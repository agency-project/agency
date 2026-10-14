from __future__ import annotations

import asyncio
import copy
import hashlib
import itertools
import json
import queue
import ssl
import threading
import time
import uuid
from concurrent.futures import Future
from contextlib import suppress
from typing import TYPE_CHECKING

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import ClientDisconnect

from ...llm import (
    API_CONN_EXCS,
    API_ERROR_EXCS,
    BAD_REQUEST_EXCS,
    RATE_LIMIT_EXCS,
    UnusableResponseError,
)
from ...llm.agllm import agllm, dispatch_attempt
from ...llm.response_check import (
    EMPTY,
    TRUNCATED,
    inspect_response,
    mark_incomplete_tool_calls,
)
from ...llm.usage_tracker import LlmUsageTracker

if TYPE_CHECKING:
    from ...configs.agconfig import agconfig as agconfig_cls
    from ...observability.agdatalogger import agDataLogger

TRANSIENT_DISPATCH_EXCS = (
    RATE_LIMIT_EXCS + API_CONN_EXCS + API_ERROR_EXCS + (ssl.SSLError, OSError, httpx.TransportError)
)

_STREAM_QUEUE_MAXSIZE = 256
_STREAM_JOIN_TIMEOUT_S = 5.0


def _annotate(span, **metadata) -> None:
    annotate = getattr(span, "annotate", None)
    if annotate is not None:
        annotate(**metadata)


def _blocks_to_message(blocks: "dict[int, dict]") -> dict:
    return {"role": "assistant", "blocks": [blocks[i] for i in sorted(blocks)]}


def _request_profile_metadata(request: dict) -> dict:
    from ...observability.profiler import agprof

    known = ("messages", "tools", "tool_choice")
    return {
        **agprof.detail_metadata("llm.tools", request.get("tools") or []),
        **agprof.detail_metadata("llm.tool_choice", request.get("tool_choice")),
        **agprof.detail_metadata(
            "llm.request_extra", {k: v for k, v in request.items() if k not in known}
        ),
    }


def _response_profile_metadata(message: dict, usage: "dict | None", stop_reason) -> dict:
    from ...observability.profiler import agprof

    usage = usage or {}
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    response = {
        **message,
        "blocks": [block for block in message.get("blocks", []) if block.get("type") != "metadata"],
    }
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "stop_reason": stop_reason,
        **agprof.detail_metadata("llm.response", response),
    }


def _payload_hash(payload: dict) -> str:
    """Content-addressable storage key for one {role, **block} transcript
    payload."""
    role = payload.get("role")
    block_type = payload.get("type")
    if block_type == "tool_use" and payload.get("id"):
        identity: dict = {"role": role, "type": block_type, "id": payload["id"]}
    elif block_type == "tool_result" and payload.get("tool_call_id"):
        identity = {"role": role, "type": block_type, "tool_call_id": payload["tool_call_id"]}
    elif block_type == "text":
        identity = {"role": role, "type": block_type, "text": payload.get("text")}
    elif block_type == "thinking":
        # Current Claude models return thinking with empty text by default,
        # so text alone would collapse every block of a run into one stored
        # payload; the signature (Anthropic) / encrypted content (OpenAI
        # Responses) is what tells them apart. Missing and "" normalize alike
        # so a rich response block and its replay still dedupe.
        identity = {
            "role": role,
            "type": block_type,
            "text": payload.get("text"),
            "signature": payload.get("signature") or "",
        }
    else:
        identity = payload
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()


def _classify_dispatch_exception(error: BaseException) -> "tuple[int, bool] | None":
    """(status_code, transient) for a known LLM-backend failure category, or
    None if *error* isn't one of them (caller re-raises it unmodified)."""
    if isinstance(error, BAD_REQUEST_EXCS):
        return 400, False
    if isinstance(error, UnusableResponseError):
        return error.status_code, error.transient
    if isinstance(error, TRANSIENT_DISPATCH_EXCS):
        return 503, True
    return None


def _retry_in_place(error: BaseException) -> bool:
    """Whether a failure *after the response started* may be asked for again
    by the host itself.

    Nothing of a response reaches the harness before it is complete, so a
    second ask is invisible to it and cannot repeat a side effect. It is
    withheld where it cannot help or would be rude: a request the provider
    rejects, a rate limit (the transport/caller owns bounded backoff), and anything that is not an ordinary exception.
    """
    if not isinstance(error, Exception):
        return False
    if isinstance(error, UnusableResponseError):
        return error.retryable
    return not isinstance(error, BAD_REQUEST_EXCS + RATE_LIMIT_EXCS)


class _ResponsePolicy:
    """llmconfig's handling of unusable responses, read once per request."""

    def __init__(self, llm) -> None:
        self.unusable_retries = max(0, int(getattr(llm, "unusable_response_retries", 0) or 0))
        self.backoff_s = max(0.0, float(getattr(llm, "unusable_response_backoff_s", 0.0) or 0.0))
        self.fail_on_empty = bool(getattr(llm, "fail_on_empty_response", True))
        self.truncation_retries = max(0, int(getattr(llm, "truncation_retries", 0) or 0))
        self.multiplier = float(getattr(llm, "truncation_output_multiplier", 2.0) or 2.0)
        self.ceiling = getattr(llm, "truncation_max_output_tokens", None)

    def next_output_cap(self, current: int) -> "int | None":
        """The larger limit to re-ask with after a cut-off response, or None
        when there is no room left to raise it."""
        raised = int(current * self.multiplier)
        if self.ceiling is not None:
            raised = min(raised, int(self.ceiling))
        return raised if raised > current else None


def _unusable_error(
    reason: str, detail: str, attempts: int, cause: "BaseException | None"
) -> UnusableResponseError:
    """The error reported to the harness once asking again has not helped.
    Keeps the original failure's own text and its upstream identifiers."""
    ids = getattr(cause, "response_ids", None) or {}
    suffix = f"reason={reason}, attempts={attempts}"
    if ids:
        suffix += ", upstream=" + ",".join(f"{k}={v}" for k, v in sorted(ids.items()))
    error = UnusableResponseError(
        reason,
        f"{detail} ({suffix})",
        status_code=getattr(cause, "status_code", None) or 502,
        details={"attempts": attempts, **ids, **(getattr(cause, "details", None) or {})},
    )
    error.__cause__ = cause
    return error


def _response_ids_of(message: "dict | None") -> dict:
    """Upstream call identifiers a backend attached to the metadata block."""
    for block in (message or {}).get("blocks") or []:
        if block.get("type") != "metadata":
            continue
        fragments = block.get("data")
        for fragment in fragments if isinstance(fragments, list) else [fragments]:
            if isinstance(fragment, dict) and isinstance(fragment.get("response_ids"), dict):
                return fragment["response_ids"]
    return {}


class _DispatchError(Exception):
    """Raised by the non-streaming path; carries what build_app()'s route
    needs to translate this into an HTTP response."""

    def __init__(self, message: str, *, status_code: int, transient: bool) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.transient = transient


class _RequestAborted(Exception):
    """Internal signal that the HTTP caller disconnected at a safe boundary."""


class _StreamHandle:
    def __init__(
        self,
        q: "queue.Queue[dict]",
        cancel_event: threading.Event,
        call_label: "str | None" = None,
    ) -> None:
        self._queue = q
        self._queue_condition = threading.Condition()
        self._cancel_event = cancel_event
        self.call_label = call_label
        self._thread: "threading.Thread | None" = None
        self._stream_ref_lock = threading.Lock()
        self._stream_ref = None
        self._stream_closed = False
        # This connection's own exchange -- kept on the handle rather than a
        # shared instance-wide list so one connection's in-progress writes
        # can never race another connection's. See
        # LlmHandlerServer.get_all_transcripts(). None until the first
        # register_stream_exchange() call creates it.
        self._entry: "dict | None" = None
        self._transcript_lock = threading.Lock()

    def get_transcript(self) -> "list[dict]":
        with self._transcript_lock:
            return [dict(self._entry)] if self._entry is not None else []

    def register_stream_exchange(self, item: "dict | None" = None, **entry_fields) -> bool:
        with self._queue_condition:
            if item is not None:
                while self._queue.full() and not self._cancel_event.is_set():
                    self._queue_condition.wait()
            if self._cancel_event.is_set():
                return False
            # Transcript fields and queue admission share the same cancel
            # linearization point. A terminal item rejected after disconnect
            # must not make an undelivered response look committed.
            with self._transcript_lock:
                if self._entry is None:
                    self._entry = {
                        "request": None,
                        "response": None,
                        "usage": None,
                        "finish_reason": None,
                        "streaming": True,
                        "ts": time.time(),
                    }
                self._entry.update(entry_fields)
            if item is not None:
                self._queue.put_nowait(item)
                self._queue_condition.notify_all()
            return True

    def _set_stream_ref(self, stream) -> None:
        with self._stream_ref_lock:
            self._stream_ref = stream
        # Cancellation can win just before the backend publishes its client.
        # Re-check after publication so the producer cannot escape the wakeup
        # that cancel_and_join() relies on.
        if self._cancel_event.is_set():
            self._close_stream()

    def _close_stream(self) -> None:
        with self._stream_ref_lock:
            if self._stream_closed or self._stream_ref is None:
                return
            self._stream_closed = True
            stream = self._stream_ref
        try:
            stream.close()
        except BaseException:
            # A failed close must remain retryable. In particular, stop()
            # retains the handle after a timeout and may call cancel again.
            with self._stream_ref_lock:
                if self._stream_ref is stream:
                    self._stream_closed = False
            raise

    def first(self) -> dict:
        with self._queue_condition:
            while self._queue.empty() and not self._cancel_event.is_set():
                self._queue_condition.wait()
            if self._cancel_event.is_set():
                return {
                    "type": "error",
                    "message": "LLM stream cancelled",
                    "transient": False,
                    "status_code": 499,
                }
            item = self._queue.get_nowait()
            self._queue_condition.notify_all()
            return item

    async def _next_item(self) -> dict:
        return await asyncio.to_thread(self.first)

    async def relay(self, first_item: dict):
        try:
            item = first_item
            while True:
                yield json.dumps(item) + "\n"
                if item["type"] in ("done", "error"):
                    return
                item = await self._next_item()
        finally:
            # This intentionally joins synchronously. Starlette cancels the
            # response task on disconnect; awaiting a shielded threadpool join
            # can let ASGI return while that detached join is still running.
            # The producer is made cancellation-aware below, so the join is
            # short and the attempt lease remains held until it is truly done.
            self.cancel_and_join()

    def cancel(self) -> None:
        # Wakes both sides of the bounded queue; this lock is the
        # linearization point after which no later item/field is admitted.
        with self._queue_condition:
            self._cancel_event.set()
            self._queue_condition.notify_all()
        self._close_stream()

    def cancel_and_join(self, timeout: "float | None" = None) -> bool:
        cancel_error: "BaseException | None" = None
        try:
            self.cancel()
        except BaseException as exc:
            cancel_error = exc
            # A transient provider-close failure leaves the close retryable.
            # Retry before an unbounded join so relay cleanup cannot strand
            # itself behind a producer that the first close failed to wake.
            with suppress(BaseException):
                self.cancel()

        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        stopped = thread is None or not thread.is_alive()
        if cancel_error is not None:
            raise cancel_error
        return stopped


class LlmHandlerServer:
    def __init__(
        self,
        agconfig: "agconfig_cls",
        data_logger: "agDataLogger",
        usage_tracker: "LlmUsageTracker",
        *,
        parent_context=None,
        request_id: "str | None" = None,
        skill_name: "str | None" = None,
    ) -> None:
        self._data_logger = data_logger
        self._usage_tracker = usage_tracker
        self._handles: "list[_StreamHandle]" = []
        self._handles_lock = threading.Lock()
        self._stopping = False
        # HTTP/UDS requests are handled on the host server's own thread, so
        # their contextvars do not automatically inherit the agent run span.
        # Keep the durable OTel context captured by HostServerManager and use
        # it explicitly for every LLM attempt span.
        self._parent_context = parent_context
        self._profile_context_provider = None
        # Non-streaming exchanges only
        self._transcript: "list[dict]" = []
        self._transcript_lock = threading.Lock()
        self._request_id = request_id
        self._skill_name = skill_name

        self.change_config(agconfig)

    def get_all_transcripts(self) -> "list[dict]":
        with self._transcript_lock:
            entries = [dict(entry) for entry in self._transcript]
        with self._handles_lock:
            handles = list(self._handles)
        for handle in handles:
            entries.extend(handle.get_transcript())
        return entries

    def get_main_transcript(self, needle: "str | None" = None) -> "list[dict]":
        entries = [
            entry
            for entry in self.get_all_transcripts()
            if isinstance(entry.get("request"), dict)
            and isinstance(entry["request"].get("messages"), list)
            and isinstance(entry.get("response"), dict)
        ]
        if not entries:
            return []
        candidates = entries
        if needle is not None:
            matching = [
                e
                for e in entries
                if any(
                    b.get("text") == needle
                    for m in e["request"]["messages"]
                    for b in m.get("blocks", [])
                )
            ]
            if matching:
                candidates = matching
        best = max(candidates, key=lambda e: len(e["request"]["messages"]))
        messages = [m for m in best["request"]["messages"] if m.get("role") != "system"]
        messages.append(best["response"])
        return messages

    def _tag_metadata_block(
        self,
        request_messages: "list[dict]",
        message: dict,
        *,
        ttft_ms: "float | None" = None,
        retries: "list[dict] | None" = None,
    ) -> None:
        """Enrich this exchange's metadata block with its own new (non-
        cumulative) prompt token count and the skill/request it belongs to."""
        blocks = message.get("blocks") or []
        metadata_block = next((b for b in blocks if b.get("type") == "metadata"), None)
        if metadata_block is None:
            return
        if retries:
            # Why earlier asks for this same turn were discarded.
            metadata_block["retries"] = retries
        if "usage" in metadata_block:
            usage = metadata_block.get("usage") or {}
            stop_reason = metadata_block.get("stop_reason")
        else:
            usage = None
            stop_reason = None
            data = metadata_block.get("data")
            if isinstance(data, list):
                for fragment in reversed(data):
                    if not isinstance(fragment, dict):
                        continue
                    if usage is None and fragment.get("usage"):
                        usage = fragment["usage"]
                    if stop_reason is None and fragment.get("stop_reason") is not None:
                        stop_reason = fragment["stop_reason"]
            usage = usage or {}
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        metadata_block["usage"] = usage
        metadata_block["stop_reason"] = stop_reason
        metadata_block["ttft_ms"] = ttft_ms
        metadata_block["new_prompt_tokens"] = self._usage_tracker.resolve_new_prompt_tokens(
            request_messages, message, prompt_tokens, completion_tokens
        )
        metadata_block["request_id"] = self._request_id
        metadata_block["skill"] = self._skill_name

    def change_config(self, agconfig: "agconfig_cls") -> None:
        self._agconfig = agconfig
        self._backend = agllm.for_config(agconfig)

    def resolve_model(self) -> str:
        return self._backend.model or ""

    def context_limit(self) -> int:
        return self._backend.fetch_context_limit()

    def _record_retry(self, call_label, attempt: int, reason: str, detail: str, **extra) -> dict:
        """Durable record that one ask for this turn was discarded and why.
        The discarded attempt's stream fragments are transient; this is what
        remains of it."""
        entry = {"attempt": attempt, "reason": reason, "detail": detail, **extra}
        agname = getattr(self._data_logger, "_default_name", None)
        self._data_logger.record_event(
            type="llm_retry",
            payload={"request_id": self._request_id, "model": self.resolve_model(), **entry},
            call_label=call_label,
            term_message=f"[{agname}] LLM    ↻  attempt {attempt} discarded: {reason}",
            print_to_terminal=False,
        )
        return entry

    def dispatch(
        self,
        request: dict,
        *,
        abort_event: "threading.Event | None" = None,
    ) -> dict:
        from ...observability.profiler import agprof

        request = self._prepare_request(request, abort_event=abort_event)
        call_label = uuid.uuid4().hex[:12]
        self._data_logger.record_event(
            type="agent_state",
            payload={
                "state": "waiting_llm",
                "causal_evidence": self._request_evidence(request),
                "request_id": self._request_id,
                "model": self.resolve_model(),
            },
            update_latest_snapshot=True,
            flush=True,
            call_label=call_label,
        )
        finalized = False

        def finalize_error(error: BaseException) -> None:
            nonlocal finalized
            self._finalize_terminal(call_label, kind="error", request=request, error=error)
            finalized = True

        def finalize_cancelled() -> None:
            nonlocal finalized
            self._finalize_terminal(call_label, kind="cancelled", request=request)
            finalized = True

        def complete_failure(error: BaseException) -> None:
            if abort_event is not None and abort_event.is_set():
                finalize_cancelled()
                raise _RequestAborted from error
            finalize_error(error)

        def aborted() -> bool:
            return abort_event is not None and abort_event.is_set()

        policy = _ResponsePolicy(self._agconfig.llm)
        backend = self._backend
        unusable_left = policy.unusable_retries
        truncation_left = policy.truncation_retries
        retries: "list[dict]" = []
        attempt = 0
        try:
            while True:
                retry: "tuple | None" = None
                next_cap = None
                with agprof.span(
                    f"llm:attempt[{attempt}]",
                    parent_context=(
                        self._profile_context_provider()
                        if self._profile_context_provider
                        else self._parent_context
                    ),
                ) as attempt_span:
                    _annotate(
                        attempt_span,
                        model=backend.model,
                        provider=type(backend).__name__,
                        call_label=call_label,
                        **agprof.detail_metadata("llm.messages", request.get("messages", [])),
                        **_request_profile_metadata(request),
                    )
                    t0 = time.perf_counter()
                    try:
                        with dispatch_attempt(
                            attempt=attempt,
                            retry_reason=retries[-1]["reason"] if retries else None,
                            call_label=call_label,
                        ):
                            result = backend.dispatch(request)
                    except BaseException as error:
                        with suppress(BaseException):
                            _annotate(
                                attempt_span,
                                outcome="failure",
                                error_type=type(error).__name__,
                            )
                        if (
                            isinstance(error, UnusableResponseError)
                            and _retry_in_place(error)
                            and unusable_left > 0
                            and not aborted()
                        ):
                            retry = (error.reason, f"{type(error).__name__}: {error}")
                        else:
                            complete_failure(error)
                            classified = _classify_dispatch_exception(error)
                            if classified is None:
                                raise
                            status_code, transient = classified
                            raise _DispatchError(
                                str(error), status_code=status_code, transient=transient
                            ) from error
                    else:
                        ttft_ms = round((time.perf_counter() - t0) * 1000, 3)
                        verdict = inspect_response(
                            result["message"],
                            result["stop_reason"],
                            empty_is_unusable=policy.fail_on_empty,
                        )
                        deliver = verdict.usable
                        if verdict.reason == TRUNCATED:
                            if truncation_left > 0:
                                next_cap = policy.next_output_cap(backend.output_token_cap())
                            if next_cap is None:
                                # No room left to raise the limit: hand the
                                # cut-off turn on, marked so no adapter can
                                # run its partial tool calls.
                                mark_incomplete_tool_calls(result["message"])
                                deliver = True
                        if deliver:
                            _annotate(
                                attempt_span,
                                outcome="success" if verdict.usable else "incomplete",
                                **_response_profile_metadata(
                                    result["message"], result["usage"], result["stop_reason"]
                                ),
                                ttft_ms=ttft_ms,
                            )
                            self._tag_metadata_block(
                                request["messages"],
                                result["message"],
                                ttft_ms=ttft_ms,
                                retries=retries,
                            )
                            with self._transcript_lock:
                                self._transcript.append(
                                    {
                                        "request": request,
                                        "response": result["message"],
                                        "usage": result["usage"],
                                        "finish_reason": result["stop_reason"],
                                        "streaming": False,
                                        "ts": time.time(),
                                    }
                                )
                            self._finalize_success(call_label, request, result["message"])
                            finalized = True
                            return result
                        _annotate(
                            attempt_span,
                            outcome="retry",
                            retry_reason=verdict.reason,
                            **_response_profile_metadata(
                                result["message"], result["usage"], result["stop_reason"]
                            ),
                        )
                        if verdict.reason != TRUNCATED and (unusable_left <= 0 or aborted()):
                            error = _unusable_error(
                                verdict.reason, verdict.detail, attempt + 1, None
                            )
                            complete_failure(error)
                            raise _DispatchError(
                                str(error), status_code=error.status_code, transient=True
                            ) from error
                        retry = (verdict.reason, verdict.detail)
                reason, detail = retry
                if reason == TRUNCATED:
                    truncation_left -= 1
                    retries.append(
                        self._record_retry(
                            call_label,
                            attempt,
                            reason,
                            detail,
                            output_token_cap=backend.output_token_cap(),
                            next_output_token_cap=next_cap,
                        )
                    )
                    backend = backend.with_output_token_cap(next_cap)
                else:
                    unusable_left -= 1
                    retries.append(self._record_retry(call_label, attempt, reason, detail))
                    delay = policy.backoff_s * (2 ** (len(retries) - 1))
                    if abort_event is not None:
                        abort_event.wait(delay)
                    elif delay:
                        time.sleep(delay)
                attempt += 1
        except BaseException as error:
            if not finalized:
                complete_failure(error)
            raise

    def _new_stream_handle(
        self,
        abort_event: "threading.Event | None" = None,
    ) -> _StreamHandle:
        return _StreamHandle(
            queue.Queue(maxsize=_STREAM_QUEUE_MAXSIZE),
            abort_event if abort_event is not None else threading.Event(),
            uuid.uuid4().hex[:12],
        )

    def start_stream(
        self,
        request: dict,
        *,
        abort_event: "threading.Event | None" = None,
        _handle: "_StreamHandle | None" = None,
    ) -> "_StreamHandle":
        from ...observability.profiler import agprof

        handle = _handle if _handle is not None else self._new_stream_handle(abort_event)
        if abort_event is not None and handle._cancel_event is not abort_event:
            raise ValueError("stream handle and abort event must use the same event")
        request = self._prepare_request(request, abort_event=handle._cancel_event)
        call_label = handle.call_label
        self._data_logger.record_event(
            type="agent_state",
            payload={
                "state": "waiting_llm",
                "causal_evidence": self._request_evidence(request),
                "request_id": self._request_id,
                "model": self.resolve_model(),
            },
            update_latest_snapshot=True,
            flush=True,
            call_label=call_label,
        )
        try:
            thread = agprof.spawn_traced(
                self._run_stream_producer,
                request,
                handle,
                daemon=True,
            )
            with self._handles_lock:
                if self._stopping:
                    raise RuntimeError("LLM handler server is stopping")
                handle._thread = thread
                # Starting and publishing share the same synchronization
                # boundary as stop()'s snapshot. Shutdown therefore either
                # rejects this producer or observes and joins it.
                thread.start()
                self._handles.append(handle)
            return handle
        except BaseException as error:
            was_cancelled = handle._cancel_event.is_set()
            with suppress(BaseException):
                handle.cancel()
            with self._handles_lock:
                if handle in self._handles:
                    self._handles.remove(handle)
            handle._thread = None
            if was_cancelled:
                self._finalize_terminal(call_label, kind="cancelled", request=request)
                raise _RequestAborted from error
            self._finalize_terminal(call_label, kind="error", request=request, error=error)
            raise

    def stop(self) -> None:
        with self._handles_lock:
            self._stopping = True
            handles = list(self._handles)
        errors: "list[BaseException]" = []
        for handle in handles:
            try:
                handle.cancel()
            except BaseException as error:
                errors.append(error)
        alive: "list[_StreamHandle]" = []
        deadline = time.monotonic() + _STREAM_JOIN_TIMEOUT_S
        for handle in handles:
            thread = handle._thread
            if thread is None:
                continue
            if thread is threading.current_thread():
                alive.append(handle)
                continue
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                alive.append(handle)
        if alive:
            raise RuntimeError(f"{len(alive)} LLM stream producer(s) did not stop")
        if errors:
            raise RuntimeError("failed to cancel an LLM stream producer") from errors[0]

    @staticmethod
    def _spawn_http_worker(target, /, *args, **kwargs) -> "tuple[threading.Thread, Future]":
        from ...observability.profiler import agprof

        result: Future = Future()

        def run() -> None:
            try:
                result.set_result(target(*args, **kwargs))
            except BaseException as error:
                result.set_exception(error)

        thread = agprof.spawn_traced(run, daemon=True)
        thread.name = "llm-http-dispatch"
        thread.start()
        return thread, result

    async def _wait_for_http_worker(
        self,
        request: Request,
        thread: threading.Thread,
        result: Future,
        abort,
        cleanup=None,
    ) -> bool:
        async_result = asyncio.wrap_future(result)
        disconnect_task = asyncio.create_task(self._wait_for_disconnect(request))

        def abort_and_join() -> None:
            errors: "list[BaseException]" = []
            try:
                abort()
            except BaseException as error:
                errors.append(error)
            finally:
                thread.join()
                if cleanup is not None:
                    try:
                        cleanup()
                    except BaseException as error:
                        errors.append(error)
                async_result.cancel()
            if errors:
                raise errors[0]

        try:
            done, _pending = await asyncio.wait(
                {async_result, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            # If completion and disconnect become visible in the same loop
            # turn, preserve the disconnect. Otherwise this receive would be
            # consumed and a later response-stage watcher could wait forever.
            if disconnect_task in done:
                abort_and_join()
                return True

            with suppress(BaseException):
                async_result.exception()
            disconnect_task.cancel()
            with suppress(asyncio.CancelledError):
                await disconnect_task
            thread.join()
            return False
        except asyncio.CancelledError:
            abort_and_join()
            raise
        finally:
            if not disconnect_task.done():
                disconnect_task.cancel()
                with suppress(asyncio.CancelledError):
                    await disconnect_task

    @staticmethod
    async def _wait_for_disconnect(request: Request) -> None:
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                return

    async def _first_stream_item(self, request: Request, handle: _StreamHandle) -> dict:
        worker, result = self._spawn_http_worker(handle.first)
        disconnected = await self._wait_for_http_worker(
            request,
            worker,
            result,
            handle.cancel,
            handle.cancel_and_join,
        )
        if disconnected:
            raise ClientDisconnect
        return result.result()

    def build_app(self) -> FastAPI:
        app = FastAPI()
        # Endpoint caches may outlive this app; only the app owns invocation state.
        app.state.llm_handler_server = self

        @app.post("/dispatch")
        async def _dispatch(http_request: Request):
            server = http_request.app.state.llm_handler_server
            request = await http_request.json()
            try:
                if request.get("stream"):
                    abort_event = threading.Event()
                    handle = server._new_stream_handle(abort_event)
                    worker, result = server._spawn_http_worker(
                        server.start_stream,
                        request,
                        abort_event=abort_event,
                        _handle=handle,
                    )
                    disconnected = await server._wait_for_http_worker(
                        http_request,
                        worker,
                        result,
                        handle.cancel,
                        handle.cancel_and_join,
                    )
                    if disconnected:
                        raise ClientDisconnect
                    result.result()
                    item = await server._first_stream_item(http_request, handle)
                    if item["type"] == "error":
                        if not handle.cancel_and_join():
                            raise RuntimeError("LLM stream producer did not stop")
                        return JSONResponse(
                            {
                                "error": {
                                    "message": item["message"],
                                    "transient": item["transient"],
                                }
                            },
                            status_code=item["status_code"],
                        )
                    return StreamingResponse(handle.relay(item), media_type="application/x-ndjson")

                abort_event = threading.Event()

                worker, result = server._spawn_http_worker(
                    server.dispatch,
                    request,
                    abort_event=abort_event,
                )
                disconnected = await server._wait_for_http_worker(
                    http_request,
                    worker,
                    result,
                    abort_event.set,
                )
                if disconnected:
                    raise ClientDisconnect
                return JSONResponse(result.result())
            except _DispatchError as e:
                return JSONResponse(
                    {"error": {"message": str(e), "transient": e.transient}},
                    status_code=e.status_code,
                )
            except _RequestAborted as error:
                raise ClientDisconnect from error

        @app.get("/resolve_model")
        def _resolve_model(http_request: Request) -> JSONResponse:
            server = http_request.app.state.llm_handler_server
            return JSONResponse({"model": server.resolve_model()})

        @app.get("/context_limit")
        def _context_limit(http_request: Request) -> JSONResponse:
            server = http_request.app.state.llm_handler_server
            return JSONResponse({"context_limit": server.context_limit()})

        return app

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _prepare_request(
        self,
        request: dict,
        *,
        abort_event: "threading.Event | None" = None,
    ) -> dict:
        prepared = copy.deepcopy(request)
        prepared.pop("agency_internal_kind", None)
        if abort_event is not None and abort_event.is_set():
            raise _RequestAborted
        return prepared

    def _finalize_terminal(
        self,
        call_label: "str | None",
        *,
        kind: str,
        request: "dict | None" = None,
        error: "BaseException | None" = None,
    ) -> None:
        """Retain the attempted prompt even when a request fails or is cancelled."""
        agname = getattr(self._data_logger, "_default_name", None)
        if kind == "cancelled":
            marker = {"cancelled": True}
            suffix = "cancelled"
        else:
            suffix = f"{type(error).__name__}: {error}"
            marker = {"error": suffix}
        self._data_logger.record_llm_exchange(
            call_label,
            exchange_type=f"llm_stream_{kind}",
            prompt_chain=self._prompt_chain(request) if request else [],
            response_chain=[(_payload_hash(marker), marker)],
            term_message=f"[{agname}] LLM    ✗  {suffix}",
            print_to_terminal=False,
        )

    def _request_evidence(self, request: dict) -> dict:
        from ...observability.agwebui.causal_graph import exchange_evidence

        return exchange_evidence([b for _, b in self._prompt_chain(request)], [])

    def _prompt_chain(self, request: dict) -> "list[tuple[str, dict]]":
        """This exchange's *complete* prompt, as an ordered list of
        (content hash, payload) pairs -- covers every block the request
        carries, not just `messages`."""
        candidates: "list[dict]" = []
        for message in request.get("messages") or []:
            role = message.get("role")
            for block in message.get("blocks") or []:
                candidates.append({"role": role, **block})
        for tool in request.get("tools") or []:
            function = tool.get("function") or {}
            candidates.append(
                {
                    "role": "tool_schema",
                    "type": "tool_definition",
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters"),
                }
            )
        if request.get("tool_choice") is not None:
            candidates.append(
                {"role": "tool_schema", "type": "tool_choice", "value": request["tool_choice"]}
            )
        # Catch-all for any field this function doesn't know to name yet --
        # same reasoning as llm.request_extra on the span side: an unnamed
        # field caused exactly this kind of silent gap once already, so a
        # *new* one must show up here too instead of requiring someone to
        # notice and hand-add it to the list above first.
        known_keys = {"messages", "tools", "tool_choice"}
        request_extra = {k: v for k, v in request.items() if k not in known_keys}
        if request_extra:
            candidates.append(
                {"role": "tool_schema", "type": "request_extra", "value": request_extra}
            )
        return [(_payload_hash(payload), payload) for payload in candidates]

    def _response_chain(self, response_message: "dict | None") -> "list[tuple[str, dict]]":
        """This exchange's *complete* response, as an ordered list of
        (content hash, payload) pairs."""
        if response_message is None:
            return []
        role = response_message.get("role", "assistant")
        candidates = [{"role": role, **block} for block in response_message.get("blocks") or []]
        return [(_payload_hash(payload), payload) for payload in candidates]

    def _finalize_success(
        self, call_label: "str | None", request: dict, response_message: "dict | None"
    ) -> None:
        prompt_chain = self._prompt_chain(request)
        response_chain = self._response_chain(response_message)
        if prompt_chain or response_chain:
            self._data_logger.record_llm_exchange(
                call_label,
                exchange_type="llm_block",
                prompt_chain=prompt_chain,
                response_chain=response_chain,
            )

    def _run_stream_producer(
        self,
        request: dict,
        handle: "_StreamHandle",
    ) -> None:
        from ...observability.profiler import agprof

        finalized = False

        def finalize_error(error: BaseException) -> None:
            nonlocal finalized
            if finalized:
                return
            self._finalize_terminal(handle.call_label, kind="error", request=request, error=error)
            finalized = True

        def finalize_cancelled() -> None:
            nonlocal finalized
            if finalized:
                return
            self._finalize_terminal(handle.call_label, kind="cancelled", request=request)
            finalized = True

        def finalize_success(response_message: dict) -> None:
            nonlocal finalized
            if finalized:
                return
            self._finalize_success(handle.call_label, request, response_message)
            finalized = True

        def publish_error(
            exception: BaseException,
            *,
            status_code: int,
            transient: bool,
            **entry_fields,
        ) -> bool:
            if finalized:
                return False
            entry_fields.setdefault("request", request)
            enqueued = handle.register_stream_exchange(
                {
                    "type": "error",
                    "message": str(exception),
                    "transient": getattr(exception, "transient", transient),
                    "status_code": getattr(exception, "status_code", status_code),
                },
                **entry_fields,
            )
            if enqueued:
                finalize_error(exception)
            else:
                finalize_cancelled()
            return enqueued

        policy = _ResponsePolicy(self._agconfig.llm)
        retries: "list[dict]" = []

        def run_attempt(attempt: int, backend) -> tuple:
            """One ask of the provider. Returns ("stop",) once this producer
            has published its terminal item, ("ok" | "truncated", message,
            usage, stop_reason, ttft_ms, verdict) for an assembled response,
            or ("retry", reason, detail, cause) for one to discard."""
            with agprof.span(
                f"llm:attempt[{attempt}]",
                parent_context=(
                    self._profile_context_provider()
                    if self._profile_context_provider
                    else self._parent_context
                ),
            ) as attempt_span:
                _annotate(
                    attempt_span,
                    model=backend.model,
                    provider=type(backend).__name__,
                    call_label=handle.call_label,
                    **agprof.detail_metadata("llm.messages", request.get("messages", [])),
                    **_request_profile_metadata(request),
                )
                t0 = time.perf_counter()
                try:
                    stream_iter = iter(
                        backend.dispatch_stream(request, on_client=handle._set_stream_ref)
                    )
                    first_item = next(stream_iter)
                except StopIteration:
                    empty_message = {"role": "assistant", "blocks": []}
                    _annotate(attempt_span, **_response_profile_metadata(empty_message, None, None))
                    if policy.fail_on_empty:
                        _annotate(attempt_span, outcome="retry", retry_reason=EMPTY)
                        return ("retry", EMPTY, "provider stream carried no events", None)
                    _annotate(attempt_span, outcome="success")
                    enqueued = handle.register_stream_exchange(
                        {
                            "type": "done",
                            "message": empty_message,
                            "usage": None,
                            "stop_reason": None,
                        },
                        request=request,
                        response=empty_message,
                        streaming=False,
                    )
                    if not enqueued:
                        finalize_cancelled()
                        return ("stop",)
                    finalize_success(empty_message)
                    return ("stop",)
                except BaseException as e:
                    # A producer thread must always publish a first terminal
                    # item. Otherwise the synchronous route remains blocked in
                    # handle.first() when setup or the first read fails.
                    _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
                    if handle._cancel_event.is_set():
                        finalize_cancelled()
                        return ("stop",)
                    if isinstance(e, UnusableResponseError) and _retry_in_place(e):
                        # The provider answered and the answer was unusable;
                        # a transport failure before any answer is left to the
                        # SDK's own retries and to the harness.
                        return ("retry", e.reason, f"{type(e).__name__}: {e}", e)
                    classified = _classify_dispatch_exception(e)
                    status_code, transient = classified if classified is not None else (500, False)
                    publish_error(e, status_code=status_code, transient=transient)
                    return ("stop",)
                ttft_ms = round((time.perf_counter() - t0) * 1000, 3)
                _annotate(attempt_span, ttft_ms=ttft_ms)

                handle.register_stream_exchange(
                    request=request, response={"role": "assistant", "blocks": []}
                )
                blocks: "dict[int, dict]" = {}
                usage = None
                stop_reason = None
                try:
                    for stream_item in itertools.chain([first_item], stream_iter):
                        if handle._cancel_event.is_set():
                            break
                        self._data_logger.record_stream_delta(
                            type="llm_stream_delta",
                            payload=stream_item,
                            call_label=handle.call_label,
                        )
                        if stream_item["type"] == "usage":
                            if stream_item.get("usage") is not None:
                                usage = stream_item["usage"]
                            if stream_item.get("stop_reason") is not None:
                                stop_reason = stream_item["stop_reason"]
                            continue
                        idx = stream_item["index"]
                        now = time.time()
                        block = blocks.setdefault(
                            idx,
                            {
                                "type": stream_item["block_type"],
                                "index": idx,
                                "text": "",
                                "signature": "",
                                "id": "",
                                "name": "",
                                "arguments": "",
                                "data": None,
                                "citations": None,
                                "ts_start": now,
                            },
                        )
                        text_piece = stream_item.get("text") or ""
                        if text_piece:
                            block["text"] += text_piece
                        citations_piece = stream_item.get("citations")
                        if citations_piece:
                            if block["citations"] is None:
                                block["citations"] = []
                            block["citations"].extend(citations_piece)
                        sig_piece = stream_item.get("signature") or ""
                        if sig_piece:
                            block["signature"] += sig_piece
                        if stream_item.get("id"):
                            block["id"] = stream_item["id"]
                        name_piece = stream_item.get("name") or ""
                        if name_piece:
                            block["name"] += name_piece
                        args_piece = stream_item.get("arguments") or ""
                        if args_piece:
                            block["arguments"] += args_piece
                        if stream_item.get("data") is not None:
                            if block["data"] is None:
                                block["data"] = []
                            block["data"].append(stream_item["data"])
                        if stream_item.get("phase"):
                            block["phase"] = stream_item["phase"]
                        block["ts_end"] = now
                        message = _blocks_to_message(blocks)
                        handle.register_stream_exchange(response=message)
                except BaseException as e:
                    partial = _blocks_to_message(blocks)
                    _annotate(
                        attempt_span,
                        outcome="failure",
                        error_type=type(e).__name__,
                        **_response_profile_metadata(partial, usage, stop_reason),
                    )
                    if handle._cancel_event.is_set():
                        finalize_cancelled()
                        return ("stop",)
                    if _retry_in_place(e):
                        if not getattr(e, "response_ids", None):
                            with suppress(Exception):
                                e.response_ids = _response_ids_of(partial)
                        return (
                            "retry",
                            getattr(e, "reason", "stream_interrupted"),
                            f"{type(e).__name__}: {e}",
                            e,
                        )
                    classified = _classify_dispatch_exception(e)
                    status_code, transient = classified if classified is not None else (500, False)
                    publish_error(
                        e,
                        status_code=status_code,
                        transient=transient,
                        streaming=False,
                        error=f"{type(e).__name__}: {e}",
                    )
                    return ("stop",)

                if handle._cancel_event.is_set():
                    finalize_cancelled()
                    return ("stop",)
                message = _blocks_to_message(blocks)
                verdict = inspect_response(
                    message, stop_reason, empty_is_unusable=policy.fail_on_empty
                )
                _annotate(
                    attempt_span,
                    outcome="success" if verdict.usable else "unusable",
                    **({} if verdict.usable else {"unusable_reason": verdict.reason}),
                    **_response_profile_metadata(message, usage, stop_reason),
                )
                if verdict.usable:
                    return ("ok", message, usage, stop_reason, ttft_ms, verdict)
                if verdict.reason == TRUNCATED:
                    return ("truncated", message, usage, stop_reason, ttft_ms, verdict)
                cause = UnusableResponseError(verdict.reason, verdict.detail)
                cause.response_ids = _response_ids_of(message)
                return ("retry", verdict.reason, verdict.detail, cause)

        def deliver(message, usage, stop_reason, ttft_ms) -> None:
            enqueued = handle.register_stream_exchange(
                {
                    "type": "done",
                    "message": message,
                    "usage": usage,
                    "stop_reason": stop_reason,
                },
                request=request,
                response=message,
                usage=usage,
                finish_reason=stop_reason,
                streaming=False,
            )
            if not enqueued:
                finalize_cancelled()
                return
            self._tag_metadata_block(request["messages"], message, ttft_ms=ttft_ms, retries=retries)
            finalize_success(message)

        try:
            backend = self._backend
            unusable_left = policy.unusable_retries
            truncation_left = policy.truncation_retries
            attempt = 0
            while True:
                if handle._cancel_event.is_set():
                    finalize_cancelled()
                    return
                with dispatch_attempt(
                    attempt=attempt,
                    retry_reason=retries[-1]["reason"] if retries else None,
                    call_label=handle.call_label,
                ):
                    outcome = run_attempt(attempt, backend)
                kind = outcome[0]
                if kind == "stop":
                    return
                if kind == "ok":
                    deliver(*outcome[1:5])
                    return
                if kind == "truncated":
                    _, message, usage, stop_reason, ttft_ms, verdict = outcome
                    next_cap = (
                        policy.next_output_cap(backend.output_token_cap())
                        if truncation_left > 0
                        else None
                    )
                    if next_cap is None:
                        # No room left to raise the limit. The turn goes on
                        # with its stop reason intact and its partial tool
                        # calls marked, so an adapter reports it as incomplete
                        # and can never run them.
                        mark_incomplete_tool_calls(message)
                        deliver(message, usage, stop_reason, ttft_ms)
                        return
                    truncation_left -= 1
                    retries.append(
                        self._record_retry(
                            handle.call_label,
                            attempt,
                            TRUNCATED,
                            verdict.detail,
                            output_token_cap=backend.output_token_cap(),
                            next_output_token_cap=next_cap,
                            usage=usage,
                            **_response_ids_of(message),
                        )
                    )
                    backend = backend.with_output_token_cap(next_cap)
                    attempt += 1
                    continue
                _, reason, detail, cause = outcome
                if unusable_left <= 0:
                    error = _unusable_error(reason, detail, attempt + 1, cause)
                    publish_error(
                        error,
                        status_code=error.status_code,
                        transient=True,
                        streaming=False,
                        error=f"{type(error).__name__}: {error}",
                    )
                    return
                unusable_left -= 1
                retries.append(
                    self._record_retry(
                        handle.call_label,
                        attempt,
                        reason,
                        detail,
                        **(getattr(cause, "response_ids", None) or {}),
                    )
                )
                # Cancellation ends the wait at once; the loop head sees it.
                handle._cancel_event.wait(policy.backoff_s * (2 ** (len(retries) - 1)))
                attempt += 1
        except BaseException as e:
            if not finalized:
                if handle._cancel_event.is_set():
                    finalize_cancelled()
                else:
                    publish_error(
                        e,
                        status_code=500,
                        transient=False,
                        streaming=False,
                        error=f"{type(e).__name__}: {e}",
                    )
        finally:
            handle._close_stream()
