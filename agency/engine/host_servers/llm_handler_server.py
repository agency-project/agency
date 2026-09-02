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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import ClientDisconnect

from ...llm import API_CONN_EXCS, API_ERROR_EXCS, BAD_REQUEST_EXCS, RATE_LIMIT_EXCS
from ...llm.agllm import agllm

if TYPE_CHECKING:
    from ...agconfig import agConfig
    from ...agdatacollector import agDataCollector

TRANSIENT_DISPATCH_EXCS = (
    RATE_LIMIT_EXCS + API_CONN_EXCS + API_ERROR_EXCS + (ssl.SSLError, OSError, httpx.TransportError)
)

_STREAM_QUEUE_MAXSIZE = 256
_STREAM_JOIN_TIMEOUT_S = 5.0
_INTERNAL_COMPACTION_KIND = "compaction"

# These values are part of InvocationHandle's deliberately small control
# vocabulary.  Keep this server independent of the harness wire protocol: the
# LLM safe boundaries are host-side coordination, not sandbox RPC messages.
_CONTROL_PHASE_PRE_GENERATION = "model"
_CONTROL_PHASE_POST_GENERATION = "boundary"
_CONTROL_PHASE_CLOSING = "closing"
_INVOCATION_CANCELLED_MESSAGE = "agent invocation cancelled"
_AGENT_DESTROYED_MESSAGE = "agent destroyed"


def _annotate(span, **metadata) -> None:
    annotate = getattr(span, "annotate", None)
    if annotate is not None:
        annotate(**metadata)


def _blocks_to_message(blocks: "dict[int, dict]") -> dict:
    return {"role": "assistant", "blocks": [blocks[i] for i in sorted(blocks)]}


def _stable_digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _request_fingerprint(request: dict) -> str:
    """Return a stable identity for a semantic generation, including retries."""
    semantic_request = {
        key: value
        for key, value in request.items()
        if key not in {"stream", "agency_internal_kind"}
    }
    return _stable_digest(semantic_request)


def _history_anchor(messages: "list[dict]") -> str:
    return "root" if not messages else _stable_digest(messages)


def _history_allows_user_turn(messages: "list[dict]") -> bool:
    """Return whether appending an invocation message preserves tool pairing."""
    outstanding: "dict[str, int]" = {}
    for message in messages:
        for block in message.get("blocks", []):
            if block.get("type") == "tool_use":
                tool_id = str(block.get("id", ""))
                outstanding[tool_id] = outstanding.get(tool_id, 0) + 1
            elif block.get("type") == "tool_result":
                tool_id = str(block.get("tool_call_id", ""))
                remaining = outstanding.get(tool_id, 0)
                if remaining == 0:
                    return False
                if remaining <= 1:
                    outstanding.pop(tool_id, None)
                else:
                    outstanding[tool_id] = remaining - 1
    return not outstanding


@dataclass(frozen=True)
class _MessageOverlay:
    sequence: int
    content: str
    anchor: str


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
        *,
        interrupt_checkpoint=None,
    ) -> None:
        self._queue = q
        self._queue_condition = threading.Condition()
        self._cancel_event = cancel_event
        self.call_label = call_label
        self._thread: "threading.Thread | None" = None
        self._stream_ref_lock = threading.Lock()
        self._stream_ref = None
        self._stream_closed = False
        self._interrupt_checkpoint = interrupt_checkpoint
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
                return self._cancelled_item()
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
        # Wake either side of the bounded queue. A disconnected consumer may
        # leave the producer waiting for capacity; a pre-first-item disconnect
        # may leave the consumer waiting for data.
        with self._queue_condition:
            # Publish and cancellation use this lock as their linearization
            # point: once cancellation wins, no later item or transcript field
            # can be admitted.
            self._cancel_event.set()
            self._queue_condition.notify_all()
        try:
            if self._interrupt_checkpoint is not None:
                self._interrupt_checkpoint(self._cancel_event)
        finally:
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

    @staticmethod
    def _cancelled_item() -> dict:
        return {
            "type": "error",
            "message": "LLM stream cancelled",
            "transient": False,
            "status_code": 499,
        }


class LlmHandlerServer:
    def __init__(
        self,
        agconfig: "agConfig",
        data_collector: "agDataCollector",
        *,
        parent_context=None,
        invocation=None,
        enable_message_overlay: bool = True,
    ) -> None:
        self._data_collector = data_collector
        self._handles: "list[_StreamHandle]" = []
        self._handles_lock = threading.Lock()
        self._stopping = False
        # HTTP/UDS requests are handled on the host server's own thread, so
        # their contextvars do not automatically inherit the agent run span.
        # Keep the durable OTel context captured by HostServerManager and use
        # it explicitly for every LLM attempt span.
        self._parent_context = parent_context
        # Non-streaming exchanges only
        self._transcript: "list[dict]" = []
        self._transcript_lock = threading.Lock()
        self._invocation = invocation
        self._enable_message_overlay = enable_message_overlay
        self._control_lock = threading.RLock()
        self._message_overlays: "list[_MessageOverlay]" = []
        self._message_sequences: "set[int]" = set()
        self.set_config(agconfig)

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

    def _record_exchange(
        self, request: dict, message: dict, usage: "dict | None", finish_reason: "str | None"
    ) -> None:
        entry = {
            "request": request,
            "response": message,
            "usage": usage,
            "finish_reason": finish_reason,
            "streaming": False,
            "ts": time.time(),
        }
        with self._transcript_lock:
            self._transcript.append(entry)

    def set_config(self, agconfig: "agConfig") -> None:
        self._agconfig = agconfig
        self._backend = agllm.for_config(agconfig)

    def resolve_model(self) -> str:
        return self._backend.model or ""

    def context_limit(self) -> int:
        return self._backend.fetch_context_limit()

    def dispatch(
        self,
        request: dict,
        *,
        abort_event: "threading.Event | None" = None,
    ) -> dict:
        from ...profiler import agprof

        request, boundary_id, controlled = self._prepare_request(
            request,
            abort_event=abort_event,
        )
        call_label = uuid.uuid4().hex[:12]
        self._data_collector.record_event(
            type="agent_state",
            payload={"state": "waiting_llm"},
            overwrite=True,
            flush=True,
            call_label=call_label,
        )
        finalized = False

        def finalize_error(error: BaseException) -> None:
            nonlocal finalized
            self._finalize_error(call_label, error)
            finalized = True

        def finalize_cancelled() -> None:
            nonlocal finalized
            self._finalize_cancelled(call_label)
            finalized = True

        def complete_failure(error: BaseException) -> None:
            try:
                completed = self._complete_failed_model_request(
                    boundary_id,
                    controlled=controlled,
                    abort_event=abort_event,
                )
            except BaseException as control_error:
                finalize_error(control_error)
                raise
            if not completed:
                finalize_cancelled()
                raise _RequestAborted from error
            finalize_error(error)

        try:
            with agprof.span("llm:attempt[0]", parent_context=self._parent_context) as attempt_span:
                _annotate(
                    attempt_span,
                    model=self._backend.model,
                    provider=type(self._backend).__name__,
                )
                try:
                    result = self._backend.dispatch(request)
                except BAD_REQUEST_EXCS as error:
                    with suppress(BaseException):
                        _annotate(
                            attempt_span,
                            outcome="failure",
                            error_type=type(error).__name__,
                        )
                    complete_failure(error)
                    raise _DispatchError(str(error), status_code=400, transient=False) from error
                except TRANSIENT_DISPATCH_EXCS as error:
                    with suppress(BaseException):
                        _annotate(
                            attempt_span,
                            outcome="failure",
                            error_type=type(error).__name__,
                        )
                    complete_failure(error)
                    raise _DispatchError(str(error), status_code=503, transient=True) from error
                except BaseException as error:
                    with suppress(BaseException):
                        _annotate(
                            attempt_span,
                            outcome="failure",
                            error_type=type(error).__name__,
                        )
                    complete_failure(error)
                    raise

                model_boundary_attempted = False
                try:
                    usage = result["usage"]
                    message = result["message"]
                    stop_reason = result["stop_reason"]
                    blocks = message["blocks"]
                    model_boundary_attempted = True
                    completed = self._complete_model_request(
                        boundary_id,
                        message,
                        controlled=controlled,
                        abort_event=abort_event,
                    )
                    if not completed:
                        raise _RequestAborted
                    _annotate(
                        attempt_span,
                        outcome="success",
                        input_tokens=(usage or {}).get("prompt_tokens", 0),
                        output_tokens=(usage or {}).get("completion_tokens", 0),
                    )
                    self._record_exchange(request, message, usage, stop_reason)
                except _RequestAborted:
                    finalize_cancelled()
                    raise
                except BaseException as error:
                    if model_boundary_attempted:
                        finalize_error(error)
                    else:
                        complete_failure(error)
                    raise
                self._data_collector.finalize_stream(call_label, type="llm_block", payloads=blocks)
                finalized = True
                return result
        except BaseException as error:
            if not finalized:
                # Span creation/entry, initial annotation, and other outer
                # infrastructure failures still cross the post-error control
                # boundary before their collector label is terminated.
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
            interrupt_checkpoint=self._interrupt_checkpoint_wait,
        )

    def start_stream(
        self,
        request: dict,
        *,
        abort_event: "threading.Event | None" = None,
        _handle: "_StreamHandle | None" = None,
    ) -> "_StreamHandle":
        from ...profiler import agprof

        handle = _handle if _handle is not None else self._new_stream_handle(abort_event)
        if abort_event is not None and handle._cancel_event is not abort_event:
            raise ValueError("stream handle and abort event must use the same event")
        request, boundary_id, controlled = self._prepare_request(
            request,
            abort_event=handle._cancel_event,
        )
        call_label = handle.call_label
        self._data_collector.record_event(
            type="agent_state",
            payload={"state": "waiting_llm"},
            overwrite=True,
            flush=True,
            call_label=call_label,
        )
        try:
            thread = agprof.spawn_traced(
                self._run_stream_producer,
                request,
                boundary_id,
                controlled,
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
            terminal_error = error
            try:
                completed = self._complete_failed_model_request(
                    boundary_id,
                    controlled=controlled,
                    abort_event=handle._cancel_event,
                )
            except BaseException as control_error:
                terminal_error = control_error
                completed = True
            with suppress(BaseException):
                handle.cancel()
            with self._handles_lock:
                if handle in self._handles:
                    self._handles.remove(handle)
            handle._thread = None
            if completed:
                self._finalize_error(call_label, terminal_error)
            else:
                self._finalize_cancelled(call_label)
                raise _RequestAborted from error
            if terminal_error is not error:
                raise terminal_error from error
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
        from ...profiler import agprof

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

        @app.post("/dispatch")
        async def _dispatch(http_request: Request):
            request = await http_request.json()
            try:
                if request.get("stream"):
                    abort_event = threading.Event()
                    handle = self._new_stream_handle(abort_event)
                    worker, result = self._spawn_http_worker(
                        self.start_stream,
                        request,
                        abort_event=abort_event,
                        _handle=handle,
                    )
                    disconnected = await self._wait_for_http_worker(
                        http_request,
                        worker,
                        result,
                        handle.cancel,
                        handle.cancel_and_join,
                    )
                    if disconnected:
                        raise ClientDisconnect
                    result.result()
                    item = await self._first_stream_item(http_request, handle)
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

                def abort_nonstream() -> None:
                    abort_event.set()
                    self._interrupt_checkpoint_wait(abort_event)

                worker, result = self._spawn_http_worker(
                    self.dispatch,
                    request,
                    abort_event=abort_event,
                )
                disconnected = await self._wait_for_http_worker(
                    http_request,
                    worker,
                    result,
                    abort_nonstream,
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
        def _resolve_model() -> JSONResponse:
            return JSONResponse({"model": self.resolve_model()})

        @app.get("/context_limit")
        def _context_limit() -> JSONResponse:
            return JSONResponse({"context_limit": self.context_limit()})

        return app

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _prepare_request(
        self,
        request: dict,
        *,
        abort_event: "threading.Event | None" = None,
    ) -> "tuple[dict, str | None, bool]":
        prepared = copy.deepcopy(request)
        internal_kind = prepared.pop("agency_internal_kind", None)
        if self._invocation is None or internal_kind == _INTERNAL_COMPACTION_KIND:
            if abort_event is not None and abort_event.is_set():
                raise _RequestAborted
            return prepared, None, False

        fingerprint = _request_fingerprint(prepared)
        boundary_id = f"llm:{fingerprint}"
        with self._control_lock:
            messages = prepared.get("messages") or []
            protocol_valid = _history_allows_user_turn(messages)
            decision = self._checkpoint_invocation(
                f"{boundary_id}:pre",
                allow_messages=self._enable_message_overlay and protocol_valid,
                phase=_CONTROL_PHASE_PRE_GENERATION,
                abort_event=abort_event,
            )
            if decision is None:
                raise _RequestAborted
            self._raise_if_stopped(decision)

            if self._enable_message_overlay and protocol_valid:
                anchor = _history_anchor(messages)
                for entry in self._decision_value(decision, "invocation_messages", ()) or ():
                    sequence = int(self._decision_value(entry, "sequence", 0))
                    if sequence in self._message_sequences:
                        continue
                    self._message_sequences.add(sequence)
                    self._message_overlays.append(
                        _MessageOverlay(
                            sequence=sequence,
                            content=str(self._decision_value(entry, "content", "")),
                            anchor=anchor,
                        )
                    )
                prepared["messages"] = self._inject_message_overlays(messages)

        return prepared, boundary_id, True

    def _inject_message_overlays(self, messages: "list[dict]") -> "list[dict]":
        overlays = sorted(self._message_overlays, key=lambda item: item.sequence)
        result: "list[dict]" = []
        inserted: "set[int]" = set()

        for overlay in overlays:
            if overlay.anchor == "root":
                result.append(self._invocation_message(overlay))
                inserted.add(overlay.sequence)

        prefix: "list[dict]" = []
        for message in messages:
            result.append(copy.deepcopy(message))
            prefix.append(message)
            anchor = _history_anchor(prefix)
            for overlay in overlays:
                if overlay.sequence not in inserted and overlay.anchor == anchor:
                    result.append(self._invocation_message(overlay))
                    inserted.add(overlay.sequence)

        # If a CLI compacted away an old anchor, retain the admitted
        # instruction at the current generation rather than silently losing it.
        for overlay in overlays:
            if overlay.sequence not in inserted:
                result.append(self._invocation_message(overlay))
        return result

    @staticmethod
    def _invocation_message(overlay: _MessageOverlay) -> dict:
        return {
            "role": "user",
            "blocks": [
                {
                    "type": "text",
                    "index": 0,
                    "text": f"[AGENCY INVOCATION MESSAGE]\n{overlay.content}",
                }
            ],
        }

    def _complete_model_request(
        self,
        boundary_id: "str | None",
        message: dict,
        *,
        controlled: bool,
        abort_event: "threading.Event | None" = None,
    ) -> bool:
        if not controlled or self._invocation is None or boundary_id is None:
            return not (abort_event is not None and abort_event.is_set())
        has_tool_calls = any(block.get("type") == "tool_use" for block in message.get("blocks", []))
        with self._control_lock:
            self._invocation._note_model_result(has_tool_calls=has_tool_calls)
            decision = self._checkpoint_invocation(
                f"{boundary_id}:post",
                allow_messages=False,
                phase=(
                    _CONTROL_PHASE_POST_GENERATION if has_tool_calls else _CONTROL_PHASE_CLOSING
                ),
                abort_event=abort_event,
            )
            if decision is None:
                return False
            self._raise_if_stopped(decision)
            return True

    def _complete_failed_model_request(
        self,
        boundary_id: "str | None",
        *,
        controlled: bool,
        abort_event: "threading.Event | None" = None,
    ) -> bool:
        if not controlled or self._invocation is None or boundary_id is None:
            return not (abort_event is not None and abort_event.is_set())
        with self._control_lock:
            decision = self._checkpoint_invocation(
                f"{boundary_id}:post-error",
                allow_messages=False,
                # Keep invocation messages sealed across a provider/CLI retry. An
                # identical retry reuses the cached pre-boundary assignment,
                # so accepting a new message here would otherwise strand it.
                phase=_CONTROL_PHASE_PRE_GENERATION,
                abort_event=abort_event,
            )
            if decision is None:
                return False
            self._raise_if_stopped(decision)
            return True

    def _checkpoint_invocation(
        self,
        boundary_id: str,
        *,
        allow_messages: bool,
        phase: str,
        abort_event: "threading.Event | None" = None,
    ):
        if abort_event is not None:
            interruptible = getattr(self._invocation, "_checkpoint_interruptibly", None)
            if callable(interruptible):
                return interruptible(
                    boundary_id,
                    allow_messages=allow_messages,
                    phase=phase,
                    abort_event=abort_event,
                )
            if abort_event.is_set():
                return None
        decision = self._invocation._checkpoint(
            boundary_id,
            allow_messages=allow_messages,
            phase=phase,
        )
        if abort_event is not None and abort_event.is_set():
            return None
        return decision

    def _interrupt_checkpoint_wait(self, abort_event: threading.Event) -> None:
        invocation = self._invocation
        if invocation is None:
            return
        abort = getattr(invocation, "_abort_checkpoint_wait", None)
        if callable(abort):
            abort(abort_event)
            return
        # Compatibility with early control implementations that exposed a
        # no-argument wake method under this name.
        interrupt = getattr(invocation, "_interrupt_checkpoint_wait", None)
        if callable(interrupt):
            try:
                interrupt(abort_event)
            except TypeError:
                interrupt()

    @staticmethod
    def _decision_value(item, name: str, default=None):
        return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)

    def _raise_if_stopped(self, decision) -> None:
        if self._decision_value(decision, "destroyed", False):
            raise _DispatchError(_AGENT_DESTROYED_MESSAGE, status_code=409, transient=False)
        if self._decision_value(decision, "cancelled", False):
            raise _DispatchError(_INVOCATION_CANCELLED_MESSAGE, status_code=409, transient=False)

    def _finalize_error(self, call_label: "str | None", error: BaseException) -> None:
        self._data_collector.finalize_stream(
            call_label,
            type="llm_stream_error",
            payloads=[{"error": f"{type(error).__name__}: {error}"}],
        )

    def _finalize_cancelled(self, call_label: "str | None") -> None:
        self._data_collector.finalize_stream(
            call_label,
            type="llm_stream_cancelled",
            payloads=[{"cancelled": True}],
        )

    def _build_kwargs(self, request: dict) -> dict:
        kwargs = self._backend.build_kwargs(request["messages"], request.get("tools"))
        if request.get("tool_choice") is not None:
            kwargs["tool_choice"] = request["tool_choice"]
        return kwargs

    def _run_stream_producer(
        self,
        request: dict,
        boundary_id: "str | None",
        controlled: bool,
        handle: "_StreamHandle",
    ) -> None:
        from ...profiler import agprof

        finalized = False

        def finalize_error(error: BaseException) -> None:
            nonlocal finalized
            if finalized:
                return
            self._finalize_error(handle.call_label, error)
            finalized = True

        def finalize_cancelled() -> None:
            nonlocal finalized
            if finalized:
                return
            self._finalize_cancelled(handle.call_label)
            finalized = True

        def finalize_success(payloads: "list[dict]") -> None:
            nonlocal finalized
            if finalized:
                return
            self._data_collector.finalize_stream(
                handle.call_label,
                type="llm_block",
                payloads=payloads,
            )
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

        try:
            if handle._cancel_event.is_set():
                finalize_cancelled()
                return
            with agprof.span("llm:attempt[0]", parent_context=self._parent_context) as attempt_span:
                _annotate(
                    attempt_span, model=self._backend.model, provider=type(self._backend).__name__
                )
                t0 = time.perf_counter()
                try:
                    stream_iter = iter(
                        self._backend.dispatch_stream(request, on_client=handle._set_stream_ref)
                    )
                    first_item = next(stream_iter)
                except StopIteration:
                    _annotate(attempt_span, outcome="success")
                    empty_message = {"role": "assistant", "blocks": []}
                    try:
                        completed = self._complete_model_request(
                            boundary_id,
                            empty_message,
                            controlled=controlled,
                            abort_event=handle._cancel_event,
                        )
                    except BaseException as e:
                        publish_error(
                            e,
                            status_code=500,
                            transient=False,
                        )
                        return
                    if not completed:
                        finalize_cancelled()
                        return
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
                        return
                    finalize_success([])
                    return
                except BAD_REQUEST_EXCS as e:
                    _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
                    try:
                        completed = self._complete_failed_model_request(
                            boundary_id,
                            controlled=controlled,
                            abort_event=handle._cancel_event,
                        )
                    except BaseException as control_error:
                        e = control_error
                        completed = True
                    if not completed:
                        finalize_cancelled()
                        return
                    publish_error(e, status_code=400, transient=False)
                    return
                except TRANSIENT_DISPATCH_EXCS as e:
                    _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
                    try:
                        completed = self._complete_failed_model_request(
                            boundary_id,
                            controlled=controlled,
                            abort_event=handle._cancel_event,
                        )
                    except BaseException as control_error:
                        e = control_error
                        completed = True
                    if not completed:
                        finalize_cancelled()
                        return
                    publish_error(e, status_code=503, transient=True)
                    return
                except BaseException as e:
                    # A producer thread must always publish a first terminal
                    # item. Otherwise the synchronous route remains blocked in
                    # handle.first() when setup or the first read fails.
                    _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
                    try:
                        completed = self._complete_failed_model_request(
                            boundary_id,
                            controlled=controlled,
                            abort_event=handle._cancel_event,
                        )
                    except BaseException as control_error:
                        e = control_error
                        completed = True
                    if not completed:
                        finalize_cancelled()
                        return
                    publish_error(e, status_code=500, transient=False)
                    return
                _annotate(attempt_span, ttft_ms=round((time.perf_counter() - t0) * 1000, 3))

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
                        self._data_collector.record_stream_delta(
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
                        block["ts_end"] = now
                        message = _blocks_to_message(blocks)
                        item = (
                            {"type": "delta", "content": text_piece}
                            if stream_item["block_type"] == "text" and text_piece
                            else None
                        )
                        handle.register_stream_exchange(item, response=message)
                except BaseException as e:
                    _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
                    try:
                        completed = self._complete_failed_model_request(
                            boundary_id,
                            controlled=controlled,
                            abort_event=handle._cancel_event,
                        )
                    except BaseException as control_error:
                        e = control_error
                        completed = True
                    if not completed:
                        finalize_cancelled()
                        return
                    publish_error(
                        e,
                        status_code=500,
                        transient=False,
                        streaming=False,
                        error=f"{type(e).__name__}: {e}",
                    )
                    return

                if handle._cancel_event.is_set():
                    finalize_cancelled()
                    return
                _annotate(
                    attempt_span,
                    outcome="success",
                    input_tokens=(usage or {}).get("prompt_tokens", 0),
                    output_tokens=(usage or {}).get("completion_tokens", 0),
                )
                message = _blocks_to_message(blocks)
                try:
                    completed = self._complete_model_request(
                        boundary_id,
                        message,
                        controlled=controlled,
                        abort_event=handle._cancel_event,
                    )
                except BaseException as e:
                    publish_error(
                        e,
                        status_code=500,
                        transient=False,
                        response=message,
                        usage=usage,
                        finish_reason=stop_reason,
                        streaming=False,
                        error=f"{type(e).__name__}: {e}",
                    )
                    return
                if not completed:
                    finalize_cancelled()
                    return
                enqueued = handle.register_stream_exchange(
                    {
                        "type": "done",
                        "message": message,
                        "usage": usage,
                        "stop_reason": stop_reason,
                    },
                    response=message,
                    usage=usage,
                    finish_reason=stop_reason,
                    streaming=False,
                )
                if not enqueued:
                    finalize_cancelled()
                    return
                finalize_success(message["blocks"])
        except BaseException as e:
            if not finalized:
                terminal_error = e
                # Failures in profiling or transcript setup still cross the
                # post-error control boundary before waking ``first()`` and
                # durably terminating the collector call label.
                try:
                    completed = self._complete_failed_model_request(
                        boundary_id,
                        controlled=controlled,
                        abort_event=handle._cancel_event,
                    )
                except BaseException as control_error:
                    terminal_error = control_error
                    completed = True
                if not completed:
                    finalize_cancelled()
                else:
                    publish_error(
                        terminal_error,
                        status_code=500,
                        transient=False,
                        streaming=False,
                        error=f"{type(terminal_error).__name__}: {terminal_error}",
                    )
        finally:
            handle._close_stream()
