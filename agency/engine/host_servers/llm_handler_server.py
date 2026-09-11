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

from ...llm import API_CONN_EXCS, API_ERROR_EXCS, BAD_REQUEST_EXCS, RATE_LIMIT_EXCS
from ...llm.agllm import agllm
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


def _payload_hash(payload: dict) -> str:
    """Content identity for one {role, **block} transcript payload.

    A tool_use/tool_result block is immutable once emitted -- its id/
    tool_call_id never changes -- but the exact same call gets
    re-serialized in a different (reduced) shape the moment it's replayed
    back into a later request through a harness's own wire protocol (e.g.
    native_harness's OpenAI-chatcompletions tool_calls array only carries
    id/name/arguments, dropping fields the original backend-native block
    carried like text/signature/data/citations/ts_start/ts_end, and often
    renumbering `index`). Hashing the whole payload would treat that
    reduced replay as new content and double-log every tool call/result,
    so key on the block's own stable identity instead when it has one."""
    role = payload.get("role")
    block_type = payload.get("type")
    if block_type == "tool_use" and payload.get("id"):
        identity: dict = {"role": role, "type": block_type, "id": payload["id"]}
    elif block_type == "tool_result" and payload.get("tool_call_id"):
        identity = {"role": role, "type": block_type, "tool_call_id": payload["tool_call_id"]}
    else:
        identity = payload
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()


def _classify_dispatch_exception(error: BaseException) -> "tuple[int, bool] | None":
    """(status_code, transient) for a known LLM-backend failure category, or
    None if *error* isn't one of them (caller re-raises it unmodified)."""
    if isinstance(error, BAD_REQUEST_EXCS):
        return 400, False
    if isinstance(error, TRANSIENT_DISPATCH_EXCS):
        return 503, True
    return None


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
        # Wake either side of the bounded queue. A disconnected consumer may
        # leave the producer waiting for capacity; a pre-first-item disconnect
        # may leave the consumer waiting for data.
        with self._queue_condition:
            # Publish and cancellation use this lock as their linearization
            # point: once cancellation wins, no later item or transcript field
            # can be admitted.
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
        recent_transcript: "list[dict] | None" = None,
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

        self._logged_block_hashes: "set[str]" = {
            _payload_hash({"role": message.get("role"), **block})
            for message in (recent_transcript or [])
            for block in (message.get("blocks") or [])
        }
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
    ) -> None:
        """Enrich this exchange's metadata block with its own new (non-
        cumulative) prompt token count and the skill/request it belongs to."""
        blocks = message.get("blocks") or []
        metadata_block = next((b for b in blocks if b.get("type") == "metadata"), None)
        if metadata_block is None:
            return
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
            payload={"state": "waiting_llm"},
            update_latest_snapshot=True,
            flush=True,
            call_label=call_label,
        )
        finalized = False

        def finalize_error(error: BaseException) -> None:
            nonlocal finalized
            self._finalize_terminal(call_label, kind="error", error=error)
            finalized = True

        def finalize_cancelled() -> None:
            nonlocal finalized
            self._finalize_terminal(call_label, kind="cancelled")
            finalized = True

        def complete_failure(error: BaseException) -> None:
            if abort_event is not None and abort_event.is_set():
                finalize_cancelled()
                raise _RequestAborted from error
            finalize_error(error)

        try:
            with agprof.span(
                "llm:attempt[0]",
                parent_context=(
                    self._profile_context_provider()
                    if self._profile_context_provider
                    else self._parent_context
                ),
            ) as attempt_span:
                _annotate(
                    attempt_span,
                    model=self._backend.model,
                    provider=type(self._backend).__name__,
                )
                t0 = time.perf_counter()
                try:
                    result = self._backend.dispatch(request)
                except BaseException as error:
                    with suppress(BaseException):
                        _annotate(
                            attempt_span,
                            outcome="failure",
                            error_type=type(error).__name__,
                        )
                    complete_failure(error)
                    classified = _classify_dispatch_exception(error)
                    if classified is None:
                        raise
                    status_code, transient = classified
                    raise _DispatchError(
                        str(error), status_code=status_code, transient=transient
                    ) from error

                ttft_ms = round((time.perf_counter() - t0) * 1000, 3)
                _annotate(
                    attempt_span,
                    outcome="success",
                    input_tokens=(result["usage"] or {}).get("prompt_tokens"),
                    output_tokens=(result["usage"] or {}).get("completion_tokens"),
                    ttft_ms=ttft_ms,
                )
                self._tag_metadata_block(request["messages"], result["message"], ttft_ms=ttft_ms)
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
            payload={"state": "waiting_llm"},
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
                self._finalize_terminal(call_label, kind="cancelled")
                raise _RequestAborted from error
            self._finalize_terminal(call_label, kind="error", error=error)
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
        error: "BaseException | None" = None,
    ) -> None:
        """Record this exchange's non-success outcome."""
        agname = getattr(self._data_logger, "_default_name", None)
        if kind == "cancelled":
            payloads = [{"cancelled": True}]
            suffix = "cancelled"
        else:
            suffix = f"{type(error).__name__}: {error}"
            payloads = [{"error": suffix}]
        self._data_logger.record_final_transcript(
            call_label,
            type=f"llm_stream_{kind}",
            payloads=payloads,
            term_message=f"[{agname}] LLM    ✗  {suffix}",
            print_to_terminal=False,
        )

    def _new_transcript_payloads(
        self, request: dict, response_message: "dict | None"
    ) -> "list[dict]":
        """Diff this exchange's full state (request + response) against
        every payload already logged this attempt, returning only what's
        new."""
        candidates: "list[dict]" = []
        for message in request.get("messages") or []:
            role = message.get("role")
            for block in message.get("blocks") or []:
                candidates.append({"role": role, **block})
        if response_message is not None:
            role = response_message.get("role", "assistant")
            for block in response_message.get("blocks") or []:
                candidates.append({"role": role, **block})
        new_payloads = []
        for payload in candidates:
            digest = _payload_hash(payload)
            if digest in self._logged_block_hashes:
                continue
            self._logged_block_hashes.add(digest)
            new_payloads.append(payload)
        return new_payloads

    def _finalize_success(
        self, call_label: "str | None", request: dict, response_message: "dict | None"
    ) -> None:
        payloads = self._new_transcript_payloads(request, response_message)
        if payloads:
            self._data_logger.record_final_transcript(
                call_label, type="llm_block", payloads=payloads
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
            self._finalize_terminal(handle.call_label, kind="error", error=error)
            finalized = True

        def finalize_cancelled() -> None:
            nonlocal finalized
            if finalized:
                return
            self._finalize_terminal(handle.call_label, kind="cancelled")
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

        try:
            if handle._cancel_event.is_set():
                finalize_cancelled()
                return
            with agprof.span(
                "llm:attempt[0]",
                parent_context=(
                    self._profile_context_provider()
                    if self._profile_context_provider
                    else self._parent_context
                ),
            ) as attempt_span:
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
                    finalize_success(empty_message)
                    return
                except BaseException as e:
                    # A producer thread must always publish a first terminal
                    # item. Otherwise the synchronous route remains blocked in
                    # handle.first() when setup or the first read fails.
                    _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
                    if handle._cancel_event.is_set():
                        finalize_cancelled()
                        return
                    classified = _classify_dispatch_exception(e)
                    status_code, transient = classified if classified is not None else (500, False)
                    publish_error(e, status_code=status_code, transient=transient)
                    return
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
                        block["ts_end"] = now
                        message = _blocks_to_message(blocks)
                        handle.register_stream_exchange(response=message)
                except BaseException as e:
                    _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
                    if handle._cancel_event.is_set():
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
                    input_tokens=(usage or {}).get("prompt_tokens"),
                    output_tokens=(usage or {}).get("completion_tokens"),
                )
                message = _blocks_to_message(blocks)
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
                self._tag_metadata_block(request["messages"], message, ttft_ms=ttft_ms)
                finalize_success(message)
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
