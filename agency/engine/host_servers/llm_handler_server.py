from __future__ import annotations

import itertools
import json
import queue
import ssl
import threading
import time
from typing import TYPE_CHECKING

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from ...llm import API_CONN_EXCS, API_ERROR_EXCS, BAD_REQUEST_EXCS, RATE_LIMIT_EXCS
from ...llm.agllm import agllm

if TYPE_CHECKING:
    from ...agconfig import agConfig

TRANSIENT_DISPATCH_EXCS = (
    RATE_LIMIT_EXCS + API_CONN_EXCS + API_ERROR_EXCS + (ssl.SSLError, OSError, httpx.TransportError)
)

_STREAM_QUEUE_MAXSIZE = 256


def _annotate(span, **metadata) -> None:
    annotate = getattr(span, "annotate", None)
    if annotate is not None:
        annotate(**metadata)


def _serialize_usage(usage) -> "dict | None":
    if usage is None:
        return None
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": getattr(usage, "total_tokens", None) or (prompt_tokens + completion_tokens),
    }


def _serialize_result_message(result) -> dict:
    choice = (result.choices or [None])[0]
    if choice is None:
        return {"role": "assistant", "content": None}
    message = choice.message
    out = {"role": "assistant", "content": getattr(message, "content", None)}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": getattr(tc, "id", "") or "",
                "type": "function",
                "function": {
                    "name": getattr(tc.function, "name", "") or "",
                    "arguments": getattr(tc.function, "arguments", "") or "",
                },
            }
            for tc in tool_calls
        ]
    return out


def _finish_reason(result) -> "str | None":
    choice = (result.choices or [None])[0]
    return getattr(choice, "finish_reason", None) if choice is not None else None


class _DispatchError(Exception):
    """Raised by the non-streaming path; carries what build_app()'s route
    needs to translate this into an HTTP response."""

    def __init__(self, message: str, *, status_code: int, transient: bool) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.transient = transient


class _StreamHandle:
    """Bundles one streaming dispatch's queue/thread/cancel-event. Not a
    thread itself -- start_stream() spawns the producer thread separately
    and assigns it here. first()/relay() are the only surface build_app()'s
    route touches; _run_stream_producer reaches in directly since it's an
    internal collaborator, not a public interface."""

    def __init__(self, q: "queue.Queue[dict]", cancel_event: threading.Event) -> None:
        self._queue = q
        self._cancel_event = cancel_event
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

    def register_stream_exchange(self, item: "dict | None" = None, **entry_fields) -> None:
        """Update this connection's transcript entry -- creating it on the
        first call -- with entry_fields, then push item (if given) onto the
        response queue. One call so the transcript and the streamed
        response can never drift out of sync with each other."""
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
            self._queue.put(item)

    def _set_stream_ref(self, stream) -> None:
        with self._stream_ref_lock:
            self._stream_ref = stream

    def _close_stream(self) -> None:
        with self._stream_ref_lock:
            if self._stream_closed or self._stream_ref is None:
                return
            self._stream_closed = True
            stream = self._stream_ref
        stream.close()

    def first(self) -> dict:
        return self._queue.get()

    def relay(self, first_item: dict):
        try:
            item = first_item
            while True:
                yield json.dumps(item) + "\n"
                if item["type"] in ("done", "error"):
                    return
                item = self._queue.get()
        finally:
            self.cancel()
            if self._thread is not None:
                self._thread.join(timeout=5.0)

    def cancel(self) -> None:
        self._cancel_event.set()
        self._close_stream()


class LlmHandlerServer:
    def __init__(self, agconfig: "agConfig", *, parent_context=None) -> None:
        self._handles: "list[_StreamHandle]" = []
        self._handles_lock = threading.Lock()
        # HTTP/UDS requests are handled on the host server's own thread, so
        # their contextvars do not automatically inherit the agent run span.
        # Keep the durable OTel context captured by HostServerManager and use
        # it explicitly for every LLM attempt span.
        self._parent_context = parent_context
        # Non-streaming exchanges only
        self._transcript: "list[dict]" = []
        self._transcript_lock = threading.Lock()
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
        entries = self.get_all_transcripts()
        if not entries:
            return []
        candidates = entries
        if needle is not None:
            matching = [
                e
                for e in entries
                if any(m.get("content") == needle for m in e["request"]["messages"])
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

    def dispatch(self, request: dict) -> dict:
        return self._dispatch_once(self._build_kwargs(request))

    def start_stream(self, request: dict) -> "_StreamHandle":
        from ...profiler import agprof

        kwargs = self._build_kwargs(request)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        q: "queue.Queue[dict]" = queue.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)
        cancel_event = threading.Event()
        handle = _StreamHandle(q, cancel_event)
        thread = agprof.spawn_traced(self._run_stream_producer, kwargs, handle, daemon=True)
        handle._thread = thread
        with self._handles_lock:
            self._handles.append(handle)
        thread.start()
        return handle

    def stop(self) -> None:
        with self._handles_lock:
            handles = list(self._handles)
        for handle in handles:
            handle.cancel()
        for handle in handles:
            if handle._thread is not None:
                handle._thread.join(timeout=5.0)

    def build_app(self) -> FastAPI:
        app = FastAPI()

        @app.post("/dispatch")
        def _dispatch(request: dict):
            if request.get("stream"):
                handle = self.start_stream(request)
                item = handle.first()
                if item["type"] == "error":
                    return JSONResponse(
                        {"error": {"message": item["message"], "transient": item["transient"]}},
                        status_code=item["status_code"],
                    )
                return StreamingResponse(handle.relay(item), media_type="application/x-ndjson")
            try:
                return JSONResponse(self.dispatch(request))
            except _DispatchError as e:
                return JSONResponse(
                    {"error": {"message": str(e), "transient": e.transient}},
                    status_code=e.status_code,
                )

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

    def _build_kwargs(self, request: dict) -> dict:
        kwargs = self._backend.build_kwargs(request["messages"], request.get("tools"))
        if request.get("tool_choice") is not None:
            kwargs["tool_choice"] = request["tool_choice"]
        return kwargs

    def _client_timeout(self) -> httpx.Timeout:
        get = self._agconfig.get
        return httpx.Timeout(
            connect=get("agllm", "http_connect_timeout", 10.0),
            read=get("agllm", "stream_timeout", 1200.0),
            write=get("agllm", "http_write_timeout", 10.0),
            pool=get("agllm", "http_pool_timeout", 10.0),
        )

    def _dispatch_once(self, kwargs: dict) -> dict:
        from ...profiler import agprof

        client = self._backend.make_client(self._client_timeout())
        try:
            with agprof.span("llm:attempt[0]", parent_context=self._parent_context) as attempt_span:
                _annotate(
                    attempt_span, model=self._backend.model, provider=type(self._backend).__name__
                )
                try:
                    result = client.chat.completions.create(**kwargs)
                except BAD_REQUEST_EXCS as e:
                    _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
                    raise _DispatchError(str(e), status_code=400, transient=False) from e
                except TRANSIENT_DISPATCH_EXCS as e:
                    _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
                    raise _DispatchError(str(e), status_code=503, transient=True) from e
                except BaseException as e:
                    _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
                    raise
                usage = _serialize_usage(getattr(result, "usage", None))
                _annotate(
                    attempt_span,
                    outcome="success",
                    input_tokens=(usage or {}).get("prompt_tokens", 0),
                    output_tokens=(usage or {}).get("completion_tokens", 0),
                )
                message = _serialize_result_message(result)
                finish_reason = _finish_reason(result)
                self._record_exchange(kwargs, message, usage, finish_reason)
                return {
                    "message": message,
                    "usage": usage,
                    "stop_reason": finish_reason,
                }
        finally:
            client.close()

    def _run_stream_producer(self, kwargs: dict, handle: "_StreamHandle") -> None:
        from ...profiler import agprof

        client = self._backend.make_client(self._client_timeout())
        handle._set_stream_ref(client)
        content_parts: "list[str]" = []
        tool_calls_raw: "dict[int, dict]" = {}
        try:
            with agprof.span("llm:attempt[0]", parent_context=self._parent_context) as attempt_span:
                _annotate(
                    attempt_span, model=self._backend.model, provider=type(self._backend).__name__
                )
                t0 = time.perf_counter()
                try:
                    stream_iter = iter(client.chat.completions.create(**kwargs))
                    first_chunk = next(stream_iter)
                except StopIteration:
                    _annotate(attempt_span, outcome="success")
                    empty_message = {"role": "assistant", "content": None}
                    handle.register_stream_exchange(
                        {"type": "done", "message": empty_message, "usage": None},
                        request=kwargs,
                        response=empty_message,
                        streaming=False,
                    )
                    return
                except BAD_REQUEST_EXCS as e:
                    _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
                    handle._queue.put(
                        {"type": "error", "message": str(e), "transient": False, "status_code": 400}
                    )
                    return
                except TRANSIENT_DISPATCH_EXCS as e:
                    _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
                    handle._queue.put(
                        {"type": "error", "message": str(e), "transient": True, "status_code": 503}
                    )
                    return
                _annotate(attempt_span, ttft_ms=round((time.perf_counter() - t0) * 1000, 3))

                usage = None
                handle.register_stream_exchange(
                    request=kwargs, response={"role": "assistant", "content": ""}
                )
                try:
                    for chunk in itertools.chain([first_chunk], stream_iter):
                        if handle._cancel_event.is_set():
                            break
                        chunk_usage = _serialize_usage(getattr(chunk, "usage", None))
                        if chunk_usage is not None:
                            usage = chunk_usage
                        text = self._accumulate_chunk(chunk, content_parts, tool_calls_raw)
                        chunk_finish_reason = _finish_reason(chunk)
                        fields = {
                            "response": {"role": "assistant", "content": "".join(content_parts)}
                        }
                        if chunk_usage is not None:
                            fields["usage"] = chunk_usage
                        if chunk_finish_reason is not None:
                            fields["finish_reason"] = chunk_finish_reason
                        item = {"type": "delta", "content": text} if text else None
                        handle.register_stream_exchange(item, **fields)
                except BaseException as e:
                    _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
                    error_item = {
                        "type": "error",
                        "message": str(e),
                        "transient": False,
                        "status_code": 500,
                    }
                    handle.register_stream_exchange(
                        error_item, streaming=False, error=f"{type(e).__name__}: {e}"
                    )
                    return

                _annotate(attempt_span, outcome="success")
                message = {"role": "assistant", "content": "".join(content_parts) or None}
                if tool_calls_raw:
                    message["tool_calls"] = [tool_calls_raw[i] for i in sorted(tool_calls_raw)]
                handle.register_stream_exchange(
                    {"type": "done", "message": message, "usage": usage},
                    response=message,
                    usage=usage,
                    streaming=False,
                )
        finally:
            handle._close_stream()

    @staticmethod
    def _accumulate_chunk(
        chunk, content_parts: "list[str]", tool_calls_raw: "dict[int, dict]"
    ) -> str:
        choice = (chunk.choices or [None])[0]
        if choice is None:
            return ""
        delta = choice.delta
        content = getattr(delta, "content", None) or ""
        if content:
            content_parts.append(content)
        for tc in getattr(delta, "tool_calls", None) or []:
            idx = getattr(tc, "index", 0)
            slot = tool_calls_raw.setdefault(
                idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            )
            tc_id = getattr(tc, "id", None)
            if tc_id:
                slot["id"] = tc_id
            fn = getattr(tc, "function", None)
            if fn is not None:
                name = getattr(fn, "name", None)
                if name:
                    slot["function"]["name"] += name
                arguments = getattr(fn, "arguments", None)
                if arguments:
                    slot["function"]["arguments"] += arguments
        return content
