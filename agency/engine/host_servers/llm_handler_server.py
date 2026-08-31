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
    from ...agDataCollector import agDataCollector

TRANSIENT_DISPATCH_EXCS = (
    RATE_LIMIT_EXCS + API_CONN_EXCS + API_ERROR_EXCS + (ssl.SSLError, OSError, httpx.TransportError)
)

_STREAM_QUEUE_MAXSIZE = 256


def _annotate(span, **metadata) -> None:
    annotate = getattr(span, "annotate", None)
    if annotate is not None:
        annotate(**metadata)


def _blocks_to_message(blocks: "dict[int, dict]") -> dict:
    return {"role": "assistant", "blocks": [blocks[i] for i in sorted(blocks)]}


class _DispatchError(Exception):
    """Raised by the non-streaming path; carries what build_app()'s route
    needs to translate this into an HTTP response."""

    def __init__(self, message: str, *, status_code: int, transient: bool) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.transient = transient


class _StreamHandle:
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
    def __init__(
        self, agconfig: "agConfig", data_collector: "agDataCollector", *, parent_context=None
    ) -> None:
        self._data_collector = data_collector
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

    def dispatch(self, request: dict) -> dict:
        from ...profiler import agprof

        self._data_collector.record_event(
            type="agent_state",
            payload={"state": "waiting_llm"},
            do_update=True,
            flush=True,
        )
        with agprof.span("llm:attempt[0]", parent_context=self._parent_context) as attempt_span:
            _annotate(
                attempt_span, model=self._backend.model, provider=type(self._backend).__name__
            )
            try:
                result = self._backend.dispatch(request)
            except BAD_REQUEST_EXCS as e:
                _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
                raise _DispatchError(str(e), status_code=400, transient=False) from e
            except TRANSIENT_DISPATCH_EXCS as e:
                _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
                raise _DispatchError(str(e), status_code=503, transient=True) from e
            except BaseException as e:
                _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
                raise
            usage = result["usage"]
            _annotate(
                attempt_span,
                outcome="success",
                input_tokens=(usage or {}).get("prompt_tokens", 0),
                output_tokens=(usage or {}).get("completion_tokens", 0),
            )
            self._record_exchange(request, result["message"], usage, result["stop_reason"])
            return result

    def start_stream(self, request: dict) -> "_StreamHandle":
        from ...profiler import agprof

        self._data_collector.record_event(
            type="agent_state",
            payload={"state": "waiting_llm"},
            do_update=True,
            flush=True,
        )
        q: "queue.Queue[dict]" = queue.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)
        cancel_event = threading.Event()
        handle = _StreamHandle(q, cancel_event)
        thread = agprof.spawn_traced(self._run_stream_producer, request, handle, daemon=True)
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

    def _run_stream_producer(self, request: dict, handle: "_StreamHandle") -> None:
        from ...profiler import agprof

        try:
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
                    handle.register_stream_exchange(
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

                if handle._cancel_event.is_set():
                    return
                _annotate(
                    attempt_span,
                    outcome="success",
                    input_tokens=(usage or {}).get("prompt_tokens", 0),
                    output_tokens=(usage or {}).get("completion_tokens", 0),
                )
                message = _blocks_to_message(blocks)
                handle.register_stream_exchange(
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
        finally:
            handle._close_stream()
