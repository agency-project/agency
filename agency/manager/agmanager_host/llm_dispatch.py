"""LLM dispatch, model resolution, and context-limit lookup for
`agmanager_host`.

The actual credentialed client construction (`ag.llm.backend.make_client()`)
and `chat.completions.create()` call only ever happen here -- this is the
one place real per-agent backend credentials exist. See
`agmanager_host.py`'s module docstring for the full two-server design.

Retry policy: exactly one dispatch attempt, no retry loop here -- same
reasoning as the old `agllm_terminus.py`'s identical comment. Once a
streaming response commits its first chunk there is no way to signal
"retry me" without a caller misinterpreting real partial output as garbage,
and retrying underneath whatever resilience the caller already has (a
harness CLI's own retry, or native's own bounded retry around this call)
would stack an uncoordinated second layer. Classify honestly, let the
caller decide."""

from __future__ import annotations

import json
import ssl
import sys
import time
from typing import TYPE_CHECKING

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ...llm import BAD_REQUEST_EXCS, API_CONN_EXCS, RATE_LIMIT_EXCS, API_ERROR_EXCS
from .config import AgHostAgentManagerFields
from .serialization import serialize_result, serialize_chunk

if TYPE_CHECKING:
    from ...agconfig import agConfig
    from ...agent import agent
    from .launch_state import LaunchRegistry

TRANSIENT_DISPATCH_EXCS = (
    RATE_LIMIT_EXCS + API_CONN_EXCS + API_ERROR_EXCS + (ssl.SSLError, OSError, httpx.TransportError)
)


def _annotate(span, **metadata) -> None:
    annotate = getattr(span, "annotate", None)
    if annotate is not None:
        annotate(**metadata)


class LLMDispatcher:
    """Owns the actual LLM call plus its request log."""

    def __init__(
        self, ag: "agent", registry: "LaunchRegistry", agconfig: "agConfig | None" = None
    ) -> None:
        self._ag = ag
        self._registry = registry
        self._agconfig = agconfig
        self.request_log: "list[dict]" = []

    def build_router(self) -> APIRouter:
        router = APIRouter()

        @router.post("/internal/dispatch")
        async def dispatch(request: Request):
            body = await request.json()
            token = body.get("token")
            launch = self._registry.get(token)
            if launch is None:
                return JSONResponse(
                    {"error": {"message": "unknown or missing token"}}, status_code=401
                )
            kwargs = body.get("kwargs") or {}
            with self._registry.lock:
                self.request_log.append({"token": token, "model": kwargs.get("model", "")})
            timeout_s = AgHostAgentManagerFields(self._agconfig).request_timeout_s
            client = self._ag.llm.backend.make_client(httpx.Timeout(timeout_s))
            model = kwargs.get("model", "")
            provider = type(self._ag.llm.backend).__name__

            if kwargs.get("stream"):
                return self._dispatch_stream(token, launch, client, kwargs, model, provider)
            return self._dispatch_once(token, launch, client, kwargs, model, provider)

        @router.post("/internal/resolve_model")
        async def resolve_model(request: Request):
            body = await request.json()
            if self._registry.get(body.get("token")) is None:
                return JSONResponse({"error": "unknown or missing token"}, status_code=401)
            return JSONResponse({"model": self._ag.llm.backend.model or ""})

        @router.post("/internal/context_limit")
        async def context_limit(request: Request):
            body = await request.json()
            if self._registry.get(body.get("token")) is None:
                return JSONResponse({"error": "unknown or missing token"}, status_code=401)
            from ...llm.agllm import agllm

            return JSONResponse({"context_limit": agllm.fetch_context_limit(self._ag.llm.backend)})

        return router

    def _dispatch_once(self, token, launch, client, kwargs, model, provider) -> JSONResponse:
        from ...profiler import agprof, agprof_derive

        dispatch_start_perf_ns = time.perf_counter_ns()
        dispatch_start_wall_ns = time.time_ns()
        run_context = launch.run_context
        span_scope = (
            agprof.span("llm:attempt[0]")
            if run_context is None
            else agprof.span("llm:attempt[0]", parent_context=run_context)
        )
        with span_scope as attempt_span:
            _annotate(attempt_span, **launch.span_attributes)
            _annotate(attempt_span, model=model, provider=provider)
            try:
                result = client.chat.completions.create(**kwargs)
                serialized = serialize_result(result)
            except BAD_REQUEST_EXCS as e:
                _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
                return JSONResponse(
                    {"error": {"message": str(e), "transient": False}}, status_code=400
                )
            except TRANSIENT_DISPATCH_EXCS as e:
                _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
                return JSONResponse(
                    {"error": {"message": str(e), "transient": True}}, status_code=503
                )
            except BaseException as e:
                _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
                raise
            usage = serialized.get("usage") or {}
            _annotate(
                attempt_span,
                outcome="success",
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
            )
            if serialized["choices"]:
                response_message = serialized["choices"][0]["message"]
                self._registry.record_transcript(token, kwargs.get("messages"), response_message)
                if not launch.exact_tool_events:
                    # Known gap vs. the old agprof_ingest.py: `before_derive_
                    # tools` there called `reconcile_derived_tool_call_ids`, a
                    # short grace-period wait that lets an in-flight Claude
                    # PostToolUse hook win a race against this transcript-
                    # derived fallback for the same tool call, cancelling the
                    # provisional exact span so a completed tool never shows up
                    # twice. Not ported here yet -- a no-op passthrough, so a
                    # tight race can double-count a tool call's span. Revisit
                    # before this replaces agprof_ingest.py for real.
                    agprof_derive.on_dispatch(
                        token,
                        kwargs.get("messages"),
                        response_message,
                        start_perf_ns=dispatch_start_perf_ns,
                        start_wall_ns=dispatch_start_wall_ns,
                        end_perf_ns=time.perf_counter_ns(),
                        end_wall_ns=time.time_ns(),
                        parent_context=run_context,
                        span_attributes=launch.span_attributes,
                        skip_tool_call_ids=set(launch.exact_tool_call_ids),
                        before_derive_tools=lambda call_ids: call_ids,
                    )
            return JSONResponse(serialized)

    def _dispatch_stream(self, token, launch, client, kwargs, model, provider) -> StreamingResponse:
        from ...profiler import agprof

        kwargs = dict(kwargs)
        stream_options = kwargs.get("stream_options")
        stream_options = dict(stream_options) if isinstance(stream_options, dict) else {}
        stream_options["include_usage"] = True
        kwargs["stream_options"] = stream_options

        run_context = launch.run_context
        span_scope = (
            agprof.span("llm:attempt[0]")
            if run_context is None
            else agprof.span("llm:attempt[0]", parent_context=run_context)
        )
        attempt_span = span_scope.__enter__()
        _annotate(attempt_span, **launch.span_attributes)
        attempt_t0 = time.perf_counter()
        dispatch_start_perf_ns = time.perf_counter_ns()
        dispatch_start_wall_ns = time.time_ns()
        span_closed = False

        def _finish_span(exc_info=(None, None, None)) -> None:
            nonlocal span_closed
            if not span_closed:
                span_closed = True
                span_scope.__exit__(*exc_info)

        try:
            stream_iter = iter(client.chat.completions.create(**kwargs))
            first_chunk = next(stream_iter)
            ttft_ms = (time.perf_counter() - attempt_t0) * 1000
            _annotate(attempt_span, model=model, provider=provider, ttft_ms=round(ttft_ms, 3))
        except StopIteration:
            _annotate(attempt_span, model=model, provider=provider, outcome="success")
            _finish_span()
            return StreamingResponse(iter(["data: [DONE]\n\n"]), media_type="text/event-stream")
        except BAD_REQUEST_EXCS as e:
            _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
            _finish_span((type(e), e, e.__traceback__))
            return JSONResponse({"error": {"message": str(e), "transient": False}}, status_code=400)
        except TRANSIENT_DISPATCH_EXCS as e:
            _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
            _finish_span((type(e), e, e.__traceback__))
            return JSONResponse({"error": {"message": str(e), "transient": True}}, status_code=503)
        except BaseException as e:
            _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
            _finish_span((type(e), e, e.__traceback__))
            raise

        content_parts: "list[str]" = []
        tool_calls_raw: "dict[int, dict]" = {}

        def _accumulate_and_serialize(chunk) -> dict:
            serialized = serialize_chunk(chunk)
            for choice in serialized["choices"]:
                delta = choice["delta"]
                if delta.get("content"):
                    content_parts.append(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tool_calls_raw.setdefault(
                        idx,
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn_delta = tc.get("function") or {}
                    if fn_delta.get("name"):
                        slot["function"]["name"] += fn_delta["name"]
                    if fn_delta.get("arguments"):
                        slot["function"]["arguments"] += fn_delta["arguments"]
            message = {"role": "assistant", "content": "".join(content_parts) or None}
            if tool_calls_raw:
                message["tool_calls"] = [tool_calls_raw[i] for i in sorted(tool_calls_raw)]
            self._registry.record_transcript(token, kwargs.get("messages"), message)
            return serialized

        def sse_gen():
            try:
                yield f"data: {json.dumps(_accumulate_and_serialize(first_chunk))}\n\n"
                for chunk in stream_iter:
                    yield f"data: {json.dumps(_accumulate_and_serialize(chunk))}\n\n"
            except BaseException as e:
                _annotate(attempt_span, outcome="failure", error_type=type(e).__name__)
                raise
            else:
                _annotate(attempt_span, outcome="success")
                transcript = self._registry.transcript_for_token(token)
                if transcript and not launch.exact_tool_events:
                    from ...profiler import agprof_derive

                    # See _dispatch_once's identical call for the known
                    # before_derive_tools gap (no reconcile grace period
                    # ported yet).
                    agprof_derive.on_dispatch(
                        token,
                        transcript[:-1],
                        transcript[-1],
                        start_perf_ns=dispatch_start_perf_ns,
                        start_wall_ns=dispatch_start_wall_ns,
                        end_perf_ns=time.perf_counter_ns(),
                        end_wall_ns=time.time_ns(),
                        parent_context=run_context,
                        span_attributes=launch.span_attributes,
                        skip_tool_call_ids=set(launch.exact_tool_call_ids),
                        before_derive_tools=lambda call_ids: call_ids,
                    )
                yield "data: [DONE]\n\n"

        class _ProfiledStreamingResponse(StreamingResponse):
            async def __call__(self, scope, receive, send) -> None:
                exc_info = (None, None, None)
                try:
                    await super().__call__(scope, receive, send)
                except BaseException:
                    exc_info = sys.exc_info()
                    raise
                finally:
                    _finish_span(exc_info)

        return _ProfiledStreamingResponse(sse_gen(), media_type="text/event-stream")


__all__ = ["LLMDispatcher", "TRANSIENT_DISPATCH_EXCS"]
