"""Host-side credential-holding LLM dispatch terminus.

`agproxy_llm.py` is (per docs/Design_harness_integration.md) moving to run
*inside* the sandbox container so every engine's LLM traffic transits one
place regardless of where it runs. Real per-agent backend credentials
(`ag.llm.backend`'s API keys) must never cross into the container, though --
so this module is the one place that still holds them and performs the
actual `client.chat.completions.create()` call. `agproxy_llm`'s routes keep
all their routing/translation logic (bearer-token extraction, Anthropic
Messages/OpenAI Responses <-> chat-completions reshaping) but no longer
construct a real backend client themselves; they forward the token plus
already-uniform chat-completions kwargs here over HTTP (a Unix domain
socket once agproxy_llm actually runs in-container; a plain loopback TCP
connection is equally valid when both sides are host-side, e.g. in tests
or before that relocation lands) and reconstruct SDK response objects from
what comes back, so `agproxy_llm_adapters.py`'s contract (real
`ChatCompletion`/`ChatCompletionChunk` objects, not dicts) is unchanged.

Deliberately holds its own token<->agent registry rather than sharing
agProxyLLM's: this is the object that's meant to keep working, on the host,
once agProxyLLM's routing half is a separate in-container process -- it
shouldn't reach into a sibling object's in-process state to do its job.
"""

from __future__ import annotations

import json
import ssl
import sys
import threading
import time
import uuid
from typing import TYPE_CHECKING

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..agconfig import GlobalConfigParam, _AgConfigViewBase
from ..agllm_backends import BAD_REQUEST_EXCS, API_CONN_EXCS, RATE_LIMIT_EXCS, API_ERROR_EXCS
from ..profiler import agprof, agprof_derive
from .agprof_ingest import get_shared_profiler_ingest

if TYPE_CHECKING:
    from ..agconfig import agConfig
    from ..agent import agent


# ---------------------------------------------------------------------------
# Retry policy -- deliberately NOT a retry loop. See module docstring's
# retry section (and the design discussion that produced it): this
# terminus always streams, and once it commits to a 200 status line (which
# Starlette sends before the generator produces even its first item),
# there is no way to signal "actually, retry me" without a caller
# misinterpreting genuinely-sent partial output as something to discard.
# Retrying here anyway -- with its own backoff -- would also stack an
# invisible second retry/timeout layer underneath whatever an external
# harness's own CLI already does on its end (they have their own
# resilience; that's how they survive real network conditions at all),
# risking duplicate real provider calls with no coordination between the
# two layers. So: exactly one attempt, classified honestly, no sleep, no
# loop. `TRANSIENT_DISPATCH_EXCS` (503, safe to retry) vs
# `BAD_REQUEST_EXCS` (400, retrying would never help) reuse the exact
# exception tuples `agllm.py`'s own retry loop already classifies by --
# same taxonomy, just not the same layer's job to act on it. The one
# caller that currently NEEDS retry protection at all -- native.py's
# in-container loop, which has no other resilience layer under it the way
# a harness CLI does -- implements its own bounded retry around the
# *outer* call to this route (`_dispatch_via_terminus` in
# `_native_in_container_entrypoint.py`), entirely independent of this
# module and of whatever any harness does.
# ---------------------------------------------------------------------------

TRANSIENT_DISPATCH_EXCS = (
    RATE_LIMIT_EXCS + API_CONN_EXCS + API_ERROR_EXCS + (ssl.SSLError, OSError, httpx.TransportError)
)


# ---------------------------------------------------------------------------
# Duck-typed response serialization -- NOT `result.model_dump()`/
# `chunk.model_dump_json()`. Confirmed a real gap during development: the
# Anthropic/Bedrock backend's non-streaming response
# (`_AnthropicNonStreamResponse`) and streaming chunks (`_FakeChunk` et al,
# in `agllm_backends/anthropic.py`) are lightweight, `__slots__`-based
# compatibility objects -- not real OpenAI SDK pydantic models -- built only
# far enough to support `agllm.py`'s own in-process attribute-access
# reassembly loop (`chunk.choices[0].delta.content`, `.tool_calls`,
# `chunk.usage`), which never needed a serialization method since it's never
# crossed a wire. Dispatching over HTTP (this module's entire purpose)
# requires serializing regardless of which backend produced the response, so
# these helpers duck-type on that same minimal attribute surface instead of
# assuming a pydantic method exists.
# ---------------------------------------------------------------------------


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


def _serialize_tool_calls(tool_calls) -> "list | None":
    if not tool_calls:
        return None
    return [
        {
            "index": getattr(tc, "index", i),
            "id": getattr(tc, "id", "") or "",
            "type": "function",
            "function": {
                "name": getattr(tc.function, "name", "") or "",
                "arguments": getattr(tc.function, "arguments", "") or "",
            },
        }
        for i, tc in enumerate(tool_calls)
    ]


def _serialize_result(result) -> dict:
    """A non-streaming chat-completion-shaped response -> a plain
    ChatCompletion-shaped dict, duck-typed off `.choices[0].message.
    {content,tool_calls}` -- see module-level docstring above."""
    choices = []
    for i, choice in enumerate(result.choices or []):
        message = choice.message
        choices.append(
            {
                "index": getattr(choice, "index", i),
                "finish_reason": getattr(choice, "finish_reason", None) or "stop",
                "message": {
                    "role": "assistant",
                    "content": getattr(message, "content", None),
                    "tool_calls": _serialize_tool_calls(getattr(message, "tool_calls", None)),
                },
            }
        )
    return {
        "id": getattr(result, "id", "") or "",
        "object": "chat.completion",
        "created": getattr(result, "created", 0) or 0,
        "model": getattr(result, "model", "") or "",
        "choices": choices,
        "usage": _serialize_usage(getattr(result, "usage", None)),
    }


def _serialize_chunk(chunk) -> dict:
    """One streaming chunk -> a plain ChatCompletionChunk-shaped dict,
    duck-typed off `.choices[0].delta.{content,tool_calls}` -- see
    module-level docstring above."""
    choices = []
    for i, choice in enumerate(chunk.choices or []):
        delta = choice.delta
        choices.append(
            {
                "index": getattr(choice, "index", i),
                "finish_reason": getattr(choice, "finish_reason", None),
                "delta": {
                    "content": getattr(delta, "content", None),
                    "tool_calls": _serialize_tool_calls(getattr(delta, "tool_calls", None)),
                },
            }
        )
    return {
        "id": getattr(chunk, "id", "") or "",
        "object": "chat.completion.chunk",
        "created": getattr(chunk, "created", 0) or 0,
        "model": getattr(chunk, "model", "") or "",
        "choices": choices,
        "usage": _serialize_usage(getattr(chunk, "usage", None)),
    }


# See agproxy_llm.py's identical note: FastAPI/Starlette resolve a route
# handler's parameter annotations from the function's *module-level*
# globals, so `Request` must be importable from here, not imported inside
# `_build_app()`.


class _AgLLMTerminusFields:
    bind_host = GlobalConfigParam("agllm_terminus", default="127.0.0.1")
    request_timeout_s = GlobalConfigParam("agllm_terminus", default=300)

    def __init__(self, agconfig: "agConfig | None" = None) -> None:
        self._agconfig = agconfig


class agLLMTerminusConfig(_AgConfigViewBase):
    _OWNER = "agllm_terminus"


class _ProfiledStreamingResponse(StreamingResponse):
    """Close a provider-attempt span after Starlette drains its body.

    The terminus has to pull the first provider chunk before constructing the
    response so pre-stream failures can still become an HTTP 400/503.  Usage,
    however, normally arrives in the final chunk.  Keeping the span scope on
    the response bridges those two points without buffering the provider
    stream or changing when the caller receives chunks.
    """

    def __init__(self, *args, finish_span, stream_finished, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._finish_span = finish_span
        self._stream_finished = stream_finished

    async def __call__(self, scope, receive, send) -> None:
        exc_info = (None, None, None)
        try:
            await super().__call__(scope, receive, send)
        except BaseException:
            exc_info = sys.exc_info()
            raise
        finally:
            try:
                self._finish_span(exc_info)
            finally:
                self._stream_finished()


def _annotate_span(span, **metadata) -> None:
    """Annotate a concrete span handle, or do nothing when profiling is off."""
    annotate = getattr(span, "annotate", None)
    if annotate is not None:
        annotate(**metadata)


def _usage_metrics(usage: "dict | None") -> dict:
    usage = usage or {}
    return {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
    }


class agLLMTerminus:
    """The one place a real per-agent LLM backend client is ever
    constructed. One process-wide instance, shared the same way
    `agProxyLLM`'s shared gateway is (see `get_shared_terminus` below)."""

    def __init__(self, agconfig: "agConfig | None" = None) -> None:
        self._agconfig = agconfig
        self._agents_by_token: "dict[str, agent]" = {}
        self._lock = threading.Lock()
        self._app = self._build_app()
        self._server = None
        self._thread: "threading.Thread | None" = None
        self.base_url: "str | None" = None
        self._uds_server = None
        self._uds_thread: "threading.Thread | None" = None
        self.uds_path: "str | None" = None
        # Every successfully-authenticated dispatch, appended as
        # {"token", "model"} -- this is the one place a test (or anything
        # else) can prove a real credentialed call actually happened,
        # regardless of which engine or process routed the request here.
        # Mirrors agProxyLLM.request_log's role for its own routing/
        # translation layer, but this one stays meaningful even once that
        # layer runs inside the container (a different process from
        # whatever's asking) -- this object never does.
        self.request_log: "list[dict]" = []
        # Phase 5 (history unification, docs/Design_harness_integration.md):
        # every dispatch's `kwargs["messages"]` already carries the FULL
        # conversation so far -- every wire-format-translating engine
        # (claude_code.py, codex.py, ...) resends it whole on each turn,
        # same as any stateless chat-completions caller must. So the LAST
        # dispatch recorded for a token, request-messages plus this turn's
        # response appended, already IS that token's complete transcript --
        # no need to accumulate turn-by-turn ourselves. Overwritten (not
        # appended) each dispatch for exactly that reason. Read by a
        # harness backend after its process exits, before `unregister()`
        # discards it (see `transcript_for_token`).
        self._last_transcript_by_token: "dict[str, list[dict]]" = {}
        self._stream_condition = threading.Condition()
        self._active_streams = 0

    # -- token <-> agent registry ---------------------------------------
    # Same shape as agProxyLLM's own registry -- see that module's
    # docstring for why a per-launch bearer token, not a per-port server,
    # is what routes a request to the right agent's real credentials.

    def register(self, token: str, ag: "agent") -> None:
        with self._lock:
            self._agents_by_token[token] = ag

    def unregister(self, token: str) -> None:
        with self._lock:
            self._agents_by_token.pop(token, None)
            self._last_transcript_by_token.pop(token, None)
        agprof_derive.forget(token)

    def _agent_for_token(self, token: "str | None"):
        if token is None:
            return None
        with self._lock:
            return self._agents_by_token.get(token)

    def transcript_for_token(self, token: "str | None") -> "list[dict] | None":
        """The full conversation (request messages + final response,
        see `_last_transcript_by_token`'s docstring) recorded for *token*'s
        most recent dispatch, or None if nothing was ever dispatched for
        it. Callers must read this before `unregister(token)`, which
        discards it."""
        if token is None:
            return None
        with self._lock:
            transcript = self._last_transcript_by_token.get(token)
            return list(transcript) if transcript is not None else None

    def _record_transcript(
        self, token: "str | None", request_messages, response_message: dict
    ) -> None:
        if token is None:
            return
        with self._lock:
            self._last_transcript_by_token[token] = list(request_messages or []) + [
                response_message
            ]

    # -- app / route ------------------------------------------------------

    def _build_app(self):
        app = FastAPI()

        @app.post("/internal/validate_token")
        async def validate_token(request: Request):
            # Lets agproxy_llm's own auth gate (every route's 401 check)
            # work without a local token registry of its own -- required
            # once that routing/translation layer can run somewhere with no
            # Python object shared with this process at all (inside the
            # sandbox container), where a same-process dict lookup is
            # simply impossible.
            body = await request.json()
            valid = self._agent_for_token(body.get("token")) is not None
            return JSONResponse({"valid": valid})

        @app.post("/internal/dispatch")
        async def dispatch(request: Request):
            body = await request.json()
            token = body.get("token")
            ag = self._agent_for_token(token)
            if ag is None:
                return JSONResponse(
                    {"error": {"message": "unknown or missing token"}}, status_code=401
                )
            kwargs = body.get("kwargs") or {}
            with self._lock:
                self.request_log.append({"token": token, "model": kwargs.get("model", "")})
            timeout_s = _AgLLMTerminusFields(self._agconfig).request_timeout_s
            client = ag.llm.backend.make_client(httpx.Timeout(timeout_s))
            model = kwargs.get("model", "")
            provider = type(ag.llm.backend).__name__
            profiler_ingest = get_shared_profiler_ingest()
            run_context = profiler_ingest.context_for_token(token)
            run_attributes = profiler_ingest.attributes_for_token(token)

            def _attempt_span():
                if run_context is None:
                    return agprof.span("llm:attempt[0]")
                return agprof.span("llm:attempt[0]", parent_context=run_context)

            if kwargs.get("stream"):
                # OpenAI-compatible providers only include the final usage
                # chunk when explicitly requested.  Anthropic/Bedrock's
                # compatibility client accepts and discards this option while
                # still producing its own final usage chunk.
                kwargs = dict(kwargs)
                stream_options = kwargs.get("stream_options")
                stream_options = dict(stream_options) if isinstance(stream_options, dict) else {}
                stream_options["include_usage"] = True
                kwargs["stream_options"] = stream_options

                # Force the FIRST chunk before returning any HTTP response
                # at all -- this is what makes the 503-vs-400 classification
                # below meaningful. A `StreamingResponse` commits its status
                # line before its generator produces anything, so any
                # exception raised only once we're already inside that
                # generator can no longer be signaled to the caller as
                # "retry me" -- see this module's retry-policy comment
                # above for why that's a hard constraint, not an oversight.
                attempt_scope = _attempt_span()
                attempt_span = attempt_scope.__enter__()
                _annotate_span(attempt_span, **run_attributes)
                attempt_t0 = time.perf_counter()
                # M3 (docs/Design_profiler_harness_integration.md §5.2):
                # separate clock captures for agprof_derive's turn/tool
                # spans, taken at the same instant as attempt_t0 above but
                # in agprof's own clock domain (perf_counter_ns for
                # ordering/duration, time_ns for OTel's wall-clock span
                # timestamps) -- see record_derived_span's docstring.
                dispatch_start_perf_ns = time.perf_counter_ns()
                dispatch_start_wall_ns = time.time_ns()
                span_closed = False

                def _finish_span(exc_info=(None, None, None)) -> None:
                    nonlocal span_closed
                    if not span_closed:
                        span_closed = True
                        attempt_scope.__exit__(*exc_info)

                try:
                    stream_iter = iter(client.chat.completions.create(**kwargs))
                    first_chunk = next(stream_iter)
                    ttft_ms = (time.perf_counter() - attempt_t0) * 1000
                    _annotate_span(
                        attempt_span,
                        model=model,
                        provider=provider,
                        ttft_ms=round(ttft_ms, 3),
                    )
                except StopIteration:
                    # A genuinely empty stream -- not an error, just
                    # nothing to reassemble or forward.
                    elapsed_ms = (time.perf_counter() - attempt_t0) * 1000
                    _annotate_span(
                        attempt_span,
                        model=model,
                        provider=provider,
                        outcome="success",
                        ttft_ms=None,
                        generation_ms=round(elapsed_ms, 3),
                        input_tokens=0,
                        output_tokens=0,
                    )
                    _finish_span()
                    return StreamingResponse(
                        iter(["data: [DONE]\n\n"]), media_type="text/event-stream"
                    )
                except BAD_REQUEST_EXCS as e:
                    _annotate_span(
                        attempt_span,
                        model=model,
                        provider=provider,
                        outcome="failure",
                        error_type=type(e).__name__,
                        status_code=400,
                        transient=False,
                        input_tokens=0,
                        output_tokens=0,
                    )
                    _finish_span(sys.exc_info())
                    try:
                        client.close()
                    except Exception as close_error:
                        print(
                            "[agllm_terminus] WARNING: failed to close client after "
                            f"bad request: {close_error}"
                        )
                    return JSONResponse(
                        {"error": {"message": str(e), "transient": False}}, status_code=400
                    )
                except TRANSIENT_DISPATCH_EXCS as e:
                    _annotate_span(
                        attempt_span,
                        model=model,
                        provider=provider,
                        outcome="failure",
                        error_type=type(e).__name__,
                        status_code=503,
                        transient=True,
                        input_tokens=0,
                        output_tokens=0,
                    )
                    _finish_span(sys.exc_info())
                    try:
                        client.close()
                    except Exception as close_error:
                        print(
                            "[agllm_terminus] WARNING: failed to close client after "
                            f"transient error: {close_error}"
                        )
                    return JSONResponse(
                        {"error": {"message": str(e), "transient": True}}, status_code=503
                    )
                except BaseException as e:
                    _annotate_span(
                        attempt_span,
                        model=model,
                        provider=provider,
                        outcome="failure",
                        error_type=type(e).__name__,
                        input_tokens=0,
                        output_tokens=0,
                    )
                    _finish_span(sys.exc_info())
                    try:
                        client.close()
                    except Exception as close_error:
                        print(
                            "[agllm_terminus] WARNING: failed to close client after "
                            f"unexpected dispatch error: {close_error}"
                        )
                    raise

                # Reassemble the streamed deltas into one final message for
                # the transcript (Phase 5) -- mirrors the exact same
                # concatenate-by-index logic every real streaming consumer
                # in this codebase already does (agllm.py's own call(),
                # _native_in_container_entrypoint.py's _dispatch_via_terminus)
                # -- purely for recording; the chunks forwarded to the
                # actual caller below are untouched.
                content_parts: "list[str]" = []
                tool_calls_raw: "dict[int, dict]" = {}
                usage: "dict | None" = None

                def _accumulate_and_serialize(chunk) -> dict:
                    nonlocal usage
                    serialized = _serialize_chunk(chunk)
                    chunk_usage = serialized.get("usage")
                    if chunk_usage is not None:
                        if usage is None:
                            usage = chunk_usage
                        else:
                            # Some compatibility backends report prompt and
                            # completion counts in different chunks.  Retain
                            # the latest non-zero value for each side.
                            prompt_tokens = chunk_usage.get("prompt_tokens") or usage.get(
                                "prompt_tokens", 0
                            )
                            completion_tokens = chunk_usage.get("completion_tokens") or usage.get(
                                "completion_tokens", 0
                            )
                            usage = {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": chunk_usage.get("total_tokens")
                                or (prompt_tokens + completion_tokens),
                            }
                    for choice in serialized["choices"]:
                        delta = choice["delta"]
                        if delta.get("content"):
                            content_parts.append(delta["content"])
                        for tc in delta.get("tool_calls") or []:
                            idx = tc.get("index", 0)
                            slot = tool_calls_raw.setdefault(
                                idx,
                                {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                },
                            )
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            fn_delta = tc.get("function") or {}
                            if fn_delta.get("name"):
                                slot["function"]["name"] += fn_delta["name"]
                            if fn_delta.get("arguments"):
                                slot["function"]["arguments"] += fn_delta["arguments"]
                    # Record after EVERY chunk, not just once at the end --
                    # a real client (agproxy_llm's own SSE consumer) closes
                    # the connection as soon as it sees `[DONE]`, which can
                    # race Starlette's own generator-draining against that
                    # disconnect and skip code placed after the loop
                    # entirely (found via a real end-to-end test: the
                    # transcript came back empty even though every chunk
                    # had clearly been forwarded). Overwriting on every
                    # chunk is cheap (a dict assignment under a lock) and
                    # means the recorded state is never more than one chunk
                    # stale even in the worst case.
                    message = {"role": "assistant", "content": "".join(content_parts) or None}
                    if tool_calls_raw:
                        message["tool_calls"] = [tool_calls_raw[i] for i in sorted(tool_calls_raw)]
                    self._record_transcript(token, kwargs.get("messages"), message)
                    return serialized

                def sse_gen():
                    try:
                        yield f"data: {json.dumps(_accumulate_and_serialize(first_chunk))}\n\n"
                        for chunk in stream_iter:
                            yield f"data: {json.dumps(_accumulate_and_serialize(chunk))}\n\n"
                    except BaseException as e:
                        elapsed_ms = (time.perf_counter() - attempt_t0) * 1000
                        _annotate_span(
                            attempt_span,
                            model=model,
                            provider=provider,
                            outcome="failure",
                            error_type=type(e).__name__,
                            generation_ms=round(max(0.0, elapsed_ms - ttft_ms), 3),
                            **_usage_metrics(usage),
                        )
                        raise
                    else:
                        elapsed_ms = (time.perf_counter() - attempt_t0) * 1000
                        _annotate_span(
                            attempt_span,
                            model=model,
                            provider=provider,
                            outcome="success",
                            generation_ms=round(max(0.0, elapsed_ms - ttft_ms), 3),
                            **_usage_metrics(usage),
                        )
                        # M3: the last chunk's _accumulate_and_serialize()
                        # call already recorded the final transcript above --
                        # deriving here, once per dispatch on completion,
                        # avoids the per-chunk _record_transcript() calls
                        # manufacturing a turn per chunk (see module
                        # docstring's compaction/chunking risk note).
                        transcript = self.transcript_for_token(token)
                        if transcript and not profiler_ingest.has_exact_events(token):
                            agprof_derive.on_dispatch(
                                token,
                                transcript[:-1],
                                transcript[-1],
                                start_perf_ns=dispatch_start_perf_ns,
                                start_wall_ns=dispatch_start_wall_ns,
                                end_perf_ns=time.perf_counter_ns(),
                                end_wall_ns=time.time_ns(),
                                parent_context=run_context,
                                span_attributes=run_attributes,
                                skip_tool_call_ids=profiler_ingest.exact_tool_call_ids(token),
                                before_derive_tools=lambda call_ids: (
                                    profiler_ingest.reconcile_derived_tool_call_ids(token, call_ids)
                                ),
                            )
                        yield "data: [DONE]\n\n"

                self._stream_started()
                try:
                    return _ProfiledStreamingResponse(
                        sse_gen(),
                        media_type="text/event-stream",
                        finish_span=_finish_span,
                        stream_finished=self._stream_finished,
                    )
                except BaseException:
                    self._stream_finished()
                    raise

            dispatch_start_perf_ns = time.perf_counter_ns()
            dispatch_start_wall_ns = time.time_ns()
            with _attempt_span() as attempt_span:
                _annotate_span(attempt_span, **run_attributes)
                _annotate_span(attempt_span, model=model, provider=provider)
                try:
                    result = client.chat.completions.create(**kwargs)
                    serialized = _serialize_result(result)
                except BaseException as e:
                    _annotate_span(
                        attempt_span,
                        outcome="failure",
                        error_type=type(e).__name__,
                        input_tokens=0,
                        output_tokens=0,
                    )
                    raise
                _annotate_span(
                    attempt_span, outcome="success", **_usage_metrics(serialized["usage"])
                )
                if serialized["choices"]:
                    response_message = serialized["choices"][0]["message"]
                    self._record_transcript(token, kwargs.get("messages"), response_message)
                    if not profiler_ingest.has_exact_events(token):
                        agprof_derive.on_dispatch(
                            token,
                            kwargs.get("messages"),
                            response_message,
                            start_perf_ns=dispatch_start_perf_ns,
                            start_wall_ns=dispatch_start_wall_ns,
                            end_perf_ns=time.perf_counter_ns(),
                            end_wall_ns=time.time_ns(),
                            parent_context=run_context,
                            span_attributes=run_attributes,
                            skip_tool_call_ids=profiler_ingest.exact_tool_call_ids(token),
                            before_derive_tools=lambda call_ids: (
                                profiler_ingest.reconcile_derived_tool_call_ids(token, call_ids)
                            ),
                        )
                return JSONResponse(serialized)

        @app.post("/internal/resolve_model")
        async def resolve_model(request: Request):
            # Lets agproxy_llm's routing/translation layer route to the
            # right model without holding a live `ag` reference itself --
            # the prerequisite for that layer to run somewhere `ag` isn't
            # reachable at all (e.g. inside the sandbox container). See
            # agproxy_llm.py's `/v1/messages`/`/v1/responses` routes.
            body = await request.json()
            ag = self._agent_for_token(body.get("token"))
            if ag is None:
                return JSONResponse({"error": "unknown or missing token"}, status_code=401)
            return JSONResponse({"model": ag.llm.backend.model or ""})

        @app.post("/internal/context_limit")
        async def context_limit(request: Request):
            # Lets a caller with no `agllm` of its own (native.py's
            # in-container entrypoint, which can't import agllm.py at all
            # -- see that module's docstring) learn this agent's model's
            # context window, the one piece of model metadata its own
            # compaction (agllm_pure.should_compact/tail_start) needs and
            # can't derive locally. Reuses agllm.fetch_context_limit's
            # real model-listing/known-limit lookup -- not a second,
            # simplified guess.
            body = await request.json()
            ag = self._agent_for_token(body.get("token"))
            if ag is None:
                return JSONResponse({"error": "unknown or missing token"}, status_code=401)
            from ..agllm import agllm

            return JSONResponse({"context_limit": agllm.fetch_context_limit(ag.llm.backend)})

        @app.post("/internal/log_warning")
        async def log_warning(request: Request):
            # Same reasoning as resolve_model -- agproxy_llm.py's own
            # warning logging (e.g. _warn_mid_array_system_messages) needs
            # `ag.terminal`, which isn't reachable without a live `ag`
            # reference either.
            body = await request.json()
            ag = self._agent_for_token(body.get("token"))
            if ag is None:
                return JSONResponse({"error": "unknown or missing token"}, status_code=401)
            ag.terminal.log("WARNING  ", body.get("message", ""))
            return JSONResponse({"ok": True})

        @app.post("/internal/check_tool_policy")
        async def check_tool_policy(request: Request):
            # Same reasoning again -- agpolicy.check_tool() takes the full
            # `ag` object (a policy implementation is arbitrary user code
            # that may inspect anything on it), so this can't be reduced to
            # a narrower duck-typed field the way resolve_model/log_warning
            # are.
            body = await request.json()
            ag = self._agent_for_token(body.get("token"))
            if ag is None:
                return JSONResponse(
                    {"decision": "deny", "reason": "unknown or missing token"}, status_code=401
                )
            from .. import agharness

            policy = agharness.default_policy(ag)
            decision = policy.check_tool(
                ag, body.get("tool_name", ""), body.get("tool_input") or {}
            )
            return JSONResponse({"decision": decision.kind, "reason": decision.reason})

        return app

    def _stream_started(self) -> None:
        with self._stream_condition:
            self._active_streams += 1

    def _stream_finished(self) -> None:
        with self._stream_condition:
            self._active_streams -= 1
            self._stream_condition.notify_all()

    def drain(self, timeout_s: "float | None" = None) -> bool:
        """Wait for streaming response finalizers without stopping the server."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._stream_condition:
            while self._active_streams:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._stream_condition.wait(timeout=remaining)
        return True

    # -- lifecycle ----------------------------------------------------------
    # Identical shape to agProxyLLM's start()/ensure_uds_started()/stop() --
    # see that module for the reasoning (uvicorn on a daemon thread, poll
    # `server.started` rather than assume a fixed delay, UDS as the
    # container-boundary-crossing mechanism via agsandbox's existing
    # bind-mount primitive).

    def start(self) -> str:
        if self.base_url is not None:
            return self.base_url

        fields = _AgLLMTerminusFields(self._agconfig)
        config = uvicorn.Config(self._app, host=fields.bind_host, port=0, log_level="warning")
        server = uvicorn.Server(config)
        self._server = server

        self._thread = threading.Thread(target=server.run, daemon=True, name="agllm_terminus")
        self._thread.start()

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not server.started:
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("agLLMTerminus server did not start within 10s")

        port = server.servers[0].sockets[0].getsockname()[1]
        self.base_url = f"http://{fields.bind_host}:{port}"
        return self.base_url

    def stop(self) -> None:
        self.drain(timeout_s=10)
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._server = None
        self._thread = None
        self.base_url = None
        self.stop_uds()

    def ensure_uds_started(self) -> str:
        if self.uds_path is not None:
            return self.uds_path

        from ..agutil import agharness_llm_gateway_dir

        sock_path = str(agharness_llm_gateway_dir() / f"agllm_terminus-{uuid.uuid4().hex}.sock")
        config = uvicorn.Config(self._app, uds=sock_path, log_level="warning")
        server = uvicorn.Server(config)
        self._uds_server = server

        self._uds_thread = threading.Thread(
            target=server.run, daemon=True, name="agllm_terminus-uds"
        )
        self._uds_thread.start()

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not server.started:
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("agLLMTerminus UDS server did not start within 10s")

        self.uds_path = sock_path
        return sock_path

    def stop_uds(self) -> None:
        if self._uds_server is not None:
            self._uds_server.should_exit = True
        if self._uds_thread is not None:
            self._uds_thread.join(timeout=10)
        self._uds_server = None
        self._uds_thread = None
        self.uds_path = None


_shared_terminus: "agLLMTerminus | None" = None
_shared_terminus_lock = threading.Lock()


def get_shared_terminus(agconfig: "agConfig | None" = None) -> agLLMTerminus:
    """One `agLLMTerminus` per process, mirroring `agproxy_llm.get_shared_gateway`."""
    global _shared_terminus
    if _shared_terminus is not None:
        return _shared_terminus
    with _shared_terminus_lock:
        if _shared_terminus is None:
            _shared_terminus = agLLMTerminus(agconfig)
            from .shared_services import register_shared_service

            register_shared_service(_shared_terminus)
        return _shared_terminus


__all__ = ["agLLMTerminus", "agLLMTerminusConfig", "get_shared_terminus"]
