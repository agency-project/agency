"""LLM backend abstraction for agllm.

An `agllm` instance builds exactly one `agllm_backend` from its config (via
`agllm_backend.for_config()`) and reuses it for every client it needs — the
streaming call in `agllm.call()`, the summarisation call in `agllm.compact()`,
and the model-listing lookup in `agllm.fetch_context_limit()`. Backend
selection logic (OpenAI-compatible vs. Amazon Bedrock, and within Bedrock,
the OpenAI-compatible Mantle gateway vs. Anthropic's native Messages API)
lives here instead of being duplicated at each call site.

Every backend's client exposes the same surface agllm.call()/.compact() use:
`.chat.completions.create(**kwargs)` (streaming or not) and `.close()`.
"""
from __future__ import annotations
import json
import os
import re
import httpx
import openai

from .agconfig import agConfig, GlobalConfigParam, DynamicConfigParam, _AgConfigViewBase

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None


# ---------------------------------------------------------------------------
# Class-based LLM config -- every per-call LLM request parameter (model,
# api_key, temperature, ...) is a DynamicConfigParam, the same descriptor
# machinery every other framework class uses for its tunables (see agllm.py's
# _AgLLMFields). `agllm_backend` inherits this class, so every concrete
# backend (_OpenAICompatibleBackend, _AnthropicBackend, ...) reads its
# parameters as plain attributes (self.model, self.api_key, ...).
#
# Descriptors are registered once, at class-body-execution time, and shared
# via ordinary inheritance. Values are NOT stored privately per instance --
# `agllm_backend.__init__` points self._agconfig at the exact agConfig the
# caller built its config on (typically via `cfg.agllm_backend.model = ...`
# before constructing anything), so a later `cfg.agllm_backend.temperature =
# 0.9` is visible on the next read, same as any other DynamicConfigParam.
# Two backends given two different (or cloned) agConfig instances still
# never see each other's values, for the usual agConfig reason: each one's
# data lives in its own `.data` dict.
#
# model_listing_timeout_seconds and default_max_tokens are different in
# kind: genuine process-wide tunables for the backend machinery itself
# (unrelated to any one call's parameters), so they stay tier-1
# (GlobalConfigParam), exactly as before.
# ---------------------------------------------------------------------------

class AgLLMBackendFields:
    """Every LLM config field used by any backend, as DynamicConfigParam
    descriptors, plus the tier-1 tunables for the backend machinery itself."""

    model_listing_timeout_seconds = GlobalConfigParam("agllm_backend", default=10.0)  # httpx timeout for the best-effort /v1/models lookup.
    # Anthropic requires max_tokens; 4096 was too small — a call with no explicit
    # max_tokens (i.e. no "max_tokens" key in llm_config) could truncate mid-tool-call
    # on a large structured tool argument, silently dropping the call entirely (see
    # the flush-on-truncation handling in _anthropic_stream_to_openai_chunks) instead
    # of erroring. compact() always passes its own explicit max_tokens and never
    # hits this fallback.
    default_max_tokens = GlobalConfigParam("agllm_backend", default=128000)

    model = DynamicConfigParam("agllm_backend", default="")
    api_key = DynamicConfigParam("agllm_backend", default=None)
    base_url = DynamicConfigParam("agllm_backend", default=None)
    provider = DynamicConfigParam("agllm_backend", default=None)
    region = DynamicConfigParam("agllm_backend", default=None)
    context_limit = DynamicConfigParam("agllm_backend", default=None)
    temperature = DynamicConfigParam("agllm_backend", default=None)
    max_completion_tokens = DynamicConfigParam("agllm_backend", default=None)
    max_tokens = DynamicConfigParam("agllm_backend", default=None)  # deprecated alias for max_completion_tokens
    top_p = DynamicConfigParam("agllm_backend", default=None)
    frequency_penalty = DynamicConfigParam("agllm_backend", default=None)
    presence_penalty = DynamicConfigParam("agllm_backend", default=None)
    n = DynamicConfigParam("agllm_backend", default=None)
    stop = DynamicConfigParam("agllm_backend", default=None)
    logprobs = DynamicConfigParam("agllm_backend", default=None)
    seed = DynamicConfigParam("agllm_backend", default=None)
    extra_body = DynamicConfigParam("agllm_backend", default=None)
    top_k = DynamicConfigParam("agllm_backend", default=None)
    repetition_penalty = DynamicConfigParam("agllm_backend", default=None)
    min_p = DynamicConfigParam("agllm_backend", default=None)
    min_tokens = DynamicConfigParam("agllm_backend", default=None)
    guided_json = DynamicConfigParam("agllm_backend", default=None)
    guided_regex = DynamicConfigParam("agllm_backend", default=None)
    workspace_id = DynamicConfigParam("agllm_backend", default=None)
    aws_access_key = DynamicConfigParam("agllm_backend", default=None)
    aws_secret_key = DynamicConfigParam("agllm_backend", default=None)
    aws_session_token = DynamicConfigParam("agllm_backend", default=None)
    aws_profile = DynamicConfigParam("agllm_backend", default=None)
    aws_region = DynamicConfigParam("agllm_backend", default=None)

    _FIELD_NAMES: "tuple[str, ...]" = (
        "model", "api_key", "base_url", "provider", "region", "context_limit",
        "temperature", "max_completion_tokens", "max_tokens", "top_p",
        "frequency_penalty", "presence_penalty", "n", "stop", "logprobs", "seed",
        "extra_body", "top_k", "repetition_penalty", "min_p", "min_tokens",
        "guided_json", "guided_regex", "workspace_id", "aws_access_key",
        "aws_secret_key", "aws_session_token", "aws_profile", "aws_region",
    )

    def as_dict(self) -> dict:
        """Snapshot of every currently-set field -- used for logging and
        checkpoint serialization, where a dict is more convenient than
        reading each attribute individually."""
        return {name: v for name in self._FIELD_NAMES if (v := getattr(self, name)) is not None}


class agLLMBackendConfig(_AgConfigViewBase):
    """View over an agConfig for pre-setting agllm_backend fields in one call::

        cfg = agLLMBackendConfig(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            model="meta-llama/Llama-3.1-8B-Instruct",
        ).agconfig
        ag = agent(agconfig=cfg)

    See `_AgConfigViewBase` in agconfig.py for the shared mechanics (targets
    an existing agConfig if given one, validates field names, composes with
    other owners' views via `agConfig(view_a, view_b, ...)`).
    """

    _OWNER = "agllm_backend"


# Common per-call generation params every OpenAI-compatible server accepts as
# top-level chat.completions.create() kwargs (see agllm.build_llm_kwargs's
# _OPENAI_GEN_PARAMS). vLLM (and other OpenAI-compatible servers with
# sampling extensions) additionally accept _VLLM_EXTRA_GEN_FIELDS via
# extra_body -- real OpenAI's API does not.
_OPENAI_GEN_FIELDS = frozenset({
    "temperature", "max_completion_tokens", "max_tokens", "top_p",
    "frequency_penalty", "presence_penalty", "n", "stop", "logprobs", "seed",
    "extra_body",
})
_VLLM_EXTRA_GEN_FIELDS = frozenset({
    "top_k", "repetition_penalty", "min_p", "min_tokens", "guided_json", "guided_regex",
})


class _AgProviderBackendConfig(agLLMBackendConfig):
    """Shared base for the per-provider *BackendConfig classes below. Each
    subclass fixes `provider` to the value that routes `for_config()` to its
    corresponding backend class, and restricts `_ALLOWED_FIELDS` to what that
    backend actually reads -- passing a field it silently ignores (e.g.
    `frequency_penalty` on `agAnthropicBackendConfig`) raises immediately
    instead of the value quietly never reaching the API call.
    """

    _PROVIDER: "ClassVar[str]"

    def __init__(self, agconfig: "agConfig | None" = None, **fields) -> None:
        super().__init__(agconfig)
        self._agconfig.set(self._OWNER, "provider", self._PROVIDER)
        if fields:
            self.update(**fields)


class agVLLMBackendConfig(_AgProviderBackendConfig):
    """agLLMBackendConfig restricted to the fields `_OpenAICompatibleBackend`
    (agllm_backend.py) reads for a vLLM (or other OpenAI-compatible) endpoint
    -- the full generation surface, including vLLM/sglang sampling extensions
    (top_k, repetition_penalty, min_p, min_tokens, guided_json, guided_regex)
    sent via extra_body. `provider` is fixed to "vllm"."""

    _PROVIDER = "vllm"
    _ALLOWED_FIELDS = frozenset({"model", "api_key", "base_url", "context_limit"}) | _OPENAI_GEN_FIELDS | _VLLM_EXTRA_GEN_FIELDS


class agOpenAIBackendConfig(_AgProviderBackendConfig):
    """agLLMBackendConfig restricted to the fields `_OpenAICompatibleBackend`
    reads for the real OpenAI API. Excludes the vLLM/sglang-only sampling
    extensions `agVLLMBackendConfig` allows (top_k, repetition_penalty,
    min_p, min_tokens, guided_json, guided_regex) -- OpenAI's API rejects
    those in extra_body. `provider` is fixed to "openai"."""

    _PROVIDER = "openai"
    _ALLOWED_FIELDS = frozenset({"model", "api_key", "base_url", "context_limit"}) | _OPENAI_GEN_FIELDS


class agAnthropicBackendConfig(_AgProviderBackendConfig):
    """agLLMBackendConfig restricted to the fields `_AnthropicBackend` (the
    first-party api.anthropic.com backend) actually forwards -- see
    `_AnthropicBedrockCompletions.create()`, which every Anthropic-family
    backend shares: only temperature, top_p, max_tokens/max_completion_tokens,
    and extra_body["top_k"] are applied; frequency_penalty, presence_penalty,
    n, stop, logprobs, seed, and the vLLM-only extras are silently dropped by
    that adapter, so they're excluded here rather than accepted and ignored.
    `provider` is fixed to "anthropic"."""

    _PROVIDER = "anthropic"
    _ALLOWED_FIELDS = frozenset({
        "model", "api_key", "base_url", "context_limit", "workspace_id",
        "temperature", "top_p", "max_completion_tokens", "max_tokens", "extra_body",
    })


class agBedrockBackendConfig(_AgProviderBackendConfig):
    """agLLMBackendConfig restricted to the fields Amazon Bedrock backends
    read. `for_config()` picks between two backends under the hood based on
    `model`: non-Anthropic models go through `_OpenAICompatibleBedrockBackend`
    (the OpenAI-compatible Mantle gateway -- full generation surface, same as
    `agVLLMBackendConfig`); Anthropic models on Bedrock go through
    `_AnthropicBedrockBackend`, which -- like `agAnthropicBackendConfig` --
    only applies temperature, top_p, max_tokens, and extra_body["top_k"],
    silently ignoring the rest. `provider` is fixed to "bedrock"."""

    _PROVIDER = "bedrock"
    _ALLOWED_FIELDS = frozenset({"model", "api_key", "region", "context_limit"}) | _OPENAI_GEN_FIELDS | _VLLM_EXTRA_GEN_FIELDS


# Exception-translation tuples so callers (agllm.call()) can catch both
# backend families without importing the anthropic package directly.
BAD_REQUEST_EXCS: tuple = (openai.BadRequestError,) + (
    (_anthropic_sdk.BadRequestError,) if _anthropic_sdk else ()
)
API_CONN_EXCS: tuple = (openai.APIConnectionError,) + (
    (_anthropic_sdk.APIConnectionError,) if _anthropic_sdk else ()
)
RATE_LIMIT_EXCS: tuple = (openai.RateLimitError,) + (
    (_anthropic_sdk.RateLimitError,) if _anthropic_sdk else ()
)
# Base/catch-all API error classes — covers mid-stream server-side error frames
# (e.g. openai._streaming raises openai.APIError directly, with no HTTP status
# to build a more specific subclass from) and any other APIStatusError subclass
# not special-cased above (e.g. InternalServerError). Callers should check the
# more specific tuples above first — BadRequestError/RateLimitError/connection
# errors are all subclasses of these and get their own handling.
API_ERROR_EXCS: tuple = (openai.APIError,) + (
    (_anthropic_sdk.APIError,) if _anthropic_sdk else ()
)

_ANTHROPIC_BEDROCK_MODEL_RE = re.compile(r"^(?:(?:us|eu|apac|global)\.)?anthropic\.")
_CACHE_CONTROL = {"type": "ephemeral"}  # prompt-caching breakpoint, default 5-minute TTL

# Bedrock's native invoke_model API has no /v1/models-style endpoint to query
# context windows from, so known Anthropic model context windows are hardcoded
# here instead. Keyed by the bare model name, after stripping the region and
# "anthropic." prefix Bedrock IDs carry — see _known_anthropic_context_window
# below. Update when new models ship.
_ANTHROPIC_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-fable-5": 1_000_000,
    "claude-mythos-5": 1_000_000,
    "claude-mythos-preview": 1_000_000,
    "claude-opus-4-8": 1_000_000,
    "claude-opus-4-7": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-opus-4-5": 1_000_000,
    "claude-opus-4-1": 1_000_000,
    "claude-opus-4-0": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-sonnet-4-5": 1_000_000,
    "claude-sonnet-4-0": 1_000_000,
    "claude-haiku-4-5": 200_000,
}


# ---------------------------------------------------------------------------
# AWS Bedrock SigV4 auth
# ---------------------------------------------------------------------------

class _BedrockSigV4Auth(httpx.Auth):
    """httpx auth handler that signs requests with AWS SigV4 for Amazon Bedrock."""

    def __init__(self, region: str, api_key: str | None = None) -> None:
        import boto3
        from botocore.credentials import Credentials
        self._region = region
        if api_key:
            parts = api_key.split(":", 2)
            if len(parts) < 2:
                raise ValueError(
                    "Bedrock api_key must be 'ACCESS_KEY_ID:SECRET_ACCESS_KEY' "
                    "or 'ACCESS_KEY_ID:SECRET_ACCESS_KEY:SESSION_TOKEN'."
                )
            self._creds = Credentials(
                access_key=parts[0],
                secret_key=parts[1],
                token=parts[2] if len(parts) == 3 else None,
            )
        else:
            creds = boto3.Session(region_name=region).get_credentials()
            if creds is None:
                raise RuntimeError(
                    "No AWS credentials found for Amazon Bedrock. "
                    "Set api_key='ACCESS_KEY_ID:SECRET_ACCESS_KEY' in the llm_config, "
                    "or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars, "
                    "or run: aws configure"
                )
            self._creds = creds

    def auth_flow(self, request: httpx.Request):
        import botocore.auth
        import botocore.awsrequest

        aws_req = botocore.awsrequest.AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content or b"",
            headers={k: v for k, v in request.headers.items()
                     if k.lower() not in ("host", "content-length")},
        )
        botocore.auth.SigV4Auth(self._creds.get_frozen_credentials(), "bedrock", self._region).add_auth(aws_req)
        for k, v in aws_req.headers.items():
            request.headers[k] = v
        yield request


# ---------------------------------------------------------------------------
# Anthropic Bedrock adapter
#
# Claude models on Amazon Bedrock are NOT served through the OpenAI-compatible
# Mantle gateway used by every other Bedrock model — Mantle's `/v1/models`
# never lists an `anthropic.*` model, and every Claude model ID 404s there.
# Claude models are only reachable through Bedrock's native invoke_model API,
# in the Anthropic Messages API shape (the `anthropic` SDK's `AnthropicBedrock`
# client), and only via an inference-profile ID (e.g. `us.anthropic.claude-
# sonnet-5`) rather than the bare `anthropic.claude-sonnet-5` foundation-model
# ID — the bare ID 400s with "on-demand throughput isn't supported."
#
# This adapter translates between the Messages API and the OpenAI
# chat.completions.create(...) interface (streaming and non-streaming) the
# rest of agllm.py is built around, so its streaming / tool-call / retry /
# compaction logic needs no changes to support Claude-on-Bedrock.
# ---------------------------------------------------------------------------

def _is_anthropic_bedrock_model(model: str) -> bool:
    return bool(_ANTHROPIC_BEDROCK_MODEL_RE.match(model or ""))


def _openai_messages_to_anthropic(messages: list[dict]) -> "tuple[str | None, list[dict]]":
    """Convert OpenAI-style chat messages into (system_text, anthropic_messages)."""
    system_parts: list[str] = []
    out: list[dict] = []

    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            if content:
                system_parts.append(content)
        elif role == "user":
            out.append({"role": "user", "content": content})
        elif role == "assistant":
            blocks: list[dict] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {})
                try:
                    tool_input = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    tool_input = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": tool_input,
                })
            out.append({"role": "assistant", "content": blocks or content})
        elif role == "tool":
            result_block = {
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": content,
            }
            prev_content = out[-1]["content"] if out and out[-1]["role"] == "user" else None
            if isinstance(prev_content, list):
                prev_content.append(result_block)
            else:
                out.append({"role": "user", "content": [result_block]})
        # unrecognized roles are dropped rather than sent to an API that would reject them
    return ("\n\n".join(system_parts) or None), out


def _openai_tools_to_anthropic(tools: "list[dict] | None") -> "list[dict] | None":
    if not tools:
        return None
    converted = []
    for t in tools:
        fn = t.get("function", t)
        converted.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return converted


class _FakeToolCallFunction:
    __slots__ = ("name", "arguments")

    def __init__(self, name: str = "", arguments: str = "") -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCallDelta:
    __slots__ = ("index", "id", "function")

    def __init__(self, index: int, id: str = "", name: str = "", arguments: str = "") -> None:
        self.index = index
        self.id = id
        self.function = _FakeToolCallFunction(name, arguments)


class _FakeDelta:
    __slots__ = ("content", "tool_calls", "reasoning_content", "model_extra")

    def __init__(self, content=None, tool_calls=None, reasoning_content=None) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content
        self.model_extra = {}


class _FakeChoice:
    __slots__ = ("delta",)

    def __init__(self, delta: _FakeDelta) -> None:
        self.delta = delta


class _FakeUsage:
    __slots__ = ("prompt_tokens", "completion_tokens")

    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeChunk:
    __slots__ = ("choices", "usage")

    def __init__(self, choices=(), usage=None) -> None:
        self.choices = list(choices)
        self.usage = usage


def _anthropic_stream_to_openai_chunks(stream):
    """Translate an Anthropic Messages-API SSE stream into OpenAI-style chunks
    matching what agllm.call()'s streaming loop expects.

    Tool-call JSON input is buffered per content block and emitted as a single
    chunk on content_block_stop, rather than streamed fragment-by-fragment.
    agllm.call() never renders partial tool-call arguments to the user (only
    text/thinking feed live_messages_fn), so nothing is lost — and emitting
    one complete chunk instead of many small ones avoids relying on every
    fragment individually surviving whatever consumes this generator (e.g.
    agutil._iter_batched's background-thread queue).
    """
    input_tokens = 0
    output_tokens = 0
    tool_blocks: dict[int, dict] = {}  # index -> {"id", "name", "json_parts"}

    for event in stream:
        etype = getattr(event, "type", None)
        if etype == "message_start":
            usage = getattr(event.message, "usage", None)
            if usage is not None:
                input_tokens = getattr(usage, "input_tokens", 0) or 0
        elif etype == "content_block_start":
            block = event.content_block
            if block.type == "tool_use":
                tool_blocks[event.index] = {"id": block.id, "name": block.name, "json_parts": []}
        elif etype == "content_block_delta":
            delta = event.delta
            kind = getattr(delta, "type", None)
            if kind == "text_delta":
                yield _FakeChunk(choices=[_FakeChoice(_FakeDelta(content=delta.text))])
            elif kind == "thinking_delta":
                yield _FakeChunk(choices=[_FakeChoice(_FakeDelta(reasoning_content=delta.thinking))])
            elif kind == "input_json_delta":
                block = tool_blocks.get(event.index)
                if block is not None:
                    block["json_parts"].append(delta.partial_json or "")
        elif etype == "content_block_stop":
            block = tool_blocks.pop(event.index, None)
            if block is not None:
                yield _FakeChunk(choices=[_FakeChoice(_FakeDelta(tool_calls=[
                    _FakeToolCallDelta(
                        index=event.index, id=block["id"], name=block["name"],
                        arguments="".join(block["json_parts"]),
                    )
                ]))])
        elif etype == "message_delta":
            usage = getattr(event, "usage", None)
            if usage is not None:
                output_tokens = getattr(usage, "output_tokens", 0) or output_tokens

    # If the stream ended (e.g. stop_reason="max_tokens") while a tool_use
    # block was still open, content_block_stop never fires for it and the
    # tool call would otherwise vanish with no trace — the assistant turn
    # comes out completely empty and callers loop forever re-requesting it.
    # Flush whatever JSON was collected so far instead; a downstream
    # json.loads() failure on truncated arguments is at least visible.
    for index in sorted(tool_blocks):
        block = tool_blocks[index]
        print(
            f"[agllm_backend] WARNING: tool_use block {block['name']!r} "
            f"(id={block['id']}) truncated mid-stream (likely hit max_tokens) "
            f"— flushing partial arguments instead of dropping the call"
        )
        yield _FakeChunk(choices=[_FakeChoice(_FakeDelta(tool_calls=[
            _FakeToolCallDelta(
                index=index, id=block["id"], name=block["name"],
                arguments="".join(block["json_parts"]),
            )
        ]))])

    yield _FakeChunk(usage=_FakeUsage(input_tokens, output_tokens))


class _FakeMessage:
    __slots__ = ("content",)

    def __init__(self, content: str) -> None:
        self.content = content


class _FakeNonStreamChoice:
    __slots__ = ("message",)

    def __init__(self, message: _FakeMessage) -> None:
        self.message = message


class _AnthropicNonStreamResponse:
    """Mimics openai.types.chat.ChatCompletion's `.choices[0].message.content`
    surface for a non-streaming Anthropic Messages API response — used by
    agllm.compact(), which doesn't stream."""
    __slots__ = ("choices",)

    def __init__(self, anthropic_message) -> None:
        text = "".join(
            b.text for b in anthropic_message.content if getattr(b, "type", None) == "text"
        )
        self.choices = [_FakeNonStreamChoice(_FakeMessage(text))]


def _with_cache_control(content):
    """Return `content` with cache_control on its last block, normalizing a
    bare string into a single text block first (cache_control attaches to a
    content block, not to a string). Caller must pass content it's safe to
    mutate — _openai_messages_to_anthropic() always builds fresh lists/dicts,
    never a reference into the caller's original messages."""
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    else:
        content = list(content)
    if content:
        content[-1] = {**content[-1], "cache_control": _CACHE_CONTROL}
    return content


class _AnthropicBedrockCompletions:
    def __init__(self, anthropic_client) -> None:
        self._client = anthropic_client

    def create(
        self, *, model, messages, stream=False, stream_options=None,
        max_tokens=None, temperature=None, top_p=None, tools=None,
        extra_body=None, **_ignored,
    ):
        system, anthropic_messages = _openai_messages_to_anthropic(messages)
        kwargs: dict = dict(
            model=model,
            messages=anthropic_messages,
            max_tokens=max_tokens or AgLLMBackendFields().default_max_tokens,
        )
        if system:
            # Breakpoint on the system prompt: it's the largest, most static
            # part of every request (agent instructions), and tools render
            # before system in Anthropic's prefix order, so this one
            # breakpoint caches tools + system together.
            kwargs["system"] = [{"type": "text", "text": system, "cache_control": _CACHE_CONTROL}]
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if extra_body and "top_k" in extra_body:
            kwargs["top_k"] = extra_body["top_k"]
        anthropic_tools = _openai_tools_to_anthropic(tools)
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        if anthropic_messages:
            # Second breakpoint on the latest turn. messages grows across
            # calls in an agent loop, so this lets the *next* call read
            # everything up to (not including) this turn from cache — the
            # standard multi-turn caching pattern. Earlier breakpoints don't
            # need to be resent; they remain valid read points.
            anthropic_messages[-1] = dict(anthropic_messages[-1])
            anthropic_messages[-1]["content"] = _with_cache_control(anthropic_messages[-1]["content"])

        if not stream:
            return _AnthropicNonStreamResponse(self._client.messages.create(**kwargs))

        raw_stream = self._client.messages.create(stream=True, **kwargs)
        return _anthropic_stream_to_openai_chunks(raw_stream)


class _AnthropicBedrockChat:
    def __init__(self, anthropic_client) -> None:
        self.completions = _AnthropicBedrockCompletions(anthropic_client)


class _AnthropicBedrockChatClient:
    """Drop-in replacement for the subset of openai.OpenAI's interface
    agllm.call()/.compact() use (`.chat.completions.create()`, `.close()`),
    wrapping an already-constructed anthropic.AnthropicBedrock client."""

    def __init__(self, anthropic_client) -> None:
        self._client = anthropic_client
        self.chat = _AnthropicBedrockChat(anthropic_client)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close:
            close()


# ---------------------------------------------------------------------------
# Backend classes
# ---------------------------------------------------------------------------

class agllm_backend(AgLLMBackendFields):
    """One backend instance per agconfig — knows how to build a client and
    answer capability questions (model listing, tokenize endpoint). Use
    `agllm_backend.for_config(agconfig)` to get the right subclass; don't
    instantiate a subclass directly.

    Inherits AgLLMBackendFields so every concrete backend reads its
    parameters as plain attributes (self.model, self.api_key, ...). Given an
    agConfig, it's stored as-is (self._agconfig) -- not copied -- so a
    caller that mutates it later (`cfg.agllm_backend.temperature = 0.9`)
    sees the change reflected on the next attribute read, same as any other
    DynamicConfigParam consumer in the framework. Given a plain dict (for
    quick/manual construction outside the agconfig-driven path), it's
    wrapped in a fresh private agConfig -- same attribute-backed reads, just
    with no caller-visible agConfig to mutate afterwards.
    """

    def __init__(self, config: "dict | agConfig") -> None:
        self._agconfig = config if isinstance(config, agConfig) else agConfig({"agllm_backend": dict(config)})

    @staticmethod
    def for_config(config: "dict | agConfig") -> "agllm_backend":
        agconfig = config if isinstance(config, agConfig) else agConfig({"agllm_backend": dict(config)})
        provider = agconfig.get("agllm_backend", "provider")
        model = agconfig.get("agllm_backend", "model", "") or ""
        if provider == "bedrock":
            if _is_anthropic_bedrock_model(model):
                return _AnthropicBedrockBackend(agconfig)
            return _OpenAICompatibleBedrockBackend(agconfig)
        if provider in ("anthropicAWS", "anthropic_aws"):
            return _AnthropicAWSBackend(agconfig)
        if provider == "anthropic":
            return _AnthropicBackend(agconfig)
        if provider == "vllm" and not agconfig.get("agllm_backend", "base_url"):
            raise ValueError(
                "agVLLMBackendConfig (provider='vllm') requires base_url "
                "-- point it at your vLLM/OpenAI-compatible endpoint (e.g. "
                "'http://localhost:8000/v1')."
            )
        return _OpenAICompatibleBackend(agconfig)

    def make_client(self, timeout: httpx.Timeout):
        """Build and return a client exposing `.chat.completions.create()` and `.close()`."""
        raise NotImplementedError

    def list_models(self) -> list:
        """Best-effort model listing, used for context-limit lookups. Exceptions
        propagate to the caller (agllm.fetch_context_limit already wraps this).
        Override to return [] for backends with no listing capability."""
        client = self.make_client(httpx.Timeout(self.model_listing_timeout_seconds))
        return list(client.models.list())

    def tokenize_url(self) -> "str | None":
        """Root URL for a vLLM-style /tokenize endpoint, or None if unsupported."""
        return None

    def known_context_limit(self, model: str) -> "int | None":
        """Static fallback context window for models with no listing API to
        query (e.g. Bedrock's native invoke_model). None if unknown — the
        caller (agllm.fetch_context_limit) falls back to _AgLLMFields.default_context_limit.default."""
        return None


class _OpenAICompatibleBackend(agllm_backend):
    """Default backend: OpenAI, vLLM, or any other OpenAI-compatible endpoint."""

    def make_client(self, timeout: httpx.Timeout) -> openai.OpenAI:
        return openai.OpenAI(
            api_key=self.api_key or "EMPTY",
            base_url=self.base_url,
            timeout=timeout,
        )

    def tokenize_url(self) -> "str | None":
        base_url: str = self.base_url or ""
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        return root or None


class _OpenAICompatibleBedrockBackend(_OpenAICompatibleBackend):
    """Bedrock models reachable through the OpenAI-compatible Mantle gateway —
    every Bedrock model except Anthropic's own (see module docstring above)."""

    def make_client(self, timeout: httpx.Timeout) -> openai.OpenAI:
        region  = self.region or "us-east-1"
        api_key = self.api_key or os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or None
        mantle_url  = f"https://bedrock-mantle.{region}.api.aws/v1"
        runtime_url = f"https://bedrock-runtime.{region}.amazonaws.com"
        # A Bedrock API key (e.g. "ABSK...") is a single opaque bearer token
        # for the Mantle gateway. AWS access/secret key pairs for SigV4 signing
        # are always "ACCESS_KEY_ID:SECRET_ACCESS_KEY[:SESSION_TOKEN]" — the
        # colon is what distinguishes the two, not any particular prefix
        # string (real Bedrock API keys don't start with "bedrock-api-key-").
        if api_key and ":" not in api_key:
            return openai.OpenAI(api_key=api_key, base_url=mantle_url, timeout=timeout)
        if not api_key:
            try:
                from aws_bedrock_token_generator import provide_token as _provide_token
                os.environ.setdefault("AWS_DEFAULT_REGION", region)
                token = _provide_token(region=region)
                return openai.OpenAI(api_key=token, base_url=mantle_url, timeout=timeout)
            except ImportError:
                pass
        return openai.OpenAI(
            api_key="bedrock",
            base_url=runtime_url,
            http_client=httpx.Client(
                auth=_BedrockSigV4Auth(region, api_key=api_key), timeout=timeout
            ),
        )

    def tokenize_url(self) -> "str | None":
        return None  # Bedrock has no vLLM-style /tokenize endpoint


def _known_anthropic_context_window(model: str) -> "int | None":
    """Look up a known context window for an Anthropic Bedrock model ID.

    Strips the optional region prefix (us./eu./apac./global.) and the
    "anthropic." prefix, then matches the remainder against known model
    names — exact match, or a prefix match to tolerate dated snapshot
    suffixes (e.g. "claude-opus-4-5-20251101-v1:0").
    """
    bare = _ANTHROPIC_BEDROCK_MODEL_RE.sub("", model or "")
    for known_id, window in _ANTHROPIC_CONTEXT_WINDOWS.items():
        if bare == known_id or bare.startswith(known_id + "-"):
            return window
    return None


class _AnthropicBedrockBackend(agllm_backend):
    """Claude models on Amazon Bedrock — native invoke_model API via the
    anthropic SDK's AnthropicBedrock client (Messages API shape)."""

    def make_client(self, timeout: httpx.Timeout) -> _AnthropicBedrockChatClient:
        if _anthropic_sdk is None:
            raise RuntimeError(
                "Anthropic models on Bedrock require the 'anthropic' package: "
                "pip install anthropic"
            )
        region = self.region or "us-east-1"
        anthropic_client = _anthropic_sdk.AnthropicBedrock(aws_region=region, timeout=timeout)
        return _AnthropicBedrockChatClient(anthropic_client)

    def list_models(self) -> list:
        return []  # Bedrock's native invoke_model API has no OpenAI-style /v1/models

    def tokenize_url(self) -> "str | None":
        return None

    def known_context_limit(self, model: str) -> "int | None":
        return _known_anthropic_context_window(model)


class _AnthropicAWSBackend(agllm_backend):
    """Claude Platform on AWS via the anthropic SDK's AnthropicAWS client.

    Auth (resolved by the SDK): SigV4 via the default AWS credential chain,
    explicit aws_access_key/aws_secret_key, or an API key (config `api_key` /
    ANTHROPIC_AWS_API_KEY). Requires workspace_id (config /
    ANTHROPIC_AWS_WORKSPACE_ID) and aws_region (config `region` or
    `aws_region` / AWS_REGION) unless base_url is set.
    """

    def _client_kwargs(self, timeout: httpx.Timeout) -> dict:
        kwargs: dict = dict(timeout=timeout)
        api_key = self.api_key or os.environ.get("ANTHROPIC_AWS_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
        for key in ("aws_access_key", "aws_secret_key", "aws_session_token", "aws_profile"):
            value = getattr(self, key)
            if value:
                kwargs[key] = value
        region = self.aws_region or self.region
        if region:
            kwargs["aws_region"] = region
        workspace_id = (
            self.workspace_id
            or os.environ.get("ANTHROPIC_AWS_WORKSPACE_ID")
            or os.environ.get("ANTHROPIC_WORKSPACE_ID")
        )
        if workspace_id:
            kwargs["workspace_id"] = workspace_id
        base_url = (
            self.base_url
            or os.environ.get("ANTHROPIC_AWS_BASE_URL")
            or os.environ.get("ANTHROPIC_BASE_URL")
        )
        if base_url:
            kwargs["base_url"] = base_url
        return kwargs

    def make_client(self, timeout: httpx.Timeout) -> _AnthropicBedrockChatClient:
        if _anthropic_sdk is None:
            raise RuntimeError(
                "provider='anthropicAWS' requires the 'anthropic' package: pip install anthropic"
            )
        if not hasattr(_anthropic_sdk, "AnthropicAWS"):
            raise RuntimeError(
                "provider='anthropicAWS' requires a recent 'anthropic' package with AnthropicAWS support"
            )
        anthropic_client = _anthropic_sdk.AnthropicAWS(**self._client_kwargs(timeout))
        return _AnthropicBedrockChatClient(anthropic_client)

    def list_models(self) -> list:
        if _anthropic_sdk is None or not hasattr(_anthropic_sdk, "AnthropicAWS"):
            return []
        client = _anthropic_sdk.AnthropicAWS(
            **self._client_kwargs(httpx.Timeout(self.model_listing_timeout_seconds))
        )
        return list(client.models.list())

    def tokenize_url(self) -> "str | None":
        return None

    def known_context_limit(self, model: str) -> "int | None":
        return _known_anthropic_context_window(model)


class _AnthropicBackend(agllm_backend):
    """Claude models via the first-party Anthropic API (api.anthropic.com) —
    the anthropic SDK's plain Anthropic client (Messages API shape). Reuses
    the same _AnthropicBedrockChatClient adapter as the Bedrock backend since
    it only depends on `.messages.create()`, which both clients expose alike.

    For Claude Platform on AWS, prefer provider='anthropicAWS' and the
    AnthropicAWS client instead — it handles SigV4/API-key auth, region-
    derived base URLs, and the workspace header natively.
    """

    def _client_kwargs(self, timeout: httpx.Timeout) -> dict:
        kwargs: dict = dict(
            api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY"),
            timeout=timeout,
        )
        workspace_id = self.workspace_id or os.environ.get("ANTHROPIC_WORKSPACE_ID")
        if workspace_id:
            kwargs["default_headers"] = {"anthropic-workspace-id": workspace_id}
        return kwargs

    def make_client(self, timeout: httpx.Timeout) -> _AnthropicBedrockChatClient:
        if _anthropic_sdk is None:
            raise RuntimeError(
                "provider='anthropic' requires the 'anthropic' package: pip install anthropic"
            )
        anthropic_client = _anthropic_sdk.Anthropic(**self._client_kwargs(timeout))
        return _AnthropicBedrockChatClient(anthropic_client)

    def list_models(self) -> list:
        # Unlike the chat-completions-shaped _AnthropicBedrockChatClient,
        # the real /v1/models listing is only on the raw anthropic client.
        if _anthropic_sdk is None:
            return []
        client = _anthropic_sdk.Anthropic(**self._client_kwargs(httpx.Timeout(self.model_listing_timeout_seconds)))
        return list(client.models.list())

    def tokenize_url(self) -> "str | None":
        return None

    def known_context_limit(self, model: str) -> "int | None":
        # Fallback only — list_models() usually finds the real max_input_tokens
        # first; this covers new models this table hasn't been updated for yet
        # falling through, and any transient failure of the live lookup.
        return _known_anthropic_context_window(model)
