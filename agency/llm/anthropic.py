"""Claude backend, via the first-party Anthropic API (api.anthropic.com).

Also houses the Anthropic Messages-API <-> OpenAI chat.completions adapter
machinery (`_AnthropicBedrockChatClient` and everything it's built from) --
`._bedrock` reuses it for both of its own Anthropic-family backends
(`_AnthropicBedrockBackend`, `_AnthropicAWSBackend`), since all three only
ever need `.messages.create()`, which every one of the underlying anthropic
SDK clients (`Anthropic`, `AnthropicBedrock`, `AnthropicAWS`) exposes alike.
"""

from __future__ import annotations
import json
import os
import re
import httpx

from .base import _AgProviderBackendConfig, agllm_backend

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None

# Matches the region + "anthropic." prefix Bedrock model IDs carry (e.g.
# "us.anthropic.claude-sonnet-5-...") -- a no-op substitution on plain
# api.anthropic.com model IDs ("claude-sonnet-5"), which carry no such
# prefix, so _known_anthropic_context_window() below is shared as-is by both
# this module's _AnthropicBackend and .bedrock's Anthropic-family backends.
_ANTHROPIC_BEDROCK_MODEL_RE = re.compile(r"^(?:(?:us|eu|apac|global)\.)?anthropic\.")

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


def _known_anthropic_context_window(model: str) -> "int | None":
    """Look up a known context window for an Anthropic model ID (either a
    plain api.anthropic.com ID or a Bedrock one).

    Strips the optional region prefix (us./eu./apac./global.) and the
    "anthropic." prefix (a no-op if neither is present), then matches the
    remainder against known model names — exact match, or a prefix match to
    tolerate dated snapshot suffixes (e.g. "claude-opus-4-5-20251101-v1:0").
    """
    bare = _ANTHROPIC_BEDROCK_MODEL_RE.sub("", model or "")
    for known_id, window in _ANTHROPIC_CONTEXT_WINDOWS.items():
        if bare == known_id or bare.startswith(known_id + "-"):
            return window
    return None


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
    _ALLOWED_FIELDS = frozenset(
        {
            "model",
            "api_key",
            "base_url",
            "context_limit",
            "workspace_id",
            "temperature",
            "top_p",
            "max_completion_tokens",
            "max_tokens",
            "extra_body",
        }
    )


_CACHE_CONTROL = {"type": "ephemeral"}  # prompt-caching breakpoint, default 5-minute TTL


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
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                try:
                    tool_input = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    tool_input = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": tool_input,
                    }
                )
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
        converted.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
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
                yield _FakeChunk(
                    choices=[_FakeChoice(_FakeDelta(reasoning_content=delta.thinking))]
                )
            elif kind == "input_json_delta":
                block = tool_blocks.get(event.index)
                if block is not None:
                    block["json_parts"].append(delta.partial_json or "")
        elif etype == "content_block_stop":
            block = tool_blocks.pop(event.index, None)
            if block is not None:
                yield _FakeChunk(
                    choices=[
                        _FakeChoice(
                            _FakeDelta(
                                tool_calls=[
                                    _FakeToolCallDelta(
                                        index=event.index,
                                        id=block["id"],
                                        name=block["name"],
                                        arguments="".join(block["json_parts"]),
                                    )
                                ]
                            )
                        )
                    ]
                )
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
        yield _FakeChunk(
            choices=[
                _FakeChoice(
                    _FakeDelta(
                        tool_calls=[
                            _FakeToolCallDelta(
                                index=index,
                                id=block["id"],
                                name=block["name"],
                                arguments="".join(block["json_parts"]),
                            )
                        ]
                    )
                )
            ]
        )

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
        self,
        *,
        model,
        messages,
        stream=False,
        stream_options=None,
        max_tokens=None,
        temperature=None,
        top_p=None,
        tools=None,
        extra_body=None,
        **_ignored,
    ):
        from .base import AgLLMBackendFields

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
            anthropic_messages[-1]["content"] = _with_cache_control(
                anthropic_messages[-1]["content"]
            )

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
    wrapping an already-constructed anthropic SDK client (`Anthropic`,
    `AnthropicBedrock`, or `AnthropicAWS` -- all three expose `.messages.create()`
    alike, which is all this wrapper needs)."""

    def __init__(self, anthropic_client) -> None:
        self._client = anthropic_client
        self.chat = _AnthropicBedrockChat(anthropic_client)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close:
            close()


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
        client = _anthropic_sdk.Anthropic(
            **self._client_kwargs(httpx.Timeout(self.model_listing_timeout_seconds))
        )
        return list(client.models.list())

    def tokenize_url(self) -> "str | None":
        return None

    def known_context_limit(self, model: str) -> "int | None":
        # Fallback only — list_models() usually finds the real max_input_tokens
        # first; this covers new models this table hasn't been updated for yet
        # falling through, and any transient failure of the live lookup.
        return _known_anthropic_context_window(model)
