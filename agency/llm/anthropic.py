"""Claude backend, via the first-party Anthropic API (api.anthropic.com)."""

from __future__ import annotations
import json
import os
import re
import httpx

from .agllm import agllm

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None


def _anthropic_sdk_timeout(timeout: httpx.Timeout):
    """The installed anthropic SDK validates its `timeout` kwarg against its
    own Timeout type -- httpx.Timeout in older SDK releases, the separate
    httpx2 package's Timeout in newer ones -- and rejects the other one with
    a TypeError (older releases fail worse: they accept it, then break deep
    in socket setup). `anthropic.Timeout` is the SDK's own stable alias for
    whichever one this installed version actually needs (confirmed via
    identity check against both), so construct through it directly instead
    of guessing which package is installed."""
    if _anthropic_sdk is None:
        return timeout
    return _anthropic_sdk.Timeout(
        connect=timeout.connect, read=timeout.read, write=timeout.write, pool=timeout.pool
    )


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
    """Look up a known context window for an Anthropic model ID (plain or
    Bedrock). Strips region/"anthropic." prefixes, then matches by exact
    or prefix (to tolerate dated snapshot suffixes)."""
    bare = _ANTHROPIC_BEDROCK_MODEL_RE.sub("", model or "")
    for known_id, window in _ANTHROPIC_CONTEXT_WINDOWS.items():
        if bare == known_id or bare.startswith(known_id + "-"):
            return window
    return None


_CACHE_CONTROL = {"type": "ephemeral"}  # prompt-caching breakpoint, default 5-minute TTL


def _serialize_sdk_object(obj):
    dump = getattr(obj, "model_dump", None)
    return dump() if dump is not None else obj


_ANTHROPIC_TYPE_PREFIX = "anthropic_"
_METADATA_BLOCK_INDEX = 2**31 - 1  # reserved index, sorts after any real content-block index


def _anthropic_native_block_type(native_type: str) -> str:
    return f"{_ANTHROPIC_TYPE_PREFIX}{native_type}"


def _flatten_unknown_fragment(fragment) -> dict:
    if not isinstance(fragment, dict):
        return {}
    if "start" in fragment or "deltas" in fragment:
        flat = dict(fragment.get("start") or {})
        for delta in fragment.get("deltas") or []:
            if not isinstance(delta, dict):
                continue
            for k, v in delta.items():
                if isinstance(v, str) and isinstance(flat.get(k), str):
                    flat[k] += v
                else:
                    flat[k] = v
        return flat
    return dict(fragment)


def _unknown_block_to_anthropic(b: dict) -> dict:
    data = b.get("data")
    fragments = data if isinstance(data, list) else [data]
    merged: dict = {}
    for fragment in fragments:
        flat = _flatten_unknown_fragment(fragment)
        for k, v in flat.items():
            if isinstance(v, str) and isinstance(merged.get(k), str):
                merged[k] += v
            else:
                merged[k] = v
    merged["type"] = b["type"][len(_ANTHROPIC_TYPE_PREFIX) :]
    return merged


def _text_block_to_anthropic(b: dict) -> dict:
    block = {"type": "text", "text": b["text"]}
    if b.get("citations"):
        block["citations"] = b["citations"]
    return block


def _agency_messages_to_anthropic(messages: list[dict]) -> "tuple[str | None, list[dict]]":
    system_parts: list[str] = []
    out: list[dict] = []

    for m in messages:
        role = m.get("role")
        blocks = m.get("blocks") or []
        if role == "system":
            text = "".join(b["text"] for b in blocks if b["type"] == "text")
            if text:
                system_parts.append(text)
        elif role == "user":
            has_non_text = any(b["type"] != "text" for b in blocks)
            if not has_non_text:
                text = "".join(b["text"] for b in blocks if b["type"] == "text")
                out.append({"role": "user", "content": text})
            else:
                content_blocks = []
                for b in blocks:
                    if b["type"] == "text":
                        content_blocks.append(_text_block_to_anthropic(b))
                    elif b["type"].startswith(_ANTHROPIC_TYPE_PREFIX):
                        content_blocks.append(_unknown_block_to_anthropic(b))
                out.append({"role": "user", "content": content_blocks})
        elif role == "assistant":
            anthropic_blocks: list[dict] = []
            for b in blocks:
                if b["type"] == "text":
                    anthropic_blocks.append(_text_block_to_anthropic(b))
                elif b["type"] == "thinking":
                    anthropic_blocks.append(
                        {
                            "type": "thinking",
                            "thinking": b["text"],
                            "signature": b.get("signature", ""),
                        }
                    )
                elif b["type"] == "tool_use":
                    try:
                        tool_input = json.loads(b["arguments"] or "{}")
                    except ValueError:
                        tool_input = {}
                    anthropic_blocks.append(
                        {"type": "tool_use", "id": b["id"], "name": b["name"], "input": tool_input}
                    )
                elif b["type"].startswith(_ANTHROPIC_TYPE_PREFIX):
                    anthropic_blocks.append(_unknown_block_to_anthropic(b))
            text_only = "".join(b["text"] for b in blocks if b["type"] == "text")
            out.append({"role": "assistant", "content": anthropic_blocks or text_only})
        elif role == "tool":
            result_block = next((b for b in blocks if b["type"] == "tool_result"), None)
            content = result_block.get("text", "") if result_block else ""
            raw_content = result_block.get("raw_content") if result_block else None
            result = {
                "type": "tool_result",
                "tool_use_id": result_block.get("tool_call_id", "") if result_block else "",
                "content": raw_content if raw_content is not None else content,
            }
            prev_content = out[-1]["content"] if out and out[-1]["role"] == "user" else None
            if isinstance(prev_content, list):
                prev_content.append(result)
            else:
                out.append({"role": "user", "content": [result]})
    return ("\n\n".join(system_parts) or None), out


def _agency_tools_to_anthropic(tools: "list[dict] | None") -> "list[dict] | None":
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


def _agency_tool_choice_to_anthropic(tool_choice):
    if tool_choice is None:
        return None
    if tool_choice == "auto":
        return {"type": "auto"}
    if tool_choice == "required":
        return {"type": "any"}
    if tool_choice == "none":
        return {"type": "none"}
    if isinstance(tool_choice, dict):
        name = tool_choice.get("function", {}).get("name") or tool_choice.get("name")
        if name:
            return {"type": "tool", "name": name}
    return None


def _with_cache_control(content):
    """Return `content` with cache_control on its last block, normalizing a
    bare string into a single text block first (cache_control attaches to a
    content block, not to a string). Caller must pass content it's safe to
    mutate — _agency_messages_to_anthropic() always builds fresh lists/dicts,
    never a reference into the caller's original messages."""
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    else:
        content = list(content)
    if content:
        content[-1] = {**content[-1], "cache_control": _CACHE_CONTROL}
    return content


class _AnthropicBackend(agllm):
    """Claude models via the first-party Anthropic API (api.anthropic.com) —
    the anthropic SDK's plain Anthropic client (Messages API shape).

    For Claude Platform on AWS, prefer provider='anthropicAWS' and the
    AnthropicAWS client instead — it handles SigV4/API-key auth, region-
    derived base URLs, and the workspace header natively.
    """

    def _client_kwargs(self, timeout: httpx.Timeout) -> dict:
        kwargs: dict = dict(
            api_key=self.agconfig.llm.api_key or os.environ.get("ANTHROPIC_API_KEY"),
            timeout=_anthropic_sdk_timeout(timeout),
        )
        workspace_id = self.agconfig.llm.workspace_id or os.environ.get("ANTHROPIC_WORKSPACE_ID")
        if workspace_id:
            kwargs["default_headers"] = {"anthropic-workspace-id": workspace_id}
        return kwargs

    def make_client(self, timeout: httpx.Timeout):
        if _anthropic_sdk is None:
            raise RuntimeError(
                "provider='anthropic' requires the 'anthropic' package: pip install anthropic"
            )
        return _anthropic_sdk.Anthropic(**self._client_kwargs(timeout))

    def list_models(self) -> list:
        if _anthropic_sdk is None:
            return []
        return list(
            self.make_client(
                httpx.Timeout(self.agconfig.llm.model_listing_timeout_seconds)
            ).models.list()
        )

    def tokenize_url(self) -> "str | None":
        return None

    def known_context_limit(self, model: str) -> "int | None":
        # Fallback only: list_models() usually finds the real value first.
        return _known_anthropic_context_window(model)

    @staticmethod
    def _close(client) -> None:
        close = getattr(client, "close", None)
        if close:
            close()

    def _format_context_agency_to_backend(self, request: dict) -> dict:
        system, anthropic_messages = _agency_messages_to_anthropic(request["messages"])
        kwargs: dict = dict(
            model=self.agconfig.llm.model or "",
            messages=anthropic_messages,
            max_tokens=self.agconfig.llm.max_completion_tokens
            or self.agconfig.llm.max_tokens
            or self.agconfig.llm.default_max_tokens,
        )
        if system:
            # Breakpoint on the system prompt: it's the largest, most static
            # part of every request (agent instructions), and tools render
            # before system in Anthropic's prefix order, so this one
            # breakpoint caches tools + system together.
            kwargs["system"] = [{"type": "text", "text": system, "cache_control": _CACHE_CONTROL}]
        if self.agconfig.llm.temperature is not None:
            kwargs["temperature"] = self.agconfig.llm.temperature
        if self.agconfig.llm.top_p is not None:
            kwargs["top_p"] = self.agconfig.llm.top_p
        extra_body = self.agconfig.llm.extra_body or {}
        if "top_k" in extra_body:
            kwargs["top_k"] = extra_body["top_k"]
        anthropic_tools = _agency_tools_to_anthropic(request.get("tools"))
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        anthropic_tool_choice = _agency_tool_choice_to_anthropic(request.get("tool_choice"))
        if anthropic_tool_choice is not None:
            kwargs["tool_choice"] = anthropic_tool_choice
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
        return kwargs

    def _call_backend(self, backend_request: dict):
        client = self.make_client(self._client_timeout())
        try:
            return client.messages.create(**backend_request)
        finally:
            self._close(client)

    def _format_context_backend_to_agency(self, raw_result) -> dict:
        blocks: "list[dict]" = []
        for i, b in enumerate(raw_result.content):
            btype = getattr(b, "type", None)
            if btype == "text":
                block = {"type": "text", "index": i, "text": b.text}
                citations = getattr(b, "citations", None)
                if citations:
                    block["citations"] = [_serialize_sdk_object(c) for c in citations]
                blocks.append(block)
            elif btype == "tool_use":
                blocks.append(
                    {
                        "type": "tool_use",
                        "index": i,
                        "id": b.id,
                        "name": b.name,
                        "arguments": json.dumps(b.input),
                    }
                )
            elif btype == "thinking":
                blocks.append(
                    {
                        "type": "thinking",
                        "index": i,
                        "text": getattr(b, "thinking", "") or "",
                        "signature": getattr(b, "signature", "") or "",
                    }
                )
            else:
                blocks.append(
                    {
                        "type": _anthropic_native_block_type(btype),
                        "index": i,
                        "data": _serialize_sdk_object(b),
                    }
                )
        usage = getattr(raw_result, "usage", None)
        input_tokens = (getattr(usage, "input_tokens", 0) or 0) if usage else 0
        output_tokens = (getattr(usage, "output_tokens", 0) or 0) if usage else 0
        cache_read_tokens = (getattr(usage, "cache_read_input_tokens", 0) or 0) if usage else 0
        cache_write_tokens = (getattr(usage, "cache_creation_input_tokens", 0) or 0) if usage else 0
        usage_dict = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
        }
        stop_reason = getattr(raw_result, "stop_reason", None)
        blocks.append(
            {
                "type": "metadata",
                "index": len(blocks),
                "usage": usage_dict,
                "stop_reason": stop_reason,
                "data": _serialize_sdk_object(raw_result),
            }
        )
        return {
            "message": {"role": "assistant", "blocks": blocks},
            "usage": usage_dict,
            "stop_reason": stop_reason,
        }

    def _call_backend_stream(self, backend_request: dict, on_client=None):
        client = self.make_client(self._client_timeout())
        if on_client is not None:
            on_client(client)
        raw_stream = client.messages.create(stream=True, **backend_request)
        return raw_stream, client

    def _format_stream_to_agency(self, raw_stream):
        """Tool-call JSON input is buffered per content block and emitted as
        a single item on content_block_stop, rather than fragment-by-
        fragment. Partial tool-call arguments are never rendered to a live
        viewer anyway, so nothing is lost — and emitting one complete item
        instead of many small ones avoids relying on every fragment
        individually surviving whatever consumes this generator."""
        input_tokens = 0
        output_tokens = 0
        cache_read_tokens = 0
        cache_write_tokens = 0
        stop_reason = None
        tool_blocks: "dict[int, dict]" = {}  # index -> {"id", "name", "json_parts"}
        unknown_blocks: "dict[int, dict]" = {}  # index -> {"native_type", "start", "deltas"}
        metadata_events: "list[dict]" = []  # raw message_start/message_delta events, for the metadata block

        for event in raw_stream:
            etype = getattr(event, "type", None)
            if etype == "message_start":
                usage = getattr(event.message, "usage", None)
                if usage is not None:
                    input_tokens = getattr(usage, "input_tokens", 0) or 0
                    cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
                    cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
                    metadata_events.append(
                        {"event": "message_start", "usage": _serialize_sdk_object(usage)}
                    )
            elif etype == "content_block_start":
                block = event.content_block
                if block.type == "tool_use":
                    tool_blocks[event.index] = {
                        "id": block.id,
                        "name": block.name,
                        "json_parts": [],
                    }
                elif block.type not in ("text", "thinking"):
                    unknown_blocks[event.index] = {
                        "native_type": block.type,
                        "start": _serialize_sdk_object(block),
                        "deltas": [],
                    }
            elif etype == "content_block_delta":
                delta = event.delta
                kind = getattr(delta, "type", None)
                if event.index in unknown_blocks:
                    unknown_blocks[event.index]["deltas"].append(_serialize_sdk_object(delta))
                elif kind == "text_delta":
                    yield {
                        "type": "block_delta",
                        "index": event.index,
                        "block_type": "text",
                        "text": delta.text,
                    }
                elif kind == "thinking_delta":
                    yield {
                        "type": "block_delta",
                        "index": event.index,
                        "block_type": "thinking",
                        "text": delta.thinking,
                    }
                elif kind == "signature_delta":
                    yield {
                        "type": "block_delta",
                        "index": event.index,
                        "block_type": "thinking",
                        "signature": delta.signature,
                    }
                elif kind == "input_json_delta":
                    block = tool_blocks.get(event.index)
                    if block is not None:
                        block["json_parts"].append(delta.partial_json or "")
                elif kind == "citations_delta":
                    yield {
                        "type": "block_delta",
                        "index": event.index,
                        "block_type": "text",
                        "citations": [_serialize_sdk_object(getattr(delta, "citation", None))],
                    }
            elif etype == "content_block_stop":
                block = tool_blocks.pop(event.index, None)
                if block is not None:
                    yield {
                        "type": "block_delta",
                        "index": event.index,
                        "block_type": "tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "arguments": "".join(block["json_parts"]),
                    }
                unknown = unknown_blocks.pop(event.index, None)
                if unknown is not None:
                    yield {
                        "type": "block_delta",
                        "index": event.index,
                        "block_type": _anthropic_native_block_type(unknown["native_type"]),
                        "data": {"start": unknown["start"], "deltas": unknown["deltas"]},
                    }
            elif etype == "message_delta":
                usage = getattr(event, "usage", None)
                if usage is not None:
                    output_tokens = getattr(usage, "output_tokens", 0) or output_tokens
                delta_stop_reason = getattr(getattr(event, "delta", None), "stop_reason", None)
                if delta_stop_reason is not None:
                    stop_reason = delta_stop_reason
                metadata_events.append(
                    {"event": "message_delta", "data": _serialize_sdk_object(event)}
                )

        # If the stream ended (e.g. stop_reason="max_tokens") while a tool_use
        # block was still open, content_block_stop never fires for it and the
        # tool call would otherwise vanish with no trace — the assistant turn
        # comes out completely empty and callers loop forever re-requesting it.
        # Flush whatever JSON was collected so far instead; a downstream
        # json.loads() failure on truncated arguments is at least visible.
        for index in sorted(tool_blocks):
            block = tool_blocks[index]
            print(
                f"[agllm] WARNING: tool_use block {block['name']!r} "
                f"(id={block['id']}) truncated mid-stream (likely hit max_tokens) "
                f"— flushing partial arguments instead of dropping the call"
            )
            yield {
                "type": "block_delta",
                "index": index,
                "block_type": "tool_use",
                "id": block["id"],
                "name": block["name"],
                "arguments": "".join(block["json_parts"]),
            }

        for index in sorted(unknown_blocks):
            unknown = unknown_blocks[index]
            yield {
                "type": "block_delta",
                "index": index,
                "block_type": _anthropic_native_block_type(unknown["native_type"]),
                "data": {"start": unknown["start"], "deltas": unknown["deltas"]},
            }

        usage_dict = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
        }
        # The metadata block is a plain block_delta -- like any other unknown
        # native block, it flows through the host server's generic block
        # accumulator with no dedicated handling required there. All
        # translation of what "usage"/"stop_reason" mean stays in this
        # backend, not the host server.
        yield {
            "type": "block_delta",
            "index": _METADATA_BLOCK_INDEX,
            "block_type": "metadata",
            "data": {
                "usage": usage_dict,
                "stop_reason": stop_reason,
                "raw_events": metadata_events,
            },
        }
        yield {
            "type": "usage",
            "usage": usage_dict,
            "stop_reason": stop_reason,
        }
