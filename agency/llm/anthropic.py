"""Claude backend, via the first-party Anthropic API (api.anthropic.com)."""

from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass

import httpx

from .agllm import agllm
from .openai_responses import ENCRYPTED_REASONING_TAG

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None

_anthropic_timeout_type = _anthropic_sdk.Timeout if _anthropic_sdk is not None else None


def _anthropic_sdk_timeout(timeout: httpx.Timeout):
    """The installed anthropic SDK validates its `timeout` kwarg against its
    own Timeout type -- httpx.Timeout in older SDK releases, the separate
    httpx2 package's Timeout in newer ones -- and rejects the other one with
    a TypeError (older releases fail worse: they accept it, then break deep
    in socket setup). `anthropic.Timeout` is the SDK's own stable alias for
    whichever one this installed version actually needs (confirmed via
    identity check against both), so construct through it directly instead
    of guessing which package is installed."""
    if _anthropic_sdk is None or _anthropic_timeout_type is None:
        return timeout
    return _anthropic_timeout_type(
        connect=timeout.connect, read=timeout.read, write=timeout.write, pool=timeout.pool
    )


# Matches the region + "anthropic." prefix Bedrock model IDs carry (e.g.
# "us.anthropic.claude-sonnet-5-...") -- a no-op substitution on plain
# api.anthropic.com model IDs ("claude-sonnet-5"), which carry no such
# prefix, so _known_anthropic_context_window() below is shared as-is by both
# this module's _AnthropicBackend and .bedrock's Anthropic-family backends.
_ANTHROPIC_BEDROCK_MODEL_RE = re.compile(r"^(?:(?:us|eu|apac|global)\.)?anthropic\.")

_EFFORT_ALL = frozenset({"low", "medium", "high", "xhigh", "max"})
_EFFORT_NO_XHIGH = frozenset({"low", "medium", "high", "max"})
_EFFORT_BASIC = frozenset({"low", "medium", "high"})


@dataclass(frozen=True)
class _AnthropicModelInfo:
    """What Agency knows about one Claude model's request surface. `None`
    means unverified: the request field passes through untouched and the
    API decides, exactly as for a model missing from the table."""

    context_window: int
    # False: tool_choice {"type": "any"} / {"type": "tool"} is a 400.
    forced_tool_choice: "bool | None" = None
    # "none": temperature other than 1.0, and any top_p/top_k, are a 400.
    # "exclusive": each is accepted alone, but temperature + top_p is a 400.
    sampling: "str | None" = None
    # Accepted output_config.effort levels; empty = effort is a 400.
    effort: "frozenset[str] | None" = None
    # Whether role:"system" is accepted inside `messages`. Informational for
    # now: mid-conversation system-class messages are still sent as `user`
    # on every model (the native form has placement rules of its own).
    mid_conversation_system: "bool | None" = None


# Keyed by the bare model name, after stripping the region and "anthropic."
# prefix Bedrock IDs carry — see _anthropic_model_info below. Context windows
# are also the static fallback for Bedrock, whose invoke_model API has no
# /v1/models endpoint; where the first-party Models API lists a model, the
# value here is its max_input_tokens. The restriction fields were verified
# against the live API on 2026-09-29 (Mythos entries: Anthropic's docs, same
# surface as their Fable counterparts). Update when new models ship.
_ANTHROPIC_MODELS: "dict[str, _AnthropicModelInfo]" = {
    "claude-fable-5-1": _AnthropicModelInfo(1_000_000, False, "none", _EFFORT_ALL, True),
    "claude-mythos-5-1": _AnthropicModelInfo(1_000_000, False, "none", _EFFORT_ALL, True),
    "claude-fable-5": _AnthropicModelInfo(1_000_000, True, "none", _EFFORT_ALL, True),
    "claude-mythos-5": _AnthropicModelInfo(1_000_000, True, "none", _EFFORT_ALL, True),
    "claude-mythos-preview": _AnthropicModelInfo(1_000_000),
    "claude-opus-5-5": _AnthropicModelInfo(1_000_000, False, "none", _EFFORT_ALL, True),
    "claude-opus-5": _AnthropicModelInfo(1_000_000, True, "none", _EFFORT_ALL, True),
    "claude-opus-4-8": _AnthropicModelInfo(1_000_000, True, "none", _EFFORT_ALL, True),
    "claude-opus-4-7": _AnthropicModelInfo(1_000_000, True, "none", _EFFORT_ALL, False),
    "claude-opus-4-6": _AnthropicModelInfo(1_000_000, True, "exclusive", _EFFORT_NO_XHIGH, False),
    "claude-opus-4-5": _AnthropicModelInfo(200_000, True, "exclusive", _EFFORT_BASIC, False),
    "claude-opus-4-1": _AnthropicModelInfo(1_000_000),
    "claude-opus-4-0": _AnthropicModelInfo(1_000_000),
    "claude-sonnet-5-5": _AnthropicModelInfo(1_000_000, False, "none", _EFFORT_ALL, True),
    "claude-sonnet-5": _AnthropicModelInfo(1_000_000, True, "none", _EFFORT_ALL, True),
    "claude-sonnet-4-6": _AnthropicModelInfo(1_000_000, True, "exclusive", _EFFORT_NO_XHIGH, False),
    "claude-sonnet-4-5": _AnthropicModelInfo(1_000_000, True, "exclusive", frozenset(), False),
    "claude-sonnet-4-0": _AnthropicModelInfo(1_000_000),
    "claude-haiku-4-5": _AnthropicModelInfo(200_000, True, "exclusive", frozenset(), False),
}

# What may follow a known bare model name and still be that same model: a
# dated snapshot ("-20251001") and/or a Bedrock version tag ("-v1:0"). Anything
# else is a different model -- "claude-fable-5-1" is a new model, not a
# snapshot of "claude-fable-5", and must never inherit its metadata.
_ANTHROPIC_SNAPSHOT_SUFFIX_RE = re.compile(r"(?:-\d{8})?(?:-v\d+(?::\d+)?)?")


def _anthropic_model_info(model: str) -> "_AnthropicModelInfo | None":
    """Look up what's known about an Anthropic model ID (plain or Bedrock).
    Strips region/"anthropic." prefixes, then matches the bare name exactly
    or followed only by a snapshot/version suffix -- never a family prefix,
    so an unlisted model inherits no restrictions."""
    bare = _ANTHROPIC_BEDROCK_MODEL_RE.sub("", model or "")
    for known_id, info in _ANTHROPIC_MODELS.items():
        if bare.startswith(known_id) and _ANTHROPIC_SNAPSHOT_SUFFIX_RE.fullmatch(
            bare[len(known_id) :]
        ):
            return info
    return None


def _known_anthropic_context_window(model: str) -> "int | None":
    info = _anthropic_model_info(model)
    return info.context_window if info is not None else None


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


# OpenAI's Responses API added "developer" as a companion/successor to
# "system" (Codex emits it); both are operator instructions. Same set as
# agllm.build_kwargs's _SYSTEM_CLASS_ROLES.
_SYSTEM_CLASS_ROLES = ("system", "developer")


def _flush_deferred_text(out: list[dict], deferred: "list[str]") -> None:
    """Emit system-class text held back while tool results were pending:
    after the tool_result blocks of the same user message when there is one
    (text may follow tool results there, never precede them), else as its
    own user message."""
    if not deferred:
        return
    prev_content = out[-1]["content"] if out and out[-1]["role"] == "user" else None
    if isinstance(prev_content, list):
        prev_content.extend({"type": "text", "text": text} for text in deferred)
    else:
        out.extend({"role": "user", "content": text} for text in deferred)
    deferred.clear()


def _agency_messages_to_anthropic(messages: list[dict]) -> "tuple[str | None, list[dict]]":
    system_parts: list[str] = []
    out: list[dict] = []
    conversation_started = False
    # tool_use ids of the latest assistant turn still waiting for a result,
    # and system-class text that arrived meanwhile: nothing may sit between
    # a tool_use and its tool_result message (a 400).
    pending_tool_ids: "set[str]" = set()
    deferred_text: "list[str]" = []

    for m in messages:
        role = m.get("role")
        blocks = m.get("blocks") or []
        if role not in _SYSTEM_CLASS_ROLES:
            conversation_started = True
            if role != "tool":
                pending_tool_ids.clear()
                _flush_deferred_text(out, deferred_text)
        if role in _SYSTEM_CLASS_ROLES:
            text = "".join(b["text"] for b in blocks if b["type"] == "text")
            if not text:
                continue
            if not conversation_started:
                system_parts.append(text)
            else:
                print(
                    f"[anthropic] WARNING: harness emitted a mid-conversation "
                    f"{role}-role message -- not valid per the Anthropic Messages "
                    "API (system must be the top-level `system` field, never a "
                    "`messages` entry); sending it as `user` content at its "
                    "original position instead (after any pending tool results)"
                )
                if pending_tool_ids:
                    deferred_text.append(text)
                else:
                    out.append({"role": "user", "content": text})
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
            dropped_thinking = 0
            for b in blocks:
                if b["type"] == "text":
                    anthropic_blocks.append(_text_block_to_anthropic(b))
                elif b["type"] == "thinking":
                    signature = b.get("signature") or ""
                    if not signature or signature.startswith(ENCRYPTED_REASONING_TAG):
                        # Only Claude-signed blocks can be replayed: an
                        # unsigned one (vLLM/Chat Completions reasoning_content,
                        # Converse reasoningText without a signature) or
                        # OpenAI encrypted reasoning from provider=
                        # "openai_responses" is a 400 ("Invalid `signature`"),
                        # on every Claude model. Only the rest of the turn is
                        # replayed; nothing is ever re-signed.
                        dropped_thinking += 1
                        continue
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
            if dropped_thinking:
                _warn_once(
                    ("thinking", "unsigned"),
                    "history carries thinking blocks without a Claude signature "
                    "(another provider's reasoning); they are not sent to Anthropic",
                )
                if not anthropic_blocks:
                    # Nothing but foreign reasoning -- an empty assistant turn
                    # would itself be rejected.
                    continue
            text_only = "".join(b["text"] for b in blocks if b["type"] == "text")
            out.append({"role": "assistant", "content": anthropic_blocks or text_only})
            pending_tool_ids = {b["id"] for b in anthropic_blocks if b["type"] == "tool_use"}
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
            pending_tool_ids.discard(result["tool_use_id"])
            if not pending_tool_ids:
                _flush_deferred_text(out, deferred_text)
    _flush_deferred_text(out, deferred_text)
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


_WARNED_ONCE: "set[tuple]" = set()


def _warn_once(key: tuple, message: str) -> None:
    """Config-driven warnings would otherwise repeat on every call of an
    agent loop; one per process per distinct (model, cause) is enough."""
    if key not in _WARNED_ONCE:
        _WARNED_ONCE.add(key)
        print(f"[anthropic] WARNING: {message}")


def _tool_choice_for_model(tool_choice, model: str, info: "_AnthropicModelInfo | None"):
    """Degrade forced tool choice to `auto` on models that reject it.

    Semantic tradeoff: `required` ("at least one tool call") and a named tool
    ("call exactly this one") cannot be enforced by the provider on these
    models -- the API 400s on {"type": "any"} / {"type": "tool"} regardless
    of thinking settings. Agency itself never requests forced choice (its
    own must-call contract, structured output via submit_output, is enforced
    by prompt + engine retry); only a harness does, and a harness validates
    its own tool loop. Degrading keeps the attempt alive where a hard error
    would end it. No steering text is injected: a per-request message that
    the harness doesn't keep in its history would edit the transcript and
    invalidate later thinking blocks. Models not verified to reject forced
    choice (older Claude, unknown IDs) keep it unchanged."""
    if not isinstance(tool_choice, dict) or tool_choice.get("type") not in ("any", "tool"):
        return tool_choice
    if info is None or info.forced_tool_choice is not False:
        return tool_choice
    requested = (
        "required" if tool_choice["type"] == "any" else f"named tool {tool_choice.get('name')!r}"
    )
    # Every occurrence, not once: each one is a request whose tool call the
    # provider no longer guarantees.
    print(
        f"[anthropic] WARNING: {model} does not support provider-level tool "
        f"enforcement; the requested tool_choice ({requested}) was converted to "
        f"`auto`, so tool use is no longer guaranteed by the provider for this request"
    )
    return {"type": "auto"}


def _sampling_for_model(sampling: dict, model: str, info: "_AnthropicModelInfo | None") -> dict:
    """Drop the sampling fields a known model would 400 on. Unknown models,
    and models whose sampling rules are unverified, get everything as set."""
    policy = info.sampling if info is not None else None
    kept = dict(sampling)
    if policy == "none":
        dropped = [k for k in ("top_p", "top_k") if k in kept]
        if "temperature" in kept and kept["temperature"] != 1:
            dropped.insert(0, "temperature")
        for key in dropped:
            del kept[key]
        if dropped:
            _warn_once(
                (model, "sampling", tuple(dropped)),
                f"{model} rejects sampling parameters (only temperature=1.0 is "
                f"accepted); not sending {', '.join(dropped)}",
            )
    elif policy == "exclusive" and "temperature" in kept and "top_p" in kept:
        # Deterministic: temperature is the more commonly set knob.
        del kept["top_p"]
        _warn_once(
            (model, "sampling", "top_p"),
            f"{model} accepts temperature or top_p but not both; sending temperature and not top_p",
        )
    return kept


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

    def retrieve_model(self, model: str):
        # GET /v1/models/{id} resolves aliases ("claude-haiku-4-5") that the
        # listing only carries under their dated ID.
        if _anthropic_sdk is None or not model:
            return None
        client = self.make_client(httpx.Timeout(self.agconfig.llm.model_listing_timeout_seconds))
        try:
            return client.models.retrieve(model)
        except _anthropic_sdk.NotFoundError:
            return None
        finally:
            self._close(client)

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
        model = self.agconfig.llm.model or ""
        info = _anthropic_model_info(model)
        system, anthropic_messages = _agency_messages_to_anthropic(request["messages"])
        kwargs: dict = dict(
            model=model,
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
        sampling: dict = {}
        if self.agconfig.llm.temperature is not None:
            sampling["temperature"] = self.agconfig.llm.temperature
        if self.agconfig.llm.top_p is not None:
            sampling["top_p"] = self.agconfig.llm.top_p
        extra_body = self.agconfig.llm.extra_body or {}
        if "top_k" in extra_body:
            sampling["top_k"] = extra_body["top_k"]
        kwargs.update(_sampling_for_model(sampling, model, info))
        anthropic_tools = _agency_tools_to_anthropic(request.get("tools"))
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        anthropic_tool_choice = _tool_choice_for_model(
            _agency_tool_choice_to_anthropic(request.get("tool_choice")), model, info
        )
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
