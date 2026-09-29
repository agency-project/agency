"""OpenAI Responses API backend (provider="openai_responses").

An opt-in alternative to `.openai`'s Chat Completions backend. It exists for
models whose tool calling only works on /v1/responses: on Chat Completions,
GPT-6 models accept function tools only with reasoning_effort="none", and
GPT-6 Astra rejects "none" outright -- so Astra cannot run an agent there at
all. On /v1/responses every GPT-6 model (Astra, Sol, GPT-6.1 Sol, Luna) can reason and call tools together.

The request stays stateless, exactly like the Chat Completions path: every
call carries the whole conversation, and store=False tells OpenAI to keep
nothing afterwards. Reasoning therefore has to travel inside the transcript:
each reasoning output item's `encrypted_content` becomes the Agency thinking
block's `signature` (tagged -- see ENCRYPTED_REASONING_TAG), and is sent back
as a reasoning input item next turn.
"""

from __future__ import annotations

from .openai import _OpenAICompatibleBackend

_RESPONSES_TYPE_PREFIX = "openai_responses_"
_METADATA_BLOCK_INDEX = 2**31 - 1  # reserved index, sorts after any real content-block index

# llmconfig fields the Chat Completions backend forwards that /v1/responses
# has no parameter for. Rejected up front rather than silently dropped, so a
# config never looks like it's doing something it isn't.
_UNSUPPORTED_LLM_FIELDS = (
    "frequency_penalty",
    "presence_penalty",
    "n",
    "stop",
    "logprobs",
    "seed",
    "top_k",
    "repetition_penalty",
    "min_p",
    "min_tokens",
    "guided_json",
    "guided_regex",
)

_CHAT_TOOL_CHOICE_STRINGS = ("auto", "required", "none")

# Prefixed to every encrypted_content this backend stores as a thinking
# block's signature. Opaque reasoning state is only valid for the provider
# that produced it: an Anthropic signature sent as encrypted_content (or the
# reverse) is a 400. The tag lives inside the opaque string, not beside it,
# because the string is the only part every harness echoes back verbatim --
# the Codex and Claude Code adapters rebuild thinking blocks from their own
# wire format and would drop any extra block field. Anthropic signatures are
# base64, so they can never start with this.
ENCRYPTED_REASONING_TAG = "openai_responses:"


def _serialize_sdk_object(obj):
    dump = getattr(obj, "model_dump", None)
    return dump() if dump is not None else obj


def _responses_native_block_type(native_type: str) -> str:
    # Same prefix harness/adapters/codex.py uses, so an item this backend
    # can't translate still round-trips to a Responses-speaking harness.
    return f"{_RESPONSES_TYPE_PREFIX}{native_type}"


def _text_of(blocks: "list[dict]") -> str:
    return "".join(b["text"] for b in blocks if b["type"] == "text")


# ---------------------------------------------------------------------------
# Agency -> Responses request
# ---------------------------------------------------------------------------


def _thinking_block_to_reasoning_item(block: dict) -> dict:
    # No "id": with store=False the API checks encrypted_content against the
    # id when one is given, and a mismatched one is a 400 ("Encrypted content
    # item_id did not match"). Harness adapters mint their own reasoning ids
    # (codex.py sends Codex "rs_<uuid>"), so the only safe choice is to omit
    # it -- which the API accepts.
    summary = [{"type": "summary_text", "text": block["text"]}] if block.get("text") else []
    encrypted_content = block["signature"][len(ENCRYPTED_REASONING_TAG) :]
    return {"type": "reasoning", "summary": summary, "encrypted_content": encrypted_content}


def _signature_from_encrypted_content(encrypted_content: "str | None") -> str:
    return f"{ENCRYPTED_REASONING_TAG}{encrypted_content}" if encrypted_content else ""


def _assistant_blocks_to_items(blocks: "list[dict]") -> "list[dict]":
    items = []
    for block in blocks:
        if block["type"] == "text" and block["text"]:
            items.append({"role": "assistant", "content": block["text"]})
        elif block["type"] == "thinking":
            # Only this backend's own reasoning is replayed. Anything else --
            # another provider's signature, or a vLLM reasoning_content trace
            # with none -- can't be verified under store=False and is
            # dropped; the conversation's text and tool calls are kept.
            if (block.get("signature") or "").startswith(ENCRYPTED_REASONING_TAG):
                items.append(_thinking_block_to_reasoning_item(block))
        elif block["type"] == "tool_use":
            items.append(
                {
                    "type": "function_call",
                    "call_id": block["id"],
                    "name": block["name"],
                    "arguments": block["arguments"] or "{}",
                }
            )
        # Other providers' native blocks (anthropic_*, openai_chatcompletions_*)
        # have no Responses meaning; the Chat Completions and Anthropic
        # backends drop foreign blocks the same way.
    return items


def _agency_messages_to_responses_input(messages: "list[dict]") -> "list[dict]":
    items: "list[dict]" = []
    for message in messages:
        role = message["role"]
        blocks = message.get("blocks") or []
        if role == "assistant":
            items.extend(_assistant_blocks_to_items(blocks))
        elif role == "tool":
            result = next((b for b in blocks if b["type"] == "tool_result"), None)
            if result is None:
                continue
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": result.get("tool_call_id", ""),
                    "output": result.get("text", ""),
                }
            )
        else:
            # system, developer and user. /v1/responses accepts system and
            # developer messages anywhere in `input`, so unlike the Chat
            # Completions path (agllm.build_kwargs) nothing is hoisted into
            # `instructions` or rewritten as a user message.
            items.append({"role": role, "content": _text_of(blocks)})
    return items


def _agency_tools_to_responses(tools: "list[dict] | None") -> "list[dict] | None":
    if not tools:
        return None
    converted = []
    for tool in tools:
        fn = tool.get("function", tool)
        responses_tool = {
            "type": "function",
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
        }
        # Only forwarded when the caller chose it: strict mode requires every
        # property be required and additionalProperties false, which
        # arbitrary MCP/native tool schemas don't satisfy.
        if "strict" in fn:
            responses_tool["strict"] = fn["strict"]
        converted.append(responses_tool)
    return converted


def _agency_tool_choice_to_responses(tool_choice):
    """Agency's tool_choice is Chat Completions-shaped; Responses flattens the
    named-function form. Anything else (e.g. a Codex hosted-tool choice,
    carried as an openai_responses_* native block) is dropped, as the Chat
    Completions backend does."""
    if tool_choice in _CHAT_TOOL_CHOICE_STRINGS:
        return tool_choice
    if not isinstance(tool_choice, dict) or tool_choice.get("type") != "function":
        return None
    name = (tool_choice.get("function") or {}).get("name") or tool_choice.get("name")
    if not name:
        return None
    return {"type": "function", "name": name}


# ---------------------------------------------------------------------------
# Responses output -> Agency blocks
# ---------------------------------------------------------------------------


def _reasoning_item_to_thinking_block(item: dict, index: int) -> dict:
    summary_texts = [part.get("text", "") for part in item.get("summary") or []]
    return {
        "type": "thinking",
        "index": index,
        "text": "\n\n".join(summary_texts),
        "signature": _signature_from_encrypted_content(item.get("encrypted_content")),
    }


def _message_item_to_blocks(item: dict, first_index: int) -> "list[dict]":
    blocks = []
    for part in item.get("content") or []:
        index = first_index + len(blocks)
        if part.get("type") == "output_text":
            blocks.append({"type": "text", "index": index, "text": part.get("text", "")})
        else:
            # e.g. a "refusal" part -- kept, like the Chat Completions
            # backend keeps its refusal field, rather than dropped.
            blocks.append(
                {
                    "type": _responses_native_block_type(part.get("type")),
                    "index": index,
                    "data": part,
                }
            )
    return blocks


def _output_items_to_blocks(output) -> "list[dict]":
    blocks: "list[dict]" = []
    for raw_item in output or []:
        item = _serialize_sdk_object(raw_item)
        item_type = item.get("type")
        if item_type == "message":
            blocks.extend(_message_item_to_blocks(item, len(blocks)))
        elif item_type == "reasoning":
            blocks.append(_reasoning_item_to_thinking_block(item, len(blocks)))
        elif item_type == "function_call":
            blocks.append(
                {
                    "type": "tool_use",
                    "index": len(blocks),
                    "id": item.get("call_id", ""),
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments") or "",
                }
            )
        else:
            blocks.append(
                {
                    "type": _responses_native_block_type(item_type),
                    "index": len(blocks),
                    "data": item,
                }
            )
    return blocks


def _stop_reason(response) -> "str | None":
    """Chat Completions finish_reason vocabulary ("stop", "tool_calls",
    "length", "content_filter"): every harness adapter already translates
    from it (claude_code._STOP_REASON_TO_ANTHROPIC, native, codex)."""
    status = getattr(response, "status", None)
    if status == "incomplete":
        reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
        if reason == "max_output_tokens":
            return "length"
        return reason or "length"
    if status != "completed":
        return status
    for raw_item in getattr(response, "output", None) or []:
        if getattr(raw_item, "type", None) == "function_call":
            return "tool_calls"
    return "stop"


def _serialize_responses_usage(usage) -> "dict | None":
    # Same keys as the Chat Completions backend's usage, so the usage
    # tracker and web UI read both paths identically.
    if usage is None:
        return None
    prompt_tokens = getattr(usage, "input_tokens", 0) or 0
    completion_tokens = getattr(usage, "output_tokens", 0) or 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": getattr(usage, "total_tokens", None) or (prompt_tokens + completion_tokens),
    }


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class _StreamBlockIndexes:
    """Assigns each streamed block a stable Agency index on first sight.

    One Responses output item can become several Agency blocks (a message
    item holds text plus possibly a refusal part), so output_index alone
    can't be the block index. Events for a block always arrive after its
    item starts, so first-sight order is also output order."""

    def __init__(self) -> None:
        self._indexes: "dict[tuple, int]" = {}

    def of(self, *key) -> int:
        return self._indexes.setdefault(key, len(self._indexes))


def _finished_item_deltas(item, output_index: int, indexes: _StreamBlockIndexes):
    """Deltas for the parts of an output item that only arrive complete, on
    response.output_item.done: the reasoning signature, a whole tool call
    (emitted once, like the Anthropic backend, instead of argument
    fragments), and any item or content part with no Agency equivalent."""
    data = _serialize_sdk_object(item)
    item_type = data.get("type")
    if item_type == "reasoning":
        if data.get("encrypted_content"):
            yield {
                "type": "block_delta",
                "index": indexes.of(output_index, "thinking"),
                "block_type": "thinking",
                "signature": _signature_from_encrypted_content(data["encrypted_content"]),
            }
        return
    if item_type == "function_call":
        yield {
            "type": "block_delta",
            "index": indexes.of(output_index, "tool_use"),
            "block_type": "tool_use",
            "id": data.get("call_id", ""),
            "name": data.get("name", ""),
            "arguments": data.get("arguments") or "",
        }
        return
    if item_type == "message":
        for content_index, part in enumerate(data.get("content") or []):
            if part.get("type") == "output_text":
                continue  # already streamed through response.output_text.delta
            yield {
                "type": "block_delta",
                "index": indexes.of(output_index, "part", content_index),
                "block_type": _responses_native_block_type(part.get("type")),
                "data": part,
            }
        return
    yield {
        "type": "block_delta",
        "index": indexes.of(output_index, "item"),
        "block_type": _responses_native_block_type(item_type),
        "data": data,
    }


class _OpenAIResponsesBackend(_OpenAICompatibleBackend):
    """OpenAI models via /v1/responses. Shares client construction, the
    explicit-base_url requirement, and model listing with the Chat
    Completions backend it subclasses; only the wire format differs."""

    def _validate_config(self) -> None:
        super()._validate_config()
        unsupported = [
            f for f in _UNSUPPORTED_LLM_FIELDS if getattr(self.agconfig.llm, f) is not None
        ]
        if unsupported:
            raise ValueError(
                f"provider='openai_responses' has no Responses API parameter for "
                f"llmconfig field(s) {unsupported}; unset them or use provider='openai'."
            )

    def _format_context_agency_to_backend(self, request: dict) -> dict:
        llm = self.agconfig.llm
        kwargs: dict = dict(
            model=llm.model or "",
            input=_agency_messages_to_responses_input(request["messages"]),
            max_output_tokens=llm.max_completion_tokens or llm.max_tokens or llm.default_max_tokens,
            store=False,
            # Without this, reasoning items come back with no
            # encrypted_content and can't be replayed under store=False.
            include=["reasoning.encrypted_content"],
        )
        if llm.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": llm.reasoning_effort}
        if llm.temperature is not None:
            kwargs["temperature"] = llm.temperature
        if llm.top_p is not None:
            kwargs["top_p"] = llm.top_p
        if llm.extra_body:
            kwargs["extra_body"] = dict(llm.extra_body)
        tools = _agency_tools_to_responses(request.get("tools"))
        if tools:
            kwargs["tools"] = tools
            # OpenAI rejects tool_choice when no tools are present, so it's
            # only ever sent alongside them.
            tool_choice = _agency_tool_choice_to_responses(request.get("tool_choice"))
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        return kwargs

    def _call_backend(self, backend_request: dict):
        client = self.make_client(self._client_timeout())
        try:
            return client.responses.create(**backend_request)
        finally:
            client.close()

    def _format_context_backend_to_agency(self, raw_result) -> dict:
        blocks = _output_items_to_blocks(getattr(raw_result, "output", None))
        usage = _serialize_responses_usage(getattr(raw_result, "usage", None))
        stop_reason = _stop_reason(raw_result)
        blocks.append(
            {
                "type": "metadata",
                "index": len(blocks),
                "usage": usage,
                "stop_reason": stop_reason,
                "data": _serialize_sdk_object(raw_result),
            }
        )
        return {
            "message": {"role": "assistant", "blocks": blocks},
            "usage": usage,
            "stop_reason": stop_reason,
        }

    def _call_backend_stream(self, backend_request: dict, on_client=None):
        client = self.make_client(self._client_timeout())
        if on_client is not None:
            on_client(client)
        raw_stream = client.responses.create(**backend_request, stream=True)
        return raw_stream, client

    def _format_stream_to_agency(self, raw_stream):
        indexes = _StreamBlockIndexes()
        for event in raw_stream:
            event_type = getattr(event, "type", None)
            if event_type == "response.output_text.delta":
                yield {
                    "type": "block_delta",
                    "index": indexes.of(event.output_index, "text"),
                    "block_type": "text",
                    "text": event.delta,
                }
            elif event_type == "response.reasoning_summary_part.added":
                if event.summary_index > 0:
                    # Separates summary parts the same way the non-streaming
                    # path joins them.
                    yield {
                        "type": "block_delta",
                        "index": indexes.of(event.output_index, "thinking"),
                        "block_type": "thinking",
                        "text": "\n\n",
                    }
            elif event_type == "response.reasoning_summary_text.delta":
                yield {
                    "type": "block_delta",
                    "index": indexes.of(event.output_index, "thinking"),
                    "block_type": "thinking",
                    "text": event.delta,
                }
            elif event_type == "response.output_item.done":
                yield from _finished_item_deltas(event.item, event.output_index, indexes)
            elif event_type in ("response.completed", "response.incomplete"):
                yield from self._final_deltas(event.response)
            elif event_type == "response.failed":
                error = getattr(event.response, "error", None)
                raise RuntimeError(f"OpenAI response failed: {getattr(error, 'message', error)}")
            elif event_type == "error":
                raise RuntimeError(f"OpenAI stream error: {getattr(event, 'message', event)}")

    @staticmethod
    def _final_deltas(response):
        usage = _serialize_responses_usage(getattr(response, "usage", None))
        stop_reason = _stop_reason(response)
        # A plain block_delta, like every other backend's metadata block: it
        # flows through the host server's generic accumulator unchanged.
        yield {
            "type": "block_delta",
            "index": _METADATA_BLOCK_INDEX,
            "block_type": "metadata",
            "data": {
                "usage": usage,
                "stop_reason": stop_reason,
                "raw_response": _serialize_sdk_object(response),
            },
        }
        yield {"type": "usage", "usage": usage, "stop_reason": stop_reason}
