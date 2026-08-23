"""Reverse-direction wire adapters for `agproxy_llm`: harness-native request
formats (Anthropic Messages API, OpenAI Responses API) <-> the
`.chat.completions.create(**kwargs)` shape every `agllm_backend.make_client()`
exposes uniformly (see `llm/base.py`).

This is deliberately the *opposite* direction from `llm/anthropic.py`'s
`_openai_messages_to_anthropic`/`_anthropic_stream_to_openai_chunks` (those
convert agency's own OpenAI-shaped request into a real outbound Anthropic API
call). Here, an off-the-shelf harness (Claude Code, Codex) is the *caller*,
speaking its own native wire format to `agproxy_llm`; this module reshapes
that request into OpenAI chat-completions kwargs, invokes the SAME uniform
`client.chat.completions.create()` every other backend uses, then reshapes
the (possibly streaming) response back into the harness's expected format.

Fidelity is bounded on purpose: this is a format *translation*, not a proxy
to the harness's real provider, so anything with no chat-completions
equivalent is dropped rather than erroring -- extended thinking blocks,
prompt-cache breakpoints (`cache_control`), and image content blocks are all
silently stripped. See docs/Design_harness_integration.md's "Design
Tensions" section: this is the documented cost of `gateway_mode="translate"`.
"""

from __future__ import annotations

import json
import time
import uuid


# ---------------------------------------------------------------------------
# Anthropic Messages API <-> OpenAI chat.completions
# ---------------------------------------------------------------------------


def _stringify_anthropic_content(content) -> str:
    """Anthropic tool_result content can be a bare string or a list of
    content blocks (text/image) -- OpenAI's `tool` role wants a single string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return "" if content is None else str(content)


def _anthropic_tool_choice_to_openai(tool_choice):
    if not tool_choice:
        return None
    kind = tool_choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "none":
        return "none"
    if kind == "any":
        return "required"
    if kind == "tool":
        return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
    return None


def anthropic_tools_to_openai(tools) -> "list[dict] | None":
    if not tools:
        return None
    converted = []
    for t in tools:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return converted


def _anthropic_system_to_text(system) -> "str | None":
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    if isinstance(system, list):
        parts = [
            b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"
        ]
        joined = "".join(parts)
        return joined or None
    return None


def anthropic_messages_to_openai(body: dict) -> dict:
    """Anthropic `POST /v1/messages` request body -> OpenAI
    `client.chat.completions.create(**kwargs)` kwargs."""
    openai_messages: list[dict] = []

    system_text = _anthropic_system_to_text(body.get("system"))

    # Claude Code injects its own mid-conversation `system`-role messages
    # directly into `messages` (a "system reminder"), not just the
    # top-level `system` field above -- confirmed against a real captured
    # request: `[('user', list), ('system', str), ('assistant', list),
    # ...]`. Every mid-array occurrence is collected here (not just the
    # first -- there is no guarantee only one ever appears) and folded
    # into the SAME leading system message below, mirroring exactly what
    # `_openai_messages_to_anthropic` (llm/anthropic.py:100-144)
    # already does for the reverse direction: accumulate every
    # `role == "system"` message's text, wherever it appears, and join
    # them into one system context -- never leave a second `system` entry
    # in the messages array. Confirmed via a real captured Bedrock-native
    # request that this is also what Claude Code's own properly-targeted
    # Bedrock client does (system-role content never appears mid-array on
    # that path either; it's absorbed into a single leading `system`
    # field). Folding preserves the content's actual semantic weight
    # (system-level authority), instead of recasting it as if the user
    # said it, and never inserts anything at its original array position.
    extra_system_parts: list[str] = []

    for m in body.get("messages", []):
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            extra_text = _anthropic_system_to_text(content)
            if extra_text:
                extra_system_parts.append(extra_text)
            continue
        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            openai_messages.append({"role": role, "content": ""})
            continue

        if role == "user":
            text_parts = []
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_result":
                    if text_parts:
                        openai_messages.append({"role": "user", "content": "".join(text_parts)})
                        text_parts = []
                    openai_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": _stringify_anthropic_content(block.get("content")),
                        }
                    )
                # image / other block types: no chat-completions text equivalent, dropped
            if text_parts:
                openai_messages.append({"role": "user", "content": "".join(text_parts)})
        elif role == "assistant":
            text_parts = []
            tool_calls = []
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        }
                    )
                # thinking / redacted_thinking blocks: no chat-completions equivalent, dropped
            msg = {"role": "assistant", "content": "".join(text_parts) or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            openai_messages.append(msg)
        # unrecognized roles are dropped rather than sent to a backend that would reject them

    combined_system = (
        "\n\n".join([system_text] + extra_system_parts)
        if system_text
        else "\n\n".join(extra_system_parts)
    )
    if combined_system:
        openai_messages.insert(0, {"role": "system", "content": combined_system})

    kwargs: dict = {
        "model": body.get("model", ""),
        "messages": openai_messages,
        "stream": bool(body.get("stream", False)),
    }
    if "max_tokens" in body:
        # Agency's canonical OpenAI-compatible request shape uses the
        # current Chat Completions parameter. GPT-5-family models reject
        # the legacy max_tokens spelling outright.
        kwargs["max_completion_tokens"] = body["max_tokens"]
    if "temperature" in body:
        kwargs["temperature"] = body["temperature"]
    if "top_p" in body:
        kwargs["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        kwargs["stop"] = body["stop_sequences"]
    tools = anthropic_tools_to_openai(body.get("tools"))
    if tools:
        kwargs["tools"] = tools
    tool_choice = _anthropic_tool_choice_to_openai(body.get("tool_choice"))
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    # "thinking" (extended thinking config): no chat-completions equivalent, dropped
    return kwargs


_FINISH_REASON_TO_STOP_REASON = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
}


def _finish_reason_to_stop_reason(finish_reason: "str | None") -> str:
    return _FINISH_REASON_TO_STOP_REASON.get(finish_reason or "", "end_turn")


def openai_response_to_anthropic_message(resp, model: str, request_id: "str | None" = None) -> dict:
    """Non-streaming OpenAI chat-completion response -> an Anthropic
    Messages API response object."""
    choice = resp.choices[0]
    message = choice.message
    content_blocks = []
    text = getattr(message, "content", None)
    if text:
        content_blocks.append({"type": "text", "text": text})
    for tc in getattr(message, "tool_calls", None) or []:
        try:
            tool_input = json.loads(tc.function.arguments or "{}")
        except (ValueError, AttributeError):
            tool_input = {}
        content_blocks.append(
            {"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": tool_input}
        )

    usage = getattr(resp, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0

    return {
        "id": request_id or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": _finish_reason_to_stop_reason(getattr(choice, "finish_reason", None)),
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def openai_chunks_to_anthropic_sse(chunks, model: str, request_id: "str | None" = None):
    """OpenAI-style streaming chunks (real SDK chunks, or the `_FakeChunk`
    shape `llm/anthropic.py`'s own Anthropic-backed shim yields --
    both expose `.choices[].delta.content`/`.tool_calls` and `.usage`) ->
    an Anthropic Messages API SSE stream."""
    request_id = request_id or f"msg_{uuid.uuid4().hex}"

    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": request_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    next_index = 0
    text_index = None
    tool_blocks: dict = {}  # openai tool-call index -> {"anthropic_index", "id", "name", "started"}
    stop_reason = "end_turn"
    input_tokens = 0
    output_tokens = 0

    for chunk in chunks:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "prompt_tokens", 0) or input_tokens
            output_tokens = getattr(usage, "completion_tokens", 0) or output_tokens

        for choice in getattr(chunk, "choices", None) or []:
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason:
                stop_reason = _finish_reason_to_stop_reason(finish_reason)

            delta = choice.delta
            content = getattr(delta, "content", None)
            if content:
                if text_index is None:
                    text_index = next_index
                    next_index += 1
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": text_index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": text_index,
                        "delta": {"type": "text_delta", "text": content},
                    },
                )

            for tc in getattr(delta, "tool_calls", None) or []:
                tc_index = getattr(tc, "index", 0)
                if tc_index not in tool_blocks:
                    if text_index is not None:
                        yield _sse(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": text_index},
                        )
                        text_index = None
                    anthropic_index = next_index
                    next_index += 1
                    tool_blocks[tc_index] = {"anthropic_index": anthropic_index}
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": anthropic_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": getattr(tc, "id", "") or "",
                                "name": getattr(tc.function, "name", "") or "",
                            },
                        },
                    )
                fn = getattr(tc, "function", None)
                arguments = getattr(fn, "arguments", None) if fn else None
                if arguments:
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": tool_blocks[tc_index]["anthropic_index"],
                            "delta": {"type": "input_json_delta", "partial_json": arguments},
                        },
                    )

    if text_index is not None:
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": text_index})
    for block in tool_blocks.values():
        yield _sse(
            "content_block_stop", {"type": "content_block_stop", "index": block["anthropic_index"]}
        )

    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# OpenAI Responses API <-> OpenAI chat.completions
# ---------------------------------------------------------------------------


def responses_tools_to_openai(tools) -> "list[dict] | None":
    """Responses API tool defs are flat (`{"type":"function","name":...,
    "description":...,"parameters":...}`) -- chat-completions nests them
    under a `function` key."""
    if not tools:
        return None
    converted = []
    for t in tools:
        if t.get("type") != "function":
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return converted or None


def _responses_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype in ("input_text", "output_text", "text"):
                parts.append(block.get("text", ""))
            # input_image / other block types: no chat-completions text equivalent, dropped
        return "".join(parts)
    return "" if content is None else str(content)


def responses_request_to_openai(body: dict) -> dict:
    """OpenAI `POST /v1/responses` request body -> `client.chat.completions.
    create(**kwargs)` kwargs."""
    openai_messages: list[dict] = []

    instructions = body.get("instructions")
    if instructions:
        openai_messages.append({"role": "system", "content": instructions})

    raw_input = body.get("input")
    if isinstance(raw_input, str):
        openai_messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        for item in raw_input:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "function_call":
                openai_messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": item.get("call_id", ""),
                                "type": "function",
                                "function": {
                                    "name": item.get("name", ""),
                                    "arguments": item.get("arguments", "{}"),
                                },
                            }
                        ],
                    }
                )
            elif itype == "function_call_output":
                openai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": _responses_content_to_text(item.get("output")),
                    }
                )
            else:
                # a plain {"role": ..., "content": [...]} message item
                role = item.get("role", "user")
                openai_messages.append(
                    {"role": role, "content": _responses_content_to_text(item.get("content"))}
                )

    kwargs: dict = {
        "model": body.get("model", ""),
        "messages": openai_messages,
        "stream": bool(body.get("stream", False)),
    }
    if "max_output_tokens" in body:
        kwargs["max_completion_tokens"] = body["max_output_tokens"]
    if "temperature" in body:
        kwargs["temperature"] = body["temperature"]
    if "top_p" in body:
        kwargs["top_p"] = body["top_p"]
    tools = responses_tools_to_openai(body.get("tools"))
    if tools:
        kwargs["tools"] = tools
    return kwargs


def openai_response_to_responses_api(resp, model: str, request_id: "str | None" = None) -> dict:
    """Non-streaming OpenAI chat-completion response -> an OpenAI Responses
    API response object."""
    request_id = request_id or f"resp_{uuid.uuid4().hex}"
    choice = resp.choices[0]
    message = choice.message
    output = []

    text = getattr(message, "content", None)
    if text:
        output.append(
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    for tc in getattr(message, "tool_calls", None) or []:
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex}",
                "call_id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
                "status": "completed",
            }
        )

    usage = getattr(resp, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0

    return {
        "id": request_id,
        "object": "response",
        "created_at": time.time(),
        "status": "completed",
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def openai_chunks_to_responses_sse(chunks, model: str, request_id: "str | None" = None):
    """OpenAI-style streaming chunks -> a Responses API SSE stream.

    Not verified against a live `codex` binary (see codex.py's docstring) --
    implemented from the documented Responses API streaming event shapes
    (`response.created` / `response.output_item.added` / `response.output_text.
    delta` / `response.output_item.done` / `response.completed`), covering the
    text-message and function-call item cases only.
    """
    request_id = request_id or f"resp_{uuid.uuid4().hex}"

    yield _sse(
        "response.created",
        {
            "type": "response.created",
            "response": {
                "id": request_id,
                "object": "response",
                "status": "in_progress",
                "model": model,
            },
        },
    )

    output_index = 0
    text_item_id = None
    text_parts: list[str] = []
    tool_blocks: dict = {}  # openai tool-call index -> {"output_index", "id", "call_id", "name", "args_parts"}
    input_tokens = 0
    output_tokens = 0

    for chunk in chunks:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "prompt_tokens", 0) or input_tokens
            output_tokens = getattr(usage, "completion_tokens", 0) or output_tokens

        for choice in getattr(chunk, "choices", None) or []:
            delta = choice.delta
            content = getattr(delta, "content", None)
            if content:
                if text_item_id is None:
                    text_item_id = f"msg_{uuid.uuid4().hex}"
                    yield _sse(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": output_index,
                            "item": {
                                "type": "message",
                                "id": text_item_id,
                                "status": "in_progress",
                                "role": "assistant",
                                "content": [],
                            },
                        },
                    )
                text_parts.append(content)
                yield _sse(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": text_item_id,
                        "output_index": output_index,
                        "delta": content,
                    },
                )

            for tc in getattr(delta, "tool_calls", None) or []:
                tc_index = getattr(tc, "index", 0)
                if tc_index not in tool_blocks:
                    if text_item_id is not None:
                        yield _sse(
                            "response.output_item.done",
                            {
                                "type": "response.output_item.done",
                                "output_index": output_index,
                                "item": {
                                    "type": "message",
                                    "id": text_item_id,
                                    "status": "completed",
                                    "role": "assistant",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": "".join(text_parts),
                                            "annotations": [],
                                        }
                                    ],
                                },
                            },
                        )
                        output_index += 1
                        text_item_id = None
                        text_parts = []
                    fc_id = f"fc_{uuid.uuid4().hex}"
                    tool_blocks[tc_index] = {
                        "output_index": output_index,
                        "id": fc_id,
                        "call_id": getattr(tc, "id", "") or "",
                        "name": getattr(tc.function, "name", "") or "",
                        "args_parts": [],
                    }
                    yield _sse(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": output_index,
                            "item": {
                                "type": "function_call",
                                "id": fc_id,
                                "call_id": tool_blocks[tc_index]["call_id"],
                                "name": tool_blocks[tc_index]["name"],
                                "arguments": "",
                                "status": "in_progress",
                            },
                        },
                    )
                    output_index += 1
                fn = getattr(tc, "function", None)
                arguments = getattr(fn, "arguments", None) if fn else None
                if arguments:
                    tool_blocks[tc_index]["args_parts"].append(arguments)

    if text_item_id is not None:
        # No tool call ever followed this text item, so output_index was
        # never advanced past it (advancing only happens on the
        # text->tool_call transition above) -- it's still the right index.
        yield _sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": {
                    "type": "message",
                    "id": text_item_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "".join(text_parts), "annotations": []}
                    ],
                },
            },
        )

    for block in tool_blocks.values():
        yield _sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": block["output_index"],
                "item": {
                    "type": "function_call",
                    "id": block["id"],
                    "call_id": block["call_id"],
                    "name": block["name"],
                    "arguments": "".join(block["args_parts"]),
                    "status": "completed",
                },
            },
        )

    yield _sse(
        "response.completed",
        {
            "type": "response.completed",
            "response": {
                "id": request_id,
                "object": "response",
                "status": "completed",
                "model": model,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            },
        },
    )


__all__ = [
    "anthropic_messages_to_openai",
    "anthropic_tools_to_openai",
    "openai_response_to_anthropic_message",
    "openai_chunks_to_anthropic_sse",
    "responses_request_to_openai",
    "responses_tools_to_openai",
    "openai_response_to_responses_api",
    "openai_chunks_to_responses_sse",
]
