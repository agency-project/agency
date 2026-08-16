"""Reverse-direction wire adapters for `agproxy_llm`: harness-native request
formats (Anthropic Messages API, OpenAI Responses API) <-> the
`.chat.completions.create(**kwargs)` shape every `agllm_backend.make_client()`
exposes uniformly (see `agllm_backends/base.py`).

This is deliberately the *opposite* direction from `agllm_backends/anthropic.py`'s
`_openai_messages_to_anthropic`/`_anthropic_stream_to_openai_chunks` (those
convert agency's own OpenAI-shaped request into a real outbound Anthropic API
call). Here, an off-the-shelf harness (Claude Code, Codex) is the *caller*,
speaking its own native wire format to `agproxy_llm`; this module reshapes
that request into OpenAI chat-completions kwargs, invokes the SAME uniform
`client.chat.completions.create()` every other backend uses, then reshapes
the (possibly streaming) response back into the harness's expected format.

Fidelity is bounded on purpose: this is a format *translation*, not a proxy
to the harness's real provider. The Anthropic adapter retains its documented
best-effort behavior for provider-only metadata. The Responses adapter is
stricter because silently dropping a Codex tool or input block changes what
the agent can do: unsupported semantic content raises an explicit translation
error instead. See docs/Design_harness_integration.md's "Design Tensions"
section for the limits of `gateway_mode="translate"`.
"""

from __future__ import annotations

import json
import time
import uuid
import warnings


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
    # `_openai_messages_to_anthropic` (agllm_backends/anthropic.py:100-144)
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
    shape `agllm_backends/anthropic.py`'s own Anthropic-backed shim yields --
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


class UnsupportedResponsesRequest(ValueError):
    """A Responses request cannot be represented by Chat Completions safely."""


def responses_tools_to_openai(tools, *, warning_handler=None) -> "list[dict] | None":
    """Responses API tool defs are flat (`{"type":"function","name":...,
    "description":...,"parameters":...}`) -- chat-completions nests them
    under a `function` key.

    Only function tools have a lossless Chat Completions representation.
    Codex may also advertise custom or namespace tools depending on its model
    metadata and enabled features. Chat Completions cannot represent those
    tools. Keep the function-tool subset so Codex's shell remains usable, but
    always surface each omission through the provided per-request warning
    handler (or a Python warning for direct callers).
    """
    if not tools:
        return None
    converted = []
    for index, t in enumerate(tools):
        if not isinstance(t, dict):
            raise UnsupportedResponsesRequest(f"Responses tool at index {index} must be an object")
        tool_type = t.get("type")
        if tool_type != "function":
            message = (
                f"Responses tool type {tool_type!r} at index {index} cannot be translated "
                "to Chat Completions and was omitted; Codex may use only the remaining "
                "function tools on this turn"
            )
            if warning_handler is None:
                warnings.warn(message, RuntimeWarning, stacklevel=2)
            else:
                warning_handler(message)
            continue
        if t.get("defer_loading"):
            raise UnsupportedResponsesRequest(
                f"Responses function tool {t.get('name', '')!r} uses defer_loading, which "
                "Chat Completions cannot represent"
            )

        parameters = t.get("parameters")
        if parameters is None:
            parameters = {"type": "object", "properties": {}}
        function = {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": parameters,
        }
        # `strict` exists in both APIs. Preserve even False/None rather than
        # relying on a backend default that may differ from Codex's request.
        if "strict" in t:
            function["strict"] = t["strict"]
        converted.append(
            {
                "type": "function",
                "function": function,
            }
        )
    return converted or None


def _responses_content_to_text(content, *, location: str) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for index, block in enumerate(content):
            if not isinstance(block, dict):
                raise UnsupportedResponsesRequest(
                    f"Responses content block {location}[{index}] must be an object"
                )
            btype = block.get("type")
            if btype in ("input_text", "output_text", "text"):
                text = block.get("text", "")
                if not isinstance(text, str):
                    raise UnsupportedResponsesRequest(
                        f"Responses text block {location}[{index}] must contain string text"
                    )
                parts.append(text)
                continue
            raise UnsupportedResponsesRequest(
                f"Responses content type {btype!r} at {location}[{index}] cannot be "
                "translated safely to Agency's Chat Completions backends"
            )
        return "".join(parts)
    raise UnsupportedResponsesRequest(
        f"Responses content at {location} must be a string or a list of text blocks"
    )


def _responses_tool_choice_to_openai(tool_choice):
    if tool_choice is None:
        return None
    if tool_choice in ("auto", "none", "required"):
        return tool_choice
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = tool_choice.get("name")
        if isinstance(name, str) and name:
            return {"type": "function", "function": {"name": name}}
    raise UnsupportedResponsesRequest(
        f"Responses tool_choice {tool_choice!r} cannot be translated to Chat Completions"
    )


def _apply_responses_reasoning_config(body: dict, kwargs: dict) -> None:
    reasoning = body.get("reasoning")
    if reasoning is None:
        return
    if not isinstance(reasoning, dict):
        raise UnsupportedResponsesRequest("Responses reasoning must be an object or null")

    unsupported = {key for key, value in reasoning.items() if key != "effort" and value is not None}
    if unsupported:
        fields = ", ".join(sorted(unsupported))
        raise UnsupportedResponsesRequest(
            f"Responses reasoning fields cannot be translated to Chat Completions: {fields}"
        )
    if reasoning.get("effort") is not None:
        kwargs["reasoning_effort"] = reasoning["effort"]


def _apply_responses_text_config(body: dict, kwargs: dict) -> None:
    text_config = body.get("text")
    if text_config is None:
        return
    if not isinstance(text_config, dict):
        raise UnsupportedResponsesRequest("Responses text configuration must be an object")

    response_format = text_config.get("format")
    if response_format not in (None, {"type": "text"}):
        raise UnsupportedResponsesRequest(
            "Responses text.format is not supported by the Agency translation proxy; "
            "use the shared Agency structured-output path instead"
        )
    if text_config.get("verbosity") is not None:
        kwargs["verbosity"] = text_config["verbosity"]


def responses_request_to_openai(body: dict, *, warning_handler=None) -> dict:
    """OpenAI `POST /v1/responses` request body -> `client.chat.completions.
    create(**kwargs)` kwargs.

    This covers the function-tool subset emitted by Codex 0.140.0. Unsupported
    tool definitions are warned and omitted while unsupported prompt content
    fails explicitly instead of being removed from the model-visible request.
    """
    openai_messages: list[dict] = []

    instructions = body.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise UnsupportedResponsesRequest("Responses instructions must be a string or null")
    if instructions:
        openai_messages.append({"role": "system", "content": instructions})

    raw_input = body.get("input")
    if isinstance(raw_input, str):
        openai_messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        for index, item in enumerate(raw_input):
            if not isinstance(item, dict):
                raise UnsupportedResponsesRequest(
                    f"Responses input item at index {index} must be an object"
                )
            itype = item.get("type")
            if itype == "function_call":
                tool_call = {
                    "id": item.get("call_id", ""),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                }
                # A Responses turn can emit several parallel function_call
                # output items. Chat Completions represents all of them on
                # one assistant message; separate consecutive assistant
                # messages also violate Anthropic/Bedrock role alternation.
                if openai_messages and openai_messages[-1].get("role") == "assistant":
                    openai_messages[-1].setdefault("tool_calls", []).append(tool_call)
                else:
                    openai_messages.append(
                        {"role": "assistant", "content": None, "tool_calls": [tool_call]}
                    )
            elif itype == "function_call_output":
                openai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": _responses_content_to_text(
                            item.get("output"), location=f"input[{index}].output"
                        ),
                    }
                )
            elif itype in (None, "message"):
                role = item.get("role")
                if role not in ("user", "assistant", "system", "developer"):
                    raise UnsupportedResponsesRequest(
                        f"Responses message role {role!r} at input[{index}] is unsupported"
                    )
                # Agency's Anthropic-compatible backend understands `system`
                # but not Chat Completions' newer `developer` role. Mapping it
                # to system preserves its instruction priority across every
                # configured Agency provider rather than silently losing it.
                if role == "developer":
                    role = "system"
                openai_messages.append(
                    {
                        "role": role,
                        "content": _responses_content_to_text(
                            item.get("content"), location=f"input[{index}].content"
                        ),
                    }
                )
            else:
                raise UnsupportedResponsesRequest(
                    f"Responses input item type {itype!r} at index {index} cannot be "
                    "translated to Chat Completions"
                )
    else:
        raise UnsupportedResponsesRequest("Responses input must be a string or a list")

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
    if "parallel_tool_calls" in body:
        kwargs["parallel_tool_calls"] = body["parallel_tool_calls"]
    if "store" in body:
        kwargs["store"] = body["store"]
    if body.get("prompt_cache_key") is not None:
        kwargs["prompt_cache_key"] = body["prompt_cache_key"]
    _apply_responses_reasoning_config(body, kwargs)
    _apply_responses_text_config(body, kwargs)
    tools = responses_tools_to_openai(body.get("tools"), warning_handler=warning_handler)
    if tools:
        kwargs["tools"] = tools
    tool_choice = _responses_tool_choice_to_openai(body.get("tool_choice"))
    if tools and tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    elif not tools and tool_choice not in (None, "auto", "none"):
        raise UnsupportedResponsesRequest(
            "Responses tool_choice requires a function tool, but no translatable function "
            "tools remain"
        )
    return kwargs


def _openai_usage_detail(usage, group: str, field: str) -> "int | None":
    details = getattr(usage, group, None)
    if details is None:
        return None
    if isinstance(details, dict):
        return details.get(field)
    return getattr(details, field, None)


def _openai_usage_to_responses(usage) -> dict:
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    result = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }

    cached_tokens = _openai_usage_detail(usage, "prompt_tokens_details", "cached_tokens")
    if cached_tokens is not None:
        result["input_tokens_details"] = {"cached_tokens": cached_tokens}
    reasoning_tokens = _openai_usage_detail(usage, "completion_tokens_details", "reasoning_tokens")
    if reasoning_tokens is not None:
        result["output_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return result


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

    usage = _openai_usage_to_responses(getattr(resp, "usage", None))

    return {
        "id": request_id,
        "object": "response",
        "created_at": time.time(),
        "status": "completed",
        "model": model,
        "output": output,
        "usage": usage,
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
    cached_input_tokens = None
    reasoning_output_tokens = None

    for chunk in chunks:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "prompt_tokens", 0) or input_tokens
            output_tokens = getattr(usage, "completion_tokens", 0) or output_tokens
            cached_tokens = _openai_usage_detail(usage, "prompt_tokens_details", "cached_tokens")
            if cached_tokens is not None:
                cached_input_tokens = cached_tokens
            reasoning_tokens = _openai_usage_detail(
                usage, "completion_tokens_details", "reasoning_tokens"
            )
            if reasoning_tokens is not None:
                reasoning_output_tokens = reasoning_tokens

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

    response_usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if cached_input_tokens is not None:
        response_usage["input_tokens_details"] = {"cached_tokens": cached_input_tokens}
    if reasoning_output_tokens is not None:
        response_usage["output_tokens_details"] = {"reasoning_tokens": reasoning_output_tokens}

    yield _sse(
        "response.completed",
        {
            "type": "response.completed",
            "response": {
                "id": request_id,
                "object": "response",
                "status": "completed",
                "model": model,
                "usage": response_usage,
            },
        },
    )


__all__ = [
    "UnsupportedResponsesRequest",
    "anthropic_messages_to_openai",
    "anthropic_tools_to_openai",
    "openai_response_to_anthropic_message",
    "openai_chunks_to_anthropic_sse",
    "responses_request_to_openai",
    "responses_tools_to_openai",
    "openai_response_to_responses_api",
    "openai_chunks_to_responses_sse",
]
