"""Preserve Claude's native request while keeping its blocks patchable."""

from __future__ import annotations

import copy
import json

from .message_store import with_tool_schema


NATIVE_REQUEST = "anthropic_native_request"
_MESSAGE = "anthropic_message"
_FIELDS = "anthropic_fields"


def _to_block(native: dict, index: int) -> dict:
    fields = copy.deepcopy(native)
    kind = fields.pop("type")
    block = {"type": kind, "index": index}
    if kind == "text":
        block["text"] = fields.pop("text")
    elif kind == "thinking":
        block["text"] = fields.pop("thinking")
        block["signature"] = fields.pop("signature")
    elif kind == "tool_use":
        block.update(id=fields.pop("id"), name=fields.pop("name"))
        block["arguments"] = json.dumps(fields.pop("input"))
    elif kind == "tool_result":
        block["tool_call_id"] = fields.pop("tool_use_id")
        if "content" in fields:
            content = fields.pop("content")
            if isinstance(content, str):
                block["text"] = content
            else:
                # Keep structured results as structured data, rather than a
                # second text copy that could silently win over a history patch.
                block["raw_content"] = content
        if "is_error" in fields:
            block["is_error"] = fields.pop("is_error")
    else:
        return {"type": "anthropic_" + kind, "index": index, "data": copy.deepcopy(native)}
    if fields:
        block[_FIELDS] = fields
    return block


def _from_block(block: dict) -> dict:
    kind = block["type"]
    fields = copy.deepcopy(block.get(_FIELDS, {}))
    fields["type"] = kind
    if kind == "text":
        fields["text"] = block["text"]
        if "citations" in block:
            fields["citations"] = copy.deepcopy(block["citations"])
    elif kind == "thinking":
        fields.update(thinking=block["text"], signature=block["signature"])
    elif kind == "tool_use":
        fields.update(id=block["id"], name=block["name"], input=json.loads(block["arguments"]))
    elif kind == "tool_result":
        fields.update(tool_use_id=block["tool_call_id"])
        if "raw_content" in block or "text" in block:
            fields["content"] = copy.deepcopy(block.get("raw_content", block.get("text")))
        if "is_error" in block:
            fields["is_error"] = block["is_error"]
    elif kind.startswith("anthropic_"):
        fields = copy.deepcopy(block["data"])
        fields["type"] = kind[len("anthropic_") :]
    else:
        raise ValueError(f"Cannot serialize {kind!r} as an Anthropic content block")
    return fields


def request_to_agency(body: dict, headers=None, query_params=None) -> dict:
    messages = []

    def add_message(role, content, location, fields):
        marker = {"location": location, "string": isinstance(content, str), "fields": fields}
        if isinstance(content, str):
            blocks = [{"type": "text", "index": 0, "text": content}]
        else:
            blocks = [_to_block(b, i) for i, b in enumerate(content)]
        # Keep tool results in the public tool-role representation. The source
        # location joins these pieces back into their original user message.
        pending = []
        for block in blocks:
            if role == "user" and block["type"] == "tool_result":
                if pending:
                    messages.append({"role": role, "blocks": pending, _MESSAGE: marker})
                    pending = []
                block["index"] = 0
                messages.append({"role": "tool", "blocks": [block], _MESSAGE: marker})
            else:
                block["index"] = len(pending)
                pending.append(block)
        if pending or not blocks:
            messages.append({"role": role, "blocks": pending, _MESSAGE: marker})

    if "system" in body:
        add_message("system", body["system"], "system", {})
    for index, message in enumerate(body.get("messages", [])):
        fields = {k: copy.deepcopy(v) for k, v in message.items() if k not in {"role", "content"}}
        add_message(message["role"], message["content"], index, fields)
    tools = None
    if "tools" in body:
        tools = []
        for tool in body["tools"]:
            # Native built-in tools need not have a function schema.
            if "input_schema" not in tool:
                tools.append({"anthropic_native_tool": copy.deepcopy(tool)})
                continue
            fn = {"name": tool["name"], "parameters": copy.deepcopy(tool["input_schema"])}
            if "description" in tool:
                fn["description"] = tool["description"]
            fields = {
                k: copy.deepcopy(v)
                for k, v in tool.items()
                if k not in {"name", "description", "input_schema"}
            }
            tools.append({"type": "function", "function": fn, _FIELDS: fields})
    options = {
        k: copy.deepcopy(v)
        for k, v in body.items()
        if k not in {"messages", "system", "tools", "tool_choice", "stream"}
    }
    safe_headers = {
        k.lower(): v
        for k, v in (headers or {}).items()
        if k.lower() in {"anthropic-beta", "anthropic-version"}
    }
    query = {"beta": query_params["beta"]} if query_params and "beta" in query_params else {}
    native = {
        "options": options,
        "headers": safe_headers,
        "query": query,
        "tools_present": "tools" in body,
    }
    return {
        "messages": with_tool_schema(
            messages, tools, body.get("tool_choice"), {NATIVE_REQUEST: native}
        )
    }


def request_from_agency(messages, tools, tool_choice, native, config) -> dict:
    body = copy.deepcopy(native["options"])
    body["messages"] = []
    previous_location = None
    for message in messages:
        marker = message.get(_MESSAGE, {})
        content = [_from_block(block) for block in message.get("blocks", [])]
        if marker.get("string") and len(content) == 1 and set(content[0]) == {"type", "text"}:
            content = content[0]["text"]
        location = marker.get("location")
        if location == "system":
            body["system"] = content
            previous_location = None
            continue
        role = "user" if message["role"] == "tool" else message["role"]
        if marker and location == previous_location:
            body["messages"][-1]["content"].extend(content)
        else:
            body["messages"].append(
                {**copy.deepcopy(marker.get("fields", {})), "role": role, "content": content}
            )
        previous_location = location if marker else None
    if tools is not None or native.get("tools_present"):
        body["tools"] = []
        for tool in tools or []:
            if "anthropic_native_tool" in tool:
                body["tools"].append(copy.deepcopy(tool["anthropic_native_tool"]))
                continue
            fn = tool["function"]
            rendered = {
                **copy.deepcopy(tool.get(_FIELDS, {})),
                "name": fn["name"],
                "input_schema": copy.deepcopy(fn["parameters"]),
            }
            if "description" in fn:
                rendered["description"] = fn["description"]
            body["tools"].append(rendered)
    if tool_choice is not None:
        body["tool_choice"] = copy.deepcopy(tool_choice)
    if config.model:
        # The configured model selects the CLI's initial model. The CLI may
        # then make a documented per-request fallback decision; remapping it
        # here would resend a refused request to the original model.
        body.setdefault("model", config.model)
    cap = (
        config.max_completion_tokens
        if config.max_completion_tokens is not None
        else config.max_tokens
    )
    if cap is not None:
        body["max_tokens"] = cap
    body.setdefault("max_tokens", config.default_max_tokens)
    for field in ("temperature", "top_p", "top_k"):
        value = getattr(config, field)
        if value is not None:
            body[field] = value
    if config.stop is not None:
        body["stop_sequences"] = [config.stop] if isinstance(config.stop, str) else config.stop
    if config.reasoning_effort is not None:
        body.setdefault("output_config", {})["effort"] = config.reasoning_effort
    body.update(copy.deepcopy(config.extra_body or {}))
    # SDK versions differ in which beta fields they accept as named arguments.
    # extra_body preserves those wire fields without depending on SDK typing.
    kwargs = {key: body.pop(key) for key in ("model", "messages", "max_tokens")}
    if body:
        kwargs["extra_body"] = body
    if native.get("headers"):
        kwargs["extra_headers"] = copy.deepcopy(native["headers"])
    if native.get("query"):
        kwargs["extra_query"] = copy.deepcopy(native["query"])
    return kwargs
