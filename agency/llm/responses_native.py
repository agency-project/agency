"""Lossless native Responses requests stored as patchable Agency blocks.

Only the Responses backend uses this representation. Cross-provider adapters
continue to use the normalized format. Every input item, tool and request option
participates in the normal message store and its patch overlay.
"""

from __future__ import annotations

import copy
import json
from .message_store import with_tool_schema

NATIVE_REQUEST = "openai_responses_native_request"


class NativeRequest(dict):
    """One native CLI request; the CLI owns its retry policy."""


class NativeResult:
    def __init__(self, value):
        self.value = value


class NativeStream:
    """Raw SDK HTTP stream; JSON types and aliased keys remain untouched."""

    def __init__(self, response, owner):
        self.response = response
        self.owner = owner

    def __iter__(self):
        data = []
        for line in self.response.iter_lines():
            if not line:
                if data:
                    value = "\n".join(data)
                    if value != "[DONE]":
                        yield json.loads(value)
                    data = []
            elif line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))
        if data and "\n".join(data) != "[DONE]":
            yield json.loads("\n".join(data))

    def close(self):
        self.owner.__exit__(None, None, None)


def native_result(body, events=None):
    from .openai_responses import _output_items_to_blocks

    raw_usage = body.get("usage") or {}
    usage = {
        "prompt_tokens": raw_usage.get("input_tokens", 0),
        "completion_tokens": raw_usage.get("output_tokens", 0),
        "total_tokens": raw_usage.get("total_tokens", 0),
    }
    for field, details, key in (
        ("cached_tokens", "input_tokens_details", "cached_tokens"),
        ("reasoning_tokens", "output_tokens_details", "reasoning_tokens"),
    ):
        value = (raw_usage.get(details) or {}).get(key)
        if value is not None:
            usage[field] = value
    reason = body.get("status")
    if reason == "completed":
        reason = (
            "tool_calls"
            if any(
                i.get("type") in {"function_call", "custom_tool_call"}
                for i in body.get("output", [])
            )
            else "stop"
        )
    elif reason == "incomplete":
        reason = (body.get("incomplete_details") or {}).get("reason", "length")
    blocks = _output_items_to_blocks(body.get("output"))
    blocks.append(
        {
            "type": "metadata",
            "index": len(blocks),
            "data": {
                "native_responses": {"events": events} if events is not None else {"response": body}
            },
            "usage": usage,
            "stop_reason": reason,
        }
    )
    return {
        "message": {"role": "assistant", "blocks": blocks},
        "usage": usage,
        "stop_reason": reason,
    }


def context_from_request(body, headers=None, query_params=None):
    messages = []
    for location in ("instructions", "input"):
        if location not in body:
            continue
        value = body[location]
        if isinstance(value, str):
            messages.append(
                {
                    "role": "system" if location == "instructions" else "user",
                    "responses_location": location,
                    "responses_string": True,
                    "blocks": [{"type": "text", "index": 0, "text": value}],
                }
            )
        elif isinstance(value, list):
            for item in value:
                # Native item boundaries and IDs are protocol state. Keeping the
                # full item in a block also preserves future/unknown item types.
                messages.append(
                    {
                        "role": item.get("role", "assistant"),
                        "responses_location": location,
                        "blocks": [
                            {
                                "type": "openai_responses_item",
                                "index": 0,
                                "data": copy.deepcopy(item),
                            }
                        ],
                    }
                )
    native = {
        "options": {
            k: copy.deepcopy(v)
            for k, v in body.items()
            if k not in {"tools", "tool_choice", "stream"}
            and not (k in {"instructions", "input"} and isinstance(v, (str, list)))
        },
        "present": [
            k
            for k in ("instructions", "input", "tools", "tool_choice")
            if k in body
            and (k not in {"instructions", "input"} or isinstance(body[k], (str, list)))
        ],
        "strings": [k for k in ("instructions", "input") if isinstance(body.get(k), str)],
        # No credentials, transport/session identifiers or SDK telemetry.
        "headers": {
            k.lower(): v
            for k, v in (headers or {}).items()
            if k.lower() in {"openai-beta", "x-openai-api-version", "originator", "user-agent"}
            or k.lower().startswith("x-codex-")
        },
        "query": dict(query_params or {}),
    }
    return {
        "messages": with_tool_schema(
            messages,
            copy.deepcopy(body.get("tools")),
            copy.deepcopy(body.get("tool_choice")),
            {NATIVE_REQUEST: native},
        )
    }


def request_from_agency(messages, tools, tool_choice, native, config):
    body = copy.deepcopy(native["options"])
    body.setdefault("model", config.model or "")
    locations = {key: [] for key in native["present"] if key in {"input", "instructions"}}
    strings = set(native.get("strings", []))
    for message in messages:
        location = message.get("responses_location", "input")
        values = locations.setdefault(location, [])
        if message.get("responses_string"):
            strings.add(location)
        for block in message.get("blocks", []):
            if block["type"] == "openai_responses_item":
                values.append(copy.deepcopy(block["data"]))
            elif block["type"] == "text":
                values.append(
                    block["text"]
                    if location in strings
                    else {"role": message["role"], "content": block["text"]}
                )
    for location, values in locations.items():
        body[location] = "".join(values) if location in strings else values
    if tools is not None or "tools" in native["present"]:
        body["tools"] = copy.deepcopy(tools or [])
    if tool_choice is not None or "tool_choice" in native["present"]:
        body["tool_choice"] = copy.deepcopy(tool_choice)
    # Explicit configuration remains an intentional override. Unset defaults
    # never change the CLI's own context, storage or generation settings.
    for field in ("temperature", "top_p"):
        value = getattr(config, field)
        if value is not None:
            body[field] = value
    cap = (
        config.max_completion_tokens
        if config.max_completion_tokens is not None
        else config.max_tokens
    )
    if cap is not None:
        body["max_output_tokens"] = cap
    for field, option in (("reasoning_effort", "effort"), ("reasoning_summary", "summary")):
        value = getattr(config, field)
        if value is not None:
            body.setdefault("reasoning", {})[option] = value
    body.update(copy.deepcopy(config.extra_body or {}))
    kwargs = NativeRequest(model=body.pop("model"), extra_body=body)
    if native.get("headers"):
        kwargs["extra_headers"] = copy.deepcopy(native["headers"])
    if native.get("query"):
        kwargs["extra_query"] = copy.deepcopy(native["query"])
    return kwargs


def response_fields(message):
    for block in message.get("blocks", []):
        if block.get("type") != "metadata":
            continue
        data = block.get("data")
        for fragment in data if isinstance(data, list) else [data]:
            if isinstance(fragment, dict) and "native_responses" in fragment:
                return fragment["native_responses"]
    return None
