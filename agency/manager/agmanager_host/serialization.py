"""Duck-typed chat-completion response/chunk serialization for
`agmanager_host`'s LLM dispatch route.

Deliberately reimplemented here rather than imported from the old
`agllm_terminus.py` (which this design is eventually meant to replace, see
`agmanager_host.py`'s module docstring): some backends' response/chunk
objects are lightweight `__slots__` compatibility shims, not real SDK
pydantic models, so `.model_dump()` can't be assumed -- see
`agllm_terminus.py`'s identical-purpose helpers for the original discovery
of this gap."""

from __future__ import annotations


def serialize_usage(usage) -> "dict | None":
    if usage is None:
        return None
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": getattr(usage, "total_tokens", None) or (prompt_tokens + completion_tokens),
    }


def serialize_tool_calls(tool_calls) -> "list | None":
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


def serialize_result(result) -> dict:
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
                    "tool_calls": serialize_tool_calls(getattr(message, "tool_calls", None)),
                },
            }
        )
    return {
        "id": getattr(result, "id", "") or "",
        "object": "chat.completion",
        "created": getattr(result, "created", 0) or 0,
        "model": getattr(result, "model", "") or "",
        "choices": choices,
        "usage": serialize_usage(getattr(result, "usage", None)),
    }


def serialize_chunk(chunk) -> dict:
    choices = []
    for i, choice in enumerate(chunk.choices or []):
        delta = choice.delta
        choices.append(
            {
                "index": getattr(choice, "index", i),
                "finish_reason": getattr(choice, "finish_reason", None),
                "delta": {
                    "content": getattr(delta, "content", None),
                    "tool_calls": serialize_tool_calls(getattr(delta, "tool_calls", None)),
                },
            }
        )
    return {
        "id": getattr(chunk, "id", "") or "",
        "object": "chat.completion.chunk",
        "created": getattr(chunk, "created", 0) or 0,
        "model": getattr(chunk, "model", "") or "",
        "choices": choices,
        "usage": serialize_usage(getattr(chunk, "usage", None)),
    }


__all__ = ["serialize_usage", "serialize_tool_calls", "serialize_result", "serialize_chunk"]
