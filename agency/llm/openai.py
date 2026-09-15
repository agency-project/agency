"""OpenAI (and, via `.vllm`, any other OpenAI-compatible endpoint) backend."""

from __future__ import annotations
import httpx
import openai

from .agllm import agllm

_HANDLED_MESSAGE_FIELDS = {"role", "content", "tool_calls", "reasoning_content", "function_call"}
_HANDLED_DELTA_FIELDS = {"role", "content", "reasoning_content", "tool_calls", "function_call"}
_CHATCOMPLETIONS_TYPE_PREFIX = "openai_chatcompletions_"
_METADATA_BLOCK_INDEX = 2**31 - 1  # reserved index, sorts after any real content-block index

_CHATCOMPLETIONS_TOOL_CHOICE_VALUES = {"auto", "required", "none"}


def _serialize_sdk_object(obj):
    dump = getattr(obj, "model_dump", None)
    return dump() if dump is not None else obj


def _chatcompletions_native_block_type(native_type: str) -> str:
    return f"{_CHATCOMPLETIONS_TYPE_PREFIX}{native_type}"


def _is_chatcompletions_tool_choice(tool_choice) -> bool:
    if isinstance(tool_choice, str) and tool_choice in _CHATCOMPLETIONS_TOOL_CHOICE_VALUES:
        return True
    return (
        isinstance(tool_choice, dict)
        and tool_choice.get("type") == "function"
        and isinstance(tool_choice.get("function"), dict)
    )


class _OpenAICompatibleBackend(agllm):
    """Default backend: OpenAI, vLLM, litellm, or any other OpenAI-compatible
    endpoint. base_url is always required (see _validate_config) -- the
    openai SDK's own default (api.openai.com) is exactly the kind of silent
    wrong-endpoint footgun this backend exists to avoid: a real OpenAI key
    still needs base_url='https://api.openai.com/v1' spelled out."""

    def _validate_config(self) -> None:
        super()._validate_config()
        if not self.agconfig.llm.base_url:
            raise ValueError(
                f"agconfig.llm with provider={self.agconfig.llm.provider!r} "
                "requires an explicit base_url."
            )

    def make_client(self, timeout: httpx.Timeout) -> openai.OpenAI:
        return openai.OpenAI(
            api_key=self.agconfig.llm.api_key or "EMPTY",
            base_url=self.agconfig.llm.base_url,
            timeout=timeout,
        )

    def tokenize_url(self) -> "str | None":
        base_url: str = self.agconfig.llm.base_url or ""
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        return root or None

    def _format_context_agency_to_backend(self, request: dict) -> dict:
        kwargs = self.build_kwargs(request["messages"], request.get("tools"))
        tool_choice = request.get("tool_choice")
        # Responses-style clients such as Codex may send tool_choice="auto"
        # even after their hosted tools were filtered from a Chat Completions
        # request. OpenAI rejects tool_choice when no tools are present.
        if (
            kwargs.get("tools")
            and tool_choice is not None
            and _is_chatcompletions_tool_choice(tool_choice)
        ):
            kwargs["tool_choice"] = tool_choice
        return kwargs

    def _call_backend(self, backend_request: dict):
        client = self.make_client(self._client_timeout())
        try:
            return client.chat.completions.create(**backend_request)
        finally:
            client.close()

    def _format_context_backend_to_agency(self, raw_result) -> dict:
        choice = (raw_result.choices or [None])[0]
        usage = _serialize_openai_usage(getattr(raw_result, "usage", None))
        if choice is None:
            return {
                "message": {
                    "role": "assistant",
                    "blocks": [
                        {
                            "type": "metadata",
                            "index": 0,
                            "usage": usage,
                            "stop_reason": None,
                            "data": _serialize_sdk_object(raw_result),
                        }
                    ],
                },
                "usage": usage,
                "stop_reason": None,
            }
        message = choice.message
        blocks = []
        content = getattr(message, "content", None)
        if content:
            blocks.append({"type": "text", "index": 0, "text": content})
        for tc in getattr(message, "tool_calls", None) or []:
            blocks.append(
                {
                    "type": "tool_use",
                    "index": len(blocks),
                    "id": getattr(tc, "id", "") or "",
                    "name": getattr(tc.function, "name", "") or "",
                    "arguments": getattr(tc.function, "arguments", "") or "",
                }
            )
        function_call = getattr(message, "function_call", None)
        if function_call is not None:
            blocks.append(
                {
                    "type": "tool_use",
                    "index": len(blocks),
                    "id": "",
                    "name": getattr(function_call, "name", "") or "",
                    "arguments": getattr(function_call, "arguments", "") or "",
                }
            )
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            blocks.append({"type": "thinking", "index": len(blocks), "text": reasoning})
        message_dump = _serialize_sdk_object(message)
        if isinstance(message_dump, dict):
            for field, value in message_dump.items():
                if field in _HANDLED_MESSAGE_FIELDS or not value:
                    continue
                blocks.append(
                    {
                        "type": _chatcompletions_native_block_type(field),
                        "index": len(blocks),
                        "data": value,
                    }
                )
        stop_reason = getattr(choice, "finish_reason", None)
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
        # Without stream_options.include_usage, the OpenAI streaming API
        # never sends a usage field on any chunk -- chunk_usage below would
        # be None for the entire stream, and every metadata block's usage
        # would be silently empty.
        raw_stream = client.chat.completions.create(
            **{**backend_request, "stream": True, "stream_options": {"include_usage": True}}
        )
        return raw_stream, client

    def _format_stream_to_agency(self, raw_stream):
        unknown_field_index: "dict[str, int]" = {}
        next_unknown_index = -3
        for chunk in raw_stream:
            chunk_usage = getattr(chunk, "usage", None)
            choice = (chunk.choices or [None])[0]
            finish_reason = None
            if choice is not None:
                delta = choice.delta
                content = getattr(delta, "content", None)
                if content:
                    yield {
                        "type": "block_delta",
                        "index": -1,
                        "block_type": "text",
                        "text": content,
                    }
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield {
                        "type": "block_delta",
                        "index": -2,
                        "block_type": "thinking",
                        "text": reasoning,
                    }
                for tc in getattr(delta, "tool_calls", None) or []:
                    fn = getattr(tc, "function", None)
                    yield {
                        "type": "block_delta",
                        "index": getattr(tc, "index", 0),
                        "block_type": "tool_use",
                        "id": getattr(tc, "id", None) or "",
                        "name": (getattr(fn, "name", None) or "") if fn else "",
                        "arguments": (getattr(fn, "arguments", None) or "") if fn else "",
                    }
                delta_dump = _serialize_sdk_object(delta)
                if isinstance(delta_dump, dict):
                    for field, value in delta_dump.items():
                        if field in _HANDLED_DELTA_FIELDS or not value:
                            continue
                        if field not in unknown_field_index:
                            unknown_field_index[field] = next_unknown_index
                            next_unknown_index -= 1
                        yield {
                            "type": "block_delta",
                            "index": unknown_field_index[field],
                            "block_type": _chatcompletions_native_block_type(field),
                            "data": value,
                        }
                finish_reason = getattr(choice, "finish_reason", None)
            if chunk_usage is not None or finish_reason is not None:
                usage = _serialize_openai_usage(chunk_usage) if chunk_usage is not None else None
                # The metadata block is a plain block_delta -- like any other
                # unknown native field, it flows through the host server's
                # generic block accumulator with no dedicated handling
                # required there. All translation of what "usage"/
                # "stop_reason" mean stays in this backend, not the host
                # server.
                yield {
                    "type": "block_delta",
                    "index": _METADATA_BLOCK_INDEX,
                    "block_type": "metadata",
                    "data": {
                        "usage": usage,
                        "stop_reason": finish_reason,
                        "raw_chunk": _serialize_sdk_object(chunk),
                    },
                }
                yield {
                    "type": "usage",
                    "usage": usage,
                    "stop_reason": finish_reason,
                }


def _serialize_openai_usage(usage) -> "dict | None":
    if usage is None:
        return None
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": getattr(usage, "total_tokens", None) or (prompt_tokens + completion_tokens),
    }
