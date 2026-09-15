"""Amazon Bedrock backend(s), plus the separate "Claude Platform on AWS"
backend (provider='anthropicAWS'/'anthropic_aws') -- grouped here as the two
AWS-hosted flavors of Anthropic access, distinct from Bedrock's actual other
models.

Claude models on Bedrock are only reachable through Bedrock's native
invoke_model API, in the Anthropic Messages API shape (the `anthropic` SDK's
`AnthropicBedrock` client), and only via an inference-profile ID (e.g.
`us.anthropic.claude-sonnet-5`) rather than the bare
`anthropic.claude-sonnet-5` foundation-model ID — the bare ID 400s with "on-
demand throughput isn't supported."

`for_config()` (`.agllm`) picks between `_AnthropicBedrockBackend` and
`_OpenAICompatibleBedrockBackend` based on the model ID
(`_is_anthropic_bedrock_model()`), and routes provider='anthropicAWS' to
`_AnthropicAWSBackend` directly (no model-based branching -- there's no
non-Anthropic equivalent on that product).
"""

from __future__ import annotations
import base64
import json
import os
import re
import httpx
import openai

from .agllm import agllm
from .openai import _OpenAICompatibleBackend
from .anthropic import (
    _AnthropicBackend,
    _ANTHROPIC_BEDROCK_MODEL_RE,
    _anthropic_sdk_timeout,
    _known_anthropic_context_window,
)

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None


def _is_anthropic_bedrock_model(model: str) -> bool:
    return bool(_ANTHROPIC_BEDROCK_MODEL_RE.match(model or ""))


# openai.gpt-5.x on Bedrock 400s on the OpenAI-compatible gateway ("isn't
# supported on this route" on bedrock-mantle) and, like Anthropic models,
# only works through the native Converse API with an inference-profile-
# prefixed ID (e.g. "us.openai.gpt-5.6-terra") -- confirmed directly against
# Bedrock for gpt-5.4/5.5/5.6-luna/5.6-sol/5.6-terra. Update if more Bedrock
# models are confirmed to need this route too.
_BEDROCK_CONVERSE_MODEL_RE = re.compile(r"^(?:(?:us|eu|apac|global)\.)?openai\.gpt-5(?:[.\-]|$)")


def _needs_bedrock_converse(model: str) -> bool:
    return bool(_BEDROCK_CONVERSE_MODEL_RE.match(model or ""))


def _require_bedrock_bearer_token(llm_config) -> str:
    """Amazon Bedrock's one supported auth mechanism: an explicit
    bearer-token api_key (what Bedrock itself calls a "Bedrock API key"),
    from either the llm config or AWS_BEARER_TOKEN_BEDROCK."""
    api_key = llm_config.api_key or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if not api_key:
        raise RuntimeError(
            "Amazon Bedrock requires a bearer-token api_key: set it in the "
            "llm_config's api_key field, or export AWS_BEARER_TOKEN_BEDROCK. "
            "SigV4 key pairs and the AWS default credential chain are not "
            "supported for provider='bedrock'."
        )
    if ":" in api_key:
        raise ValueError(
            "Amazon Bedrock only accepts a bearer-token api_key now, not a "
            "SigV4 'ACCESS_KEY_ID:SECRET_ACCESS_KEY[:SESSION_TOKEN]' pair."
        )
    return api_key


class _OpenAICompatibleBedrockBackend(_OpenAICompatibleBackend):
    """Bedrock models reached through bedrock-runtime's own OpenAI-compatible
    endpoint (its /openai/v1 path) — every Bedrock model except Anthropic's
    own and openai.gpt-5.x (see module docstring above)."""

    def _validate_config(self) -> None:
        agllm._validate_config(self)

    def make_client(self, timeout: httpx.Timeout) -> openai.OpenAI:
        region = self.agconfig.llm.region or "us-east-1"
        api_key = _require_bedrock_bearer_token(self.agconfig.llm)
        runtime_url = f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1"
        return openai.OpenAI(api_key=api_key, base_url=runtime_url, timeout=timeout)

    def tokenize_url(self) -> "str | None":
        return None  # Bedrock has no vLLM-style /tokenize endpoint


class _AnthropicBedrockBackend(_AnthropicBackend):
    """Claude models on Amazon Bedrock — native invoke_model API via the
    anthropic SDK's AnthropicBedrock client (Messages API shape)."""

    def make_client(self, timeout: httpx.Timeout):
        if _anthropic_sdk is None:
            raise RuntimeError(
                "Anthropic models on Bedrock require the 'anthropic' package: pip install anthropic"
            )
        region = self.agconfig.llm.region or "us-east-1"
        api_key = _require_bedrock_bearer_token(self.agconfig.llm)
        return _anthropic_sdk.AnthropicBedrock(
            aws_region=region, api_key=api_key, timeout=_anthropic_sdk_timeout(timeout)
        )

    def list_models(self) -> list:
        return []  # Bedrock's native invoke_model API has no OpenAI-style /v1/models

    def tokenize_url(self) -> "str | None":
        return None

    def known_context_limit(self, model: str) -> "int | None":
        return _known_anthropic_context_window(model)


# ---------------------------------------------------------------------------
# Bedrock native Converse API -- for models that reject Mantle (see
# _needs_bedrock_converse above), starting with openai.gpt-5.x. Bedrock's own
# unified message shape, distinct from both the OpenAI chat.completions shape
# (openai.py) and the Anthropic Messages shape (anthropic.py) -- e.g. a
# ContentBlock is a single-key dict ({"text": ...} | {"toolUse": {...}} |
# {"toolResult": {...}} | {"reasoningContent": {...}}), not a "type"-tagged
# dict.
# ---------------------------------------------------------------------------

_CONVERSE_TYPE_PREFIX = "bedrock_converse_"
_CONVERSE_METADATA_BLOCK_INDEX = (
    2**31 - 1
)  # reserved index, sorts after any real content-block index


def _converse_native_block_type(native_type: str) -> str:
    return f"{_CONVERSE_TYPE_PREFIX}{native_type}"


_BLOB_B64_KEY = "__bedrock_blob_b64__"


def _bytes_to_jsonable(value):
    """Some Converse fields (e.g. reasoningContent.redactedContent) are
    boto3-typed as a raw bytes blob, not JSON-serializable -- and this value
    flows through JSON-lines transport (host_services_client ->
    llm_handler_server) on its way into an agency block. Marked and
    base64-encoded here so it survives that hop; _jsonable_to_bytes() below
    reverses it before replaying the block back to Converse."""
    if isinstance(value, (bytes, bytearray)):
        return {_BLOB_B64_KEY: base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, dict):
        return {k: _bytes_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_bytes_to_jsonable(v) for v in value]
    return value


def _jsonable_to_bytes(value):
    if isinstance(value, dict):
        if set(value.keys()) == {_BLOB_B64_KEY}:
            return base64.b64decode(value[_BLOB_B64_KEY])
        return {k: _jsonable_to_bytes(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable_to_bytes(v) for v in value]
    return value


def _flatten_unknown_converse_fragment(fragment) -> dict:
    """Undo the streaming path's {"start": ..., "deltas": [...]} accumulation
    wrapper (see _format_stream_to_agency) so a round-tripped unknown block
    reconstructs the same shape Converse itself sent -- concatenating string
    fields across deltas. A non-streaming block's `data` has no such wrapper
    (it's already the final value), so it passes through unchanged. Mirrors
    anthropic.py's identical _flatten_unknown_fragment for its own unknown
    blocks."""
    if not isinstance(fragment, dict):
        return {}
    # Undo the base64-blob marking applied by _bytes_to_jsonable before this
    # was transported/logged as JSON, so blob fields (e.g.
    # reasoningContent.redactedContent) concatenate as bytes below rather
    # than as opaque marker dicts.
    fragment = _jsonable_to_bytes(fragment)
    if "start" in fragment or "deltas" in fragment:
        flat = dict(fragment.get("start") or {})
        for delta in fragment.get("deltas") or []:
            if not isinstance(delta, dict):
                continue
            for k, v in delta.items():
                prev = flat.get(k)
                if isinstance(v, str) and isinstance(prev, str):
                    flat[k] = prev + v
                elif isinstance(v, (bytes, bytearray)) and isinstance(prev, (bytes, bytearray)):
                    flat[k] = bytes(prev) + bytes(v)
                else:
                    flat[k] = v
        return flat
    return dict(fragment)


def _unknown_block_to_converse(b: dict) -> dict:
    return {
        b["type"][len(_CONVERSE_TYPE_PREFIX) :]: _flatten_unknown_converse_fragment(b.get("data"))
    }


def _agency_messages_to_converse(messages: list[dict]) -> "tuple[str | None, list[dict]]":
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
            content = []
            for b in blocks:
                if b["type"] == "text":
                    content.append({"text": b["text"]})
                elif b["type"].startswith(_CONVERSE_TYPE_PREFIX):
                    content.append(_unknown_block_to_converse(b))
            out.append({"role": "user", "content": content or [{"text": ""}]})
        elif role == "assistant":
            content = []
            for b in blocks:
                if b["type"] == "text":
                    content.append({"text": b["text"]})
                elif b["type"] == "thinking":
                    reasoning_text: dict = {"text": b["text"]}
                    if b.get("signature"):
                        reasoning_text["signature"] = b["signature"]
                    content.append({"reasoningContent": {"reasoningText": reasoning_text}})
                elif b["type"] == "tool_use":
                    try:
                        tool_input = json.loads(b["arguments"] or "{}")
                    except ValueError:
                        tool_input = {}
                    content.append(
                        {"toolUse": {"toolUseId": b["id"], "name": b["name"], "input": tool_input}}
                    )
                elif b["type"].startswith(_CONVERSE_TYPE_PREFIX):
                    content.append(_unknown_block_to_converse(b))
            # A turn can end with genuinely no text/tool_use/thinking (e.g.
            # gpt-5.6-terra ending a tool-loop with a bare stop, nothing else
            # in the message) -- Converse rejects an empty content array on
            # replay ("The content field in the Message object ... is
            # empty."), confirmed directly against Bedrock. Same fallback the
            # user-role branch above already uses.
            out.append({"role": "assistant", "content": content or [{"text": ""}]})
        elif role == "tool":
            result_block = next((b for b in blocks if b["type"] == "tool_result"), None)
            result = {
                "toolResult": {
                    "toolUseId": result_block.get("tool_call_id", "") if result_block else "",
                    "content": [{"text": result_block.get("text", "") if result_block else ""}],
                    "status": "success",
                }
            }
            prev = out[-1] if out and out[-1]["role"] == "user" else None
            if prev is not None:
                prev["content"].append(result)
            else:
                out.append({"role": "user", "content": [result]})
    return ("\n\n".join(system_parts) or None), out


def _agency_tools_to_converse(tools: "list[dict] | None") -> "list[dict] | None":
    if not tools:
        return None
    converted = []
    for t in tools:
        fn = t.get("function", t)
        converted.append(
            {
                "toolSpec": {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "inputSchema": {
                        "json": fn.get("parameters") or {"type": "object", "properties": {}}
                    },
                }
            }
        )
    return converted


def _agency_tool_choice_to_converse(tool_choice):
    """Converse's toolChoice has no "none" option (unlike OpenAI/Anthropic) --
    the caller drops `tools` entirely for that case instead of calling this."""
    if tool_choice == "auto":
        return {"auto": {}}
    if tool_choice == "required":
        return {"any": {}}
    if isinstance(tool_choice, dict):
        name = tool_choice.get("function", {}).get("name") or tool_choice.get("name")
        if name:
            return {"tool": {"name": name}}
    return None


class _BedrockConverseBackend(agllm):
    """Bedrock models that reject Mantle and need the native Converse API
    instead (see _needs_bedrock_converse) -- via boto3's bedrock-runtime
    client directly (no vendor SDK covers this shape)."""

    def make_client(self, timeout: httpx.Timeout):
        import boto3
        from botocore.config import Config as _BotoConfig

        region = self.agconfig.llm.region or "us-east-1"
        api_key = _require_bedrock_bearer_token(self.agconfig.llm)
        # boto3 has no per-client passthrough for a Bedrock API key bearer
        # token -- it's only ever resolved via this env var, read fresh on
        # every signed request (AWS's own documented mechanism for Bedrock
        # API keys). Last write wins if different Bedrock bearer tokens are
        # used concurrently in one process.
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key
        return boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=_BotoConfig(connect_timeout=timeout.connect, read_timeout=timeout.read),
        )

    def list_models(self) -> list:
        return []  # Bedrock's native Converse API has no OpenAI-style /v1/models

    def tokenize_url(self) -> "str | None":
        return None

    def _format_context_agency_to_backend(self, request: dict) -> dict:
        system, converse_messages = _agency_messages_to_converse(request["messages"])
        kwargs: dict = dict(modelId=self.agconfig.llm.model or "", messages=converse_messages)
        if system:
            kwargs["system"] = [{"text": system}]
        inference_config: dict = {
            "maxTokens": (
                self.agconfig.llm.max_completion_tokens
                or self.agconfig.llm.max_tokens
                or self.agconfig.llm.default_max_tokens
            )
        }
        if self.agconfig.llm.temperature is not None:
            inference_config["temperature"] = self.agconfig.llm.temperature
        if self.agconfig.llm.top_p is not None:
            inference_config["topP"] = self.agconfig.llm.top_p
        if self.agconfig.llm.stop is not None:
            stop = self.agconfig.llm.stop
            inference_config["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)
        kwargs["inferenceConfig"] = inference_config
        tool_choice = request.get("tool_choice")
        converse_tools = (
            None if tool_choice == "none" else _agency_tools_to_converse(request.get("tools"))
        )
        if converse_tools:
            tool_config: dict = {"tools": converse_tools}
            converse_tool_choice = _agency_tool_choice_to_converse(tool_choice)
            if converse_tool_choice is not None:
                tool_config["toolChoice"] = converse_tool_choice
            kwargs["toolConfig"] = tool_config
        extra_body = self.agconfig.llm.extra_body or {}
        if extra_body:
            kwargs["additionalModelRequestFields"] = extra_body
        return kwargs

    def _call_backend(self, backend_request: dict):
        client = self.make_client(self._client_timeout())
        return client.converse(**backend_request)

    def _format_context_backend_to_agency(self, raw_result) -> dict:
        message = raw_result.get("output", {}).get("message", {})
        blocks: "list[dict]" = []
        for content_block in message.get("content", []):
            if "text" in content_block:
                blocks.append({"type": "text", "index": len(blocks), "text": content_block["text"]})
            elif "toolUse" in content_block:
                tu = content_block["toolUse"]
                blocks.append(
                    {
                        "type": "tool_use",
                        "index": len(blocks),
                        "id": tu.get("toolUseId", ""),
                        "name": tu.get("name", ""),
                        "arguments": json.dumps(tu.get("input", {})),
                    }
                )
            elif (
                "reasoningContent" in content_block
                and "reasoningText" in content_block["reasoningContent"]
            ):
                # Plain-text reasoning only -- a model that instead returns
                # opaque encrypted reasoning (openai.gpt-5.x's
                # {"redactedContent": ...}) falls through to the generic
                # unknown-block branch below, which round-trips the whole
                # reasoningContent value byte-for-byte. Mapping *that* to a
                # "thinking" block would silently lose the opaque payload and
                # reconstruct an empty reasoningText on replay -- confirmed
                # directly against Bedrock to 400 ("This model doesn't support
                # the reasoningContent.reasoningText.text field").
                reasoning_text = content_block["reasoningContent"]["reasoningText"]
                blocks.append(
                    {
                        "type": "thinking",
                        "index": len(blocks),
                        "text": reasoning_text.get("text", ""),
                        "signature": reasoning_text.get("signature", ""),
                    }
                )
            else:
                ((key, value),) = content_block.items()
                blocks.append(
                    {
                        "type": _converse_native_block_type(key),
                        "index": len(blocks),
                        "data": _bytes_to_jsonable(value),
                    }
                )
        usage = raw_result.get("usage") or {}
        input_tokens = usage.get("inputTokens", 0) or 0
        output_tokens = usage.get("outputTokens", 0) or 0
        usage_dict = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": usage.get("totalTokens") or (input_tokens + output_tokens),
            "cache_read_tokens": usage.get("cacheReadInputTokens", 0) or 0,
            "cache_write_tokens": usage.get("cacheWriteInputTokens", 0) or 0,
        }
        stop_reason = raw_result.get("stopReason")
        blocks.append(
            {
                "type": "metadata",
                "index": len(blocks),
                "usage": usage_dict,
                "stop_reason": stop_reason,
                "data": _bytes_to_jsonable(
                    {k: v for k, v in raw_result.items() if k != "ResponseMetadata"}
                ),
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
        raw_stream = client.converse_stream(**backend_request)["stream"]
        return raw_stream, client

    def _format_stream_to_agency(self, raw_stream):
        input_tokens = 0
        output_tokens = 0
        cache_read_tokens = 0
        cache_write_tokens = 0
        stop_reason = None
        tool_blocks: "dict[int, dict]" = {}  # index -> {"id", "name", "json_parts"}
        unknown_blocks: "dict[int, dict]" = {}  # index -> {"native_type", "start", "deltas"}
        raw_events: "list[dict]" = []

        for event in raw_stream:
            raw_events.append(event)
            if "contentBlockStart" in event:
                start_event = event["contentBlockStart"]
                idx = start_event.get("contentBlockIndex", 0)
                block_start = start_event.get("start") or {}
                if "toolUse" in block_start:
                    tu = block_start["toolUse"]
                    tool_blocks[idx] = {
                        "id": tu.get("toolUseId", ""),
                        "name": tu.get("name", ""),
                        "json_parts": [],
                    }
                elif block_start:
                    ((native_type, value),) = block_start.items()
                    unknown_blocks[idx] = {"native_type": native_type, "start": value, "deltas": []}
            elif "contentBlockDelta" in event:
                delta_event = event["contentBlockDelta"]
                idx = delta_event.get("contentBlockIndex", 0)
                delta = delta_event.get("delta") or {}
                if "text" in delta:
                    yield {
                        "type": "block_delta",
                        "index": idx,
                        "block_type": "text",
                        "text": delta["text"],
                    }
                elif "toolUse" in delta:
                    block = tool_blocks.get(idx)
                    if block is not None:
                        block["json_parts"].append(delta["toolUse"].get("input", "") or "")
                elif "reasoningContent" in delta:
                    rc = delta["reasoningContent"]
                    if "text" in rc:
                        yield {
                            "type": "block_delta",
                            "index": idx,
                            "block_type": "thinking",
                            "text": rc["text"],
                        }
                    elif "signature" in rc:
                        yield {
                            "type": "block_delta",
                            "index": idx,
                            "block_type": "thinking",
                            "signature": rc["signature"],
                        }
                    else:
                        # Opaque encrypted reasoning (e.g. openai.gpt-5.x's
                        # {"redactedContent": ...}) -- these models send no
                        # contentBlockStart for this block, so the entry is
                        # created lazily here instead. Preserved as opaque
                        # native data (see _format_context_backend_to_agency's
                        # identical reasoning) rather than mapped to
                        # "thinking", which would silently lose the payload
                        # and break replay (confirmed directly against
                        # Bedrock: replaying an empty reasoningText gets a
                        # 400 "This model doesn't support the
                        # reasoningContent.reasoningText.text field").
                        entry = unknown_blocks.setdefault(
                            idx, {"native_type": "reasoningContent", "start": None, "deltas": []}
                        )
                        entry["deltas"].append(rc)
                elif idx in unknown_blocks:
                    unknown_blocks[idx]["deltas"].append(delta)
            elif "contentBlockStop" in event:
                idx = event["contentBlockStop"].get("contentBlockIndex", 0)
                block = tool_blocks.pop(idx, None)
                if block is not None:
                    yield {
                        "type": "block_delta",
                        "index": idx,
                        "block_type": "tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "arguments": "".join(block["json_parts"]),
                    }
                unknown = unknown_blocks.pop(idx, None)
                if unknown is not None:
                    yield {
                        "type": "block_delta",
                        "index": idx,
                        "block_type": _converse_native_block_type(unknown["native_type"]),
                        "data": _bytes_to_jsonable(
                            {"start": unknown["start"], "deltas": unknown["deltas"]}
                        ),
                    }
            elif "messageStop" in event:
                stop_reason = event["messageStop"].get("stopReason")
            elif "metadata" in event:
                usage = event["metadata"].get("usage") or {}
                input_tokens = usage.get("inputTokens", 0) or input_tokens
                output_tokens = usage.get("outputTokens", 0) or output_tokens
                cache_read_tokens = usage.get("cacheReadInputTokens", 0) or cache_read_tokens
                cache_write_tokens = usage.get("cacheWriteInputTokens", 0) or cache_write_tokens

        # If the stream ended (e.g. stopReason="max_tokens") while a tool_use
        # block was still open, contentBlockStop never fires for it -- flush
        # whatever JSON was collected so far instead of silently dropping the
        # call (same reasoning as anthropic.py's identical flush).
        for idx in sorted(tool_blocks):
            block = tool_blocks[idx]
            print(
                f"[agllm] WARNING: tool_use block {block['name']!r} (id={block['id']}) "
                f"truncated mid-stream (likely hit max_tokens) -- flushing partial "
                f"arguments instead of dropping the call"
            )
            yield {
                "type": "block_delta",
                "index": idx,
                "block_type": "tool_use",
                "id": block["id"],
                "name": block["name"],
                "arguments": "".join(block["json_parts"]),
            }
        for idx in sorted(unknown_blocks):
            unknown = unknown_blocks[idx]
            yield {
                "type": "block_delta",
                "index": idx,
                "block_type": _converse_native_block_type(unknown["native_type"]),
                "data": _bytes_to_jsonable(
                    {"start": unknown["start"], "deltas": unknown["deltas"]}
                ),
            }

        usage_dict = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
        }
        yield {
            "type": "block_delta",
            "index": _CONVERSE_METADATA_BLOCK_INDEX,
            "block_type": "metadata",
            "data": _bytes_to_jsonable(
                {"usage": usage_dict, "stop_reason": stop_reason, "raw_events": raw_events}
            ),
        }
        yield {"type": "usage", "usage": usage_dict, "stop_reason": stop_reason}


class _AnthropicAWSBackend(_AnthropicBackend):
    """Claude Platform on AWS via the anthropic SDK's AnthropicAWS client.

    Auth (resolved by the SDK): SigV4 via the default AWS credential chain,
    explicit aws_access_key/aws_secret_key, or an API key (config `api_key` /
    ANTHROPIC_AWS_API_KEY). Requires workspace_id (config /
    ANTHROPIC_AWS_WORKSPACE_ID) and aws_region (config `region` or
    `aws_region` / AWS_REGION) unless base_url is set.
    """

    def _client_kwargs(self, timeout: httpx.Timeout) -> dict:
        kwargs: dict = dict(timeout=_anthropic_sdk_timeout(timeout))
        api_key = self.agconfig.llm.api_key or os.environ.get("ANTHROPIC_AWS_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
        for key in ("aws_access_key", "aws_secret_key", "aws_session_token", "aws_profile"):
            value = getattr(self.agconfig.llm, key)
            if value:
                kwargs[key] = value
        region = self.agconfig.llm.aws_region or self.agconfig.llm.region
        if region:
            kwargs["aws_region"] = region
        workspace_id = (
            self.agconfig.llm.workspace_id
            or os.environ.get("ANTHROPIC_AWS_WORKSPACE_ID")
            or os.environ.get("ANTHROPIC_WORKSPACE_ID")
        )
        if workspace_id:
            kwargs["workspace_id"] = workspace_id
        base_url = (
            self.agconfig.llm.base_url
            or os.environ.get("ANTHROPIC_AWS_BASE_URL")
            or os.environ.get("ANTHROPIC_BASE_URL")
        )
        if base_url:
            kwargs["base_url"] = base_url
        return kwargs

    def make_client(self, timeout: httpx.Timeout):
        if _anthropic_sdk is None:
            raise RuntimeError(
                "provider='anthropicAWS' requires the 'anthropic' package: pip install anthropic"
            )
        if not hasattr(_anthropic_sdk, "AnthropicAWS"):
            raise RuntimeError(
                "provider='anthropicAWS' requires a recent 'anthropic' package with AnthropicAWS support"
            )
        return _anthropic_sdk.AnthropicAWS(**self._client_kwargs(timeout))

    def list_models(self) -> list:
        if _anthropic_sdk is None or not hasattr(_anthropic_sdk, "AnthropicAWS"):
            return []
        client = _anthropic_sdk.AnthropicAWS(
            **self._client_kwargs(httpx.Timeout(self.agconfig.llm.model_listing_timeout_seconds))
        )
        return list(client.models.list())

    def tokenize_url(self) -> "str | None":
        return None

    def known_context_limit(self, model: str) -> "int | None":
        return _known_anthropic_context_window(model)
