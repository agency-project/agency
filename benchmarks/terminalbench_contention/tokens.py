"""Prompt-token estimation for the OpenAI Chat Completions wire format.

The previous benchmark charged ``ceil(request UTF-8 bytes / 2)`` for the prompt.
JSON framing, escaping and tool schemas made that roughly double the tokens the
model reported. This module counts what the model is actually sent:

* every message's role, content, name and tool calls, with the documented
  per-message framing overhead;
* tool (function) schemas, rendered the way OpenAI models receive them;
* reply priming and tool-choice overhead.

Luna's tokenizer is not published to this repository. ``o200k_base`` is the
configurable default because it is the encoding of the current OpenAI chat model
family; ``calibrate.py`` measures the estimate against model-reported usage from
recorded requests. A multiplicative ``safety_factor`` and additive
``safety_tokens`` make the estimate deliberately conservative, and both are
recorded with every admission so the operator can re-tune them after verifying
the gateway. When the tokenizer cannot be loaded the estimator falls back to a
conservative bytes-per-token bound and says so in its output; it never silently
returns a smaller number.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

DEFAULT_ENCODING = "o200k_base"


@dataclass(frozen=True)
class EstimatorConfig:
    encoding: str = DEFAULT_ENCODING
    # Chat framing: <|start|>{role}<|message|>{content}<|end|> per message, and
    # the assistant reply is primed with <|start|>assistant<|message|>.
    tokens_per_message: int = 3
    tokens_per_name: int = 1
    tokens_reply_priming: int = 3
    # A tool call is rendered as a recipient header plus its JSON arguments.
    tokens_per_tool_call: int = 4
    # "typescript": tools rendered as the `namespace functions { ... }` block the
    # model receives. "json": tokens of the compact JSON schema (an upper bound
    # for typical schemas; kept for calibration comparisons).
    tool_schema_mode: str = "typescript"
    tokens_tools_preamble: int = 12
    tokens_per_tool: int = 4
    # Conservative margins applied to the tokenizer count.
    safety_factor: float = 1.05
    safety_tokens: int = 16
    # Fallback when the tokenizer is unavailable: UTF-8 bytes / this value. Three
    # bytes per token over-estimates English text and code for o200k_base.
    fallback_bytes_per_token: float = 3.0

    def validate(self) -> None:
        if self.tool_schema_mode not in ("typescript", "json"):
            raise ValueError(f"Unknown tool_schema_mode {self.tool_schema_mode!r}")
        if self.safety_factor < 1.0:
            raise ValueError("safety_factor below 1.0 would under-reserve tokens")
        if self.safety_tokens < 0 or self.fallback_bytes_per_token <= 0:
            raise ValueError("safety_tokens must be >= 0 and fallback_bytes_per_token > 0")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PromptEstimate:
    """Breakdown of one request's prompt estimate. All values are tokens."""

    message_tokens: int = 0
    tool_schema_tokens: int = 0
    protocol_tokens: int = 0
    raw_tokens: int = 0  # tokenizer count before safety margins
    estimated_tokens: int = 0  # what admission charges for the prompt
    method: str = "tokenizer"  # "tokenizer" | "fallback_bytes"
    encoding: str = DEFAULT_ENCODING
    messages: int = 0
    tools: int = 0
    by_role: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class TokenizerUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=4)
def load_encoding(name: str = DEFAULT_ENCODING):
    """Load a tiktoken encoding. The BPE file is read from TIKTOKEN_CACHE_DIR
    (populated during preparation) so measured runs never fetch it."""
    try:
        import tiktoken

        return tiktoken.get_encoding(name)
    except Exception as exc:  # missing package, missing cache, no network
        raise TokenizerUnavailable(f"Cannot load tokenizer {name!r}: {exc}") from exc


def tokenizer_fingerprint(name: str = DEFAULT_ENCODING) -> dict:
    """Identify the tokenizer actually in use, for the experiment manifest."""
    try:
        import tiktoken

        encoding = load_encoding(name)
        cache = os.environ.get("TIKTOKEN_CACHE_DIR")
        files = sorted(str(p.name) for p in Path(cache).iterdir()) if cache else []
        return dict(
            available=True,
            encoding=name,
            vocabulary_size=encoding.n_vocab,
            tiktoken_version=tiktoken.__version__,
            cache_dir=cache,
            cache_files=files,
        )
    except Exception as exc:
        return dict(available=False, encoding=name, error=str(exc))


def _content_text(content) -> str:
    """Text the model sees for a message's content (string or content parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                else:  # unknown part type: count its serialized form
                    parts.append(json.dumps(part, ensure_ascii=False, separators=(",", ":")))
            else:
                parts.append(str(part))
        return "".join(parts)
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


_TS_PRIMITIVES = {"string": "string", "number": "number", "integer": "number", "boolean": "boolean"}


def _ts_type(schema, indent: int) -> str:
    if not isinstance(schema, dict):
        return "any"
    if "enum" in schema and isinstance(schema["enum"], list):
        return " | ".join(json.dumps(v, ensure_ascii=False) for v in schema["enum"])
    for key in ("anyOf", "oneOf"):
        if isinstance(schema.get(key), list):
            return " | ".join(_ts_type(s, indent) for s in schema[key])
    kind = schema.get("type")
    if isinstance(kind, list):
        return " | ".join(_ts_type({**schema, "type": k}, indent) for k in kind)
    if kind in _TS_PRIMITIVES:
        return _TS_PRIMITIVES[kind]
    if kind == "null":
        return "null"
    if kind == "array":
        return _ts_type(schema.get("items"), indent) + "[]"
    if kind == "object" or "properties" in schema:
        return _ts_object(schema, indent)
    return "any"


def _ts_object(schema: dict, indent: int) -> str:
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return "object"
    required = set(schema.get("required") or [])
    pad = " " * indent
    lines = ["{"]
    for name, prop in properties.items():
        if isinstance(prop, dict):
            description = prop.get("description")
            if isinstance(description, str) and description:
                lines.extend(f"{pad}// {line}" for line in description.splitlines())
            if "default" in prop:
                lines.append(f"{pad}// default: {json.dumps(prop['default'], ensure_ascii=False)}")
        optional = "" if name in required else "?"
        lines.append(f"{pad}{name}{optional}: {_ts_type(prop, indent + 1)},")
    lines.append(" " * max(indent - 1, 0) + "}")
    return "\n".join(lines)


def render_tools_typescript(tools: list) -> str:
    """Render function tools as the TypeScript namespace models are prompted with."""
    lines = ["# Tools", "", "## functions", "", "namespace functions {", ""]
    for tool in tools:
        function = tool.get("function", tool) if isinstance(tool, dict) else {}
        description = function.get("description")
        if isinstance(description, str) and description:
            lines.extend(f"// {line}" for line in description.splitlines())
        parameters = function.get("parameters")
        name = function.get("name", "")
        if isinstance(parameters, dict) and parameters.get("properties"):
            lines.append(f"type {name} = (_: {_ts_object(parameters, 1)}) => any;")
        else:
            lines.append(f"type {name} = () => any;")
        lines.append("")
    lines.append("} // namespace functions")
    return "\n".join(lines)


class TokenEstimator:
    """Counts prompt tokens for one Chat Completions request body."""

    def __init__(self, config: EstimatorConfig | None = None, *, encoding=None) -> None:
        self.config = config or EstimatorConfig()
        self.config.validate()
        self._encoding = encoding
        self._fallback_reason: str | None = None
        if self._encoding is None:
            try:
                self._encoding = load_encoding(self.config.encoding)
            except TokenizerUnavailable as exc:
                self._fallback_reason = str(exc)

    @property
    def method(self) -> str:
        return "tokenizer" if self._encoding is not None else "fallback_bytes"

    @property
    def fallback_reason(self) -> "str | None":
        return self._fallback_reason

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is not None:
            # Text is untrusted model/tool output; special-token strings in it
            # are ordinary text, never control tokens.
            return len(self._encoding.encode(text, disallowed_special=()))
        return math.ceil(len(text.encode("utf-8")) / self.config.fallback_bytes_per_token)

    def _message_tokens(self, message: dict) -> int:
        cfg = self.config
        total = cfg.tokens_per_message
        total += self.count_text(str(message.get("role", "")))
        total += self.count_text(_content_text(message.get("content")))
        if message.get("name"):
            total += cfg.tokens_per_name + self.count_text(str(message["name"]))
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {} if isinstance(call, dict) else {}
            total += cfg.tokens_per_tool_call
            total += self.count_text(str(function.get("name", "")))
            total += self.count_text(str(function.get("arguments", "")))
        function_call = message.get("function_call")
        if isinstance(function_call, dict):
            total += cfg.tokens_per_tool_call
            total += self.count_text(str(function_call.get("name", "")))
            total += self.count_text(str(function_call.get("arguments", "")))
        return total

    def _tool_tokens(self, tools: list) -> int:
        cfg = self.config
        if not tools:
            return 0
        if cfg.tool_schema_mode == "json":
            body = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
        else:
            body = render_tools_typescript(tools)
        return self.count_text(body) + cfg.tokens_tools_preamble + cfg.tokens_per_tool * len(tools)

    def estimate(self, payload: dict) -> PromptEstimate:
        """Estimate prompt tokens for a decoded Chat Completions request body."""
        cfg = self.config
        messages = payload.get("messages") or []
        tools = payload.get("tools") or payload.get("functions") or []
        by_role: dict = {}
        message_tokens = 0
        for message in messages:
            if not isinstance(message, dict):
                continue
            tokens = self._message_tokens(message)
            message_tokens += tokens
            role = str(message.get("role", ""))
            by_role[role] = by_role.get(role, 0) + tokens
        tool_tokens = self._tool_tokens(tools if isinstance(tools, list) else [])
        protocol = cfg.tokens_reply_priming
        tool_choice = payload.get("tool_choice")
        if isinstance(tool_choice, dict):
            protocol += self.count_text(json.dumps(tool_choice, separators=(",", ":")))
        response_format = payload.get("response_format")
        if isinstance(response_format, dict):
            protocol += self.count_text(
                json.dumps(response_format, ensure_ascii=False, separators=(",", ":"))
            )
        raw = message_tokens + tool_tokens + protocol
        estimated = math.ceil(raw * cfg.safety_factor) + cfg.safety_tokens
        return PromptEstimate(
            message_tokens=message_tokens,
            tool_schema_tokens=tool_tokens,
            protocol_tokens=protocol,
            raw_tokens=raw,
            estimated_tokens=estimated,
            method=self.method,
            encoding=cfg.encoding,
            messages=len(messages),
            tools=len(tools) if isinstance(tools, list) else 0,
            by_role=by_role,
        )

    def estimate_bytes(self, body: bytes) -> PromptEstimate:
        """Estimate from raw request bytes; an undecodable body is charged by size."""
        try:
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("request body is not a JSON object")
        except (ValueError, UnicodeDecodeError):
            raw = math.ceil(len(body) / self.config.fallback_bytes_per_token)
            return PromptEstimate(
                raw_tokens=raw,
                estimated_tokens=math.ceil(raw * self.config.safety_factor)
                + self.config.safety_tokens,
                method="fallback_bytes",
                encoding=self.config.encoding,
            )
        return self.estimate(payload)


def completion_cap(payload: dict, default: int) -> int:
    """The output-token cap the request declares, as the provider sees it."""
    for key in ("max_completion_tokens", "max_tokens", "max_output_tokens"):
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value > 0:
            return int(value) * max(1, int(payload.get("n") or 1))
    return int(default)


def legacy_estimate(body: bytes, payload: dict) -> int:
    """The previous benchmark's charge, kept only for side-by-side telemetry."""
    return math.ceil(len(body) / 2) + int(
        payload.get("max_completion_tokens", payload.get("max_tokens", 4096)) or 4096
    )
