from __future__ import annotations

import re
from dataclasses import dataclass


_ANTHROPIC_BEDROCK_MODEL_RE = re.compile(r"^(?:(?:us|eu|apac|global)\.)?anthropic\.")

_OPENAI_GEN_PARAMS = {
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "n",
    "stop",
    "logprobs",
    "seed",
}

_EXTRA_BODY_GEN_PARAMS = {
    "top_k",
    "repetition_penalty",
    "min_p",
    "min_tokens",
    "guided_json",
    "guided_regex",
}


@dataclass(frozen=True)
class agconfig:
    """Normalized view of user-provided LLM configuration.

    The public API still accepts plain dicts. This class centralizes aliases
    and provider-specific request parameter names so the rest of the framework
    does not need to know which backend calls an output-token budget
    max_completion_tokens versus max_tokens.
    """

    raw: dict

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_user(cls, config: "dict | agconfig") -> "agconfig":
        if isinstance(config, agconfig):
            return config
        if not isinstance(config, dict):
            raise TypeError("llm_config must be a dict or agconfig")
        return cls(config)

    # ------------------------------------------------------------------
    # Property Getters
    # ------------------------------------------------------------------

    @property
    def provider(self) -> str | None:
        return self.raw.get("provider")

    @property
    def model(self) -> str:
        return self.raw.get("model", "")

    @property
    def context_limit(self) -> int | None:
        if "context_limit" not in self.raw:
            return None
        return int(self.raw["context_limit"])

    @property
    def max_output_tokens(self) -> int | None:
        if "max_completion_tokens" in self.raw:
            return self.raw["max_completion_tokens"]
        if "max_output_tokens" in self.raw:
            return self.raw["max_output_tokens"]
        if "max_tokens" in self.raw:
            return self.raw["max_tokens"]
        return None

    @property
    def uses_anthropic_messages_api(self) -> bool:
        provider = self.provider
        return (
            provider in ("anthropic", "anthropicAWS", "anthropic_aws")
            or (provider == "bedrock" and bool(_ANTHROPIC_BEDROCK_MODEL_RE.match(self.model or "")))
        )

    def completion_token_wire_key(self) -> str:
        if self.uses_anthropic_messages_api:
            return "max_tokens"
        return "max_completion_tokens"

    # ------------------------------------------------------------------
    # Dict Views
    # ------------------------------------------------------------------

    def backend_config(self) -> dict:
        """Return a plain dict suitable for backend selection/client auth."""
        return dict(self.raw)

    def sanitized_dict(self) -> dict:
        return {k: v for k, v in self.raw.items() if k != "api_key"}

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _warn_aliases(self) -> None:
        if "max_tokens" in self.raw:
            print("[agllm] WARNING: llm_config['max_tokens'] is deprecated; use 'max_completion_tokens' instead.")

    def _extra_body(self) -> dict:
        extra_body: dict = dict(self.raw.get("extra_body") or {})
        for param in _EXTRA_BODY_GEN_PARAMS:
            if param in self.raw:
                extra_body[param] = self.raw[param]
        return extra_body

    @staticmethod
    def wire_messages(messages: list[dict]) -> list[dict]:
        wire_messages: list[dict] = []
        for m in messages:
            wire_msg = {k: v for k, v in m.items() if not k.startswith("_")}
            if wire_msg.get("content") is None:
                wire_msg["content"] = ""
            wire_messages.append(wire_msg)
        return wire_messages

    # ------------------------------------------------------------------
    # Request Builders
    # ------------------------------------------------------------------

    def build_chat_kwargs(self, messages: list[dict], openai_tools: "list | None" = None) -> dict:
        self._warn_aliases()
        kwargs: dict = dict(
            model=self.raw.get("model", "gpt-4o"),
            messages=agconfig.wire_messages(messages),
        )
        for param in _OPENAI_GEN_PARAMS:
            if param in self.raw:
                kwargs[param] = self.raw[param]
        max_output_tokens = self.max_output_tokens
        if max_output_tokens is not None:
            kwargs[self.completion_token_wire_key()] = max_output_tokens
        extra_body = self._extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
        if openai_tools:
            kwargs["tools"] = openai_tools
        return kwargs

    def build_compact_kwargs(self, messages: list[dict], max_output_tokens: int) -> dict:
        kwargs: dict = dict(
            model=self.raw.get("model", "gpt-4o"),
            messages=agconfig.wire_messages(messages),
            **{self.completion_token_wire_key(): max_output_tokens},
        )
        if "extra_body" in self.raw:
            kwargs["extra_body"] = self.raw["extra_body"]
        return kwargs
