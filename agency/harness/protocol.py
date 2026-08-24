"""Shared host ↔ sandbox protocol types for one harness attempt."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptPayload:
    system_instruction: str
    user_content: "str | list[dict]"
    output_instruction: "str | None" = None


@dataclass
class HarnessAttemptRequest:
    prompt: PromptPayload
    harness: str
    max_steps: "int | None" = None


@dataclass
class HarnessAttemptResult:
    ok: bool
    final_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    session_id: "str | None" = None
    error_message: str = ""


__all__ = ["PromptPayload", "HarnessAttemptRequest", "HarnessAttemptResult"]
