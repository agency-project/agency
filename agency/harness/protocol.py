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
    resume_session_id: "str | None" = None
    # Session files are opaque bytes, so the JSON protocol carries them as base64.
    prior_session_blob_b64: "str | None" = None
    # Python tool closures cannot cross the JSON host/sandbox boundary.
    # The host rejects non-empty custom tool sets before sending a request;
    # an explicit empty replacement may still suppress built-in tools.
    suppress_builtin_tools: bool = False


@dataclass
class HarnessAttemptResult:
    ok: bool
    final_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    session_id: "str | None" = None
    # Base64-encoded adapter session state paired with ``session_id``.
    session_blob_b64: "str | None" = None
    error_message: str = ""


__all__ = ["PromptPayload", "HarnessAttemptRequest", "HarnessAttemptResult"]
