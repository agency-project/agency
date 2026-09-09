"""Shared host ↔ sandbox protocol types for one harness attempt."""

from __future__ import annotations

from dataclasses import dataclass


ATTEMPT_TOKEN_HEADER = "X-Agency-Attempt-Token"


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
    # Fresh for each host -> sandbox RPC. The long-lived daemon and host
    # gateway accept only the token belonging to the currently active attempt.
    attempt_token: "str | None" = None
    # Explicit sandbox tools only; host tools and the skill stay on the host.
    sandbox_mcp_tools_b64: "str | None" = None
    # Static summary of this attempt's syscall policy (see agpolicy) -- lets
    # the daemon's _HostSyscallPolicy decide locally, without a host round
    # trip, whenever a trapped syscall has no hook (see daemon.py). Only
    # hook *names* cross this boundary, never the hook callables themselves.
    syscall_default_to_deny: bool = False
    syscall_hooked_names: "list[str] | None" = None


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


__all__ = [
    "ATTEMPT_TOKEN_HEADER",
    "PromptPayload",
    "HarnessAttemptRequest",
    "HarnessAttemptResult",
]
