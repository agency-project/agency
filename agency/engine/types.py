from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agcontext import agcontext
    from ..agdata import agdata


@dataclass
class PromptPayload:
    prompt: "str | list"
    output_format_instruction: "str | None"
    extra_system: "str | None"


@dataclass
class HarnessAttemptResult:
    ok: bool
    final_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    session_id: "str | None" = None
    error_message: str = ""


@dataclass
class ExecutionResult:
    output: "agdata"
    context: "agcontext"
    delta: "list[dict]"
    ok: bool = True
    error_message: str = ""
