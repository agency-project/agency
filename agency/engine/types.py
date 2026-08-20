from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agcontext import agcontext
    from ..agdata import agdata
    from ..sandbox.agsandbox import agSandbox


@dataclass
class ProvisionedSandbox:
    sandbox: "agSandbox | None"
    uds_path: "str | None"


@dataclass
class PromptPayload:
    prompt: "str | list"
    output_format_instruction: "str | None"
    extra_system: "str | None"


@dataclass
class HarnessManagerHandle:
    base_url: str
    uds_path: "str | None"
    pid: "int | None" = None


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
