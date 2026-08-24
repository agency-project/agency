from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agcontext import agcontext
    from ..agdata import agdata


@dataclass
class ExecutionResult:
    output: "agdata"
    context: "agcontext"
    delta: "list[dict]"
    ok: bool = True
    error_message: str = ""
