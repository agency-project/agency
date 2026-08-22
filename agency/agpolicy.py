from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .sandbox.events import agsyscallevent


@dataclass
class agpolicy:
    tool_hooks: "dict[str, Callable[[dict], bool | tuple[bool, str]]] | None" = None
    syscall_hooks: "dict[str, Callable[[agsyscallevent], bool | tuple[bool, str]]] | None" = None
    default_to_deny: bool = False
