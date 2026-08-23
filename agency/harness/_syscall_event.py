"""Architecture-neutral policy event shared by ptrace and native hooks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class agsyscallevent:
    """One syscall or semantic tool call resolved for ``agpolicy``.

    ``argv``/``envp``/``path`` carry kernel-resolved syscall arguments when
    ptrace is active. ``tool_name``/``tool_args`` carry the corresponding
    harness semantic fields for native hooks. Keeping this definition outside
    the x86_64-only ptrace module lets both paths hand policy the exact same
    concrete event type on every architecture.
    """

    syscall: str
    pid: int
    tid: int
    argv: "list[str] | None"
    envp: "dict[str, str] | None"
    path: "str | None"
    timestamp: float
    tool_name: "str | None" = None
    tool_args: "dict | None" = None


__all__ = ["agsyscallevent"]
