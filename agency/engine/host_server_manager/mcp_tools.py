from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...agskill import agskill
    from ...sandbox.agsandbox import agSandbox


class HostMcpTools:
    def __init__(self, sandbox: "agSandbox", skill: "agskill") -> None:
        raise NotImplementedError

    def reserve_cpu(self, count: float) -> bool:
        raise NotImplementedError

    def cpu_release(self, count: float) -> None:
        raise NotImplementedError

    def daemon_release(self, pid: int) -> None:
        raise NotImplementedError

    def submit_output(self, field: str, value: "object") -> None:
        raise NotImplementedError

    def ask_human(self, question: str) -> str:
        raise NotImplementedError

    def collected_output(self) -> dict:
        raise NotImplementedError
