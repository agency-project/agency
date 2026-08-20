from __future__ import annotations

from typing import TYPE_CHECKING

from .types import HarnessAttemptResult, PromptPayload

if TYPE_CHECKING:
    from ..agconfig import agConfig
    from ..agskill import agskill
    from ..sandbox.agsandbox import agSandbox
    from .host_servers.host_server_manager import HostServerManager


class HarnessManagerBridge:
    def __init__(self) -> None:
        raise NotImplementedError

    def ensure_launched(
        self, sandbox: "agSandbox", host_server_manager: "HostServerManager"
    ) -> "int | None":
        raise NotImplementedError

    def run_attempt(
        self,
        agconfig: "agConfig",
        skill: "agskill",
        harness_manager_pid: "int | None",
        prompt: PromptPayload,
    ) -> HarnessAttemptResult:
        raise NotImplementedError

    def wait(self, harness_manager_pid: "int | None") -> None:
        raise NotImplementedError

    def stop(self, harness_manager_pid: "int | None") -> None:
        raise NotImplementedError
