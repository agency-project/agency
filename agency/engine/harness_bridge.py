from __future__ import annotations

from typing import TYPE_CHECKING

from .types import HarnessAttemptResult, PromptPayload

if TYPE_CHECKING:
    from ..agconfig import agConfig
    from ..agskill import agskill
    from ..sandbox.agsandbox import agSandbox


class HarnessManagerBridge:
    def __init__(self) -> None:
        # The bridge is a host-side lifecycle/client object.  Constructing it
        # must not launch the sandbox-side harness manager; that remains a
        # lazy operation behind ensure_launched().
        pass

    def ensure_launched(self, sandbox: "agSandbox", host_uds_path: str) -> "int | None":
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
