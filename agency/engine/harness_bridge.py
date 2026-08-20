from __future__ import annotations

from typing import TYPE_CHECKING

from .types import HarnessAttemptResult, HarnessManagerHandle, PromptPayload

if TYPE_CHECKING:
    from ..agconfig import agConfig
    from ..agskill import agskill
    from .host_server_manager.host_server_manager import HostServerManager
    from .policy_manager import AgentHarnessPolicyManager
    from .types import ProvisionedSandbox


class HarnessManagerBridge:
    def __init__(self, policy_manager: "AgentHarnessPolicyManager") -> None:
        raise NotImplementedError

    def ensure_launched(
        self, provisioned: "ProvisionedSandbox", host_server_manager: "HostServerManager"
    ) -> HarnessManagerHandle:
        raise NotImplementedError

    def run_attempt(
        self,
        agconfig: "agConfig",
        skill: "agskill",
        harness: HarnessManagerHandle,
        prompt: PromptPayload,
    ) -> HarnessAttemptResult:
        raise NotImplementedError

    def wait(self, harness: HarnessManagerHandle) -> None:
        raise NotImplementedError

    def stop(self, harness: HarnessManagerHandle) -> None:
        raise NotImplementedError
