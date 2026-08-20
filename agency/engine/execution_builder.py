from __future__ import annotations

from typing import TYPE_CHECKING

from .types import (
    ExecutionResult,
    HarnessAttemptResult,
    HarnessManagerHandle,
    PromptPayload,
    ProvisionedSandbox,
)

if TYPE_CHECKING:
    from ..agconfig import agConfig
    from ..agcontext import agcontext
    from ..agdata import agdata
    from ..agskill import agskill
    from ..sandbox.agsandbox import agSandbox
    from .harness_bridge import HarnessManagerBridge
    from .host_server_manager.host_server_manager import HostServerManager
    from .sandbox_provisioner import SandboxProvisioner


class ExecutionBuilder:
    def __init__(
        self,
        sandbox_provisioner: "SandboxProvisioner",
        harness_bridge: "HarnessManagerBridge",
    ) -> None:
        raise NotImplementedError

    def build_and_run(
        self,
        context: "agcontext",
        sandbox: "agSandbox",
        agconfig: "agConfig",
        skill: "agskill",
        skill_input: "agdata",
    ) -> ExecutionResult:
        raise NotImplementedError

    def build_container_and_uds(self, sandbox: "agSandbox") -> ProvisionedSandbox:
        raise NotImplementedError

    def build_host_side_server(
        self,
        context: "agcontext",
        agconfig: "agConfig",
        skill: "agskill",
        provisioned: ProvisionedSandbox,
    ) -> "HostServerManager":
        raise NotImplementedError

    def build_prompt_payload(self, skill: "agskill", skill_input: "agdata") -> PromptPayload:
        raise NotImplementedError

    def launch_harness_manager(
        self, provisioned: ProvisionedSandbox, host_server_manager: "HostServerManager"
    ) -> HarnessManagerHandle:
        raise NotImplementedError

    def run_agent_harness(
        self,
        agconfig: "agConfig",
        skill: "agskill",
        prompt: PromptPayload,
        harness: HarnessManagerHandle,
    ) -> HarnessAttemptResult:
        raise NotImplementedError

    def wait_for_completion(
        self, harness: HarnessManagerHandle, attempt: HarnessAttemptResult
    ) -> ExecutionResult:
        raise NotImplementedError
