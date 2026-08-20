from __future__ import annotations

from typing import TYPE_CHECKING

from .types import (
    ExecutionResult,
    HarnessAttemptResult,
    PromptPayload,
)

if TYPE_CHECKING:
    from ..agconfig import agConfig
    from ..agcontext import agcontext
    from ..agdata import agdata
    from ..agent import agent
    from ..agresources import agResourcePool
    from ..agskill import agskill
    from ..sandbox.agsandbox import agSandbox
    from .harness_bridge import HarnessManagerBridge
    from .host_servers.host_server_manager import HostServerManager
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
        agent: "agent",
        context: "agcontext",
        skill: "agskill",
        skill_input: "agdata",
        resource_pool: "agResourcePool",
    ) -> ExecutionResult:
        raise NotImplementedError

    def build_container_and_uds(self, sandbox: "agSandbox") -> "agSandbox":
        raise NotImplementedError

    def build_host_side_server(
        self,
        agent: "agent",
        skill: "agskill",
        sandbox: "agSandbox",
        resource_pool: "agResourcePool",
    ) -> "HostServerManager":
        raise NotImplementedError

    def build_prompt_payload(self, skill: "agskill", skill_input: "agdata") -> PromptPayload:
        raise NotImplementedError

    def launch_harness_manager(
        self, sandbox: "agSandbox", host_server_manager: "HostServerManager"
    ) -> "int | None":
        raise NotImplementedError

    def run_agent_harness(
        self,
        agconfig: "agConfig",
        skill: "agskill",
        prompt: PromptPayload,
        harness_manager_pid: "int | None",
    ) -> HarnessAttemptResult:
        raise NotImplementedError

    def wait_for_completion(
        self, harness_manager_pid: "int | None", attempt: HarnessAttemptResult
    ) -> ExecutionResult:
        raise NotImplementedError
