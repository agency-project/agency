from __future__ import annotations

from typing import TYPE_CHECKING

from ..harness.protocol import HarnessAttemptResult, PromptPayload
from .host_servers.host_server_manager import HostServerManager
from .types import ExecutionResult

if TYPE_CHECKING:
    from ..agconfig import agConfig
    from ..agcontext import agcontext
    from ..agdata import agdata
    from ..agent import agent
    from ..agresources import agResourcePool
    from ..agskill import agskill
    from .clients import SandboxInteractionClient


class AgentEngine:
    """Host-side execution owner for one agent."""

    def __init__(self, agent: "agent") -> None:
        self._agent = agent
        self._host_server_manager: "HostServerManager | None" = None
        self._sandbox_interaction_client: "SandboxInteractionClient | None" = None

    def set_config(self, agconfig: "agConfig") -> None:
        if self._host_server_manager is not None:
            self._host_server_manager.set_config(agconfig)

    @property
    def host_server_manager(self) -> "HostServerManager":
        if self._host_server_manager is None:
            raise RuntimeError("execute() has not built a HostServerManager yet")
        return self._host_server_manager

    def execute(
        self,
        context: "agcontext",
        skill: "agskill",
        skill_input: "agdata",
        resource_pool: "agResourcePool",
        max_steps: "int | None" = None,
    ) -> ExecutionResult:
        """Execute one declarative skill request through the host services."""
        self._host_server_manager = HostServerManager(
            self._agent, self._agent.sandbox, skill, resource_pool
        )
        self._host_server_manager.start()
        try:
            self._ensure_harness_manager_launched()
            prompt = self._build_prompt_payload(skill, skill_input)
            retries_left = skill.max_output_schema_retries
            attempt: "HarnessAttemptResult | None" = None
            while True:
                attempt = self._run_attempt(prompt)
                if not attempt.ok:
                    break
                missing = self._missing_output_fields(skill)
                if not missing or retries_left <= 0:
                    break
                retries_left -= 1
                prompt = self._build_retry_prompt(
                    missing, system_instruction=prompt.system_instruction
                )
            return self._build_execution_result(context, skill, attempt)
        finally:
            self._host_server_manager.stop()

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _ensure_harness_manager_launched(self) -> None:
        raise NotImplementedError

    def _build_prompt_payload(self, skill: "agskill", skill_input: "agdata") -> PromptPayload:
        from ..harness import agharness

        return PromptPayload(
            system_instruction=skill._build_system_prompt(),
            user_content=agharness.build_user_turn_prompt(skill, skill_input),
            output_instruction=agharness.build_output_format_instruction(skill),
        )

    def _build_retry_prompt(
        self, missing: "list[str]", *, system_instruction: str = ""
    ) -> PromptPayload:
        return PromptPayload(
            system_instruction=system_instruction,
            user_content=(
                "[HARNESS SYSTEM] You have not yet provided all required output "
                f"fields. Still missing: {missing}. Call the submit_output tool "
                "once for each of them."
            ),
            output_instruction=None,
        )

    def _run_attempt(self, prompt: PromptPayload) -> HarnessAttemptResult:
        interaction = self._host_server_manager.interaction_server
        waiter = interaction.expect_attempt_result()
        try:
            self._send_run_attempt(prompt)
            return interaction.wait_for_attempt_result(waiter)
        except BaseException:
            interaction.cancel_expected_attempt(waiter)
            raise

    def _send_run_attempt(self, prompt: PromptPayload) -> None:
        """Send one attempt directly to the sandbox daemon server.

        The host-side protocol boundary is explicit now; the daemon client
        will implement it once the sandbox-side server exists.
        """
        raise NotImplementedError("sandbox daemon client is not configured")

    def _missing_output_fields(self, skill: "agskill") -> "list[str]":
        if skill.output_schema is None:
            return []
        collected = self._host_server_manager.host_mcp_server.collected_output()
        required = set(skill.output_schema._data.keys())
        return sorted(required - set(collected.keys()))

    def _build_execution_result(
        self, context: "agcontext", skill: "agskill", attempt: "HarnessAttemptResult | None"
    ) -> ExecutionResult:
        raise NotImplementedError
