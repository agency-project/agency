from __future__ import annotations

from typing import TYPE_CHECKING

from .host_servers.host_server_manager import HostServerManager
from .types import ExecutionResult, HarnessAttemptResult, PromptPayload

if TYPE_CHECKING:
    from ..agconfig import agConfig
    from ..agcontext import agcontext
    from ..agdata import agdata
    from ..agent import agent
    from ..agresources import agResourcePool
    from ..agskill import agskill


class agentEngine:
    def __init__(self, agent: "agent") -> None:
        self._agent = agent
        self._host_server_manager: "HostServerManager | None" = None

    def set_config(self, agconfig: "agConfig") -> None:
        self._agent.change_config(agconfig)
        if self._host_server_manager is not None:
            self._host_server_manager.set_config(agconfig)

    @property
    def host_server_manager(self) -> "HostServerManager":
        if self._host_server_manager is None:
            raise RuntimeError("run() has not built a HostServerManager yet")
        return self._host_server_manager

    def run(
        self,
        context: "agcontext",
        skill: "agskill",
        skill_input: "agdata",
        resource_pool: "agResourcePool",
    ) -> ExecutionResult:
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
                attempt = self._host_server_manager.harness_interaction_server.run_prompt(prompt)
                if not attempt.ok:
                    break
                missing = self._missing_output_fields(skill)
                if not missing or retries_left <= 0:
                    break
                retries_left -= 1
                prompt = self._build_retry_prompt(missing)
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
            prompt=agharness.build_user_turn_prompt(skill, skill_input),
            output_format_instruction=agharness.build_output_format_instruction(skill),
            extra_system=None,
        )

    def _build_retry_prompt(self, missing: "list[str]") -> PromptPayload:
        return PromptPayload(
            prompt=(
                "[HARNESS SYSTEM] You have not yet provided all required output "
                f"fields. Still missing: {missing}. Call the submit_output tool "
                "once for each of them."
            ),
            output_format_instruction=None,
            extra_system=None,
        )

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
