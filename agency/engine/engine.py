from __future__ import annotations

from typing import TYPE_CHECKING

from .types import CompletedResult

if TYPE_CHECKING:
    from ..agcontext import agcontext
    from ..agdata import agdata
    from ..agent import agent
    from ..agresources import agResourcePool
    from ..agskill import agskill
    from .execution_builder import ExecutionBuilder
    from .sandbox_provisioner import SandboxProvisioner


class agentEngine:
    """Thin public facade for one agent execution.

    The engine is the composition root, not a lifecycle state machine. Its
    sole execution operation delegates the complete transaction to an
    :class:`ExecutionBuilder`.
    """

    def __init__(
        self,
        agent: "agent",
        context: "agcontext",
        skill: "agskill",
        skill_input: "agdata",
        resource_pool: "agResourcePool",
        *,
        execution_builder: "ExecutionBuilder | None" = None,
        sandbox_provisioner: "SandboxProvisioner | None" = None,
    ) -> None:
        self._agent = agent
        self._context = context
        self._skill = skill
        self._skill_input = skill_input
        self._resource_pool = resource_pool

        if execution_builder is None:
            from .execution_builder import ExecutionBuilder
            from .harness_bridge import HarnessManagerBridge
            from .sandbox_provisioner import SandboxProvisioner

            execution_builder = ExecutionBuilder(
                sandbox_provisioner=sandbox_provisioner or SandboxProvisioner(),
                harness_bridge=HarnessManagerBridge(),
            )
        self._execution_builder = execution_builder

    def execute(self) -> CompletedResult:
        """Delegate one complete execution transaction to the builder."""
        return self._execution_builder.execute(
            agent=self._agent,
            context=self._context,
            skill=self._skill,
            skill_input=self._skill_input,
            resource_pool=self._resource_pool,
        )
