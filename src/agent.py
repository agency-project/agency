from __future__ import annotations
from .agdata import agdata
from .agskill import agskill
from .tool import tool


class agent:
    """Orchestrator that maintains shared history and delegates to named agskills.

    An agent holds:
    - llm_config  : provider/model settings passed to every agskill
    - history     : shared conversation context (agdata with a 'messages' list)
    - tools       : pool of tool objects available to agskills by default
    - agskills     : list of agskill skills the user can invoke by name

    Calling agent.run(skill_name, input) finds the named agskill, runs its
    internal ReAct loop, updates the shared history, and returns the result.
    """

    def __init__(
        self,
        llm_config: dict,
        agskills: list[agskill] | None = None,
        tools: list[tool] | None = None,
    ):
        self.llm_config = llm_config
        self.agskills: list[agskill] = agskills or []
        self.tools: list[tool] = tools or []
        self.history = agdata(messages=[])

    def run(self, func_name: str, input: agdata, max_steps: int = 10) -> agdata:
        """Invoke the named agskill and return its result."""
        af = next((f for f in self.agskills if f.name == func_name), None)
        if af is None:
            return agdata(error=f"agskill not found: {func_name!r}")

        result, updated_history = af.run(
            self.llm_config, input, self.history, self.tools, max_steps
        )
        self.history = updated_history
        return result

    def __repr__(self) -> str:
        names = [f.name for f in self.agskills]
        return f"agent(agskills={names!r})"
