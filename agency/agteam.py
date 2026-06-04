from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import agent as _Agent


class agteam:
    """Base class for a self-contained group of agents, skills, and tools.

    Subclass and override :meth:`setup` to define tools, skills, and agents,
    and :meth:`run` to implement the workflow.  Configuration specific to the
    team (topic, output path, …) is passed as keyword arguments to
    ``__init__`` and becomes instance attributes.

    Example::

        class PaperCrawlerTeam(agteam):
            llm_config = LLM_CONFIG

            def setup(self):
                self.search_arxiv = agtool(...)
                self.find_papers   = agskill(..., tools=[self.search_arxiv])
                self.summarise     = agskill(...)
                self.compile       = agskill(...)
                self.agent         = self.make_agent(
                    [self.find_papers, self.summarise, self.compile]
                )

            def run(self):
                papers = self.agent.run("find_papers", agdata(topic=self.topic)).papers
                ...

        team = PaperCrawlerTeam(topic="KV cache")
        team.run()

    Class attributes
    ----------------
    llm_config : dict
        Default LLM configuration shared by all instances unless overridden
        at construction time.
    """

    llm_config: dict = {}

    def __init__(self, llm_config: dict | None = None, **config) -> None:
        # Instance-level llm_config: explicit arg > class attribute
        self.llm_config: dict = llm_config if llm_config is not None else type(self).llm_config
        # Expose every config kwarg as a plain attribute
        for k, v in config.items():
            setattr(self, k, v)
        self._agents: list[_Agent] = []
        self.setup()

    # ------------------------------------------------------------------
    # Override points
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Define tools, skills, and agents.  Called once at construction."""

    def run(self) -> object:
        """Execute the team's workflow.  Must be overridden by subclasses."""
        raise NotImplementedError(f"{type(self).__name__} must implement run()")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def make_agent(self, agskills, tools=None, **kwargs) -> "_Agent":
        """Create an agent registered with this team.

        Parameters mirror :class:`agent`; ``llm_config`` defaults to
        ``self.llm_config``.  The created agent is appended to
        :attr:`agents` and returned.
        """
        from .agent import agent as _agent_cls
        ag = _agent_cls(
            llm_config=self.llm_config,
            agskills=agskills,
            tools=tools,
            **kwargs,
        )
        self._agents.append(ag)
        return ag

    @property
    def agents(self) -> list["_Agent"]:
        """All agents created by :meth:`make_agent` for this team."""
        return list(self._agents)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(agents={len(self._agents)})"
