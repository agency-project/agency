from __future__ import annotations
import functools
import weakref
from concurrent.futures import ThreadPoolExecutor, Future
from typing import TYPE_CHECKING

from ._context import _active_team

if TYPE_CHECKING:
    from .agent import agent as _Agent


class agteam:
    """Base class for a self-contained group of agents, skills, and tools.

    Subclass and override :meth:`setup` to define tools, skills, and agents,
    and :meth:`run` to implement the workflow.  Configuration specific to the
    team (topic, output path, …) is passed as keyword arguments to
    ``__init__`` and becomes instance attributes.

    ``run()`` is always non-blocking — it starts the workflow in a background
    thread and returns a pending :class:`agdata` immediately.  Field access
    on the returned value blocks until the workflow finishes::

        class PaperCrawlerTeam(agteam):
            llm_config = LLM_CONFIG

            def setup(self):
                self.find_papers   = agskill(...)
                self.summarise     = agskill(...)
                self.compile       = agskill(...)
                self.agent         = agent(agskills=[self.find_papers,
                                                     self.summarise,
                                                     self.compile])

            def run(self):
                papers = self.agent.run("find_papers", agdata(topic=self.topic)).papers
                summaries = [agent(self.agent).run("summarise", agdata(**p)) for p in papers]
                return self.agent.run("compile", agdata(summaries=summaries))

        result = PaperCrawlerTeam(topic="KV cache").run()  # returns immediately
        print(result.report_path)                           # blocks here

    Parallel fan-out is just a list comprehension::

        teams   = [PaperCrawlerTeam(topic=t) for t in topics]
        pending = [t.run() for t in teams]          # all start immediately
        for r in pending:
            print(r.report_path)                    # blocks per team as needed

    Class attributes
    ----------------
    llm_config : dict
        Default LLM configuration shared by all instances unless overridden
        at construction time.
    """

    llm_config: dict = {}

    # Shared pool across all agteam instances.
    _pool: ThreadPoolExecutor = ThreadPoolExecutor()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "run" in cls.__dict__:
            _wrap_run(cls)

    def __init__(self, llm_config: dict | None = None, **config) -> None:
        # Instance-level llm_config: explicit arg > class attribute
        self.llm_config: dict = llm_config if llm_config is not None else type(self).llm_config
        # Expose every config kwarg as a plain attribute
        for k, v in config.items():
            setattr(self, k, v)
        self._agents: weakref.WeakSet = weakref.WeakSet()
        self._run_future: Future | None = None
        token = _active_team.set(self)
        try:
            self.setup()
        finally:
            _active_team.reset(token)

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

    @property
    def agents(self) -> list["_Agent"]:
        """All agents tracked by this team (setup + dynamic run-time forks)."""
        return list(self._agents)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(agents={len(self._agents)})"


def _wrap_run(cls) -> None:
    """Replace cls.run with a non-blocking version that runs in a thread pool.

    The wrapped run() sets _active_team for the duration of the background
    task, stores the Future on self._run_future, and returns a pending agdata
    whose fields block until the task completes.
    """
    original = cls.__dict__["run"]

    @functools.wraps(original)
    def _async_run(self, *args, **kwargs):
        from .agdata import agdata
        future: Future = Future()

        def _task() -> None:
            token = _active_team.set(self)
            try:
                result = original(self, *args, **kwargs)
                future.set_result(result if isinstance(result, agdata) else agdata(result=result))
            except Exception as exc:
                future.set_exception(exc)
            finally:
                _active_team.reset(token)

        self._run_future = future
        agteam._pool.submit(_task)
        return agdata(_future=future)

    cls.run = _async_run
