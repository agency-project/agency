from __future__ import annotations
import functools
import weakref
from concurrent.futures import Future
from contextvars import ContextVar
from typing import TYPE_CHECKING

from .configs.agconfig import agconfig as agconfig_cls
from .observability.profiler import agprof

if TYPE_CHECKING:
    from .agent import agent as _Agent

# Set to the active agteam instance while its run() method is executing.
# Used by agent.__init__ to auto-register fork agents with the enclosing team.
_active_team: ContextVar = ContextVar("_active_team", default=None)


class agteam:
    """Base class for a self-contained group of agents, skills, and tools.

    Subclass and override :meth:`setup` to define tools, skills, and agents,
    and :meth:`run` to implement the workflow.  Configuration specific to the
    team (topic, output path, …) is passed as keyword arguments to
    ``__init__`` and becomes instance attributes.

    ``run()`` is always non-blocking — it starts the workflow in a background
    thread and returns a pending :class:`agdata` immediately.  Field access
    on the returned value blocks until the workflow finishes::

        from agency.configs.agconfig import agconfig, llmconfig

        cfg = agconfig(
            llmconfig(
                model="claude-sonnet-5",
                provider="anthropic",
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )
        )

        class PaperCrawlerTeam(agteam):
            agconfig = cfg

            def setup(self):
                self.find_papers   = agskill(...)
                self.summarise     = agskill(...)
                self.compile       = agskill(...)
                self.agent         = agent()

            def run(self):
                papers = self.agent.run(self.find_papers, agdata(topic=self.topic)).papers
                summaries = [agent.fork(self.agent).run(self.summarise, agdata(**p)) for p in papers]
                return self.agent.run(self.compile, agdata(summaries=summaries))

        result = PaperCrawlerTeam(topic="KV cache").run()  # returns immediately
        print(result.report_path)                           # blocks here

    Parallel fan-out is just a list comprehension::

        teams   = [PaperCrawlerTeam(topic=t) for t in topics]
        pending = [t.run() for t in teams]          # all start immediately
        for r in pending:
            print(r.report_path)                    # blocks per team as needed

    Class attributes
    ----------------
    agconfig : agconfig_cls | None
        Default LLM configuration (and any other agconfig-based settings)
        shared by all instances unless overridden at construction time.
        Agents created with no explicit ``agconfig=`` inside ``setup()``/
        ``run()`` inherit this automatically.
    """

    agconfig: "agconfig_cls | None" = None

    # Global weak registry of all live agteam instances.
    _live_teams: "weakref.WeakSet[agteam]" = weakref.WeakSet()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "run" in cls.__dict__:
            _wrap_run(cls)

    def __init__(self, agconfig: "agconfig_cls | None" = None, **config) -> None:
        # Instance-level agconfig: explicit arg > class attribute. Cloned so
        # this team's own agconfig is independent of whatever source it was
        # built from -- mutating that source afterward must not silently
        # change an already-constructed team (or the agents it already spawned).
        _src_agconfig = agconfig if agconfig is not None else type(self).agconfig
        self.agconfig: "agconfig_cls | None" = (
            _src_agconfig.clone() if _src_agconfig is not None else None
        )
        # Expose every config kwarg as a plain attribute
        for k, v in config.items():
            setattr(self, k, v)
        self._agents: weakref.WeakSet = weakref.WeakSet()
        self._run_future: Future | None = None
        agteam._live_teams.add(self)

        parent = _active_team.get(None)
        self._parent_team: "agteam | None" = parent

        from .orchestrator import get_orchestrator
        from .agname import agname as _agname

        _base = config.get("name") or f"{type(self).__name__}"
        self.team_name: str = _agname.allocate_agname(_base, prefix="team")
        parent_team_name = parent.team_name if parent is not None else None

        self.data_logger = get_orchestrator(self.agconfig).data_logger
        self.data_logger.record_event(
            type="team_created",
            payload={"team": self.team_name, "parent_team": parent_team_name},
            name=self.team_name,
            object="agteam",
            term_message=f"[{self.team_name}] CREATED  parent={parent_team_name}",
        )

        token = _active_team.set(self)
        try:
            self.setup()
        finally:
            _active_team.reset(token)

        self.data_logger.record_event(
            type="team_registered",
            payload={"team_name": self.team_name, "agents": [a.agname for a in self._agents]},
            name=self.team_name,
            object="agteam",
            update_latest_snapshot=True,
        )

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

    def change_config(self, agconfig: "agconfig_cls") -> None:
        """Replace this team's agconfig with a clone of the given one, and
        push that same clone down to every agent this team has spawned so
        far (via ``agent.change_config``). Agents created afterward pick up
        the new ``self.agconfig`` automatically, the same way they do at
        construction."""
        self.agconfig = agconfig.clone() if agconfig is not None else agconfig_cls()
        for a in self._agents:
            a.change_config(self.agconfig)

    def get_config_copy(self) -> "agconfig_cls | None":
        """Return a clone of this team's agconfig, or None if it has none."""
        return self.agconfig.clone() if self.agconfig is not None else None

    @property
    def agents(self) -> list["_Agent"]:
        """All agents tracked by this team (setup + dynamic run-time forks)."""
        return list(self._agents)

    @classmethod
    def all(cls) -> "list[agteam]":
        """Return all currently live agteam instances in this process."""
        return list(cls._live_teams)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(agents={len(self._agents)})"


def _wrap_run(cls) -> None:
    """Replace cls.run with a non-blocking version that runs in a daemon thread.

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
            import traceback

            token = _active_team.set(self)
            try:
                result = original(self, *args, **kwargs)
                if not isinstance(result, agdata):
                    as_pending = getattr(result, "_as_pending_agdata", None)
                    if callable(as_pending):
                        result = as_pending()
                future.set_result(result if isinstance(result, agdata) else agdata(result=result))
            except Exception as exc:
                tb = traceback.format_exc()
                self.data_logger.record_event(
                    type="team_run_failed",
                    payload={"team": self.team_name, "traceback": tb},
                    name=self.team_name,
                    object="agteam",
                    term_message=tb,
                )
                future.set_exception(exc)
            finally:
                _active_team.reset(token)

        self._run_future = future
        agprof.spawn_traced(_task).start()
        return agdata(_future=future)

    cls.run = _async_run
