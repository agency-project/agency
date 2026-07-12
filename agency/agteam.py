from __future__ import annotations
import functools
import threading
import weakref
from concurrent.futures import Future
from typing import TYPE_CHECKING

from ._context import _active_team
from .agconfig import agConfig

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

        cfg = agConfig()
        cfg.agllm_backend.model = "claude-sonnet-5"
        cfg.agllm_backend.provider = "anthropic"
        cfg.agllm_backend.api_key = os.environ["ANTHROPIC_API_KEY"]

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
    agconfig : agConfig | None
        Default LLM configuration (and any other agconfig-based settings)
        shared by all instances unless overridden at construction time.
        Agents created with no explicit ``agconfig=`` inside ``setup()``/
        ``run()`` inherit this automatically.
    """

    agconfig: "agConfig | None" = None

    # Global weak registry of all live agteam instances.
    _live_teams: "weakref.WeakSet[agteam]" = weakref.WeakSet()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "run" in cls.__dict__:
            _wrap_run(cls)

    def __init__(self, agconfig: "agConfig | None" = None, **config) -> None:
        # Instance-level agconfig: explicit arg > class attribute. Cloned so
        # this team's own agconfig is independent of whatever source it was
        # built from -- mutating that source afterward must not silently
        # change an already-constructed team (or the agents it already spawned).
        _src_agconfig = agconfig if agconfig is not None else type(self).agconfig
        self.agconfig: "agConfig | None" = (
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

        # Log team creation to both terminal and file
        from .agent import agent as _Agent
        from .agname import agname as _agname
        from .aglog import aglog as _aglog
        import sys

        _base = config.get("name") or f"{type(self).__name__}"
        self.team_name: str = _agname.allocate_agname(_base)
        parent_team_name = parent.team_name if parent is not None else None
        log_dir = _Agent.log_dir
        self._log = _aglog(log_dir / "_teams.jsonl" if log_dir else None)
        self._log._lifecycle(
            "created",
            team=self.team_name,
            parent_team=parent_team_name,
        )
        if parent_team_name is not None:
            print(
                f"  [agteam] {self.team_name} created inside {parent_team_name}",
                file=sys.stderr,
                flush=True,
            )

        token = _active_team.set(self)
        try:
            self.setup()
        finally:
            _active_team.reset(token)

        try:
            from . import agwebui as _agwebui

            if _agwebui._active is not None:
                _agwebui._active.emitter.team_registered(
                    self.team_name,
                    [a.agname for a in self._agents],
                )
        except Exception as _e:
            print(f"[agteam] WARNING: team_registered push failed for {self.team_name}: {_e}")

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

    def change_config(self, agconfig: "agConfig") -> None:
        """Replace this team's agconfig with a clone of the given one, and
        push that same clone down to every agent this team has spawned so
        far (via ``agent.change_config``). Agents created afterward pick up
        the new ``self.agconfig`` automatically, the same way they do at
        construction."""
        self.agconfig = agconfig.clone()
        for a in self._agents:
            a.change_config(self.agconfig)

    def get_config_copy(self) -> "agConfig | None":
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
            import sys
            import traceback

            token = _active_team.set(self)
            try:
                result = original(self, *args, **kwargs)
                future.set_result(result if isinstance(result, agdata) else agdata(result=result))
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                future.set_exception(exc)
            finally:
                _active_team.reset(token)

        self._run_future = future
        threading.Thread(target=_task, daemon=True).start()
        return agdata(_future=future)

    cls.run = _async_run
