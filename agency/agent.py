from __future__ import annotations
import io
import json
import os
import queue
import tarfile
import threading
import uuid as _uuid_mod
import weakref
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from .agutil import agency_tmp_root as _agency_tmp_root

# Single run-level ID for the default log directory.
# [REFACTOR] combine with the RUN_ID in agcontainer
_RUN_ID = _uuid_mod.uuid4().hex[:12]
_RUN_TS = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Same root as the UDS gateway (see agutil.agency_tmp_root for why it is
# hardcoded rather than following $TMPDIR) -- one location policy for every
# host-side runtime path agency owns, instead of logs and sockets diverging.
_DEFAULT_LOG_DIR = _agency_tmp_root() / f"{_RUN_TS}_{_RUN_ID}"

# Global weak registry of all live agent instances.
_live_agents: "weakref.WeakSet[agent]" = weakref.WeakSet()


def _llm_config_snapshot(agconfig: "agConfig") -> dict:
    """Backend config fields for logging/checkpointing, minus the secret
    api_key -- a fresh, cheap (no network I/O) construction each call, not a
    persisted instance."""
    return {k: v for k, v in agllm.for_config(agconfig).as_dict().items() if k != "api_key"}


from .agdata import agdata
from .agcontext import agcontext
from .agDataCollector import _ts
from .agcollector import AgentDataCollectorFacade
from .sandbox.agsandbox import agSandbox, agSandboxConfig
from .sandbox import agSandboxBackendConfig
from .agresources import agResourcePool
from .llm.agllm import agllm
from .agconfig import agConfig, DynamicConfigParam, _AgConfigViewBase

from .agname import agname as _agname  # [REFACTOR] Why underscore?
from .profiler import agprof

if TYPE_CHECKING:
    from .engine import AgentEngine


# Exists only to register agent's config fields (via __set_name__ at import
# time). Reads use a throwaway instance -- _AgAgentFields(agconfig) -- since
# these values are needed in a classmethod (load()) and an instance method
# (save()) that doesn't otherwise inherit from this class.
class _AgAgentFields:
    checkpoint_save_timeout_s = DynamicConfigParam("agent", default=600)
    checkpoint_load_timeout_s = DynamicConfigParam("agent", default=600)
    harness = DynamicConfigParam(
        "agent", default="native"
    )  # Looked up via agharness_backend.for_config() by AgentEngine. "native" runs
    # agency's own react loop as a persistent in-container process
    # (agharness_backends/native.py); any other value names an external
    # harness engine (claude_code/codex/opencode/grok).

    def __init__(self, agconfig=None) -> None:
        self._agconfig = agconfig


class agAgentConfig(_AgConfigViewBase):
    """View over an agConfig for pre-setting agent tunables in one call::

        cfg = agConfig(agAgentConfig(checkpoint_save_timeout_s=300))

    See `_AgConfigViewBase` in agconfig.py for the shared mechanics.
    """

    _OWNER = "agent"


# [REFACTOR] Remove
def _classvar_or_agconfig(agconfig: "agConfig | None", name: str, classvar_default):
    """Resolve one of agent's own knobs (log_dir, output_dir, ...): the plain
    ClassVar default, optionally overridden by agconfig.

    Deliberately NOT a ConfigParam descriptor: agent's docstring documents
    ``agent.log_dir = Path(...)`` as a supported class-level override, and
    assigning to a class attribute that holds a descriptor replaces the
    descriptor itself (silently breaking it for every future instance) --
    so these fields stay plain ClassVars, resolved via this helper instead.
    """
    return classvar_default if agconfig is None else agconfig.get("agent", name, classvar_default)


_SETTLED_LEAF_STATES = ("inactive", "finished", "error", "paused")


class agent_state:
    """Atomically owned live state used by orchestration and inspection."""

    def __init__(self, agname: str) -> None:
        self.agname = agname
        self.state = "inactive"
        self.skill: "str | None" = None
        self.tool: "str | None" = None
        self.blocked_on: "agent | None" = None
        self._lock = threading.RLock()

    def snapshot(self) -> "tuple[str, str | None, str | None]":
        with self._lock:
            return self.state, self.skill, self.tool

    def update_state(
        self, state: str, skill: "str | None" = None, tool: "str | None" = None
    ) -> None:
        with self._lock:
            self.state, self.skill, self.tool = state, skill, tool


# [REFACTOR] Check how it works
class agent:
    """Orchestrator that maintains shared history and delegates to named agskills.

    An agent holds all runtime state (LLM config, conversation context, sandbox,
    logging infrastructure) and delegates execution to agskill objects.

    agent.run(skill, input)
        Always non-blocking.  Returns a pending agdata immediately.  Calls on
        the same agent are serialized through the history chain.  Calls on
        different agents (forks) run concurrently.

    agent.fork(existing_agent)
        Blocks until the source agent's in-flight task completes, then
        deep-copies the resolved history and snapshots the parent's container.

    Class-level configuration (set once before creating agents)::

        agent.log_dir        = Path("runs/logs")
        agent.agresource_pool  = agResourcePool()
        agent.ping_interval_s = 300
        agent.poll_interval_s = 5
        agent.max_outer_iters = 144
    """

    # [REFACTOR] Move to agconfig
    log_dir: ClassVar[Path | None] = None
    output_dir: ClassVar[Path | None] = None
    agresource_pool: ClassVar[agResourcePool] = agResourcePool(mark_gpus=False)
    ping_interval_s: ClassVar[int] = 300
    poll_interval_s: ClassVar[int] = 5
    max_outer_iters: ClassVar[int] = 144

    # Tier-1-style fallback: agent(agconfig=...) not given -> use this if set.
    # Same "set once before creating agents" convention as the ClassVars
    # above, so scripts that construct agents directly (agent(agname=...),
    # with no agconfig= kwarg) still pick up a run-wide agConfig.
    default_agconfig: "ClassVar[agConfig | None]" = None  # [REFACTOR] Remove

    def __init__(
        self,
        agname: str | None = None,
        *,
        sandbox: "agSandbox | None" = None,
        agconfig: "agConfig | None" = None,
        harness: "str | None" = None,
    ):
        with agprof.span("agent:create"):
            self._initialize(agname, sandbox, agconfig, harness)

    # [REFACTOR] Why separate?
    def _initialize(
        self,
        agname: "str | None",
        sandbox: "agSandbox | None",
        agconfig: "agConfig | None",
        harness: "str | None",
    ) -> None:
        _src_agconfig = agconfig if agconfig is not None else agent.default_agconfig

        if _src_agconfig is None or not _src_agconfig.data.get("agllm_backend"):
            from ._context import _active_team as _at

            _t = _at.get(None)
            if (
                _t is not None and _t.agconfig is not None and _t.agconfig.data.get("agllm_backend")
            ):  # [REFACTOR] Why do we have auto team-config inheritance only when agllm_backend exists?
                # Adopt the team's agconfig outright (not just for the LLM
                # fields) -- log_dir/output_dir/sandbox settings etc. should
                # also come from it, matching "agents inherit the team's
                # agconfig automatically" (see agteam's docstring).
                _src_agconfig = _t.agconfig
            else:
                raise TypeError(
                    "agent() requires an agconfig with LLM fields set "
                    "(e.g. cfg.agllm_backend.model = ...) when called "
                    "outside an agteam context"
                )

        # Cloned so this agent's own agconfig is independent of whatever
        # source it was built from (an explicit agconfig=, agent.default_agconfig,
        # or the active agteam's agconfig) -- mutating that source afterward
        # must not silently change an already-constructed agent. Use
        # ag.change_config(new_cfg) to change it live -- see that method.
        self.agconfig: "agConfig | None" = (
            _src_agconfig.clone() if _src_agconfig is not None else None
        )  # [REFACTOR] Should use a single setter method, also no default agconfig

        self.agname: _agname = _agname.allocate_agname(agname)
        self._parent_agent_id: "str | None" = (
            None  # [REFACTOR]  Why do we need to keep reference of parent agent id?
        )

        self.harness: str = (
            harness if harness is not None else _AgAgentFields(self.agconfig).harness
        )  # [REFACTOR] Change to config only
        self.ctx: agcontext = agcontext()
        # Sandbox is created lazily on first skill run; container provisioning
        # is expensive and agents may be constructed without ever running a skill.
        self.sandbox: "agSandbox | None" = sandbox
        self.engine: "AgentEngine | None" = None

        _log_dir_val = _classvar_or_agconfig(self.agconfig, "log_dir", agent.log_dir)
        log_dir = Path(_log_dir_val) if _log_dir_val is not None else _DEFAULT_LOG_DIR

        self.data_collector = AgentDataCollectorFacade(
            agname=str(self.agname),
            default_db_path=log_dir / "agency.sqlite3",
        )
        self._state = agent_state(str(self.agname))
        self.inbox: queue.Queue[str] = queue.Queue()

        _live_agents.add(self)

        from ._context import _active_team

        _team = _active_team.get(None)
        if _team is not None:
            _team._agents.add(self)

        team_name = _team.team_name if _team is not None else None

        _llm_config = _llm_config_snapshot(self.agconfig)
        _context_limit = _llm_config.get("context_limit")
        ctx = f"  context={_context_limit}" if _context_limit else "  context=unknown"
        team_tag = f"  team={team_name}" if team_name else ""
        self.data_collector.record_event(
            type="agent_created",
            payload={
                "agname": self.agname,
                "team": team_name,
                "llm_config": _llm_config,
                "context_limit": _context_limit,
            },
            term_message=(
                f"[{self.agname}] CREATED  model={_llm_config.get('model') or '?'}{ctx}{team_tag}"
            ),
        )
        self._set_ui_state("inactive")
        self.change_config(self.agconfig)

    def change_config(self, agconfig: "agConfig") -> None:
        """Replace this agent's agconfig with a clone of the given one, and
        push that same clone down to every sub-object that holds its own
        independent copy (``self.data_collector``, ``self.sandbox`` if one
        has been created, and ``self.engine``). Reassigning ``self.agconfig``
        alone does not reach those clones, so this is the supported way to
        change live config (e.g. ``max_completion_tokens``) after
        construction -- also called by __init__/fork/load themselves, once
        their sub-objects exist, so config-snapshot recording lives in one
        place."""
        self.agconfig = agconfig.clone()
        self.data_collector.set_config(self.agconfig)
        if self.sandbox is not None:
            self.sandbox.change_config(self.agconfig)
        if self.engine is not None:
            self.engine.set_config(self.agconfig)
        self.data_collector.record_event(
            type="agent_config",
            payload=self.agconfig.dynamic_snapshot(),
            do_update=True,
        )

    def get_config_copy(self) -> "agConfig | None":
        """Return a clone of this agent's agconfig, or None if it has none."""
        return self.agconfig.clone() if self.agconfig is not None else None

    # ------------------------------------------------------------------
    # Properties # [REFACTOR] Why as properties?
    # ------------------------------------------------------------------

    @property
    def output_path(self) -> Path | None:
        out_dir = _classvar_or_agconfig(self.agconfig, "output_dir", agent.output_dir)
        if out_dir is None:
            return None
        return Path(out_dir) / self.agname

    @property
    def container_output_path(self) -> str | None:
        out_dir = _classvar_or_agconfig(self.agconfig, "output_dir", agent.output_dir)
        if out_dir is None:
            return None
        return f"/agent_output/{self.agname}"

    @property
    def history(self) -> agdata:
        """Return the committed transcript after this agent becomes idle."""
        from .orchestrator import peek_orchestrator

        orchestrator = peek_orchestrator()
        if orchestrator is not None:
            orchestrator.wait_for_agent(self)
        return agdata(messages=self.ctx.get_resolved_transcript())

    @history.setter
    def history(self, value: agdata) -> None:
        from .orchestrator import peek_orchestrator

        orchestrator = peek_orchestrator()
        if orchestrator is not None:
            orchestrator.wait_for_agent(self)
        self.ctx.set_transcript(value._data.get("messages", []))

    def _set_ui_state(
        self, state: str, skill: "str | None" = None, tool: "str | None" = None
    ) -> None:
        self._state.update_state(state, skill, tool)
        self.data_collector.record_event(
            type="agent_state",
            payload={"state": state, "skill": skill, "tool": tool},
            do_update=True,
        )
        try:
            from . import agwebui as _agwebui

            if _agwebui._active is not None:
                _agwebui._active.emitter.agent_state(self.agname, state, skill, tool)
        except Exception as exc:
            print(f"[agent] WARNING: agent_state push failed for {self.agname}: {exc}")

    def is_paused(self) -> bool:
        return self._state.state == "paused"

    def is_settled(self, _seen: "set[str] | None" = None) -> bool:
        from .orchestrator import peek_orchestrator

        orchestrator = peek_orchestrator()
        if orchestrator is not None and not orchestrator.is_agent_idle(self):
            return orchestrator.is_agent_settled(self)
        producer = self._state.blocked_on
        if producer is not None:
            seen = _seen if _seen is not None else set()
            if str(producer.agname) in seen:
                return True
            seen.add(str(self.agname))
            return producer.is_settled(seen)
        return self._state.state in _SETTLED_LEAF_STATES

    # ------------------------------------------------------------------
    # UI / history helpers — called by agskill during execution
    # ------------------------------------------------------------------

    def _next_inbox_msg(self) -> "dict | None":
        """Return the next typed inbox entry, or None if empty."""
        try:
            return self.inbox.get_nowait()
        except queue.Empty:
            return None

    def _drain_inbox(self, messages: list) -> bool:
        """Drain pending typed inbox entries. Returns True if any were appended."""
        had_inbox = False
        while True:
            msg = self._next_inbox_msg()
            if msg is None:
                break
            messages.append(msg)
            had_inbox = True
        return had_inbox

    # ------------------------------------------------------------------
    # Pause / resume
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Request that this agent's harness manager stop at its next safe
        checkpoint. Non-blocking — delivered as an inbox entry the harness
        manager drains via check_inbox()."""
        self.inbox.put({"type": "pause"})
        self.data_collector.record_event(
            type="agent_pause_requested",
            payload={"agname": self.agname},
            term_message=f"[{self.agname}] PAUSE ▶  requested",
        )

    def resume(self) -> None:
        """Clear a pause request. Non-blocking — delivered as an inbox entry
        the harness manager drains via check_inbox()."""
        self.inbox.put({"type": "resume"})
        self.data_collector.record_event(
            type="agent_resumed",
            payload={"agname": self.agname},
            term_message=f"[{self.agname}] PAUSE ✓  resumed",
        )

    # ------------------------------------------------------------------
    # Execution — delegates to agskill
    # ------------------------------------------------------------------

    def _ensure_sandbox(self) -> agSandbox:
        """Create the sandbox lazily on the admitted engine thread."""
        if self.sandbox is not None:
            return self.sandbox
        sandbox_config = self.agconfig
        agent_output_dir = self.output_path
        if agent_output_dir is not None:
            sandbox_config = sandbox_config.clone() if sandbox_config else agConfig()
            agSandboxConfig(sandbox_config).add_mount(
                "agent_output", agent_output_dir, "/agent_output"
            )
        self.sandbox = agSandbox(self.agname, agconfig=sandbox_config)
        return self.sandbox

    def run(self, skill, skill_input: agdata, max_steps: "int | None" = None) -> agdata:
        """Submit a request; admission creates its fresh engine and thread."""
        if max_steps is None:
            return skill.run(self, skill_input)
        return skill.run(self, skill_input, max_steps=max_steps)

    async def asyncio_run(
        self,
        skill,
        skill_input: "agdata",
        max_steps: "int | None" = None,
    ) -> "agdata":
        """Async wrapper around run() for use in asyncio event loops."""
        import asyncio

        loop = asyncio.get_event_loop()
        pending = self.run(skill, skill_input, max_steps)
        await loop.run_in_executor(None, pending._resolve)
        return pending

    # ------------------------------------------------------------------
    # Destructor
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        """Best-effort: log destruction. The sandbox (if any) cleans itself up
        via agSandbox.__del__ once this agent's reference to it is gone."""
        _live_agents.discard(self)
        try:
            self.data_collector.record_event(
                type="agent_state",
                payload={"state": "agent_exit"},
                do_update=True,
                flush=True,
            )
            self.data_collector.record_event(
                type="agent_destroyed",
                payload={"agname": self.agname},
                term_message=f"[{self.agname}] DESTROYED",
                flush=True,
            )
        except Exception as _e:
            print(f"[agent] WARNING: __del__ log failed for {getattr(self, 'agname', '?')}: {_e}")

    # ------------------------------------------------------------------
    # Fork
    # ------------------------------------------------------------------
    # [REFACTOR] Why not call _init()_?
    @classmethod
    def fork(cls, src: "agent", agname: str | None = None) -> "agent":
        """Return an independent agent forked from *src*."""
        ag: agent = cls.__new__(cls)
        ag.agname = _agname.allocate_agname(agname)
        ag._parent_agent_id = str(src.agname)
        # Cloned so the fork's own agconfig is independent of src's -- see
        # the matching comment in __init__.
        ag.agconfig = src.agconfig.clone() if src.agconfig is not None else None
        ag.harness = src.harness
        ag.engine = None
        from .orchestrator import peek_orchestrator

        orchestrator = peek_orchestrator()
        if orchestrator is not None:
            orchestrator.wait_for_agent(src)
        src.ctx.resolve_prev_dependencies()
        ag.ctx = src.ctx.copy()
        _out_dir = _classvar_or_agconfig(ag.agconfig, "output_dir", cls.output_dir)
        _out = Path(_out_dir) / ag.agname if _out_dir else None
        sb_cfg = ag.agconfig
        if _out is not None:
            sb_cfg = sb_cfg.clone() if sb_cfg else agConfig()
            agSandboxConfig(sb_cfg).add_mount("agent_output", _out, "/agent_output")
        ag.sandbox = (
            src.sandbox.fork(ag.agname, agconfig=sb_cfg) if src.sandbox is not None else None
        )
        _log_dir_val = _classvar_or_agconfig(ag.agconfig, "log_dir", cls.log_dir)
        log_dir = Path(_log_dir_val) if _log_dir_val is not None else _DEFAULT_LOG_DIR
        ag.data_collector = AgentDataCollectorFacade(
            agname=str(ag.agname),
            default_db_path=log_dir / "agency.sqlite3",
        )
        ag._state = agent_state(str(ag.agname))
        ag.inbox = queue.Queue()
        _live_agents.add(ag)

        from ._context import _active_team

        _team = _active_team.get(None)
        if _team is not None:
            _team._agents.add(ag)
        team_name = _team.team_name if _team is not None else None

        ag.data_collector.record_event(
            type="agent_forked",
            payload={
                "agname": ag.agname,
                "parent_agname": src.agname,
                "team": team_name,
                "llm_config": _llm_config_snapshot(ag.agconfig),
            },
            term_message=f"[{ag.agname}] FORKED   from {src.agname}",
        )
        ag._set_ui_state("inactive")
        ag.change_config(ag.agconfig)
        return ag

    # ------------------------------------------------------------------
    # Live agent registry
    # ------------------------------------------------------------------

    @classmethod
    def all(cls) -> "list[agent]":
        return list(_live_agents)

    @classmethod
    def save_all(cls, directory: "Path | str") -> "list[Path]":
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for ag in cls.all():
            path = directory / f"{ag.agname}.ckpt"
            ag.save(path)
            paths.append(path)
        return paths

    @classmethod
    def load_all(
        cls,
        directory: "Path | str",
        agconfig: "agConfig | None" = None,
    ) -> "list[agent]":
        directory = Path(directory)
        live_names = {ag.agname: ag for ag in cls.all()}
        restored: list[agent] = []

        for ckpt in sorted(directory.glob("*.ckpt")):
            with tarfile.open(ckpt, "r:gz") as tar:
                state = json.loads(tar.extractfile("state.json").read())
            agname = state["agname"]

            if agname in live_names:
                existing = live_names[agname]
                existing.data_collector.record_event(
                    type="agent_load_skipped",
                    payload={"agname": agname, "ckpt": ckpt.name},
                    term_message=(
                        f"[{agname}] CKPT     load_all: already live, skipping {ckpt.name}"
                    ),
                )
                restored.append(existing)
            else:
                ag = cls.load(ckpt, agconfig=agconfig)
                restored.append(ag)

        return restored

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: "Path | str") -> None:
        """Checkpoint this agent to a single .ckpt file."""
        path = Path(path)
        image_tag = f"agency/ckpt-{self.agname}"

        from .orchestrator import peek_orchestrator

        orchestrator = peek_orchestrator()
        if orchestrator is not None and not orchestrator.is_agent_idle(self):
            self.data_collector.record_event(
                type="agent_checkpoint_waiting",
                payload={"agname": self.agname},
                term_message=f"[{self.agname}] CKPT ⏳  waiting for in-flight task to complete...",
            )
            orchestrator.wait_for_agent(self)
        self.ctx.resolve_prev_dependencies()

        state = {
            "agname": self.agname,
            "parent_agent_id": self._parent_agent_id,
            "harness": self.harness,
            "llm_config": _llm_config_snapshot(self.agconfig),
            "history": self.ctx.recent_transcript,
            "ts": _ts(),
        }
        if self.sandbox is not None and self.sandbox._checkpoint_image is not None:
            # Recorded so load() knows which backend's image format
            # container.tar is in -- a chroot snapshot directory and a
            # docker/podman image tag are unrelated formats.
            state["sandbox_image_kind"] = self.sandbox.image_kind
        if self.ctx.harness_sessions:
            # See docs/Design_harness_history.md -- travels with the
            # agent's own checkpoint, not with container.tar, so it's
            # available regardless of which sandbox this checkpoint is
            # later restored onto.
            state["harness_sessions"] = self.ctx.harness_sessions
        state_bytes = json.dumps(state, indent=2).encode()

        path.parent.mkdir(parents=True, exist_ok=True)

        if self.sandbox is not None and self.sandbox._checkpoint_image is not None:
            backend_cls = type(self.sandbox._backend)
            backend_cls.tag_image(self.sandbox._checkpoint_image, image_tag)
            try:
                _save_timeout = _AgAgentFields(self.agconfig).checkpoint_save_timeout_s
                # Scrub the owning process's PID before embedding -- it's
                # meaningless (and, since a .ckpt file can be restored by
                # an unrelated process on a different host entirely,
                # potentially misleading) once outside this process's own
                # lifetime. load() re-stamps the actually-current
                # restoring process's PID after import. See
                # relabel_owner_pid()'s docstring.
                backend_cls.relabel_owner_pid(image_tag, None, _save_timeout)
                image_bytes = backend_cls.export_image(image_tag, _save_timeout)
                with tarfile.open(path, "w:gz") as tar:
                    for name, data in [("state.json", state_bytes), ("container.tar", image_bytes)]:
                        info = tarfile.TarInfo(name=name)
                        info.size = len(data)
                        tar.addfile(info, io.BytesIO(data))
            finally:
                backend_cls.delete_image(image_tag, force=True)
        else:
            with tarfile.open(path, "w:gz") as tar:
                info = tarfile.TarInfo(name="state.json")
                info.size = len(state_bytes)
                tar.addfile(info, io.BytesIO(state_bytes))

        size_kb = path.stat().st_size // 1024
        self.data_collector.record_event(
            type="agent_saved",
            payload={"agname": self.agname, "path": str(path), "size_kb": size_kb},
            term_message=f"[{self.agname}] CKPT ✓   saved → {path}  ({size_kb} KB)",
        )

    @classmethod
    def load(
        cls,
        path: "Path | str",
        agconfig: "agConfig | None" = None,
    ) -> "agent":
        """Restore an agent from a checkpoint file created by agent.save().

        The checkpointed LLM config (everything except ``api_key``, which
        ``save()`` strips) is merged into ``agconfig``'s ``agllm_backend``
        fields -- a field already set explicitly on ``agconfig`` (e.g.
        ``cfg.agllm_backend.api_key = ...``, to restore the secret ``save()``
        dropped) wins over the checkpointed value.
        """
        path = Path(path)
        image_tag = f"agency/ckpt-restore-{_uuid_mod.uuid4().hex[:8]}"

        with tarfile.open(path, "r:gz") as tar:
            state = json.loads(tar.extractfile("state.json").read())
            container_member = next(
                (m for m in tar.getmembers() if m.name == "container.tar"), None
            )
            image_bytes = tar.extractfile(container_member).read() if container_member else None

        checkpoint: str | None = None
        image_kind = state.get("sandbox_image_kind", "container")
        if image_bytes is not None:
            _load_timeout = _AgAgentFields(agconfig).checkpoint_load_timeout_s
            backend_cls = agSandbox.backend_for_image_kind(image_kind)
            backend_cls.import_image(image_bytes, _load_timeout)
            original_tag = f"agency/ckpt-{state['agname']}"
            backend_cls.tag_image(original_tag, image_tag)
            backend_cls.delete_image(original_tag)
            # Stamp the actually-current restoring process's own PID --
            # save() scrubbed whatever PID this image carried before
            # embedding it (see relabel_owner_pid()'s docstring), so
            # without this the restored image would carry no owner
            # evidence at all, same as a never-labelled image.
            backend_cls.relabel_owner_pid(image_tag, os.getpid(), _load_timeout)
            checkpoint = image_tag

        ag: agent = cls.__new__(cls)
        ag.agname = _agname.claim_unique_agname(state["agname"])
        ag._parent_agent_id = state.get("parent_agent_id")
        _base_agconfig = agconfig if agconfig is not None else agent.default_agconfig
        ag.agconfig = _base_agconfig.clone() if _base_agconfig is not None else agConfig()
        _already_set = (
            _base_agconfig.data.get("agllm_backend", {}) if _base_agconfig is not None else {}
        )
        for k, v in state.get("llm_config", {}).items():
            if k not in _already_set:
                ag.agconfig.set("agllm_backend", k, v)
        # Accept the old checkpoint key so existing snapshots remain loadable.
        ag.harness = state.get("harness", state.get("engine", "native"))
        ag.engine = None
        ag.ctx = agcontext(
            recent_transcript=list(state.get("history", [])),
            harness_sessions=state.get("harness_sessions", {}),
        )
        _out_dir = _classvar_or_agconfig(ag.agconfig, "output_dir", cls.output_dir)
        _out = Path(_out_dir) / ag.agname if _out_dir else None
        sb_cfg = ag.agconfig
        if _out is not None:
            sb_cfg = sb_cfg.clone() if sb_cfg else agConfig()
            agSandboxConfig(sb_cfg).add_mount("agent_output", _out, "/agent_output")
        if checkpoint and image_kind == "chroot":
            # Force the matching backend -- auto-detection (podman/docker
            # preferred when usable) would otherwise reconstruct this
            # sandbox with a backend that can't make sense of a chroot
            # snapshot tag. Container-kind checkpoints don't need this: auto
            # picking podman vs. docker for them was already safe before
            # chroot existed.
            sb_cfg = sb_cfg.clone() if sb_cfg else agConfig()
            agSandboxBackendConfig(sb_cfg).update(backend="chroot")
        ag.sandbox = (
            agSandbox(ag.agname, checkpoint_image=checkpoint, agconfig=sb_cfg)
            if checkpoint
            else None
        )

        _log_dir_val = _classvar_or_agconfig(ag.agconfig, "log_dir", agent.log_dir)
        log_dir = Path(_log_dir_val) if _log_dir_val is not None else _DEFAULT_LOG_DIR
        ag.data_collector = AgentDataCollectorFacade(
            agname=str(ag.agname),
            default_db_path=log_dir / "agency.sqlite3",
        )
        ag._state = agent_state(str(ag.agname))
        ag.inbox: queue.Queue = queue.Queue()

        _live_agents.add(ag)

        ag.data_collector.record_event(
            type="agent_loaded",
            payload={
                "agname": ag.agname,
                "source": str(path),
                "checkpoint_ts": state.get("ts"),
            },
            term_message=f"[{ag.agname}] LOADED   from {path}",
        )
        ag._set_ui_state("inactive")
        ag.change_config(ag.agconfig)

        return ag

    def __repr__(self) -> str:
        return f"agent(agname={self.agname!r})"
