from __future__ import annotations
import copy
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
from typing import ClassVar

# Single run-level ID for the default log directory.
_RUN_ID = _uuid_mod.uuid4().hex[:12]
_RUN_TS = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
_DEFAULT_LOG_DIR = Path(f"/tmp/agency/{_RUN_TS}_{_RUN_ID}")

# Global weak registry of all live agent instances.
_live_agents: "weakref.WeakSet[agent]" = weakref.WeakSet()


from .agdata import agdata
from .agcontext import agcontext
from .aglog import aglog, _ts
from .agterm import agterm
from .agsandbox import agSandbox, agSandboxConfig
from .agsandbox_backends import agSandboxBackendConfig
from .agresources import agResourcePool
from .agllm import agllm
from .agconfig import agConfig, DynamicConfigParam, _AgConfigViewBase

from .agname import agname as _agname
from .profiler import agprof


# Exists only to register agent's config fields (via __set_name__ at import
# time). Reads use a throwaway instance -- _AgAgentFields(agconfig) -- since
# these values are needed in a classmethod (load()) and an instance method
# (save()) that doesn't otherwise inherit from this class.
class _AgAgentFields:
    checkpoint_save_timeout_s = DynamicConfigParam("agent", default=600)
    checkpoint_load_timeout_s = DynamicConfigParam("agent", default=600)

    def __init__(self, agconfig=None) -> None:
        self._agconfig = agconfig


class agAgentConfig(_AgConfigViewBase):
    """View over an agConfig for pre-setting agent tunables in one call::

        cfg = agConfig(agAgentConfig(checkpoint_save_timeout_s=300))

    See `_AgConfigViewBase` in agconfig.py for the shared mechanics.
    """

    _OWNER = "agent"


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


# States that mean "this agent's worker thread will not make forward
# progress until something external (a resume, or an upstream producer)
# unblocks it". Used by agent.is_settled() -- the wait_all_* helpers in
# agpause.py never check it directly, they call is_settled() on each agent.
_SETTLED_LEAF_STATES = ("inactive", "finished", "error", "paused")


class agent_state:
    """Single owner of one agent's live status: the display fields a human or
    the webui sees (state/skill/tool), the synchronization primitives pause
    coordination needs (run_allowed/paused_ack/blocked_on), and the lock that
    makes every transition atomic. One instance lives on agent._state.

    Lives here rather than in agpause.py because it's the general-purpose
    status container set on every skill run (every LLM call, every tool
    dispatch) — pausing is just one consumer of it, via run_allowed/
    paused_ack/blocked_on. agpause.py's coordination code (_BlockCtx,
    wait_all_paused, ...) only ever touches these fields through plain
    attribute access on an agent it's given, so it never needs to import
    this class at all.

    update_state() is the *only* way to change the display fields — no
    caller ever takes the lock itself. That matters because a plain "set
    these fields" swap is atomic under the GIL, but a read-then-conditionally
    -write sequence is not: pause()'s "relabel to 'pausing' unless already
    paused/blocked" check used to run outside any lock, so it could read a
    stale "skill" state right before the worker thread's checkpoint wrote
    "paused", then overwrite that "paused" back to "pausing" moments later.
    Folding the guard condition into update_state() itself means the whole
    check-and-set is one atomic operation, and no future caller can
    reintroduce that race by forgetting to lock around its own check.
    """

    def __init__(self, agname: str) -> None:
        self.agname = agname
        self.state: str = "inactive"
        self.skill: "str | None" = None
        self.tool: "str | None" = None
        # Set == allowed to run. Cleared by pause(), set by resume().
        self.run_allowed = threading.Event()
        self.run_allowed.set()
        # Set the moment a worker thread actually blocks at a checkpoint —
        # the real "pause took effect" acknowledgment. is_paused() reads
        # this directly rather than the (cosmetic, racy) state string.
        self.paused_ack = threading.Event()
        # While this agent's worker thread is blocked resolving another
        # agent's pending future, points at that upstream agent so
        # is_settled() can recurse through the dependency chain.
        self.blocked_on: "agent | None" = None
        self._lock = threading.RLock()

    def snapshot(self) -> "tuple[str, str | None, str | None]":
        """Atomically read (state, skill, tool) together — reading the three
        fields one at a time would let a concurrent update_state() call
        interleave between them and hand back a mismatched combination."""
        with self._lock:
            return self.state, self.skill, self.tool

    def update_state(
        self,
        new_state: str,
        skill: "str | None" = None,
        tool: "str | None" = None,
        *,
        unless_in: "tuple[str, ...]" = (),
    ) -> bool:
        """Atomically apply (new_state, skill, tool) unless the current state
        is one of *unless_in* (checked under the same lock as the write, so
        the guard can never race a concurrent transition). Emits to the
        webui outside the lock. Returns False if the guard blocked the write."""
        with self._lock:
            if self.state in unless_in:
                return False
            self.state, self.skill, self.tool = new_state, skill, tool
        self._emit()
        return True

    def _emit(self) -> None:
        try:
            from . import agwebui as _agwebui

            if _agwebui._active is not None:
                from .agterm import agterm as _agterm
                from .agwebui.emitter import ansi_to_hex as _ansi_to_hex
                from ._context import _active_team as _at

                _ansi = _agterm._agname_colors.get(self.agname)
                _color = _ansi_to_hex(_ansi) if _ansi else None
                _team = _at.get(None)
                _team_name = _team.team_name if _team is not None else None
                _agwebui._active.emitter.agent_state(
                    self.agname, self.state, self.skill, self.tool, color=_color, team=_team_name
                )
        except Exception as _e:
            print(f"[agent] WARNING: agent_state push failed for {self.agname}: {_e}")


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
    default_agconfig: "ClassVar[agConfig | None]" = None

    # Global token counter — accumulates across all agents and skill calls.
    _global_input_tokens: ClassVar[int] = 0
    _global_output_tokens: ClassVar[int] = 0
    _global_token_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def _add_global_tokens(cls, inp: int, out: int) -> None:
        with cls._global_token_lock:
            cls._global_input_tokens += inp
            cls._global_output_tokens += out

    @classmethod
    def global_token_usage(cls) -> dict:
        """Framework-wide cumulative token usage across all agents and skill calls."""
        with cls._global_token_lock:
            inp = cls._global_input_tokens
            out = cls._global_output_tokens
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}

    def __init__(
        self,
        agname: str | None = None,
        *,
        llm: "agllm | None" = None,
        sandbox: "agSandbox | None" = None,
        agconfig: "agConfig | None" = None,
    ):
        with agprof.span("agent:create"):
            _src_agconfig = agconfig if agconfig is not None else agent.default_agconfig

            if llm is None:
                if _src_agconfig is None or not _src_agconfig.data.get("agllm_backend"):
                    from ._context import _active_team as _at

                    _t = _at.get(None)
                    if (
                        _t is not None
                        and _t.agconfig is not None
                        and _t.agconfig.data.get("agllm_backend")
                    ):
                        # Adopt the team's agconfig outright (not just for the LLM
                        # fields) -- log_dir/output_dir/sandbox settings etc. should
                        # also come from it, matching "agents inherit the team's
                        # agconfig automatically" (see agteam's docstring).
                        _src_agconfig = _t.agconfig
                    else:
                        raise TypeError(
                            "agent() requires an agconfig with LLM fields set "
                            "(e.g. cfg.agllm_backend.model = ...), or llm=, "
                            "when called outside an agteam context"
                        )

            # Cloned so this agent's own agconfig is independent of whatever
            # source it was built from (an explicit agconfig=, agent.default_agconfig,
            # or the active agteam's agconfig) -- mutating that source afterward
            # must not silently change an already-constructed agent. Use
            # ag.change_config(new_cfg) to change it live -- see that method.
            self.agconfig: "agConfig | None" = (
                _src_agconfig.clone() if _src_agconfig is not None else None
            )

            self.agname: _agname = _agname.allocate_agname(agname)

            self.llm: agllm = llm if llm is not None else agllm(self.agconfig)
            self.ctx: agcontext = agcontext()
            # Sandbox is created lazily on first skill run; container provisioning
            # is expensive and agents may be constructed without ever running a skill.
            self.sandbox: "agSandbox | None" = sandbox

            _log_dir_val = _classvar_or_agconfig(self.agconfig, "log_dir", agent.log_dir)
            log_dir = Path(_log_dir_val) if _log_dir_val is not None else _DEFAULT_LOG_DIR
            log_path = log_dir / f"{self.agname}_timeline.jsonl"
            self.log = aglog(path=log_path, agconfig=self.agconfig)
            self._full_history: list[dict] = []
            self._full_history_path: Path = log_dir / f"{self.agname}_history.jsonl"
            self._full_history_path.parent.mkdir(parents=True, exist_ok=True)
            self.terminal = agterm(self.agname)

            self._snapshot_messages: list[dict] = []
            self.inbox: queue.Queue[str] = queue.Queue()
            self._state = agent_state(str(self.agname))

            _live_agents.add(self)

            from ._context import _active_team

            _team = _active_team.get(None)
            if _team is not None:
                _team._agents.add(self)

            team_name = _team.team_name if _team is not None else None

            ctx = (
                f"  context={self.llm.context_limit}"
                if self.llm.context_limit
                else "  context=unknown"
            )
            team_tag = f"  team={team_name}" if team_name else ""
            self.terminal.log("CREATED  ", f"model={self.llm.backend.model or '?'}{ctx}{team_tag}")
            self.log._lifecycle(
                "created",
                agname=self.agname,
                team=team_name,
                llm_config={k: v for k, v in self.llm.backend.as_dict().items() if k != "api_key"},
                context_limit=self.llm.context_limit,
            )
            self._emit_config()

    def change_config(self, agconfig: "agConfig") -> None:
        """Replace this agent's agconfig with a clone of the given one, and
        push that same clone down to every sub-object that holds its own
        independent copy (``self.llm`` -- and its backend --, ``self.log``,
        and ``self.sandbox`` if one has been created). Reassigning
        ``self.agconfig`` alone does not reach those clones, so this is the
        supported way to change live config (e.g. ``max_completion_tokens``)
        after construction."""
        self.agconfig = agconfig.clone()
        self.llm.change_config(self.agconfig)
        self.log.change_config(self.agconfig)
        if self.sandbox is not None:
            self.sandbox.change_config(self.agconfig)
        self._emit_config()

    def get_config_copy(self) -> "agConfig | None":
        """Return a clone of this agent's agconfig, or None if it has none."""
        return self.agconfig.clone() if self.agconfig is not None else None

    # ------------------------------------------------------------------
    # Properties
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
    def token_usage(self) -> dict:
        return self.log.token_usage

    @property
    def history(self) -> agdata:
        """Return the current history, blocking until any in-flight task finishes."""
        return agdata(messages=self.ctx.get_resolved_messages())

    @history.setter
    def history(self, value: agdata) -> None:
        self.ctx.set_messages(value._data.get("messages", []))

    @property
    def full_history(self) -> list[dict]:
        """Append-only transcript: every message ever sent/received."""
        return list(self._full_history)

    def set_full_history(self, history: list[dict]) -> None:
        self._full_history = copy.deepcopy(history)

    def reset_full_history(self) -> None:
        self._full_history = []

    # ------------------------------------------------------------------
    # UI / history helpers — called by agskill during execution
    # ------------------------------------------------------------------

    def _append_full_history(self, msg: dict) -> None:
        """Append one message to the append-only full history (thread-safe write)."""
        self._full_history.append(msg)
        with self._full_history_path.open("a") as f:
            f.write(json.dumps(msg) + "\n")
        if "role" not in msg:
            self._push_live_messages(self._snapshot_messages)

    def _set_ui_state(self, state: str, skill: str | None = None, tool: str | None = None) -> None:
        self._state.update_state(state, skill, tool)

    def _emit_config(self) -> None:
        """Push this agent's current dynamic-config snapshot to the webui,
        so its config editor can show/edit it without a round trip into this
        (isolated) execution process. Called on construction and after every
        change_config()."""
        if self.agconfig is None:
            return
        try:
            from . import agwebui as _agwebui

            if _agwebui._active is not None:
                _agwebui._active.emitter.agent_config(self.agname, self.agconfig.dynamic_snapshot())
        except Exception as _e:
            print(f"[agent] WARNING: agent_config push failed for {self.agname}: {_e}")

    def _push_live_messages(self, messages: list) -> None:
        self._snapshot_messages = list(messages)
        try:
            from . import agwebui as _agwebui

            if _agwebui._active is not None:
                event_entries = [e for e in getattr(self, "_full_history", []) if "role" not in e]
                _agwebui._active.emitter.push_messages(self.agname, list(messages) + event_entries)
        except Exception as _e:
            print(f"[agent] WARNING: push_messages failed for {self.agname}: {_e}")

    def _next_inbox_msg(self) -> "str | None":
        """Return the next message from the inbox, or None if empty."""
        try:
            return self.inbox.get_nowait()
        except queue.Empty:
            return None

    def _drain_inbox(self, messages: list) -> bool:
        """Drain pending inbox messages into the conversation. Returns True if any were appended."""
        had_inbox = False
        while True:
            msg = self._next_inbox_msg()
            if msg is None:
                break
            inbox_msg = {"role": "user", "content": msg}
            messages.append(inbox_msg)
            had_inbox = True
            if self._push_live_messages:
                self._push_live_messages(messages[1:])
            if self._append_full_history:
                self._append_full_history(inbox_msg)
        return had_inbox

    # ------------------------------------------------------------------
    # Pause / resume
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Request that this agent stop at its next safe checkpoint (the top
        of its ReAct loop, or before one starts). Non-blocking — the agent
        may still be mid-LLM-call or mid-tool-call for a while after this
        returns. Use agpause.wait_all_paused([...]) to confirm it actually
        stopped."""
        self._state.run_allowed.clear()
        self._state.paused_ack.clear()
        # Relabel to "pausing" unless already paused/blocked/terminal --
        # update_state()'s unless_in guard makes this check-and-set atomic
        # against the worker thread's own concurrent transitions (e.g. the
        # checkpoint writing "paused"), so this can never clobber a state
        # that already reflects a genuine stop.
        self._state.update_state(
            "pausing",
            skill=self._state.skill,
            tool=self._state.tool,
            unless_in=("inactive", "finished", "error", "paused", "blocked_on_dependency"),
        )
        self.terminal.log("PAUSE ▶  ", "requested")

    def resume(self) -> None:
        """Clear a pause request. Non-blocking — use
        agpause.wait_all_resumed([...]) to confirm execution actually
        continued past the checkpoint."""
        self._state.run_allowed.set()
        self.terminal.log("PAUSE ✓  ", "resumed")

    def is_paused(self) -> bool:
        """True once this agent has actually stopped at its checkpoint (not
        merely requested — see pause()). Reads the real synchronization
        primitive directly rather than the (cosmetic, independently-settable)
        display state string."""
        return self._state.paused_ack.is_set()

    def is_settled(self, _seen: "set[str] | None" = None) -> bool:
        """True if this agent is not making forward progress right now:
        either it's paused/inactive/finished/errored, or its worker thread is
        transitively blocked waiting on an upstream agent that is itself
        settled. The recursive case is what lets wait_all_paused() confirm a
        whole dependency chain has stopped instead of deadlocking on an agent
        that will never reach its own checkpoint because an upstream
        producer it's waiting on is paused first.

        Checks _state.blocked_on first, ahead of the display state string: a
        pause() request can legitimately relabel the display state (e.g. to
        "pausing") while the agent is still parked inside a blocking
        future.result() call — blocked_on is the reliable signal for that,
        independent of whatever cosmetic label the state string carries."""
        producer = self._state.blocked_on
        if producer is not None:
            _seen = _seen if _seen is not None else set()
            if producer.agname in _seen:
                return True  # cycle guard — shouldn't happen, but never hang on one
            _seen.add(self.agname)
            return producer.is_settled(_seen)
        return self._state.state in _SETTLED_LEAF_STATES

    def _check_pause(self, skill: "str | None" = None, tool: "str | None" = None) -> None:
        """Checkpoint: block here while a pause is in effect. Called once
        before the ReAct loop starts and again at the top of every iteration
        — never mid-LLM-call or mid-tool-call, so an in-flight call always
        finishes before a pause takes effect."""
        if self._state.run_allowed.is_set():
            return
        prev_state, prev_skill, prev_tool = self._state.snapshot()
        self._state.update_state("paused", skill=skill, tool=tool)
        self._state.paused_ack.set()
        self._state.run_allowed.wait()
        self._state.paused_ack.clear()
        if prev_state in (None, "pausing", "paused", "blocked_on_dependency"):
            prev_state = "skill"
        self._state.update_state(prev_state, skill=prev_skill, tool=prev_tool)

    def push_token_count_update_to_ui(self, skill_inp: int, skill_out: int) -> None:
        """Push a live token update to the webui (called from agskill mid-loop)."""
        try:
            from . import agwebui as _agwebui

            if _agwebui._active is None:
                return
            _gl = agent.global_token_usage()
            _before = self.log.token_usage
            _agwebui._active.emitter.token_update(
                self.agname,
                _before["input_tokens"] + skill_inp,
                _before["output_tokens"] + skill_out,
                _gl["input_tokens"],
                _gl["output_tokens"],
            )
        except Exception as _e:
            print(f"[agent] WARNING: live token_update push failed for {self.agname}: {_e}")

    # ------------------------------------------------------------------
    # Execution — delegates to agskill
    # ------------------------------------------------------------------

    def run(self, skill, skill_input: agdata, max_steps: "int | None" = None) -> agdata:
        """Submit the skill and return a pending agdata immediately.

        Delegates all threading, sandboxing, and execution to skill.run(self, ...).
        Calls on the same agent are serialized via the context future chain.
        """
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
            self.terminal.log("DESTROYED", "")
            self.log._lifecycle("destroyed", agname=self.agname)
        except Exception as _e:
            print(f"[agent] WARNING: __del__ log failed for {getattr(self, 'agname', '?')}: {_e}")

    # ------------------------------------------------------------------
    # Fork
    # ------------------------------------------------------------------

    @classmethod
    def fork(cls, src: "agent", agname: str | None = None) -> "agent":
        """Return an independent agent forked from *src*."""
        ag: agent = cls.__new__(cls)
        ag.agname = _agname.allocate_agname(agname)
        # Cloned so the fork's own agconfig is independent of src's -- see
        # the matching comment in __init__.
        ag.agconfig = src.agconfig.clone() if src.agconfig is not None else None
        ag.llm = agllm(ag.agconfig)
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
        log_path = log_dir / f"{ag.agname}_timeline.jsonl"
        ag.log = aglog(path=log_path, agconfig=ag.agconfig)
        ag._full_history = []
        ag._full_history_path = log_dir / f"{ag.agname}_history.jsonl"
        ag._full_history_path.parent.mkdir(parents=True, exist_ok=True)
        ag.terminal = agterm(ag.agname)
        ag._snapshot_messages = []
        ag.inbox = queue.Queue()
        ag._state = agent_state(str(ag.agname))
        _live_agents.add(ag)

        from ._context import _active_team

        _team = _active_team.get(None)
        if _team is not None:
            _team._agents.add(ag)
        team_name = _team.team_name if _team is not None else None

        ag.terminal.log("FORKED   ", f"from {src.agname}")
        ag.log._lifecycle(
            "forked",
            agname=ag.agname,
            parent_agname=src.agname,
            team=team_name,
            llm_config={k: v for k, v in ag.llm.backend.as_dict().items() if k != "api_key"},
        )
        ag._emit_config()
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
                existing.terminal.log(
                    "CKPT     ", f"load_all: {agname} already live, skipping {ckpt.name}"
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

        if self.ctx.is_pending():
            self.terminal.log("CKPT ⏳  ", "waiting for in-flight task to complete...")
        self.ctx.resolve_prev_dependencies()

        state = {
            "agname": self.agname,
            "llm_config": {k: v for k, v in self.llm.backend.as_dict().items() if k != "api_key"},
            "history": self.ctx.messages,
            "ts": _ts(),
        }
        if self.sandbox is not None and self.sandbox._checkpoint_image is not None:
            # Recorded so load() knows which backend's image format
            # container.tar is in -- a chroot snapshot directory and a
            # docker/podman image tag are unrelated formats.
            state["sandbox_image_kind"] = self.sandbox.image_kind
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
        self.terminal.log("CKPT ✓   ", f"saved → {path}  ({size_kb} KB)")
        self.log._lifecycle("saved", agname=self.agname, path=str(path), size_kb=size_kb)

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
        _base_agconfig = agconfig if agconfig is not None else agent.default_agconfig
        ag.agconfig = _base_agconfig.clone() if _base_agconfig is not None else agConfig()
        _already_set = (
            _base_agconfig.data.get("agllm_backend", {}) if _base_agconfig is not None else {}
        )
        for k, v in state.get("llm_config", {}).items():
            if k not in _already_set:
                ag.agconfig.set("agllm_backend", k, v)
        ag.llm = agllm(ag.agconfig)
        ag.ctx = agcontext(messages=list(state.get("history", [])))
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
        ag.log = aglog(path=log_dir / f"{ag.agname}_timeline.jsonl", agconfig=ag.agconfig)
        ag._full_history: list[dict] = []
        ag._full_history_path: Path = log_dir / f"{ag.agname}_history.jsonl"
        ag._full_history_path.parent.mkdir(parents=True, exist_ok=True)
        ag.terminal = agterm(ag.agname)
        ag._snapshot_messages: list[dict] = []
        ag.inbox: queue.Queue = queue.Queue()
        ag._state = agent_state(str(ag.agname))

        _live_agents.add(ag)

        ag.terminal.log("LOADED   ", f"from {path}")
        ag.log._lifecycle(
            "loaded", agname=ag.agname, source=str(path), checkpoint_ts=state.get("ts")
        )
        ag._emit_config()

        return ag

    def __repr__(self) -> str:
        return f"agent(agname={self.agname!r})"
