from __future__ import annotations
import copy
import io
import json
import queue
import tarfile
import threading
import uuid as _uuid_mod
import weakref
from datetime import datetime
from pathlib import Path
from typing import ClassVar

# Single run-level ID for the default log directory.
_RUN_ID  = _uuid_mod.uuid4().hex[:12]
_RUN_TS  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
_DEFAULT_LOG_DIR = Path(f"/tmp/agency/{_RUN_TS}_{_RUN_ID}")

# Global weak registry of all live agent instances.
_live_agents: "weakref.WeakSet[agent]" = weakref.WeakSet()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT_SAVE_TIMEOUT_S = 600
CHECKPOINT_LOAD_TIMEOUT_S = 600

from .agdata import agdata
from .agcontext import agcontext
from .aglog import aglog, _ts
from .agterm import agterm
from .agsandbox import agSandbox
from .agresources import agResourcePool
from .agllm import agllm

from .agname import agname as _agname


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

    log_dir:          ClassVar[Path | None]          = None
    output_dir:       ClassVar[Path | None]          = None
    agresource_pool:  ClassVar[agResourcePool]       = agResourcePool(mark_gpus=False)
    ping_interval_s:  ClassVar[int]                  = 300
    poll_interval_s:  ClassVar[int]                  = 5
    max_outer_iters:  ClassVar[int]                  = 144

    # Global token counter — accumulates across all agents and skill calls.
    _global_input_tokens:  ClassVar[int]             = 0
    _global_output_tokens: ClassVar[int]             = 0
    _global_token_lock:    ClassVar[threading.Lock]  = threading.Lock()

    @classmethod
    def _add_global_tokens(cls, inp: int, out: int) -> None:
        with cls._global_token_lock:
            cls._global_input_tokens  += inp
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
        llm_config: "dict | list[dict] | None" = None,
        agname: str | None = None,
        *,
        llm: "agllm | None" = None,
        sandbox: "agSandbox | None" = None,
    ):
        if llm is None:
            if llm_config is None:
                from ._context import _active_team as _at
                _t = _at.get(None)
                if _t is not None:
                    llm_config = _t.llm_config
                else:
                    raise TypeError("agent() requires llm_config or llm= when called outside an agteam context")
            llm_config = agllm.pick_llm_config(llm_config)

        self.agname: _agname = _agname.allocate_agname(agname)

        # True when the sandbox was provided by the caller; agskill will NOT stop
        # or destroy it at the end of skill runs.  Managed via get_agent_sandbox_as_external_sandbox() /
        # set_agent_sandbox_to_external_sandbox() — the sandbox property setter handles cleanup but does not
        # touch this flag, so agskill's internal provisioning leaves it intact.
        self.is_external_sandbox: bool = sandbox is not None

        self.llm: agllm                = llm if llm is not None else agllm(llm_config)
        self.ctx: agcontext            = agcontext()
        # Private backing field — accessed via the sandbox property.
        # Sandbox is created lazily on first skill run; container provisioning
        # is expensive and agents may be constructed without ever running a skill.
        self._sandbox: "agSandbox | None" = sandbox

        log_dir  = Path(agent.log_dir) if agent.log_dir is not None else _DEFAULT_LOG_DIR
        log_path = log_dir / f"{self.agname}_timeline.jsonl"
        self.log  = aglog(path=log_path)
        self._full_history: list[dict] = []
        self._full_history_path: Path = log_dir / f"{self.agname}_history.jsonl"
        self._full_history_path.parent.mkdir(parents=True, exist_ok=True)
        self.terminal = agterm(self.agname)

        self._snapshot_messages: list[dict] = []
        self.inbox: queue.Queue[str] = queue.Queue()
        self._ui_state: dict = {"state": "inactive", "skill": None, "tool": None}

        _live_agents.add(self)

        from ._context import _active_team
        _team = _active_team.get(None)
        if _team is not None:
            _team._agents.add(self)

        team_name = _team.team_name if _team is not None else None

        ctx = f"  context={self.llm.context_limit}" if self.llm.context_limit else "  context=unknown"
        team_tag = f"  team={team_name}" if team_name else ""
        self.terminal.log("CREATED  ", f"model={self.llm.config.get('model','?')}{ctx}{team_tag}")
        self.log._lifecycle(
            "created",
            agname=self.agname,
            team=team_name,
            llm_config={k: v for k, v in self.llm.config.items() if k != "api_key"},
            context_limit=self.llm.context_limit,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def output_path(self) -> Path | None:
        if agent.output_dir is None:
            return None
        return Path(agent.output_dir) / self.agname

    @property
    def container_output_path(self) -> str | None:
        if agent.output_dir is None:
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

    # ------------------------------------------------------------------
    # LLM config helpers
    # ------------------------------------------------------------------

    def set_llm_config(self, llm_config: dict) -> None:
        """Replace the agent's LLM config and refresh the context limit."""
        self.llm = agllm(dict(llm_config))

    @property
    def sandbox(self) -> "agSandbox | None":
        """Read the current sandbox without transferring ownership.

        Use get_agent_sandbox_as_external_sandbox() to take ownership (sets is_external_sandbox=True).
        Assigning via ag.sandbox = sb destroys any existing agent-owned sandbox
        and updates the backing field, but does NOT touch is_external_sandbox —
        agskill uses this path for internal provisioning.
        """
        return self._sandbox

    @sandbox.setter
    def sandbox(self, value: "agSandbox | None") -> None:
        if self._sandbox is not None and not self.is_external_sandbox:
            try:
                self._sandbox.destroy()
            except Exception as _e:
                print(f"[agent] WARNING: failed to destroy old sandbox on assignment: {_e}")
        self._sandbox = value

    def get_agent_sandbox_as_external_sandbox(self) -> "agSandbox | None":
        """Return the agent's sandbox and transfer ownership to the caller.

        Sets is_external_sandbox=True so agskill will no longer stop or destroy
        the container at the end of skill runs.  The caller is responsible for
        the sandbox lifecycle from this point on.

        WARNING: agsandbox objects wrap live Docker containers.  Cleanup relies
        on __del__ and atexit handlers, which may not run on SIGKILL or during
        interpreter shutdown.  In long-running processes, call destroy() explicitly.
        """
        self.is_external_sandbox = True
        return self._sandbox

    def set_agent_sandbox_to_external_sandbox(self, sandbox: "agSandbox | None") -> None:
        """Attach an external sandbox and transfer ownership to this agent.

        Destroys any existing agent-owned sandbox before attaching the new one.
        Sets is_external_sandbox=True so agskill will not manage this sandbox's
        lifecycle.  Pass None to detach — the next skill run provisions a fresh
        agent-owned sandbox.

        WARNING: sharing a sandbox across agents means only the last holder
        should destroy it.  Use is_external_sandbox to check ownership.
        """
        self.sandbox = sandbox          # property setter handles cleanup
        self.is_external_sandbox = sandbox is not None

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

    def _set_ui_state(self, state: str, skill: str | None = None,
                      tool: str | None = None) -> None:
        self._ui_state = {"state": state, "skill": skill, "tool": tool}
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
                _agwebui._active.emitter.agent_state(self.agname, state, skill, tool, color=_color, team=_team_name)
        except Exception as _e:
            print(f"[agent] WARNING: agent_state push failed for {self.agname}: {_e}")

    def _push_live_messages(self, messages: list) -> None:
        self._snapshot_messages = list(messages)
        try:
            from . import agwebui as _agwebui
            if _agwebui._active is not None:
                event_entries = [e for e in getattr(self, "_full_history", []) if "role" not in e]
                _agwebui._active.emitter.push_messages(
                    self.agname, list(messages) + event_entries
                )
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
                _before["input_tokens"]  + skill_inp,
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
        """Best-effort: log destruction and destroy the sandbox container."""
        _live_agents.discard(self)
        try:
            self.terminal.log("DESTROYED", "")
            self.log._lifecycle("destroyed", agname=self.agname)
        except Exception as _e:
            print(f"[agent] WARNING: __del__ log failed for {getattr(self, 'agname', '?')}: {_e}")
        if not getattr(self, "is_external_sandbox", False):
            try:
                sb = getattr(self, "_sandbox", None)
                if sb is not None:
                    sb.destroy()
            except Exception as _e:
                print(f"[agent] WARNING: sandbox.destroy() failed in __del__ for {getattr(self, 'agname', '?')}: {_e}")

    # ------------------------------------------------------------------
    # Fork
    # ------------------------------------------------------------------

    @classmethod
    def fork(cls, src: "agent", agname: str | None = None) -> "agent":
        """Return an independent agent forked from *src*."""
        ag: agent = cls.__new__(cls)
        ag.agname = _agname.allocate_agname(agname)
        ag.is_external_sandbox = False
        ag.llm = agllm(src.llm.config)
        src.ctx.resolve_prev_dependencies()
        ag.ctx = src.ctx.copy()
        _out = Path(cls.output_dir) / ag.agname if cls.output_dir else None
        ag._sandbox = src.sandbox.fork(ag.agname, output_dir=_out) if src.sandbox is not None else None
        log_dir  = Path(cls.log_dir) if cls.log_dir is not None else _DEFAULT_LOG_DIR
        log_path = log_dir / f"{ag.agname}_timeline.jsonl"
        ag.log   = aglog(path=log_path)
        ag._full_history = []
        ag._full_history_path = log_dir / f"{ag.agname}_history.jsonl"
        ag._full_history_path.parent.mkdir(parents=True, exist_ok=True)
        ag.terminal = agterm(ag.agname)
        ag._snapshot_messages = []
        ag.inbox  = queue.Queue()
        ag._ui_state = {"state": "inactive", "skill": None, "tool": None}
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
            llm_config={k: v for k, v in ag.llm.config.items() if k != "api_key"},
        )
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
        llm_config: dict,
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
                existing.terminal.log("CKPT     ", f"load_all: {agname} already live, skipping {ckpt.name}")
                restored.append(existing)
            else:
                ag = cls.load(ckpt, llm_config=llm_config)
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
            "agname":     self.agname,
            "llm_config": {k: v for k, v in self.llm.config.items() if k != "api_key"},
            "history":    self.ctx.messages,
            "ts":         _ts(),
        }
        state_bytes = json.dumps(state, indent=2).encode()

        path.parent.mkdir(parents=True, exist_ok=True)

        if self.sandbox is not None and self.sandbox._checkpoint_image is not None:
            agSandbox.tag_image(self.sandbox._checkpoint_image, image_tag)
            try:
                image_bytes = agSandbox.export_image(image_tag, CHECKPOINT_SAVE_TIMEOUT_S)
                with tarfile.open(path, "w:gz") as tar:
                    for name, data in [("state.json", state_bytes), ("container.tar", image_bytes)]:
                        info = tarfile.TarInfo(name=name)
                        info.size = len(data)
                        tar.addfile(info, io.BytesIO(data))
            finally:
                agSandbox.delete_image(image_tag, force=True)
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
        llm_config: dict,
    ) -> "agent":
        """Restore an agent from a checkpoint file created by agent.save()."""
        path = Path(path)
        image_tag = f"agency/ckpt-restore-{_uuid_mod.uuid4().hex[:8]}"

        with tarfile.open(path, "r:gz") as tar:
            state       = json.loads(tar.extractfile("state.json").read())
            container_member = next((m for m in tar.getmembers() if m.name == "container.tar"), None)
            image_bytes = tar.extractfile(container_member).read() if container_member else None

        checkpoint: str | None = None
        if image_bytes is not None:
            agSandbox.import_image(image_bytes, CHECKPOINT_LOAD_TIMEOUT_S)
            original_tag = f"agency/ckpt-{state['agname']}"
            agSandbox.tag_image(original_tag, image_tag)
            agSandbox.delete_image(original_tag)
            checkpoint = image_tag

        ag: agent = cls.__new__(cls)
        ag.agname        = _agname.claim_unique_agname(state["agname"])
        ag.is_external_sandbox = False
        ag.llm           = agllm({**state.get("llm_config", {}), **llm_config})
        ag.ctx           = agcontext(messages=list(state.get("history", [])))
        _out = Path(cls.output_dir) / ag.agname if cls.output_dir else None
        ag._sandbox      = agSandbox(ag.agname, output_dir=_out, checkpoint_image=checkpoint) if checkpoint else None

        log_dir  = Path(agent.log_dir) if agent.log_dir is not None else _DEFAULT_LOG_DIR
        ag.log   = aglog(path=log_dir / f"{ag.agname}_timeline.jsonl")
        ag._full_history: list[dict] = []
        ag._full_history_path: Path = log_dir / f"{ag.agname}_history.jsonl"
        ag._full_history_path.parent.mkdir(parents=True, exist_ok=True)
        ag.terminal = agterm(ag.agname)
        ag._snapshot_messages: list[dict] = []
        ag.inbox: queue.Queue = queue.Queue()
        ag._ui_state: dict = {"state": "inactive", "skill": None, "tool": None}

        _live_agents.add(ag)

        ag.terminal.log("LOADED   ", f"from {path}")
        ag.log._lifecycle("loaded", agname=ag.agname, source=str(path), checkpoint_ts=state.get("ts"))

        return ag

    def __repr__(self) -> str:
        return f"agent(agname={self.agname!r})"
