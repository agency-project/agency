from __future__ import annotations
import copy
import io
import json
import queue
import tarfile
import threading
import uuid as _uuid_mod
import weakref
from concurrent.futures import Future
from datetime import datetime
from pathlib import Path
from typing import ClassVar

# Single run-level ID for the default log directory.
# Created once at import time so all agents in one process share it.
_RUN_ID  = _uuid_mod.uuid4().hex[:12]
_RUN_TS  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
_DEFAULT_LOG_DIR = Path(f"/tmp/agency/{_RUN_TS}_{_RUN_ID}")

# Global weak registry of all live agent instances.
# WeakSet entries disappear automatically when agents are garbage-collected.
_live_agents: "weakref.WeakSet[agent]" = weakref.WeakSet()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT_SAVE_TIMEOUT_S = 600  # Timeout in seconds for `subprocess.run` when exporting a container image during agent.save().
CHECKPOINT_LOAD_TIMEOUT_S = 600  # Timeout in seconds for `subprocess.run` when loading a container image during agent.load().
SKILL_ERROR_LOG_TRUNCATE = 300  # Maximum characters of an error string shown in the terminal log line after a skill failure.

from .agdata import agdata, agerror
from .agutil import format_exception
from .agskill import agskill, AGSKILL_REACT_MAX_STEPS
from .aglog import aglog, _ts
from .agterm import agterm
from .agsandbox import agSandbox
from .agresources import agResourcePool
from .agcompaction import _prune_tool_outputs
from .agllm import agllm

from .agname import agname as _agname



class agent:
    """Orchestrator that maintains shared history and delegates to named agskills.

    An agent holds:
    - llm_config  : provider/model settings passed to every agskill
    - history     : shared conversation context (agdata with a 'messages' list)
    - tools       : pool of tool objects available to agskills by default
    - agskills    : list of agskill objects the user can invoke by name
    - sandbox     : container (docker or podman) isolating this agent's filesystem

    agent.run(skill_name, input)
        Always non-blocking.  Returns a pending agdata immediately.  Accessing
        any field of the result blocks until the skill — including all background
        processes it launched — has fully completed and resources are released.

        Calls on the *same* agent object are automatically serialized through
        the history chain.  Calls on different agents (forks) run concurrently.

    agent.fork(existing_agent)
        Blocks until the source agent's in-flight task completes, then
        deep-copies the resolved history and snapshots the parent's container
        filesystem.  The fork starts from the parent's exact state; its
        subsequent writes are isolated.

    Class-level configuration (set once before creating agents)::

        agent.log_dir        = Path("runs/logs")
        agent.agresource_pool  = agResourcePool()    # auto-detected by default
        agent.ping_interval_s = 300  # max seconds between process-status re-entries
        agent.poll_interval_s = 5    # granularity of the liveness poll inside that window
        agent.max_outer_iters = 144   # safety cap (~12 hours at 5-minute intervals)
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
        """Framework-wide cumulative token usage across all agents and skill calls.

        Returns {"input_tokens": int, "output_tokens": int, "total_tokens": int}.

        Example::
            usage = agent.global_token_usage()
            print(usage["total_tokens"])
        """
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
        # Need llm_config when no pre-built agllm is provided.
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

        # Track whether sandbox/llm were provided externally.
        # External objects are NOT destroyed by this agent — the caller owns them.
        self._external_sandbox: bool = sandbox is not None

        self.llm: agllm                = llm if llm is not None else agllm(llm_config)
        self._history: agdata          = agdata(messages=[])
        # Sandbox is created lazily on first _task() call to avoid the
        # gpu-detection subprocess cost at agent construction time.
        self.sandbox: agSandbox | None = sandbox

        log_dir  = Path(agent.log_dir) if agent.log_dir is not None else _DEFAULT_LOG_DIR
        log_path = log_dir / f"{self.agname}_timeline.jsonl"
        self.log  = aglog(path=log_path)
        self._full_history: list[dict] = []
        self._full_history_path: Path = log_dir / f"{self.agname}_history.jsonl"
        self._full_history_path.parent.mkdir(parents=True, exist_ok=True)
        self._term = agterm(self.agname)

        # Last fully-resolved message list — updated at the end of every skill
        # run and read (without blocking) by the UI for the history pane.
        self._snapshot_messages: list[dict] = []

        # Per-agent inbox — user messages injected from the UI between ReAct iterations.
        self._inbox: queue.Queue[str] = queue.Queue()

        # UI state dict — written from the agent thread, read by the UI timer.
        # Keys: state ("inactive"|"skill"|"llm"|"tool"|"proc_wait"|"human"),
        #       skill (str|None), tool (str|None)
        self._ui_state: dict = {"state": "inactive", "skill": None, "tool": None}

        _live_agents.add(self)

        # Auto-register with the enclosing agteam if run() is on the call stack.
        from ._context import _active_team
        _team = _active_team.get(None)
        if _team is not None:
            _team._agents.add(self)

        team_name = _team.team_name if _team is not None else None

        ctx = f"  context={self.llm.context_limit}" if self.llm.context_limit else "  context=unknown"
        team_tag = f"  team={team_name}" if team_name else ""
        self._term.log("CREATED  ", f"model={self.llm.config.get('model','?')}{ctx}{team_tag}")
        self.log._lifecycle(
            "created",
            agname=self.agname,
            team=team_name,
            llm_config={k: v for k, v in self.llm.config.items() if k != "api_key"},
            context_limit=self.llm.context_limit,
        )

    # ------------------------------------------------------------------
    # History property — blocks until the current chain link resolves
    # ------------------------------------------------------------------

    @property
    def output_path(self) -> Path | None:
        """Host-side output directory for this agent, or None if output_dir is not set.

        Files written to ``container_output_path`` inside the container appear here.
        """
        if agent.output_dir is None:
            return None
        return Path(agent.output_dir) / self.agname

    @property
    def container_output_path(self) -> str | None:
        """Path inside the container where this agent should write output files.

        Mounted read-write from ``output_path`` on the host.
        """
        if agent.output_dir is None:
            return None
        return f"/agent_output/{self.agname}"

    @property
    def token_usage(self) -> dict:
        """Cumulative token usage for this agent across all completed skill calls.

        Returns {"input_tokens": int, "output_tokens": int, "total_tokens": int}.

        For framework-wide totals across all agents use ``agent.global_token_usage()``.
        """
        return self.log.token_usage

    @property
    def history(self) -> agdata:
        """Return the current history, blocking until any in-flight task finishes."""
        self._history._resolve()
        return self._history

    @history.setter
    def history(self, value: agdata) -> None:
        self._history = value

    @property
    def full_history(self) -> list[dict]:
        """Append-only transcript: every message ever sent/received, including
        thinking blocks. Never compacted or pruned."""
        return list(self._full_history)

    def set_llm_config(self, llm_config: dict) -> None:
        """Replace the agent's LLM config and refresh the context limit."""
        self.llm = agllm(dict(llm_config))

    def set_full_history(self, history: list[dict]) -> None:
        """Replace the agent's full history with a deep copy of *history*."""
        self._full_history = copy.deepcopy(history)

    def reset_full_history(self) -> None:
        """Clear the agent's full history."""
        self._full_history = []

    def _append_full_history(self, msg: dict) -> None:
        """Append one message to the append-only full history (thread-safe write)."""
        self._full_history.append(msg)
        with self._full_history_path.open("a") as f:
            f.write(json.dumps(msg) + "\n")
        if "role" not in msg:
            # Event entry (skill_error, llm_retry, etc.) — push immediately so it
            # appears in the webui history panel without waiting for the next LLM turn.
            self._push_live_messages(self._snapshot_messages)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

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
                # Append event entries (skill_error, llm_retry, etc.) to the snapshot
                # so they appear in the webui history panel alongside conversation messages.
                event_entries = [e for e in getattr(self, "_full_history", []) if "role" not in e]
                _agwebui._active.emitter.push_messages(
                    self.agname, list(messages) + event_entries
                )
        except Exception as _e:
            print(f"[agent] WARNING: push_messages failed for {self.agname}: {_e}")
            
    def run(self, skill: "agskill", input: agdata, max_steps: int = AGSKILL_REACT_MAX_STEPS) -> agdata:
        """Submit the skill and return a pending agdata immediately.

        The future resolves only after:
        1. The ReAct loop finishes.
        2. All background processes started via ``bash`` have exited.
        3. All acquired resources (GPU/CPU) have been released.

        Calls on the same agent are serialized via the history chain.
        Calls on different agents (forks) run concurrently.
        """
        skill_name = skill.name
        af         = skill

        prev_history = self._history
        result_future: Future[agdata] = Future()
        history_future: Future[agdata] = Future()
        ts_start = _ts()
        pool = agent.agresource_pool

        def _task() -> None:
            try:
                # ── 1. Unblock: wait for any in-flight predecessor skill to finish,
                #    then resolve any lazy input futures passed by the caller.
                prev_history._resolve()
                input.resolve_input()

                # ── 2. Provision sandbox — created once on first run and reused
                #    across subsequent runs via its internal checkpoint image.
                if not self._external_sandbox and self.sandbox is None:
                    _out = Path(agent.output_dir) / self.agname if agent.output_dir else None
                    self.sandbox = agSandbox(self.agname, output_dir=_out)

                history_before = list(prev_history._data.get("messages", []))

                self._term.log("SKILL ▶  ", f"{skill_name}  input={list(input._data.keys())}")
                self._set_ui_state("skill", skill=skill_name)

                # ── 3. Snapshot cumulative log usage before this skill so the live
                #    token callback can compute the correct agent-total mid-skill.
                _log_usage_before = self.log.token_usage

                def _live_token_update(skill_inp: int, skill_out: int) -> None:
                    try:
                        from . import agwebui as _agwebui
                        if _agwebui._active is None:
                            return
                        _prev = _log_usage_before
                        _gl   = agent.global_token_usage()
                        _agwebui._active.emitter.token_update(
                            self.agname,
                            _prev["input_tokens"]  + skill_inp,
                            _prev["output_tokens"] + skill_out,
                            # Use the committed global total as-is. Adding
                            # skill_inp here would race with concurrent agents
                            # doing the same, producing out-of-order values.
                            # The committed global is locked and monotonically
                            # increasing, so it never produces negative rates.
                            _gl["input_tokens"],
                            _gl["output_tokens"],
                        )
                    except Exception as _e:
                        print(f"[agent] WARNING: live token_update push failed for {self.agname}: {_e}")

                def _drain_inbox() -> str | None:
                    try:
                        return self._inbox.get_nowait()
                    except queue.Empty:
                        return None

                self._append_full_history({
                    "type": "skill_start",
                    "skill": skill_name,
                    "ts": ts_start,
                })

                # ── 4. Run the ReAct loop — all input prep, LLM calls, tool
                #    dispatch, output recovery, and sandbox cleanup happen inside.
                outer_input_tokens  = 0
                outer_output_tokens = 0
                outer_result, outer_history, outer_delta, _tok = af.run(
                    self.llm, input, prev_history,
                    self.sandbox, pool, max_steps, term=self._term, log=self.log,
                    _state_fn=self._set_ui_state,
                    _live_messages_fn=self._push_live_messages,
                    _inbox_fn=_drain_inbox,
                    _full_history_fn=self._append_full_history,
                    _token_update_fn=_live_token_update,
                    _ping_interval_s=agent.ping_interval_s,
                    _poll_interval_s=agent.poll_interval_s,
                    _agname=self.agname,
                )
                outer_input_tokens  = _tok[0]
                outer_output_tokens = _tok[1]

            except Exception as exc:
                # ── 4a. Unexpected exception — wrap in agerror so the caller
                #     gets a clean result instead of a dangling future.
                outer_result  = agerror(format_exception(exc))
                outer_history = prev_history
                outer_delta   = []
                history_before = list(prev_history._data.get("messages", []))
                self._term.log("SKILL ✗  ", f"{skill_name}  exception={exc}")
            finally:
                # ── 5. Teardown — release GPU slot, commit container filesystem
                #    to a checkpoint image, then stop the container.
                #    External sandboxes are left running — the caller owns them.
                _had_error = outer_result is not None and bool(outer_result._data.get("error"))
                self._set_ui_state("error" if _had_error else "finished")
                if self.sandbox._gpu_id is not None:
                    pool.release_gpu(self.sandbox._gpu_id)
                if not self._external_sandbox:
                    self.sandbox.stop(commit=True)

            # ── 6. Log result and commit token counts.
            ts_end = _ts()
            assert outer_result is not None
            input_dict  = input.to_dict()
            result_dict = outer_result.to_dict()
            if result_dict.get("error"):
                self._term.log("SKILL ✗  ", f"{skill_name}  error={str(result_dict['error'])[:SKILL_ERROR_LOG_TRUNCATE]}")
                self._append_full_history({"type": "skill_error", "skill": skill_name,
                                           "error": str(result_dict["error"])})
            else:
                self._term.log("SKILL ✓  ", f"{skill_name}  output={list(result_dict.keys())}")
            try:
                self.log._record(skill_name, ts_start, ts_end,
                                 input_dict, result_dict,
                                 len(outer_history._data.get("messages", [])),
                                 history_before=history_before,
                                 history_delta=outer_delta,
                                 input_tokens=outer_input_tokens,
                                 output_tokens=outer_output_tokens)
                agent._add_global_tokens(outer_input_tokens, outer_output_tokens)
                _ag_usage  = self.log.token_usage
                _gl_usage  = agent.global_token_usage()
                try:
                    from . import agwebui as _agwebui
                    if _agwebui._active is not None:
                        _agwebui._active.emitter.token_update(
                            self.agname,
                            _ag_usage["input_tokens"],
                            _ag_usage["output_tokens"],
                            _gl_usage["input_tokens"],
                            _gl_usage["output_tokens"],
                        )
                except Exception as _e:
                    print(f"[agent] WARNING: post-skill token_update push failed for {self.agname}: {_e}")
            except Exception as log_exc:
                self._term.log("SKILL ✗  ", f"[log error] {log_exc}")

            # ── 7. Resolve result future — unblocks the caller immediately so it
            #    can process the result while pruning runs in the background.
            self._snapshot_messages = list(outer_history._data.get("messages", []))
            result_future.set_result(outer_result)

            # ── 8. Prune history — trim oversized tool outputs from the shared
            #    history before resolving history_future so the next skill in
            #    the chain always starts with a compact context.
            try:
                pruned_msgs = _prune_tool_outputs(
                    outer_history._data.get("messages", [])
                )
                if pruned_msgs is not outer_history._data.get("messages", []):
                    outer_history = agdata(messages=pruned_msgs)
                    self._term.log("PRUNE    ", f"{skill_name}  history pruned to {len(pruned_msgs)} msgs")
            except Exception as prune_exc:
                self._term.log("PRUNE ✗  ", f"{skill_name}  pruning failed: {prune_exc}")

            # ── 9. Resolve history future — unblocks the next chained run() call.
            history_future.set_result(outer_history)

        threading.Thread(target=_task, daemon=True).start()

        self._history = agdata(_future=history_future)
        return agdata(_future=result_future)

    # ------------------------------------------------------------------
    # Backward-compat fork helper
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        """Best-effort: log destruction and destroy the sandbox container."""
        _live_agents.discard(self)
        try:
            self._term.log("DESTROYED", "")
            self.log._lifecycle("destroyed", agname=self.agname)
        except Exception as _e:
            print(f"[agent] WARNING: __del__ log failed for {getattr(self, 'agname', '?')}: {_e}")
        if not getattr(self, "_external_sandbox", False):
            try:
                if self.sandbox is not None:
                    self.sandbox.destroy()
            except Exception as _e:
                print(f"[agent] WARNING: sandbox.destroy() failed in __del__ for {getattr(self, 'agname', '?')}: {_e}")

    @classmethod
    def fork(cls, src: "agent", agname: str | None = None) -> "agent":
        """Return an independent agent forked from *src*.

        The fork starts with a deep copy of *src*'s conversation history and,
        if *src* has a sandbox checkpoint, a tagged copy of that image so the
        fork's first task resumes from the same container filesystem state.
        """
        ag: agent = cls.__new__(cls)
        ag.agname = _agname.allocate_agname(agname)
        ag._external_sandbox = False
        ag.llm = agllm(src.llm.config)
        src._history._resolve()
        ag._history = copy.deepcopy(src._history)
        _out = Path(cls.output_dir) / ag.agname if cls.output_dir else None
        ag.sandbox = src.sandbox.fork(ag.agname, output_dir=_out) if src.sandbox is not None else None
        # Initialise the remaining agent bookkeeping fields
        log_dir  = Path(cls.log_dir) if cls.log_dir is not None else _DEFAULT_LOG_DIR
        log_path = log_dir / f"{ag.agname}_timeline.jsonl"
        ag.log   = aglog(path=log_path)
        ag._full_history = []
        ag._full_history_path = log_dir / f"{ag.agname}_history.jsonl"
        ag._full_history_path.parent.mkdir(parents=True, exist_ok=True)
        ag._term = agterm(ag.agname)
        ag._snapshot_messages = []
        ag._inbox  = queue.Queue()
        ag._ui_state = {"state": "inactive", "skill": None, "tool": None}
        _live_agents.add(ag)

        from ._context import _active_team
        _team = _active_team.get(None)
        if _team is not None:
            _team._agents.add(ag)
        team_name = _team.team_name if _team is not None else None

        ag._term.log("FORKED   ", f"from {src.agname}")
        ag.log._lifecycle(
            "forked",
            agname=ag.agname,
            parent_agname=src.agname,
            team=team_name,
            llm_config={k: v for k, v in ag.llm.config.items() if k != "api_key"},
        )
        return ag

    async def asyncio_run(
        self,
        skill: "agskill",
        input: "agdata",
        max_steps: int = AGSKILL_REACT_MAX_STEPS,
    ) -> "agdata":
        """Async wrapper around ``run()`` for use in asyncio event loops.

        Submits the skill to the thread pool (same as ``run()``) and awaits
        completion without blocking the event loop thread.  The returned
        ``agdata`` is fully resolved — no further blocking on field access.

        Equivalent to ``await Runner.run(agent, input)`` in openai-agents.

        Example — FastAPI endpoint::

            @app.post("/ask")
            async def ask(question: str):
                result = await ag.asyncio_run(qa_skill, agdata(question=question))
                return {"answer": result.answer}

        Example — parallel execution with asyncio.gather::

            results = await asyncio.gather(*[
                agent(parent).asyncio_run(translate_skill, agdata(text=msg))
                for _ in range(3)
            ])
        """
        import asyncio
        loop = asyncio.get_event_loop()
        pending = self.run(skill, input, max_steps)
        await loop.run_in_executor(None, pending._resolve)
        return pending

    # ------------------------------------------------------------------
    # Live agent registry
    # ------------------------------------------------------------------

    @classmethod
    def all(cls) -> "list[agent]":
        """Return all currently live agent instances in this process."""
        return list(_live_agents)

    @classmethod
    def save_all(cls, directory: "Path | str") -> "list[Path]":
        """Checkpoint every live agent to *directory*/<agname>.ckpt.

        Waits for any in-flight run() on each agent before snapshotting.
        Returns the list of paths written.
        """
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
        """Restore all ``*.ckpt`` files from *directory*.

        Behaviour for each checkpoint found:

        - **agname already live**: the checkpoint is skipped and the existing
          agent is returned as-is.  The running agent is never overwritten —
          a live agent always takes precedence over a checkpoint on disk.
        - **agname not live**: a new agent is restored from the checkpoint and
          added to the live registry.

        Returns the full list (both existing and newly restored agents).
        """
        directory = Path(directory)
        live_names = {ag.agname: ag for ag in cls.all()}
        restored: list[agent] = []

        for ckpt in sorted(directory.glob("*.ckpt")):
            with tarfile.open(ckpt, "r:gz") as tar:
                state = json.loads(tar.extractfile("state.json").read())
            agname = state["agname"]

            if agname in live_names:
                existing = live_names[agname]
                existing._term.log("CKPT     ", f"load_all: {agname} already live, skipping {ckpt.name}")
                restored.append(existing)
            else:
                ag = cls.load(ckpt, llm_config=llm_config)
                restored.append(ag)

        return restored

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: "Path | str") -> None:
        """Checkpoint this agent to a single .ckpt file.

        Blocks until any in-flight ``run()`` completes so history and the
        container filesystem are consistent before the snapshot is taken.

        The file is a gzip-compressed tar archive containing:
          - ``state.json``   — history, agname, llm_config keys, skill names
          - ``container.tar`` — full Docker image export of the container filesystem

        The api_key is never written to disk.
        """
        path = Path(path)
        image_tag = f"agency/ckpt-{self.agname}"

        if self._history.is_pending():
            self._term.log("CKPT ⏳  ", "waiting for in-flight task to complete...")
        self._history._resolve()

        # Build state dict
        state = {
            "agname":     self.agname,
            "llm_config": {k: v for k, v in self.llm.config.items() if k != "api_key"},
            "history":    self._history._data.get("messages", []),
            "ts":         _ts(),
        }
        state_bytes = json.dumps(state, indent=2).encode()

        path.parent.mkdir(parents=True, exist_ok=True)

        if self.sandbox is not None and self.sandbox._checkpoint_image is not None:
            # Tag checkpoint for export, then clean up the temp export tag.
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
            # Agent never used a container — save history only (no filesystem state)
            with tarfile.open(path, "w:gz") as tar:
                info = tarfile.TarInfo(name="state.json")
                info.size = len(state_bytes)
                tar.addfile(info, io.BytesIO(state_bytes))

        size_kb = path.stat().st_size // 1024
        self._term.log("CKPT ✓   ", f"saved → {path}  ({size_kb} KB)")
        self.log._lifecycle("saved", agname=self.agname, path=str(path), size_kb=size_kb)

    @classmethod
    def load(
        cls,
        path: "Path | str",
        llm_config: dict,
    ) -> "agent":
        """Restore an agent from a checkpoint file created by ``agent.save()``.

        Always creates a **new** agent instance — it never modifies or replaces
        any currently live agent, even if an agent with the same agname already
        exists.  If the checkpoint's agname is already taken in this process,
        ``ValueError`` is raised; discard the conflicting agent first or use
        ``agent.load_all()`` which handles this automatically.

        The caller must re-supply ``llm_config`` (api_key is never stored).
        The restored agent has the same conversation history and container
        filesystem as at checkpoint time and can continue running skills immediately.
        """
        path = Path(path)
        image_tag = f"agency/ckpt-restore-{_uuid_mod.uuid4().hex[:8]}"

        with tarfile.open(path, "r:gz") as tar:
            state       = json.loads(tar.extractfile("state.json").read())
            container_member = next((m for m in tar.getmembers() if m.name == "container.tar"), None)
            image_bytes = tar.extractfile(container_member).read() if container_member else None

        checkpoint: str | None = None
        if image_bytes is not None:
            # Load image — docker restores the original tag (agency/ckpt-{agname})
            agSandbox.import_image(image_bytes, CHECKPOINT_LOAD_TIMEOUT_S)
            original_tag = f"agency/ckpt-{state['agname']}"
            # Re-tag to a unique name so concurrent restores don't collide,
            # then remove the original tag
            agSandbox.tag_image(original_tag, image_tag)
            agSandbox.delete_image(original_tag)
            checkpoint = image_tag

        # Build agent without going through normal __init__ to avoid creating a fresh container
        ag: agent = cls.__new__(cls)
        ag.agname        = _agname.claim_unique_agname(state["agname"])
        ag._external_sandbox = False
        ag.llm           = agllm({**state.get("llm_config", {}), **llm_config})
        ag._history      = agdata(messages=list(state.get("history", [])))
        _out = Path(cls.output_dir) / ag.agname if cls.output_dir else None
        # Create sandbox eagerly here — the checkpoint image tag is a temporary
        # unique tag that must be owned by the sandbox immediately; it can't wait.
        ag.sandbox       = agSandbox(ag.agname, output_dir=_out, checkpoint_image=checkpoint) if checkpoint else None

        log_dir  = Path(agent.log_dir) if agent.log_dir is not None else _DEFAULT_LOG_DIR
        ag.log   = aglog(path=log_dir / f"{ag.agname}_timeline.jsonl")
        ag._full_history: list[dict] = []
        ag._full_history_path: Path = log_dir / f"{ag.agname}_history.jsonl"
        ag._full_history_path.parent.mkdir(parents=True, exist_ok=True)
        ag._term = agterm(ag.agname)
        ag._snapshot_messages: list[dict] = []
        ag._inbox: queue.Queue = queue.Queue()
        ag._ui_state: dict = {"state": "inactive", "skill": None, "tool": None}

        _live_agents.add(ag)

        ag._term.log("LOADED   ", f"from {path}")
        ag.log._lifecycle("loaded", agname=ag.agname, source=str(path), checkpoint_ts=state.get("ts"))

        return ag

    def __repr__(self) -> str:
        return f"agent(agname={self.agname!r})"
