from __future__ import annotations
import copy
import io
import json
import queue
import subprocess
import tarfile
import time
import uuid as _uuid_mod
import weakref
from concurrent.futures import ThreadPoolExecutor, Future
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

from .agdata import agdata
from .agskill import agskill
from .agtool import agtool
from .aglog import aglog, _ts
from .agterm import agterm
from .agsandbox import agSandbox, get_container_runtime
from .agresources import agResourcePool
from .agcompaction import fetch_context_limit, _prune_tool_outputs
from .tools import make_sandboxed_tools

_NOUNS = [
    "alex", "andy", "arch", "bake", "bale", "band", "bart", "base",
    "beam", "bear", "beef", "bell", "bill", "bird", "blue", "boat",
    "bolt", "bond", "bone", "bonk", "book", "boss", "brim", "buzz",
    "byte", "cage", "cake", "cane", "cant", "cape", "cart", "cask",
    "cave", "chip", "clam", "clay", "coal", "coil", "coin", "colt",
    "cord", "core", "corn", "cove", "crab", "crag", "crow", "dale",
    "dart", "deer", "dome", "dove", "down", "drum", "duck", "dune",
    "dust", "east", "edge", "evil", "fang", "fast", "fate", "fawn",
    "felt", "fern", "film", "fire", "fish", "fist", "flat", "flaw",
    "flux", "foam", "font", "fork", "frog", "fuse", "gale", "game",
    "gate", "gear", "glen", "greg", "grip", "gust", "hail", "hare",
    "hawk", "haze", "hemp", "hill", "hind", "hole", "hoop", "hull",
    "ibex", "ivan", "jake", "jane", "joey", "juke", "kite", "kodo",
    "ksen", "lake", "land", "lard", "lash", "lava", "leaf", "lego",
    "lily", "lion", "lord", "love", "lynx", "made", "many", "mark",
    "mean", "mert", "mess", "meta", "mick", "mill", "mink", "moba",
    "moon", "moth", "mule", "must", "nail", "nate", "next", "node",
    "onix", "pain", "park", "peat", "pier", "pike", "pile", "pine",
    "plug", "pony", "pool", "pork", "puma", "rain", "rate", "real",
    "reef", "rest", "rice", "road", "rock", "roll", "rope", "rust",
    "sage", "salt", "sand", "seal", "shot", "silk", "slag", "snow",
    "soda", "soil", "sold", "song", "spam", "star", "surf", "swan",
    "tack", "tail", "tide", "tire", "toad", "tony", "tool", "tree",
    "tuna", "turf", "vast", "vent", "vine", "wake", "ward", "wasp",
    "well", "wick", "wind", "wire", "wolf", "wood", "yang", "zinc",
]

_noun_index:      int           = 0
_noun_counters:   dict[str, int] = {}
_allocated_agnames: set[str]    = set()
_agname_lock      = __import__("threading").Lock()


def _register_agname(full_name: str) -> str:
    """Register an already-final name as in-use, raising if taken."""
    with _agname_lock:
        if full_name in _allocated_agnames:
            raise ValueError(f"agname {full_name!r} is already in use by another agent")
        _allocated_agnames.add(full_name)
    return full_name


def _allocate_agname(name: str) -> str:
    """Return a unique name in the form <name>_NNN and register it as in-use."""
    with _agname_lock:
        n = _noun_counters.get(name, 0)
        _noun_counters[name] = n + 1
        full = f"{name}_{n:03d}"
        _allocated_agnames.add(full)
    return full


def _generate_agname() -> str:
    """Return a unique agname in the form <noun>_<3-digit number>.

    Nouns are assigned in order from _NOUNS, cycling back to the start after
    the last entry. The suffix increments independently per noun, so arch_000
    and arch_001 are the first and second agents that received 'arch'.
    """
    global _noun_index
    with _agname_lock:
        noun = _NOUNS[_noun_index % len(_NOUNS)]
        _noun_index += 1
        n = _noun_counters.get(noun, 0)
        _noun_counters[noun] = n + 1
        name = f"{noun}_{n:03d}"
        _allocated_agnames.add(name)
    return name


def _resolve_input(inp: agdata) -> None:
    """Resolve any pending agdata values nested inside inp, in-place.

    Handles:
    - inp itself being pending (resolves before inspecting fields)
    - top-level field values that are pending agdata
    - list fields whose elements are pending agdata
    """
    inp._resolve()
    for val in inp._data.values():
        if isinstance(val, agdata):
            val._resolve()
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, agdata):
                    item._resolve()


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

    agent(existing_agent)
        Copy constructor.  Blocks until the source agent's in-flight task
        completes, then deep-copies the resolved history and snapshots the
        parent's container via ``podman commit``.  The fork starts from the
        parent's exact filesystem state; its subsequent writes are isolated.

    Class-level configuration (set once before creating agents)::

        agent.log_dir        = Path("runs/logs")
        agent.agresource_pool  = agResourcePool()    # auto-detected by default
        agent.ping_interval_s = 300  # max seconds between process-status re-entries
        agent.poll_interval_s = 5    # granularity of the liveness poll inside that window
        agent.max_outer_iters = 144   # safety cap (~12 hours at 5-minute intervals)
    """

    _pool: ClassVar[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=256)
    log_dir:          ClassVar[Path | None]          = None
    output_dir:       ClassVar[Path | None]          = None
    agresource_pool:  ClassVar[agResourcePool]       = agResourcePool()
    ping_interval_s:  ClassVar[int]                  = 300
    poll_interval_s:  ClassVar[int]                  = 5
    max_outer_iters:  ClassVar[int]                  = 144

    def __init__(
        self,
        llm_config: "dict | agent | None" = None,
        agskills: list[agskill] | None = None,
        tools: list[agtool] | None = None,
        agname: str | None = None,
    ):
        # Inject llm_config from the enclosing agteam when not supplied.
        if llm_config is None:
            from ._context import _active_team as _at
            _t = _at.get(None)
            if _t is not None:
                llm_config = _t.llm_config
            else:
                raise TypeError("agent() requires llm_config when called outside an agteam context")

        self.agname = _generate_agname() if agname is None else _allocate_agname(agname)
        pool = agent.agresource_pool

        # Per-agent output subdir: <output_dir>/<agname>/
        _out = Path(agent.output_dir) / self.agname if agent.output_dir else None

        if isinstance(llm_config, agent):
            src = llm_config
            self.llm_config    = src.llm_config
            self._context_limit: int | None = src._context_limit
            self.agskills      = list(agskills if agskills is not None else src.agskills)
            # Block until source's in-flight task finishes, then deep-copy history
            src._history._resolve()
            self._history: agdata = copy.deepcopy(src._history)
            # Snapshot parent container → fork starts from parent's exact state
            self.sandbox = agSandbox(self.agname, parent_agname=src.agname, output_dir=_out)
        else:
            self.llm_config    = llm_config
            self._context_limit = fetch_context_limit(llm_config)
            self.agskills      = list(agskills or [])
            self._history      = agdata(messages=[])
            self.sandbox       = agSandbox(self.agname, output_dir=_out)

        # Build sandboxed tool list; user-supplied tools override if provided
        if tools is not None:
            self.tools = list(tools)
        else:
            self.tools = make_sandboxed_tools(self.sandbox, pool)

        log_dir  = Path(agent.log_dir) if agent.log_dir is not None else _DEFAULT_LOG_DIR
        log_path = log_dir / f"{self.agname}.jsonl"
        self.log  = aglog(path=log_path)
        self._full_history: list[dict] = []
        self._full_history_path: Path = log_dir / f"{self.agname}_full.jsonl"
        self._full_history_path.parent.mkdir(parents=True, exist_ok=True)
        self._term = agterm(self.agname)

        # Wire terminal + file logging into every tool.
        for t in self.tools:
            t.attach_logger(self._term, self.log)

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

        if isinstance(llm_config, agent):
            self._term.log("FORKED   ", f"from {src.agname}  skills={[s.name for s in self.agskills]}")
            self.log._lifecycle(
                "forked",
                agname=self.agname,
                parent_agname=src.agname,
                agskills=[s.name for s in self.agskills],
                tools=[t.name for t in self.tools],
                llm_config={k: v for k, v in self.llm_config.items() if k != "api_key"},
            )
        else:
            ctx = f"  context={self._context_limit}" if self._context_limit else "  context=unknown"
            self._term.log("CREATED  ", f"skills={[s.name for s in self.agskills]}  model={self.llm_config.get('model','?')}{ctx}")
            self.log._lifecycle(
                "created",
                agname=self.agname,
                agskills=[s.name for s in self.agskills],
                tools=[t.name for t in self.tools],
                llm_config={k: v for k, v in self.llm_config.items() if k != "api_key"},
                context_limit=self._context_limit,
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

    def _append_full_history(self, msg: dict) -> None:
        """Append one message to the append-only full history (thread-safe write)."""
        self._full_history.append(msg)
        with self._full_history_path.open("a") as f:
            f.write(json.dumps(msg) + "\n")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _set_ui_state(self, state: str, skill: str | None = None,
                      tool: str | None = None) -> None:
        self._ui_state = {"state": state, "skill": skill, "tool": tool}

    def _push_live_messages(self, messages: list) -> None:
        self._snapshot_messages = list(messages)
        try:
            from . import agui as _agui
            if _agui._active is not None:
                _agui._active._app.call_from_thread(_agui._active._app._render_history)
        except Exception:
            pass

    def run(self, skill_name: str, input: agdata, max_steps: int = 10) -> agdata:
        """Submit the skill and return a pending agdata immediately.

        The future resolves only after:
        1. The ReAct loop finishes.
        2. All background processes started via ``bash`` have exited.
        3. All acquired resources (GPU/CPU) have been released.

        Calls on the same agent are serialized via the history chain.
        Calls on different agents (forks) run concurrently.
        """
        if not any(f.name == skill_name for f in self.agskills):
            raise ValueError(f"agskill not found: {skill_name!r}")

        prev_history = self._history
        result_future: Future[agdata] = Future()
        history_future: Future[agdata] = Future()
        ts_start = _ts()
        pool = agent.agresource_pool

        def _task() -> None:
            try:
                prev_history._resolve()
                _resolve_input(input)

                history_before = list(prev_history._data.get("messages", []))

                af = next(f for f in self.agskills if f.name == skill_name)

                self._term.log("SKILL ▶  ", f"{skill_name}  input={list(input._data.keys())}")
                self._set_ui_state("skill", skill=skill_name)

                # ----------------------------------------------------------
                # Outer monitoring loop
                # Runs the ReAct loop, then waits for any background PIDs.
                # If long-running PIDs remain after min_wait_s, re-enters the
                # ReAct loop with a status ping so the agent can respond.
                # ----------------------------------------------------------
                current_input   = input
                current_history = prev_history
                outer_result:    agdata | None = None
                outer_history:   agdata        = prev_history
                outer_delta:     list[dict]    = []
                is_continuation  = False

                for _outer_iter in range(agent.max_outer_iters):
                    def _drain_inbox() -> str | None:
                        try:
                            return self._inbox.get_nowait()
                        except queue.Empty:
                            return None

                    def _compact_log(**kw) -> None:
                        self.log._lifecycle("compacted", agname=self.agname, **kw)

                    # Skill-start marker in full history
                    self._append_full_history({
                        "type": "skill_start",
                        "skill": skill_name,
                        "ts": ts_start,
                    })

                    result, new_history, history_delta = af.run(
                        self.llm_config, current_input, current_history,
                        self.tools, max_steps, term=self._term,
                        _is_continuation=is_continuation,
                        _state_fn=self._set_ui_state,
                        _live_messages_fn=self._push_live_messages,
                        _inbox_fn=_drain_inbox,
                        _context_limit=self._context_limit,
                        _compact_log_fn=_compact_log,
                        _full_history_fn=self._append_full_history,
                    )
                    outer_result  = result
                    outer_history = new_history
                    outer_delta.extend(history_delta)

                    # Snapshot which PIDs were outstanding when this ReAct
                    # iteration ended — used below to detect completion.
                    pids_at_end = set(self.sandbox._watched_pids)

                    if not pids_at_end:
                        break  # no background work — skill is done

                    # Background processes detected — log the start of the wait.
                    summary = self.sandbox.pid_status_summary()
                    self._set_ui_state("proc_wait", skill=skill_name)
                    self._term.log("PROCS ▶  ", f"{skill_name}  monitoring: {summary}")
                    self.log._lifecycle("procs_started", agname=self.agname,
                                        skill=skill_name, pids=list(pids_at_end),
                                        summary=summary)

                    # Poll liveness at poll_interval_s granularity for up to
                    # ping_interval_s total.  Break as soon as all PIDs are
                    # gone — whether that takes 2 seconds or 5 minutes.
                    deadline = time.monotonic() + agent.ping_interval_s
                    while time.monotonic() < deadline:
                        time.sleep(agent.poll_interval_s)
                        if not self.sandbox.get_live_pids():
                            break

                    live_now = self.sandbox.get_live_pids()

                    if not live_now:
                        # All background processes finished — re-enter so the
                        # agent can read their output and act on the results.
                        self._term.log("PROCS ✓  ", f"{skill_name}  all processes completed, re-entering agent")
                        self.log._lifecycle("procs_completed", agname=self.agname,
                                            skill=skill_name)
                        current_input = agdata(
                            _event="process_completed",
                            message=(
                                "Background processes have completed. "
                                "Read their output and act on the results."
                            ),
                        )
                        current_history = new_history
                        is_continuation = True
                        continue

                    # Processes still running after ping_interval_s — ping agent.
                    summary = self.sandbox.pid_status_summary()
                    self._term.log("PROCS ⏳  ", f"{skill_name}  still running: {summary}")
                    self.log._lifecycle("procs_ping", agname=self.agname,
                                        skill=skill_name, pids=list(live_now),
                                        summary=summary)
                    current_input   = agdata(
                        _event="process_update",
                        message=(
                            f"Background processes are still running: {summary}. "
                            f"You may check their output, wait, or proceed if appropriate. "
                            f"If any of these processes are intentional long-running services "
                            f"(daemons, servers, monitors) that should not block completion, "
                            f"call daemon_release(pid) for each such PID to release it from "
                            f"monitoring."
                        ),
                    )
                    current_history = new_history
                    is_continuation = True

            except Exception as exc:
                outer_result  = agdata(error=str(exc))
                outer_history = prev_history
                outer_delta   = []
                history_before = list(prev_history._data.get("messages", []))
                self._term.log("SKILL ✗  ", f"{skill_name}  exception={exc}")
            finally:
                self._set_ui_state("inactive")
                self.sandbox.release_resources(pool)

            # Log and resolve futures after all background work is done
            ts_end = _ts()
            assert outer_result is not None
            input_dict  = input.to_dict()
            result_dict = outer_result.to_dict()
            if result_dict.get("error"):
                self._term.log("SKILL ✗  ", f"{skill_name}  error={str(result_dict['error'])[:80]}")
            else:
                self._term.log("SKILL ✓  ", f"{skill_name}  output={list(result_dict.keys())}")
            try:
                self.log._record(skill_name, ts_start, ts_end,
                                 input_dict, result_dict,
                                 len(outer_history._data.get("messages", [])),
                                 history_before=history_before,
                                 history_delta=outer_delta)
            except Exception as log_exc:
                self._term.log("SKILL ✗  ", f"[log error] {log_exc}")

            self._snapshot_messages = list(outer_history._data.get("messages", []))
            result_future.set_result(outer_result)

            # Post-skill history pruning — trim old oversized tool outputs from
            # the shared history before unblocking the next run() on this agent.
            # Runs after result_future so the caller can unblock immediately;
            # history_future holds until pruning is done so the dependency chain
            # sees clean history.
            pruned_msgs = _prune_tool_outputs(
                outer_history._data.get("messages", [])
            )
            if pruned_msgs is not outer_history._data.get("messages", []):
                outer_history = agdata(messages=pruned_msgs)
                self._term.log("PRUNE    ", f"{skill_name}  history pruned to {len(pruned_msgs)} msgs")

            history_future.set_result(outer_history)

        agent._pool.submit(_task)

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
        except Exception:
            pass
        try:
            self.sandbox.destroy()
        except Exception:
            pass

    def fork(self) -> "agent":
        """Return an independent copy of this agent (same as agent(self))."""
        return agent(self)

    async def asyncio_run(
        self,
        skill_name: str,
        input: "agdata",
        max_steps: int = 10,
    ) -> "agdata":
        """Async wrapper around ``run()`` for use in asyncio event loops.

        Submits the skill to the thread pool (same as ``run()``) and awaits
        completion without blocking the event loop thread.  The returned
        ``agdata`` is fully resolved — no further blocking on field access.

        Equivalent to ``await Runner.run(agent, input)`` in openai-agents.

        Example — FastAPI endpoint::

            @app.post("/ask")
            async def ask(question: str):
                result = await ag.asyncio_run("qa", agdata(question=question))
                return {"answer": result.answer}

        Example — parallel execution with asyncio.gather::

            results = await asyncio.gather(*[
                agent(parent).asyncio_run("translate", agdata(text=msg))
                for _ in range(3)
            ])
        """
        import asyncio
        loop = asyncio.get_event_loop()
        pending = self.run(skill_name, input, max_steps)
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
        agskills: "list[agskill] | None" = None,
        tools: "list[agtool] | None" = None,
    ) -> "list[agent]":
        """Restore all ``*.ckpt`` files from *directory*.

        Behaviour for each checkpoint found:

        - **agname already live**: the checkpoint is skipped and the existing
          agent is returned as-is.  The running agent is never overwritten —
          a live agent always takes precedence over a checkpoint on disk.
        - **agname not live**: a new agent is restored from the checkpoint and
          added to the live registry.

        ``agskills`` and ``tools`` are shared across all restored agents.
        Returns the full list (both existing and newly restored agents).
        """
        directory = Path(directory)
        live_names = {ag.agname: ag for ag in cls.all()}
        restored: list[agent] = []

        for ckpt in sorted(directory.glob("*.ckpt")):
            # Peek at the agname without loading the full image
            with tarfile.open(ckpt, "r:gz") as tar:
                state = json.loads(tar.extractfile("state.json").read())
            agname = state["agname"]

            if agname in live_names:
                # Already running — skip, return existing
                existing = live_names[agname]
                existing._term.log("CKPT     ", f"load_all: {agname} already live, skipping {ckpt.name}")
                restored.append(existing)
            else:
                ag = cls.load(ckpt, llm_config=llm_config, agskills=agskills, tools=tools)
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
        runtime = self.sandbox._runtime
        image_tag = f"agency/ckpt-{self.agname}"

        # 1. Wait for any in-flight run() to complete so history and the
        #    container filesystem are in a consistent, quiescent state.
        if self._history.is_pending():
            self._term.log("CKPT ⏳  ", "waiting for in-flight task to complete...")
        self._history._resolve()

        # 2. Snapshot the now-idle container to an image
        self.sandbox._run([runtime, "commit", self.sandbox._container_name(), image_tag], check=True)

        try:
            # 2. Export image to bytes
            result = subprocess.run(
                [runtime, "save", image_tag],
                capture_output=True, check=True, timeout=600,
            )
            image_bytes = result.stdout

            # 3. Build state dict
            state = {
                "agname":      self.agname,
                "llm_config":  {k: v for k, v in self.llm_config.items() if k != "api_key"},
                "history":     self._history._data.get("messages", []),
                "skill_names": [s.name for s in self.agskills],
                "ts":          _ts(),
            }
            state_bytes = json.dumps(state, indent=2).encode()

            # 4. Bundle into a single .tar.gz
            path.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(path, "w:gz") as tar:
                for name, data in [("state.json", state_bytes), ("container.tar", image_bytes)]:
                    info = tarfile.TarInfo(name=name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))

        finally:
            subprocess.run([runtime, "rmi", "-f", image_tag], capture_output=True)

        size_kb = path.stat().st_size // 1024
        self._term.log("CKPT ✓   ", f"saved → {path}  ({size_kb} KB)")
        self.log._lifecycle("saved", agname=self.agname, path=str(path), size_kb=size_kb)

    @classmethod
    def load(
        cls,
        path: "Path | str",
        llm_config: dict,
        agskills: "list[agskill] | None" = None,
        tools: "list[agtool] | None" = None,
    ) -> "agent":
        """Restore an agent from a checkpoint file created by ``agent.save()``.

        Always creates a **new** agent instance — it never modifies or replaces
        any currently live agent, even if an agent with the same agname already
        exists.  If the checkpoint's agname is already taken in this process,
        ``ValueError`` is raised; discard the conflicting agent first or use
        ``agent.load_all()`` which handles this automatically.

        The caller must re-supply ``llm_config`` (api_key is never stored) and
        ``agskills`` (Python code is not serialised).  The restored agent has
        the same conversation history and container filesystem as at checkpoint
        time and can continue running skills immediately.
        """
        path = Path(path)
        runtime = get_container_runtime()
        image_tag = f"agency/ckpt-restore-{_uuid_mod.uuid4().hex[:8]}"

        with tarfile.open(path, "r:gz") as tar:
            state       = json.loads(tar.extractfile("state.json").read())
            image_bytes = tar.extractfile("container.tar").read()

        # Load image — docker restores the original tag (agency/ckpt-{agname})
        subprocess.run(
            [runtime, "load"],
            input=image_bytes, capture_output=True, check=True, timeout=600,
        )
        original_tag = f"agency/ckpt-{state['agname']}"
        # Re-tag to a unique name so concurrent restores don't collide,
        # then remove the original tag
        subprocess.run([runtime, "tag", original_tag, image_tag], capture_output=True, check=True)
        subprocess.run([runtime, "rmi", original_tag], capture_output=True)

        # Build agent without going through normal __init__ to avoid creating a fresh container
        ag: agent = cls.__new__(cls)
        ag.agname    = _register_agname(state["agname"])
        ag.llm_config = {**state.get("llm_config", {}), **llm_config}
        ag.agskills   = list(agskills or [])
        ag._history   = agdata(messages=list(state.get("history", [])))

        _out = Path(agent.output_dir) / ag.agname if agent.output_dir else None
        ag.sandbox = agSandbox(ag.agname, restore_image=image_tag, output_dir=_out)

        pool = agent.agresource_pool
        ag.tools = list(tools) if tools is not None else make_sandboxed_tools(ag.sandbox, pool)

        log_dir  = Path(agent.log_dir) if agent.log_dir is not None else _DEFAULT_LOG_DIR
        ag.log   = aglog(path=log_dir / f"{ag.agname}.jsonl")
        ag._term = agterm(ag.agname)

        for t in ag.tools:
            t.attach_logger(ag._term, ag.log)

        _live_agents.add(ag)

        ag._term.log("LOADED   ", f"from {path}  skills={state.get('skill_names', [])}")
        ag.log._lifecycle("loaded", agname=ag.agname, source=str(path), checkpoint_ts=state.get("ts"))

        return ag

    def __repr__(self) -> str:
        names = [f.name for f in self.agskills]
        return f"agent(agname={self.agname!r}, agskills={names!r})"
