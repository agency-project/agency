from __future__ import annotations
import copy
import io
import json
import queue
import shlex
import subprocess
import tarfile
import threading
import uuid as _uuid_mod
import weakref
from concurrent.futures import Future
from datetime import datetime
from pathlib import Path
from typing import ClassVar, get_args, get_origin

# Single run-level ID for the default log directory.
# Created once at import time so all agents in one process share it.
_RUN_ID  = _uuid_mod.uuid4().hex[:12]
_RUN_TS  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
_DEFAULT_LOG_DIR = Path(f"/tmp/agency/{_RUN_TS}_{_RUN_ID}")

# Global weak registry of all live agent instances.
# WeakSet entries disappear automatically when agents are garbage-collected.
_live_agents: "weakref.WeakSet[agent]" = weakref.WeakSet()

# Round-robin counter for multi-server llm_config lists.
_llm_config_counter: int = 0
_llm_config_lock: threading.Lock = threading.Lock()


def _pick_llm_config(llm_config: "dict | list[dict]") -> dict:
    """Return a single config dict, round-robining across a list."""
    if not isinstance(llm_config, list):
        return llm_config
    global _llm_config_counter
    with _llm_config_lock:
        idx = _llm_config_counter % len(llm_config)
        _llm_config_counter += 1
    return llm_config[idx]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT_SAVE_TIMEOUT_S = 600  # Timeout in seconds for `subprocess.run` when exporting a container image during agent.save().
CHECKPOINT_LOAD_TIMEOUT_S = 600  # Timeout in seconds for `subprocess.run` when loading a container image during agent.load().
SKILL_ERROR_LOG_TRUNCATE = 80  # Maximum characters of an error string shown in the terminal log line after a skill failure.

from .agdata import agdata, _fmt_exc
from .agtype import agtype
from .agskill import agskill, AGSKILL_REACT_MAX_STEPS
from .agtool import agtool
from .aglog import aglog, _ts
from .agterm import agterm
from .agsandbox import agSandbox, get_container_runtime, _RUN_ID as _SANDBOX_RUN_ID
from .agresources import agResourcePool
from .agcompaction import fetch_context_limit, _prune_tool_outputs

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

# Lowercase alphanumeric alphabet used for agent ID suffixes.
# 4 digits → 36⁴ = 1 679 616 unique values per noun.
_B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def _b36_suffix(n: int, width: int = 4) -> str:
    """Encode *n* as a fixed-width base-36 string (0000…0009, 000a…)."""
    base = len(_B36)
    digits = []
    for _ in range(width):
        digits.append(_B36[n % base])
        n //= base
    return "".join(reversed(digits))


def _register_agname(full_name: str) -> str:
    """Register an already-final name as in-use, raising if taken."""
    with _agname_lock:
        if full_name in _allocated_agnames:
            raise ValueError(f"agname {full_name!r} is already in use by another agent")
        _allocated_agnames.add(full_name)
    return full_name


def _allocate_agname(name: str) -> str:
    """Return a unique name in the form <name>_XXX and register it as in-use.

    XXX is a 4-character base-36 suffix (1 679 616 unique values per noun).
    """
    with _agname_lock:
        n = _noun_counters.get(name, 0)
        _noun_counters[name] = n + 1
        full = f"{name}_{_b36_suffix(n)}"
        _allocated_agnames.add(full)
    return full


def _generate_agname() -> str:
    """Return a unique agname in the form <noun>_XXX.

    Nouns are assigned in order from _NOUNS, cycling back to the start after
    the last entry. The suffix is a 5-character base-36 string (60 466 176
    unique values per noun), so arch_00000 and arch_00001 are the first and
    second agents that received 'arch', giving names like arch_0000, arch_0001.
    """
    global _noun_index
    with _agname_lock:
        noun = _NOUNS[_noun_index % len(_NOUNS)]
        _noun_index += 1
        n = _noun_counters.get(noun, 0)
        _noun_counters[noun] = n + 1
        name = f"{noun}_{_b36_suffix(n)}"
        _allocated_agnames.add(name)
    return name


# String fields longer than this many characters are offloaded to a file in
# the agent's sandbox instead of being inlined in the LLM context window.
INPUT_OFFLOAD_CHARS: int = 2000


def _offload_large_fields(
    inp: agdata, sandbox: "agSandbox", skill_name: str,
    schema: "agdata | None" = None,
    suffix: str = "",
) -> tuple[list[str], list[str]]:
    """Write oversized string fields to /workspace/inputs/ in the sandbox.

    Called after agtype fields have already been prepared (so agfile inputs are
    already short file paths).  Each remaining field whose string value still
    exceeds INPUT_OFFLOAD_CHARS is replaced in-place with a short reference.
    Returns (paths_written, field_names) so the caller can delete files and
    build an auto-offload note for the system prompt.

    Fields already managed by an agtype subclass (e.g. agimage data URLs) are
    skipped so their prepared values are not replaced by sandbox file references.
    """
    from .agtype import agtype
    agtype_keys: set[str] = set()
    if schema is not None:
        for key, hint in schema._data.items():
            if isinstance(hint, type) and issubclass(hint, agtype):
                agtype_keys.add(key)
            elif get_origin(hint) is list:
                args = get_args(hint)
                if args and isinstance(args[0], type) and issubclass(args[0], agtype):
                    agtype_keys.add(key)

    paths: list[str] = []
    fields: list[str] = []
    for key, val in list(inp._data.items()):
        if key in agtype_keys:
            continue
        if isinstance(val, str):
            if len(val) <= INPUT_OFFLOAD_CHARS:
                continue
            path = f"/workspace/inputs/{skill_name}_{key}{suffix}.txt"
            try:
                sandbox.write_file(path, val)
                inp._data[key] = (
                    f"(content saved to {path} — use the read tool to access it)"
                )
                paths.append(path)
                fields.append(key)
            except Exception as _e:
                print(f"[agent] WARNING: failed to offload input field '{key}' to {path}: {_e}")
        elif isinstance(val, list):
            new_vals = list(val)
            offloaded_any = False
            for i, item in enumerate(val):
                if not isinstance(item, str) or len(item) <= INPUT_OFFLOAD_CHARS:
                    continue
                path = f"/workspace/inputs/{skill_name}_{key}_{i}{suffix}.txt"
                try:
                    sandbox.write_file(path, item)
                    new_vals[i] = path
                    paths.append(path)
                    offloaded_any = True
                except Exception as _e:
                    print(f"[agent] WARNING: failed to offload input list field '{key}[{i}]' to {path}: {_e}")
            if offloaded_any:
                inp._data[key] = new_vals
                fields.append(key)
    return paths, fields


def _prepare_agtype_inputs(
    inp: agdata, schema: "agdata | None", sandbox: "agSandbox", skill_name: str,
    suffix: str = "",
) -> list[str]:
    """Prepare agtype input fields before the skill runs.

    For each schema field whose hint is an agtype subclass (or list[agtype]),
    calls ``hint.prepare()`` which may transform the value and write sandbox
    files.  Returns all paths written for cleanup.
    """
    if schema is None:
        return []
    paths: list[str] = []
    for key, hint in schema._data.items():
        # Direct agtype subclass
        if isinstance(hint, type) and issubclass(hint, agtype):
            val = inp._data.get(key)
            try:
                new_val, written = hint.prepare(val, sandbox, skill_name, key, suffix=suffix)
                inp._data[key] = new_val
                paths.extend(written)
            except Exception as _e:
                print(f"[agent] WARNING: {hint.__name__}.prepare failed for field '{key}': {_e}")
        # list[agtype subclass]
        elif get_origin(hint) is list:
            args = get_args(hint)
            if args and isinstance(args[0], type) and issubclass(args[0], agtype):
                inner = args[0]
                vals = inp._data.get(key)
                if isinstance(vals, list):
                    new_vals = []
                    for v in vals:
                        try:
                            new_v, written = inner.prepare(v, sandbox, skill_name, key, suffix=suffix)
                            paths.extend(written)
                        except Exception as _e:
                            print(f"[agent] WARNING: {inner.__name__}.prepare failed for list field '{key}': {_e}")
                            new_v = v
                        new_vals.append(new_v)
                    inp._data[key] = new_vals
    return paths


def _recover_agtype_outputs(
    result: agdata, schema: "agdata | None", sandbox: "agSandbox"
) -> list[str]:
    """Recover agtype output fields after the skill finishes.

    For each schema field whose hint is an agtype subclass, calls
    ``hint.recover()`` which may read sandbox files back into Python values.
    Returns all paths written for cleanup.
    """
    if schema is None or result.is_error():
        return []
    paths: list[str] = []
    for key, hint in schema._data.items():
        if not (isinstance(hint, type) and issubclass(hint, agtype)):
            continue
        val = result._data.get(key)
        try:
            new_val, written = hint.recover(val, sandbox)
            result._data[key] = new_val
            paths.extend(written)
        except Exception as _e:
            print(f"[agent] WARNING: {hint.__name__}.recover failed for field '{key}': {_e}")
    return paths


def _remove_offloaded_fields(paths: list[str], sandbox: "agSandbox") -> None:
    """Delete files previously written by offload/agfile helpers."""
    for path in paths:
        try:
            sandbox._container_exec(f"rm -f {shlex.quote(path)}", shell="sh")
        except Exception as _e:
            print(f"[agent] WARNING: failed to remove offloaded file {path}: {_e}")


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
        llm_config: "dict | agent | None" = None,
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

        # Resolve a list of configs to a single one via round-robin.
        llm_config = _pick_llm_config(llm_config)

        self.agname = _generate_agname() if agname is None else _allocate_agname(agname)
        pool = agent.agresource_pool

        if isinstance(llm_config, agent):
            src = llm_config
            self.llm_config    = src.llm_config
            self._context_limit: int | None = src._context_limit
            # Block until source's in-flight task finishes, then deep-copy history
            src._history._resolve()
            self._history: agdata = copy.deepcopy(src._history)
            # Copy parent's checkpoint as this fork's starting state — no docker run yet
            self._checkpoint: str | None = None
            if src._checkpoint:
                fork_tag = f"agency/lifecycle-{_SANDBOX_RUN_ID}-{self.agname}"
                subprocess.run(
                    [get_container_runtime(), "tag", src._checkpoint, fork_tag],
                    capture_output=True, check=True,
                )
                self._checkpoint = fork_tag
            self.sandbox: agSandbox | None = None
        else:
            self.llm_config    = llm_config
            self._context_limit = fetch_context_limit(llm_config)
            self._history      = agdata(messages=[])
            self._checkpoint: str | None = None
            self.sandbox:     agSandbox | None = None

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

        if isinstance(llm_config, agent):
            self._term.log("FORKED   ", f"from {src.agname}")
            self.log._lifecycle(
                "forked",
                agname=self.agname,
                parent_agname=src.agname,
                team=team_name,
                llm_config={k: v for k, v in self.llm_config.items() if k != "api_key"},
            )
        else:
            ctx = f"  context={self._context_limit}" if self._context_limit else "  context=unknown"
            team_tag = f"  team={team_name}" if team_name else ""
            self._term.log("CREATED  ", f"model={self.llm_config.get('model','?')}{ctx}{team_tag}")
            self.log._lifecycle(
                "created",
                agname=self.agname,
                team=team_name,
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
        self.llm_config = dict(llm_config)
        self._context_limit = fetch_context_limit(self.llm_config)

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
            _offloaded_paths: list[str] = []   # declared before try for reliable finally cleanup
            _out = Path(agent.output_dir) / self.agname if agent.output_dir else None
            try:
                prev_history._resolve()
                _resolve_input(input)
                self.sandbox = agSandbox(self.agname, lifecycle_image=self._checkpoint, output_dir=_out)
                self._checkpoint = None

                history_before = list(prev_history._data.get("messages", []))

                self._term.log("SKILL ▶  ", f"{skill_name}  input={list(input._data.keys())}")
                self._set_ui_state("skill", skill=skill_name)

                # Snapshot cumulative log usage before this skill so the live
                # token callback can compute the correct agent-total mid-skill.
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

                # Prepare agtype fields first (agfile → file path), then offload
                # any remaining oversized plain-string fields.
                # Timestamp suffix ensures each invocation writes to a unique path
                # so persistent agents always see new file names and re-read them.
                import time as _time
                _input_suffix = f"_{int(_time.time() * 1000)}"
                _offloaded_paths.extend(
                    _prepare_agtype_inputs(input, af.input_schema, self.sandbox, skill_name,
                                           suffix=_input_suffix)
                )
                auto_paths, auto_fields = _offload_large_fields(
                    input, self.sandbox, skill_name, schema=af.input_schema,
                    suffix=_input_suffix
                )
                _offloaded_paths.extend(auto_paths)

                # Build extra system prompt note for auto-offloaded fields so the
                # agent knows they are temporary and how to access them.
                _extra_system: str | None = None
                if auto_fields:
                    field_list = ", ".join(f"`{f}`" for f in auto_fields)
                    _extra_system = (
                        f"\nNote: The following input fields contain large content "
                        f"that has been automatically saved to temporary files in "
                        f"your sandbox: {field_list}. The file paths are shown in "
                        f"the input JSON. Use the read tool to access the full "
                        f"content. WARNING: these files are temporary and will be "
                        f"automatically deleted after this task ends."
                    )

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

                outer_input_tokens  = 0
                outer_output_tokens = 0
                outer_result, outer_history, outer_delta, _tok = af.run(
                    self.llm_config, input, prev_history,
                    self.sandbox, pool, max_steps, term=self._term, log=self.log,
                    _state_fn=self._set_ui_state,
                    _live_messages_fn=self._push_live_messages,
                    _inbox_fn=_drain_inbox,
                    _context_limit=self._context_limit,
                    _compact_log_fn=_compact_log,
                    _full_history_fn=self._append_full_history,
                    _extra_system=_extra_system,
                    _token_update_fn=_live_token_update,
                    _ping_interval_s=agent.ping_interval_s,
                    _poll_interval_s=agent.poll_interval_s,
                    _agname=self.agname,
                )
                outer_input_tokens  = _tok[0]
                outer_output_tokens = _tok[1]

                # Recover agtype output fields from sandbox into the result agdata.
                if outer_result is not None:
                    _offloaded_paths.extend(
                        _recover_agtype_outputs(outer_result, af.output_schema, self.sandbox)
                    )

            except Exception as exc:
                outer_result  = agdata(error=_fmt_exc(exc))
                outer_history = prev_history
                outer_delta   = []
                history_before = list(prev_history._data.get("messages", []))
                self._term.log("SKILL ✗  ", f"{skill_name}  exception={exc}")
            finally:
                _had_error = outer_result is not None and bool(outer_result._data.get("error"))
                self._set_ui_state("error" if _had_error else "finished")
                if self.sandbox is not None:
                    _remove_offloaded_fields(_offloaded_paths, self.sandbox)
                    if self.sandbox._gpu_id is not None:
                        pool.release_gpu(self.sandbox._gpu_id)
                    if self.sandbox._lifecycle_image:
                        self._checkpoint = self.sandbox._lifecycle_image
                    self.sandbox.destroy()
                    self.sandbox = None

            # Log and resolve futures after all background work is done
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

            self._snapshot_messages = list(outer_history._data.get("messages", []))
            result_future.set_result(outer_result)

            # Post-skill history pruning — trim old oversized tool outputs from
            # the shared history before unblocking the next run() on this agent.
            # Runs after result_future so the caller can unblock immediately;
            # history_future holds until pruning is done so the dependency chain
            # sees clean history.
            try:
                pruned_msgs = _prune_tool_outputs(
                    outer_history._data.get("messages", [])
                )
                if pruned_msgs is not outer_history._data.get("messages", []):
                    outer_history = agdata(messages=pruned_msgs)
                    self._term.log("PRUNE    ", f"{skill_name}  history pruned to {len(pruned_msgs)} msgs")
            except Exception as prune_exc:
                self._term.log("PRUNE ✗  ", f"{skill_name}  pruning failed: {prune_exc}")

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
        try:
            if self.sandbox is not None:
                self.sandbox.destroy()
        except Exception as _e:
            print(f"[agent] WARNING: sandbox.destroy() failed in __del__ for {getattr(self, 'agname', '?')}: {_e}")
        try:
            if self._checkpoint:
                subprocess.run(
                    [get_container_runtime(), "rmi", "-f", self._checkpoint],
                    capture_output=True,
                )
        except Exception as _e:
            print(f"[agent] WARNING: checkpoint rmi failed in __del__ for {getattr(self, 'agname', '?')}: {_e}")

    def fork(self) -> "agent":
        """Return an independent copy of this agent (same as agent(self))."""
        return agent(self)

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
        runtime = get_container_runtime()
        image_tag = f"agency/ckpt-{self.agname}"

        if self._history.is_pending():
            self._term.log("CKPT ⏳  ", "waiting for in-flight task to complete...")
        self._history._resolve()

        # Build state dict
        state = {
            "agname":     self.agname,
            "llm_config": {k: v for k, v in self.llm_config.items() if k != "api_key"},
            "history":    self._history._data.get("messages", []),
            "ts":         _ts(),
        }
        state_bytes = json.dumps(state, indent=2).encode()

        path.parent.mkdir(parents=True, exist_ok=True)

        if self._checkpoint is not None:
            # Retag checkpoint for export — keeps _checkpoint intact for next task
            subprocess.run(
                [runtime, "tag", self._checkpoint, image_tag],
                capture_output=True, check=True,
            )
            try:
                result = subprocess.run(
                    [runtime, "save", image_tag],
                    capture_output=True, check=True, timeout=CHECKPOINT_SAVE_TIMEOUT_S,
                )
                image_bytes = result.stdout
                with tarfile.open(path, "w:gz") as tar:
                    for name, data in [("state.json", state_bytes), ("container.tar", image_bytes)]:
                        info = tarfile.TarInfo(name=name)
                        info.size = len(data)
                        tar.addfile(info, io.BytesIO(data))
            finally:
                subprocess.run([runtime, "rmi", "-f", image_tag], capture_output=True)
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
        runtime = get_container_runtime()
        image_tag = f"agency/ckpt-restore-{_uuid_mod.uuid4().hex[:8]}"

        with tarfile.open(path, "r:gz") as tar:
            state       = json.loads(tar.extractfile("state.json").read())
            container_member = next((m for m in tar.getmembers() if m.name == "container.tar"), None)
            image_bytes = tar.extractfile(container_member).read() if container_member else None

        checkpoint: str | None = None
        if image_bytes is not None:
            # Load image — docker restores the original tag (agency/ckpt-{agname})
            subprocess.run(
                [runtime, "load"],
                input=image_bytes, capture_output=True, check=True, timeout=CHECKPOINT_LOAD_TIMEOUT_S,
            )
            original_tag = f"agency/ckpt-{state['agname']}"
            # Re-tag to a unique name so concurrent restores don't collide,
            # then remove the original tag
            subprocess.run([runtime, "tag", original_tag, image_tag], capture_output=True, check=True)
            subprocess.run([runtime, "rmi", original_tag], capture_output=True)
            checkpoint = image_tag

        # Build agent without going through normal __init__ to avoid creating a fresh container
        ag: agent = cls.__new__(cls)
        ag.agname        = _register_agname(state["agname"])
        ag.llm_config    = {**state.get("llm_config", {}), **llm_config}
        ag._history      = agdata(messages=list(state.get("history", [])))
        ag._context_limit = fetch_context_limit(ag.llm_config)
        ag._checkpoint   = checkpoint  # None for history-only saves; consumed by first _task()
        ag.sandbox       = None

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
