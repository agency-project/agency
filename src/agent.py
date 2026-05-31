from __future__ import annotations
import copy
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import ClassVar
from .agdata import agdata
from .agskill import agskill
from .agtool import agtool
from .aglog import aglog, _ts
from .agterm import agterm


def _resolve_input(inp: agdata) -> None:
    """Resolve any pending agdata values nested inside inp, in-place.

    Handles:
    - inp itself being pending (resolves before inspecting fields)
    - top-level field values that are pending agdata
    - list fields whose elements are pending agdata
    """
    inp._resolve()
    for key, val in list(inp._data.items()):
        if isinstance(val, agdata):
            val._resolve()
        elif isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, agdata):
                    item._resolve()


class agent:
    """Orchestrator that maintains shared history and delegates to named agskills.

    An agent holds:
    - llm_config  : provider/model settings passed to every agskill
    - history     : shared conversation context (agdata with a 'messages' list)
    - tools       : pool of tool objects available to agskills by default
    - agskills    : list of agskill objects the user can invoke by name

    agent.run(skill_name, input)
        Always non-blocking.  Returns a pending agdata immediately.  Accessing
        any field of the result blocks until the skill completes.

        Calls on the *same* agent object are automatically serialized through
        the history chain — each call waits for the previous one's history
        update before starting.  This ensures sequential coherence without
        any explicit synchronization by the caller.

    agent(existing_agent)
        Copy constructor.  Blocks until the source agent's in-flight task
        completes, then deep-copies the resolved history.  The new agent is
        fully independent — its subsequent run() calls do not affect the
        original agent's history chain.

    Parallelism pattern:
        # These run concurrently — each fork is independent
        results = [agent(ag).run("skill", inp) for inp in many_inputs]
        # Accessing result fields blocks until each finishes
        values  = [r.value for r in results]
    """

    _pool: ClassVar[ThreadPoolExecutor] = ThreadPoolExecutor()  # shared across all instances
    log_dir: ClassVar[Path | None] = None  # set once; all agents (incl. forks) log here

    def __init__(
        self,
        llm_config: "dict | agent",
        agskills: list[agskill] | None = None,
        tools: list[agtool] | None = None,
    ):
        self.uuid = str(uuid.uuid4())
        if isinstance(llm_config, agent):
            src = llm_config
            self.llm_config = src.llm_config
            self.agskills   = list(agskills if agskills is not None else src.agskills)
            self.tools      = list(tools    if tools    is not None else src.tools)
            # Block until source's in-flight task finishes, then deep-copy
            src._history._resolve()
            self._history: agdata = copy.deepcopy(src._history)
        else:
            self.llm_config = llm_config
            self.agskills   = list(agskills or [])
            self.tools      = list(tools or [])
            self._history   = agdata(messages=[])
        log_path = Path(agent.log_dir) / f"{self.uuid}.jsonl" if agent.log_dir is not None else None
        self.log  = aglog(path=log_path)
        self._term = agterm(self.uuid)
        if isinstance(llm_config, agent):
            self._term.log("FORKED   ", f"from {src.uuid[:8]}  skills={[s.name for s in self.agskills]}")
            self.log._lifecycle(
                "forked",
                uuid=self.uuid,
                parent_uuid=src.uuid,
                agskills=[s.name for s in self.agskills],
                tools=[t.name for t in self.tools],
                llm_config={k: v for k, v in self.llm_config.items() if k != "api_key"},
            )
        else:
            self._term.log("CREATED  ", f"skills={[s.name for s in self.agskills]}  model={self.llm_config.get('model','?')}")
            self.log._lifecycle(
                "created",
                uuid=self.uuid,
                agskills=[s.name for s in self.agskills],
                tools=[t.name for t in self.tools],
                llm_config={k: v for k, v in self.llm_config.items() if k != "api_key"},
            )

    # ------------------------------------------------------------------
    # History property — blocks until the current chain link resolves
    # ------------------------------------------------------------------

    @property
    def history(self) -> agdata:
        """Return the current history, blocking until any in-flight task finishes."""
        self._history._resolve()
        return self._history

    @history.setter
    def history(self, value: agdata) -> None:
        self._history = value

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, skill_name: str, input: agdata, max_steps: int = 10) -> agdata:
        """Submit the skill and return a pending agdata immediately.

        Calls on the same agent are automatically serialized via the history
        chain.  Calls on different agents (forks created with agent(self))
        run concurrently.
        """
        prev_history = self._history  # capture current chain link

        result_future: Future[agdata] = Future()
        history_future: Future[agdata] = Future()
        ts_start = _ts()

        def _task() -> None:
            try:
                # Serialize: wait for the previous task in this chain to finish
                prev_history._resolve()
                # Resolve any pending inputs (including list fields)
                _resolve_input(input)

                history_before = list(prev_history._data.get("messages", []))

                af = next((f for f in self.agskills if f.name == skill_name), None)
                if af is None:
                    err = agdata(error=f"agskill not found: {skill_name!r}")
                    self._term.log("SKILL ✗  ", f"{skill_name!r} not found")
                    self.log._record(skill_name, ts_start, _ts(),
                                     input.to_dict(), err.to_dict(), len(history_before),
                                     history_before=history_before, history_delta=[])
                    result_future.set_result(err)
                    history_future.set_result(prev_history)
                    return

                self._term.log("SKILL ▶  ", f"{skill_name}  input={list(input._data.keys())}")
                result, new_history, history_delta = af.run(
                    self.llm_config, input, prev_history, self.tools, max_steps,
                    term=self._term,
                )
                ts_end = _ts()
                # Serialise inputs/outputs now — handles nested agdata (e.g. lists of agdata)
                input_dict  = input.to_dict()
                result_dict = result.to_dict()
                if result_dict.get("error"):
                    self._term.log("SKILL ✗  ", f"{skill_name}  error={str(result_dict['error'])[:80]}")
                else:
                    self._term.log("SKILL ✓  ", f"{skill_name}  output={list(result_dict.keys())}")
                # Log to file; wrap so a logging failure never kills the task
                try:
                    self.log._record(skill_name, ts_start, ts_end,
                                     input_dict, result_dict,
                                     len(new_history._data.get("messages", [])),
                                     history_before=history_before,
                                     history_delta=history_delta)
                except Exception as log_exc:
                    self._term.log("SKILL ✗  ", f"[log error] {log_exc}")
                result_future.set_result(result)
                history_future.set_result(new_history)
            except Exception as exc:
                err = agdata(error=str(exc))
                history_before = list(prev_history._data.get("messages", []))
                self._term.log("SKILL ✗  ", f"{skill_name}  exception={exc}")
                if not result_future.done():
                    try:
                        self.log._record(skill_name, ts_start, _ts(),
                                         input.to_dict(), err.to_dict(), len(history_before),
                                         history_before=history_before, history_delta=[])
                    except Exception:
                        pass
                    result_future.set_result(err)
                if not history_future.done():
                    history_future.set_result(prev_history)

        agent._pool.submit(_task)

        # Advance the history chain link for subsequent calls on this agent
        self._history = agdata(_future=history_future)
        return agdata(_future=result_future)

    # ------------------------------------------------------------------
    # Backward-compat fork helper
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        """Best-effort: log destruction when the agent is garbage-collected."""
        try:
            self._term.log("DESTROYED", "")
            self.log._lifecycle("destroyed", uuid=self.uuid)
        except Exception:
            pass  # never raise in __del__

    def fork(self) -> "agent":
        """Return an independent copy of this agent (same as agent(self))."""
        return agent(self)

    def __repr__(self) -> str:
        names = [f.name for f in self.agskills]
        return f"agent(uuid={self.uuid[:8]!r}, agskills={names!r})"
