from __future__ import annotations
import copy
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import ClassVar
from .agdata import agdata
from .agskill import agskill
from .agtool import agtool
from .aglog import aglog, _ts
from .agterm import agterm
from .agsandbox import agSandbox
from .agresources import agResourcePool
from .tools import make_sandboxed_tools


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

    _pool: ClassVar[ThreadPoolExecutor] = ThreadPoolExecutor()
    log_dir:          ClassVar[Path | None]          = None
    output_dir:       ClassVar[Path | None]          = None
    agresource_pool:  ClassVar[agResourcePool]       = agResourcePool()
    ping_interval_s:  ClassVar[int]                  = 300
    poll_interval_s:  ClassVar[int]                  = 5
    max_outer_iters:  ClassVar[int]                  = 144

    def __init__(
        self,
        llm_config: "dict | agent",
        agskills: list[agskill] | None = None,
        tools: list[agtool] | None = None,
    ):
        self.uuid = str(uuid.uuid4())
        pool = agent.agresource_pool

        if isinstance(llm_config, agent):
            src = llm_config
            self.llm_config = src.llm_config
            self.agskills   = list(agskills if agskills is not None else src.agskills)
            # Block until source's in-flight task finishes, then deep-copy history
            src._history._resolve()
            self._history: agdata = copy.deepcopy(src._history)
            # Snapshot parent container → fork starts from parent's exact state
            self.sandbox = agSandbox(self.uuid, parent_uuid=src.uuid,
                                     output_dir=Path(agent.output_dir) if agent.output_dir else None)
        else:
            self.llm_config = llm_config
            self.agskills   = list(agskills or [])
            self._history   = agdata(messages=[])
            self.sandbox    = agSandbox(self.uuid,
                                        output_dir=Path(agent.output_dir) if agent.output_dir else None)

        # Build sandboxed tool list; user-supplied tools override if provided
        if tools is not None:
            self.tools = list(tools)
        else:
            self.tools = make_sandboxed_tools(self.sandbox, pool)

        log_path = Path(agent.log_dir) / f"{self.uuid}.jsonl" if agent.log_dir is not None else None
        self.log  = aglog(path=log_path)
        self._term = agterm(self.uuid)

        # Wire terminal + file logging into every tool.
        for t in self.tools:
            t.attach_logger(self._term, self.log)

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

        The future resolves only after:
        1. The ReAct loop finishes.
        2. All background processes started via ``bash`` have exited.
        3. All acquired resources (GPU/CPU) have been released.

        Calls on the same agent are serialized via the history chain.
        Calls on different agents (forks) run concurrently.
        """
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
                    result, new_history, history_delta = af.run(
                        self.llm_config, current_input, current_history,
                        self.tools, max_steps, term=self._term,
                        _is_continuation=is_continuation,
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
                    self._term.log("PROCS ▶  ", f"{skill_name}  monitoring: {summary}")
                    self.log._lifecycle("procs_started", uuid=self.uuid,
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
                        self.log._lifecycle("procs_completed", uuid=self.uuid,
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
                    self.log._lifecycle("procs_ping", uuid=self.uuid,
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

            result_future.set_result(outer_result)
            history_future.set_result(outer_history)

        agent._pool.submit(_task)

        self._history = agdata(_future=history_future)
        return agdata(_future=result_future)

    # ------------------------------------------------------------------
    # Backward-compat fork helper
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        """Best-effort: log destruction and destroy the sandbox container."""
        try:
            self._term.log("DESTROYED", "")
            self.log._lifecycle("destroyed", uuid=self.uuid)
        except Exception:
            pass
        try:
            self.sandbox.destroy()
        except Exception:
            pass

    def fork(self) -> "agent":
        """Return an independent copy of this agent (same as agent(self))."""
        return agent(self)

    def __repr__(self) -> str:
        names = [f.name for f in self.agskills]
        return f"agent(uuid={self.uuid[:8]!r}, agskills={names!r})"
