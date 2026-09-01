from __future__ import annotations

import atexit
import heapq
import itertools
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..agcollector import GlobalDataCollector, ScopedDataCollector
from ..agconfig import StaticConfigParam, _AgConfigViewBase
from ..agcontext import agcontext
from ..agdata import agdata, agerror
from ..engine import AgentEngine
from ..agDataCollector import _ts
from ..agutil import format_exception
from ..profiler import agprof
from .scheduler import ExecutionScheduler

if TYPE_CHECKING:
    from ..agconfig import agConfig
    from ..agent import agent
    from ..agskill import agskill


class _AgOrchestratorFields:
    max_concurrent_engines = StaticConfigParam("agorchestrator", default=None)
    db_path = StaticConfigParam("agorchestrator", default=None)
    flush_batch_size = StaticConfigParam("agorchestrator", default=500)
    flush_interval_s = StaticConfigParam("agorchestrator", default=1.0)

    def __init__(self, agconfig: "agConfig | None" = None) -> None:
        self._agconfig = agconfig


class agOrchestratorConfig(_AgConfigViewBase):
    """Configuration for the process-wide global agent orchestrator."""

    _OWNER = "agorchestrator"


@dataclass(frozen=True)
class OrchestratorSnapshot:
    state: str
    max_concurrent_engines: "int | None"
    ready_count: int
    blocked_count: int
    running_count: int
    submitted_total: int
    completed_total: int
    failed_total: int
    agents: dict
    db_path: str
    persistence_error: "str | None" = None
    telemetry_error: "str | None" = None


@dataclass
class _ExecutionRequest:
    request_id: str
    sequence: int
    agent: "agent"
    skill: "agskill"
    skill_input: agdata
    max_steps: "int | None"
    result_future: "Future[agdata]"
    parent_context: object
    ts_start: str
    submitted_perf_ns: int
    submitted_wall_ns: int
    state: str = "submitted"
    dependencies: set[Future] = field(default_factory=set)
    producer_ids: set[str] = field(default_factory=set)
    scoped_collector: "ScopedDataCollector | None" = None
    engine: "AgentEngine | None" = None
    run_span: object = None
    phase_span: object = None
    phase_name: "str | None" = None
    phase_started_wall: "float | None" = None
    engine_started_wall: "float | None" = None


@dataclass
class _RunCompletion:
    output: agdata
    context: agcontext
    failed: bool
    error_message: str = ""


class GlobalAgentOrchestrator(_AgOrchestratorFields):
    """Event-driven process-wide scheduler for agent engine executions."""

    def __init__(
        self,
        agconfig: "agConfig | None" = None,
        *,
        default_db_path: "str | Path",
    ) -> None:
        super().__init__(agconfig)
        if self.max_concurrent_engines is not None and (
            not isinstance(self.max_concurrent_engines, int)
            or isinstance(self.max_concurrent_engines, bool)
            or self.max_concurrent_engines <= 0
        ):
            raise ValueError("max_concurrent_engines must be a positive integer or None")
        db_path = self.db_path if self.db_path is not None else default_db_path
        self.data_collector = GlobalDataCollector(
            db_path,
            flush_batch_size=self.flush_batch_size,
            flush_interval_s=self.flush_interval_s,
        )
        self.data_collector.start()
        try:
            from .. import agwebui as _agwebui

            if _agwebui._active is not None:
                _agwebui._active.emitter.bind_collector(self.data_collector)
        except Exception as exc:
            print(f"[agorchestrator] WARNING: Web UI collector binding failed: {exc}")

        self._events: "list[tuple[int, str, object]]" = []
        self._event_cond = threading.Condition(threading.RLock())
        self._event_sequence = itertools.count()
        self._request_sequence = itertools.count()
        self._requests: dict[str, _ExecutionRequest] = {}
        self._future_producers: dict[Future, str] = {}
        self._active_by_agent: dict[agent, str] = {}
        self._outstanding_by_agent: dict[agent, set[str]] = {}
        self._submitted_total = 0
        self._completed_total = 0
        self._failed_total = 0
        self._accepting = True
        self._state = "running"
        self._shutdown_ack: "Future[None] | None" = None
        self._stop_loop = False
        self.scheduler = ExecutionScheduler(self)
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_main,
            daemon=True,
            name="agency-global-scheduler",
        )
        self._scheduler_thread.start()
        self.data_collector.record_event(
            "scheduler_started",
            {"max_concurrent_engines": self.max_concurrent_engines},
            source="orchestrator",
            do_update=True,
        )
        with self._event_cond:
            self._refresh_snapshot_locked()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        ag: "agent",
        skill: "agskill",
        skill_input: agdata,
        max_steps: "int | None" = None,
    ) -> agdata:
        result_future: Future[agdata] = Future()
        accepted: Future[None] = Future()
        called_from_scheduler = threading.current_thread() is self._scheduler_thread
        with self._event_cond:
            if not self._accepting:
                raise RuntimeError("global agent orchestrator is shut down")
            self._post_locked(
                "submit",
                (
                    ag,
                    skill,
                    skill_input,
                    max_steps,
                    result_future,
                    agprof.current_span_context(),
                    time.perf_counter_ns(),
                    time.time_ns(),
                    accepted,
                ),
            )
        # This waits only for registration on the scheduler thread, never for
        # dependencies, an engine slot, or execution itself.
        if not called_from_scheduler:
            accepted.result()
        return agdata(_future=result_future)

    def snapshot(self) -> OrchestratorSnapshot:
        with self._event_cond:
            data = self._snapshot_dict_locked()
        collector = self.data_collector.snapshot()
        return OrchestratorSnapshot(
            **data,
            db_path=str(self.data_collector.db_path),
            persistence_error=collector.get("persistence_error"),
            telemetry_error=collector.get("telemetry_error"),
        )

    def wait_for_agent(self, ag: "agent", timeout_s: "float | None" = None) -> None:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._event_cond:
            while self._outstanding_by_agent.get(ag):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"agent {ag.agname} did not become idle within {timeout_s}s")
                self._event_cond.wait(timeout=remaining)

    def is_agent_idle(self, ag: "agent") -> bool:
        with self._event_cond:
            return not self._outstanding_by_agent.get(ag)

    def is_agent_settled(self, ag: "agent") -> bool:
        with self._event_cond:
            outstanding = self._outstanding_by_agent.get(ag, set())
            if not outstanding:
                return True
            for request_id in outstanding:
                request = self._requests.get(request_id)
                if request is not None and request.state in ("ready", "running"):
                    return False
            return True

    def flush(self, timeout_s: "float | None" = None) -> None:
        self.data_collector.flush(timeout_s=timeout_s)

    def shutdown(self, wait: bool = True, timeout_s: "float | None" = None) -> None:
        with self._event_cond:
            if self._state == "stopped":
                return
            if self._shutdown_ack is None:
                self._shutdown_ack = Future()
                self._accepting = False
                self._post_locked("shutdown", self._shutdown_ack)
            ack = self._shutdown_ack
        if not wait:
            return
        ack.result(timeout=timeout_s)
        self._scheduler_thread.join(timeout=timeout_s)

    # ------------------------------------------------------------------
    # Scheduler loop and transitions
    # ------------------------------------------------------------------

    def _post(self, kind: str, value: object) -> None:
        with self._event_cond:
            self._post_locked(kind, value)

    def _post_locked(self, kind: str, value: object) -> None:
        heapq.heappush(self._events, (next(self._event_sequence), kind, value))
        self._event_cond.notify_all()

    def _scheduler_main(self) -> None:
        try:
            while True:
                with self._event_cond:
                    while not self._events and not self._stop_loop:
                        self._event_cond.wait()
                    if self._stop_loop:
                        break
                    _seq, kind, value = heapq.heappop(self._events)
                    if kind == "submit":
                        self._handle_submit_locked(value)
                    elif kind == "dependency_done":
                        # The event is only a wakeup. execute() resolves the
                        # complete wait pool before it schedules anything.
                        pass
                    elif kind == "engine_done":
                        self._handle_engine_done_locked(value)
                    elif kind == "shutdown":
                        self._state = "stopping"
                    self.scheduler.execute()
                    finalize_shutdown = self._finish_shutdown_if_possible_locked()
                    self._refresh_snapshot_locked()
                if finalize_shutdown:
                    # Never wait for the writer while holding the scheduler
                    # lock: collector subscribers are allowed to inspect the
                    # orchestrator snapshot during their callback.
                    self.data_collector.shutdown()
                    with self._event_cond:
                        if self._shutdown_ack is not None and not self._shutdown_ack.done():
                            self._shutdown_ack.set_result(None)
                        self._stop_loop = True
                        self._event_cond.notify_all()
        except BaseException as exc:
            print(f"[agorchestrator] FATAL scheduler failure: {format_exception(exc)}")
            with self._event_cond:
                for request in list(self._requests.values()):
                    if request.state != "running":
                        self._fail_request_locked(
                            request, f"scheduler failed: {format_exception(exc)}"
                        )
                if self._shutdown_ack is not None and not self._shutdown_ack.done():
                    self._shutdown_ack.set_exception(exc)

    def _handle_submit_locked(self, value: object) -> None:
        (
            ag,
            skill,
            skill_input,
            max_steps,
            result_future,
            parent_context,
            submitted_perf_ns,
            submitted_wall_ns,
            accepted,
        ) = value
        sequence = next(self._request_sequence)
        request_id = f"run{sequence}"
        request = _ExecutionRequest(
            request_id=request_id,
            sequence=sequence,
            agent=ag,
            skill=skill,
            skill_input=skill_input,
            max_steps=max_steps,
            result_future=result_future,
            parent_context=parent_context,
            ts_start=_ts(),
            submitted_perf_ns=submitted_perf_ns,
            submitted_wall_ns=submitted_wall_ns,
        )
        request.run_span = agprof.start_external_span(
            f"{request_id}:{skill.name}:{ag.agname}",
            start_perf_ns=submitted_perf_ns,
            start_wall_ns=submitted_wall_ns,
            metadata={
                "agency.run_id": request_id,
                "agency.agent_id": str(ag.agname),
                "agency.parent_agent_id": getattr(ag, "_parent_agent_id", None),
            },
            parent_context=parent_context,
        )
        request.scoped_collector = self.data_collector.scoped(
            agname=str(ag.agname), request_id=request_id, skill=skill.name
        )
        self._requests[request_id] = request
        self._future_producers[result_future] = request_id
        self._outstanding_by_agent.setdefault(ag, set()).add(request_id)
        self._submitted_total += 1
        self._publish_request_locked("request_submitted", request, {})
        accepted.set_result(None)

    def _launch_request_locked(self, request: _ExecutionRequest) -> None:
        request.state = "running"
        self._end_phase_span_locked(request)
        request.agent._state.blocked_on = None
        self._active_by_agent[request.agent] = request.request_id
        request.engine_started_wall = time.time()
        request.agent._set_ui_state("skill", skill=request.skill.name)
        self._publish_request_locked("request_started", request, {})
        try:
            # An engine belongs to exactly one dispatched request.  The agent
            # retains a compatibility/inspection reference to the latest one.
            request.engine = AgentEngine(request.agent)
            request.agent.engine = request.engine
            thread = agprof.spawn_traced(self._engine_worker, request)
            thread.name = f"agency-engine-{request.request_id}-{request.agent.agname}"
            thread.start()
        except BaseException as exc:
            self._post_locked(
                "engine_done",
                (
                    request.request_id,
                    _RunCompletion(
                        output=agerror(format_exception(exc)),
                        context=request.agent.ctx,
                        failed=True,
                        error_message=format_exception(exc),
                    ),
                ),
            )

    def _has_capacity_locked(self) -> bool:
        limit = self.max_concurrent_engines
        return limit is None or len(self._active_by_agent) < limit

    def _engine_worker(self, request: _ExecutionRequest) -> None:
        label = f"{request.request_id}:{request.skill.name}:{request.agent.agname}"
        agprof.thread_name(label)
        try:
            with agprof.span("engine:execute", parent_context=request.parent_context):
                agprof.annotate(
                    **{
                        "agency.run_id": request.request_id,
                        "agency.agent_id": str(request.agent.agname),
                        "agency.parent_agent_id": getattr(request.agent, "_parent_agent_id", None),
                    }
                )
                completion = self._perform_execution(request)
                agprof.annotate(
                    outcome="failure" if completion.failed else "success",
                    error_type="skill_error" if completion.failed else None,
                )
        except BaseException as exc:
            message = format_exception(exc)
            completion = _RunCompletion(
                output=agerror(message),
                context=request.agent.ctx,
                failed=True,
                error_message=message,
            )
        finally:
            self._post("engine_done", (request.request_id, completion))

    def _perform_execution(self, request: _ExecutionRequest) -> _RunCompletion:
        ag = request.agent
        skill = request.skill
        committed_ctx = ag.ctx.copy()
        working_ctx = committed_ctx.copy()
        local_skill_input = request.skill_input
        history_before = list(committed_ctx.recent_transcript)
        scoped = request.scoped_collector
        if scoped is None:
            raise RuntimeError("dispatched request has no scoped data collector")

        try:
            with agprof.span("resolve"):
                self.scheduler.materialize_dependencies(request.skill_input)
            local_skill_input = agdata(**dict(request.skill_input._data))
            scoped.record_event(
                type="skill_start",
                payload={"skill": skill.name, "ts": request.ts_start},
                term_message=(
                    f"[{ag.agname}] SKILL ▶  {skill.name}  "
                    f"input={list(local_skill_input._data.keys())}"
                ),
            )
            sandbox = ag._ensure_sandbox()
            engine = request.engine
            if engine is None:
                raise RuntimeError("dispatched request has no AgentEngine")
            engine._scoped_data_collector = scoped
            try:
                outer_result = engine.execute(
                    context=working_ctx,
                    skill=skill,
                    skill_input=local_skill_input,
                    resource_pool=type(ag).agresource_pool,
                    sandbox=sandbox,
                    max_steps=request.max_steps,
                )
            finally:
                engine._scoped_data_collector = None
            updated_ctx = working_ctx
        except Exception as exc:
            outer_result = agerror(format_exception(exc))
            updated_ctx = committed_ctx

        self._finish_execution_log(
            request,
            local_skill_input,
            outer_result,
            updated_ctx,
            history_before,
        )
        failed = bool(outer_result._data.get("error"))
        return _RunCompletion(
            output=outer_result,
            context=updated_ctx,
            failed=failed,
            error_message=str(outer_result._data.get("error", "")),
        )

    def _finish_execution_log(
        self,
        request: _ExecutionRequest,
        local_skill_input: agdata,
        result: agdata,
        updated_ctx: agcontext,
        history_before: list[dict],
    ) -> None:
        from ..agskill import _AgSkillFields

        ag = request.agent
        skill = request.skill
        scoped = request.scoped_collector
        if scoped is None:
            return
        try:
            ts_end = _ts()
            input_dict = local_skill_input.to_dict()
            result_dict = result.to_dict()
            if result_dict.get("error"):
                truncate = _AgSkillFields(ag.agconfig).error_log_truncate
                scoped.record_event(
                    type="skill_error",
                    payload={"skill": skill.name, "error": str(result_dict["error"])},
                    term_message=(
                        f"[{ag.agname}] SKILL ✗  {skill.name}  "
                        f"error={str(result_dict['error'])[:truncate]}"
                    ),
                )
            else:
                scoped.record_event(
                    type="skill_success",
                    payload={"skill": skill.name, "output_fields": list(result_dict)},
                    term_message=(
                        f"[{ag.agname}] SKILL ✓  {skill.name}  "
                        f"output={list(result_dict)}"
                    ),
                )
            transcript = list(updated_ctx.recent_transcript)
            scoped.record_event(
                type="skill_call",
                payload={
                    "skill": skill.name,
                    "ts_start": request.ts_start,
                    "ts_end": ts_end,
                    "input": input_dict,
                    "output": result_dict,
                    "history_len": len(transcript),
                    "history_before": history_before,
                    "history_delta": transcript,
                },
            )
            scoped.record_event(
                type="live_messages",
                payload={"messages": transcript},
                do_update=True,
            )
            try:
                from .. import agwebui as _agwebui

                if _agwebui._active is not None:
                    _agwebui._active.emitter.push_messages(ag.agname, transcript)
            except Exception as exc:
                print(f"[agorchestrator] WARNING: message UI update failed: {exc}")
        except Exception as exc:
            print(f"[agorchestrator] WARNING: execution logging failed: {exc}")

    def _handle_engine_done_locked(self, value: object) -> None:
        request_id, completion = value
        request = self._requests.get(request_id)
        if request is None:
            return
        completed_wall = time.time()
        if request.engine_started_wall is not None:
            self.data_collector.record_span(
                "engine:execution",
                request.engine_started_wall,
                completed_wall,
                {"outcome": "failure" if completion.failed else "success"},
                source="profiler_adapter",
                agname=str(request.agent.agname),
                request_id=request.request_id,
                skill=request.skill.name,
            )
        request.agent.ctx = completion.context
        self._active_by_agent.pop(request.agent, None)
        request.state = "failed" if completion.failed else "completed"
        self._completed_total += 1
        if completion.failed:
            self._failed_total += 1
        if not request.result_future.done():
            request.result_future.set_result(completion.output)
        self._end_run_span_locked(request, completion.failed, completion.error_message)
        self._finish_request_locked(request)
        self._publish_request_locked(
            "request_failed" if completion.failed else "request_completed",
            request,
            {"error": completion.error_message} if completion.failed else {},
        )
        self._update_agent_display_locked(request.agent, completion.failed)

    def _finish_request_locked(self, request: _ExecutionRequest) -> None:
        outstanding = self._outstanding_by_agent.get(request.agent)
        if outstanding is not None:
            outstanding.discard(request.request_id)
            if not outstanding:
                self._outstanding_by_agent.pop(request.agent, None)
        self._future_producers.pop(request.result_future, None)
        self._requests.pop(request.request_id, None)
        self._event_cond.notify_all()

    def _fail_request_locked(self, request: _ExecutionRequest, message: str) -> None:
        if request.state in ("completed", "failed", "running"):
            return
        request.state = "failed"
        self._completed_total += 1
        self._failed_total += 1
        output = agerror(message)
        if not request.result_future.done():
            request.result_future.set_result(output)
        self._end_phase_span_locked(request)
        self._end_run_span_locked(request, True, message)
        self._finish_request_locked(request)
        self._publish_request_locked("request_failed", request, {"error": message})
        self._update_agent_display_locked(request.agent, True)

    def _update_agent_display_locked(self, ag: "agent", last_failed: bool) -> None:
        outstanding = [
            self._requests[rid]
            for rid in self._outstanding_by_agent.get(ag, ())
            if rid in self._requests
        ]
        if ag in self._active_by_agent:
            return
        ready = min(
            (r for r in outstanding if r.state == "ready"), key=lambda r: r.sequence, default=None
        )
        if ready is not None:
            ag._state.blocked_on = None
            ag._set_ui_state("queued", skill=ready.skill.name)
            return
        blocked = min(
            (r for r in outstanding if r.state == "blocked"),
            key=lambda r: r.sequence,
            default=None,
        )
        if blocked is not None:
            self.scheduler.set_agent_blocked(blocked)
            return
        ag._state.blocked_on = None
        ag._set_ui_state("error" if last_failed else "finished")

    # ------------------------------------------------------------------
    # State, data collection, and shutdown
    # ------------------------------------------------------------------

    def _start_phase_span_locked(self, request: _ExecutionRequest, name: str) -> None:
        self._end_phase_span_locked(request)
        request.phase_name = name
        request.phase_started_wall = time.time()
        request.phase_span = agprof.start_external_span(
            name,
            start_perf_ns=time.perf_counter_ns(),
            start_wall_ns=time.time_ns(),
            metadata={
                "agency.run_id": request.request_id,
                "agency.agent_id": str(request.agent.agname),
            },
            parent_context=request.parent_context,
        )

    def _end_phase_span_locked(self, request: _ExecutionRequest) -> None:
        span = request.phase_span
        request.phase_span = None
        phase_name = request.phase_name
        phase_started_wall = request.phase_started_wall
        request.phase_name = None
        request.phase_started_wall = None
        ended_wall = time.time()
        if span is not None:
            span.end(end_perf_ns=time.perf_counter_ns(), end_wall_ns=time.time_ns())
        if phase_name is not None and phase_started_wall is not None:
            self.data_collector.record_span(
                phase_name,
                phase_started_wall,
                ended_wall,
                {},
                source="profiler_adapter",
                agname=str(request.agent.agname),
                request_id=request.request_id,
                skill=request.skill.name,
            )

    def _end_run_span_locked(
        self, request: _ExecutionRequest, failed: bool, error_message: str = ""
    ) -> None:
        span = request.run_span
        request.run_span = None
        ended_wall = time.time()
        if span is not None:
            metadata = {"outcome": "failure" if failed else "success"}
            if failed:
                metadata.update(error_type="skill_error", error_message=error_message)
            span.end(
                end_perf_ns=time.perf_counter_ns(),
                end_wall_ns=time.time_ns(),
                metadata=metadata,
            )
        attributes = {"outcome": "failure" if failed else "success"}
        if failed:
            attributes["error_message"] = error_message
        self.data_collector.record_span(
            "request:submission_to_completion",
            request.submitted_wall_ns / 1_000_000_000,
            ended_wall,
            attributes,
            source="profiler_adapter",
            agname=str(request.agent.agname),
            request_id=request.request_id,
            skill=request.skill.name,
        )

    def _publish_request_locked(
        self, event_type: str, request: _ExecutionRequest, payload: dict
    ) -> None:
        self.data_collector.record_event(
            event_type,
            {"state": request.state, "submission_sequence": request.sequence, **payload},
            source="orchestrator",
            agname=str(request.agent.agname),
            request_id=request.request_id,
            skill=request.skill.name,
            do_update=True,
        )

    def _snapshot_dict_locked(self) -> dict:
        requests = list(self._requests.values())
        agents: dict[str, dict] = {}
        for ag, request_ids in self._outstanding_by_agent.items():
            agents[str(ag.agname)] = {
                "active_request_id": self._active_by_agent.get(ag),
                "request_ids": sorted(
                    request_ids,
                    key=lambda rid: self._requests[rid].sequence if rid in self._requests else -1,
                ),
            }
        return {
            "state": self._state,
            "max_concurrent_engines": self.max_concurrent_engines,
            "ready_count": sum(request.state == "ready" for request in requests),
            "blocked_count": sum(request.state == "blocked" for request in requests),
            "running_count": len(self._active_by_agent),
            "submitted_total": self._submitted_total,
            "completed_total": self._completed_total,
            "failed_total": self._failed_total,
            "agents": agents,
        }

    def _refresh_snapshot_locked(self) -> None:
        snapshot = self._snapshot_dict_locked()
        self.data_collector.update_runtime(snapshot)
        self.data_collector.record_event(
            "scheduler_state",
            {key: value for key, value in snapshot.items() if key not in ("agents", "state")},
            source="orchestrator",
            do_update=True,
        )

    def _finish_shutdown_if_possible_locked(self) -> bool:
        if self._state != "stopping":
            return False
        if self._active_by_agent or any(r.state == "ready" for r in self._requests.values()):
            return False
        if any(kind == "dependency_done" for _seq, kind, _value in self._events):
            return False
        blocked = [r for r in self._requests.values() if r.state == "blocked"]
        for request in blocked:
            self._fail_request_locked(
                request,
                "orchestrator shut down before the request's dependencies resolved",
            )
        self._state = "stopped"
        self.data_collector.record_event(
            "scheduler_stopped", {}, source="orchestrator", do_update=True
        )
        self._refresh_snapshot_locked()
        return True


_global_orchestrator: "GlobalAgentOrchestrator | None" = None
_global_lock = threading.Lock()


def get_orchestrator(
    agconfig: "agConfig | None" = None,
    *,
    default_db_path: "str | Path | None" = None,
) -> GlobalAgentOrchestrator:
    global _global_orchestrator
    with _global_lock:
        if _global_orchestrator is None:
            if default_db_path is None:
                from ..agent import _DEFAULT_LOG_DIR

                default_db_path = _DEFAULT_LOG_DIR / "agency.sqlite3"
            _global_orchestrator = GlobalAgentOrchestrator(
                agconfig,
                default_db_path=default_db_path,
            )
        return _global_orchestrator


def peek_orchestrator() -> "GlobalAgentOrchestrator | None":
    return _global_orchestrator


def _reset_orchestrator_for_tests() -> None:
    global _global_orchestrator
    with _global_lock:
        orchestrator = _global_orchestrator
        _global_orchestrator = None
    if orchestrator is not None:
        try:
            orchestrator.shutdown(timeout_s=10)
        except Exception:  # noqa: S110 - test cleanup is best effort
            pass


def _shutdown_at_exit() -> None:
    orchestrator = peek_orchestrator()
    if orchestrator is not None:
        try:
            orchestrator.shutdown(wait=False)
        except Exception:  # noqa: S110 - interpreter teardown is best effort
            pass


atexit.register(_shutdown_at_exit)


__all__ = [
    "GlobalAgentOrchestrator",
    "OrchestratorSnapshot",
    "agOrchestratorConfig",
    "get_orchestrator",
]
