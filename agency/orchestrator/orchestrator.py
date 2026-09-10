from __future__ import annotations

import atexit
import contextvars
import heapq
import itertools
import sys
import threading
import time
from concurrent.futures import Future, InvalidStateError, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..observability.agdatalogger import agDataLogger, resolve_global_db_path
from ..configs.agconfig import agconfig as agconfig_cls, dataloggerconfig
from ..agcontext import agcontext
from ..agdata import agdata, agerror, agcanceled
from ..engine import AgentEngine
from ..observability.agdatalogger import _ts
from ..utils.agutil import _DEFAULT_LOG_DIR, format_exception
from ..observability.profiler import agprof
from .agresources import agResourcePool
from .scheduler import ExecutionScheduler

if TYPE_CHECKING:
    from ..agent import agent
    from ..agskill import agskill


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
    queue_depth: int = 0


@dataclass
class _ExecutionRequest:
    request_id: str
    sequence: int
    kind: str
    agent: "agent"
    skill: "agskill | None"
    skill_input: "agdata | None"
    max_steps: "int | None"
    result_future: "Future[agdata]"
    context_future: "Future[agcontext]"
    context_dependency: agcontext
    parent_context: object
    ts_start: str
    submitted_perf_ns: int
    submitted_wall_ns: int
    state: str = "submitted"
    cancelled: bool = False
    message: "str | None" = None
    dependencies: set[Future] = field(default_factory=set)
    producer_ids: set[str] = field(default_factory=set)
    engine: "AgentEngine | None" = None
    run_span: object = None
    phase_span: object = None
    terminal_output: "agdata | None" = None
    terminal_error: str = ""
    terminal_state: str = "failed"
    terminal_event: str = "request_failed"
    terminal_counts_as_failure: bool = True


@dataclass
class _RunCompletion:
    output: agdata
    context: agcontext
    outcome: str
    error_message: str = ""

    @property
    def failed(self) -> bool:
        return self.outcome == "failed"


class GlobalAgentOrchestrator:
    """Event-driven process-wide scheduler for agent engine executions."""

    def __init__(
        self,
        agconfig: "agconfig_cls | None" = None,
        *,
        default_db_path: "str | Path | None" = None,
    ) -> None:
        self.agconfig = agconfig if agconfig is not None else agconfig_cls()
        self._validate_max_concurrent_engines()
        if default_db_path is None:
            default_db_path = resolve_global_db_path(_DEFAULT_LOG_DIR)
        db_path = self.agconfig.orchestrator.db_path or default_db_path
        assert db_path is not None
        self.data_logger = agDataLogger(self._scoped_data_logger_config(str(db_path)))
        self.data_logger.start()
        self.agresource_pool = agResourcePool(
            mark_gpus=False,
            data_logger=self.data_logger,
        )
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
        self._scheduler_failure: "BaseException | None" = None
        self._shutdown_ack: "Future[None] | None" = None
        self._stop_loop = False
        # The scheduler remains the sole authority for admission/capacity.  A
        # very large ceiling preserves the public ``None`` (unlimited) setting
        # without imposing ThreadPoolExecutor's much smaller implicit default;
        # workers are still created lazily and reused after a request finishes.
        execution_worker_limit = self.agconfig.orchestrator.max_concurrent_engines or sys.maxsize
        self._execution_workers = ThreadPoolExecutor(
            max_workers=execution_worker_limit,
            thread_name_prefix="agency-execution",
        )
        self.scheduler = ExecutionScheduler(self)
        self._scheduler_thread = agprof.spawn_traced(self._scheduler_main, daemon=True)
        self._scheduler_thread.name = "agency-global-scheduler"
        self._scheduler_thread.start()
        self._record_global_event(
            "scheduler_started",
            {"max_concurrent_engines": self.agconfig.orchestrator.max_concurrent_engines},
            update_latest_snapshot=True,
        )
        with self._event_cond:
            self._record_scheduler_snapshot()

    def _validate_max_concurrent_engines(self) -> None:
        limit = self.agconfig.orchestrator.max_concurrent_engines
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
        ):
            raise ValueError("max_concurrent_engines must be a positive integer or None")

    def _scoped_data_logger_config(self, db_path: "str | None" = None) -> "agconfig_cls":
        """Build the data logger's own narrow agconfig from this
        orchestrator's current fields. *db_path* is only needed on first
        construction -- agDataLogger.change_config() carries the existing
        db_path forward on its own when this orchestrator's own db_path is
        still unset."""
        return agconfig_cls(
            dataloggerconfig(
                db_path=db_path if db_path is not None else self.agconfig.orchestrator.db_path,
                flush_batch_size=self.agconfig.orchestrator.flush_batch_size,
                flush_interval_s=self.agconfig.orchestrator.flush_interval_s,
            )
        )

    def change_config(self, agconfig: "agconfig_cls") -> None:
        """Replace this orchestrator's agconfig and cascade to every child
        that holds its own agconfig composition: the global data logger
        (its own narrow, rescoped config) and the default resource pool."""
        self.agconfig = agconfig
        self._validate_max_concurrent_engines()
        self.data_logger.change_config(self._scoped_data_logger_config())
        self.agresource_pool.change_config(self.agconfig)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        ag: "agent",
        skill: "agskill",
        skill_input: object,
        max_steps: "int | None" = None,
    ) -> agdata:
        """Atomically publish one skill execution and return its bare result agdata.

        The orchestrator condition is acquired before the per-agent submission
        lock everywhere these two locks are needed.  Publication into the
        authoritative context chain and scheduler registration therefore have
        one linearization point and cannot be observed in different orders.
        """
        self._validate_max_steps(max_steps)
        if not isinstance(skill_input, agdata):
            as_pending = getattr(skill_input, "_as_pending_agdata", None)
            if not callable(as_pending):
                raise TypeError("skill_input must be an agdata or pending result handle")
            skill_input = as_pending()

        result_future: "Future[agdata]" = Future()
        context_future: "Future[agcontext]" = Future()
        cycle_ack: Future[None] | None = None
        with self._event_cond:
            if not self._accepting:
                raise RuntimeError("global agent orchestrator is shut down")
            with ag._submission_lock:
                predecessor = ag.context
                new_context = agcontext(_future=context_future)
                ag.context = new_context
                try:
                    request = self._register_submission_locked(
                        ag,
                        kind="skill",
                        skill=skill,
                        skill_input=skill_input,
                        max_steps=max_steps,
                        result_future=result_future,
                        context_future=context_future,
                        context_dependency=predecessor,
                    )
                except BaseException:
                    if ag.context is new_context:
                        ag.context = predecessor
                    raise
            if threading.current_thread() is not self._scheduler_thread:
                cycle_ack = Future()
            self._post_locked("schedule", (request.request_id, cycle_ack))
        if cycle_ack is not None:
            cycle_ack.result()
        result = agdata(_future=result_future)
        # Keep execution identity outside user data, and retain it after wait()
        # clears _future. No scheduler request needs to survive completion.
        object.__setattr__(result, "_execution_id", request.request_id)
        object.__setattr__(result, "_execution_agent", ag)
        return result

    @staticmethod
    def _validate_max_steps(max_steps: "int | None") -> None:
        if max_steps is not None and (
            not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0
        ):
            raise ValueError("max_steps must be a positive integer or None")

    def submit_context_message(self, ag: "agent", message: str) -> None:
        """Atomically publish one orchestrator-owned host-only context request.

        A plain enqueue -- nothing is returned. Ordering into the context
        chain is already guaranteed synchronously here, before this method
        returns, so there is nothing for a caller to hold a handle to.
        """
        result_future: "Future[agdata]" = Future()
        context_future: "Future[agcontext]" = Future()
        cycle_ack: Future[None] | None = None
        with self._event_cond:
            if not self._accepting:
                raise RuntimeError("global agent orchestrator is shut down")
            with ag._submission_lock:
                predecessor = ag.context
                new_context = agcontext(_future=context_future)
                ag.context = new_context
                try:
                    request = self._register_submission_locked(
                        ag,
                        kind="context_message",
                        skill=None,
                        skill_input=None,
                        max_steps=None,
                        result_future=result_future,
                        context_future=context_future,
                        context_dependency=predecessor,
                        message=message,
                    )
                except BaseException:
                    if ag.context is new_context:
                        ag.context = predecessor
                    raise
            if threading.current_thread() is not self._scheduler_thread:
                cycle_ack = Future()
            self._post_locked("schedule", (request.request_id, cycle_ack))
        if cycle_ack is not None:
            cycle_ack.result()

    def redirect_request(self, ag: "agent", request_id: str, message: str) -> bool:
        with self._event_cond:
            request = self._requests.get(request_id)
            if request is None or request.agent is not ag or request.state != "running":
                return False
            engine = request.engine
        # Never hold the scheduler lock during PTY/network I/O. This engine
        # belongs only to the requested execution, even if it finishes now.
        return engine.redirect(message) if engine is not None else False

    def cancel_request(self, future: "Future") -> bool:
        """Mark whichever request produced *future* as cancelled. Returns
        True iff that request was in "running" state at this instant --
        agent.cancel() uses this as a best-effort gate for whether to also
        try killing a live harness process; it is not a precise "a harness
        is definitely running right now" signal (the worker dispatched for
        a "running" request may not have reached the harness launch yet),
        just a cheap way to skip that attempt when it's certain there's
        nothing to reach (request still blocked/ready, never dispatched).

        No scheduler scan and no wake-event: an already-running request is
        only ever observed by the engine's own checkpoints, and a still-
        blocked request will naturally reach one once its own predecessor
        resolves via the normal flow -- nothing here needs to be faster than
        that.
        """
        with self._event_cond:
            request_id = self._future_producers.get(future)
            if request_id is None:
                return False
            request = self._requests.get(request_id)
            if request is None:
                return False
            request.cancelled = True
            return request.state == "running"

    def snapshot(self) -> OrchestratorSnapshot:
        with self._event_cond:
            data = self._snapshot_dict_locked()
        return OrchestratorSnapshot(
            **data,
            db_path=str(self.data_logger.db_path),
        )

    def flush(self, timeout_s: "float | None" = None) -> None:
        self.data_logger.flush()

    def shutdown(self, wait: bool = True, timeout_s: "float | None" = None) -> None:
        with self._event_cond:
            if self._state != "stopped" and self._shutdown_ack is None:
                self._shutdown_ack = Future()
                self._accepting = False
                self._post_locked("shutdown", self._shutdown_ack)
            ack = self._shutdown_ack
        # Result callbacks execute synchronously on the publishing thread.  A
        # callback may request shutdown, but the scheduler cannot wait for or
        # join its own event loop from inside that callback.
        if threading.current_thread() is self._scheduler_thread:
            wait = False
        if not wait:
            return
        if ack is not None:
            ack.result(timeout=timeout_s)
        self._scheduler_thread.join(timeout=timeout_s)
        # The scheduler reaches ``stopped`` only after every accepted engine
        # request has completed.  This join therefore only reaps idle reusable
        # workers (and any rejected work item left by a failed thread start).
        self._execution_workers.shutdown(wait=True, cancel_futures=True)

    # ------------------------------------------------------------------
    # Scheduler loop and transitions
    # ------------------------------------------------------------------

    def _post(self, kind: str, value: object) -> None:
        with self._event_cond:
            self._post_locked(kind, value)

    def _post_locked(self, kind: str, value: object) -> None:
        heapq.heappush(self._events, (next(self._event_sequence), kind, value))
        self._event_cond.notify_all()

    @staticmethod
    def _settle_future(future: Future, value: object, label: str) -> None:
        """Publish a value without allowing a hostile callback to strand state."""
        if future.done():
            return
        try:
            future.set_result(value)
        except InvalidStateError:
            pass
        except BaseException as exc:
            # concurrent.futures already catches ordinary callback Exceptions;
            # this protects scheduler invariants from BaseException subclasses.
            print(f"[agorchestrator] WARNING: {label} callback failed: {format_exception(exc)}")

    @staticmethod
    def _reject_future(future: Future, exc: BaseException, label: str) -> None:
        """Reject an acknowledgement while preserving the drain state machine."""
        if future.done():
            return
        try:
            future.set_exception(exc)
        except InvalidStateError:
            pass
        except BaseException as callback_exc:
            print(
                f"[agorchestrator] WARNING: {label} callback failed: "
                f"{format_exception(callback_exc)}"
            )

    def _scheduler_main(self) -> None:
        try:
            while True:
                finalize_shutdown = False
                with self._event_cond:
                    while not self._events and not self._stop_loop:
                        self._event_cond.wait()
                    if self._stop_loop:
                        break
                    _seq, kind, value = heapq.heappop(self._events)
                    cycle_ack: "Future[None] | None" = None
                    try:
                        if kind == "dependency_done":
                            # The event is only a wakeup. execute() resolves the
                            # complete wait pool before it schedules anything.
                            pass
                        elif kind in {"schedule", "control_changed"}:
                            # Registration/control methods already changed the
                            # authoritative state while holding this condition.
                            if kind == "schedule" and isinstance(value, tuple):
                                _request_id, cycle_ack = value
                        elif kind == "engine_done":
                            self._handle_engine_done_locked(value)
                        elif kind == "pass_through_done":
                            self._handle_pass_through_done_locked(value)
                        elif kind == "shutdown":
                            if self._scheduler_failure is None:
                                self._state = "stopping"
                        # A fatal transition closes scheduling permanently, but
                        # the event loop remains alive to drain engine and context
                        # completion events that were already in flight.
                        if self._scheduler_failure is None:
                            self.scheduler.execute()
                    except BaseException as exc:
                        self._enter_scheduler_failure_locked(exc, cycle_ack)
                    else:
                        if cycle_ack is not None:
                            self._settle_future(cycle_ack, None, "schedule acknowledgement")
                    finalize_shutdown = self._finish_shutdown_if_possible_locked()
                    self._record_scheduler_snapshot()
                if finalize_shutdown:
                    try:
                        self.data_logger.stop()
                    except Exception as exc:
                        print(f"[agorchestrator] WARNING: global logger shutdown failed: {exc}")
                    with self._event_cond:
                        if self._shutdown_ack is not None:
                            self._settle_future(
                                self._shutdown_ack,
                                None,
                                "shutdown acknowledgement",
                            )
                        self._stop_loop = True
                        self._event_cond.notify_all()
        finally:
            # A callback running on the scheduler thread cannot synchronously
            # join that same thread.  Closing here guarantees its non-blocking
            # shutdown request still retires the reusable workers once drained.
            self._execution_workers.shutdown(wait=False, cancel_futures=True)

    def _enter_scheduler_failure_locked(
        self,
        exc: BaseException,
        cycle_ack: "Future[None] | None",
    ) -> None:
        """Close admission and enter a completion-only drain after a fatal event."""
        if cycle_ack is not None:
            self._reject_future(cycle_ack, exc, "schedule rejection")
        if self._scheduler_failure is not None:
            return

        self._scheduler_failure = exc
        self._accepting = False
        self._state = "failed"
        message = f"scheduler failed: {format_exception(exc)}"
        print(f"[agorchestrator] FATAL scheduler failure: {format_exception(exc)}")

        # Every submitter whose request was already registered must be released
        # from its cycle acknowledgement.  Retain only events required to drain
        # active engines and asynchronous context pass-through continuations.
        drain_events: list[tuple[int, str, object]] = []
        while self._events:
            event = heapq.heappop(self._events)
            _sequence, kind, value = event
            if kind == "schedule" and isinstance(value, tuple):
                _request_id, ack = value
                if isinstance(ack, Future):
                    self._reject_future(ack, exc, "queued schedule rejection")
            if kind in {"engine_done", "pass_through_done", "shutdown"}:
                drain_events.append(event)
        self._events = drain_events
        heapq.heapify(self._events)

        for request in list(self._requests.values()):
            if request.state != "running":
                self._fail_request_locked(request, message)

    def _register_submission_locked(
        self,
        ag: "agent",
        *,
        kind: str,
        skill: "agskill | None",
        skill_input: "agdata | None",
        max_steps: "int | None",
        result_future: "Future[agdata]",
        context_future: "Future[agcontext]",
        context_dependency: agcontext,
        message: "str | None" = None,
    ) -> _ExecutionRequest:
        parent_context = agprof.current_span_context()
        submitted_perf_ns = time.perf_counter_ns()
        submitted_wall_ns = time.time_ns()
        sequence = next(self._request_sequence)
        request_id = f"run{sequence}"
        request = _ExecutionRequest(
            request_id=request_id,
            sequence=sequence,
            kind=kind,
            agent=ag,
            skill=skill,
            skill_input=skill_input,
            max_steps=max_steps,
            result_future=result_future,
            context_future=context_future,
            parent_context=parent_context,
            ts_start=_ts(),
            submitted_perf_ns=submitted_perf_ns,
            submitted_wall_ns=submitted_wall_ns,
            context_dependency=context_dependency,
            state="submitted",
            message=message,
        )
        skill_name = skill.name if skill is not None else kind
        request.run_span = agprof.start_external_span(
            f"{request_id}:{skill_name}:{ag.agname}",
            start_perf_ns=submitted_perf_ns,
            start_wall_ns=submitted_wall_ns,
            metadata={
                "request_kind": kind,
                "harness": ag.harness,
                "request_id": request_id,
                "skill": skill_name,
                "agency.run_id": request_id,
                "agency.agent_id": str(ag.agname),
                "agency.parent_agent_id": getattr(ag, "_parent_agent_id", None),
            },
            parent_context=parent_context,
            data_span_name="request:submission_to_completion",
        )
        if request.run_span is not None:
            request.parent_context = request.run_span.context()
        self._requests[request_id] = request
        self._future_producers[result_future] = request_id
        self._future_producers[context_future] = request_id
        self._outstanding_by_agent.setdefault(ag, set()).add(request_id)
        self._submitted_total += 1
        self._record_request_event("request_submitted", request, {})
        return request

    def _launch_request_locked(self, request: _ExecutionRequest) -> None:
        if request.skill is None:
            raise RuntimeError("skill request has no skill")
        try:
            # Always launch: a cancelled invocation costs only an AgentEngine
            # construction here -- its own pre-checkpoint returns a controlled
            # error before any harness is ever spawned. Every exception from
            # this point must produce an engine_done event, or the request
            # could look active forever with no worker started.
            request.state = "running"
            self._end_phase_span_locked(request)
            self._active_by_agent[request.agent] = request.request_id
            request.agent.record_state("running_skill", skill=request.skill.name)
            self._record_request_event("request_started", request, {})
            # An engine belongs to exactly one dispatched request.  The agent
            # retains a compatibility/inspection reference to the latest one.
            request.engine = AgentEngine(request.agent)
            request.agent.engine = request.engine
            self._submit_engine_worker(request)
        except BaseException as exc:
            self._post_locked(
                "engine_done",
                (
                    request.request_id,
                    _RunCompletion(
                        output=agerror(format_exception(exc)),
                        context=request.context_dependency.copy(),
                        outcome="failed",
                        error_message=format_exception(exc),
                    ),
                ),
            )

    def _submit_engine_worker(self, request: _ExecutionRequest) -> None:
        """Dispatch one request without allowing a rejected item to run later.

        ``ThreadPoolExecutor.submit`` puts its work item on the queue before it
        tries to start a new worker.  If ``Thread.start`` then fails, submit
        raises but that queued item remains.  Hold the item behind this gate so
        a later healthy worker drains it as a no-op instead of executing a
        request which the scheduler has already failed and released.
        """
        admission = threading.Event()
        accepted = False

        def run_if_accepted() -> None:
            admission.wait()
            if accepted:
                # Reused threads must not carry ContextVar/OTel state from one
                # invocation into the next.  The durable profiling parent is
                # supplied explicitly by _engine_worker from the request.
                contextvars.Context().run(self._engine_worker, request)

        try:
            self._execution_workers.submit(run_if_accepted)
        except BaseException:
            admission.set()
            raise
        accepted = True
        admission.set()

    def _has_capacity_locked(self) -> bool:
        limit = self.agconfig.orchestrator.max_concurrent_engines
        return limit is None or len(self._active_by_agent) < limit

    def _engine_worker(self, request: _ExecutionRequest) -> None:
        if request.skill is None:
            raise RuntimeError("engine worker received a request without a skill")
        label = f"{request.request_id}:{request.skill.name}:{request.agent.agname}"
        agprof.thread_name(label)
        try:
            with agprof.span("engine:execute", parent_context=request.parent_context):
                agprof.annotate(
                    **{
                        "request_kind": request.kind,
                        "request_id": request.request_id,
                        "skill": request.skill.name,
                        "agency.run_id": request.request_id,
                        "agency.agent_id": str(request.agent.agname),
                        "agency.parent_agent_id": getattr(request.agent, "_parent_agent_id", None),
                    }
                )
                completion = self._perform_execution(request)
                agprof.annotate(
                    outcome="success" if completion.outcome == "succeeded" else "failure",
                    lifecycle_outcome=completion.outcome,
                    error_type="skill_error" if completion.failed else None,
                )
        except BaseException as exc:
            message = format_exception(exc)
            completion = _RunCompletion(
                output=agerror(message),
                context=request.context_dependency.copy(),
                outcome="failed",
                error_message=message,
            )
        finally:
            self._post("engine_done", (request.request_id, completion))

    def _perform_execution(self, request: _ExecutionRequest) -> _RunCompletion:
        ag = request.agent
        skill = request.skill
        if skill is None or request.skill_input is None:
            raise RuntimeError("skill execution request is incomplete")
        committed_context = request.context_dependency.copy()
        working_context = committed_context.copy()
        local_skill_input = request.skill_input
        history_before = list(committed_context.recent_transcript)
        logger = ag.data_logger
        execution_started = False

        if request.cancelled:
            return self._controlled_completion(request, committed_context)

        try:
            with agprof.span("resolve"):
                self.scheduler.materialize_dependencies(request.skill_input)
            local_skill_input = agdata(**dict(request.skill_input._data))
            logger.record_event(
                type="skill_start",
                payload={"skill": skill.name, "ts": request.ts_start},
                term_message=(
                    f"[{ag.agname}] SKILL ▶  {skill.name}  "
                    f"input={list(local_skill_input._data.keys())}"
                ),
                flush=True,
            )
            sandbox = ag._ensure_sandbox()
            engine = request.engine
            if engine is None:
                raise RuntimeError("dispatched request has no AgentEngine")
            execution_started = True
            outer_result = engine.execute(
                context=working_context,
                skill=skill,
                skill_input=local_skill_input,
                resource_pool=self.agresource_pool,
                sandbox=sandbox,
                max_steps=request.max_steps,
                is_cancelled=lambda: request.cancelled,
                request_id=request.request_id,
            )
            # A skill-level failure is a rolled-back transaction even when the
            # engine reports it as an agerror rather than raising.  Never
            # publish mutations made to the working context on that path.
            updated_context = (
                committed_context if isinstance(outer_result, agerror) else working_context
            )
        except Exception as exc:
            outer_result = agerror(format_exception(exc))
            updated_context = committed_context

        if request.cancelled:
            return self._controlled_completion(request, committed_context)

        if execution_started and isinstance(outer_result, agerror):
            # Preserve exactly the canonical rollback notice on a clean copy;
            # every other mutation made by the failed working transaction is
            # intentionally discarded.  Controlled cancellation returned above
            # and therefore never receives this notice.
            updated_context = committed_context.copy()
            ag._record_context_notice(updated_context)

        self._record_execution_results(
            request,
            local_skill_input,
            outer_result,
            updated_context,
            history_before,
        )
        failed = bool(outer_result._data.get("error"))
        return _RunCompletion(
            output=outer_result,
            context=updated_context,
            outcome="failed" if failed else "succeeded",
            error_message=str(outer_result._data.get("error", "")),
        )

    def _controlled_completion(
        self,
        request: _ExecutionRequest,
        committed_context: agcontext,
    ) -> _RunCompletion:
        message = "agent invocation cancelled"
        try:
            request.agent.data_logger.record_event(
                type="skill_cancelled",
                payload={"skill": request.skill.name if request.skill is not None else None},
                term_message=f"[{request.agent.agname}] SKILL ■  cancelled",
                flush=True,
            )
        except Exception as exc:
            print(f"[agorchestrator] WARNING: controlled completion logging failed: {exc}")
        return _RunCompletion(
            output=agcanceled(message),
            context=committed_context,
            outcome="cancelled",
            error_message=message,
        )

    def _record_execution_results(
        self,
        request: _ExecutionRequest,
        local_skill_input: agdata,
        result: agdata,
        updated_context: agcontext,
        history_before: list[dict],
    ) -> None:
        def logging_safe(value):
            if isinstance(value, bytes):
                return {"type": "bytes", "size_bytes": len(value)}
            if isinstance(value, dict):
                return {key: logging_safe(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [logging_safe(item) for item in value]
            return value

        ag = request.agent
        skill = request.skill
        logger = ag.data_logger
        try:
            ts_end = _ts()
            input_dict = logging_safe(local_skill_input.to_dict())
            result_dict = logging_safe(result.to_dict())
            if result_dict.get("error"):
                logger.record_event(
                    type="skill_error",
                    payload={"skill": skill.name, "error": str(result_dict["error"])},
                    term_message=(
                        f"[{ag.agname}] SKILL ✗  {skill.name}  error={result_dict['error']}"
                    ),
                )
            else:
                logger.record_event(
                    type="skill_success",
                    payload={"skill": skill.name, "output_fields": list(result_dict)},
                    term_message=(
                        f"[{ag.agname}] SKILL ✓  {skill.name}  output={list(result_dict)}"
                    ),
                )
            transcript = list(updated_context.recent_transcript)
            logger.record_event(
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
            logger.record_event(
                type="live_messages",
                payload={"messages": transcript},
                update_latest_snapshot=True,
                flush=True,
            )
        except Exception as exc:
            print(f"[agorchestrator] WARNING: execution logging failed: {exc}")

    def _handle_engine_done_locked(self, value: object) -> None:
        request_id, completion = value
        request = self._requests.get(request_id)
        if request is None:
            return
        self._settle_future(request.context_future, completion.context, "output-context")
        request.state = "completed" if completion.outcome == "succeeded" else completion.outcome
        self._completed_total += 1
        if completion.failed:
            self._failed_total += 1
        self._end_run_span_locked(
            request,
            completion.failed,
            completion.error_message,
            outcome=completion.outcome,
        )
        event_type = {
            "succeeded": "request_completed",
            "failed": "request_failed",
            "cancelled": "request_cancelled",
        }[completion.outcome]
        self._record_request_event(
            event_type,
            request,
            {"error": completion.error_message} if completion.outcome != "succeeded" else {},
        )
        self._finish_request_locked(request)
        self._active_by_agent.pop(request.agent, None)
        self._update_agent_display_locked(request.agent)
        self._settle_future(request.result_future, completion.output, "invocation result")

    def _complete_message_locked(self, request: _ExecutionRequest) -> None:
        """Commit one ready host-only message on the scheduler event loop.

        The event condition is the outer publication lock.  Taking the agent
        submission lock beneath it keeps the sequence-number bump atomic with
        any other context mutation, while the operation stays entirely free
        of engine, sandbox, harness, and capacity ownership.
        """
        if request.kind != "context_message":
            raise RuntimeError("context-only completion requires a context_message request")
        if request.state not in {"submitted", "blocked"}:
            return

        try:
            output_context = request.context_dependency.copy()
            with request.agent._submission_lock:
                last_retained_sequence = max(
                    (
                        int(entry.get("sequence", 0))
                        for entry in output_context.retained_messages
                        if isinstance(entry, dict)
                    ),
                    default=0,
                )
                request.agent._ensure_sequence_at_least(last_retained_sequence)
                sequence = request.agent._next_sequence()
            output_context.append_retained_message(
                {
                    "sequence": sequence,
                    "type": "message",
                    "role": "user",
                    "content": request.message,
                    "source": "queue_message",
                }
            )
        except BaseException as exc:
            self._fail_request_locked(request, format_exception(exc))
            return

        result = agdata()
        request.state = "completed"
        self._end_phase_span_locked(request)
        self._settle_future(request.context_future, output_context, "message output-context")
        self._completed_total += 1
        self._end_run_span_locked(request, False, outcome="succeeded")
        self._record_request_event("request_completed", request, {})
        self._finish_request_locked(request)
        self._update_agent_display_locked(request.agent)
        self._settle_future(request.result_future, result, "message result")

    def _finish_request_locked(self, request: _ExecutionRequest) -> None:
        outstanding = self._outstanding_by_agent.get(request.agent)
        if outstanding is not None:
            outstanding.discard(request.request_id)
            if not outstanding:
                self._outstanding_by_agent.pop(request.agent, None)
        self._future_producers.pop(request.result_future, None)
        self._future_producers.pop(request.context_future, None)
        self._requests.pop(request.request_id, None)
        self._event_cond.notify_all()

    def _pass_through_context_future(self, request: _ExecutionRequest) -> bool:
        """Set pass-through context now, or arrange an event-driven continuation."""
        if request.context_future.done():
            return True
        predecessor = request.context_dependency
        predecessor_future = predecessor._future
        if predecessor_future is None or predecessor_future.done():
            try:
                context = predecessor.copy()
            except BaseException:
                context = agcontext()
            self._settle_future(request.context_future, context, "pass-through context")
            return True

        def _propagate(finished: "Future[agcontext]", request_id: str = request.request_id) -> None:
            self._post("pass_through_done", (request_id, finished))

        predecessor_future.add_done_callback(_propagate)
        return False

    def _handle_pass_through_done_locked(self, value: object) -> None:
        request_id, finished = value
        request = self._requests.get(request_id)
        if request is None or request.state != "settling_terminal":
            return
        if not request.context_future.done():
            try:
                context = finished.result().copy()
            except BaseException:
                context = agcontext()
            self._settle_future(request.context_future, context, "pass-through context")
        self._finish_terminal_request_locked(request)

    def _fail_request_locked(self, request: _ExecutionRequest, message: str) -> None:
        self._begin_terminal_request_locked(
            request,
            output=agerror(message),
            message=message,
            terminal_state="failed",
            event_type="request_failed",
            counts_as_failure=True,
        )

    def _begin_terminal_request_locked(
        self,
        request: _ExecutionRequest,
        *,
        output: agdata,
        message: str,
        terminal_state: str,
        event_type: str,
        counts_as_failure: bool,
    ) -> None:
        if request.state in {
            "completed",
            "failed",
            "cancelled",
            "settling_terminal",
            "running",
        }:
            return
        request.state = "settling_terminal"
        self._completed_total += 1
        if counts_as_failure:
            self._failed_total += 1
        request.terminal_output = output
        request.terminal_error = message
        request.terminal_state = terminal_state
        request.terminal_event = event_type
        request.terminal_counts_as_failure = counts_as_failure
        self._end_phase_span_locked(request)
        self._end_run_span_locked(
            request,
            counts_as_failure,
            message,
            outcome=terminal_state,
        )
        self._record_request_event(event_type, request, {"error": message})
        if self._pass_through_context_future(request):
            self._finish_terminal_request_locked(request)

    def _finish_terminal_request_locked(self, request: _ExecutionRequest) -> None:
        output = request.terminal_output or agerror(request.terminal_error or "request failed")
        request.state = request.terminal_state
        self._finish_request_locked(request)
        self._update_agent_display_locked(request.agent)
        self._settle_future(request.result_future, output, "terminal submission result")

    def _update_agent_display_locked(self, ag: "agent") -> None:
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
            ag.record_state("queued", skill=self._request_label(ready))
            return
        blocked = min(
            (r for r in outstanding if r.state == "blocked"),
            key=lambda r: r.sequence,
            default=None,
        )
        if blocked is not None:
            self.scheduler.set_agent_blocked(blocked)
            return
        ag.record_state("agent_idle")

    # ------------------------------------------------------------------
    # State, data collection, and shutdown
    # ------------------------------------------------------------------

    def _start_phase_span_locked(self, request: _ExecutionRequest, name: str) -> None:
        self._end_phase_span_locked(request)
        request.phase_span = agprof.start_external_span(
            name,
            start_perf_ns=time.perf_counter_ns(),
            start_wall_ns=time.time_ns(),
            metadata={
                "request_kind": request.kind,
                "request_id": request.request_id,
                "skill": self._request_label(request),
                "agency.run_id": request.request_id,
                "agency.agent_id": str(request.agent.agname),
            },
            parent_context=request.parent_context,
        )

    def _end_phase_span_locked(self, request: _ExecutionRequest) -> None:
        span = request.phase_span
        request.phase_span = None
        if span is not None:
            span.end(end_perf_ns=time.perf_counter_ns(), end_wall_ns=time.time_ns())

    def _end_run_span_locked(
        self,
        request: _ExecutionRequest,
        failed: bool,
        error_message: str = "",
        *,
        outcome: "str | None" = None,
    ) -> None:
        span = request.run_span
        request.run_span = None
        if span is None:
            return
        outcome = outcome or ("failed" if failed else "succeeded")
        profiler_outcome = "success" if outcome == "succeeded" else "failure"
        metadata = {"outcome": profiler_outcome, "lifecycle_outcome": outcome}
        if failed:
            metadata.update(error_type="skill_error", error_message=error_message)
        span.end(
            end_perf_ns=time.perf_counter_ns(),
            end_wall_ns=time.time_ns(),
            metadata=metadata,
        )

    def _record_request_event(
        self, event_type: str, request: _ExecutionRequest, payload: dict
    ) -> None:
        try:
            self.data_logger.record_event(
                event_type,
                {
                    "request_kind": request.kind,
                    "state": request.state,
                    "submission_sequence": request.sequence,
                    "request_id": request.request_id,
                    "skill": self._request_label(request),
                    **payload,
                },
                name=str(request.agent.agname),
                object="agent",
                update_latest_snapshot=True,
            )
        except Exception as exc:
            print(f"[agorchestrator] WARNING: event recording failed: {exc}")

    @staticmethod
    def _request_label(request: _ExecutionRequest) -> str:
        return request.skill.name if request.skill is not None else request.kind

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
            "max_concurrent_engines": self.agconfig.orchestrator.max_concurrent_engines,
            "ready_count": sum(request.state == "ready" for request in requests),
            "blocked_count": sum(request.state == "blocked" for request in requests),
            "running_count": len(self._active_by_agent),
            "submitted_total": self._submitted_total,
            "completed_total": self._completed_total,
            "failed_total": self._failed_total,
            "agents": agents,
        }

    def _record_global_event(
        self,
        event_type: str,
        payload: dict,
        *,
        update_latest_snapshot: bool = False,
    ) -> None:
        try:
            self.data_logger.record_event(
                event_type,
                payload,
                name="scheduler",
                object="orchestrator",
                update_latest_snapshot=update_latest_snapshot,
            )
        except Exception as exc:
            print(f"[agorchestrator] WARNING: global event recording failed: {exc}")

    def _record_scheduler_snapshot(self) -> None:
        snapshot = self._snapshot_dict_locked()
        try:
            self.data_logger.record_event(
                "scheduler_state",
                {key: value for key, value in snapshot.items() if key not in ("agents", "state")},
                name="scheduler",
                object="orchestrator",
                update_latest_snapshot=True,
            )
        except Exception as exc:
            print(f"[agorchestrator] WARNING: scheduler telemetry failed: {exc}")

    def _finish_shutdown_if_possible_locked(self) -> bool:
        if self._state not in {"stopping", "failed"}:
            return False
        if self._active_by_agent or any(r.state == "ready" for r in self._requests.values()):
            return False
        if any(kind == "dependency_done" for _seq, kind, _value in self._events):
            return False
        pending = [
            request
            for request in self._requests.values()
            if request.state in {"submitted", "blocked"}
        ]
        for request in pending:
            self._fail_request_locked(
                request,
                "orchestrator shut down before the request's dependencies resolved",
            )
        self._force_unmanaged_terminal_contexts_locked()
        if self._requests:
            return False
        self._state = "stopped"
        self._record_global_event("scheduler_stopped", {}, update_latest_snapshot=True)
        return True

    def _force_unmanaged_terminal_contexts_locked(self) -> None:
        """Break shutdown-only waits on predecessor futures we do not own.

        During normal operation a terminal request faithfully waits for its
        predecessor.  Once admission is closed, an unresolved external future
        has no managed producer that this scheduler can drain, so retaining the
        wait would violate the stronger shutdown invariant that every public
        future settles.
        """
        made_progress = True
        while made_progress:
            made_progress = False
            for request in list(self._requests.values()):
                if request.state != "settling_terminal":
                    continue
                if request.context_future.done():
                    self._finish_terminal_request_locked(request)
                    made_progress = True
                    continue
                predecessor_future = request.context_dependency._future
                if predecessor_future is None or predecessor_future.done():
                    try:
                        context = request.context_dependency.copy()
                    except BaseException:
                        context = agcontext()
                else:
                    producer_id = self._future_producers.get(predecessor_future)
                    if producer_id is not None and producer_id in self._requests:
                        continue
                    context = agcontext()
                self._settle_future(
                    request.context_future,
                    context,
                    "forced shutdown context",
                )
                self._finish_terminal_request_locked(request)
                made_progress = True


_global_orchestrator: "GlobalAgentOrchestrator | None" = None
_global_lock = threading.Lock()


def get_orchestrator(
    agconfig: "agconfig_cls | None" = None,
    *,
    default_db_path: "str | Path | None" = None,
) -> GlobalAgentOrchestrator:
    global _global_orchestrator
    with _global_lock:
        if _global_orchestrator is None:
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
    "get_orchestrator",
]
