"""Host interaction server routes used during a sandbox harness attempt."""

from __future__ import annotations

import threading
import time
import uuid
from typing import TYPE_CHECKING, Callable

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ...harness._syscall_event import agsyscallevent
from ...observability.profiler import agprof

# Slack absorbed by translating a reporter's wall-clock timestamp into this
# host's own perf_counter_ns domain (see _wall_clock_to_perf_ns) -- generous
# enough for typical wall-clock sync and the jitter between two independent
# host-local conversions, not a negotiated per-caller value.
_CLOCK_SLACK_NS = 5_000_000

if TYPE_CHECKING:
    from ...observability.agdatalogger import agDataLogger
    from ...agskill import agskill


class HostInteractionServer:
    def __init__(
        self,
        skill: "agskill",
        data_logger: "agDataLogger",
        agname: str,
        *,
        is_cancelled: "Callable[[], bool] | None" = None,
        parent_context=None,
        profile_attributes: dict | None = None,
    ) -> None:
        self._profile_context = parent_context
        self._profile_attributes = dict(profile_attributes or {})
        self._policy = skill.policy
        self._data_logger = data_logger
        self._agname = agname  # for _record_admission's term_message tag
        # Bound by HostServerManager to the exact orchestrator request.  The
        # sandbox never supplies an invocation id and therefore cannot target
        # another request's cancellation state.
        self._is_cancelled = is_cancelled if is_cancelled is not None else (lambda: False)
        # Correlates an admission decision with its later completion report,
        # across the tool/syscall admission <-> completion round trip (which
        # may cross the daemon<->host UDS bridge, or an external harness's
        # own PreToolUse/PostToolUse hook subprocess). Value: (kind, name,
        # attributes, start_ts). Only admitted (allowed) calls are stashed --
        # a denied call never really runs, so it has nothing to complete.
        self._pending_calls: "dict[str, tuple]" = {}
        self._pending_calls_lock = threading.Lock()
        # Spans a caller opened via record_span(span_id=..., end_ts=None) and
        # has not yet closed. Generic -- any reporter (this class's own
        # admission/completion boundaries, or a harness reporting its own
        # internal spans) can open/close/nest through the same dict.
        self._open_spans: "dict[str, object]" = {}
        self._profile_pid = -2 - agprof.next_index("remote-profile")

    def check_tool(self, tool_name: str, tool_input: dict) -> "tuple[bool, str | None]":
        if self._is_cancelled():
            return False, "agent invocation stopped"
        hook = (self._policy.tool_hooks or {}).get(tool_name)
        if hook is None:
            return (not self._policy.default_to_deny, None)
        try:
            result = hook(tool_input)
        except Exception as exc:
            return (False, f"hook raised: {exc}")
        return result if isinstance(result, tuple) else (result, None)

    def check_syscall(self, syscall: "agsyscallevent") -> "tuple[bool, str | None]":
        hook = (self._policy.syscall_hooks or {}).get(syscall.syscall)
        if hook is None:
            return (not self._policy.default_to_deny, None)
        try:
            result = hook(syscall)
        except Exception as exc:
            return (False, f"hook raised: {exc}")
        return result if isinstance(result, tuple) else (result, None)

    _ADMISSION_LABEL = {"tool": "TOOL   ", "syscall": "SYSCALL"}

    def _record_admission(
        self, kind: str, name: str, attributes: dict, allowed: bool, reason: "str | None" = None
    ) -> str:
        call_id = uuid.uuid4().hex
        self.record_event(
            f"{kind}_call",
            {**attributes, "call_id": call_id, "allowed": allowed},
        )
        label = self._ADMISSION_LABEL[kind]
        if kind == "tool":
            args_text = repr(attributes["arguments"])
        elif attributes.get("address") is not None:
            args_text = f"{attributes['address']}:{attributes.get('port')}"
        else:
            args_text = repr(attributes["argv"] or attributes["path"])
        if attributes.get("program"):
            args_text = f"{args_text}  program={attributes['program']}"
        args_suffix = f"  args={args_text}"
        if allowed:
            term_message = f"[{self._agname}] {label} ▶  {name}{args_suffix}"
        else:
            term_message = (
                f"[{self._agname}] {label} ✗  {name}{args_suffix}  "
                f"DENIED: {reason or 'no reason given'}"
            )
        self.record_event(
            "agent_state",
            {"state": f"running_{kind}" if allowed else f"{kind}_denied", kind: name},
            update_latest_snapshot=True,
            term_message=term_message,
            print_to_terminal=False,
            flush=True,
        )
        if allowed:
            with self._pending_calls_lock:
                started = time.time_ns()
                profile_span = agprof.start_external_span(
                    f"{kind}:{name}",
                    start_perf_ns=time.perf_counter_ns(),
                    start_wall_ns=started,
                    metadata={
                        **self._profile_attributes,
                        "call_id": call_id,
                        "timing": "hook_boundary",
                        "provenance": "host_observed",
                    },
                    parent_context=self.current_open_context(),
                )
                self._pending_calls[call_id] = (kind, name, attributes, started / 1e9, profile_span)
        return call_id

    def _record_completion(self, kind: str, call_id: str, extra: dict) -> None:
        with self._pending_calls_lock:
            pending = self._pending_calls.get(call_id)
            if pending is not None and pending[0] == kind:
                self._pending_calls.pop(call_id)
            else:
                pending = None
        if pending is None:
            # Unknown, already-completed, or never-admitted (denied) call --
            # a no-op, not an error: callers report completion best-effort
            # and shouldn't have to track admission outcomes themselves.
            return
        _kind, name, attributes, start_ts, profile_span = pending
        end_perf_ns, end_wall_ns = time.perf_counter_ns(), time.time_ns()
        end_ts = end_wall_ns / 1e9
        outcome = "failure" if extra.get("error") is not None else "success"
        if kind == "syscall" and extra.get("return_value") is None and extra.get("error") is None:
            outcome = "unknown"
        timing = "hook_boundary"
        duration_ns = extra.get("duration_ns")
        overrides = {}
        if duration_ns is not None:
            started_wall_ns = extra.get("started_wall_ns")
            measured_start = (
                self._wall_clock_to_perf_ns(started_wall_ns / 1e9)[0]
                if isinstance(started_wall_ns, int)
                else None
            )
            if (
                isinstance(duration_ns, int)
                and not isinstance(duration_ns, bool)
                and duration_ns >= 0
                and measured_start is not None
                and profile_span is not None
                and measured_start >= profile_span._t0 - _CLOCK_SLACK_NS
                and measured_start + duration_ns <= end_perf_ns + _CLOCK_SLACK_NS
            ):
                timing = "exact"
                measured_end = measured_start + duration_ns
                measured_wall = end_wall_ns + measured_end - end_perf_ns
                end_perf_ns, end_wall_ns = measured_end, measured_wall
                overrides = {
                    "start_perf_ns": measured_start,
                    "start_wall_ns": measured_wall - duration_ns,
                }
                start_ts, end_ts = (measured_wall - duration_ns) / 1e9, measured_wall / 1e9
            else:
                agprof.telemetry_error("invalid_tool_durations")
        if profile_span is not None:
            profile_span.end(
                end_perf_ns=end_perf_ns,
                end_wall_ns=end_wall_ns,
                metadata={
                    "outcome": outcome,
                    "timing": timing,
                    "provenance": "container_asserted" if timing == "exact" else "host_observed",
                    "clock_uncertainty_ns": _CLOCK_SLACK_NS if timing == "exact" else 0,
                },
                **overrides,
            )
        term_message = None
        if kind == "tool":
            mark = {"success": "✓", "failure": "✗", "unknown": "?"}[outcome]
            result_text = repr(extra["error"] if outcome == "failure" else extra.get("result"))
            term_message = (
                f"[{self._agname}] {self._ADMISSION_LABEL[kind]} {mark}  {name}  ={result_text}"
            )
        self.record_event(
            f"{kind}_result",
            {**attributes, **extra, "call_id": call_id},
            term_message=term_message,
            print_to_terminal=False,
        )
        self.record_span(
            f"{kind}:{name}",
            start_ts,
            end_ts,
            {**attributes, "call_id": call_id, "outcome": outcome, "timing": timing},
        )

    def finalize_profile(self) -> None:
        """Retired attempts cannot finish later; retain unmatched starts explicitly."""
        with self._pending_calls_lock:
            pending = list(self._pending_calls.values())
            self._pending_calls.clear()
        for span in self._open_spans.values():
            agprof.interrupt_external_span(span)
        self._open_spans.clear()
        for _kind, _name, _attrs, _start, span in pending:
            agprof.interrupt_external_span(span)
        if pending:
            agprof.telemetry_error("unmatched_completions", len(pending))

    @staticmethod
    def _wall_clock_to_perf_ns(wall_ts: float) -> "tuple[int, int]":
        """Approximate this host's own perf_counter_ns for a wall-clock
        timestamp reported by anyone (this process included), using this
        host's own simultaneous clock readings."""
        now_perf_ns = time.perf_counter_ns()
        now_wall_ns = time.time_ns()
        target_wall_ns = int(wall_ts * 1e9)
        return now_perf_ns + (target_wall_ns - now_wall_ns), target_wall_ns

    def current_open_context(self):
        """Context of the most recently opened, still-open reported span, if
        any -- otherwise the request-level fallback."""
        if self._open_spans:
            return next(reversed(self._open_spans.values())).context()
        return self._profile_context

    def profile_settings(self) -> dict:
        """Whether profiling is on, and automatic-function-sampling settings."""
        settings = agprof._auto_settings
        harness = self._profile_attributes.get("harness")
        if harness in agprof._engine_coverage:
            agprof._engine_coverage[harness]["automatic_functions"] = (
                "host_and_remote_python" if settings else "disabled"
            )
            agprof._engine_coverage[harness]["retries"] = "container_reported"
        return {
            "enabled": agprof.enabled(),
            "automatic": None
            if settings is None
            else {
                key: settings[key]
                for key in ("min_duration_ms", "max_depth", "max_events", "include_dependencies")
            },
        }

    def record_samples(self, samples: "list[dict]") -> dict:
        """Ingest a batch of already-measured function-call samples."""
        if not isinstance(samples, list) or len(samples) > 128:
            return {"ok": False, "error": "sample batch exceeds 128 events"}
        harness = self._profile_attributes.get("harness") or "remote"
        rejected = agprof.ingest_auto_samples(
            self._profile_pid, samples, thread_label=f"{harness} thread"
        )
        return {"ok": rejected == 0, "rejected": rejected}

    def admit_tool_call(self, tool_name: str, tool_input: dict) -> dict:
        """Admission + telemetry entry point for a tool call"""
        allowed, reason = self.check_tool(tool_name, tool_input)
        call_id = self._record_admission(
            "tool", tool_name, {"tool": tool_name, "arguments": tool_input}, allowed, reason
        )
        return {"allowed": allowed, "reason": reason, "call_id": call_id}

    def complete_tool_call(
        self,
        call_id: str,
        result: object = None,
        error: "str | None" = None,
        duration_ns: int | None = None,
        started_wall_ns: int | None = None,
    ) -> None:
        self._record_completion(
            "tool",
            call_id,
            {
                "result": result,
                "error": error,
                **({"duration_ns": duration_ns} if duration_ns is not None else {}),
                **({"started_wall_ns": started_wall_ns} if started_wall_ns is not None else {}),
            },
        )

    def admit_syscall(self, syscall: "agsyscallevent") -> dict:
        """Admission + telemetry entry point for a syscall"""
        allowed, reason = self.check_syscall(syscall)
        call_id = self._record_admission(
            "syscall",
            syscall.syscall,
            {
                "syscall": syscall.syscall,
                "pid": syscall.pid,
                "path": syscall.path,
                "argv": syscall.argv,
                "program": syscall.program,
                "address": syscall.address,
                "port": syscall.port,
            },
            allowed,
            reason,
        )
        return {"allowed": allowed, "reason": reason, "call_id": call_id}

    def complete_syscall(
        self, call_id: str, return_value: "int | None" = None, error: "str | None" = None
    ) -> None:
        self._record_completion("syscall", call_id, {"return_value": return_value, "error": error})

    def record_event(
        self,
        type: str,
        payload: dict,
        call_label: "str | None" = None,
        update_latest_snapshot: bool = False,
        term_message: "str | None" = None,
        print_to_terminal: bool = True,
        flush: bool = False,
    ) -> None:
        self._data_logger.record_event(
            type,
            payload,
            call_label=call_label,
            update_latest_snapshot=update_latest_snapshot,
            term_message=term_message,
            print_to_terminal=print_to_terminal,
            flush=flush,
        )

    def record_span(
        self,
        name: str,
        start_ts: "float | None",
        end_ts: "float | None",
        attributes: dict,
        cpu_ms: "float | None" = None,
        runqueue_ms: "float | None" = None,
        blocked_ms: "float | None" = None,
        parent: "str | None" = None,
        call_label: "str | None" = None,
        span_id: "str | None" = None,
    ) -> None:
        """Log one flat historical span row -- and, when *span_id* is given,
        also fold it into the live profiler trace with correct nesting."""
        if span_id is not None:
            opened = self._open_spans.pop(span_id, None)
            if opened is None and start_ts is not None and len(self._open_spans) < 1024:
                parent_span = self._open_spans.get(parent)
                start_perf_ns, start_wall_ns = self._wall_clock_to_perf_ns(start_ts)
                opened = agprof.start_external_span(
                    name,
                    start_perf_ns=start_perf_ns,
                    start_wall_ns=start_wall_ns,
                    parent_context=(
                        parent_span.context() if parent_span is not None else self._profile_context
                    ),
                    metadata={**self._profile_attributes, **attributes},
                )
            if end_ts is None:
                if opened is not None:
                    self._open_spans[span_id] = opened
                return
            if start_ts is None and opened is not None:
                start_ts = opened._wall0 / 1e9
            if opened is not None:
                end_perf_ns, end_wall_ns = self._wall_clock_to_perf_ns(end_ts)
                opened.end(
                    end_perf_ns=end_perf_ns, end_wall_ns=end_wall_ns, metadata=dict(attributes)
                )
        self._data_logger.record_span(
            name,
            start_ts,
            end_ts,
            attributes,
            cpu_ms=cpu_ms,
            runqueue_ms=runqueue_ms,
            blocked_ms=blocked_ms,
            parent=parent,
            call_label=call_label,
        )

    def build_app(self) -> FastAPI:
        app = FastAPI()

        @app.post("/profile_settings")
        def _profile_settings() -> JSONResponse:
            return JSONResponse(self.profile_settings())

        @app.post("/record_samples")
        def _record_samples(request: dict) -> JSONResponse:
            return JSONResponse(self.record_samples(request.get("samples", [])))

        @app.post("/check_tool")
        def _check_tool(request: dict) -> JSONResponse:
            return JSONResponse(self.admit_tool_call(request["tool_name"], request["tool_input"]))

        @app.post("/complete_tool")
        def _complete_tool(request: dict) -> JSONResponse:
            self.complete_tool_call(
                request["call_id"],
                request.get("result"),
                request.get("error"),
                request.get("duration_ns"),
                request.get("started_wall_ns"),
            )
            return JSONResponse({"ok": True})

        @app.post("/check_syscall")
        def _check_syscall(request: dict) -> JSONResponse:
            return JSONResponse(self.admit_syscall(agsyscallevent(**request)))

        @app.post("/complete_syscall")
        def _complete_syscall(request: dict) -> JSONResponse:
            self.complete_syscall(
                request["call_id"], request.get("return_value"), request.get("error")
            )
            return JSONResponse({"ok": True})

        @app.post("/record_event")
        def _record_event(request: dict) -> JSONResponse:
            self.record_event(
                request["type"],
                request["payload"],
                call_label=request.get("call_label"),
                update_latest_snapshot=request.get("update_latest_snapshot", False),
                term_message=request.get("term_message"),
                flush=request.get("flush", False),
            )
            return JSONResponse({"ok": True})

        @app.post("/record_span")
        def _record_span(request: dict) -> JSONResponse:
            self.record_span(
                request["name"],
                request.get("start_ts"),
                request.get("end_ts"),
                request["attributes"],
                cpu_ms=request.get("cpu_ms"),
                runqueue_ms=request.get("runqueue_ms"),
                blocked_ms=request.get("blocked_ms"),
                parent=request.get("parent"),
                call_label=request.get("call_label"),
                span_id=request.get("span_id"),
            )
            return JSONResponse({"ok": True})

        return app
