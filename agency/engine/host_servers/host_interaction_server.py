"""Host interaction server routes used during a sandbox harness attempt."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from contextlib import suppress
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ...harness._syscall_event import agsyscallevent
from ...observability.profiler import agprof

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
        invocation=None,
        admit_tools: bool = True,
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
        # another request's lifecycle state.
        self._invocation = invocation
        self._admit_tools = admit_tools
        # Correlates an admission decision with its later completion report,
        # across the tool/syscall admission <-> completion round trip (which
        # may cross the daemon<->host UDS bridge, or an external harness's
        # own PreToolUse/PostToolUse hook subprocess). Value: (kind, name,
        # attributes, start_ts). Only admitted (allowed) calls are stashed --
        # a denied call never really runs, so it has nothing to complete.
        self._pending_calls: "dict[str, tuple]" = {}
        self._pending_calls_lock = threading.Lock()
        self._remote_spans: dict[str, object] = {}
        self._remote_turn: str | None = None
        self._profile_session_id = None
        self._clock_uncertainty_ns = 0
        self._profile_pid = -2 - agprof.next_index("remote-profile")

    def checkpoint(self, boundary_id: str, *, allow_messages: bool, phase: str) -> dict:
        if self._invocation is None:
            result = {"cancelled": False, "destroyed": False, "invocation_messages": []}
            if phase == "action":
                result["action_admitted"] = True
            return result
        decision = self._invocation._checkpoint(
            boundary_id,
            allow_messages=allow_messages,
            phase=phase,
        )
        return self._serialize_checkpoint_decision(decision)

    @staticmethod
    def _serialize_checkpoint_decision(decision) -> dict:
        def value(item, name: str, default=None):
            return (
                item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)
            )

        result = {
            "cancelled": bool(value(decision, "cancelled", False)),
            "destroyed": bool(value(decision, "destroyed", False)),
            "invocation_messages": [
                {
                    "sequence": int(value(entry, "sequence", 0)),
                    "content": str(value(entry, "content", "")),
                }
                for entry in (value(decision, "invocation_messages", ()) or ())
            ],
        }
        if value(decision, "action_admitted", False):
            result["action_admitted"] = True
        return result

    def _checkpoint_interruptibly(
        self,
        boundary_id: str,
        *,
        allow_messages: bool,
        phase: str,
        abort_event: threading.Event,
    ) -> "dict | None":
        if self._invocation is None:
            return (
                None
                if abort_event.is_set()
                else self.checkpoint(boundary_id, allow_messages=allow_messages, phase=phase)
            )
        checkpoint = getattr(self._invocation, "_checkpoint_interruptibly", None)
        if checkpoint is None:
            # Preserve duck-typed test and third-party invocation fakes that
            # implement only the established private checkpoint seam.
            decision = self._invocation._checkpoint(
                boundary_id,
                allow_messages=allow_messages,
                phase=phase,
            )
        else:
            decision = checkpoint(
                boundary_id,
                allow_messages=allow_messages,
                phase=phase,
                abort_event=abort_event,
            )
        return None if decision is None else self._serialize_checkpoint_decision(decision)

    def _abort_checkpoint_wait(self, abort_event: threading.Event) -> None:
        abort = getattr(self._invocation, "_abort_checkpoint_wait", None)
        if abort is None:
            abort_event.set()
            return
        abort(abort_event)

    def check_tool(self, tool_name: str, tool_input: dict) -> "tuple[bool, str | None]":
        if self._invocation is not None and self._admit_tools:
            decision = self._invocation._checkpoint(
                "external:action", allow_messages=False, phase="action"
            )
            if decision.destroyed or decision.cancelled:
                return False, "agent invocation stopped"
            if not decision.action_admitted:
                return (
                    False,
                    "Invocation redirected. Return to the model before taking another action.",
                )
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
        else:
            args_text = repr(attributes["argv"] or attributes["path"])
        if len(args_text) > 200:
            args_text = f"{args_text[:200]}…"
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
                    parent_context=self.profile_parent_context(),
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
            measured_start = extra.get("started_perf_ns")
            if (
                isinstance(duration_ns, int)
                and not isinstance(duration_ns, bool)
                and duration_ns >= 0
                and isinstance(measured_start, int)
                and profile_span is not None
                and measured_start >= profile_span._t0 - self._clock_uncertainty_ns
                and measured_start + duration_ns <= end_perf_ns + self._clock_uncertainty_ns
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
                    "clock_uncertainty_ns": self._clock_uncertainty_ns if timing == "exact" else 0,
                },
                **overrides,
            )
        term_message = None
        if kind == "tool":
            mark = {"success": "✓", "failure": "✗", "unknown": "?"}[outcome]
            result_text = repr(extra["error"] if outcome == "failure" else extra.get("result"))
            if len(result_text) > 200:
                result_text = f"{result_text[:200]}…"
            term_message = (
                f"[{self._agname}] {self._ADMISSION_LABEL[kind]} {mark}  {name}  ={result_text}"
            )
        self.record_event(
            f"{kind}_result", {**attributes, **extra, "call_id": call_id}, term_message=term_message
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
        for span in self._remote_spans.values():
            agprof.interrupt_external_span(span)
        self._remote_spans.clear()
        self._remote_turn = None
        for _kind, _name, _attrs, _start, span in pending:
            agprof.interrupt_external_span(span)
        if pending:
            agprof.telemetry_error("unmatched_completions", len(pending))

    def profile_parent_context(self):
        turn = self._remote_spans.get(self._remote_turn)
        return turn.context() if turn is not None else self._profile_context

    def profile_config(self) -> dict:
        self._profile_session_id = agprof._profile_session_id
        settings = agprof._auto_settings
        engine = self._profile_attributes.get("harness")
        if engine == "native" and engine in agprof._engine_coverage:
            agprof._engine_coverage[engine]["automatic_functions"] = (
                "host_and_native_python" if settings else "disabled"
            )
            agprof._engine_coverage[engine]["retries"] = "container_reported"
        return {
            "enabled": agprof.enabled(),
            "session_id": self._profile_session_id,
            "host_perf_ns": time.perf_counter_ns(),
            "automatic": None
            if settings is None
            else {
                key: settings[key]
                for key in ("min_duration_ms", "max_depth", "max_events", "include_dependencies")
            },
        }

    def profile_events(self, request: dict) -> dict:
        """Accept bounded telemetry only inside the manager's authenticated attempt fence."""
        if (
            not agprof.enabled()
            or request.get("session_id") != self._profile_session_id
            or self._profile_session_id != agprof._profile_session_id
        ):
            return {"ok": False, "error": "inactive profile session"}
        events = request.get("events", [])
        if not isinstance(events, list) or len(events) > 128:
            return {"ok": False, "error": "profile batch exceeds 128 events"}
        uncertainty = request.get("clock_uncertainty_ns", 0)
        if not isinstance(uncertainty, int) or uncertainty < 0 or uncertainty > 1_000_000_000:
            return {"ok": False, "error": "invalid clock uncertainty"}
        self._clock_uncertainty_ns = uncertainty
        now = time.perf_counter_ns()
        rejected = 0
        for event in events:
            try:
                name = event["name"]
                if not isinstance(name, str) or not name or len(name) > 512:
                    raise ValueError("invalid name")
                started = int(event["perf_ns"])
                if started < agprof._session_started_ns or started > now + 1_000_000_000:
                    raise ValueError("timestamp outside profile session")
                wall = time.time_ns() + started - time.perf_counter_ns()
                kind = event["kind"]
                if kind == "automatic":
                    duration = int(event["duration_ns"])
                    if duration < 0 or started + duration > now + 1_000_000_000:
                        raise ValueError("invalid interval")
                    settings = agprof._auto_settings
                    if settings is None:
                        continue
                    if len(agprof._auto_records) >= settings["max_events"]:
                        agprof.telemetry_error("remote_auto_dropped")
                        continue
                    agprof._auto_records.append(
                        (
                            self._profile_pid,
                            int(event["tid"]),
                            name,
                            str(event.get("filename", ""))[:1024],
                            int(event.get("lineno", 0)),
                            started,
                            duration,
                            str(event.get("outcome", "unknown"))[:32],
                            "Native Python",
                        )
                    )
                elif kind == "start":
                    identifier = event["id"]
                    if (
                        not isinstance(identifier, str)
                        or not identifier
                        or len(identifier) > 128
                        or identifier in self._remote_spans
                    ):
                        raise ValueError("invalid or duplicate span id")
                    if len(self._remote_spans) >= 1024:
                        raise ValueError("too many open remote spans")
                    parent_id = event.get("parent_id")
                    if parent_id is not None and parent_id not in self._remote_spans:
                        raise ValueError("unknown parent")
                    parent = self._remote_spans.get(parent_id)
                    if parent is not None and started < parent._t0:
                        raise ValueError("child starts before parent")
                    span = agprof.start_external_span(
                        name,
                        start_perf_ns=started,
                        start_wall_ns=wall,
                        parent_context=parent.context() if parent else self._profile_context,
                        metadata={
                            **self._profile_attributes,
                            "timing": "exact",
                            "provenance": "container_asserted",
                            "clock_uncertainty_ns": max(
                                0, min(int(request.get("clock_uncertainty_ns", 0)), 1_000_000_000)
                            ),
                        },
                    )
                    self._remote_spans[identifier] = span
                    if name.startswith("turn"):
                        self._remote_turn = identifier
                elif kind == "end":
                    identifier = event["id"]
                    span = self._remote_spans.get(identifier)
                    if span is None or started < span._t0:
                        raise ValueError("unmatched end")
                    self._remote_spans.pop(identifier)
                    outcome = event.get("outcome", "unknown")
                    if outcome not in ("success", "failure", "unknown"):
                        outcome = "unknown"
                    span.end(end_perf_ns=started, end_wall_ns=wall, metadata={"outcome": outcome})
                    if self._remote_turn == identifier:
                        self._remote_turn = None
                else:
                    raise ValueError("invalid event kind")
            except (KeyError, ValueError, TypeError, OverflowError):
                rejected += 1
                agprof.telemetry_error("remote_events_rejected")
        dropped = request.get("dropped", 0)
        if isinstance(dropped, int) and dropped > 0:
            agprof.telemetry_error("remote_events_dropped", dropped)
        return {"ok": rejected == 0, "rejected": rejected}

    def admit_tool_call(self, tool_name: str, tool_input: dict) -> dict:
        """Admission + telemetry entry point for a tool call: decide
        allow/deny via `check_tool()`, then record a `tool_call` event and
        (if allowed) open a pending span completed later by
        `complete_tool_call()`."""
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
        started_perf_ns: int | None = None,
    ) -> None:
        self._record_completion(
            "tool",
            call_id,
            {
                "result": result,
                "error": error,
                **({"duration_ns": duration_ns} if duration_ns is not None else {}),
                **({"started_perf_ns": started_perf_ns} if started_perf_ns is not None else {}),
            },
        )

    def admit_syscall(self, syscall: "agsyscallevent") -> dict:
        """Admission + telemetry entry point for a syscall, symmetric with
        `admit_tool_call()`."""
        allowed, reason = self.check_syscall(syscall)
        call_id = self._record_admission(
            "syscall",
            syscall.syscall,
            {
                "syscall": syscall.syscall,
                "pid": syscall.pid,
                "path": syscall.path,
                "argv": syscall.argv,
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
        flush: bool = False,
    ) -> None:
        self._data_logger.record_event(
            type,
            payload,
            call_label=call_label,
            update_latest_snapshot=update_latest_snapshot,
            term_message=term_message,
            flush=flush,
        )

    def record_span(
        self,
        name: str,
        start_ts: float,
        end_ts: float,
        attributes: dict,
        cpu_ms: "float | None" = None,
        runqueue_ms: "float | None" = None,
        blocked_ms: "float | None" = None,
        parent: "str | None" = None,
        call_label: "str | None" = None,
    ) -> None:
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

        @app.post("/profile/config")
        def _profile_config() -> JSONResponse:
            return JSONResponse(self.profile_config())

        @app.post("/profile/events")
        def _profile_events(request: dict) -> JSONResponse:
            return JSONResponse(self.profile_events(request))

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
                request.get("started_perf_ns"),
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

        @app.post("/checkpoint")
        async def _checkpoint(request: dict, raw_request: Request) -> JSONResponse:
            boundary_id = request.get("boundary_id")
            phase = request.get("phase")
            allow_messages = request.get("allow_messages")
            if (
                not isinstance(boundary_id, str)
                or not boundary_id
                or not isinstance(phase, str)
                or not phase
                or not isinstance(allow_messages, bool)
            ):
                return JSONResponse({"error": "invalid lifecycle checkpoint"}, status_code=400)

            abort_event = threading.Event()
            checkpoint_task = asyncio.create_task(
                asyncio.to_thread(
                    self._checkpoint_interruptibly,
                    boundary_id,
                    allow_messages=allow_messages,
                    phase=phase,
                    abort_event=abort_event,
                )
            )

            async def wait_for_disconnect() -> None:
                while True:
                    message = await raw_request.receive()
                    if message["type"] == "http.disconnect":
                        return

            disconnect_task = asyncio.create_task(wait_for_disconnect())
            try:
                done, _pending = await asyncio.wait(
                    (checkpoint_task, disconnect_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if checkpoint_task in done:
                    result = checkpoint_task.result()
                    if result is None:
                        return JSONResponse(
                            {"error": "lifecycle checkpoint aborted"}, status_code=499
                        )
                    return JSONResponse(result)

                # Do not cancel the to_thread task: that abandons its worker
                # while the invocation condition remains blocked. Signal the
                # wait itself, wake the condition, and join the worker first.
                self._abort_checkpoint_wait(abort_event)
                await checkpoint_task
                return JSONResponse(
                    {"error": "lifecycle checkpoint client disconnected"},
                    status_code=499,
                )
            finally:
                disconnect_task.cancel()
                with suppress(asyncio.CancelledError):
                    await disconnect_task
                if not checkpoint_task.done():
                    self._abort_checkpoint_wait(abort_event)
                    with suppress(asyncio.CancelledError):
                        await asyncio.shield(checkpoint_task)

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
                request["start_ts"],
                request["end_ts"],
                request["attributes"],
                cpu_ms=request.get("cpu_ms"),
                runqueue_ms=request.get("runqueue_ms"),
                blocked_ms=request.get("blocked_ms"),
                parent=request.get("parent"),
                call_label=request.get("call_label"),
            )
            return JSONResponse({"ok": True})

        return app
