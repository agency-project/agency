"""Host interaction server routes used during a sandbox harness attempt."""

from __future__ import annotations

import asyncio
import threading
from contextlib import suppress
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ...harness._syscall_event import agsyscallevent

if TYPE_CHECKING:
    from ...observability.agdatalogger import agDataLogger
    from ...agskill import agskill


class HostInteractionServer:
    def __init__(
        self,
        skill: "agskill",
        data_logger: "agDataLogger",
        *,
        invocation=None,
        admit_tools: bool = True,
    ) -> None:
        self._policy = skill.policy
        self._data_logger = data_logger
        # Bound by HostServerManager to the exact orchestrator request.  The
        # sandbox never supplies an invocation id and therefore cannot target
        # another request's lifecycle state.
        self._invocation = invocation
        self._admit_tools = admit_tools

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

        @app.post("/check_tool")
        def _check_tool(request: dict) -> JSONResponse:
            allowed, reason = self.check_tool(request["tool_name"], request["tool_input"])
            return JSONResponse({"allowed": allowed, "reason": reason})

        @app.post("/check_syscall")
        def _check_syscall(request: dict) -> JSONResponse:
            allowed, reason = self.check_syscall(agsyscallevent(**request))
            return JSONResponse({"allowed": allowed, "reason": reason})

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
