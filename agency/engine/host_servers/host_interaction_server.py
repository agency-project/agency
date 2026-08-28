"""Host interaction server routes used during a sandbox harness attempt."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ...harness._syscall_event import agsyscallevent

if TYPE_CHECKING:
    from ...agDataCollector import agDataCollector
    from ...agent import agent
    from ...agskill import agskill


class HostInteractionServer:
    def __init__(self, agent: "agent", skill: "agskill", data_collector: "agDataCollector") -> None:
        self._agent = agent
        self._policy = skill.policy
        self._data_collector = data_collector

    def check_tool(self, tool_name: str, tool_input: dict) -> "tuple[bool, str | None]":
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

    def update_state(
        self, new_state: str, skill: "str | None" = None, tool: "str | None" = None
    ) -> None:
        self._agent._state.update_state(new_state, skill, tool)

    def record_event(
        self, type: str, payload: dict, call_label: "str | None" = None, do_update: bool = False
    ) -> None:
        self._data_collector.record_event(type, payload, call_label=call_label, do_update=do_update)

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
        self._data_collector.record_span(
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

        @app.post("/update_state")
        def _update_state(request: dict) -> JSONResponse:
            self.update_state(request["state"], request.get("skill"), request.get("tool"))
            return JSONResponse({"ok": True})

        @app.post("/record_event")
        def _record_event(request: dict) -> JSONResponse:
            self.record_event(
                request["type"],
                request["payload"],
                call_label=request.get("call_label"),
                do_update=request.get("do_update", False),
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
