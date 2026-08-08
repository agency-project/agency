"""Host-side correlation registry for harness profiler traffic.

Each harness launch already has a unique bearer token used to authenticate
its LLM traffic.  This module resolves that credential to the launch's agent
and active ``run{N}`` span context on the host.  The token remains only a dict
key: it is never copied into span attributes or profiler summaries.

The registry deliberately belongs to its own service rather than reaching
into agLLMTerminus's token map.  A later in-container emitter can use the same
mapping over its own UDS listener without coupling telemetry ingestion to the
LLM server's event loop.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..profiler import agprof

if TYPE_CHECKING:
    from ..agent import agent


@dataclass(frozen=True)
class _Registration:
    agent: "agent"
    run_context: object | None
    span_attributes: dict


class agProfilerIngest:
    """Token-to-run correlation state shared by all harness backends."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._registrations: "dict[str, _Registration]" = {}

    def register(self, token: str, ag: "agent", *, run_context=None) -> None:
        if run_context is None:
            run_context = agprof.current_span_context()
        current_attributes = agprof.current_span_attributes()
        attributes = {
            key: current_attributes[key]
            for key in (
                "agency.run_id",
                "agency.agent_id",
                "agency.parent_agent_id",
            )
            if key in current_attributes
        }
        attributes.setdefault("agency.agent_id", str(ag.agname))
        parent_agent_id = getattr(ag, "_parent_agent_id", None)
        if parent_agent_id is not None:
            attributes.setdefault("agency.parent_agent_id", str(parent_agent_id))
        with self._lock:
            self._registrations[token] = _Registration(ag, run_context, attributes)

    def unregister(self, token: str) -> None:
        with self._lock:
            self._registrations.pop(token, None)

    def agent_for_token(self, token: "str | None"):
        registration = self._registration_for_token(token)
        return registration.agent if registration is not None else None

    def context_for_token(self, token: "str | None"):
        registration = self._registration_for_token(token)
        return registration.run_context if registration is not None else None

    def attributes_for_token(self, token: "str | None") -> dict:
        registration = self._registration_for_token(token)
        return dict(registration.span_attributes) if registration is not None else {}

    def _registration_for_token(self, token: "str | None") -> "_Registration | None":
        if token is None:
            return None
        with self._lock:
            return self._registrations.get(token)


_shared_profiler_ingest: "agProfilerIngest | None" = None
_shared_profiler_ingest_lock = threading.Lock()


def get_shared_profiler_ingest() -> agProfilerIngest:
    global _shared_profiler_ingest
    if _shared_profiler_ingest is not None:
        return _shared_profiler_ingest
    with _shared_profiler_ingest_lock:
        if _shared_profiler_ingest is None:
            _shared_profiler_ingest = agProfilerIngest()
        return _shared_profiler_ingest


__all__ = ["agProfilerIngest", "get_shared_profiler_ingest"]
