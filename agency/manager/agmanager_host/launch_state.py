"""Per-launch state and registry for `agmanager_host`.

One agent runs many separate skill launches over its lifetime (and, in
principle, concurrent ones from subagents) -- each with its own `skill`/
`output_schema`, collected structured-output fields, transcript, and
profiler run context. `LaunchRegistry` is the shared per-launch scoping
every other module in this package (`llm_dispatch.py`, `control_routes.py`,
`mcp_tools.py`, `profiler_ingest.py`) reads and writes through, keyed by a
short-lived launch token -- same shape the old per-token dicts on
`agllm_terminus`/`agmcp_server`/`agprof_ingest` had, just no longer needing
to *also* look up which agent a token belongs to (there is only ever one,
for this instance). See `agmanager_host.py`'s module docstring for the
full two-server design this is part of.

`lock`/`launches` are exposed as plain public attributes, not hidden behind
only the convenience methods below -- `profiler_ingest.py` needs to mutate
a `_LaunchState`'s own fields (`open_remote_spans`, `clock_offsets`, ...)
as part of a larger atomic sequence, which the simple accessor methods here
don't cover; it uses `registry.lock`/`registry.launches` directly for that,
the same way the original monolithic class used `self._lock`/`self._launches`
internally."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...agent import agent
    from ...agskill import agskill


@dataclass
class _LaunchState:
    """Per-launch (per skill `execute()` call) state."""

    skill: "agskill | None" = None
    exact_events: bool = False
    exact_tool_events: bool = False
    collected_output: dict = field(default_factory=dict)
    transcript: "list[dict] | None" = None
    run_context: object | None = None
    span_attributes: dict = field(default_factory=dict)
    clock_offsets: "tuple[int, int] | None" = None
    clock_sync_estimate: "tuple[int, int] | None" = None
    # Value type is profiler_ingest._RemoteSpanStart -- kept as a loose
    # `object` here (not imported) so profiler_ingest.py can import THIS
    # module without a cycle; it never needs to import from here the other
    # way around.
    open_remote_spans: "dict[str, object]" = field(default_factory=dict)
    exact_tool_call_ids: set = field(default_factory=set)


class LaunchHandle:
    """Convenience wrapper a caller can hold onto for the duration of one
    launch instead of threading a bare token string through its own code --
    purely ergonomic, `registry`/`token` are both still directly usable."""

    __slots__ = ("registry", "token")

    def __init__(self, registry: "LaunchRegistry", token: str) -> None:
        self.registry = registry
        self.token = token

    def collected_output(self) -> dict:
        return self.registry.collected_output(self.token)

    def transcript(self) -> "list[dict] | None":
        return self.registry.transcript_for_token(self.token)

    def unregister(self) -> None:
        self.registry.unregister(self.token)

    def __enter__(self) -> "LaunchHandle":
        return self

    def __exit__(self, *exc_info) -> None:
        self.unregister()


class LaunchRegistry:
    """One per agent, owned by `agHostAgentManager` and shared by every
    feature module that needs to look up or mutate a launch's state."""

    def __init__(self, ag: "agent") -> None:
        self._ag = ag
        self.lock = threading.Lock()
        self.launches: "dict[str, _LaunchState]" = {}

    def register(
        self,
        token: "str | None" = None,
        *,
        skill: "agskill | None" = None,
        exact_events: bool = False,
        exact_tool_events: bool = False,
    ) -> LaunchHandle:
        token = token or uuid.uuid4().hex
        from ...profiler import agprof

        run_context = agprof.current_span_context()
        current_attributes = agprof.current_span_attributes()
        span_attributes = {
            key: current_attributes[key]
            for key in ("agency.run_id", "agency.agent_id", "agency.parent_agent_id")
            if key in current_attributes
        }
        span_attributes.setdefault("agency.agent_id", str(self._ag.agname))
        parent_agent_id = getattr(self._ag, "_parent_agent_id", None)
        if parent_agent_id is not None:
            span_attributes.setdefault("agency.parent_agent_id", str(parent_agent_id))
        with self.lock:
            self.launches[token] = _LaunchState(
                skill=skill,
                exact_events=exact_events,
                exact_tool_events=exact_events or exact_tool_events,
                run_context=run_context,
                span_attributes=span_attributes,
            )
        return LaunchHandle(self, token)

    def unregister(self, token: str) -> None:
        from ...profiler import agprof

        with self.lock:
            launch = self.launches.pop(token, None)
        if launch is None:
            return
        for open_span in launch.open_remote_spans.values():
            agprof.interrupt_external_span(open_span.handle)

    def get(self, token: "str | None") -> "_LaunchState | None":
        if token is None:
            return None
        with self.lock:
            return self.launches.get(token)

    def collected_output(self, token: str) -> dict:
        with self.lock:
            launch = self.launches.get(token)
            return dict(launch.collected_output) if launch is not None else {}

    def transcript_for_token(self, token: "str | None") -> "list[dict] | None":
        launch = self.get(token)
        if launch is None or launch.transcript is None:
            return None
        with self.lock:
            return list(launch.transcript)

    def record_transcript(self, token: str, request_messages, response_message: dict) -> None:
        with self.lock:
            launch = self.launches.get(token)
            if launch is not None:
                launch.transcript = list(request_messages or []) + [response_message]


__all__ = ["LaunchHandle", "LaunchRegistry"]
