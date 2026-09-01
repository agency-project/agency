from __future__ import annotations

import heapq
from concurrent.futures import Future
from typing import TYPE_CHECKING, Callable

from ..agdata import agdata, agerror
from ..agutil import format_exception

if TYPE_CHECKING:
    from .orchestrator import GlobalAgentOrchestrator, _ExecutionRequest


class ExecutionScheduler:
    """Resolve the wait pool, promote ready work, and apply a schedule policy."""

    def __init__(self, orchestrator: "GlobalAgentOrchestrator") -> None:
        self._orchestrator = orchestrator
        self._ready: list[tuple[int, str]] = []
        # Policy seam: a future scheduler can replace this callable while the
        # default continues to launch every eligible ready request.
        self.schedule: Callable[[], None] = self.default_schedule

    def execute(self) -> None:
        """Run one complete dependency-resolution and scheduling cycle."""
        owner = self._orchestrator
        resolved: list[_ExecutionRequest] = []
        wait_pool = sorted(
            (
                request
                for request in owner._requests.values()
                if request.state in ("submitted", "blocked")
            ),
            key=lambda request: request.sequence,
        )
        for pending_exec in wait_pool:
            if self.resolve_dependency(pending_exec):
                resolved.append(pending_exec)

        self._fail_dependency_cycles()
        self._promote_resolved(resolved)
        self.schedule()

    def resolve_dependency(self, pending_exec: "_ExecutionRequest") -> bool:
        """Return true when one wait-pool request is ready for promotion."""
        owner = self._orchestrator
        dependencies: set[Future] = set()
        failure = self._discover_dependencies(pending_exec.skill_input, dependencies)
        if failure is not None:
            owner._fail_request_locked(pending_exec, failure)
            return False

        previous_dependencies = pending_exec.dependencies
        previous_producer_ids = pending_exec.producer_ids
        new_dependencies = dependencies - pending_exec.dependencies
        pending_exec.dependencies = dependencies
        pending_exec.producer_ids = {
            producer_id
            for future in dependencies
            if (producer_id := owner._future_producers.get(future)) is not None
        }
        for dependency in new_dependencies:
            dependency.add_done_callback(
                lambda finished, rid=pending_exec.request_id: owner._post(
                    "dependency_done", (rid, finished)
                )
            )

        if dependencies:
            transitioned = pending_exec.state != "blocked"
            dependencies_changed = dependencies != previous_dependencies
            producers_changed = pending_exec.producer_ids != previous_producer_ids
            if transitioned:
                owner._start_phase_span_locked(pending_exec, "sync:dependency_wait")
            pending_exec.state = "blocked"
            if transitioned or producers_changed:
                self.set_agent_blocked(pending_exec)
            if transitioned or dependencies_changed:
                owner._publish_request_locked(
                    "request_blocked",
                    pending_exec,
                    {"dependency_count": len(dependencies)},
                )
            return False
        return True

    def set_agent_blocked(self, pending_exec: "_ExecutionRequest") -> None:
        owner = self._orchestrator
        producer_agent = None
        for producer_id in pending_exec.producer_ids:
            producer = owner._requests.get(producer_id)
            if producer is not None:
                producer_agent = producer.agent
                break
        pending_exec.agent._state.blocked_on = producer_agent
        if pending_exec.agent not in owner._active_by_agent:
            pending_exec.agent._set_ui_state("blocked_on_dependency", skill=pending_exec.skill.name)

    def _promote_resolved(self, resolved: "list[_ExecutionRequest]") -> None:
        owner = self._orchestrator
        for pending_exec in resolved:
            if pending_exec.request_id not in owner._requests or pending_exec.state not in (
                "submitted",
                "blocked",
            ):
                continue
            owner._start_phase_span_locked(pending_exec, "sync:scheduler_queue")
            pending_exec.state = "ready"
            pending_exec.agent._state.blocked_on = None
            heapq.heappush(self._ready, (pending_exec.sequence, pending_exec.request_id))
            if pending_exec.agent not in owner._active_by_agent:
                pending_exec.agent._set_ui_state("queued", skill=pending_exec.skill.name)
            owner._publish_request_locked("request_ready", pending_exec, {})

    def default_schedule(self) -> None:
        """Launch every eligible ready request allowed by current capacity."""
        owner = self._orchestrator
        while owner._has_capacity_locked():
            skipped: list[tuple[int, str]] = []
            selected: "_ExecutionRequest | None" = None
            while self._ready:
                entry = heapq.heappop(self._ready)
                pending_exec = owner._requests.get(entry[1])
                if pending_exec is None or pending_exec.state != "ready":
                    continue
                if pending_exec.agent in owner._active_by_agent:
                    skipped.append(entry)
                    continue
                selected = pending_exec
                break
            for entry in skipped:
                heapq.heappush(self._ready, entry)
            if selected is None:
                return
            owner._launch_request_locked(selected)

    def materialize_dependencies(self, value: object) -> object:
        """Replace completed dependency wrappers with their resolved values."""
        seen: set[int] = set()

        def materialize(current: object) -> object:
            if isinstance(current, agdata):
                marker = id(current)
                if marker in seen:
                    return current
                seen.add(marker)
                future = object.__getattribute__(current, "_future")
                if future is not None:
                    if not future.done():
                        raise RuntimeError(
                            "scheduler dispatched a request with an unresolved dependency"
                        )
                    resolved = future.result()
                    if isinstance(resolved, agerror):
                        raise RuntimeError(resolved.error)
                    resolved = materialize(resolved)
                    object.__setattr__(
                        current,
                        "_data",
                        object.__getattribute__(resolved, "_data"),
                    )
                    object.__setattr__(current, "_future", None)
                data = object.__getattribute__(current, "_data")
                for key, nested in list(data.items()):
                    data[key] = materialize(nested)
                return current
            if isinstance(current, dict):
                for key, nested in list(current.items()):
                    current[key] = materialize(nested)
                return current
            if isinstance(current, list):
                for index, nested in enumerate(current):
                    current[index] = materialize(nested)
                return current
            if isinstance(current, tuple):
                return tuple(materialize(nested) for nested in current)
            return current

        return materialize(value)

    def _discover_dependencies(self, value: object, found: set[Future]) -> "str | None":
        seen: set[int] = set()

        def visit(current: object, *, from_future: bool = False) -> "str | None":
            if isinstance(current, agerror):
                return current.error if from_future else None
            if isinstance(current, agdata):
                marker = id(current)
                if marker in seen:
                    return None
                seen.add(marker)
                future = object.__getattribute__(current, "_future")
                if future is not None:
                    if not future.done():
                        found.add(future)
                        return None
                    if future.cancelled():
                        return "dependency was cancelled"
                    try:
                        resolved = future.result()
                    except BaseException as exc:
                        return f"dependency failed: {format_exception(exc)}"
                    return visit(resolved, from_future=True)
                for nested in object.__getattribute__(current, "_data").values():
                    failure = visit(nested)
                    if failure is not None:
                        return failure
                return None
            if isinstance(current, dict):
                values = current.values()
            elif isinstance(current, (list, tuple)):
                values = current
            else:
                return None
            marker = id(current)
            if marker in seen:
                return None
            seen.add(marker)
            for nested in values:
                failure = visit(nested)
                if failure is not None:
                    return failure
            return None

        return visit(value)

    def _fail_dependency_cycles(self) -> None:
        owner = self._orchestrator
        graph = {
            request_id: set(request.producer_ids) & owner._requests.keys()
            for request_id, request in owner._requests.items()
            if request.state == "blocked"
        }
        index = 0
        stack: list[str] = []
        on_stack: set[str] = set()
        indices: dict[str, int] = {}
        lowlinks: dict[str, int] = {}
        cycles: list[set[str]] = []

        def connect(node: str) -> None:
            nonlocal index
            indices[node] = lowlinks[node] = index
            index += 1
            stack.append(node)
            on_stack.add(node)
            for target in graph.get(node, ()):
                if target not in graph:
                    continue
                if target not in indices:
                    connect(target)
                    lowlinks[node] = min(lowlinks[node], lowlinks[target])
                elif target in on_stack:
                    lowlinks[node] = min(lowlinks[node], indices[target])
            if lowlinks[node] == indices[node]:
                component: set[str] = set()
                while True:
                    member = stack.pop()
                    on_stack.remove(member)
                    component.add(member)
                    if member == node:
                        break
                if len(component) > 1 or node in graph.get(node, ()):
                    cycles.append(component)

        for node in graph:
            if node not in indices:
                connect(node)
        for cycle in cycles:
            message = "dependency cycle detected among requests: " + ", ".join(sorted(cycle))
            for request_id in list(cycle):
                request = owner._requests.get(request_id)
                if request is not None:
                    owner._fail_request_locked(request, message)


__all__ = ["ExecutionScheduler"]
