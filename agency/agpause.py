from __future__ import annotations
import threading
import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from concurrent.futures import Future
    from .agent import agent

# Thread-local pointer to "the agent whose worker thread this is", set for
# the lifetime of an agskill._task() run. Lets agdata/agcontext attribute a
# blocking future.result() call to the specific agent that's waiting, so a
# cross-agent dependency shows up as an observable state instead of just a
# thread parked with no explanation.
_worker_agent = threading.local()


def current_worker_agent() -> "agent | None":
    return getattr(_worker_agent, "value", None)


def set_current_worker_agent(ag: "agent | None") -> None:
    _worker_agent.value = ag


def tag_producer(future: "Future", producer: "agent") -> None:
    """Record which agent will eventually resolve *future*, so a downstream
    waiter can be attributed to a specific upstream agent instead of just
    looking blocked. Plain attribute on a stdlib Future — no subclassing
    needed since Future instances support arbitrary attributes."""
    future._producer_ref = weakref.ref(producer)  # type: ignore[attr-defined]


def producer_of(future: "Future") -> "agent | None":
    ref = getattr(future, "_producer_ref", None)
    return ref() if ref is not None else None


class _NullBlockCtx:
    def __enter__(self) -> "_NullBlockCtx":
        return self

    def __exit__(self, *exc) -> bool:
        return False


class _BlockCtx:
    """Marks the current worker agent as blocked_on_dependency on *producer*
    for the duration of a blocking future.result() call, restoring its prior
    display state afterward."""

    def __init__(self, waiter: "agent", producer: "agent") -> None:
        self.waiter = waiter
        self.producer = producer
        self._prev: "tuple[str, str | None, str | None]" = ("skill", None, None)

    def __enter__(self) -> "_BlockCtx":
        state = self.waiter._state
        with state._lock:
            self._prev = state.snapshot()
            state.blocked_on = self.producer
            state.update_state("blocked_on_dependency", skill=self._prev[1], tool=self._prev[2])
        return self

    def __exit__(self, *exc) -> bool:
        state = self.waiter._state
        with state._lock:
            state.blocked_on = None
            restore = self._prev[0]
            if restore in (None, "pausing", "paused", "blocked_on_dependency"):
                restore = "skill"
            state.update_state(restore, skill=self._prev[1], tool=self._prev[2])
        return False


def note_blocked_on(producer: "agent | None"):
    """Context manager wrapping a blocking future.result() call. No-op unless
    the calling thread is a known agent worker thread waiting on a *different*
    agent's future — same-agent chaining (an agent waiting on its own prior
    run) needs no extra tagging since the agent's own state already reflects
    that prior run's state."""
    waiter = current_worker_agent()
    if waiter is None or producer is None or producer is waiter:
        return _NullBlockCtx()
    return _BlockCtx(waiter, producer)
