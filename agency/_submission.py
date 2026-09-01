"""Public handles and ordered nodes for an agent's context-future chain."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, InvalidStateError
from typing import TYPE_CHECKING

from ._agent_control import InvocationHandle
from .agcontext import agcontext
from .agdata import agdata, agerror

if TYPE_CHECKING:
    from .agent import agent


TERMINAL_INVOCATION_STATES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED", "DESTROYED"})


async def _await_concurrent_future(future: Future):
    """Await without letting cancellation of one waiter cancel shared work."""
    return await asyncio.shield(asyncio.wrap_future(future))


class Submission:
    """One ordered node in an agent's authoritative context chain."""

    def __init__(self, ag: "agent") -> None:
        self._agent = ag
        self.ordering_id = 0
        self.predecessor_context: agcontext | None = None
        self._context_future: Future[agcontext] = Future()
        self.output_context = agcontext(_future=self._context_future)
        self._request_id: str | None = None

    def _bind(self, ordering_id: int, predecessor: agcontext) -> None:
        self.ordering_id = ordering_id
        self.predecessor_context = predecessor

    def _bind_request(self, request_id: str) -> None:
        self._request_id = request_id


class _AgdataResultHandle:
    """Compatibility surface shared by Invocation and MessageSubmission."""

    result: agdata
    _result_future: Future[agdata]

    def _as_pending_agdata(self) -> agdata:
        return self.result

    def _resolve(self) -> None:
        self.result._resolve()

    def wait(self, timeout: float | None = None):
        if timeout is None:
            self.result.wait()
        else:
            self._result_future.result(timeout=timeout)
            self.result.wait()
        return self

    def is_pending(self) -> bool:
        return self.result.is_pending()

    def to_dict(self) -> dict:
        return self.result.to_dict()

    def to_json(self) -> str:
        return self.result.to_json()

    @property
    def _future(self):
        return self._result_future

    @property
    def _data(self):
        self.result._resolve()
        return self.result._data

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.result, name)

    def __await__(self):
        return self._await_result().__await__()

    async def _await_result(self) -> agdata:
        resolved = await _await_concurrent_future(self._result_future)
        resolved.wait()
        return resolved


class Invocation(Submission, InvocationHandle, _AgdataResultHandle):
    """Awaitable identity, result dependency, and controls for one skill run."""

    def __init__(self, ag: "agent", invocation_id: int, skill_name: str, *, ready: bool) -> None:
        Submission.__init__(self, ag)
        InvocationHandle.__init__(self, ag._control, invocation_id, skill_name)
        self._result_future: Future[agdata] = Future()
        self.result = agdata(_future=self._result_future)
        self._state = "QUEUED" if ready else "PREPARED"
        self._ready = ready

    @property
    def state(self) -> str:
        with self._control._condition:
            if self._phase == "paused" and self._state == "RUNNING":
                return "PAUSED"
            return self._state

    def start(self) -> None:
        """Release this invocation if prepared, without changing its order."""
        self._agent._start_invocation(self)

    def pause(self) -> None:
        super().pause()
        self._notify_control_change()

    def resume(self) -> None:
        super().resume()
        self._notify_control_change()

    def cancel(self) -> None:
        with self._control._condition:
            if self._state in TERMINAL_INVOCATION_STATES:
                return
        super().cancel()
        with self._control._condition:
            if self._cancelled and not self._closed:
                self._state = "CANCELLING"
        self._notify_control_change()

    def _notify_control_change(self) -> None:
        notify = getattr(self._agent, "_notify_invocation_control", None)
        if callable(notify):
            notify(self)

    def _release(self) -> bool:
        with self._control._condition:
            if self._state != "PREPARED":
                return False
            self._state = "QUEUED"
            self._ready = True
            self._control._condition.notify_all()
            return True

    def _mark_running(self) -> None:
        with self._control._condition:
            self._state = "RUNNING"

    def _close_without_activation(self) -> None:
        with self._control._condition:
            self._closed = True
            self._phase = "closing"
            self._pause_requested = False
            self._control._condition.notify_all()

    def _request_destroy(self) -> None:
        self._control.request_destroy(self)
        with self._control._condition:
            if self._destroyed and not self._closed:
                self._state = "CANCELLING"
        self._notify_control_change()

    def _mark_terminal(self, result: agdata) -> None:
        with self._control._condition:
            if self._destroyed:
                self._state = "DESTROYED"
            elif self._cancelled:
                self._state = "CANCELLED"
            elif isinstance(result, agerror) or bool(result._data.get("error")):
                self._state = "FAILED"
            else:
                self._state = "SUCCEEDED"

    def __repr__(self) -> str:
        return f"Invocation(id={self.ordering_id}, skill={self.skill_name!r}, state={self.state})"


class MessageSubmission(Submission, _AgdataResultHandle):
    """Awaitable receipt for one ordered, host-only retained message."""

    def __init__(self, ag: "agent", message: str) -> None:
        super().__init__(ag)
        self.message = message
        self._result_future: Future[agdata] = Future()
        self.result = agdata(_future=self._result_future)
        self._state = "QUEUED"
        self._commit_claimed = False
        self._destroy_requested = False

    @property
    def state(self) -> str:
        lock = getattr(self._agent, "_submission_lock", None)
        if lock is None:
            return self._state
        with lock:
            return self._state

    def _request_destroy(self) -> None:
        if not self._commit_claimed:
            self._destroy_requested = True
            self._state = "CANCELLING"

    def __getattr__(self, name: str):
        if name in {"start", "steer", "pause", "resume", "cancel"}:
            raise AttributeError(f"MessageSubmission has no {name}() control")
        return super().__getattr__(name)

    def __repr__(self) -> str:
        return f"MessageSubmission(id={self.ordering_id}, state={self.state})"


class CloseHandle:
    """Reusable awaitable returned by :meth:`agent.destroy`."""

    def __init__(self) -> None:
        self._future: Future[None] = Future()

    def done(self) -> bool:
        return self._future.done()

    def wait(self, timeout: float | None = None) -> "CloseHandle":
        self._future.result(timeout=timeout)
        return self

    def __await__(self):
        return self._await_close().__await__()

    async def _await_close(self) -> "CloseHandle":
        await _await_concurrent_future(self._future)
        return self

    def _settle(self) -> None:
        try:
            self._future.set_result(None)
        except InvalidStateError:
            # Concurrent cleanup paths are deliberately idempotent.  Future's
            # done()/set_result() pair is not itself atomic, so losing this
            # benign race must not leak out of destruction.
            pass

    def __repr__(self) -> str:
        return f"CloseHandle(done={self.done()})"


__all__ = [
    "CloseHandle",
    "Invocation",
    "MessageSubmission",
    "Submission",
    "TERMINAL_INVOCATION_STATES",
]
