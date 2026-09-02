"""Thread-safe lifecycle state shared by public invocations and execution layers.

The global orchestrator owns scheduling and engine construction.  This module
owns only the desired control state for one agent and its exact public
``Invocation`` objects.  Execution code observes that state at explicit safe
boundaries.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass


class AgentDestroyedError(RuntimeError):
    """Raised synchronously when an operation targets a closing agent."""


@dataclass(frozen=True)
class InvocationMessage:
    sequence: int
    content: str


@dataclass(frozen=True)
class InvocationDecision:
    cancelled: bool
    destroyed: bool
    invocation_messages: tuple[InvocationMessage, ...] = ()


@dataclass(frozen=True)
class InvocationFinish:
    decision: InvocationDecision
    sequence: int | None = None


class InvocationHandle:
    """Safe-boundary control state for exactly one invocation.

    The public :class:`~agency.Invocation` subclasses this type.  There is no
    second lifecycle object for the engine to drift away from: the scheduler,
    engine, and harness all receive this same object.
    """

    _MESSAGE_REJECT_PHASES = frozenset({"model", "closing"})
    _VALID_PHASES = frozenset(
        {"starting", "boundary", "infrastructure", "model", "tool", "paused", "closing"}
    )

    def __init__(self, control: "AgentControl", invocation_id: int, skill_name: str) -> None:
        self._control = control
        self.invocation_id = invocation_id
        self.skill_name = skill_name
        self._phase = "starting"
        self._phase_before_pause = "starting"
        self._pause_requested = False
        self._cancelled = False
        self._destroyed = False
        self._closed = False
        self._completion_claimed = False
        self._finish: InvocationFinish | None = None
        self._pending_messages: deque[InvocationMessage] = deque()
        self._boundary_messages: dict[str, tuple[InvocationMessage, ...]] = {}

    @property
    def phase(self) -> str:
        with self._control._condition:
            return self._phase

    def _checkpoint(
        self,
        boundary_id: str,
        *,
        allow_messages: bool,
        phase: str,
    ) -> InvocationDecision:
        """Observe pause/suspension gates and deliver invocation messages at a boundary.

        Messages assigned to a ``boundary_id`` are cached.  A retried request
        using the same boundary therefore sees the same overlay without
        consuming the queue a second time.
        """
        decision = self._checkpoint_wait(
            boundary_id,
            allow_messages=allow_messages,
            phase=phase,
            abort_event=None,
        )
        assert decision is not None
        return decision

    def _checkpoint_interruptibly(
        self,
        boundary_id: str,
        *,
        allow_messages: bool,
        phase: str,
        abort_event: threading.Event,
    ) -> "InvocationDecision | None":
        """Wait at a boundary until control admits it or its caller disconnects.

        ``None`` means the caller-owned operation was aborted.  In that case
        no invocation message is assigned, consumed, or cached for ``boundary_id``.
        The ordinary ``_checkpoint`` API remains non-interruptible and always
        returns an ``InvocationDecision``.
        """
        return self._checkpoint_wait(
            boundary_id,
            allow_messages=allow_messages,
            phase=phase,
            abort_event=abort_event,
        )

    def _checkpoint_wait(
        self,
        boundary_id: str,
        *,
        allow_messages: bool,
        phase: str,
        abort_event: "threading.Event | None",
    ) -> "InvocationDecision | None":
        if not isinstance(boundary_id, str) or not boundary_id:
            raise ValueError("boundary_id must be a non-empty string")
        requested_phase = self._normalize_phase(phase)

        with self._control._condition:
            if abort_event is not None and abort_event.is_set():
                return None
            if not self._closed and self._phase != "closing":
                self._phase = requested_phase
            while (
                not self._closed
                and self._phase != "closing"
                and (self._pause_requested or self._control._suspend_requested)
                and not (self._cancelled or self._destroyed)
            ):
                self._phase_before_pause = self._phase
                self._phase = "paused"
                if self._control._suspend_requested and not self._control._is_closing_unlocked():
                    self._control._lifecycle = "suspended"
                self._control._condition.notify_all()
                self._control._condition.wait()
                if self._phase == "paused" and not self._closed:
                    self._phase = self._phase_before_pause
                if abort_event is not None and abort_event.is_set():
                    # This waiter is no longer parked, so restore its prior
                    # phase while preserving the requested pause/suspension.
                    # A later retry will park at the same gate.
                    self._control._condition.notify_all()
                    return None

            if abort_event is not None and abort_event.is_set():
                return None

            effective_phase = self._phase
            assigned = self._boundary_messages.get(boundary_id)
            if assigned is None:
                can_assign = (
                    allow_messages
                    and effective_phase != "closing"
                    and not self._cancelled
                    and not self._destroyed
                )
                assigned = tuple(self._pending_messages) if can_assign else ()
                if can_assign:
                    self._pending_messages.clear()
                self._boundary_messages[boundary_id] = assigned

            return InvocationDecision(
                cancelled=self._cancelled,
                destroyed=self._destroyed,
                invocation_messages=assigned,
            )

    def _abort_checkpoint_wait(self, abort_event: threading.Event) -> None:
        """Abort one interruptible wait and wake it without timeout polling."""
        abort_event.set()
        with self._control._condition:
            self._control._condition.notify_all()

    def _note_model_result(self, has_tool_calls: bool) -> None:
        """Advance the model/tool phase and establish the final-answer fence."""
        with self._control._condition:
            if not self._closed and self._phase != "closing":
                self._phase = "tool" if has_tool_calls else "closing"
                if not has_tool_calls:
                    self._pause_requested = False
            self._control._condition.notify_all()

    def _claim_completion(self) -> bool:
        """Atomically win the cancellation/destruction-versus-commit race."""
        with self._control._condition:
            # The claim is a monotonic race result.  Once this invocation wins,
            # later lifecycle transitions cannot retroactively change that
            # answer for another completion-fence check.
            if self._completion_claimed:
                return True
            if self._cancelled or self._destroyed or self._control._is_closing_unlocked():
                return False
            self._completion_claimed = True
            self._phase = "closing"
            self._pause_requested = False
            self._control._condition.notify_all()
            return True

    def send_message(self, message: str) -> None:
        self._control._send_message(self, message)

    def pause(self) -> None:
        self._control._pause(self)

    def resume(self) -> None:
        self._control._resume(self)

    def cancel(self) -> None:
        self._control._cancel(self)

    def is_cancelled(self) -> bool:
        with self._control._condition:
            return self._cancelled

    def is_destroyed(self) -> bool:
        with self._control._condition:
            return self._destroyed

    def is_pause_requested(self) -> bool:
        with self._control._condition:
            return self._pause_requested

    @classmethod
    def _normalize_phase(cls, phase: str) -> str:
        value = str(getattr(phase, "value", phase)).lower()
        if value == "finalizing":
            value = "closing"
        if value not in cls._VALID_PHASES:
            raise ValueError(f"unknown invocation phase: {phase!r}")
        return value


class AgentControl:
    """Own one active invocation and the independent agent lifecycle gate."""

    def __init__(self, *, initial_sequence: int = 0) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._lifecycle = "active"
        self._suspend_requested = False
        self._active: InvocationHandle | None = None
        self._next_invocation_id = 1
        self._sequence = max(0, int(initial_sequence))

    def _is_closing_unlocked(self) -> bool:
        return self._lifecycle in {"destroying", "destroyed"}

    def assert_alive(self, operation: str = "operation") -> None:
        with self._condition:
            if self._is_closing_unlocked():
                raise AgentDestroyedError(f"cannot {operation}: agent is destroyed")

    def allocate_invocation_id(self) -> int:
        with self._condition:
            self.assert_alive("submit a skill")
            value = self._next_invocation_id
            self._next_invocation_id += 1
            return value

    def next_sequence(self, *, allow_destroyed: bool = False) -> int:
        with self._condition:
            if not allow_destroyed:
                self.assert_alive("record a message")
            self._sequence += 1
            return self._sequence

    def ensure_sequence_at_least(self, sequence: int) -> None:
        with self._condition:
            self._sequence = max(self._sequence, int(sequence))

    def begin_invocation(self, skill_name: str) -> InvocationHandle:
        handle = InvocationHandle(self, self.allocate_invocation_id(), skill_name)
        self.activate_invocation(handle)
        return handle

    def activate_invocation(self, handle: InvocationHandle) -> InvocationHandle:
        with self._condition:
            self.assert_alive("start a skill invocation")
            if handle._control is not self:
                raise RuntimeError("invocation handle belongs to another agent control")
            if self._active is not None:
                raise RuntimeError("agent already has an active skill invocation")
            if handle._cancelled or handle._destroyed:
                raise RuntimeError("cannot activate a cancelling invocation")
            self._active = handle
            if handle._phase != "closing":
                handle._phase = "starting"
            self._condition.notify_all()
            return handle

    def finish_invocation(
        self,
        handle: InvocationHandle,
        *,
        reserve_sequence: bool = False,
    ) -> InvocationFinish:
        with self._condition:
            if handle._control is not self:
                raise RuntimeError("invocation handle belongs to another agent control")
            if handle._finish is not None:
                return handle._finish
            if self._active is not handle:
                raise RuntimeError("cannot finish a non-active skill invocation")
            handle._phase = "closing"
            decision = InvocationDecision(handle._cancelled, handle._destroyed)
            sequence = None
            if reserve_sequence and not (decision.cancelled or decision.destroyed):
                self._sequence += 1
                sequence = self._sequence
            handle._closed = True
            handle._pause_requested = False
            self._active = None
            if self._suspend_requested and not self._is_closing_unlocked():
                self._lifecycle = "suspended"
            finish = InvocationFinish(decision=decision, sequence=sequence)
            handle._finish = finish
            self._condition.notify_all()
            return finish

    def end_invocation(self, handle: InvocationHandle) -> None:
        self.finish_invocation(handle)

    def active_invocation(self) -> InvocationHandle | None:
        with self._condition:
            return self._active

    def _send_message(self, handle: InvocationHandle, message: str) -> InvocationMessage:
        with self._condition:
            self.assert_alive("send a message")
            if not isinstance(message, str):
                raise TypeError("message must be a string")
            if not message.strip():
                raise ValueError("message must be a non-empty string")
            if handle._control is not self:
                raise RuntimeError("invocation handle belongs to another agent control")
            if handle._cancelled or handle._destroyed or handle._closed:
                raise RuntimeError("cannot send message: skill invocation is ending")
            if handle._phase in InvocationHandle._MESSAGE_REJECT_PHASES:
                raise RuntimeError(f"cannot send message while invocation phase is {handle._phase}")
            self._sequence += 1
            entry = InvocationMessage(self._sequence, message)
            handle._pending_messages.append(entry)
            self._condition.notify_all()
            return entry

    def _pause(self, handle: InvocationHandle) -> None:
        with self._condition:
            self.assert_alive("pause")
            if handle._cancelled or handle._destroyed or handle._closed:
                raise RuntimeError("cannot pause: skill invocation is ending")
            if handle._phase == "closing":
                raise RuntimeError("cannot pause: final-answer fence has been reached")
            handle._pause_requested = True
            self._condition.notify_all()

    def _resume(self, handle: InvocationHandle) -> None:
        with self._condition:
            self.assert_alive("resume an invocation")
            if handle._cancelled or handle._destroyed or handle._closed:
                raise RuntimeError("cannot resume: skill invocation is ending")
            if handle._phase == "closing":
                raise RuntimeError("cannot resume: final-answer fence has been reached")
            handle._pause_requested = False
            self._condition.notify_all()

    def _cancel(self, handle: InvocationHandle) -> None:
        with self._condition:
            if (
                handle._cancelled
                or handle._destroyed
                or handle._closed
                or handle._completion_claimed
            ):
                return
            self.assert_alive("cancel an invocation")
            handle._cancelled = True
            handle._pause_requested = False
            self._condition.notify_all()

    def suspend(self) -> None:
        with self._condition:
            self.assert_alive("suspend")
            if self._suspend_requested:
                return
            self._suspend_requested = True
            self._lifecycle = "suspending" if self._active is not None else "suspended"
            self._condition.notify_all()

    def resume_agent(self) -> None:
        with self._condition:
            self.assert_alive("resume")
            self._suspend_requested = False
            self._lifecycle = "active"
            self._condition.notify_all()

    def destroy(self) -> bool:
        """Close admission immediately, returning ``True`` only once."""
        with self._condition:
            if self._is_closing_unlocked():
                return False
            self._lifecycle = "destroying"
            self._suspend_requested = False
            handle = self._active
            if handle is not None and not (handle._closed or handle._completion_claimed):
                handle._destroyed = True
                handle._cancelled = True
                handle._pause_requested = False
            self._condition.notify_all()
            return True

    def request_destroy(self, handle: InvocationHandle) -> None:
        with self._condition:
            if handle._closed or handle._completion_claimed:
                return
            handle._destroyed = True
            handle._cancelled = True
            handle._pause_requested = False
            self._condition.notify_all()

    def mark_destroyed(self) -> None:
        with self._condition:
            self._lifecycle = "destroyed"
            self._condition.notify_all()

    def is_destroyed(self) -> bool:
        with self._condition:
            return self._is_closing_unlocked()

    def is_fully_destroyed(self) -> bool:
        with self._condition:
            return self._lifecycle == "destroyed"

    def lifecycle_state(self) -> str:
        with self._condition:
            return self._lifecycle

    def is_suspended(self) -> bool:
        with self._condition:
            return self._suspend_requested

    def is_pause_requested(self) -> bool:
        with self._condition:
            return bool(self._active and self._active._pause_requested)

    def is_paused_actual(self) -> bool:
        with self._condition:
            return bool(self._active and self._active._phase == "paused")


__all__ = [
    "AgentControl",
    "AgentDestroyedError",
    "InvocationDecision",
    "InvocationFinish",
    "InvocationHandle",
    "InvocationMessage",
]
