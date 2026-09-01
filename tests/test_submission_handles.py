from __future__ import annotations

import asyncio
import threading

import pytest

from agency import (
    Agent,
    CloseHandle,
    Invocation,
    MessageSubmission,
    Submission,
    agdata,
    agent,
)
from agency._agent_control import AgentControl


class _FakeAgent:
    def __init__(self) -> None:
        self._control = AgentControl()
        self._submission_lock = threading.RLock()
        self.control_changes = []

    def _start_invocation(self, invocation: Invocation) -> None:
        invocation._release()

    def _notify_invocation_control(self, invocation: Invocation) -> None:
        self.control_changes.append(invocation)


async def _await(value):
    return await value


def test_public_submission_exports_and_agent_alias():
    assert Agent is agent
    assert issubclass(Invocation, Submission)
    assert issubclass(MessageSubmission, Submission)


def test_invocation_result_wait_await_and_field_proxy_support_literal_result_field():
    owner = _FakeAgent()
    invocation = Invocation(owner, 1, "answer", ready=True)
    output = agdata(result="literal", other=42)
    invocation._result_future.set_result(output)
    invocation._mark_terminal(output)

    assert invocation.wait() is invocation
    assert asyncio.run(_await(invocation)) is output
    assert invocation.other == 42
    assert invocation.result.result == "literal"
    assert invocation.to_dict() == {"result": "literal", "other": 42}
    assert invocation.state == "SUCCEEDED"


def test_cancelling_one_async_waiter_does_not_cancel_shared_invocation_future():
    owner = _FakeAgent()
    invocation = Invocation(owner, 1, "async", ready=True)

    async def scenario():
        waiter = asyncio.ensure_future(invocation)
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        invocation._result_future.set_result(agdata(answer="still running"))
        return await invocation

    assert asyncio.run(scenario()).answer == "still running"
    assert invocation._result_future.cancelled() is False


def test_invocation_prepare_start_and_control_state_are_idempotent():
    owner = _FakeAgent()
    invocation = Invocation(owner, 1, "prepared", ready=False)

    assert invocation.state == "PREPARED"
    invocation.start()
    invocation.start()
    assert invocation.state == "QUEUED"

    invocation.pause()
    assert invocation.is_pause_requested()
    invocation.resume()
    assert not invocation.is_pause_requested()
    invocation.cancel()
    invocation.cancel()
    assert invocation.state == "CANCELLING"
    assert owner.control_changes[-1] is invocation


def test_steering_is_fifo_replayed_per_boundary_and_rejected_after_final_answer():
    owner = _FakeAgent()
    invocation = Invocation(owner, 1, "steer", ready=True)
    invocation.steer("first")
    invocation.steer("second")

    first = invocation._checkpoint("tool-1", allow_steering=True, phase="tool")
    retry = invocation._checkpoint("tool-1", allow_steering=True, phase="tool")
    assert [entry.instructions for entry in first.steering] == ["first", "second"]
    assert retry.steering == first.steering

    invocation._note_model_result(has_tool_calls=False)
    with pytest.raises(RuntimeError, match="phase is closing"):
        invocation.steer("too late")


def test_message_submission_is_pending_data_without_skill_controls():
    owner = _FakeAgent()
    receipt = MessageSubmission(owner, "remember")
    receipt._result_future.set_result(agdata(accepted=True))

    assert receipt.wait().accepted is True
    for control in ("start", "steer", "pause", "resume", "cancel"):
        assert not hasattr(receipt, control)


def test_close_handle_is_reusable_waitable_and_awaitable():
    close = CloseHandle()
    assert close.done() is False
    close._settle()
    close._settle()
    assert close.wait() is close
    assert asyncio.run(_await(close)) is close
