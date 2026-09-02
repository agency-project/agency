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

    def _notify_invocation_control(self, invocation: Invocation) -> None:
        self.control_changes.append(invocation)


async def _await(value):
    return await value


def test_public_submission_exports_and_agent_alias():
    assert Agent is agent
    assert issubclass(Invocation, Submission)
    assert issubclass(MessageSubmission, Submission)
    assert hasattr(Invocation, "send_message")
    assert not hasattr(Invocation, "steer")
    assert hasattr(agent, "queue_message")
    assert not hasattr(agent, "send")
    assert not hasattr(agent, "steer")
    assert not hasattr(agent, "cancel")
    assert not hasattr(agent, "pause")
    assert hasattr(agent, "suspend")
    assert hasattr(agent, "resume")
    assert hasattr(agent, "destroy")


def test_invocation_result_wait_await_and_field_proxy_support_literal_result_field():
    owner = _FakeAgent()
    invocation = Invocation(owner, 1, "answer")
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
    invocation = Invocation(owner, 1, "async")

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


def test_invocation_is_queued_immediately_and_control_state_is_idempotent():
    owner = _FakeAgent()
    invocation = Invocation(owner, 1, "queued")

    assert invocation.state == "QUEUED"
    assert not hasattr(type(invocation), "start")
    assert not hasattr(agent, "prepare")
    assert not hasattr(agent, "start")

    invocation.pause()
    assert invocation.is_pause_requested()
    invocation.resume()
    assert not invocation.is_pause_requested()
    invocation.cancel()
    invocation.cancel()
    assert invocation.state == "CANCELLING"
    assert owner.control_changes[-1] is invocation


def test_invocation_messages_are_fifo_replayed_and_rejected_after_final_answer():
    owner = _FakeAgent()
    invocation = Invocation(owner, 1, "message")
    invocation.send_message("first")
    invocation.send_message("second")

    first = invocation._checkpoint("tool-1", allow_messages=True, phase="tool")
    retry = invocation._checkpoint("tool-1", allow_messages=True, phase="tool")
    assert [entry.content for entry in first.invocation_messages] == ["first", "second"]
    assert retry.invocation_messages == first.invocation_messages

    invocation._note_model_result(has_tool_calls=False)
    with pytest.raises(RuntimeError, match="phase is closing"):
        invocation.send_message("too late")


def test_sending_a_message_does_not_resume_a_paused_invocation():
    owner = _FakeAgent()
    invocation = Invocation(owner, 1, "paused-message")

    invocation.pause()
    invocation.send_message("wait until explicitly resumed")

    assert invocation.is_pause_requested() is True


def test_interruptible_checkpoint_abort_wakes_pause_without_consuming_messages():
    control = AgentControl()
    invocation = control.begin_invocation("interruptible")
    abort_event = threading.Event()
    outcome = {}
    invocation.send_message("deliver after reconnect")
    invocation.pause()

    worker = threading.Thread(
        target=lambda: outcome.setdefault(
            "decision",
            invocation._checkpoint_interruptibly(
                "native:tool:1",
                allow_messages=True,
                phase="boundary",
                abort_event=abort_event,
            ),
        ),
        daemon=True,
    )
    worker.start()
    with control._condition:
        assert control._condition.wait_for(
            lambda: invocation._phase == "paused",
            timeout=2.0,
        )

    invocation._abort_checkpoint_wait(abort_event)
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert outcome == {"decision": None}
    assert invocation.is_pause_requested() is True
    assert control.is_paused_actual() is False

    invocation.resume()
    retry = invocation._checkpoint(
        "native:tool:1",
        allow_messages=True,
        phase="boundary",
    )
    assert [entry.content for entry in retry.invocation_messages] == ["deliver after reconnect"]


def test_completion_claim_remains_won_after_agent_and_invocation_close():
    control = AgentControl()
    invocation = control.begin_invocation("commit")

    assert invocation._claim_completion() is True
    assert control.destroy() is True
    assert invocation._claim_completion() is True

    control.finish_invocation(invocation)
    assert invocation._claim_completion() is True


@pytest.mark.parametrize("losing_transition", ["cancel", "destroy"])
def test_completion_claim_stays_lost_when_control_wins_first(losing_transition):
    control = AgentControl()
    invocation = control.begin_invocation("rollback")

    if losing_transition == "cancel":
        invocation.cancel()
    else:
        control.destroy()

    assert invocation._claim_completion() is False
    assert invocation._claim_completion() is False


def test_message_submission_is_pending_data_without_skill_controls():
    owner = _FakeAgent()
    receipt = MessageSubmission(owner, "remember")
    receipt._result_future.set_result(agdata(accepted=True))

    assert receipt.wait().accepted is True
    for control in ("start", "send_message", "pause", "resume", "cancel"):
        assert not hasattr(receipt, control)


def test_close_handle_is_reusable_waitable_and_awaitable():
    close = CloseHandle()
    assert close.done() is False
    close._settle()
    close._settle()
    assert close.wait() is close
    assert asyncio.run(_await(close)) is close
