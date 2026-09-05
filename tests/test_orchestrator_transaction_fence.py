"""Orchestrator integration coverage for the engine completion/commit fence."""

from __future__ import annotations

import threading

import pytest

from agency import Invocation, agdata, agerror, agent, agskill
from agency.configs.agconfig import agconfig
from agency.agcontext import agcontext
from agency.engine import AgentEngine


class _TransactionSandbox:
    """Small sandbox double that exposes deterministic transaction boundaries."""

    def __init__(self, *, block_commit: bool = False) -> None:
        self._lock = threading.RLock()
        self._checkpoint_image = None
        self._block_commit = block_commit
        self.commit_entered = threading.Event()
        self.release_commit = threading.Event()
        self.events: list[str] = []

    def change_config(self, _agconfig) -> None:
        pass

    def commit(self) -> None:
        self.events.append("commit")
        self.commit_entered.set()
        if self._block_commit:
            assert self.release_commit.wait(timeout=2)

    def rm_container(self) -> None:
        self.events.append("discard")

    def _has_pending_background_work(self) -> bool:
        self.events.append("check_background_work")
        return False

    def stop(self) -> None:
        self.events.append("stop")


def _agent(tmp_path, sandbox: _TransactionSandbox) -> agent:
    config = agconfig(log_dir=str(tmp_path), api_key="test", model="m")
    return agent(sandbox=sandbox, agconfig=config)


def _committed_context() -> agcontext:
    return agcontext(
        recent_transcript=[{"role": "user", "content": "committed transcript"}],
        harness_sessions={"native": {"session_id": "committed-session"}},
        retained_messages=[
            {
                "sequence": 5,
                "type": "message",
                "role": "user",
                "content": "committed retained context",
                "source": "test",
            }
        ],
        harness_message_cursors={"native": 5},
    )


def _mutate_working_context(context: agcontext) -> None:
    context.recent_transcript.append(
        {"role": "assistant", "content": "uncommitted harness transcript"}
    )
    context.harness_sessions["native"] = {"session_id": "uncommitted-session"}
    context.retained_messages.append(
        {
            "sequence": 6,
            "type": "message",
            "role": "system",
            "content": "uncommitted retained mutation",
            "source": "test-harness",
        }
    )


@pytest.mark.parametrize(
    ("control", "expected_error", "expected_state"),
    [
        ("cancel", "agent invocation cancelled", "CANCELLED"),
        ("destroy", "agent destroyed", "DESTROYED"),
    ],
)
def test_control_after_harness_success_wins_before_completion_claim_and_discards(
    monkeypatch,
    tmp_path,
    control,
    expected_error,
    expected_state,
):
    """A safe-boundary control winner cannot leak harness success into commit."""
    sandbox = _TransactionSandbox()
    ag = _agent(tmp_path, sandbox)
    ag.context = _committed_context()
    harness_succeeded = threading.Event()
    release_harness_return = threading.Event()

    def successful_harness(self, context, *_args, **_kwargs):
        _mutate_working_context(context)
        harness_succeeded.set()
        assert release_harness_return.wait(timeout=2)
        return agdata(ok=True)

    monkeypatch.setattr(AgentEngine, "_execute_harness", successful_harness)
    invocation = ag.run(agskill("controlled-transaction", ""), agdata())
    assert harness_succeeded.wait(timeout=2)

    close = None
    try:
        if control == "cancel":
            invocation.cancel()
        else:
            close = ag.destroy()
    finally:
        release_harness_return.set()

    assert invocation.wait(timeout=2).to_dict() == {"error": expected_error}
    assert invocation.state == expected_state
    assert sandbox.events == ["discard"]
    output_context = invocation._context_future.result(timeout=2)
    assert output_context.recent_transcript == [{"role": "user", "content": "committed transcript"}]
    assert output_context.harness_sessions == {"native": {"session_id": "committed-session"}}
    assert output_context.harness_message_cursors == {"native": 5}
    assert output_context.retained_messages == _committed_context().retained_messages
    assert all(
        entry.get("source") != "context_notice" for entry in output_context.retained_messages
    )
    if close is not None:
        assert close.wait(timeout=2).done() is True


def test_destroy_after_engine_claim_during_commit_preserves_success_on_second_claim(
    monkeypatch, tmp_path
):
    """The orchestrator's second claim is idempotent after engine commit wins."""
    sandbox = _TransactionSandbox(block_commit=True)
    ag = _agent(tmp_path, sandbox)
    ag.context = _committed_context()
    claims: list[bool] = []
    original_claim = Invocation._claim_completion

    def record_claim(invocation: Invocation) -> bool:
        claimed = original_claim(invocation)
        claims.append(claimed)
        return claimed

    def successful_harness(self, context, *_args, **_kwargs):
        _mutate_working_context(context)
        return agdata(ok=True)

    monkeypatch.setattr(Invocation, "_claim_completion", record_claim)
    monkeypatch.setattr(AgentEngine, "_execute_harness", successful_harness)
    invocation = ag.run(agskill("commit-winner", ""), agdata())

    assert sandbox.commit_entered.wait(timeout=2)
    assert claims == [True]
    close = ag.destroy()
    assert close.done() is False
    assert invocation.is_destroyed() is False
    sandbox.release_commit.set()

    assert invocation.wait(timeout=2).to_dict() == {"ok": True}
    assert invocation.state == "SUCCEEDED"
    assert claims == [True, True]
    assert sandbox.events == ["commit", "check_background_work", "stop"]
    output_context = invocation._context_future.result(timeout=2)
    assert output_context.recent_transcript[-1] == {
        "role": "assistant",
        "content": "uncommitted harness transcript",
    }
    assert output_context.harness_sessions == {"native": {"session_id": "uncommitted-session"}}
    assert output_context.retained_messages[-1]["source"] == "test-harness"
    assert close.wait(timeout=2).done() is True


def test_ordinary_failure_discards_and_appends_exactly_one_context_notice(monkeypatch, tmp_path):
    sandbox = _TransactionSandbox()
    ag = _agent(tmp_path, sandbox)
    ag.context = _committed_context()

    def failed_harness(self, context, *_args, **_kwargs):
        _mutate_working_context(context)
        return agerror("ordinary harness failure")

    monkeypatch.setattr(AgentEngine, "_execute_harness", failed_harness)
    invocation = ag.run(agskill("ordinary-failure", ""), agdata())

    assert invocation.wait(timeout=2).to_dict() == {"error": "ordinary harness failure"}
    assert invocation.state == "FAILED"
    assert sandbox.events == ["discard"]
    output_context = invocation._context_future.result(timeout=2)
    assert output_context.recent_transcript == [{"role": "user", "content": "committed transcript"}]
    assert output_context.harness_sessions == {"native": {"session_id": "committed-session"}}
    assert output_context.harness_message_cursors == {"native": 5}
    assert len(output_context.retained_messages) == 2
    assert output_context.retained_messages[0] == _committed_context().retained_messages[0]
    notice = output_context.retained_messages[1]
    assert notice["sequence"] == 6
    assert notice["type"] == "message"
    assert notice["role"] == "system"
    assert notice["source"] == "context_notice"
    assert "previous skill call failed" in notice["content"]
    assert "workspace changes have been discarded" in notice["content"]
