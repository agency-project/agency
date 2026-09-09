"""Orchestrator integration coverage for the engine completion/commit fence."""

from __future__ import annotations

import threading


from agency import agdata, agerror, agent, agskill
from agency.configs.agconfig import agconfig, agentconfig, llmconfig
from agency.agcontext import agcontext
from agency.engine import AgentEngine
from agency.orchestrator import get_orchestrator


def _request_for(handle: agdata):
    orchestrator = get_orchestrator()
    future = object.__getattribute__(handle, "_future")
    with orchestrator._event_cond:
        request_id = orchestrator._future_producers[future]
        return orchestrator._requests[request_id]


class _TransactionSandbox:
    """Small sandbox double that exposes deterministic transaction boundaries."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._checkpoint_image = None
        self.events: list[str] = []

    def change_config(self, _agconfig) -> None:
        pass

    def commit(self) -> None:
        self.events.append("commit")

    def rm_container(self) -> None:
        self.events.append("discard")

    def _has_pending_background_work(self) -> bool:
        self.events.append("check_background_work")
        return False

    def stop(self) -> None:
        self.events.append("stop")


def _agent(tmp_path, sandbox: _TransactionSandbox) -> agent:
    config = agconfig(
        agentconfig(log_dir=str(tmp_path)),
        llmconfig(api_key="test", model="m"),
    )
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


def test_cancel_after_harness_success_wins_before_commit_and_discards(monkeypatch, tmp_path):
    """A safe-boundary cancel cannot leak harness success into commit."""
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
    request = _request_for(invocation)
    assert harness_succeeded.wait(timeout=2)

    try:
        ag.cancel(invocation)
    finally:
        release_harness_return.set()

    assert invocation.wait(timeout=2).to_dict() == {"error": "agent invocation cancelled"}
    assert request.cancelled is True
    assert sandbox.events == ["discard"]
    output_context = request.context_future.result(timeout=2)
    assert output_context.recent_transcript == [{"role": "user", "content": "committed transcript"}]
    assert output_context.harness_sessions == {"native": {"session_id": "committed-session"}}
    assert output_context.harness_message_cursors == {"native": 5}
    assert output_context.retained_messages == _committed_context().retained_messages
    assert all(
        entry.get("source") != "context_notice" for entry in output_context.retained_messages
    )


def test_ordinary_failure_discards_and_appends_exactly_one_context_notice(monkeypatch, tmp_path):
    sandbox = _TransactionSandbox()
    ag = _agent(tmp_path, sandbox)
    ag.context = _committed_context()

    def failed_harness(self, context, *_args, **_kwargs):
        _mutate_working_context(context)
        return agerror("ordinary harness failure")

    monkeypatch.setattr(AgentEngine, "_execute_harness", failed_harness)
    invocation = ag.run(agskill("ordinary-failure", ""), agdata())
    request = _request_for(invocation)

    assert invocation.wait(timeout=2).to_dict() == {"error": "ordinary harness failure"}
    assert sandbox.events == ["discard"]
    output_context = request.context_future.result(timeout=2)
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
