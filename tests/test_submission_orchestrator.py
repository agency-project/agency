"""Canonical submission semantics implemented by the global orchestrator."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from unittest.mock import MagicMock

from agency import Invocation, MessageSubmission, agdata, agent, agskill
from agency.agconfig import agConfig
from agency.engine import AgentEngine
from agency.orchestrator import agOrchestratorConfig, get_orchestrator


def _agent(tmp_path, *, max_engines=None) -> agent:
    config = agConfig(
        agOrchestratorConfig(max_concurrent_engines=max_engines),
        {"agent": {"log_dir": str(tmp_path)}},
        {"agllm_backend": {"api_key": "test", "model": ""}},
    )
    sandbox = MagicMock()
    sandbox._lock = threading.RLock()
    sandbox._checkpoint_image = None
    return agent(sandbox=sandbox, agconfig=config)


def test_agent_and_skill_run_return_the_exact_orchestrator_invocation(monkeypatch, tmp_path):
    gate: Future[agdata] = Future()

    def execute(self, *, skill_input, **_kwargs):
        return agdata(label=skill_input.label)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    skill = agskill("identity", "")
    first_agent = _agent(tmp_path)
    second_agent = _agent(tmp_path)

    by_agent = first_agent.run(
        skill,
        agdata(label="agent.run", gate=agdata(_future=gate)),
    )
    by_skill = skill.run(
        second_agent,
        agdata(label="agskill.run", gate=agdata(_future=gate)),
    )

    assert isinstance(by_agent, Invocation)
    assert isinstance(by_skill, Invocation)
    assert by_agent.result.is_pending()
    assert by_skill.result.is_pending()

    orchestrator = get_orchestrator()
    with orchestrator._event_cond:
        agent_request = orchestrator._requests[by_agent._request_id]
        skill_request = orchestrator._requests[by_skill._request_id]
        assert agent_request.submission is by_agent
        assert skill_request.submission is by_skill
        assert agent_request.result_future is by_agent._result_future
        assert skill_request.result_future is by_skill._result_future
        assert agent_request.context_future is by_agent._context_future
        assert skill_request.context_future is by_skill._context_future

    gate.set_result(agdata(open=True))
    assert by_agent.wait(timeout=2).label == "agent.run"
    assert by_skill.wait(timeout=2).label == "agskill.run"


def test_direct_and_nested_invocation_dependencies_materialize(monkeypatch, tmp_path):
    producer_started = threading.Event()
    release_producer = threading.Event()
    consumer_started = threading.Event()

    def execute(self, *, skill, skill_input, **_kwargs):
        if skill.name == "produce":
            producer_started.set()
            assert release_producer.wait(timeout=2)
            return agdata(answer=42)
        consumer_started.set()
        if skill.name == "direct":
            return agdata(observed=skill_input.answer)
        return agdata(observed=skill_input.payload["items"][0][0].answer)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    producer_agent = _agent(tmp_path)
    direct_agent = _agent(tmp_path)
    nested_agent = _agent(tmp_path)

    producer = producer_agent.run(agskill("produce", ""), agdata())
    assert producer_started.wait(timeout=2)
    direct = direct_agent.run(agskill("direct", ""), producer)
    nested = nested_agent.run(
        agskill("nested", ""),
        agdata(payload={"items": [(producer,)]}),
    )

    assert not consumer_started.wait(timeout=0.1)
    release_producer.set()

    assert direct.wait(timeout=2).observed == 42
    assert nested.wait(timeout=2).observed == 42


def test_message_sent_before_dispatch_targets_exact_invocation_at_first_boundary(
    monkeypatch, tmp_path
):
    dependency: Future[agdata] = Future()
    observed = []

    def execute(self, *, invocation, **_kwargs):
        decision = invocation._checkpoint(
            "test:first-valid-boundary",
            allow_messages=True,
            phase="boundary",
        )
        observed.extend(entry.content for entry in decision.invocation_messages)
        return agdata(done=True)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    ag = _agent(tmp_path)
    invocation = ag.run(
        agskill("queued-message", ""),
        agdata(dependency=agdata(_future=dependency)),
    )
    context_head = ag.context

    invocation.redirect("deliver after dispatch")

    assert ag.context is context_head
    assert ag.engine is None
    dependency.set_result(agdata(open=True))
    assert invocation.wait(timeout=2).done is True
    assert observed == ["deliver after dispatch"]


def test_concurrent_run_and_send_publish_and_register_in_one_order(monkeypatch, tmp_path):
    execution_started = threading.Event()
    execution_order: list[str] = []
    gate_dependency: Future[agdata] = Future()

    def execute(self, *, skill_input, **_kwargs):
        execution_order.append(skill_input.label)
        execution_started.set()
        return agdata(label=skill_input.label)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    ag = _agent(tmp_path)
    skill = agskill("concurrent", "")
    gate = ag.run(skill, agdata(label="gate", dependency=agdata(_future=gate_dependency)))
    barrier = threading.Barrier(3)
    submissions: dict[str, Invocation | MessageSubmission] = {}
    failures: list[BaseException] = []

    def submit_run() -> None:
        try:
            barrier.wait(timeout=2)
            submissions["run"] = ag.run(skill, agdata(label="run"))
        except BaseException as exc:  # surface caller-thread failures in the test
            failures.append(exc)

    def submit_context_message() -> None:
        try:
            barrier.wait(timeout=2)
            submissions["message"] = ag.queue_message("concurrent message")
        except BaseException as exc:  # surface caller-thread failures in the test
            failures.append(exc)

    callers = [
        threading.Thread(target=submit_run),
        threading.Thread(target=submit_context_message),
    ]
    for caller in callers:
        caller.start()
    barrier.wait(timeout=2)
    for caller in callers:
        caller.join(timeout=2)
        assert not caller.is_alive()

    assert failures == []
    run = submissions["run"]
    message = submissions["message"]
    assert isinstance(run, Invocation)
    assert isinstance(message, MessageSubmission)
    assert {run.ordering_id, message.ordering_id} == {2, 3}
    ordered = sorted((run, message), key=lambda item: item.ordering_id)
    predecessor = gate.output_context
    for submission in ordered:
        assert submission.predecessor_context is predecessor
        predecessor = submission.output_context
    assert ag.context is ordered[-1].output_context
    assert not execution_started.is_set()

    orchestrator = get_orchestrator()
    with orchestrator._event_cond:
        requests = [orchestrator._requests[submission._request_id] for submission in ordered]
        assert [request.submission for request in requests] == ordered
        assert [request.sequence for request in requests] == sorted(
            request.sequence for request in requests
        )
        assert [request.kind for request in requests] == [
            "context_message" if isinstance(submission, MessageSubmission) else "skill"
            for submission in ordered
        ]

    gate_dependency.set_result(agdata(open=True))
    assert gate.wait(timeout=2).label == "gate"
    assert run.wait(timeout=2).label == "run"
    assert message.wait(timeout=2).to_dict() == {}
    expected = [
        "gate",
        *[submission.result.label for submission in ordered if isinstance(submission, Invocation)],
    ]
    assert execution_order == expected


def test_managed_dependency_cycle_through_invocations_fails_without_an_engine(
    monkeypatch, tmp_path
):
    execute = MagicMock()
    monkeypatch.setattr(AgentEngine, "execute", execute)
    forward: Future[object] = Future()
    first_agent = _agent(tmp_path)
    second_agent = _agent(tmp_path)
    skill = agskill("cycle", "")

    first = first_agent.run(
        skill,
        agdata(value=agdata(_future=forward)),
    )
    second = second_agent.run(skill, agdata(value=first))
    forward.set_result(second)

    assert "dependency cycle detected" in first.wait(timeout=2).error
    assert "dependency cycle detected" in second.wait(timeout=2).error
    execute.assert_not_called()
