"""Canonical submission semantics implemented by the global orchestrator."""

from __future__ import annotations

import importlib
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


def test_prepared_head_blocks_later_run_without_creating_an_engine(monkeypatch, tmp_path):
    constructed: list[AgentEngine] = []
    sandbox_creations: list[tuple[tuple, dict]] = []
    execution_started = threading.Event()
    order: list[str] = []
    original_init = AgentEngine.__init__

    agent_module = importlib.import_module("agency.agent")

    def tracked_init(self, *args, **kwargs):
        constructed.append(self)
        original_init(self, *args, **kwargs)

    def execute(self, *, skill_input, **_kwargs):
        order.append(skill_input.label)
        execution_started.set()
        return agdata(label=skill_input.label)

    fake_sandbox = MagicMock()
    fake_sandbox._lock = threading.RLock()
    fake_sandbox._checkpoint_image = None

    def tracked_sandbox(*args, **kwargs):
        sandbox_creations.append((args, kwargs))
        return fake_sandbox

    monkeypatch.setattr(AgentEngine, "__init__", tracked_init)
    monkeypatch.setattr(AgentEngine, "execute", execute)
    config = agConfig(
        agOrchestratorConfig(max_concurrent_engines=1),
        {"agent": {"log_dir": str(tmp_path)}},
        {"agllm_backend": {"api_key": "test", "model": ""}},
    )
    ag = agent(agconfig=config)
    orchestrator = get_orchestrator(ag.agconfig)
    assert ag.sandbox is None
    monkeypatch.setattr(agent_module, "agSandbox", tracked_sandbox)
    skill = agskill("ordered", "")

    prepared = ag.prepare(skill, agdata(label="prepared"))
    later = ag.run(skill, agdata(label="later"))

    assert prepared.state == "PREPARED"
    assert not execution_started.is_set()
    assert constructed == []
    assert sandbox_creations == []
    assert not orchestrator._execution_workers._threads
    assert ag.sandbox is None
    assert ag.engine is None
    assert orchestrator.snapshot().running_count == 0

    prepared.start()
    assert prepared.wait(timeout=2).label == "prepared"
    assert later.wait(timeout=2).label == "later"
    assert order == ["prepared", "later"]
    assert len(constructed) == 2
    assert constructed[0] is not constructed[1]
    assert len(sandbox_creations) == 1
    assert len(orchestrator._execution_workers._threads) == 1


def test_invocation_start_never_bypasses_an_earlier_prepared_head(monkeypatch, tmp_path):
    execution_started = threading.Event()
    order: list[str] = []

    def execute(self, *, skill_input, **_kwargs):
        order.append(skill_input.label)
        execution_started.set()
        return agdata(label=skill_input.label)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    ag = _agent(tmp_path)
    skill = agskill("ordered", "")

    first = ag.prepare(skill, agdata(label="first"))
    second = ag.prepare(skill, agdata(label="second"))
    third = ag.run(skill, agdata(label="third"))

    second.start()
    assert second.state == "QUEUED"
    assert first.state == "PREPARED"
    assert not execution_started.is_set()

    first.start()
    assert first.wait(timeout=2).label == "first"
    assert second.wait(timeout=2).label == "second"
    assert third.wait(timeout=2).label == "third"
    assert order == ["first", "second", "third"]


def test_agent_start_releases_only_its_preexisting_snapshot(monkeypatch, tmp_path):
    first_started = threading.Event()
    release_first = threading.Event()
    order: list[str] = []

    def execute(self, *, skill_input, **_kwargs):
        order.append(skill_input.label)
        if skill_input.label == "one":
            first_started.set()
            assert release_first.wait(timeout=2)
        return agdata(label=skill_input.label)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    ag = _agent(tmp_path)
    skill = agskill("snapshot", "")

    one = ag.prepare(skill, agdata(label="one"))
    two = ag.prepare(skill, agdata(label="two"))
    ag.start()
    assert first_started.wait(timeout=2)

    later = ag.prepare(skill, agdata(label="later"))
    release_first.set()
    assert one.wait(timeout=2).label == "one"
    assert two.wait(timeout=2).label == "two"
    assert later.state == "PREPARED"
    assert order == ["one", "two"]

    ag.start()
    assert later.wait(timeout=2).label == "later"
    assert order == ["one", "two", "later"]


def test_prepared_agent_uses_no_capacity_while_another_agent_progresses(monkeypatch, tmp_path):
    other_started = threading.Event()
    release_other = threading.Event()
    order: list[str] = []

    def execute(self, *, skill_input, **_kwargs):
        order.append(skill_input.label)
        if skill_input.label == "other":
            other_started.set()
            assert release_other.wait(timeout=2)
        return agdata(label=skill_input.label)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    blocked_agent = _agent(tmp_path, max_engines=1)
    other_agent = _agent(tmp_path, max_engines=1)
    skill = agskill("capacity", "")

    prepared = blocked_agent.prepare(skill, agdata(label="prepared"))
    behind = blocked_agent.run(skill, agdata(label="behind"))
    other = other_agent.run(skill, agdata(label="other"))

    assert other_started.wait(timeout=2)
    assert order == ["other"]
    assert blocked_agent.engine is None
    snapshot = get_orchestrator().snapshot()
    assert snapshot.running_count == 1
    assert prepared.state == "PREPARED"
    assert behind.is_pending()

    release_other.set()
    assert other.wait(timeout=2).label == "other"
    prepared.start()
    assert prepared.wait(timeout=2).label == "prepared"
    assert behind.wait(timeout=2).label == "behind"
    assert order == ["other", "prepared", "behind"]


def test_concurrent_run_prepare_and_send_publish_and_register_in_one_order(monkeypatch, tmp_path):
    execution_started = threading.Event()
    execution_order: list[str] = []

    def execute(self, *, skill_input, **_kwargs):
        execution_order.append(skill_input.label)
        execution_started.set()
        return agdata(label=skill_input.label)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    ag = _agent(tmp_path)
    skill = agskill("concurrent", "")
    gate = ag.prepare(skill, agdata(label="gate"))
    barrier = threading.Barrier(4)
    submissions: dict[str, Invocation | MessageSubmission] = {}
    failures: list[BaseException] = []

    def submit_run() -> None:
        try:
            barrier.wait(timeout=2)
            submissions["run"] = ag.run(skill, agdata(label="run"))
        except BaseException as exc:  # surface caller-thread failures in the test
            failures.append(exc)

    def submit_prepared() -> None:
        try:
            barrier.wait(timeout=2)
            submissions["prepared"] = ag.prepare(skill, agdata(label="prepared"))
        except BaseException as exc:  # surface caller-thread failures in the test
            failures.append(exc)

    def submit_message() -> None:
        try:
            barrier.wait(timeout=2)
            submissions["message"] = ag.send("concurrent message")
        except BaseException as exc:  # surface caller-thread failures in the test
            failures.append(exc)

    callers = [
        threading.Thread(target=submit_run),
        threading.Thread(target=submit_prepared),
        threading.Thread(target=submit_message),
    ]
    for caller in callers:
        caller.start()
    barrier.wait(timeout=2)
    for caller in callers:
        caller.join(timeout=2)
        assert not caller.is_alive()

    assert failures == []
    run = submissions["run"]
    prepared = submissions["prepared"]
    message = submissions["message"]
    assert isinstance(run, Invocation)
    assert isinstance(prepared, Invocation)
    assert isinstance(message, MessageSubmission)
    assert {run.ordering_id, prepared.ordering_id, message.ordering_id} == {2, 3, 4}
    ordered = sorted((run, prepared, message), key=lambda item: item.ordering_id)
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
            "message" if isinstance(submission, MessageSubmission) else "skill"
            for submission in ordered
        ]

    prepared.start()
    gate.start()
    assert gate.wait(timeout=2).label == "gate"
    assert run.wait(timeout=2).label == "run"
    assert prepared.wait(timeout=2).label == "prepared"
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
