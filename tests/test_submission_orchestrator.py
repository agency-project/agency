"""Canonical submission semantics implemented by the global orchestrator."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from unittest.mock import MagicMock

from agency import agdata, agent, agskill
from agency.configs.agconfig import agconfig, agentconfig, llmconfig, orchestratorconfig
from agency.engine import AgentEngine
from agency.orchestrator import get_orchestrator


def _agent(tmp_path, *, max_engines=None) -> agent:
    config = agconfig(
        orchestratorconfig(max_concurrent_engines=max_engines),
        agentconfig(log_dir=str(tmp_path)),
        llmconfig(api_key="test", model="m"),
    )
    sandbox = MagicMock()
    sandbox._lock = threading.RLock()
    sandbox._checkpoint_image = None
    return agent(sandbox=sandbox, agconfig=config)


def _request_for(handle: agdata):
    """Look up the orchestrator's internal request behind a bare handle.

    The only correlation key a returned handle carries is its own result
    future -- everything else (ordering, context chaining) is orchestrator-
    internal now, reachable only via ``_future_producers``.
    """
    orchestrator = get_orchestrator()
    future = object.__getattribute__(handle, "_future")
    with orchestrator._event_cond:
        request_id = orchestrator._future_producers[future]
        return orchestrator._requests[request_id]


def test_agent_and_skill_run_return_bare_pending_agdata_bound_to_one_request(monkeypatch, tmp_path):
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

    assert type(by_agent) is agdata
    assert type(by_skill) is agdata
    assert by_agent.is_pending()
    assert by_skill.is_pending()

    agent_request = _request_for(by_agent)
    skill_request = _request_for(by_skill)
    assert agent_request.result_future is object.__getattribute__(by_agent, "_future")
    assert skill_request.result_future is object.__getattribute__(by_skill, "_future")
    assert agent_request.agent is first_agent
    assert skill_request.agent is second_agent

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
    submissions: dict[str, object] = {}
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
            ag.queue_message("concurrent message")
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
    assert not execution_started.is_set()
    run = submissions["run"]

    orchestrator = get_orchestrator()
    gate_request = _request_for(gate)
    run_request = _request_for(run)
    with orchestrator._event_cond:
        message_request = next(
            r
            for r in orchestrator._requests.values()
            if r.agent is ag and r.kind == "context_message"
        )
    # The context-dependency chain (not any field on a returned handle) is
    # the only thing that ever needed to prove concurrent submissions landed
    # in one order: each depends on its predecessor's context future.
    ordered = sorted([run_request, message_request], key=lambda r: r.sequence)
    assert ordered[0].context_dependency._future is gate_request.context_future
    assert ordered[1].context_dependency._future is ordered[0].context_future
    assert object.__getattribute__(ag.context, "_future") is ordered[-1].context_future

    gate_dependency.set_result(agdata(open=True))
    assert gate.wait(timeout=2).label == "gate"
    assert run.wait(timeout=2).label == "run"
    assert message_request.result_future.result(timeout=2).to_dict() == {}
    # Only the skill invocation ("run") ever reaches execute() -- the queued
    # message never does, regardless of which of the two landed first.
    assert execution_order == ["gate", "run"]


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


def test_materialization_preserves_cyclic_containers_and_resolves_their_dependencies():
    scheduler = get_orchestrator().scheduler
    future = Future()
    dependency = agdata(_future=future)
    items = []
    payload = {"items": items, "dependency": dependency}
    items.append(payload)
    found = set()
    assert scheduler._discover_dependencies(payload, found) is None
    assert found == {future}
    future.set_result(agdata(answer=42))
    resolved = scheduler.materialize_dependencies(payload)
    assert resolved["items"][0] is resolved
    assert resolved["dependency"].answer == 42
