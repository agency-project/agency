from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock

from agency.agconfig import agConfig
from agency.agcontext import agcontext
from agency.agdata import agdata, agerror
from agency.agent import agent
from agency.orchestrator import agOrchestratorConfig, get_orchestrator
from agency.orchestrator.scheduler import ExecutionScheduler
from agency.agskill import agskill
from agency.engine import AgentEngine


def _agent(tmp_path, *, max_engines=None):
    config = agConfig(
        agOrchestratorConfig(max_concurrent_engines=max_engines),
        {"agent": {"log_dir": str(tmp_path)}},
        {"agllm_backend": {"api_key": "test", "model": ""}},
    )
    sandbox = MagicMock()
    sandbox._lock = threading.RLock()
    return agent(sandbox=sandbox, agconfig=config)


def _result(context, **output):
    return agdata(**output)


def test_orchestrator_package_layout(tmp_path):
    ag = _agent(tmp_path)
    orchestrator = get_orchestrator(ag.agconfig)

    assert isinstance(orchestrator.scheduler, ExecutionScheduler)


def test_unresolved_dependency_uses_no_engine_thread_or_slot(monkeypatch, tmp_path):
    called = threading.Event()

    def execute(self, *, context, **_kwargs):
        called.set()
        return _result(context, ok=True)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    upstream: Future[agdata] = Future()
    ag = _agent(tmp_path)
    pending = ag.run(agskill("s", ""), agdata(value=agdata(_future=upstream)))

    assert not called.wait(0.1)
    snapshot = get_orchestrator().snapshot()
    assert snapshot.blocked_count == 1
    assert snapshot.running_count == 0
    assert ag._current_state == "waiting_on_dependency"
    assert ag.engine is None

    upstream.set_result(agdata(value=42))
    assert pending.ok is True
    assert called.is_set()
    assert isinstance(ag.engine, AgentEngine)


def test_optional_global_capacity_refills_on_completion_notification(monkeypatch, tmp_path):
    release_first = threading.Event()
    first_started = threading.Event()
    second_started = threading.Event()
    lock = threading.Lock()
    calls = 0

    def execute(self, *, context, **_kwargs):
        nonlocal calls
        with lock:
            calls += 1
            call = calls
        if call == 1:
            first_started.set()
            assert release_first.wait(2)
        else:
            second_started.set()
        return _result(context, call=call)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    first_agent = _agent(tmp_path, max_engines=1)
    second_agent = _agent(tmp_path, max_engines=1)
    skill = agskill("s", "")

    first = first_agent.run(skill, agdata())
    second = second_agent.run(skill, agdata())
    assert first_started.wait(1)
    assert not second_started.wait(0.1)
    snapshot = get_orchestrator().snapshot()
    assert (snapshot.running_count, snapshot.ready_count) == (1, 1)
    assert second_agent._current_state == "queued"
    assert second_agent.engine is None

    release_first.set()
    assert second_started.wait(1)
    assert isinstance(second_agent.engine, AgentEngine)
    assert {first.call, second.call} == {1, 2}


def test_completion_cycle_promotes_dependencies_before_scheduling(monkeypatch, tmp_path):
    release_producer = threading.Event()
    producer_started = threading.Event()
    order: list[str] = []

    def execute(self, *, context, skill_input, **_kwargs):
        label = skill_input.label
        order.append(label)
        if label == "producer":
            producer_started.set()
            assert release_producer.wait(2)
        return _result(context, label=label)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    producer_agent = _agent(tmp_path, max_engines=1)
    dependent_agent = _agent(tmp_path, max_engines=1)
    independent_agent = _agent(tmp_path, max_engines=1)
    skill = agskill("s", "")

    producer = producer_agent.run(skill, agdata(label="producer"))
    assert producer_started.wait(1)
    dependent = dependent_agent.run(
        skill,
        agdata(label="dependent", dependency=producer),
    )
    independent = independent_agent.run(skill, agdata(label="independent"))
    snapshot = get_orchestrator().snapshot()
    assert (snapshot.running_count, snapshot.blocked_count, snapshot.ready_count) == (1, 1, 1)

    release_producer.set()

    assert dependent.label == "dependent"
    assert independent.label == "independent"
    assert order == ["producer", "dependent", "independent"]


def test_same_agent_requests_never_reorder_even_when_earlier_is_blocked(monkeypatch, tmp_path):
    """A later, dependency-free request for the same agent must wait behind
    an earlier request that's still blocked on an external dependency --
    agent context (conversation history) is an implicit dependency between
    consecutive same-agent calls, so same-agent submission order is never
    reordered (only cross-agent ready work may run ahead of a blocked
    request)."""
    order: list[str] = []
    ready_started = threading.Event()

    def execute(self, *, context, skill_input, **_kwargs):
        label = skill_input.label
        order.append(label)
        if label == "ready":
            ready_started.set()
        context.recent_transcript = [
            *context.recent_transcript,
            {"role": "user", "content": label},
        ]
        return _result(context, label=label)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    dependency: Future[agdata] = Future()
    ag = _agent(tmp_path)
    skill = agskill("s", "")

    blocked = ag.run(
        skill,
        agdata(label="blocked", dependency=agdata(_future=dependency)),
    )
    ready = ag.run(skill, agdata(label="ready"))

    assert not ready_started.wait(0.2)

    dependency.set_result(agdata(value=1))
    assert blocked.label == "blocked"
    assert ready.label == "ready"
    assert order == ["blocked", "ready"]
    assert [message["content"] for message in ag.history.messages] == ["blocked", "ready"]


def test_dependency_error_fails_without_launching_engine(monkeypatch, tmp_path):
    execute = MagicMock()
    monkeypatch.setattr(AgentEngine, "execute", execute)
    dependency: Future[agdata] = Future()
    ag = _agent(tmp_path)
    result = ag.run(agskill("s", ""), agdata(value=agdata(_future=dependency)))

    dependency.set_result(agerror("upstream failed"))

    assert "upstream failed" in result.error
    execute.assert_not_called()
    assert get_orchestrator().snapshot().failed_total == 1


def test_nested_dict_list_and_tuple_dependencies_are_materialized(monkeypatch, tmp_path):
    seen = []

    def execute(self, *, context, skill_input, **_kwargs):
        seen.append(skill_input.nested["items"][0][0].answer)
        return _result(context, ok=True)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    dependency: Future[agdata] = Future()
    ag = _agent(tmp_path)
    pending = ag.run(
        agskill("s", ""),
        agdata(nested={"items": [(agdata(_future=dependency),)]}),
    )

    assert get_orchestrator().snapshot().blocked_count == 1
    dependency.set_result(agdata(answer=42))
    assert pending.ok is True
    assert seen == [42]


def test_cancelled_and_exceptional_dependencies_fail_without_engine(monkeypatch, tmp_path):
    execute = MagicMock()
    monkeypatch.setattr(AgentEngine, "execute", execute)
    first_future: Future[agdata] = Future()
    second_future: Future[agdata] = Future()
    first_agent = _agent(tmp_path)
    second_agent = _agent(tmp_path)
    skill = agskill("s", "")
    cancelled = first_agent.run(skill, agdata(value=agdata(_future=first_future)))
    exceptional = second_agent.run(skill, agdata(value=agdata(_future=second_future)))

    first_future.cancel()
    second_future.set_exception(ValueError("bad upstream"))

    assert "cancelled" in cancelled.error
    assert "bad upstream" in exceptional.error
    execute.assert_not_called()


def test_default_capacity_allows_different_agents_to_run_concurrently(monkeypatch, tmp_path):
    release = threading.Event()
    both_started = threading.Event()
    lock = threading.Lock()
    started = 0

    def execute(self, *, context, **_kwargs):
        nonlocal started
        with lock:
            started += 1
            if started == 2:
                both_started.set()
        assert release.wait(2)
        return _result(context, ok=True)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    first_agent = _agent(tmp_path)
    second_agent = _agent(tmp_path)
    skill = agskill("s", "")
    first = first_agent.run(skill, agdata())
    second = second_agent.run(skill, agdata())

    assert both_started.wait(1)
    assert get_orchestrator().snapshot().running_count == 2
    release.set()
    assert first.ok is True
    assert second.ok is True


def test_each_dispatched_request_gets_a_fresh_engine(monkeypatch, tmp_path):
    engines = []

    def execute(self, *, context, **_kwargs):
        engines.append(self)
        return _result(context, run=len(engines))

    monkeypatch.setattr(AgentEngine, "execute", execute)
    ag = _agent(tmp_path)
    skill = agskill("s", "")

    first = ag.run(skill, agdata())
    assert first.run == 1
    first_engine = ag.engine
    second = ag.run(skill, agdata())
    assert second.run == 2

    assert len(engines) == 2
    assert engines[0] is first_engine
    assert engines[1] is ag.engine
    assert engines[0] is not engines[1]


def test_engine_exception_preserves_context_and_releases_agent_slot(monkeypatch, tmp_path):
    ag = _agent(tmp_path)
    ag.context = agcontext(recent_transcript=[{"role": "user", "content": "committed"}])

    def fail(self, **_kwargs):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(AgentEngine, "execute", fail)
    failed = ag.run(agskill("fail", ""), agdata())
    assert "engine exploded" in failed.error
    assert ag.history.messages == [{"role": "user", "content": "committed"}]

    def succeed(self, *, context, **_kwargs):
        context.recent_transcript.append({"role": "assistant", "content": "recovered"})
        return _result(context, ok=True)

    monkeypatch.setattr(AgentEngine, "execute", succeed)
    assert ag.run(agskill("next", ""), agdata()).ok is True
    assert [message["content"] for message in ag.history.messages] == [
        "committed",
        "recovered",
    ]


def test_completion_callback_can_submit_followup_without_deadlocking(monkeypatch, tmp_path):
    def execute(self, *, context, skill_input, **_kwargs):
        return _result(context, value=skill_input.value)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    ag = _agent(tmp_path)
    skill = agskill("s", "")
    first = ag.run(skill, agdata(value=1))
    first_future = object.__getattribute__(first, "_future")
    callback_finished = threading.Event()
    followup = []

    def submit_followup(_future):
        followup.append(ag.run(skill, agdata(value=2)))
        callback_finished.set()

    first_future.add_done_callback(submit_followup)

    assert first.value == 1
    assert callback_finished.wait(1)
    assert followup[0].value == 2


def test_completion_callback_can_read_committed_history(monkeypatch, tmp_path):
    def execute(self, *, context, **_kwargs):
        context.recent_transcript.append({"role": "assistant", "content": "committed"})
        return _result(context, ok=True)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    ag = _agent(tmp_path)
    result = ag.run(agskill("s", ""), agdata())
    result_future = object.__getattribute__(result, "_future")
    callback_finished = threading.Event()
    seen = []

    def read_history(_future):
        seen.extend(ag.history.messages)
        callback_finished.set()

    result_future.add_done_callback(read_history)

    assert result.ok is True
    assert callback_finished.wait(1)
    assert seen == [{"role": "assistant", "content": "committed"}]


def test_thread_start_failure_resolves_request_and_releases_capacity(monkeypatch, tmp_path):
    ag = _agent(tmp_path, max_engines=1)
    orchestrator = get_orchestrator(ag.agconfig)

    class BrokenThread:
        name = "broken"

        def start(self):
            raise RuntimeError("cannot start thread")

    monkeypatch.setattr(
        "agency.orchestrator.orchestrator.agprof.spawn_traced",
        lambda *_args, **_kwargs: BrokenThread(),
    )
    result = ag.run(agskill("s", ""), agdata())

    assert "cannot start thread" in result.error
    assert orchestrator.snapshot().running_count == 0
    assert orchestrator.snapshot().failed_total == 1


def test_managed_dependency_cycle_fails_all_members(monkeypatch, tmp_path):
    execute = MagicMock()
    monkeypatch.setattr(AgentEngine, "execute", execute)
    forward: Future[agdata] = Future()
    first_agent = _agent(tmp_path)
    second_agent = _agent(tmp_path)
    skill = agskill("s", "")

    first = first_agent.run(skill, agdata(value=agdata(_future=forward)))
    second = second_agent.run(skill, agdata(value=first))
    forward.set_result(second)

    assert "dependency cycle detected" in first.error
    assert "dependency cycle detected" in second.error
    execute.assert_not_called()


def test_shutdown_rejects_new_submissions(monkeypatch, tmp_path):
    def execute(self, *, context, **_kwargs):
        return _result(context, ok=True)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    ag = _agent(tmp_path)
    assert ag.run(agskill("s", ""), agdata()).ok is True
    orchestrator = get_orchestrator()
    orchestrator.shutdown(timeout_s=2)

    assert orchestrator.snapshot().state == "stopped"
    try:
        ag.run(agskill("later", ""), agdata())
    except RuntimeError as exc:
        assert "shut down" in str(exc)
    else:
        raise AssertionError("submission after shutdown should fail")


def test_shutdown_fails_permanently_blocked_requests(monkeypatch, tmp_path):
    monkeypatch.setattr(AgentEngine, "execute", MagicMock())
    dependency: Future[agdata] = Future()
    ag = _agent(tmp_path)
    result = ag.run(
        agskill("blocked", ""),
        agdata(value=agdata(_future=dependency)),
    )
    orchestrator = get_orchestrator()

    orchestrator.shutdown(timeout_s=2)

    assert "shut down" in result.error
    assert orchestrator.snapshot().state == "stopped"


def test_profiler_adapter_persists_intervals_to_agent_database(monkeypatch, tmp_path):
    dependency: Future[agdata] = Future()

    def execute(self, *, context, **_kwargs):
        return _result(context, ok=True)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    ag = _agent(tmp_path)
    result = ag.run(
        agskill("profiled", ""),
        agdata(value=agdata(_future=dependency)),
    )
    dependency.set_result(agdata(value=1))
    assert result.ok is True
    ag.data_collector.flush()

    connection = sqlite3.connect(ag.data_collector._configs.db_path)
    try:
        rows = connection.execute("SELECT name,attributes FROM spans ORDER BY id").fetchall()
    finally:
        connection.close()
    names = {row[0] for row in rows}
    assert {
        "sync:dependency_wait",
        "sync:scheduler_queue",
        "engine:execution",
        "request:submission_to_completion",
    } <= names
    correlated = [json.loads(attributes) for _name, attributes in rows]
    assert all(item["request_id"] == "run0" for item in correlated)
    assert all(item["skill"] == "profiled" for item in correlated)


def test_agents_keep_separate_data_collectors(tmp_path):
    first = _agent(tmp_path)
    second = _agent(tmp_path)

    first_path = Path(first.data_collector._configs.db_path)
    second_path = Path(second.data_collector._configs.db_path)
    assert first_path != second_path
    assert first_path.name == f"{first.agname}_data.sqlite3"
    assert second_path.name == f"{second.agname}_data.sqlite3"
    assert first_path.exists()
    assert second_path.exists()
