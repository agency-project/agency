"""Regression coverage for orchestrator lifecycle failure boundaries."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from agency import agdata, agerror, agent, agskill
from agency.configs.agconfig import agconfig, agentconfig, llmconfig, orchestratorconfig
from agency.agcontext import agcontext
from agency.engine import AgentEngine
from agency.orchestrator import get_orchestrator


def _agent(tmp_path, *, max_engines: int | None = None) -> agent:
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
    orchestrator = get_orchestrator()
    future = object.__getattribute__(handle, "_future")
    with orchestrator._event_cond:
        request_id = orchestrator._future_producers[future]
        return orchestrator._requests[request_id]


def test_agerror_rolls_output_context_back_to_committed_predecessor(monkeypatch, tmp_path):
    seed = [{"role": "user", "content": "committed"}]

    def execute(self, *, context, **_kwargs):
        context.recent_transcript.append({"role": "assistant", "content": "must be discarded"})
        context.harness_sessions["mutated"] = {"session_id": "must-be-discarded"}
        return agerror("ordinary skill failure")

    monkeypatch.setattr(AgentEngine, "execute", execute)
    ag = _agent(tmp_path)
    ag.context = agcontext(
        recent_transcript=seed,
        harness_sessions={"stable": {"session_id": "committed"}},
        retained_messages=[
            {
                "sequence": 7,
                "type": "message",
                "role": "user",
                "content": "stable retained context",
                "source": "test",
            }
        ],
    )

    invocation = ag.run(agskill("fails", ""), agdata())
    request = _request_for(invocation)

    assert invocation.wait(timeout=2).to_dict() == {"error": "ordinary skill failure"}
    output_context = request.context_future.result(timeout=2)
    assert output_context.recent_transcript == seed
    assert output_context.harness_sessions == {"stable": {"session_id": "committed"}}
    assert output_context.retained_messages == [
        {
            "sequence": 7,
            "type": "message",
            "role": "user",
            "content": "stable retained context",
            "source": "test",
        },
        {
            "sequence": 8,
            "type": "message",
            "role": "system",
            "content": (
                "Note: the previous skill call failed. Its sandbox workspace changes "
                "have been discarded and the workspace has been reverted to the last "
                "successful checkpoint."
            ),
            "source": "context_notice",
        },
    ]
    assert ag.history.messages == seed


def test_scheduler_fatal_event_settles_submit_and_closes_admission(monkeypatch, tmp_path):
    ag = _agent(tmp_path)
    orchestrator = get_orchestrator()
    real_execute = orchestrator.scheduler.execute
    calls = 0

    def fail_once() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("deterministic scheduler failure")
        real_execute()

    monkeypatch.setattr(orchestrator.scheduler, "execute", fail_once)
    first_finished = threading.Event()
    first_errors: list[BaseException] = []

    def first_submit() -> None:
        try:
            ag.run(agskill("first", ""), agdata())
        except BaseException as exc:
            first_errors.append(exc)
        finally:
            first_finished.set()

    first_caller = threading.Thread(target=first_submit, daemon=True)
    first_caller.start()
    assert first_finished.wait(timeout=2), "fatal scheduler event left submit() blocked"
    assert len(first_errors) == 1
    assert "deterministic scheduler failure" in str(first_errors[0])

    # Publication happened before the fatal scheduling cycle. Its context node
    # still has to settle even though submit() reports the scheduler exception.
    assert ag.context._future is not None
    ag.context._future.result(timeout=2)

    rejected: list[BaseException] = []

    def rejected_submit() -> None:
        try:
            ag.run(agskill("rejected", ""), agdata())
        except BaseException as exc:
            rejected.append(exc)

    second_caller = threading.Thread(target=rejected_submit, daemon=True)
    second_caller.start()
    second_caller.join(timeout=2)
    post_fatal_hung = second_caller.is_alive()
    if post_fatal_hung:
        # Unstick the legacy fatal path so this regression can fail without
        # leaving a caller or the autouse orchestrator teardown blocked.
        with orchestrator._event_cond:
            orchestrator._accepting = False
            for _sequence, kind, value in tuple(orchestrator._events):
                if kind == "schedule" and isinstance(value, tuple):
                    _request_id, cycle_ack = value
                    if cycle_ack is not None and not cycle_ack.done():
                        cycle_ack.set_exception(RuntimeError("test cleanup after scheduler death"))
            for request in list(orchestrator._requests.values()):
                if request.state != "running":
                    orchestrator._fail_request_locked(request, "test cleanup after scheduler death")
            orchestrator._state = "stopped"
            orchestrator._stop_loop = True
        second_caller.join(timeout=2)

    shutdown_errors: list[BaseException] = []

    def shutdown() -> None:
        try:
            orchestrator.shutdown(wait=True, timeout_s=2)
        except BaseException as exc:
            shutdown_errors.append(exc)

    shutdown_caller = threading.Thread(target=shutdown, daemon=True)
    shutdown_caller.start()
    shutdown_caller.join(timeout=3)
    assert post_fatal_hung is False, "post-fatal submission waited on a dead scheduler"
    assert len(rejected) == 1
    assert isinstance(rejected[0], RuntimeError)
    assert not shutdown_caller.is_alive(), "fatal scheduler state made shutdown hang"
    assert shutdown_errors == []


def test_shutdown_settles_request_with_unmanaged_predecessor(tmp_path):
    predecessor_future: Future[agcontext] = Future()
    ag = _agent(tmp_path)
    ag.context = agcontext(_future=predecessor_future)
    invocation = ag.run(agskill("blocked", ""), agdata())
    request = _request_for(invocation)
    orchestrator = get_orchestrator()
    callback_observed_context = threading.Event()
    object.__getattribute__(invocation, "_future").add_done_callback(
        lambda _future: (
            callback_observed_context.set()
            if request.context_future.done()
            else pytest.fail("shutdown published result before output context")
        )
    )
    shutdown_errors: list[BaseException] = []

    def shutdown() -> None:
        try:
            orchestrator.shutdown(wait=True, timeout_s=2)
        except BaseException as exc:
            shutdown_errors.append(exc)

    shutdown_caller = threading.Thread(target=shutdown, daemon=True)
    shutdown_caller.start()
    shutdown_caller.join(timeout=3)
    hung = shutdown_caller.is_alive()
    timed_out = hung or bool(shutdown_errors)
    if timed_out and not predecessor_future.done():
        # Keep the regression failure bounded on implementations which still
        # wait forever for an external producer during shutdown.
        predecessor_future.set_result(agcontext())
        shutdown_caller.join(timeout=2)

    assert timed_out is False, "shutdown waited indefinitely for an unmanaged predecessor"
    assert shutdown_errors == []
    assert predecessor_future.done() is False
    assert invocation.wait(timeout=2).error == (
        "orchestrator shut down before the request's dependencies resolved"
    )
    assert callback_observed_context.wait(timeout=2)


def test_shutdown_wait_from_result_callback_does_not_self_deadlock(monkeypatch, tmp_path):
    monkeypatch.setattr(
        AgentEngine,
        "execute",
        lambda self, **_kwargs: agdata(ok=True),
    )
    ag = _agent(tmp_path)
    orchestrator = get_orchestrator()
    invocation = ag.run(agskill("callback", ""), agdata())
    callback_finished = threading.Event()
    callback_errors: list[BaseException] = []

    def shutdown_from_callback(_future) -> None:
        try:
            orchestrator.shutdown(wait=True, timeout_s=0.5)
        except BaseException as exc:
            callback_errors.append(exc)
        finally:
            callback_finished.set()

    object.__getattribute__(invocation, "_future").add_done_callback(shutdown_from_callback)

    assert invocation.wait(timeout=2).ok is True
    assert callback_finished.wait(timeout=2)
    assert callback_errors == []


@dataclass(frozen=True)
class _DataclassEnvelope:
    payload: object


class _DumpModel:
    """Small model_dump/model_copy protocol object without Pydantic internals."""

    def __init__(self, payload: object) -> None:
        self.payload = payload

    def model_dump(self) -> dict[str, object]:
        return {"payload": self.payload}

    def model_copy(self, *, update: dict[str, object]):
        return type(self)(update.get("payload", self.payload))


def test_structured_dependencies_find_and_materialize_dataclass_and_dump_model(
    monkeypatch, tmp_path
):
    producer_started = threading.Event()
    release_producer = threading.Event()
    consumer_started = {
        "dataclass": threading.Event(),
        "model_dump": threading.Event(),
    }

    def execute(self, *, skill, skill_input, **_kwargs):
        if skill.name == "producer":
            producer_started.set()
            assert release_producer.wait(timeout=2)
            return agdata(answer=42)
        consumer_started[skill.name].set()
        return agdata(observed=skill_input.envelope.payload.answer)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    producer_agent = _agent(tmp_path)
    dataclass_agent = _agent(tmp_path)
    model_agent = _agent(tmp_path)

    producer = producer_agent.run(agskill("producer", ""), agdata())
    assert producer_started.wait(timeout=2)
    dataclass_consumer = dataclass_agent.run(
        agskill("dataclass", ""),
        agdata(envelope=_DataclassEnvelope(producer)),
    )
    model_consumer = model_agent.run(
        agskill("model_dump", ""),
        agdata(envelope=_DumpModel(producer)),
    )

    assert not consumer_started["dataclass"].wait(timeout=0.1)
    assert not consumer_started["model_dump"].wait(timeout=0.1)
    release_producer.set()

    assert producer.wait(timeout=2).answer == 42
    assert dataclass_consumer.wait(timeout=2).observed == 42
    assert model_consumer.wait(timeout=2).observed == 42

    direct = agdata(
        dataclass_value=_DataclassEnvelope(producer),
        model_value=_DumpModel(producer),
    )
    direct.resolve_input_dependencies()
    assert isinstance(direct.dataclass_value.payload, agdata)
    assert direct.dataclass_value.payload.answer == 42
    assert isinstance(direct.model_value.payload, agdata)
    assert direct.model_value.payload.answer == 42
