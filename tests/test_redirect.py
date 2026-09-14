"""Execution-scoped redirects through the real public API and harness manager."""

import threading
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agency import agdata, agent, agskill
from agency.configs.agconfig import agconfig, agentconfig, llmconfig
from agency.engine.engine import AgentEngine
from agency.harness.daemon import HarnessManager
from agency.harness.protocol import HarnessAttemptRequest, HarnessAttemptResult, PromptPayload
from agency.engine.clients import HarnessInteractionClient
from fastapi.testclient import TestClient


@pytest.fixture
def runs(monkeypatch, tmp_path):
    manager = HarnessManager("/unused/sandbox", "/unused/host", "test", "claude_code")
    manager._harness_api = MagicMock()
    # Exercise HTTP serialization without opening a network socket.
    transport = TestClient(manager._interaction_server.build_app())
    client = HarnessInteractionClient.__new__(HarnessInteractionClient)
    client._client = transport
    started = {name: threading.Event() for name in ("A", "B", "C")}
    finish = {name: threading.Event() for name in started}
    terminal = {name: threading.Event() for name in started}
    received, contexts, control_handles = [], {}, {}
    delivery_entered, delivery_release = threading.Event(), threading.Event()
    scenario = SimpleNamespace(block_delivery=False, fail_delivery=False)

    def execute(self, *, context, skill_input, request_id, **kwargs):
        name = skill_input.name
        contexts[name] = list(context.retained_messages)
        self._sandbox_interaction_client = client
        self._services_closed = False
        self._request_id = request_id

        def attempt(request):
            handle = control_handles[name] = MagicMock()
            manager._register_control_handle(handle)

            def redirect(message):
                if scenario.block_delivery:
                    delivery_entered.set()
                    assert delivery_release.wait(5)
                if scenario.fail_delivery or terminal[name].is_set():
                    raise BrokenPipeError("target exited")
                received.append((name, message))
                return True

            manager._register_redirect(redirect)
            started[name].set()
            assert finish[name].wait(5)
            terminal[name].set()
            return HarnessAttemptResult(ok=True)

        manager._attempt_handler = attempt
        try:
            result = manager._dispatch_attempt(
                HarnessAttemptRequest(
                    prompt=PromptPayload("", name),
                    harness="claude_code",
                    attempt_token=request_id,
                    request_id=request_id,
                )
            )
        finally:
            with self._services_lock:
                self._sandbox_interaction_client = None
                self._services_closed = True
        assert result.ok
        return agdata(name=name)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    sandbox = MagicMock()
    ag = agent(
        sandbox=sandbox,
        agconfig=agconfig(
            agentconfig(log_dir=str(tmp_path)),
            llmconfig(api_key="test", model="m"),
        ),
    )

    skill = agskill("redirect-test", "")
    state = SimpleNamespace(
        ag=ag,
        skill=skill,
        manager=manager,
        started=started,
        finish=finish,
        terminal=terminal,
        received=received,
        contexts=contexts,
        controls=control_handles,
        delivery_entered=delivery_entered,
        delivery_release=delivery_release,
        scenario=scenario,
    )
    try:
        yield state
    finally:
        delivery_release.set()
        for event in finish.values():
            event.set()
        transport.close()


def submit(runs, name, **data):
    return runs.ag.run(runs.skill, agdata(name=name, **data))


def finish(runs, result, name):
    runs.finish[name].set()
    assert result.wait(timeout=5).name == name


def retained(runs, name):
    return [entry["content"] for entry in runs.contexts[name]]


def test_redirect_active_target_and_sequential_runs_keep_their_identity(runs):
    for name in ("A", "B", "C"):
        result = submit(runs, name)
        assert runs.started[name].wait(2)
        assert runs.ag.redirect(result, f"to {name}") is None
        finish(runs, result, name)
    assert runs.received == [("A", "to A"), ("B", "to B"), ("C", "to C")]
    assert all(retained(runs, name) == [] for name in ("A", "B", "C"))


def test_redirect_before_target_starts_queues_without_waiting(runs):
    dependency = Future()
    result = submit(runs, "A", dependency=agdata(_future=dependency))
    runs.ag.redirect(result, "future context")
    assert not runs.started["A"].is_set()
    dependency.set_result(agdata())
    finish(runs, result, "A")
    later = submit(runs, "B")
    finish(runs, later, "B")
    assert retained(runs, "A") == []
    assert retained(runs, "B") == ["future context"]
    assert runs.received == []


def test_redirect_after_resolved_target_finishes_queues(runs):
    result = submit(runs, "A")
    finish(runs, result, "A")
    assert result.to_dict() == {"name": "A"}  # metadata survives resolution, stays private
    runs.ag.redirect(result, "after A")
    later = submit(runs, "B")
    finish(runs, later, "B")
    assert runs.received == []
    assert retained(runs, "B") == ["after A"]


def test_late_redirect_for_a_cannot_interrupt_active_b(runs, monkeypatch):
    first = submit(runs, "A")
    finish(runs, first, "A")
    second = submit(runs, "B")
    assert runs.started["B"].wait(2)
    rpc = MagicMock(wraps=runs.manager.redirect)
    monkeypatch.setattr(runs.manager._interaction_server, "_redirect_handler", rpc)
    runs.ag.redirect(first, "late A")
    rpc.assert_not_called()
    assert runs.received == []
    assert runs.controls["B"].mock_calls == []
    finish(runs, second, "B")
    third = submit(runs, "C")
    finish(runs, third, "C")
    assert retained(runs, "B") == []
    assert retained(runs, "C") == ["late A"]


@pytest.mark.parametrize("target_exits", [False, True])
def test_redirect_completion_race_delivers_or_queues_once_and_never_reaches_next_run(
    runs, target_exits
):
    first = submit(runs, "A")
    assert runs.started["A"].wait(2)
    runs.scenario.block_delivery = True
    worker = threading.Thread(target=runs.ag.redirect, args=(first, "racing message"))
    worker.start()
    assert runs.delivery_entered.wait(2)
    second = submit(runs, "B")
    if target_exits:
        runs.finish["A"].set()
        assert runs.terminal["A"].wait(2)
    assert not runs.started["B"].is_set()
    runs.delivery_release.set()
    worker.join(3)
    assert not worker.is_alive()
    finish(runs, first, "A")
    finish(runs, second, "B")
    third = submit(runs, "C")
    finish(runs, third, "C")
    assert runs.received == ([] if target_exits else [("A", "racing message")])
    assert retained(runs, "C").count("racing message") == int(target_exits)
    assert len(runs.received) + retained(runs, "C").count("racing message") == 1
    assert runs.controls["B"].mock_calls == []


def test_unsupported_adapter_and_transport_failure_each_queue_once(runs, monkeypatch):
    result = submit(runs, "A")
    assert runs.started["A"].wait(2)
    runs.manager._redirect_handler = None
    runs.ag.redirect(result, "unsupported")
    monkeypatch.setattr(runs.manager, "_redirect_handler", MagicMock(side_effect=OSError("closed")))
    runs.ag.redirect(result, "failed")
    finish(runs, result, "A")
    later = submit(runs, "B")
    finish(runs, later, "B")
    assert retained(runs, "B") == ["unsupported", "failed"]


def test_redirect_rejects_unrelated_data_without_waiting(runs):
    unresolved = agdata(_future=Future())
    with pytest.raises(ValueError, match="skill result from this agent"):
        runs.ag.redirect(unresolved, "message")
    for bad in ("", "   ", "\n"):
        with pytest.raises(ValueError, match="non-empty"):
            runs.ag.redirect(unresolved, bad)
    with pytest.raises(TypeError, match="string"):
        runs.ag.redirect(unresolved, 123)


def test_daemon_rejects_a_delayed_rpc_for_a_while_b_is_active(runs):
    first = submit(runs, "A")
    finish(runs, first, "A")
    second = submit(runs, "B")
    assert runs.started["B"].wait(2)
    # Bypass the host's completed-request shortcut, simulating an RPC that was
    # already in transit when A finished. The daemon must independently fence it.
    assert (
        runs.manager.redirect(object.__getattribute__(first, "_execution_id"), "stale RPC") is False
    )
    assert runs.received == []
    assert runs.controls["B"].mock_calls == []
    finish(runs, second, "B")


def test_delayed_cancel_rpc_cannot_kill_a_successor(runs):
    first = submit(runs, "A")
    finish(runs, first, "A")
    second = submit(runs, "B")
    assert runs.started["B"].wait(2)
    with TestClient(runs.manager._interaction_server.build_app()) as client:
        response = client.post(
            "/control/cancel", json={"request_id": object.__getattribute__(first, "_execution_id")}
        )
    assert response.status_code == 200
    runs.controls["B"].kill.assert_not_called()
    finish(runs, second, "B")


def test_cancel_after_admission_before_process_registration_kills_only_that_attempt(runs):
    manager = runs.manager
    manager._current_request_id = "A"
    manager.control("cancel", request_id="A")
    handle = MagicMock()
    manager._register_control_handle(handle)
    handle.kill.assert_called_once_with()
    manager._clear_control_handle()
    result = submit(runs, "B")
    assert runs.started["B"].wait(2)
    runs.controls["B"].kill.assert_not_called()
    finish(runs, result, "B")
