"""Deterministic lifecycle regressions for orchestrator-owned Invocations."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest

from agency import (
    AgentDestroyedError,
    CloseHandle,
    MessageSubmission,
    agdata,
    agerror,
    agent,
    agskill,
)
from agency.configs.agconfig import agconfig
from agency.agcontext import agcontext
from agency.engine import AgentEngine
from agency.orchestrator import get_orchestrator


def _agent(tmp_path, *, max_engines: int | None = None) -> agent:
    config = agconfig(
        max_concurrent_engines=max_engines,
        log_dir=str(tmp_path),
        api_key="test",
        model="m",
    )
    sandbox = MagicMock()
    sandbox._lock = threading.RLock()
    sandbox._checkpoint_image = None
    return agent(sandbox=sandbox, agconfig=config)


def _request_state(invocation) -> str:
    orchestrator = get_orchestrator()
    with orchestrator._event_cond:
        return orchestrator._requests[invocation._request_id].state


def _drain_scheduler() -> None:
    """Wait until every event already posted to the scheduler has run."""
    orchestrator = get_orchestrator()
    acknowledged: Future[None] = Future()
    with orchestrator._event_cond:
        orchestrator._post_locked("schedule", (None, acknowledged))
    acknowledged.result(timeout=2)


def _terminal_error(invocation, expected: str) -> None:
    assert invocation.wait(timeout=2).to_dict() == {"error": expected}


def test_cancel_blocked_and_ready_without_building_engines(monkeypatch, tmp_path):
    capacity_started = threading.Event()
    release_capacity = threading.Event()
    callback_finished = {name: threading.Event() for name in ("seeded_ready", "blocked", "ready")}
    constructed_for: list[agent] = []
    executed: list[str] = []
    followups = []
    callback_contexts: dict[str, list[dict]] = {}
    original_init = AgentEngine.__init__

    def tracked_init(self, owner):
        constructed_for.append(owner)
        original_init(self, owner)

    def execute(self, *, skill_input, invocation, **_kwargs):
        assert invocation._agent is self._agent
        executed.append(skill_input.label)
        if skill_input.label == "holds-capacity":
            capacity_started.set()
            assert release_capacity.wait(timeout=2)
        return agdata(label=skill_input.label)

    monkeypatch.setattr(AgentEngine, "__init__", tracked_init)
    monkeypatch.setattr(AgentEngine, "execute", execute)

    capacity_agent = _agent(tmp_path, max_engines=1)
    seeded_ready_agent = _agent(tmp_path, max_engines=1)
    blocked_agent = _agent(tmp_path, max_engines=1)
    ready_agent = _agent(tmp_path, max_engines=1)
    capacity = capacity_agent.run(agskill("capacity", ""), agdata(label="holds-capacity"))
    assert capacity_started.wait(timeout=2)

    seed = [{"role": "user", "content": "committed before cancellation"}]
    seeded_ready_agent.context = agcontext(recent_transcript=seed)
    seeded_ready = seeded_ready_agent.run(
        agskill("seeded-ready", ""), agdata(label="must-not-run-seeded-ready")
    )
    unresolved: Future[agdata] = Future()
    blocked = blocked_agent.run(
        agskill("blocked", ""),
        agdata(label="must-not-run-blocked", dependency=agdata(_future=unresolved)),
    )
    ready = ready_agent.run(agskill("ready", ""), agdata(label="must-not-run-ready"))

    assert _request_state(seeded_ready) == "ready"
    assert _request_state(blocked) == "blocked"
    assert _request_state(ready) == "ready"

    def seeded_ready_done(_future) -> None:
        assert seeded_ready._context_future.done()
        callback_contexts["seeded_ready"] = seeded_ready._context_future.result().recent_transcript
        followups.append(
            seeded_ready_agent.run(agskill("followup", ""), agdata(label="callback-followup"))
        )
        callback_finished["seeded_ready"].set()

    seeded_ready._result_future.add_done_callback(seeded_ready_done)
    for name, invocation in (("blocked", blocked), ("ready", ready)):

        def observe_context(_future, *, name=name, invocation=invocation) -> None:
            if not invocation._context_future.done():
                pytest.fail("result callback ran before its output context settled")
            callback_contexts[name] = invocation._context_future.result().recent_transcript
            callback_finished[name].set()

        invocation._result_future.add_done_callback(observe_context)

    seeded_ready.cancel()
    blocked.cancel()
    ready.cancel()

    _terminal_error(seeded_ready, "agent invocation cancelled")
    _terminal_error(blocked, "agent invocation cancelled")
    _terminal_error(ready, "agent invocation cancelled")
    for finished in callback_finished.values():
        assert finished.wait(timeout=2)
    assert callback_contexts == {
        "seeded_ready": seed,
        "blocked": [],
        "ready": [],
    }
    assert seeded_ready.state == blocked.state == ready.state == "CANCELLED"
    assert unresolved.done() is False
    assert constructed_for == [capacity_agent]
    assert blocked_agent.engine is None
    assert ready_agent.engine is None

    release_capacity.set()
    assert capacity.wait(timeout=2).label == "holds-capacity"
    assert followups[0].wait(timeout=2).label == "callback-followup"
    assert executed == ["holds-capacity", "callback-followup"]


def test_running_invocation_cancel_is_observed_by_its_exact_safe_boundary(monkeypatch, tmp_path):
    entered_engine = threading.Event()
    enter_boundary = threading.Event()
    boundary_observed = threading.Event()
    callback_finished = threading.Event()
    seen = {}

    def execute(self, *, context, invocation, **_kwargs):
        seen["invocation"] = invocation
        context.recent_transcript.append({"role": "assistant", "content": "uncommitted mutation"})
        entered_engine.set()
        assert enter_boundary.wait(timeout=2)
        decision = invocation._checkpoint("test:after-tool", allow_messages=True, phase="tool")
        seen["decision"] = decision
        boundary_observed.set()
        if decision.cancelled:
            return agerror("agent invocation cancelled")
        return agdata(unexpected_success=True)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    ag = _agent(tmp_path)
    seed = [{"role": "user", "content": "committed"}]
    ag.context = agcontext(recent_transcript=seed)
    invocation = ag.run(agskill("controlled", ""), agdata())
    assert entered_engine.wait(timeout=2)

    invocation._result_future.add_done_callback(
        lambda _future: (
            pytest.fail("result callback ran before context settlement")
            if not invocation._context_future.done()
            else callback_finished.set()
        )
    )
    invocation.cancel()
    assert invocation._result_future.done() is False
    enter_boundary.set()

    assert boundary_observed.wait(timeout=2)
    _terminal_error(invocation, "agent invocation cancelled")
    assert callback_finished.wait(timeout=2)
    assert seen["invocation"] is invocation
    assert seen["decision"].cancelled is True
    assert invocation.state == "CANCELLED"
    assert invocation._context_future.result().recent_transcript == seed
    assert invocation._context_future.result().retained_messages == []
    assert ag.history.messages == seed


def test_suspend_uses_no_capacity_and_agent_resume_preserves_invocation_pause(
    monkeypatch, tmp_path
):
    other_started = threading.Event()
    release_other = threading.Event()
    suspended_started = threading.Event()

    def execute(self, *, skill_input, invocation, **_kwargs):
        assert invocation._agent is self._agent
        if skill_input.label == "other":
            other_started.set()
            assert release_other.wait(timeout=2)
        elif skill_input.label == "suspended":
            suspended_started.set()
        return agdata(label=skill_input.label)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    suspended_agent = _agent(tmp_path, max_engines=1)
    other_agent = _agent(tmp_path, max_engines=1)
    suspended_agent.suspend()
    queued = suspended_agent.run(agskill("queued", ""), agdata(label="suspended"))
    queued.pause()

    assert suspended_agent.is_suspended() is True
    assert _request_state(queued) == "ready"
    assert suspended_agent.engine is None
    other = other_agent.run(agskill("other", ""), agdata(label="other"))
    assert other_started.wait(timeout=2)
    assert get_orchestrator().snapshot().running_count == 1

    suspended_agent.resume()
    _drain_scheduler()
    assert suspended_agent.is_suspended() is False
    assert queued.is_pause_requested() is True
    assert suspended_started.is_set() is False
    assert suspended_agent.engine is None

    release_other.set()
    assert other.wait(timeout=2).label == "other"
    _drain_scheduler()
    assert _request_state(queued) == "ready"
    assert suspended_started.is_set() is False
    assert suspended_agent.engine is None

    queued.resume()
    assert suspended_started.wait(timeout=2)
    assert queued.wait(timeout=2).label == "suspended"


def test_destroy_settles_mixed_lifecycle_states_context_first_and_is_reusable(
    monkeypatch, tmp_path
):
    active_started = threading.Event()
    enter_paused_boundary = threading.Event()
    boundary_released_by_destroy = threading.Event()
    allow_engine_return = threading.Event()
    tail_history_read = threading.Event()
    constructed_for: list[agent] = []
    callback_context_done: dict[str, bool] = {}
    executed: list[str] = []
    original_init = AgentEngine.__init__

    def tracked_init(self, owner):
        constructed_for.append(owner)
        original_init(self, owner)

    def execute(self, *, context, skill_input, invocation, **_kwargs):
        executed.append(skill_input.label)
        assert skill_input.label == "active"
        context.recent_transcript.append(
            {"role": "assistant", "content": "destroyed working context"}
        )
        active_started.set()
        assert enter_paused_boundary.wait(timeout=2)
        decision = invocation._checkpoint(
            "test:before-commit", allow_messages=True, phase="boundary"
        )
        boundary_released_by_destroy.set()
        assert allow_engine_return.wait(timeout=2)
        if decision.destroyed:
            return agerror("agent destroyed")
        return agdata(unexpected_success=True)

    monkeypatch.setattr(AgentEngine, "__init__", tracked_init)
    monkeypatch.setattr(AgentEngine, "execute", execute)
    active_agent = _agent(tmp_path, max_engines=1)
    ready_agent = _agent(tmp_path, max_engines=1)
    seed = [{"role": "user", "content": "stable history"}]
    active_agent.context = agcontext(recent_transcript=seed)

    active = active_agent.run(agskill("active", ""), agdata(label="active"))
    assert active_started.wait(timeout=2)
    active.pause()
    enter_paused_boundary.set()
    with active._control._condition:
        assert active._control._condition.wait_for(lambda: active.state == "PAUSED", timeout=2)

    unresolved: Future[agdata] = Future()
    blocked = active_agent.run(
        agskill("blocked", ""),
        agdata(label="blocked", dependency=agdata(_future=unresolved)),
    )
    message = active_agent.queue_message("must be discarded by destruction")
    ready = ready_agent.run(agskill("ready", ""), agdata(label="ready"))

    assert _request_state(active) == "running"
    assert _request_state(blocked) == "blocked"
    assert isinstance(message, MessageSubmission)
    assert _request_state(message) == "blocked"
    assert _request_state(ready) == "ready"

    invocations = {
        "active": active,
        "blocked": blocked,
        "message": message,
        "ready": ready,
    }
    callback_finished = {name: threading.Event() for name in invocations}
    for name, invocation in invocations.items():

        def observe_context(_future, *, name=name, invocation=invocation) -> None:
            callback_context_done[name] = invocation._context_future.done()
            callback_finished[name].set()

        invocation._result_future.add_done_callback(observe_context)

    def read_settled_tail_history(_future) -> None:
        assert message._context_future.done()
        assert active_agent.history.messages == seed
        tail_history_read.set()

    message._result_future.add_done_callback(read_settled_tail_history)

    active_close = active_agent.destroy()
    ready_close = ready_agent.destroy()
    assert isinstance(active_close, CloseHandle)
    assert active_agent.destroy() is active_close
    assert ready_agent.destroy() is ready_close
    with pytest.raises(AgentDestroyedError):
        active_agent.run(agskill("rejected", ""), agdata())
    with pytest.raises(AgentDestroyedError):
        active_agent.queue_message("rejected")

    assert boundary_released_by_destroy.wait(timeout=2)
    assert active._result_future.done() is False
    assert active_close.done() is False
    allow_engine_return.set()

    for invocation in invocations.values():
        _terminal_error(invocation, "agent destroyed")
        assert invocation.state == "DESTROYED"
    assert tail_history_read.wait(timeout=2)
    for finished in callback_finished.values():
        assert finished.wait(timeout=2)
    assert callback_context_done == {name: True for name in invocations}
    assert active._context_future.result().retained_messages == []
    assert blocked._context_future.result().retained_messages == []
    assert message._context_future.result().retained_messages == []
    assert message._context_future.result().recent_transcript == seed
    assert unresolved.done() is False
    assert executed == ["active"]
    assert constructed_for == [active_agent]
    assert ready_agent.engine is None

    assert active_close.wait(timeout=2) is active_close
    assert active_close.wait(timeout=2) is active_close
    assert ready_close.wait(timeout=2) is ready_close
    assert active_close.done() is True
    assert active_agent.lifecycle_state == "DESTROYED"
    assert ready_agent.lifecycle_state == "DESTROYED"
