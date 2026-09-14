"""Deterministic lifecycle regressions for orchestrator-owned requests."""

from __future__ import annotations

import threading
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
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
    """Look up the orchestrator's internal request behind a bare handle --
    the only correlation key a returned handle carries is its own future."""
    orchestrator = get_orchestrator()
    future = object.__getattribute__(handle, "_future")
    with orchestrator._event_cond:
        request_id = orchestrator._future_producers.get(future)
        return orchestrator._requests.get(request_id) if request_id is not None else None


def _request_state(handle: agdata) -> str:
    return _request_for(handle).state


def _terminal_error(handle: agdata, expected: str) -> None:
    assert handle.wait(timeout=2).to_dict() == {"error": expected}


def test_cancel_takes_effect_only_once_naturally_launched_not_immediately(monkeypatch, tmp_path):
    """Verifies the deliberate trade-off: cancelling something not yet
    launched resolves no sooner than it would have anyway -- there is no
    scheduler-side scan or wake-event for cancel, only the engine's own
    pre/post checkpoints once a request actually reaches one."""
    capacity_started = threading.Event()
    release_capacity = threading.Event()
    executed: list[str] = []
    constructed_for: list[agent] = []
    original_init = AgentEngine.__init__

    def tracked_init(self, owner):
        constructed_for.append(owner)
        original_init(self, owner)

    def execute(self, *, skill_input, is_cancelled, **_kwargs):
        executed.append(skill_input.label)
        if skill_input.label == "holds-capacity":
            capacity_started.set()
            assert release_capacity.wait(timeout=2)
        if is_cancelled():
            return agerror("agent invocation cancelled")
        return agdata(label=skill_input.label)

    monkeypatch.setattr(AgentEngine, "__init__", tracked_init)
    monkeypatch.setattr(AgentEngine, "execute", execute)

    capacity_agent = _agent(tmp_path, max_engines=1)
    blocked_agent = _agent(tmp_path, max_engines=1)
    ready_agent = _agent(tmp_path, max_engines=1)
    capacity = capacity_agent.run(agskill("capacity", ""), agdata(label="holds-capacity"))
    assert capacity_started.wait(timeout=2)

    unresolved: Future[agdata] = Future()
    blocked = blocked_agent.run(
        agskill("blocked", ""),
        agdata(label="must-not-run-blocked", dependency=agdata(_future=unresolved)),
    )
    ready = ready_agent.run(agskill("ready", ""), agdata(label="must-not-run-ready"))
    assert _request_state(blocked) == "blocked"
    assert _request_state(ready) == "ready"

    blocked_agent.cancel(blocked)
    ready_agent.cancel(ready)

    # Neither resolves yet: capacity is still fully held, and the dependency
    # never resolves -- there is nothing that would settle them sooner.
    with pytest.raises(FutureTimeoutError):
        blocked.wait(timeout=0.2)
    with pytest.raises(FutureTimeoutError):
        ready.wait(timeout=0.2)
    assert constructed_for == [capacity_agent]

    release_capacity.set()
    assert capacity.wait(timeout=2).label == "holds-capacity"
    # `ready` now gets its turn -- an engine is constructed (cheap), but its
    # pre-checkpoint returns the controlled error before any harness spawns.
    _terminal_error(ready, "agent invocation cancelled")
    assert executed == ["holds-capacity"]
    assert constructed_for == [capacity_agent, ready_agent]

    # `blocked` is still stuck behind its own never-resolving dependency --
    # cancelling it doesn't fix that pre-existing hang, by design.
    with pytest.raises(FutureTimeoutError):
        blocked.wait(timeout=0.2)
    unresolved.set_result(agdata(unused=True))
    _terminal_error(blocked, "agent invocation cancelled")
    assert executed == ["holds-capacity"]


def test_running_invocation_cancel_is_observed_by_its_exact_safe_boundary(monkeypatch, tmp_path):
    entered_engine = threading.Event()
    enter_boundary = threading.Event()
    boundary_observed = threading.Event()
    callback_finished = threading.Event()
    seen = {}

    def execute(self, *, context, is_cancelled, **_kwargs):
        context.recent_transcript.append({"role": "assistant", "content": "uncommitted mutation"})
        entered_engine.set()
        assert enter_boundary.wait(timeout=2)
        cancelled = is_cancelled()
        seen["cancelled"] = cancelled
        boundary_observed.set()
        if cancelled:
            return agerror("agent invocation cancelled")
        return agdata(unexpected_success=True)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    ag = _agent(tmp_path)
    seed = [{"role": "user", "content": "committed"}]
    ag.context = agcontext(recent_transcript=seed)
    invocation = ag.run(agskill("controlled", ""), agdata())
    assert entered_engine.wait(timeout=2)
    request = _request_for(invocation)

    object.__getattribute__(invocation, "_future").add_done_callback(
        lambda _future: (
            pytest.fail("result callback ran before context settlement")
            if not request.context_future.done()
            else callback_finished.set()
        )
    )
    ag.cancel(invocation)
    assert not object.__getattribute__(invocation, "_future").done()
    enter_boundary.set()

    assert boundary_observed.wait(timeout=2)
    _terminal_error(invocation, "agent invocation cancelled")
    assert callback_finished.wait(timeout=2)
    assert seen["cancelled"] is True
    assert request.context_future.result().recent_transcript == seed
    assert request.context_future.result().retained_messages == []
    assert ag.history.messages == seed
