from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

from agency import Agent, agdata, agent


def test_public_agent_exports_and_lifecycle_surface():
    assert Agent is agent
    assert not hasattr(agent, "send")
    assert not hasattr(agent, "steer")
    assert not hasattr(agent, "suspend")
    assert not hasattr(agent, "is_suspended")
    assert not hasattr(agent, "destroy")
    assert not hasattr(agent, "lifecycle_state")
    assert not hasattr(agent, "is_settled")
    assert hasattr(agent, "queue_message")
    assert hasattr(agent, "cancel")
    assert hasattr(agent, "pause")
    assert hasattr(agent, "resume")
    assert hasattr(agent, "redirect")


def _agent(tmp_path) -> agent:
    from agency.configs.agconfig import agconfig, agentconfig, llmconfig

    config = agconfig(
        agentconfig(log_dir=str(tmp_path)),
        llmconfig(api_key="test", model="m"),
    )
    sandbox = MagicMock()
    return agent(sandbox=sandbox, agconfig=config)


def test_cancel_never_marks_the_handle_itself(tmp_path):
    # cancel() is purely a future-identity lookup through the orchestrator --
    # verified end-to-end (a real submission actually getting cancelled) in
    # test_orchestrator_lifecycle.py. Here we only need the negative: the
    # handle agent.run() returns is never itself mutated by cancel(). Checked
    # via object.__getattribute__ (bypassing agdata.__getattr__), since the
    # future is intentionally never resolved and a plain hasattr()/getattr()
    # would block forever trying to resolve it.
    ag = _agent(tmp_path)
    handle = agdata(_future=Future())
    ag.cancel(handle)
    assert object.__getattribute__(handle, "_data") == {}


def test_cancel_on_an_already_settled_or_unknown_future_is_a_harmless_no_op(tmp_path):
    ag = _agent(tmp_path)
    settled: "Future[agdata]" = Future()
    settled.set_result(agdata(answer="done"))
    ag.cancel(agdata(_future=settled))  # never registered -- no-op
    ag.cancel(agdata())  # no future at all -- no-op


def test_cancel_cannot_cancel_another_agents_result(tmp_path, monkeypatch):
    from agency import agskill
    from agency.engine import AgentEngine

    owner = _agent(tmp_path)
    other = _agent(tmp_path)
    dependency = Future()
    monkeypatch.setattr(AgentEngine, "execute", lambda *args, **kwargs: agdata(answer=42))
    result = owner.run(agskill("owned", ""), agdata(input=agdata(_future=dependency)))
    other.cancel(result)
    dependency.set_result(agdata())
    assert result.wait(timeout=2).to_dict() == {"answer": 42}


def test_pause_resume_gate(tmp_path):
    ag = _agent(tmp_path)
    assert ag.is_paused() is False

    ag.pause()
    assert ag.is_paused() is True

    ag.resume()
    assert ag.is_paused() is False


def test_pause_resume_skips_stale_daemon_when_latest_engine_services_are_closed(tmp_path):
    ag = _agent(tmp_path)
    daemon = MagicMock()
    ag.sandbox._agency_harness_daemon_handles = {str(ag.agname): daemon}
    ag.engine = SimpleNamespace(_services_closed=True)

    ag.pause()
    ag.resume()

    daemon.client.assert_not_called()


def test_pause_resume_skips_unreachable_cached_daemon(tmp_path, monkeypatch):
    ag = _agent(tmp_path)
    daemon = MagicMock()
    ag.sandbox._agency_harness_daemon_handles = {str(ag.agname): daemon}
    ag.engine = SimpleNamespace(_services_closed=False)
    monkeypatch.setattr("agency.engine.harness_daemon_launcher._is_ready", lambda _handle: False)

    ag.pause()
    ag.resume()

    daemon.client.assert_not_called()
