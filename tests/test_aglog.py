"""Tests for aglog — automatic skill call logging on agent."""

import threading
from unittest.mock import MagicMock

import pytest
from agency.agdata import agdata
from agency.agskill import agskill
from agency.aglog import aglog
from agency.agent import agent
from agency.agconfig import agConfig
from agency.engine import AgentEngine
from agency.engine.types import ExecutionResult


@pytest.fixture(autouse=True)
def _route_unit_execution_stubs_through_agent_engine(monkeypatch):
    """Exercise logging through the current AgentEngine dispatch seam."""
    real_execute = AgentEngine.execute

    def execute(self, *, context, skill, skill_input, resource_pool, sandbox, max_steps=None):
        stub = getattr(skill, "_test_execute", None)
        if stub is None:
            return real_execute(
                self,
                context=context,
                skill=skill,
                skill_input=skill_input,
                resource_pool=resource_pool,
                sandbox=sandbox,
                max_steps=max_steps,
            )
        output, updated_context, delta = stub(
            self._agent, context, skill_input, max_steps=max_steps
        )
        return ExecutionResult(output=output, context=updated_context, delta=delta)

    monkeypatch.setattr(AgentEngine, "execute", execute)


def make_agent(tools=None) -> agent:
    kwargs = {}
    if tools is not None:
        kwargs["tools"] = tools
    cfg = agConfig({"agllm_backend": {"api_key": "k", "model": "m"}})
    sandbox = MagicMock()
    sandbox._lock = threading.RLock()
    return agent(agconfig=cfg, sandbox=sandbox, **kwargs)


def make_skill(name: str = "s", out: dict | None = None):
    from agency.agcontext import agcontext

    sk = agskill(name, "")

    def fake_execute_react(ag, prev_ctx, inp, max_steps=None, **_):
        new_msgs = list(prev_ctx.messages) + [{"role": "user", "content": name}]
        return agdata(**(out or {"ok": True})), agcontext(messages=new_msgs), []

    sk._test_execute = fake_execute_react
    return sk


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


def test_agent_has_log():
    ag = make_agent()
    assert isinstance(ag.log, aglog)


def test_log_empty_initially():
    ag = make_agent()
    # no skill calls yet — entries and len are skill-only
    assert len(ag.log) == 0
    assert ag.log.entries == []
    # but the "created" lifecycle event is already present
    assert len(ag.log.events) == 1
    assert ag.log.events[0]["event"] == "created"


def test_log_records_after_run():
    skill = make_skill("search")
    ag = make_agent()
    result = ag.run(skill, agdata(query="test"))
    _ = result.ok  # wait for completion
    assert len(ag.log) == 1


def test_log_entry_fields():
    skill = make_skill("s", out={"value": 42})
    ag = make_agent()
    result = ag.run(skill, agdata(x=1))
    _ = result.value

    entry = ag.log.entries[0]
    assert entry["skill"] == "s"
    assert entry["input"] == {"x": 1}
    assert entry["output"] == {"value": 42}
    assert entry["history_len"] == 1  # one message appended
    assert "ts_start" in entry
    assert "ts_end" in entry


def test_log_timestamps_are_iso8601():
    from datetime import datetime

    skill = make_skill()
    ag = make_agent()
    _ = ag.run(skill, agdata()).ok

    e = ag.log.entries[0]
    datetime.fromisoformat(e["ts_start"])  # raises if invalid
    datetime.fromisoformat(e["ts_end"])


def test_log_ts_end_after_ts_start():
    skill = make_skill()
    ag = make_agent()
    _ = ag.run(skill, agdata()).ok

    e = ag.log.entries[0]
    assert e["ts_end"] >= e["ts_start"]


# ---------------------------------------------------------------------------
# Multiple calls accumulate in order
# ---------------------------------------------------------------------------


def test_log_accumulates_multiple_calls():
    skill_a = make_skill("a")
    skill_b = make_skill("b")
    ag = make_agent()
    ag.run(skill_a, agdata(step=1))
    ag.run(skill_b, agdata(step=2))
    _ = ag.history  # wait for both

    assert len(ag.log) == 2
    assert ag.log.entries[0]["skill"] == "a"
    assert ag.log.entries[1]["skill"] == "b"


def test_log_history_len_grows():
    skill = make_skill("s")
    ag = make_agent()
    ag.run(skill, agdata())
    ag.run(skill, agdata())
    _ = ag.history

    lens = [e["history_len"] for e in ag.log.entries]
    assert lens[0] == 1
    assert lens[1] == 2  # history grows with each call


# ---------------------------------------------------------------------------
# Error path is also logged
# ---------------------------------------------------------------------------


def test_invalid_skill_arg_raises_and_nothing_logged():
    ag = make_agent()
    # Passing a string (old API) or a non-agskill object should raise
    with pytest.raises((TypeError, AttributeError, ValueError)):
        ag.run("not_a_skill_object", agdata())
    assert len(ag.log) == 0


# ---------------------------------------------------------------------------
# Fork gets a fresh empty log
# ---------------------------------------------------------------------------


def test_fork_has_independent_log():
    skill = make_skill("s")
    ag = make_agent()
    _ = ag.run(skill, agdata()).ok  # parent logs one call

    fork = agent.fork(ag)
    assert len(fork.log) == 0  # fork starts fresh

    _ = fork.run(skill, agdata()).ok
    assert len(fork.log) == 1
    assert len(ag.log) == 1  # parent unaffected


# ---------------------------------------------------------------------------
# dump() output
# ---------------------------------------------------------------------------


def test_dump_empty():
    ag = make_agent()
    # dump shows the lifecycle "created" event even before any skill runs
    assert "CREATED" in ag.log.dump()


def test_dump_contains_skill_name():
    skill = make_skill("my_skill")
    ag = make_agent()
    _ = ag.run(skill, agdata(x=1)).ok
    assert "my_skill" in ag.log.dump()


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------


def test_created_event_logged():
    ag = make_agent()
    ev = ag.log.events[0]
    assert ev["type"] == "lifecycle"
    assert ev["event"] == "created"
    assert ev["agname"] == ag.agname
    assert "ts" in ev


def test_forked_event_logged():
    parent = make_agent()
    fork = agent.fork(parent)
    ev = fork.log.events[0]
    assert ev["event"] == "forked"
    assert ev["agname"] == fork.agname
    assert ev["parent_agname"] == parent.agname


def test_forked_agent_has_own_agname():
    parent = make_agent()
    fork = agent.fork(parent)
    assert fork.agname != parent.agname


def test_destroyed_event_logged():
    """Destruction is recorded on the agent's own log.

    `del` only drops the last *reference*; it does not promise the object is
    finalized on that line. It usually looks like it does -- refcounting
    collects immediately when nothing else holds the agent -- but inside a
    full suite something transiently can (a live thread, an interpreter
    frame, a cycle awaiting the generational collector), and then `__del__`
    runs later, at an unrelated moment. Asserting on the log immediately
    after `del` was therefore reading a race, which is exactly how this
    failed: a run showed CREATED and DESTROYED one millisecond apart while
    this log still ended at "created" -- the DESTROYED belonged to a
    *different* agent, an earlier test's, finalized in this test's window
    under a recycled agname (conftest clears the name registry per test).
    Collect explicitly and give finalization a bounded moment to land.
    """
    import gc
    import time

    ag = make_agent()
    log = ag.log  # keep a reference to the log after the agent dies
    del ag

    deadline = time.time() + 5.0
    while time.time() < deadline:
        gc.collect()
        if any(e.get("event") == "destroyed" for e in log.events):
            break
        time.sleep(0.02)

    assert [e.get("event") for e in log.events][-1] == "destroyed"


def test_events_includes_skill_and_lifecycle():
    skill = make_skill("s")
    ag = make_agent()
    _ = ag.run(skill, agdata()).ok
    types = [e["type"] for e in ag.log.events]
    assert "lifecycle" in types  # at least the "created" event
    assert "skill" in types


def test_skill_entry_has_type_field():
    skill = make_skill("s", out={"v": 1})
    ag = make_agent()
    _ = ag.run(skill, agdata()).v
    skill_entries = ag.log.entries
    assert skill_entries[0]["type"] == "skill"


def test_dump_shows_lifecycle_and_skills():
    skill = make_skill("my_skill")
    ag = make_agent()
    _ = ag.run(skill, agdata()).ok
    d = ag.log.dump()
    assert "CREATED" in d
    assert "my_skill" in d


# ---------------------------------------------------------------------------
# change_config / get_config_copy
# ---------------------------------------------------------------------------


def test_log_change_config_replaces_agconfig():
    log = aglog(agconfig=agConfig({"agllm_backend": {"temperature": 0.7}}))
    log.change_config(agConfig({"agllm_backend": {"temperature": 0.2}}))
    assert log._agconfig.get("agllm_backend", "temperature") == 0.2


def test_log_change_config_clones_given_agconfig():
    log = aglog(agconfig=agConfig())
    new_cfg = agConfig({"agllm_backend": {"temperature": 0.2}})
    log.change_config(new_cfg)
    new_cfg.agllm_backend.temperature = 0.9
    assert log._agconfig.get("agllm_backend", "temperature") == 0.2


def test_log_get_config_copy_returns_clone_not_same_object():
    cfg = agConfig({"agllm_backend": {"temperature": 0.7}})
    log = aglog(agconfig=cfg)
    copy = log.get_config_copy()
    assert copy is not log._agconfig


def test_log_get_config_copy_reflects_current_values():
    log = aglog(agconfig=agConfig({"agllm_backend": {"temperature": 0.7}}))
    assert log.get_config_copy().agllm_backend.temperature == 0.7


def test_mutating_log_get_config_copy_does_not_affect_log():
    log = aglog(agconfig=agConfig({"agllm_backend": {"temperature": 0.7}}))
    copy = log.get_config_copy()
    copy.agllm_backend.temperature = 0.1
    assert log._agconfig.get("agllm_backend", "temperature") == 0.7


def test_log_get_config_copy_none_when_no_agconfig():
    log = aglog()
    assert log.get_config_copy() is None


def test_log_change_config_none_clears_agconfig():
    log = aglog(agconfig=agConfig({"agllm_backend": {"temperature": 0.7}}))
    log.change_config(None)
    assert log.get_config_copy() is None
