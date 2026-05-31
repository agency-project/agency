"""Tests for aglog — automatic skill call logging on agent."""
import pytest
from src.agdata import agdata
from src.agskill import agskill
from src.aglog import aglog
from src.agent import agent


def make_agent(**kwargs) -> agent:
    return agent(llm_config={"api_key": "k", "model": "m"}, **kwargs)


def make_skill(name: str = "s", out: dict | None = None):
    sk = agskill(name, "")
    def fake_run(llm_cfg, inp, hist, tools, ms, **_):
        return agdata(**(out or {"ok": True})), agdata(
            messages=list(hist._data.get("messages", [])) + [{"role": "user", "content": name}]
        ), []
    sk.run = fake_run
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
    ag = make_agent(agskills=[make_skill("search")])
    result = ag.run("search", agdata(query="test"))
    _ = result.ok   # wait for completion
    assert len(ag.log) == 1


def test_log_entry_fields():
    ag = make_agent(agskills=[make_skill("s", out={"value": 42})])
    result = ag.run("s", agdata(x=1))
    _ = result.value

    entry = ag.log.entries[0]
    assert entry["skill"] == "s"
    assert entry["input"] == {"x": 1}
    assert entry["output"] == {"value": 42}
    assert entry["history_len"] == 1   # one message appended
    assert "ts_start" in entry
    assert "ts_end" in entry


def test_log_timestamps_are_iso8601():
    from datetime import datetime
    ag = make_agent(agskills=[make_skill()])
    _ = ag.run("s", agdata()).ok

    e = ag.log.entries[0]
    datetime.fromisoformat(e["ts_start"])   # raises if invalid
    datetime.fromisoformat(e["ts_end"])


def test_log_ts_end_after_ts_start():
    ag = make_agent(agskills=[make_skill()])
    _ = ag.run("s", agdata()).ok

    e = ag.log.entries[0]
    assert e["ts_end"] >= e["ts_start"]


# ---------------------------------------------------------------------------
# Multiple calls accumulate in order
# ---------------------------------------------------------------------------

def test_log_accumulates_multiple_calls():
    ag = make_agent(agskills=[make_skill("a"), make_skill("b")])
    ag.run("a", agdata(step=1))
    ag.run("b", agdata(step=2))
    _ = ag.history   # wait for both

    assert len(ag.log) == 2
    assert ag.log.entries[0]["skill"] == "a"
    assert ag.log.entries[1]["skill"] == "b"


def test_log_history_len_grows():
    ag = make_agent(agskills=[make_skill("s")])
    ag.run("s", agdata())
    ag.run("s", agdata())
    _ = ag.history

    lens = [e["history_len"] for e in ag.log.entries]
    assert lens[0] == 1
    assert lens[1] == 2   # history grows with each call


# ---------------------------------------------------------------------------
# Error path is also logged
# ---------------------------------------------------------------------------

def test_log_records_unknown_skill_error():
    ag = make_agent()
    result = ag.run("ghost", agdata())
    _ = result.error   # resolve

    assert len(ag.log) == 1
    e = ag.log.entries[0]
    assert e["skill"] == "ghost"
    assert "error" in e["output"]
    assert "ghost" in e["output"]["error"]


# ---------------------------------------------------------------------------
# Fork gets a fresh empty log
# ---------------------------------------------------------------------------

def test_fork_has_independent_log():
    ag = make_agent(agskills=[make_skill("s")])
    _ = ag.run("s", agdata()).ok   # parent logs one call

    fork = agent(ag)
    assert len(fork.log) == 0   # fork starts fresh

    _ = fork.run("s", agdata()).ok
    assert len(fork.log) == 1
    assert len(ag.log) == 1   # parent unaffected


# ---------------------------------------------------------------------------
# dump() output
# ---------------------------------------------------------------------------

def test_dump_empty():
    ag = make_agent()
    # dump shows the lifecycle "created" event even before any skill runs
    assert "CREATED" in ag.log.dump()


def test_dump_contains_skill_name():
    ag = make_agent(agskills=[make_skill("my_skill")])
    _ = ag.run("my_skill", agdata(x=1)).ok
    assert "my_skill" in ag.log.dump()


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------

def test_created_event_logged():
    ag = make_agent()
    ev = ag.log.events[0]
    assert ev["type"] == "lifecycle"
    assert ev["event"] == "created"
    assert ev["uuid"] == ag.uuid
    assert "ts" in ev


def test_forked_event_logged():
    parent = make_agent()
    fork = agent(parent)
    ev = fork.log.events[0]
    assert ev["event"] == "forked"
    assert ev["uuid"] == fork.uuid
    assert ev["parent_uuid"] == parent.uuid


def test_forked_agent_has_own_uuid():
    parent = make_agent()
    fork = agent(parent)
    assert fork.uuid != parent.uuid


def test_destroyed_event_logged():
    ag = make_agent()
    log = ag.log          # keep a reference to the log after the agent dies
    del ag
    events = log.events
    assert events[-1]["event"] == "destroyed"


def test_events_includes_skill_and_lifecycle():
    ag = make_agent(agskills=[make_skill("s")])
    _ = ag.run("s", agdata()).ok
    types = [e["type"] for e in ag.log.events]
    assert "lifecycle" in types   # at least the "created" event
    assert "skill" in types


def test_skill_entry_has_type_field():
    ag = make_agent(agskills=[make_skill("s", out={"v": 1})])
    _ = ag.run("s", agdata()).v
    skill_entries = ag.log.entries
    assert skill_entries[0]["type"] == "skill"


def test_dump_shows_lifecycle_and_skills():
    ag = make_agent(agskills=[make_skill("my_skill")])
    _ = ag.run("my_skill", agdata()).ok
    d = ag.log.dump()
    assert "CREATED" in d
    assert "my_skill" in d
