"""Sanitized evidence fixtures never invoke commands or billable models."""

import pytest

from agency.observability.agwebui.trajectory import Trajectory
from agency.observability.agwebui.trajectory_scenarios import (
    MULTI_AGENT_ENTRIES,
    multi_agent_events,
)


def project(count, hierarchy=False, clock=None):
    model = Trajectory(
        f"trajectory-agents-{count}", "Synthetic roster example", mode="review", source="synthetic"
    )
    for event in multi_agent_events(count, hierarchy=hierarchy):
        if clock is None or event["ts"] <= clock:
            model.apply(event)
    return model.snapshot()


@pytest.mark.parametrize("count", [3, 10, 32])
def test_roster_fixture_is_deterministic_and_contains_only_authored_evidence(count):
    assert multi_agent_events(count) == multi_agent_events(count)
    rows = multi_agent_events(count)
    assert all(row["source"] == "synthetic_fixture" for row in rows)
    assert len({row["id"] for row in rows}) == len(rows)
    run = project(count)
    assert run["source"] == "synthetic" and run["mode"] == "review"
    assert len(run["agents"]) == count
    assert len(run["actions"]) == 2 * count + 2
    assert run["status"] == "completed" and run["duration"] == 60
    assert run["counters"] == {}
    assert "no agents, commands, or models executed" in run["task"]
    assert all(agent["status"] == "completed" for agent in run["agents"])
    assert all(agent["parent"] is None for agent in run["agents"])
    assert run["actions"][-1]["outcome"] == "incomplete"


def test_roster_fixture_preserves_concurrent_boundaries_and_multiple_explicit_tool_links():
    active = project(10, clock=4)
    coordinator = [a for a in active["actions"] if a["agent"] == "coordinator"]
    assert len([a for a in coordinator if a["outcome"] == "running"]) == 2
    completed = project(10)
    calls = {action["id"]: action for action in completed["actions"]}
    model = calls["coordinator:model:example-response"]
    assert model["tool_call_ids"] == ["example-tool-1", "example-tool-2"]
    assert calls["coordinator:inspect"]["model_id"] == model["id"]
    assert calls["coordinator:check"]["model_id"] == model["id"]
    assert calls["coordinator:inspect"]["start"] < calls["coordinator:check"]["start"]
    assert calls["coordinator:inspect"]["end_ts"] > calls["coordinator:check"]["end_ts"]
    assert calls["worker-01:inspect"]["model_id"] is None


def test_hidden_failed_tool_does_not_change_recorded_agent_completion():
    run = project(32)
    hidden = next(agent for agent in run["agents"] if agent["id"] == "worker-31")
    failure = next(action for action in run["actions"] if action["id"] == "worker-31:check")
    assert failure["outcome"] == "failed"
    assert "synthetic expected value" in failure["result_preview"]
    assert hidden["status"] == "completed"


def test_hierarchy_exists_only_when_authored_parent_metadata_is_requested():
    explicit = project(10, hierarchy=True)
    agents = {agent["id"]: agent for agent in explicit["agents"]}
    assert agents["coordinator"]["parent"] is None
    assert agents["worker-01"]["parent"] == "coordinator"
    assert agents["worker-09"]["parent"] == "worker-01"
    assert agents["worker-09"]["parent_source"] == "recorded_parent_agname"
    assert all(agent["parent"] is None for agent in project(32)["agents"])


def test_large_output_keeps_safe_bounded_preview_and_original_authored_evidence():
    rows = multi_agent_events(10)
    original = next(
        row
        for row in rows
        if row["type"] == "tool_result"
        and row["actor"] == "worker-01"
        and row["payload"]["call_id"] == "inspect"
    )
    assert len(original["payload"]["result"]) > 100000
    assert "<script>" in original["payload"]["result"]
    action = next(
        action for action in project(10)["actions"] if action["id"] == "worker-01:inspect"
    )
    assert action["output_truncated"]
    assert len(action["result"]) == 8000
    assert len(action["result_preview"]) <= 240


def test_catalog_entries_are_explicit_examples_and_reject_arbitrary_counts():
    assert [entry["id"] for entry in MULTI_AGENT_ENTRIES] == [
        "trajectory-agents-10",
        "trajectory-agents-32",
    ]
    assert all(entry["source"] == "synthetic" for entry in MULTI_AGENT_ENTRIES)
    with pytest.raises(ValueError, match="Supported synthetic roster sizes"):
        multi_agent_events(100)
