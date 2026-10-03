import json

import pytest
from fastapi.testclient import TestClient

from agency.observability.agwebui import investigator
from agency.observability.agwebui.investigator_demo import demo_run


def span(name, start, duration, **attrs):
    return {
        "ph": "X",
        "ts": start * 1e6,
        "dur": duration * 1e6,
        "name": name,
        "args": {"agency.agent_id": "worker", **attrs},
    }


def test_trace_units_outcomes_and_coverage_gaps_are_preserved():
    trace = {
        "traceEvents": [
            span(
                "tool:Bash",
                2,
                3,
                **{
                    "tool.arguments": '{"command":"pytest -q"}',
                    "outcome": "failure",
                    "span_id": "tool-1",
                    "cpu_ms": "n/a",
                },
            ),
            span("llm:attempt[0]", 10, 2, input_tokens=42),
            None,
        ]
    }
    run = investigator.normalize_trace(trace, {"id": "real", "source": "recorded"})
    assert run["duration"] == 10
    assert run["actions"][0]["duration"] == 3
    assert run["actions"][0]["outcome"] == "failed"
    assert run["actions"][0]["metadata"]["cpu_ms"] == "n/a"
    gaps = [i for i in run["intervals"] if i["kind"] == "unknown"]
    assert len(gaps) == 1
    assert gaps[0]["start"] == 3 and gaps[0]["duration"] == 5
    assert all(i["kind"] != "cpu" for i in run["intervals"])


@pytest.mark.parametrize(
    ("command", "kind"),
    [
        ("rg -n dispatch tests/test_agorchestrator.py", "search"),
        ("sed -n '1,20p' tests/test_agorchestrator.py", "read"),
        ("pytest tests/test_agorchestrator.py -q", "test"),
        ("apply_patch tests/test_agorchestrator.py", "edit"),
        ("git diff --check", "review"),
    ],
)
def test_command_classification_uses_action_not_test_filename(command, kind):
    assert investigator.classify("tool:Bash", command) == kind


def test_context_hashes_track_repetition_addition_and_drop_per_prompt():
    first = json.dumps(
        [{"role": "system", "content": "Be precise"}, {"role": "user", "content": "Original task"}]
    )
    context, previous = investigator.context_blocks(first, {})
    assert all(b["change"] == "added" for b in context["blocks"])
    second = json.dumps(
        [{"role": "system", "content": "Be precise"}, {"role": "tool", "content": "New output"}]
    )
    context, _ = investigator.context_blocks(second, previous)
    assert [b["change"] for b in context["blocks"]] == ["repeated", "added"]
    assert context["dropped"][0]["preview"] == "Original task"
    assert context["chars"] == len("Be preciseNew output")


def test_truncated_prompt_recovers_only_complete_messages_without_false_drops():
    complete = json.dumps(
        [{"role": "system", "content": "Instructions"}, {"role": "user", "content": "Task"}]
    )
    _, previous = investigator.context_blocks(complete, {})
    partial = '[{"role":"system","content":"Instructions"}, {"role":"tool","content":"cut'
    context, carried = investigator.context_blocks(partial, previous)
    assert context["truncated"]
    assert len(context["blocks"]) == 1
    assert context["blocks"][0]["change"] == "repeated"
    assert context["dropped"] == []
    assert len(carried) == 2


def test_context_histories_are_scoped_to_agents():
    prompt = json.dumps([{"role": "system", "content": "shared"}])
    trace = {
        "traceEvents": [
            span("llm:attempt[0]", 0, 1, **{"agency.agent_id": "a", "llm.messages": prompt}),
            span("llm:attempt[0]", 2, 1, **{"agency.agent_id": "b", "llm.messages": prompt}),
        ]
    }
    run = investigator.normalize_trace(trace, {})
    assert all(a["context"]["blocks"][0]["change"] == "added" for a in run["actions"])


def test_recovery_is_a_separate_episode_and_each_action_has_one_owner():
    run = demo_run("demo-baseline")
    occurrences = [id for e in run["episodes"] for id in e["actions"]]
    assert len(occurrences) == len(set(occurrences)) == len(run["actions"])
    failure = next(e for e in run["episodes"] if "a06" in e["actions"])
    recovered = next(e for e in run["episodes"] if "a09" in e["actions"])
    assert failure["status"] == "failed"
    assert recovered["status"] == "completed"
    assert failure["id"] != recovered["id"]
    assert all(e["end"] >= e["start"] for e in run["episodes"])


def test_demo_is_isolated_and_covers_all_investigative_projections():
    baseline, contention = demo_run("demo-baseline"), demo_run("demo-contention")
    assert baseline["source"] == "synthetic"
    assert all(a["source"] == "synthetic" for a in baseline["actions"])
    assert baseline["edges"] and baseline["handoff"] and baseline["counters"]
    assert any(o["status"] == "unvalidated" for o in baseline["obligations"])
    assert any(a["outcome"] == "failed" for a in baseline["actions"])
    assert contention["duration"] > baseline["duration"]
    assert len(contention["actions"]) > len(baseline["actions"])


def test_missing_usage_and_evaluator_remain_unknown():
    run = investigator.normalize_trace({"traceEvents": [span("llm:attempt[0]", 0, 1)]}, {})
    assert run["actions"][0]["tokens"] is None
    assert run["actions"][0]["outcome"] == "unknown"
    assert run["obligations"][-1]["status"] == "unvalidated"
    assert not run["edges"]


def test_counter_labels_and_observed_parent_relationships_are_normalized():
    name = "agency-counter-v1:" + json.dumps({"label": "workload_total cpu %"})
    trace = {
        "traceEvents": [
            span("tool:read", 0, 2, **{"agency.parent_agent_id": "supervisor"}),
            {"ph": "C", "ts": 1e6, "name": name, "args": {"value": 150}},
        ]
    }
    run = investigator.normalize_trace(trace, {})
    assert run["counters"]["workload_total cpu %"] == [[1, 150]]
    assert run["edges"][0]["from"] == "supervisor"
    assert run["edges"][0]["source"] == "recorded"


def test_endpoints_use_allowlisted_run_ids_and_hide_local_paths(monkeypatch, tmp_path):
    from agency.observability.agwebui import server

    path = tmp_path / "agprof.trace.json"
    path.write_text(json.dumps({"traceEvents": [span("tool:read", 0, 1)]}))
    entry = {"id": "saved", "title": "Saved fixture", "source": "recorded", "path": str(path)}
    monkeypatch.setattr(investigator, "catalog", lambda *args: [entry])
    client = TestClient(server.app)
    catalog = client.get("/api/investigator/runs").json()["runs"]
    assert all("path" not in e for e in catalog)
    response = client.get("/api/investigator/runs/saved")
    assert response.status_code == 200 and "path" not in response.json()
    assert client.get("/api/investigator/runs/not-a-run").status_code == 404
    assert client.get("/api/investigator/runs/saved/trace").status_code == 200
    assert client.get("/api/investigator/runs/demo-baseline").json()["source"] == "synthetic"
    assert client.get("/profiler").status_code == 200


def test_malformed_optional_summary_does_not_hide_valid_trace(tmp_path):
    path = tmp_path / "agprof.trace.json"
    path.write_text(json.dumps({"traceEvents": [span("tool:read", 0, 1)]}))
    (tmp_path / "summary.json").write_text("{partial")
    run = investigator.load_run(str(path), path.stat().st_mtime, json.dumps({"id": "test"}))
    assert len(run["actions"]) == 1


def test_native_purpose_is_joined_to_the_actual_tool_completion(tmp_path):
    path = tmp_path / "events.jsonl"
    events = [
        {
            "kind": "tool_annotation",
            "timestamp_ns": 10_000_000_000,
            "agent_id": "worker",
            "event_id": "call-a",
            "tool_name": "retrieve",
            "annotation": {
                "status": "valid",
                "raw": {"purpose": "Find scheduler references", "workstream_ids": ["investigate"]},
            },
            "arguments": {"query": "scheduler"},
        },
        {
            "kind": "tool_result",
            "timestamp_ns": 12_000_000_000,
            "duration_ns": 500_000_000,
            "agent_id": "worker",
            "event_id": "call-a",
            "call_id": "agency-a",
            "tool_name": "retrieve",
            "result": "Found two documents",
            "category": "ok",
        },
        {
            "kind": "model_exchange",
            "timestamp_ns": 13_000_000_000,
            "duration_ns": 600_000_000,
            "agent_id": "worker",
            "response": {"usage": {"prompt_tokens": 120, "completion_tokens": 12}},
        },
    ]
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n{truncated")
    run = investigator.normalize_native_events(
        path, {"id": "annotations", "source": "recorded", "model": "test-model"}
    )
    action = run["actions"][0]
    assert action["intent"] == "Find scheduler references"
    assert action["metadata"]["call_id"] == "agency-a"
    assert action["metadata"]["workstream_ids"] == ["investigate"]
    assert action["kind"] == "search"
    assert action["duration"] == 0.5
    assert run["actions"][1]["tokens"] == 120
    assert run["actions"][1]["context"] is None
    assert run["coverage"]["partial_lines"] == 1


def test_overview_sampling_is_bounded_and_preserves_peak_observations():
    samples = [[i / 10, 500 if i == 1234 else i % 13] for i in range(10001)]
    reduced = investigator.reduce_samples(samples)
    assert len(reduced) <= 800
    assert max(point[1] for point in reduced) == 500
    assert min(point[1] for point in reduced) == 0
    assert reduced == sorted(reduced)


def test_recorded_model_response_is_available_in_action_inspection():
    response = json.dumps(
        {
            "role": "assistant",
            "blocks": [
                {"type": "text", "text": "Inspect the queue before changing its ordering."},
                {"type": "tool_use", "name": "read", "arguments": '{"file_path":"scheduler.py"}'},
            ],
        }
    )
    trace = {"traceEvents": [span("llm:attempt[0]", 0, 2, **{"llm.response": response})]}
    run = investigator.normalize_trace(trace, {})
    assert "Inspect the queue" in run["actions"][0]["result"]
    assert "read(" in run["actions"][0]["result"]
    assert "llm.response" not in run["actions"][0]["metadata"]
