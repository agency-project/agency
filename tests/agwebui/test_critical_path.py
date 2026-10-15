"""Deterministic evidence/correlation cases for the historical profiler."""

import json

from agency.observability.agwebui.causal_graph import exchange_evidence
from agency.observability.agwebui.critical_path import build_profile, longest_path


def action(key, start, duration, kind="model", agent="a", **extra):
    return {
        "id": key,
        "agent": agent,
        "name": key,
        "kind": kind,
        "start": start,
        "duration": duration,
        "outcome": "success",
        **extra,
    }


def run(*actions, duration=50, **extra):
    return {
        "id": "recorded",
        "title": "Historical workload",
        "duration": duration,
        "status": "completed",
        "actions": list(actions),
        "clock": {"shared": True},
        **extra,
    }


def use(key):
    return {
        "type": "tool_use",
        "role": "assistant",
        "id": key,
        "name": "Bash",
        "input": {"command": key},
    }


def result(key):
    return {"type": "tool_result", "role": "user", "tool_call_id": key, "content": "ok"}


def test_single_sequential_agent_full_model_tool_chain():
    one, two = use("one"), use("two")
    trace = run(
        action(
            "m1",
            0,
            2,
            causal_evidence=exchange_evidence([], [one]),
            metadata={"ttft_ms": 250, "stream": True},
            model="recorded-model",
            tokens=10,
            output_tokens=20,
        ),
        action("t1", 2, 3, "tool", model_id="m1", model_tool_call_id="one"),
        action("m2", 5, 2, causal_evidence=exchange_evidence([one, result("one")], [two])),
        action("t2", 7, 4, "tool", model_id="m2", model_tool_call_id="two"),
        action(
            "m3",
            11,
            1,
            causal_evidence=exchange_evidence([one, result("one"), two, result("two")], []),
        ),
        duration=12,
    )
    profile = build_profile(trace)
    assert profile["critical_path"]["duration"] == 12
    assert profile["critical_path"]["operation_ids"] == ["m1", "t1", "m2", "t2", "m3"]
    assert profile["summary"]["breakdown"] == {"LLM": 5, "Tool": 7, "Wait": 0, "System": 0}
    assert profile["summary"]["aggregate_work"] == 12
    assert profile["summary"]["critical_path_coverage"] == 1
    assert not profile["coverage"]["complete"]  # temporal coverage is not causal completeness
    llm = profile["operations"][0]
    assert llm["model"] == "recorded-model"
    assert llm["metrics"]["tokens"] == 10
    assert llm["metrics"]["streaming_ms"] == 1750
    assert profile["bounds"] == {"start": 0, "end": 12}


def test_nested_process_not_added_to_tool_wall_time_and_interval_resources():
    identity = "123-456"
    profile = build_profile(
        run(
            action("bash", 0, 12, "tool", metadata={"call_id": "host-one"}),
            action(
                "gcc",
                1,
                10,
                "process",
                metadata={
                    "agency.process_identity": identity,
                    "tool_call_id": "host-one",
                    "agency.namespace_pid": 7,
                    "agency.process_start_ticks": 456,
                    "agency.pid_namespace": "pid:[99]",
                    "agency.sandbox_id": "box",
                },
            ),
            duration=12,
            clock={"shared": True},
            summary={
                "process_metrics": [
                    {
                        "identity": identity,
                        "pid": 123,
                        "namespace_pid": 7,
                        "start_ticks": 456,
                        "pid_namespace": "pid:[99]",
                        "sandbox": "box",
                        "name": "gcc",
                        "cpu_time_seconds": 8,
                        "rss_peak_mb": 220,
                    }
                ]
            },
            counters={
                f"process:{identity}:rss_mb": [[-1, 999], [1, 20], [5, 220], [11, 30], [15, 999]],
                f"process:{identity}:cpu_percent": [[1, 50], [5, 95]],
            },
        )
    )
    assert profile["critical_path"]["duration"] == 12
    assert profile["summary"]["aggregate_work"] == 12
    process = profile["processes"][0]
    assert process["owner_ids"] == ["bash"]
    assert process["start"] == 1
    assert process["end"] == 11
    assert process["metrics"]["cpu_time_seconds"] == 8
    tool = next(o for o in profile["operations"] if o["id"] == "bash")
    window = tool["metrics"]["process_resources"][0]["sampled_window"]
    assert window["rss_mb"]["observed_peak"] == 220
    assert window["cpu_percent"]["observed_peak"] == 95
    assert "cpu_time_seconds" not in tool["metrics"]


def test_unobserved_tool_issuance_consumption_gap_remains_unexplained_wait():
    tool = use("missing")
    profile = build_profile(
        run(
            action("first", 0, 1, causal_evidence=exchange_evidence([], [tool])),
            action("next", 5, 1, causal_evidence=exchange_evidence([tool, result("missing")], [])),
            duration=6,
        )
    )
    assert profile["critical_path"]["duration"] == 6
    assert profile["summary"]["critical_path_measured"] == 2
    assert profile["summary"]["critical_path_wait"] == 4
    assert profile["summary"]["breakdown"]["Tool"] == 0
    assert profile["coverage"]["malformed_edges"] == []


def test_incomplete_prompt_suffix_does_not_make_retained_history_active():
    tool = use("one")
    prefix = exchange_evidence([tool, result("one")], [], complete=False)
    profile = build_profile(
        run(
            action("model", 0, 2, causal_evidence=exchange_evidence([], [tool])),
            action("tool", 2, 3, "tool", model_id="model", model_tool_call_id="one"),
            action("consumer", 10, 1, causal_evidence=prefix),
            duration=11,
        )
    )
    assert not any(
        edge["blocking"]
        for edge in profile["edges"]
        if edge["target"] == "consumer" and edge["kind"] != "duration"
    )
    assert next(edge for edge in profile["edges"] if edge["kind"] == "issued_tool")["blocking"]
    assert profile["critical_path"]["duration"] == 5
    assert any("Incomplete prompt" in issue["reason"] for issue in profile["coverage"]["issues"])


def test_serial_tools_from_one_response_use_explicit_executor_order():
    first, second = use("one"), use("two")
    profile = build_profile(
        run(
            action("m", 0, 2, causal_evidence=exchange_evidence([], [first, second])),
            action(
                "a",
                2,
                2,
                "tool",
                model_id="m",
                model_tool_call_id="one",
                metadata={"call_id": "host-a", "tool_batch_id": "batch"},
            ),
            action(
                "b",
                4,
                4,
                "tool",
                model_id="m",
                model_tool_call_id="two",
                metadata={
                    "call_id": "host-b",
                    "tool_batch_id": "batch",
                    "preceding_tool_call_id": "host-a",
                    "preceding_model_tool_call_id": "one",
                },
            ),
            action(
                "final",
                8,
                2,
                causal_evidence=exchange_evidence(
                    [first, second, result("one"), result("two")], []
                ),
            ),
            duration=10,
        )
    )
    assert profile["summary"]["critical_path_measured"] == 10
    assert profile["summary"]["critical_path_wait"] == 0
    assert profile["critical_path"]["operation_ids"] == ["m", "a", "b", "final"]
    assert next(edge for edge in profile["edges"] if edge["kind"] == "serial_tool")["blocking"]


def test_namespace_and_start_generation_required_pid_alone_never_attributes():
    base = {"pid": 50, "namespace_pid": 2, "pid_namespace": "pid:[4]", "sandbox": "box"}
    profile = build_profile(
        run(
            action(
                "qualified",
                0,
                3,
                "tool",
                metadata={
                    "agency.sandbox_id": "box",
                    "agency.namespace_pid": 2,
                    "agency.pid_namespace": "pid:[4]",
                    "agency.process_start_ticks": 20,
                },
            ),
            action("pid_only", 0, 3, "tool", metadata={"pid": 50}),
            summary={
                "process_metrics": [
                    {**base, "identity": "50-10", "start_ticks": 10},
                    {**base, "identity": "50-20", "start_ticks": 20},
                ]
            },
        )
    )
    by_id = {p["identity"]: p for p in profile["processes"]}
    assert by_id["50-10"]["owner_ids"] == []
    assert by_id["50-20"]["owner_ids"] == ["qualified"]
    assert next(o for o in profile["operations"] if o["id"] == "pid_only")["process_ids"] == []


def test_duplicate_process_identity_and_contradictory_generation_excluded():
    row = {
        "identity": "50-10",
        "pid": 50,
        "namespace_pid": 2,
        "pid_namespace": "pid:[4]",
        "sandbox": "box",
        "start_ticks": 10,
    }
    duplicate = build_profile(
        run(
            action("tool", 0, 3, "tool", metadata={"agency.process_identity": "50-10"}),
            summary={"process_metrics": [row, {**row, "rss_peak_mb": 999}]},
        )
    )
    assert duplicate["processes"] == []
    assert any("Duplicate process" in issue["reason"] for issue in duplicate["coverage"]["issues"])
    contradictory = build_profile(
        run(
            action(
                "tool",
                0,
                3,
                "tool",
                metadata={"agency.process_identity": "50-10", "agency.process_start_ticks": 20},
            ),
            summary={"process_metrics": [row]},
        )
    )
    assert contradictory["processes"][0]["owner_ids"] == []
    assert any(
        "contradicts namespace" in issue["reason"] for issue in contradictory["coverage"]["issues"]
    )


def test_boolean_timestamps_and_durations_are_not_measurements():
    profile = build_profile(run(action("start", True, 2), action("duration", 0, True), duration=5))
    assert profile["critical_path"]["duration"] == 0
    assert all(op["duration"] is None for op in profile["operations"])


def test_nonstreaming_and_invalid_token_timing_are_not_reported_as_ttft():
    profile = build_profile(
        run(
            action("complete", 0, 2, metadata={"streaming": False, "ttft_ms": 1900}),
            action("invalid", 3, 1, metadata={"streaming": True, "ttft_ms": 2000}),
            duration=4,
        )
    )
    first, invalid = profile["operations"]
    assert "ttft_ms" not in first["metrics"] and first["metrics"]["response_latency_ms"] == 1900
    assert (
        "streaming_ms" not in invalid["metrics"] and invalid["metrics"]["recorded_ttft_ms"] == 2000
    )


def test_recorded_model_attempts_correlate_retries_by_logical_request_identity():
    profile = build_profile(
        run(
            action(
                "first",
                0,
                2,
                name="llm:attempt[0]",
                outcome="failed",
                metadata={"call_label": "request"},
            ),
            action("retry", 3, 4, name="llm:attempt[1]", metadata={"call_label": "request"}),
            duration=7,
        )
    )
    assert profile["critical_path"]["duration"] == 7
    assert profile["summary"]["critical_path_measured"] == 6
    assert profile["summary"]["critical_path_wait"] == 1
    assert (
        next(edge for edge in profile["edges"] if edge["kind"] == "llm_retry")["confidence"]
        == "reconstructed"
    )
    assert profile["operations"][1]["metrics"]["retry_count"] == 1


def test_three_independent_concurrent_agents_no_sum_or_fabricated_edge():
    profile = build_profile(
        run(
            action("one", 0, 15, agent="a"),
            action("two", 0, 40, agent="b"),
            action("three", 0, 25, agent="c"),
            duration=40,
        )
    )
    assert profile["critical_path"]["duration"] == 40
    assert profile["summary"]["aggregate_work"] == 80
    assert profile["critical_path"]["operation_ids"] == ["two"]
    assert not any(e["kind"] != "duration" and e["blocking"] for e in profile["edges"])
    assert profile["coverage"]["status"] == "candidate"
    assert profile["coverage"]["cross_agent_dependencies"] == 0


def test_explicit_fanout_join_chooses_worker_b_without_counting_wait_twice():
    dependencies = [{"source": "launch", "target": key, "kind": "spawn"} for key in ("a", "b", "c")]
    dependencies += [
        {"source": key, "target": "join", "kind": "join", "target_phase": "end"}
        for key in ("a", "b", "c")
    ]
    dependencies += [{"source": "launch", "target": "join"}, {"source": "join", "target": "final"}]
    profile = build_profile(
        run(
            action("launch", 0, 2, "tool", "orchestrator"),
            action("a", 2, 15, agent="a"),
            action("b", 2, 40, agent="b"),
            action("c", 2, 25, agent="c"),
            action("join", 2, 40, "wait", "orchestrator"),
            action("final", 42, 3, agent="orchestrator"),
            duration=45,
            dependencies=dependencies,
        )
    )
    assert profile["critical_path"]["duration"] == 45
    assert "b" in profile["critical_path"]["operation_ids"]
    assert "a" not in profile["critical_path"]["operation_ids"]
    assert "c" not in profile["critical_path"]["operation_ids"]
    assert profile["summary"]["breakdown"] == {"LLM": 43, "Tool": 2, "Wait": 0, "System": 0}
    assert sum(s["duration"] for s in profile["critical_path"]["segments"]) == 45
    assert profile["coverage"]["cross_agent_dependencies"] == 6


def test_disconnected_late_worker_join_gate_disclosed_separately_from_candidate():
    profile = build_profile(
        run(
            action("early", 0, 30, agent="a"),
            action("late", 29, 11, agent="b"),
            action("join", 0, 40, "wait", "orchestrator"),
            action("final", 40, 1, agent="orchestrator"),
            duration=41,
            dependencies=[
                {"source": "early", "target": "join", "target_phase": "end"},
                {"source": "late", "target": "join", "target_phase": "end"},
                {"source": "join", "target": "final"},
            ],
        )
    )
    gate = profile["joins"][0]
    assert gate["latest_prerequisite_ids"] == ["late"]
    assert gate["latest_completion_time"] == 40
    assert (
        next(
            edge
            for edge in profile["edges"]
            if edge["source"] == "early" and edge["target"] == "join"
        )["gating"]
        is False
    )
    assert (
        next(
            edge
            for edge in profile["edges"]
            if edge["source"] == "late" and edge["target"] == "join"
        )["gating"]
        is True
    )
    assert "late" in profile["critical_path"]["gating_operation_ids"]
    assert gate["candidate_path_matches_gate"] is False
    assert any(
        "startup causality" in limitation for limitation in profile["coverage"]["limitations"]
    )
    assert not profile["coverage"]["complete"]


def test_recorded_submission_sequence_establishes_late_worker_startup_prefix():
    profile = build_profile(
        run(
            action(
                "launch-a",
                0,
                25,
                "setup",
                "workflow",
                name="sync:submission",
                metadata={"submitted_request_id": "a"},
            ),
            action(
                "launch-b",
                25,
                1,
                "setup",
                "workflow",
                name="sync:submission",
                metadata={"submitted_request_id": "b", "preceding_submission_span_id": "launch-a"},
            ),
            action(
                "request-a",
                24,
                20,
                "request",
                "a",
                request_id="a",
                metadata={"launcher_span_id": "launch-a"},
            ),
            action(
                "request-b",
                25,
                20,
                "request",
                "b",
                request_id="b",
                metadata={"launcher_span_id": "launch-b"},
            ),
            action("work-a", 24, 20, agent="a", request_id="a"),
            action("work-b", 25, 20, agent="b", request_id="b"),
            action("join", 26, 19, "wait", "workflow"),
            action("final", 45, 1, agent="workflow"),
            duration=46,
            dependencies=[
                {"source": "request-a", "target": "join", "target_phase": "end"},
                {"source": "request-b", "target": "join", "target_phase": "end"},
                {"source": "join", "target": "final"},
            ],
        )
    )
    assert profile["critical_path"]["duration"] == 46
    assert "work-b" in profile["critical_path"]["operation_ids"]
    assert "work-a" not in profile["critical_path"]["operation_ids"]
    assert profile["summary"]["breakdown"]["System"] == 25
    assert next(edge for edge in profile["edges"] if edge["kind"] == "submission_sequence")[
        "blocking"
    ]


def test_request_prerequisite_resolves_completion_to_admission_not_submission():
    profile = build_profile(
        run(
            action("producer-envelope", 0, 40, "request", "worker", request_id="p"),
            action("producer-model", 0, 40, agent="worker", request_id="p"),
            action(
                "consumer-envelope",
                0,
                45,
                "request",
                "consumer",
                request_id="c",
                metadata={"dependency_request_ids": '["p"]'},
            ),
            action(
                "consumer-engine",
                40,
                5,
                "execution",
                "consumer",
                request_id="c",
                name="engine:execute",
            ),
            action("final", 40, 5, agent="consumer", request_id="c"),
            duration=45,
        )
    )
    assert profile["critical_path"]["duration"] == 45
    assert profile["summary"]["critical_path_measured"] == 45
    dependency = next(e for e in profile["edges"] if e["kind"] == "request_dependency")
    assert dependency["target"] == "consumer-engine" and dependency["target_phase"] == "start"
    assert dependency["source_time"] == dependency["target_time"] == 40
    assert not any(
        e["kind"] == "duration"
        and e["source"] in {"producer-envelope", "consumer-envelope", "consumer-engine"}
        for e in profile["edges"]
    )


def test_canonical_db_dependency_payload_and_wait_end_join():
    profile = build_profile(
        run(
            action("p", 0, 20, "request", "worker", request_id="p"),
            action("work", 0, 20, agent="worker", request_id="p"),
            action("c", 0, 25, "request", "consumer", request_id="c"),
            action(
                "wait", 0, 20, "dependency", "consumer", request_id="c", name="sync:dependency_wait"
            ),
            action("engine", 20, 5, "execution", "consumer", request_id="c", name="engine:execute"),
            action("model", 20, 5, agent="consumer", request_id="c"),
            duration=25,
            request_dependencies=[
                {
                    "request_id": "c",
                    "dependencies": [{"request_id": "p", "agent_id": "worker", "kind": "input"}],
                }
            ],
        )
    )
    assert profile["critical_path"]["duration"] == 25
    assert "work" in profile["critical_path"]["operation_ids"]
    assert profile["summary"]["critical_path_measured"] == 25
    assert (
        next(e for e in profile["edges"] if e["kind"] == "request_dependency")["target_phase"]
        == "end"
    )


def test_parent_agent_and_span_containment_are_not_blocking():
    profile = build_profile(
        run(
            action("parent", 0, 12, "tool", "parent"),
            action("child", 1, 10, "tool", "child", metadata={"parent_span_id": "parent"}),
            agents=[{"id": "parent"}, {"id": "child", "parent": "parent"}],
            duration=12,
        )
    )
    assert profile["critical_path"]["duration"] == 12
    assert profile["summary"]["aggregate_work"] == 22
    assert next(e for e in profile["edges"] if e["kind"] == "containment")["blocking"] is False
    assert profile["coverage"]["cross_agent_dependencies"] == 0


def test_incomplete_missing_end_clock_and_backwards_evidence_disclosed():
    profile = build_profile(
        run(
            action("open", 0, 999, outcome="running"),
            action("negative", 1, -5, "tool"),
            action("no-start", None, 9),
            action("first", 20, 3),
            action("second", 2, 4, agent="b"),
            dependencies=[{"source": "first", "target": "second"}],
            clock={"shared": False},
            duration=30,
        )
    )
    assert profile["critical_path"]["duration"] == 4
    assert profile["summary"]["critical_path_measured"] == 4
    assert next(o for o in profile["operations"] if o["id"] == "negative")["duration"] is None
    assert any("contradicts" in issue["reason"] for issue in profile["coverage"]["issues"])
    assert any("clock" in limitation.lower() for limitation in profile["coverage"]["limitations"])
    assert profile["summary"]["unattributed_time"] == 23
    assert profile["summary"]["path_unattributed_time"] == 26
    json.dumps(profile, allow_nan=False)


def test_syscalls_aggregated_out_and_process_without_samples_inspectable():
    profile = build_profile(
        run(
            *(action(f"syscall-{i}", 0, 0.01, "syscall", name="execve") for i in range(10000)),
            action("bash", 0, 2, "tool", metadata={"call_id": "call"}),
            action(
                "process",
                0.1,
                1.8,
                "process",
                metadata={
                    "tool_call_id": "call",
                    "agency.namespace_pid": 3,
                    "agency.pid_namespace": "pid:[9]",
                    "agency.process_start_ticks": 50,
                },
            ),
            duration=2,
        )
    )
    assert len(profile["operations"]) == 2
    assert profile["summary"]["process_count"] == 1
    assert profile["processes"][0]["owner_ids"] == ["bash"]
    assert profile["critical_path"]["duration"] == 2


def test_nested_tool_intervals_union_aggregate_work_and_never_22_seconds():
    profile = build_profile(
        run(
            action("outer", 0, 12, "tool"),
            action("inner", 1, 10, "tool", metadata={"parent_span_id": "outer"}),
            dependencies=[
                {"source": "outer", "target": "inner", "source_phase": "start"},
                {"source": "inner", "target": "outer", "target_phase": "end"},
            ],
            duration=12,
        )
    )
    assert profile["critical_path"]["duration"] == 12
    assert profile["summary"]["aggregate_work"] == 12
    segments = profile["critical_path"]["segments"]
    assert all(a["end"] <= b["start"] + 1e-6 for a, b in zip(segments, segments[1:]))


def test_zero_time_cycle_and_bad_weights_rejected_explicitly():
    nodes = [{"id": "a", "time": 0}, {"id": "b", "time": 0}]
    edges = [
        {
            "id": s + t,
            "source": s,
            "target": t,
            "weight": 0,
            "confidence": "recorded",
            "blocking": True,
        }
        for s, t in (("a", "b"), ("b", "a"))
    ]
    assert "Cyclic" in longest_path(nodes, edges)["error"]
    bad = [{**edges[0], "weight": 100}, {**edges[0], "id": "absent", "target": "absent"}]
    result = longest_path(nodes, bad)
    assert result["duration"] == 0
    assert len(result["malformed_edges"]) == 2


def test_large_parallel_graph_is_compact_and_linear_in_operations():
    profile = build_profile(
        run(
            *(
                action(f"tool-{i}", i % 100, 1, "tool", agent=f"agent-{i % 32}")
                for i in range(10000)
            ),
            duration=100,
        )
    )
    assert len(profile["operations"]) == 10000
    assert len(profile["edges"]) == 10000
    assert profile["critical_path"]["duration"] == 1
    assert len(profile["coverage"]["issues"]) <= 200


def test_historical_loader_reduction_and_omissions_remain_visible():
    profile = build_profile(
        run(
            action("one", 0, 1),
            coverage={
                "omitted_operations": 8,
                "historical_warnings": ["Damaged optional summary"],
                "resource_sampling": "Counter extrema reduced",
            },
            duration=10,
        )
    )
    limits = profile["coverage"]["limitations"]
    assert any("8 operation" in item for item in limits)
    assert "Damaged optional summary" in limits
    assert "Counter extrema reduced" in limits


def test_synthetic_unmeasured_tool_nodes_bounded_and_omissions_disclosed(monkeypatch):
    from agency.observability.agwebui import critical_path

    monkeypatch.setattr(critical_path, "MAX_PROJECTED_OPERATIONS", 5)
    tools = [use(str(index)) for index in range(100)]
    profile = build_profile(
        run(
            action("first", 0, 1, causal_evidence=exchange_evidence([], tools)),
            action(
                "next",
                10,
                1,
                causal_evidence=exchange_evidence(
                    [*tools, *(result(str(index)) for index in range(100))], []
                ),
            ),
            duration=11,
        )
    )
    assert len(profile["operations"]) <= 5
    assert any("projection" in limitation for limitation in profile["coverage"]["limitations"])
    assert profile["summary"]["breakdown"]["Tool"] == 0


def test_shared_sandbox_pressure_is_context_not_tool_owned_latency():
    profile = build_profile(
        run(
            action("one", 0, 4, "tool", metadata={"agency.sandbox_id": "shared"}),
            action("two", 0, 4, "tool", "b", metadata={"agency.sandbox_id": "shared"}),
            duration=4,
            sandbox_owners={"shared": ["a", "b"]},
            counters={
                "sandbox:shared:memory pressure some": [[1, 5], [2, 12]],
                "sandbox:other:memory pressure some": [[1, 999]],
            },
        )
    )
    sample = profile["operations"][0]["metrics"]["sandbox_resources"][0]
    assert sample["scope"] == "shared_sandbox"
    assert sample["shared_with"] == ["a", "b"]
    assert sample["sampled_window"]["memory pressure some"]["observed_peak"] == 12
    assert "does not establish latency causality" in sample["basis"]
    assert profile["critical_path"]["duration"] == 4
