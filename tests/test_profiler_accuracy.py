"""Measurement provenance must survive aggregation and serialization."""

from agency.observability.profiler import agprof


def summary(records):
    return agprof._build_run_summary(
        records,
        [],
        [],
        started_ns=0,
        ended_ns=1_000_000_000,
        sample_hz=0,
        sample_gpu=False,
        gpu_sampling_available=False,
    )


def test_estimates_and_unknown_outcomes_are_not_exact_successes():
    result = summary(
        [
            (1, "tool:read", 0, 5_000_000, None, None, {"timing": "derived"}),
            (1, "tool:read", 0, 2_000_000, None, None, {"timing": "exact", "outcome": "failure"}),
            (1, "tool:read", 0, 8_000_000, None, None, {"timing": "hook_boundary"}),
        ]
    )
    tools = result["tool_metrics"]
    assert (tools["succeeded"], tools["failed"], tools["unknown"]) == (0, 1, 2)
    assert tools["latency"]["p95_ms"] == 2
    assert tools["latency_by_timing"]["derived"]["p95_ms"] == 5
    assert result["data_source"] == "mixed"
    assert result["span_metrics"][0]["cpu_ms"] is None
    assert result["span_metrics"][0]["blocked_ms"] is None
    assert "Unknown tool outcomes: 2" in agprof._render_summary_markdown(result)


def test_unreported_retries_and_resources_are_unavailable():
    result = summary([(1, "llm:attempt[0]", 0, 10, None, None, {"outcome": "success"})])
    assert result["llm_metrics"]["retries"] is None
    assert result["coverage"]["resources"]["state"] == "unavailable"
    assert result["coverage"]["tools"]["state"] == "unavailable"
    assert result["sampling"]["lossless"] is False


def test_admission_spans_reach_profile_and_keep_parent_on_retirement(monkeypatch, tmp_path):
    import time
    from types import SimpleNamespace
    from agency.agpolicy import agpolicy
    from agency.engine.host_servers.host_interaction_server import HostInteractionServer

    class Logger:
        def record_event(self, *args, **kwargs):
            pass

        def record_span(self, *args, **kwargs):
            pass

    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    with agprof.session(tmp_path, sample_hz=0, auto_functions=False):
        run = agprof.start_external_span(
            "run0:test:agent", start_perf_ns=time.perf_counter_ns(), start_wall_ns=time.time_ns()
        )
        server = HostInteractionServer(
            SimpleNamespace(policy=agpolicy()), Logger(), parent_context=run.context()
        )
        completed = server.admit_tool_call("read", {})["call_id"]
        lost = server.admit_tool_call("write", {})["call_id"]
        # A completion of the wrong kind cannot consume a tool admission.
        server.complete_syscall(lost, 0)
        server.complete_tool_call(completed, error="failed")
        server.complete_tool_call(completed)
        server.finalize_profile()
        run.end(end_perf_ns=time.perf_counter_ns(), end_wall_ns=time.time_ns())
    records = {r[1]: r for r in agprof.profile_records()}
    assert records["tool:read"][8] == records["run0:test:agent"][7]
    assert records["tool:read"][6]["outcome"] == "failure"
    result = agprof.summary_metrics()
    assert result["tool_metrics"]["completed"] == 1
    assert result["tool_metrics"]["interrupted"] == 1
    assert result["tool_metrics"]["latency"]["p95_ms"] is None
    assert result["sampling"]["telemetry_errors"]["unmatched_completions"] == 1


def test_missing_wait_counter_is_unknown_in_trace_and_legacy_summary(monkeypatch):
    from agency.observability.profiler import agprof_trace

    records = [(1, "work", 0, 100, 25, None, {"outcome": "success"})]
    row = agprof._build_summary(records)["work"]
    assert row["cpu_ms"] == 25 / 1e6
    assert row["runq_ms"] is None
    assert row["blocked_ms"] is None
    monkeypatch.setattr(agprof, "_last_summary", {"work": row})
    assert "n/a" in agprof.summary_table()
    trace = agprof_trace.build_trace(records, [], [], observations=[])
    event = next(e for e in trace["traceEvents"] if e["name"] == "work")
    assert event["args"]["blocked_ms"] == "n/a"


def test_partial_usage_does_not_look_like_a_complete_token_total():
    result = summary(
        [
            (1, "llm:attempt[0]", 0, 10, 0, 0, {"input_tokens": 10, "output_tokens": 5}),
            (1, "llm:attempt[0]", 0, 10, 0, 0, {}),
        ]
    )["llm_metrics"]
    assert result["input_tokens"] is None
    assert result["reported_input_tokens"] == 10
    assert result["usage_missing_attempts"] == 1
    assert result["failed_attempts"] == 0
