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
