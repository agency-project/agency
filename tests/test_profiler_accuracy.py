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
