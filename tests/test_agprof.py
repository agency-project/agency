"""Tests for environment-controlled profiler lifecycle scopes."""

import asyncio
import json
import os
import threading
import time
from contextlib import contextmanager

import pytest

from agency.profiler import agprof
from agency.profiler import agprof_trace


def test_complete_summary_includes_per_process_workload_and_gpu_metrics(monkeypatch):
    second = 1_000_000_000
    mebibyte = 2**20
    records = [
        (7, "turn0", 0, 100_000_000, 25_000_000, 5_000_000),
    ]
    monkeypatch.setattr(
        agprof,
        "_process_info",
        {
            "101-10": {
                "identity": "101-10",
                "pid": 101,
                "comm": "python",
                "cmdline": "python benchmark.py",
                "sandbox": None,
                "cgroup": "/agprof.slice/run.scope",
                "display_name": "python (PID 101)",
                "trace_pid": 101,
                "first_seen_ns": 0,
            }
        },
    )
    samples = [
        (0, "workload:cpu_us", 1_000_000.0),
        (0, "workload:memory_mb", 100.0),
        (0, "workload:io_r", 0.0),
        (0, "proc:101-10:cpu_s", 1.0),
        (0, "proc:101-10:rss_mb", 30.0),
        (0, "proc:101-10:vms_mb", 90.0),
        (0, "proc:101-10:io_r", 0.0),
        (0, "cg:writer:cpu_us", 0.0),
        (0, "gpu0:util_pct", 10.0),
        (second, "workload:cpu_us", 1_500_000.0),
        (second, "workload:memory_mb", 120.0),
        (second, "workload:io_r", 2.0 * mebibyte),
        (second, "proc:101-10:cpu_s", 1.2),
        (second, "proc:101-10:rss_mb", 40.0),
        (second, "proc:101-10:vms_mb", 100.0),
        (second, "proc:101-10:io_r", 1.0 * mebibyte),
        (second, "cg:writer:cpu_us", 250_000.0),
        (second, "gpu0:util_pct", 30.0),
    ]

    summary = agprof._build_run_summary(
        records,
        samples,
        [(0, 0, 2 * second, "writer")],
        started_ns=0,
        ended_ns=2 * second,
        sample_hz=10.0,
        sample_gpu=True,
        gpu_sampling_available=True,
    )

    assert summary["duration_ms"] == 2000.0
    span = summary["span_metrics"][0]
    assert span["label"] == "turn"
    assert span["calls"] == 1
    assert span["started"] == 1
    assert span["wall_ms"] == 100.0
    assert span["cpu_ms"] == 25.0
    assert span["runqueue_ms"] == 5.0
    assert span["blocked_ms"] == 70.0
    assert span["cpu_percent"] == 25.0
    assert span["p50_ms"] == 100.0
    assert span["p95_ms"] == 100.0
    resources = {row["name"]: row for row in summary["resource_metrics"]}
    assert resources["workload_total:cpu_pct"]["mean"] == 50.0
    assert resources["workload_total:memory_mb"]["mean"] == 110.0
    assert resources["workload_total:io_read_mb_s"]["mean"] == 2.0
    assert resources["workload_total:io_read_mb_s"]["total"] == 2.0
    assert resources["process:101-10:cpu_pct"]["mean"] == 20.0
    assert resources["process:101-10:rss_mb"]["mean"] == 35.0
    assert resources["process:101-10:vms_mb"]["max"] == 100.0
    assert resources["process:101-10:io_read_mb_s"]["total"] == 1.0
    assert summary["process_metrics"][0]["pid"] == 101
    assert summary["process_metrics"][0]["name"] == "python"
    assert summary["workload_metrics"]["cpu_average_percent"] == 50.0
    assert resources["sandbox:writer:cpu_pct"]["mean"] == 25.0
    assert resources["sandbox:writer:cpu_pct"]["total"] == 0.25
    assert resources["gpu0:util_pct"]["mean"] == 20.0
    assert summary["gpu_lease_metrics"] == [
        {
            "gpu_id": 0,
            "label": "writer",
            "leases": 1,
            "total_ms": 2000.0,
            "mean_ms": 2000.0,
            "max_ms": 2000.0,
        }
    ]


def test_stop_writes_json_and_markdown_summaries(monkeypatch, tmp_path):
    class FakeProfiler:
        def stop(self):
            pass

    monkeypatch.setattr(agprof, "_session", object())
    monkeypatch.setattr(agprof, "_profiler", FakeProfiler())
    monkeypatch.setattr(agprof, "_out_dir", tmp_path)
    monkeypatch.setattr(agprof, "_sampler", None)
    monkeypatch.setattr(agprof, "_session_started_ns", time.perf_counter_ns() - 1_000_000)
    monkeypatch.setattr(agprof, "_session_sample_hz", 0.0)
    monkeypatch.setattr(agprof, "_session_sample_gpu", False)
    monkeypatch.setattr(
        agprof,
        "_records",
        [(7, "stage:work", 0, 1_000_000, 500_000, 100_000)],
    )
    monkeypatch.setattr(agprof, "_samples", [])
    monkeypatch.setattr(agprof, "_leases", [])
    monkeypatch.setattr(agprof, "_leases_open", {})
    calls = []
    build_observations = agprof._resource_observations
    original_write_summaries = agprof._write_summary_files
    original_write_trace = agprof_trace.write_trace

    def observe_once(*args, **kwargs):
        calls.append("observations")
        return build_observations(*args, **kwargs)

    def write_summaries(*args, **kwargs):
        calls.append("summaries")
        return original_write_summaries(*args, **kwargs)

    def write_trace_first(*args, **kwargs):
        calls.append("trace")
        return original_write_trace(*args, **kwargs)

    monkeypatch.setattr(agprof, "_resource_observations", observe_once)
    monkeypatch.setattr(agprof, "_write_summary_files", write_summaries)
    monkeypatch.setattr(agprof_trace, "write_trace", write_trace_first)
    agprof.stop()

    machine_summary = json.loads((tmp_path / "summary.json").read_text())
    human_summary = (tmp_path / "summary.md").read_text()
    trace = json.loads((tmp_path / "agprof.trace.json").read_text())
    assert machine_summary["schema_version"] == 5
    assert "workload_metrics" in machine_summary
    assert machine_summary["process_metrics"] == []
    assert "host_metrics" not in machine_summary
    assert machine_summary["span_metrics"][0]["label"] == "stage:work"
    assert "# agprof summary" in human_summary
    assert "| stage:work |" in human_summary
    span_event = next(event for event in trace["traceEvents"] if event.get("name") == "stage:work")
    assert span_event["ph"] == "X"
    assert span_event["args"]["blocked_ms"] == 0.4
    assert calls == ["observations", "trace", "summaries"]
    assert agprof.summary_metrics() == machine_summary


@pytest.mark.parametrize("failure", [KeyboardInterrupt("interrupt"), SystemExit("exit")])
def test_stop_reports_and_reraises_fatal_trace_failure(monkeypatch, tmp_path, failure):
    class FakeProfiler:
        def stop(self):
            pass

    monkeypatch.setattr(agprof, "_session", object())
    monkeypatch.setattr(agprof, "_profiler", FakeProfiler())
    monkeypatch.setattr(agprof, "_out_dir", tmp_path)
    monkeypatch.setattr(agprof, "_sampler", None)
    monkeypatch.setattr(agprof, "_session_started_ns", time.perf_counter_ns())
    monkeypatch.setattr(agprof, "_records", [])
    monkeypatch.setattr(agprof, "_samples", [])
    monkeypatch.setattr(agprof, "_leases", [])
    monkeypatch.setattr(agprof, "_leases_open", {})
    messages = []
    monkeypatch.setattr(agprof, "_agprof_print", messages.append)
    monkeypatch.setattr(
        agprof_trace,
        "write_trace",
        lambda *args, **kwargs: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(
        agprof,
        "_write_summary_files",
        lambda *args, **kwargs: pytest.fail("trace must be attempted before summaries"),
    )

    with pytest.raises(type(failure)):
        agprof.stop()

    assert messages[0].startswith("[agprof] writing trace:")
    assert messages[1] == f"[agprof] WARNING: trace output failed: {failure}"


def test_agprof_print_flushes(monkeypatch):
    calls = []
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: calls.append((args, kwargs)))

    agprof._agprof_print("[agprof] message")

    assert calls == [(("[agprof] message",), {"flush": True})]


def test_perfetto_trace_emits_spans_counters_process_tracks_and_gpu_leases():
    second = 1_000_000_000
    process_info = {
        "101-10": {
            "display_name": "python (PID 101)",
            "trace_pid": 101,
            "first_seen_ns": 0,
            "cgroup": "/workload",
            "cmdline": "python job.py",
            "comm": "python",
        }
    }
    records = [
        (
            7,
            "run0:test:agent",
            second,
            500_000_000,
            100_000_000,
            25_000_000,
            {"outcome": "success"},
            0x123,
            None,
        ),
        (
            7,
            "tool:read",
            1_100_000_000,
            100_000_000,
            20_000_000,
            None,
            {},
            0x456,
            0x123,
        ),
    ]
    samples = [
        (second, "proc:101-10:cpu_s", 1.0),
        (1_200_000_000, "proc:101-10:cpu_s", 1.04),
    ]

    trace = agprof_trace.build_trace(
        records,
        samples,
        [(0, 1_250_000_000, 1_500_000_000, "agent")],
        process_info=process_info,
        started_ns=second,
        pid=42,
    )
    events = trace["traceEvents"]
    run = next(event for event in events if event.get("name") == "run0:test:agent")
    tool = next(event for event in events if event.get("name") == "tool:read")
    counter = next(event for event in events if event.get("ph") == "C")
    lease = next(event for event in events if event.get("cat") == "gpu_lease")

    assert run["ts"] == 0.0
    assert run["dur"] == 500_000.0
    assert run["args"]["cpu_ms"] == 100.0
    assert run["args"]["blocked_ms"] == 375.0
    assert run["args"]["span_id"] == "0000000000000123"
    assert tool["args"]["parent_span_id"] == "0000000000000123"
    assert tool["args"]["runqueue_ms"] == "n/a"
    assert counter == {
        "ph": "C",
        "pid": 101,
        "tid": 0,
        "ts": 200_000.0,
        "name": "cpu_percent",
        "cat": "resource",
        "args": {"value": 20.0},
    }
    assert lease["tid"] == "gpu0-lease"
    assert lease["ts"] == 250_000.0
    assert lease["dur"] == 250_000.0
    assert any(
        event.get("name") == "process_name"
        and event.get("pid") == 101
        and event["args"]["name"] == "python (PID 101)"
        for event in events
    )


def test_perfetto_trace_keeps_interrupted_spans():
    trace = agprof_trace.build_trace(
        [],
        [],
        interrupted_spans=[
            {
                "thread_id": 9,
                "label": "run0:test:agent",
                "started_ns": 1_000_000,
                "duration_ms": 2.5,
                "reason": "session stopped",
                "outcome": "interrupted",
            }
        ],
        started_ns=1_000_000,
        pid=42,
        observations=[],
    )

    span = next(event for event in trace["traceEvents"] if event.get("ph") == "X")
    assert span["ts"] == 0.0
    assert span["dur"] == 2500.0
    assert span["args"] == {"reason": "session stopped", "outcome": "interrupted"}


def test_write_trace_streams_json_without_building_one_giant_string(monkeypatch, tmp_path):
    records = [(7, "stage:work", 10, 1_000, 500, 100, {"outcome": "success"})]
    monkeypatch.setattr(
        agprof_trace.json,
        "dumps",
        lambda *args, **kwargs: pytest.fail("write_trace must stream with json.dump"),
    )

    path = agprof_trace.write_trace(
        tmp_path,
        records,
        [],
        started_ns=10,
        pid=42,
        observations=[],
    )

    trace = json.loads(path.read_text())
    assert trace["displayTimeUnit"] == "ms"
    assert any(event.get("name") == "stage:work" for event in trace["traceEvents"])


def test_derived_rollups_include_outcomes_percentiles_tokens_energy_and_interruptions():
    second = 1_000_000_000
    records = [
        (1, "run0:writer:a", 0, second, 100_000_000, 0, {"outcome": "success"}),
        (2, "run1:writer:b", 0, 2 * second, 100_000_000, 0, {"outcome": "failure"}),
        (
            1,
            "llm:attempt[0]",
            0,
            400_000_000,
            10_000_000,
            0,
            {
                "outcome": "success",
                "ttft_ms": 100.0,
                "generation_ms": 300.0,
                "input_tokens": 20,
                "output_tokens": 30,
            },
        ),
        (
            2,
            "llm:attempt[0]",
            0,
            600_000_000,
            10_000_000,
            0,
            {"outcome": "failure", "retrying": True},
        ),
        (
            2,
            "llm:attempt[1]",
            0,
            800_000_000,
            10_000_000,
            0,
            {
                "outcome": "success",
                "ttft_ms": 200.0,
                "generation_ms": 600.0,
                "input_tokens": 40,
                "output_tokens": 60,
            },
        ),
        (1, "tool:write", 0, 200_000_000, 1_000_000, 0, {"outcome": "success"}),
        (2, "tool:read", 0, 300_000_000, 1_000_000, 0, {"outcome": "failure"}),
    ]
    samples = [
        (0, "gpu0:power_w", 10.0),
        (second, "gpu0:power_w", 20.0),
    ]
    interrupted = [
        {
            "thread_id": 3,
            "label": "run2:writer:c",
            "started_ns": 500_000_000,
            "duration_ms": 1500.0,
            "outcome": "interrupted",
        },
        {
            "thread_id": 3,
            "label": "tool:ask_human",
            "started_ns": 600_000_000,
            "duration_ms": 1400.0,
            "outcome": "interrupted",
        },
    ]

    summary = agprof._build_run_summary(
        records,
        samples,
        [],
        interrupted_spans=interrupted,
        started_ns=0,
        ended_ns=2 * second,
        sample_hz=10.0,
        sample_gpu=True,
        gpu_sampling_available=True,
    )

    assert summary["run_metrics"]["started"] == 3
    assert summary["run_metrics"]["completed"] == 2
    assert summary["run_metrics"]["failed"] == 1
    assert summary["run_metrics"]["interrupted"] == 1
    assert summary["run_metrics"]["completed_per_second"] == 1.0
    assert summary["run_metrics"]["p50_ms"] == 1500.0
    assert summary["llm_metrics"]["calls"] == 2
    assert summary["llm_metrics"]["retries"] == 1
    assert summary["llm_metrics"]["input_tokens"] == 60
    assert summary["llm_metrics"]["output_tokens"] == 90
    assert summary["llm_metrics"]["ttft"]["p50_ms"] == 150.0
    assert summary["llm_metrics"]["output_tokens_per_second"] == 100.0
    assert summary["tool_metrics"]["started"] == 3
    assert summary["tool_metrics"]["failed"] == 1
    assert summary["tool_metrics"]["interrupted"] == 1
    assert summary["gpu_metrics"][0]["energy_j"] == 15.0
    assert summary["incomplete_spans"][0]["outcome"] == "interrupted"
    markdown = agprof._render_summary_markdown(summary)
    assert "### Tool outcomes" in markdown
    assert "| read | 1/1 | 0 | 1 | 0 |" in markdown
    assert "| ask_human | 0/1 | 0 | 0 | 1 | n/a | n/a |" in markdown
    assert "## Per-process metrics" in markdown
    assert "## Workload aggregate" in markdown
    assert "Host metrics" not in markdown


def test_stop_snapshots_open_spans_as_interrupted(monkeypatch, tmp_path):
    class FakeSpan:
        def __init__(self):
            self.attributes = {}
            self.ended = False

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def end(self, **kwargs):
            self.ended = True

    class FakeProfiler:
        def stop(self):
            pass

    monkeypatch.setattr(agprof, "_session", object())
    monkeypatch.setattr(agprof, "_profiler", FakeProfiler())
    monkeypatch.setattr(agprof, "_out_dir", tmp_path)
    monkeypatch.setattr(agprof, "_sampler", None)
    monkeypatch.setattr(agprof, "_session_started_ns", time.perf_counter_ns() - 1_000_000)
    monkeypatch.setattr(agprof, "_session_sample_hz", 0.0)
    monkeypatch.setattr(agprof, "_session_sample_gpu", False)
    monkeypatch.setattr(agprof, "_records", [])
    monkeypatch.setattr(agprof, "_samples", [])
    monkeypatch.setattr(agprof, "_leases", [])
    monkeypatch.setattr(agprof, "_leases_open", {})
    monkeypatch.setattr(agprof, "_open_spans", {})
    monkeypatch.setattr(agprof, "_interrupted_spans", [])
    active = object.__new__(agprof._TimedSpan)
    active._name = "run0:test:agent"
    active._tid = 7
    active._t0 = time.perf_counter_ns() - 1_000_000
    active._metadata = {}
    active._interrupted = False
    active._span = FakeSpan()
    active._scope = None
    agprof._open_spans[id(active)] = active
    agprof.stop()
    active.__exit__(None, None, None)

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["run_metrics"]["started"] == 1
    assert summary["run_metrics"]["completed"] == 0
    assert summary["run_metrics"]["interrupted"] == 1
    assert summary["incomplete_spans"][0]["label"] == "run0:test:agent"
    assert not agprof._records
    assert active._span.ended
    assert active._span.attributes["outcome"] == "interrupted"


def _automatic_probe(value):
    time.sleep(0.002)
    return value + 1


def test_real_session_keeps_semantic_spans_and_automatic_calls(monkeypatch, tmp_path):
    pytest.importorskip("opentelemetry.sdk.trace")
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)

    with agprof.session(
        tmp_path,
        sample_hz=0,
        sample_gpu=False,
        auto_include=[__file__],
        auto_min_duration_ms=0,
    ):
        with agprof.span("run0:test:agent"):
            assert _automatic_probe(41) == 42

    trace = json.loads((tmp_path / "agprof.trace.json").read_text())
    events = trace["traceEvents"]
    assert any(
        event.get("cat") == "agprof" and event.get("name") == "run0:test:agent" for event in events
    )
    automatic = [
        event
        for event in events
        if event.get("cat") == "python.auto" and event.get("name", "").endswith("._automatic_probe")
    ]
    assert len(automatic) == 1
    assert automatic[0]["ts"] >= 0
    assert automatic[0]["args"]["automatic"] is True
    assert agprof.summary_metrics()["automatic_function_metrics"]["captured"] > 0


def test_trace_emits_semantic_thread_name_and_sampled_process_lifetime():
    process_info = {
        "101-10": {
            "pid": 101,
            "trace_pid": 101,
            "display_name": "Python — worker.py (PID 101)",
            "cgroup": "/workload",
            "cmdline": "python worker.py",
            "comm": "python",
            "first_seen_ns": 1_000,
            "last_seen_ns": 6_000,
        }
    }
    trace = agprof_trace.build_trace(
        [],
        [],
        process_info=process_info,
        observations=[],
        started_ns=0,
        pid=101,
        profile_root_pid=101,
        automatic_records=[
            (101, 7, "test.Worker.run", __file__, 1, 2_000, 2_000, "return", "Python worker")
        ],
        thread_labels={(101, 7): (90, "Agency tool dispatcher")},
    )
    events = trace["traceEvents"]
    assert any(
        event.get("name") == "thread_name"
        and event.get("tid") == 7
        and event["args"]["name"] == "Agency tool dispatcher"
        for event in events
    )
    lifetime = next(event for event in events if event.get("name") == "alive (sampled)")
    assert lifetime["pid"] == 101
    assert lifetime["dur"] == 5.0


def test_real_otel_session_records_nested_parent_ids(monkeypatch, tmp_path):
    pytest.importorskip("opentelemetry.sdk.trace")
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        with agprof.span("run0:test:agent"):
            with agprof.span("sandbox:exec"):
                agprof.annotate(operation="read")

    records = {record[1]: record for record in agprof._records}
    run = records["run0:test:agent"]
    sandbox = records["sandbox:exec"]
    assert len(run) == 9
    assert sandbox[8] == run[7]
    assert sandbox[6]["operation"] == "read"
    assert sandbox[6]["agency.wall_ns"] > 0
    assert sandbox[6]["agency.cpu_ns"] >= 0
    assert json.loads((tmp_path / "summary.json").read_text())["llm_metrics"]["calls"] == 0


def test_external_span_completion_can_retime_exact_interval(monkeypatch, tmp_path):
    pytest.importorskip("opentelemetry.sdk.trace")
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        with agprof.span("run0:test:agent"):
            original_perf_ns = time.perf_counter_ns()
            original_wall_ns = time.time_ns()
            external = agprof.start_external_span(
                "tool:bash",
                start_perf_ns=original_perf_ns,
                start_wall_ns=original_wall_ns,
                metadata={"timing": "exact"},
                parent_context=agprof.current_span_context(),
            )
            end_perf_ns = original_perf_ns + 10_000_000
            end_wall_ns = original_wall_ns + 10_000_000
            exact_start_perf_ns = end_perf_ns - 2_000_000
            exact_start_wall_ns = end_wall_ns - 2_000_000
            with pytest.raises(ValueError, match="overridden together"):
                external.end(
                    end_perf_ns=end_perf_ns,
                    end_wall_ns=end_wall_ns,
                    start_perf_ns=exact_start_perf_ns,
                )
            external.end(
                end_perf_ns=end_perf_ns,
                end_wall_ns=end_wall_ns,
                start_perf_ns=exact_start_perf_ns,
                start_wall_ns=exact_start_wall_ns,
                metadata={"outcome": "success"},
            )

    records = {record[1]: record for record in agprof._records}
    run = records["run0:test:agent"]
    tool = records["tool:bash"]
    assert tool[2] == exact_start_perf_ns
    assert tool[3] == 2_000_000
    assert tool[8] == run[7]
    assert external._span.start_time == exact_start_wall_ns
    assert external._span.end_time == end_wall_ns


def test_open_external_span_is_interrupted_at_profiler_stop(monkeypatch, tmp_path):
    pytest.importorskip("opentelemetry.sdk.trace")
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        with agprof.span("run0:test:agent"):
            external = agprof.start_external_span(
                "process:sleep",
                start_perf_ns=time.perf_counter_ns(),
                start_wall_ns=time.time_ns(),
                metadata={"pid": 42, "executable": "sleep", "timing": "exact"},
                parent_context=agprof.current_span_context(),
            )
            external.update("process:sleep", executable="sleep")

    summary = json.loads((tmp_path / "summary.json").read_text())
    incomplete = next(
        span for span in summary["incomplete_spans"] if span["label"] == "process:sleep"
    )
    assert incomplete["outcome"] == "interrupted"
    assert incomplete["pid"] == 42
    assert external._interrupted
    assert external._span.end_time is not None
    external.end(end_perf_ns=time.perf_counter_ns(), end_wall_ns=time.time_ns())
    assert not any(record[1] == "process:sleep" for record in agprof._records)


def test_cancel_external_span_is_silent_and_idempotent(monkeypatch, tmp_path):
    pytest.importorskip("opentelemetry.sdk.trace")
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        external = agprof.start_external_span(
            "tool:duplicate",
            start_perf_ns=time.perf_counter_ns(),
            start_wall_ns=time.time_ns(),
            metadata={"tool_call_id": "call-1"},
        )
        agprof.cancel_external_span(external)
        agprof.cancel_external_span(external)

    assert external._ended
    assert external._cancelled
    assert external._span is None
    assert id(external) not in agprof._open_spans
    assert not any(record[1] == "tool:duplicate" for record in agprof._records)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert not any(span["label"] == "tool:duplicate" for span in summary["incomplete_spans"])


def test_external_span_registration_is_atomic_with_stop(monkeypatch, tmp_path):
    """stop cannot drain between the session check and live-span registration."""
    pytest.importorskip("opentelemetry.sdk.trace")
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)

    real_observed_span = agprof._ObservedSpan
    constructor_entered = threading.Event()
    allow_registration = threading.Event()
    stop_attempted = threading.Event()
    stop_finished = threading.Event()
    result = {}
    errors = []

    def blocking_observed_span(*args, **kwargs):
        constructor_entered.set()
        if not allow_registration.wait(timeout=5):
            raise AssertionError("test did not release external span registration")
        return real_observed_span(*args, **kwargs)

    def register_external() -> None:
        try:
            result["span"] = agprof.start_external_span(
                "process:race",
                start_perf_ns=time.perf_counter_ns(),
                start_wall_ns=time.time_ns(),
                metadata={"pid": 42},
            )
        except BaseException as exc:  # surfaced in the main test thread
            errors.append(exc)

    def stop_session() -> None:
        stop_attempted.set()
        try:
            agprof.stop()
        except BaseException as exc:  # surfaced in the main test thread
            errors.append(exc)
        finally:
            stop_finished.set()

    agprof.start(tmp_path, sample_hz=0, sample_gpu=False)
    monkeypatch.setattr(agprof, "_ObservedSpan", blocking_observed_span)
    register_thread = threading.Thread(target=register_external)
    stop_thread = threading.Thread(target=stop_session)
    try:
        register_thread.start()
        assert constructor_entered.wait(timeout=2)
        state_lock_was_free = agprof._state_lock.acquire(blocking=False)
        if state_lock_was_free:
            agprof._state_lock.release()
        assert not state_lock_was_free
        stop_thread.start()
        assert stop_attempted.wait(timeout=2)
        # start_external_span owns _state_lock until registration completes,
        # so stop must still be waiting rather than draining an empty map.
        assert not stop_finished.wait(timeout=0.05)
    finally:
        allow_registration.set()
        register_thread.join(timeout=5)
        stop_thread.join(timeout=5)
        if agprof.enabled():
            agprof.stop()

    assert not register_thread.is_alive()
    assert not stop_thread.is_alive()
    assert errors == []
    assert result["span"]._interrupted
    assert agprof._open_spans == {}
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert any(span["label"] == "process:race" for span in summary["incomplete_spans"])


def test_concurrent_async_spans_keep_annotations_task_local(monkeypatch, tmp_path):
    """An open span in one coroutine must not own another task's annotate()."""
    pytest.importorskip("opentelemetry.sdk.trace")
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)

    async def overlap_spans():
        first_open = asyncio.Event()
        second_open = asyncio.Event()
        first_annotated = asyncio.Event()

        async def first():
            with agprof.span("async:first"):
                first_open.set()
                await second_open.wait()
                agprof.annotate(owner="first")
                first_annotated.set()

        async def second():
            await first_open.wait()
            with agprof.span("async:second"):
                second_open.set()
                await first_annotated.wait()
                agprof.annotate(owner="second")

        await asyncio.gather(first(), second())

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        asyncio.run(overlap_spans())

    records = {record[1]: record for record in agprof._records}
    assert records["async:first"][6]["owner"] == "first"
    assert records["async:second"][6]["owner"] == "second"
    assert records["async:first"][8] is None
    assert records["async:second"][8] is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "workload"),
        ("", "workload"),
        ("workload", "workload"),
        ("WORKLOAD", "workload"),
        ("invalid", "workload"),
        ("process", "process"),
        (" PROCESS ", "process"),
    ],
)
def test_profile_scope(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("AGENCY_PROFILE_SCOPE", raising=False)
    else:
        monkeypatch.setenv("AGENCY_PROFILE_SCOPE", value)

    assert agprof.profile_scope() == expected


def test_non_linux_environment_profiling_fails_before_cgroup_or_profiler(monkeypatch):
    monkeypatch.setenv("AGENCY_PROFILE", "1")
    monkeypatch.setattr(agprof.sys, "platform", "darwin")
    monkeypatch.setattr(
        agprof,
        "_ensure_environment_cgroup",
        lambda: pytest.fail("must reject before cgroup setup"),
    )
    monkeypatch.setattr(
        agprof,
        "_maybe_autostart",
        lambda: pytest.fail("must reject before profiler startup"),
    )

    with pytest.raises(RuntimeError, match="profiling is Linux-only"):
        agprof._initialize_environment_profiling()


def test_environment_cgroup_prefers_system_slice(monkeypatch):
    captured = {}
    monkeypatch.delenv("AGENCY_PROFILE_CGROUP", raising=False)
    monkeypatch.delenv("AGENCY_PROFILE_USER_SCOPE", raising=False)
    monkeypatch.setattr(agprof, "_system_cgroup_available", lambda: True)
    monkeypatch.setattr(agprof.sys, "orig_argv", ["/venv/bin/python", "bench.py", "--quick"])
    monkeypatch.setattr(agprof.os, "getuid", lambda: 1234)
    monkeypatch.setattr(agprof.os, "getgid", lambda: 5678)
    monkeypatch.setenv("USER", "benchmark")
    monkeypatch.setenv("HOME", "/home/benchmark")
    monkeypatch.setattr(agprof.uuid, "uuid4", lambda: type("U", (), {"hex": "abcdef012345"})())
    monkeypatch.setattr(
        agprof.os,
        "execvp",
        lambda executable, argv: captured.update(executable=executable, argv=argv),
    )

    agprof._ensure_environment_cgroup()

    assert captured["executable"] == "sudo"
    command = captured["argv"]
    slice_arg = next(arg for arg in command if arg.startswith("--slice="))
    slice_name = slice_arg.split("=", 1)[1]
    assert slice_name.startswith("agprof-")
    assert f"AGENCY_PROFILE_CGROUP_PARENT={slice_name}" in command
    assert any(
        arg == f"AGENCY_PROFILE_CGROUP=/sys/fs/cgroup/agprof.slice/{slice_name}" for arg in command
    )
    assert command[-3:] == ["/venv/bin/python", "bench.py", "--quick"]


def test_environment_cgroup_falls_back_to_user_scope(monkeypatch):
    captured = {}
    monkeypatch.delenv("AGENCY_PROFILE_CGROUP", raising=False)
    monkeypatch.delenv("AGENCY_PROFILE_USER_SCOPE", raising=False)
    monkeypatch.setattr(agprof, "_system_cgroup_available", lambda: False)
    monkeypatch.setattr(agprof, "_user_cgroup_available", lambda: True)
    monkeypatch.setattr(agprof.sys, "orig_argv", ["/venv/bin/python", "bench.py", "--quick"])
    monkeypatch.setattr(agprof.uuid, "uuid4", lambda: type("U", (), {"hex": "abcdef012345"})())
    monkeypatch.setattr(
        agprof.os,
        "execvp",
        lambda executable, argv: captured.update(executable=executable, argv=argv),
    )

    agprof._ensure_environment_cgroup()

    assert captured["executable"] == "systemd-run"
    command = captured["argv"]
    assert command[:3] == ["systemd-run", "--user", "--scope"]
    unit = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--unit="))
    assert unit.startswith("agprof-") and unit.endswith(".scope")
    assert f"AGENCY_PROFILE_USER_SCOPE={unit}" in command
    assert not any(arg.startswith("AGENCY_PROFILE_CGROUP=") for arg in command)
    assert not any(arg.startswith("AGENCY_PROFILE_CGROUP_PARENT=") for arg in command)
    assert command[-3:] == ["/venv/bin/python", "bench.py", "--quick"]


def test_environment_cgroup_falls_back_when_system_exec_fails(monkeypatch):
    commands = []
    monkeypatch.delenv("AGENCY_PROFILE_CGROUP", raising=False)
    monkeypatch.delenv("AGENCY_PROFILE_USER_SCOPE", raising=False)
    monkeypatch.setattr(agprof, "_system_cgroup_available", lambda: True)
    monkeypatch.setattr(agprof, "_user_cgroup_available", lambda: True)
    monkeypatch.setattr(agprof.sys, "orig_argv", ["/venv/bin/python", "bench.py"])

    def fake_execvp(executable, argv):
        commands.append((executable, argv))
        if executable == "sudo":
            raise OSError("sudo disappeared after the capability probe")

    monkeypatch.setattr(agprof.os, "execvp", fake_execvp)

    agprof._ensure_environment_cgroup()

    assert [executable for executable, _argv in commands] == ["sudo", "systemd-run"]


def test_system_cgroup_probe_checks_systemd_run_permission(monkeypatch):
    captured = []
    monkeypatch.setattr(
        agprof,
        "_command_succeeds",
        lambda command: captured.append(command) or True,
    )

    assert agprof._system_cgroup_available() is True
    assert captured == [["sudo", "-n", "systemd-run", "--version"]]


def test_environment_cgroup_user_scope_child_discovers_current_cgroup(tmp_path, monkeypatch):
    scope = tmp_path / "agprof-12ab.scope"
    scope.mkdir()
    for name in ("cpu.stat", "memory.current", "cgroup.procs"):
        (scope / name).write_text("")
    monkeypatch.delenv("AGENCY_PROFILE_CGROUP", raising=False)
    monkeypatch.setenv("AGENCY_PROFILE_CGROUP_PARENT", "agprof-stale.slice")
    monkeypatch.setenv("AGENCY_PROFILE_USER_SCOPE", scope.name)
    monkeypatch.setattr(agprof, "_current_cgroup_dir", lambda: scope)
    monkeypatch.setattr(
        agprof.os,
        "execvp",
        lambda *_args: pytest.fail("user-scope child must not re-exec"),
    )

    agprof._ensure_environment_cgroup()

    assert os.environ["AGENCY_PROFILE_CGROUP"] == str(scope)
    assert "AGENCY_PROFILE_CGROUP_PARENT" not in os.environ
    assert agprof.container_cgroup_parent() is None


def test_environment_cgroup_fails_when_neither_systemd_path_is_available(monkeypatch):
    monkeypatch.delenv("AGENCY_PROFILE_CGROUP", raising=False)
    monkeypatch.delenv("AGENCY_PROFILE_USER_SCOPE", raising=False)
    monkeypatch.setattr(agprof, "_system_cgroup_available", lambda: False)
    monkeypatch.setattr(agprof, "_user_cgroup_available", lambda: False)

    with pytest.raises(RuntimeError, match="passwordless sudo is unavailable"):
        agprof._ensure_environment_cgroup()


def test_workload_cgroup_sampler_reads_aggregate_cpu_memory_and_io(tmp_path, monkeypatch):
    (tmp_path / "cpu.stat").write_text("usage_usec 2500000\nuser_usec 2000000\n")
    (tmp_path / "memory.current").write_text(str(64 * 2**20))
    (tmp_path / "io.stat").write_text("8:0 rbytes=1048576 wbytes=2097152 rios=1 wios=2\n")
    sampler = object.__new__(agprof._Sampler)
    sampler._process_cgroup = tmp_path
    monkeypatch.setattr(agprof, "_samples", [])

    sampler._tick_workload_cgroup(99)

    assert agprof._samples == [
        (99, "workload:cpu_us", 2_500_000.0),
        (99, "workload:memory_mb", 64.0),
        (99, "workload:io_r", 1_048_576.0),
        (99, "workload:io_w", 2_097_152.0),
    ]


def _write_fake_proc(proc_root, pid, *, start_ticks, cpu_ticks, comm="python", io=True):
    process_dir = proc_root / str(pid)
    process_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "S",
        "1",
        "1",
        "1",
        "0",
        "-1",
        "0",
        "0",
        "0",
        "0",
        "0",
        str(cpu_ticks),
        "0",
        "0",
        "0",
        "20",
        "0",
        "1",
        "0",
        str(start_ticks),
        str(100 * 2**20),
        "256",
    ]
    (process_dir / "stat").write_text(f"{pid} ({comm}) " + " ".join(fields))
    (process_dir / "cmdline").write_bytes(f"{comm}\0worker.py\0".encode())
    if io:
        (process_dir / "io").write_text("read_bytes: 1048576\nwrite_bytes: 2097152\n")


def test_sampler_discovers_recursive_cgroup_pids_and_samples_each_process(tmp_path, monkeypatch):
    cgroup = tmp_path / "cgroup"
    child_cgroup = cgroup / "docker.scope"
    child_cgroup.mkdir(parents=True)
    (cgroup / "cgroup.procs").write_text("101\n")
    (child_cgroup / "cgroup.procs").write_text("202\n")
    proc_root = tmp_path / "proc"
    _write_fake_proc(proc_root, 101, start_ticks=10, cpu_ticks=150)
    _write_fake_proc(proc_root, 202, start_ticks=20, cpu_ticks=75, comm="container-python")

    sampler = object.__new__(agprof._Sampler)
    sampler._process_cgroup = cgroup
    sampler._proc_root = proc_root
    sampler._clock_ticks = 100
    sampler._page_mb = 4096 / 2**20
    monkeypatch.setattr(agprof, "_samples", [])
    monkeypatch.setattr(agprof, "_process_info", {})
    monkeypatch.setattr(agprof, "_cg_registry", {"worker": str(child_cgroup)})

    sampler._tick_processes(99)

    assert set(agprof._process_info) == {"101-10", "202-20"}
    assert agprof._process_info["101-10"]["display_name"] == "Python — worker.py (PID 101)"
    assert agprof._process_info["202-20"]["display_name"] == (
        "sandbox:worker — container-python (PID 202)"
    )
    assert (99, "proc:101-10:cpu_s", 1.5) in agprof._samples
    assert (99, "proc:101-10:rss_mb", 1.0) in agprof._samples
    assert (99, "proc:101-10:vms_mb", 100.0) in agprof._samples
    assert (99, "proc:202-20:io_w", 2_097_152.0) in agprof._samples


def test_sampler_skips_pid_that_exits_between_discovery_and_proc_read(tmp_path, monkeypatch):
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "cgroup.procs").write_text("303\n")
    sampler = object.__new__(agprof._Sampler)
    sampler._process_cgroup = cgroup
    sampler._proc_root = tmp_path / "proc"
    sampler._clock_ticks = 100
    sampler._page_mb = 4096 / 2**20
    monkeypatch.setattr(agprof, "_samples", [])
    monkeypatch.setattr(agprof, "_process_info", {})

    sampler._tick_processes(99)

    assert agprof._samples == []
    assert agprof._process_info == {}


def test_sampler_keeps_pid_reuse_as_two_process_identities(tmp_path, monkeypatch):
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "cgroup.procs").write_text("404\n")
    proc_root = tmp_path / "proc"
    sampler = object.__new__(agprof._Sampler)
    sampler._process_cgroup = cgroup
    sampler._proc_root = proc_root
    sampler._clock_ticks = 100
    sampler._page_mb = 4096 / 2**20
    monkeypatch.setattr(agprof, "_samples", [])
    monkeypatch.setattr(agprof, "_process_info", {})

    _write_fake_proc(proc_root, 404, start_ticks=10, cpu_ticks=10)
    sampler._tick_processes(1)
    _write_fake_proc(proc_root, 404, start_ticks=20, cpu_ticks=5)
    sampler._tick_processes(2)

    assert set(agprof._process_info) == {"404-10", "404-20"}
    assert agprof._process_info["404-10"]["trace_pid"] == 404
    assert agprof._process_info["404-20"]["trace_pid"] >= 1_000_000_000


def test_sampler_updates_container_label_when_registry_arrives_late(tmp_path, monkeypatch):
    cgroup = tmp_path / "cgroup"
    container_cgroup = cgroup / "docker.scope"
    container_cgroup.mkdir(parents=True)
    (container_cgroup / "cgroup.procs").write_text("505\n")
    proc_root = tmp_path / "proc"
    _write_fake_proc(proc_root, 505, start_ticks=50, cpu_ticks=10)
    sampler = object.__new__(agprof._Sampler)
    sampler._process_cgroup = cgroup
    sampler._proc_root = proc_root
    sampler._clock_ticks = 100
    sampler._page_mb = 4096 / 2**20
    monkeypatch.setattr(agprof, "_samples", [])
    monkeypatch.setattr(agprof, "_process_info", {})
    monkeypatch.setattr(agprof, "_cg_registry", {})

    sampler._tick_processes(1)
    assert agprof._process_info["505-50"]["sandbox"] is None
    agprof._cg_registry["worker"] = str(container_cgroup)
    sampler._tick_processes(2)

    assert agprof._process_info["505-50"]["sandbox"] == "worker"
    assert agprof._process_info["505-50"]["display_name"] == (
        "sandbox:worker — worker.py (PID 505)"
    )


def test_container_cgroup_parent_accepts_only_profiler_slice(monkeypatch):
    monkeypatch.delenv("AGENCY_PROFILE_USER_SCOPE", raising=False)
    monkeypatch.setenv("AGENCY_PROFILE_CGROUP_PARENT", "agprof-12ab.slice")
    assert agprof.container_cgroup_parent() == "agprof-12ab.slice"
    monkeypatch.setenv("AGENCY_PROFILE_CGROUP_PARENT", "../../system.slice")
    assert agprof.container_cgroup_parent() is None


def test_container_cgroup_parent_is_disabled_in_user_scope(monkeypatch):
    monkeypatch.setenv("AGENCY_PROFILE_USER_SCOPE", "agprof-12ab.scope")
    monkeypatch.setenv("AGENCY_PROFILE_CGROUP_PARENT", "agprof-12ab.slice")
    assert agprof.container_cgroup_parent() is None


def test_workload_scope_owns_session_exactly_around_workload(monkeypatch):
    events = []
    profiler = object()
    monkeypatch.setenv("AGENCY_PROFILE", "1")
    monkeypatch.delenv("AGENCY_PROFILE_SCOPE", raising=False)
    monkeypatch.setenv("AGENCY_PROFILE_DIR", "custom-trace")
    monkeypatch.setattr(agprof, "enabled", lambda: False)
    monkeypatch.setattr(
        agprof,
        "start",
        lambda out_dir: events.append(("start", out_dir)) or profiler,
    )
    monkeypatch.setattr(agprof, "stop", lambda: events.append(("stop", None)))

    with agprof.workload() as active:
        events.append(("workload", None))
        assert active is profiler

    assert events == [
        ("start", "custom-trace"),
        ("workload", None),
        ("stop", None),
    ]


def test_workload_scope_stops_session_when_workload_raises(monkeypatch):
    events = []
    monkeypatch.setenv("AGENCY_PROFILE", "true")
    monkeypatch.setenv("AGENCY_PROFILE_SCOPE", "workload")
    monkeypatch.setattr(agprof, "enabled", lambda: False)
    monkeypatch.setattr(agprof, "start", lambda out_dir: events.append("start"))
    monkeypatch.setattr(agprof, "stop", lambda: events.append("stop"))

    with pytest.raises(RuntimeError, match="boom"):
        with agprof.workload():
            events.append("workload")
            raise RuntimeError("boom")

    assert events == ["start", "workload", "stop"]


@pytest.mark.parametrize(
    ("profile_value", "scope"),
    [
        ("", "workload"),
        ("0", "workload"),
        ("1", "process"),
    ],
)
def test_workload_context_does_not_own_other_lifecycles(monkeypatch, profile_value, scope):
    monkeypatch.setenv("AGENCY_PROFILE", profile_value)
    monkeypatch.setenv("AGENCY_PROFILE_SCOPE", scope)
    monkeypatch.setattr(
        agprof,
        "start",
        lambda out_dir: pytest.fail("workload context must not start profiling"),
    )
    monkeypatch.setattr(
        agprof,
        "stop",
        lambda: pytest.fail("workload context must not stop profiling"),
    )

    with agprof.workload():
        pass


def test_workload_context_preserves_explicit_active_session(monkeypatch):
    existing = object()
    monkeypatch.setenv("AGENCY_PROFILE", "1")
    monkeypatch.setenv("AGENCY_PROFILE_SCOPE", "workload")
    monkeypatch.setattr(agprof, "enabled", lambda: True)
    monkeypatch.setattr(agprof, "_profiler", existing)
    monkeypatch.setattr(
        agprof,
        "start",
        lambda out_dir: pytest.fail("an explicit session is already active"),
    )
    monkeypatch.setattr(
        agprof,
        "stop",
        lambda: pytest.fail("must not stop an explicit session"),
    )

    with agprof.workload() as active:
        assert active is existing


def test_process_scope_is_the_only_environment_autostart(monkeypatch):
    events = []
    monkeypatch.setenv("AGENCY_PROFILE", "1")
    monkeypatch.setenv("AGENCY_PROFILE_SCOPE", "process")
    monkeypatch.setenv("AGENCY_PROFILE_DIR", "process-trace")
    monkeypatch.setattr(agprof, "start", lambda out_dir: events.append(("start", out_dir)))
    monkeypatch.setattr(
        agprof,
        "_install_process_profile_signal_handlers",
        lambda: events.append(("signals", None)),
    )
    monkeypatch.setattr(agprof.atexit, "register", lambda fn: events.append(("register", fn)))

    agprof._maybe_autostart()

    assert events == [
        ("start", "process-trace"),
        ("signals", None),
        ("register", agprof._shutdown_process_profile),
    ]


def test_process_shutdown_drains_shared_services_before_stopping_profiler(monkeypatch):
    from agency.agharness_internal import shared_services

    events = []
    monkeypatch.setattr(agprof, "_process_shutdown_started", False)
    monkeypatch.setattr(
        shared_services,
        "drain_shared_services",
        lambda: events.append("drain"),
    )
    monkeypatch.setattr(agprof, "stop", lambda: events.append("stop"))

    agprof._shutdown_process_profile()
    agprof._shutdown_process_profile()

    assert events == ["drain", "stop"]


def test_process_signal_handler_shuts_down_then_restores_and_reraises(monkeypatch):
    events = []
    monkeypatch.setattr(
        agprof,
        "_shutdown_process_profile",
        lambda: events.append(("shutdown", None)),
    )
    monkeypatch.setattr(
        agprof.signal,
        "signal",
        lambda signum, handler: events.append(("signal", signum, handler)),
    )
    monkeypatch.setattr(
        agprof.signal,
        "raise_signal",
        lambda signum: events.append(("raise", signum)),
    )

    agprof._process_profile_signal_handler(agprof.signal.SIGTERM, None)

    assert events == [
        ("signal", agprof.signal.SIGTERM, agprof.signal.SIG_IGN),
        ("signal", agprof.signal.SIGINT, agprof.signal.SIG_IGN),
        ("shutdown", None),
        ("signal", agprof.signal.SIGTERM, agprof.signal.SIG_DFL),
        ("raise", agprof.signal.SIGTERM),
    ]


def test_spawn_traced_preserves_parent_span_across_thread(monkeypatch, tmp_path):
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        with agprof.span("parent"):
            thread = agprof.spawn_traced(lambda: _record_child_span())
            thread.start()
            thread.join(timeout=2)
            assert not thread.is_alive()

    records = {record[1]: record for record in agprof._records}
    assert records["child"][8] == records["parent"][7]


def _record_child_span() -> None:
    with agprof.span("child"):
        pass


def _double(x):
    return x * 2


def _boom(_x):
    raise RuntimeError("mapped task blew up")


def test_agmap_tasks_are_traced_children_of_the_enclosing_span(monkeypatch, tmp_path):
    """agmap's fan-out is the framework's highest-fan-out spawn site: a bare
    thread there would make every mapped task a disconnected trace root
    (silently -- a disconnected root is a valid trace). Each task span must
    instead parent to whatever ran the map, and must stay a *task* label so
    run_metrics keeps counting only agent runs."""
    from agency.agmap import agmap

    monkeypatch.setattr(agprof, "_require_linux", lambda: None)

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        with agprof.span("parent"):
            results = agmap(_double, [1, 2, 3])

    assert [task.result for task in results] == [2, 4, 6]

    records = {record[1]: record for record in agprof._records}
    for index in range(3):
        record = records[f"agmap:_double[{index}]"]
        assert record[8] == records["parent"][7]  # parent_span_id -> the map's span
        assert record[6]["outcome"] == "success"

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["run_metrics"]["started"] == 0


def test_agmap_task_error_is_annotated_as_a_failure(monkeypatch, tmp_path):
    """A mapped task's exception becomes an agerror rather than propagating,
    so the span cannot see it as a raised exception -- the outcome has to be
    annotated explicitly or a failed fan-out reads as all-success."""
    from agency.agmap import agmap

    monkeypatch.setattr(agprof, "_require_linux", lambda: None)

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        result = agmap(_boom, 1)

    assert "mapped task blew up" in result._data["error"]

    record = {record[1]: record for record in agprof._records}["agmap:_boom[0]"]
    assert record[6]["outcome"] == "failure"
    assert record[6]["error_type"] == "agmap_task_error"


@pytest.mark.parametrize("scope", [None, "workload", "invalid"])
def test_default_and_invalid_scopes_do_not_autostart(monkeypatch, scope):
    monkeypatch.setenv("AGENCY_PROFILE", "1")
    if scope is None:
        monkeypatch.delenv("AGENCY_PROFILE_SCOPE", raising=False)
    else:
        monkeypatch.setenv("AGENCY_PROFILE_SCOPE", scope)
    monkeypatch.setattr(
        agprof,
        "start",
        lambda out_dir: pytest.fail("non-process scope must not autostart"),
    )
    monkeypatch.setattr(
        agprof.atexit,
        "register",
        lambda fn: pytest.fail("non-process scope must not register process cleanup"),
    )

    agprof._maybe_autostart()


def test_webui_marks_only_supplied_function_as_workload(monkeypatch, tmp_path):
    import agency.agwebui as agwebui_module

    events = []

    @contextmanager
    def workload():
        events.append("profile-start")
        try:
            yield
        finally:
            events.append("profile-stop")

    class FakeProcess:
        def poll(self):
            return None

        def terminate(self):
            events.append("server-stop")

        def wait(self, timeout):
            return 0

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def setsockopt(self, *args):
            pass

        def connect_ex(self, address):
            return 1

    monkeypatch.setattr(agwebui_module.agprof, "workload", workload)
    monkeypatch.setattr(agwebui_module.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(agwebui_module.urllib.request, "urlopen", lambda *args, **kwargs: None)
    monkeypatch.setattr("socket.socket", lambda *args, **kwargs: FakeSocket())
    monkeypatch.setattr(agwebui_module.atexit, "register", lambda fn: None)

    def fn():
        events.append("workload")

    agwebui_module.agwebui.run(fn, run_dir=tmp_path, port=17860, linger=False)

    assert events[:3] == ["profile-start", "workload", "profile-stop"]
