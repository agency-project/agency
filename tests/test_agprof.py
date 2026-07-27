"""Tests for environment-controlled profiler lifecycle scopes."""

import json
import time
from contextlib import contextmanager

import pytest

from agency.profiler import agprof


def test_complete_summary_includes_spans_resources_and_gpu_leases():
    second = 1_000_000_000
    mebibyte = 2**20
    records = [
        (7, "turn0", 0, 100_000_000, 25_000_000, 5_000_000),
    ]
    samples = [
        (0, "host:cpu_s", 1.0),
        (0, "host:rss_mb", 100.0),
        (0, "host:io_r", 0.0),
        (0, "cg:writer:cpu_us", 0.0),
        (0, "gpu0:util_pct", 10.0),
        (second, "host:cpu_s", 1.5),
        (second, "host:rss_mb", 120.0),
        (second, "host:io_r", 2.0 * mebibyte),
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
    assert resources["host:cpu_pct"]["mean"] == 50.0
    assert resources["host:rss_mb"]["mean"] == 110.0
    assert resources["host:io_read_mb_s"]["mean"] == 2.0
    assert resources["host:io_read_mb_s"]["total"] == 2.0
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
    monkeypatch.setattr(
        agprof,
        "_inject_trace_args",
        lambda out_dir, records, interrupted_spans: None,
    )

    agprof.stop()

    machine_summary = json.loads((tmp_path / "summary.json").read_text())
    human_summary = (tmp_path / "summary.md").read_text()
    assert machine_summary["schema_version"] == 2
    assert machine_summary["span_metrics"][0]["label"] == "stage:work"
    assert "# agprof summary" in human_summary
    assert "| stage:work |" in human_summary
    assert agprof.summary_metrics() == machine_summary


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


def test_stop_snapshots_open_spans_as_interrupted(monkeypatch, tmp_path):
    class FakeRecordFunction:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

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
    monkeypatch.setattr(
        agprof,
        "_inject_trace_args",
        lambda out_dir, records, interrupted_spans: None,
    )

    active = agprof._TimedSpan(FakeRecordFunction, "run0:test:agent")
    active.__enter__()
    agprof.stop()
    active.__exit__(None, None, None)

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["run_metrics"]["started"] == 1
    assert summary["run_metrics"]["completed"] == 0
    assert summary["run_metrics"]["interrupted"] == 1
    assert summary["incomplete_spans"][0]["label"] == "run0:test:agent"
    assert not agprof._records


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
    monkeypatch.setattr(agprof.atexit, "register", lambda fn: events.append(("register", fn)))

    agprof._maybe_autostart()

    assert events == [("start", "process-trace"), ("register", agprof.stop)]


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
