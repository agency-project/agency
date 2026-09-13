import json
import pytest

from agency.observability.profiler.agprof_trace import build_trace, write_trace
from agency.observability.profiler.presentation import COUNTER_PREFIX


def record(name, span_id, parent=None, *, agent=None, side=None, sandbox=None, tid=7, start=1000):
    metadata = {
        k: v
        for k, v in {
            "agency.agent_id": agent,
            "agency.execution_side": side,
            "agency.sandbox_id": sandbox,
        }.items()
        if v is not None
    }
    return (tid, name, start, 1000, None, None, metadata, span_id, parent)


def trace(records, **kwargs):
    return build_trace(records, [], pid=42, process_info={}, started_ns=0, **kwargs)


def slices(document):
    return {e["name"]: e for e in document["traceEvents"] if e["ph"] == "X"}


def path(event):
    return json.loads(event["args"]["agency_presentation"])["path"]


def test_reused_worker_routes_each_agent_and_retains_execution_identity(tmp_path):
    records = [record("a", 1, agent="a", start=1000), record("b", 2, agent="b", start=3000)]
    document = trace(records)
    a, b = (slices(document)[name] for name in ("a", "b"))
    assert a["tid"] != b["tid"]
    assert a["args"]["agency.source_tid"] == b["args"]["agency.source_tid"] == 7
    assert path(a) == [["agent:a", "Agent a"], ["host", "Host"]]
    assert path(b)[0] == ["agent:b", "Agent b"]
    output = write_trace(tmp_path, records, [], pid=42, process_info={}, started_ns=0)
    assert json.loads(output.read_text()) == document


def test_host_llm_inherits_agent_but_not_sandbox_execution_side():
    records = [
        record("native", 1, agent="a", sandbox="s", side="sandbox"),
        record("llm", 2, 1, side="host", tid=8),
    ]
    document = trace(records)
    events = slices(document)
    assert path(events["native"])[1] == ["sandbox:s", "Sandbox s"]
    assert path(events["llm"])[1] == ["host", "Host"]
    flows = [e for e in document["traceEvents"] if e["ph"] in ("s", "f")]
    assert [e["tid"] for e in flows] == [events["native"]["tid"], events["llm"]["tid"]]
    assert flows[0]["ts"] == events["native"]["ts"]


def test_automatic_ownership_uses_captured_context_and_unknown_workers_stay_unknown():
    document = trace(
        [record("agent", 1, agent="a")],
        automatic_records=[
            (
                42,
                8,
                "owned",
                __file__,
                1,
                1000,
                1000,
                "return",
                "worker",
                {"agency.context_span_id": 1},
            ),
            (42, 8, "unknown", __file__, 1, 3000, 1000, "return", "worker", {}),
            (42, 42, "main", __file__, 1, 1000, 1000, "return", "main", {}),
        ],
    )
    events = slices(document)
    assert path(events["owned"])[0][0] == "agent:a"
    assert path(events["unknown"])[0][0] == "unattributed"
    assert path(events["main"])[0][0] == "workflow"


def test_shared_sandbox_and_owned_process_counters_have_one_canonical_path():
    records = [
        record("a", 1, agent="a", sandbox="shared", side="sandbox"),
        record("b", 2, agent="b", sandbox="shared", side="sandbox"),
    ]
    process = {
        "identity": "99-100",
        "pid": 99,
        "trace_pid": 99,
        "sandbox": "shared",
        "display_name": "python 99",
    }
    document = build_trace(
        records,
        [],
        pid=42,
        started_ns=0,
        process_info={"99-100": process},
        observations=[
            {
                "timestamp_ns": 1000,
                "name": "process:99-100:rss_mb",
                "trace_name": "rss_mb",
                "value": 5,
                "process_identity": "99-100",
                "trace_pid": 99,
            },
        ],
    )
    events = slices(document)
    assert path(events["a"]) == path(events["b"])
    assert path(events["a"])[0][0] == "shared"
    counter = next(e for e in document["traceEvents"] if e["ph"] == "C")
    spec = json.loads(counter["name"].removeprefix(COUNTER_PREFIX))
    assert spec["path"][:2] == path(events["a"])[:2]
    assert spec["path"][-2][0] == "process:99-100"
    assert counter["args"] == {"value": 5}


def test_remote_samples_require_explicit_sandbox_ownership():
    auto = lambda name, meta: (-2, 1, name, __file__, 1, 1000, 1000, "return", "native", meta)
    events = slices(
        trace(
            [record("agent", 1, agent="a", sandbox="s")],
            automatic_records=[
                auto(
                    "owned",
                    {
                        "agency.agent_id": "a",
                        "agency.sandbox_id": "s",
                        "agency.execution_side": "sandbox",
                    },
                ),
                auto("unowned", {}),
            ],
        )
    )
    assert "owned" in events and "unowned" not in events
    assert events["owned"]["args"]["agency.source_pid"] == -2


def test_malformed_parent_cycle_does_not_loop():
    document = trace([record("a", 1, 2, agent="a"), record("b", 2, 1)])
    assert len(slices(document)) == 2


def test_overlapping_agents_do_not_share_a_lane_or_infer_a_relationship():
    document = trace([record("a", 1, agent="a"), record("b", 2, agent="b")])
    events = slices(document)
    assert events["a"]["tid"] != events["b"]["tid"]
    assert events["a"]["ts"] == events["b"]["ts"]
    assert not any(e["ph"] in ("s", "f") for e in document["traceEvents"])


def test_interrupted_child_inherits_completed_parent_ownership():
    document = trace(
        [record("parent", 1, agent="a")],
        interrupted_spans=[
            {
                "thread_id": 8,
                "label": "child",
                "started_ns": 1000,
                "duration_ms": 1,
                "span_id": "0000000000000002",
                "parent_span_id": "0000000000000001",
            }
        ],
    )
    assert path(slices(document)["child"])[0][0] == "agent:a"


def test_packing_reuses_idle_lanes_without_moving_nested_calls_apart():
    records = [
        (7, "outer", 1000, 9000, None, None, {"agency.agent_id": "a"}, 1, None),
        record("child", 2, 1, tid=7, start=2000),
        record("parallel", 3, agent="a", tid=8, start=2500),
        record("later", 4, agent="a", tid=9, start=11000),
    ]
    document = trace(records)
    events = slices(document)
    assert events["outer"]["tid"] == events["child"]["tid"] == events["later"]["tid"]
    assert events["parallel"]["tid"] != events["outer"]["tid"]
    assert events["later"]["args"]["agency.source_tid"] == 9
    assert events["outer"]["args"]["agency.source_tid"] == 7
    assert events["outer"]["dur"] == 9
    labels = [
        e["args"]["name"]
        for e in document["traceEvents"]
        if e["ph"] == "M" and e["name"] == "thread_name"
    ]
    assert labels == ["Host · Lane 1", "Host · Lane 2"]


def test_packing_can_reuse_a_gap_in_a_source_thread():
    events = slices(
        trace(
            [
                record("early", 1, agent="a", tid=7),
                record("gap", 2, agent="a", tid=8, start=3000),
                record("late", 3, agent="a", tid=7, start=5000),
            ]
        )
    )
    assert len({e["tid"] for e in events.values()}) == 1


def mapped_trace(process_override=None):
    process = {
        "identity": "99-100",
        "pid": 99,
        "trace_pid": 99,
        "sandbox": "s",
        "display_name": "Python PID 99",
        "start_ticks": 100,
        "pid_namespace": "pid:[123]",
        "namespace_pid": 4,
    }
    process.update(process_override or {})
    metadata = {
        "agency.agent_id": "a",
        "agency.sandbox_id": "s",
        "agency.execution_side": "sandbox",
        "agency.reporter_id": -2,
        "agency.namespace_pid": 4,
        "agency.process_start_ticks": 100,
        "agency.pid_namespace": "pid:[123]",
    }
    records = [
        (7, "native", 1000, 1000, None, None, metadata, 1, None),
        record("llm", 2, 1, tid=8, side="host"),
    ]
    return build_trace(
        records,
        [],
        pid=42,
        started_ns=0,
        process_info={"99-100": process},
        observations=[
            {
                "timestamp_ns": 1000,
                "name": "process:99-100:rss_mb",
                "trace_name": "rss_mb",
                "value": 5,
                "process_identity": "99-100",
                "trace_pid": 99,
            }
        ],
    )


def test_namespace_pid_and_start_time_join_spans_to_process_counters():
    document = mapped_trace()
    events = slices(document)
    counter = next(e for e in document["traceEvents"] if e["ph"] == "C")
    spec = json.loads(counter["name"].removeprefix(COUNTER_PREFIX))
    assert path(events["native"]) == spec["path"][:-1]
    assert events["native"]["args"]["agency.host_pid"] == 99
    assert events["native"]["args"]["agency.namespace_pid"] == 4
    assert events["native"]["args"]["agency.process_identity"] == "99-100"
    assert "agency.host_pid" not in events["llm"]["args"]
    assert "agency.namespace_pid" not in events["llm"]["args"]


@pytest.mark.parametrize(
    "override",
    [
        {"start_ticks": 101},
        {"namespace_pid": 5},
        {"pid_namespace": "pid:[456]"},
        {"sandbox": "different"},
        {"pid_namespace": None},
    ],
)
def test_process_mapping_never_guesses_from_pid_or_name(override):
    native = slices(mapped_trace(override))["native"]
    assert "agency.host_pid" not in native["args"]
    assert path(native)[-1][0] == "reporter:-2"


def test_shared_sandbox_describes_navigation_without_duplicate_events():
    document = trace(
        [
            record("a", 1, agent="a", sandbox="s", side="sandbox"),
            record("b", 2, agent="b", sandbox="s", side="sandbox", start=3000),
        ]
    )
    events = slices(document)
    assert len(events) == 2
    for event in events.values():
        spec = json.loads(event["args"]["agency_presentation"])
        assert spec["path"][:2] == [["shared", "Shared resources"], ["sandbox:s", "Sandbox s"]]
        assert spec["shared_with"] == ["a", "b"]


def test_worker_ownership_is_propagated_without_sharing_annotation_handles(monkeypatch, tmp_path):
    from agency.observability.profiler import agprof
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    snapshots = []
    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        with agprof.span("parent"):
            agprof.annotate(**{"agency.agent_id": "a"})

            def child():
                snapshots.append(agprof._capture_execution_context())
                assert not agprof._span_stack.get()
                agprof.annotate(**{"agency.agent_id": "wrong"})

            thread = agprof.spawn_traced(child)
            thread.start()
            thread.join(timeout=2)
            assert not thread.is_alive()
            assert agprof.current_span_attributes()["agency.agent_id"] == "a"
        with ThreadPoolExecutor(max_workers=1) as executor:

            def owned_task(agent):
                with agprof.execution_context({"agency.agent_id": agent}):
                    return agprof._capture_execution_context()

            assert executor.submit(owned_task, "a").result()["agency.agent_id"] == "a"
            assert executor.submit(owned_task, "b").result()["agency.agent_id"] == "b"
            assert (
                "agency.agent_id" not in executor.submit(agprof._capture_execution_context).result()
            )
    assert snapshots[0]["agency.agent_id"] == "a"


def test_sampler_discovers_registered_sandbox_sibling_and_reads_namespace_identity(
    monkeypatch, tmp_path
):
    from agency.observability.profiler import agprof

    host = tmp_path / "host"
    sandbox = tmp_path / "sandbox/container"
    unrelated = tmp_path / "unrelated"
    for directory, pid in [(host, 42), (sandbox, 99), (unrelated, 100)]:
        directory.mkdir(parents=True)
        (directory / "cgroup.procs").write_text(str(pid))
    proc = tmp_path / "proc/99"
    (proc / "ns").mkdir(parents=True)
    fields = ["R"] + ["0"] * 21
    fields[19:22] = ["100", "1048576", "2"]
    (proc / "stat").write_text("99 (python) " + " ".join(fields))
    (proc / "status").write_text("Name:\tpython\nNSpid:\t99\t4\n")
    (proc / "ns/pid").symlink_to("pid:[123]")
    monkeypatch.setattr(agprof, "_cg_registry", {"s": str(sandbox)})
    monkeypatch.setattr(agprof, "_process_info", {})
    monkeypatch.setattr(agprof, "_samples", [])
    sampler = agprof._Sampler.__new__(agprof._Sampler)
    sampler._process_cgroup = host
    sampler._proc_root = tmp_path / "proc"
    sampler._clock_ticks = 100
    sampler._page_mb = 4096 / 2**20
    assert set(sampler._workload_pids()) == {42, 99}
    sampler._tick_processes(1000)
    info = agprof._process_info["99-100"]
    assert info["sandbox"] == "s"
    assert info["namespace_pid"] == 4
    assert info["pid_namespace"] == "pid:[123]"
    assert info["start_ticks"] == 100
    assert any(sample[1] == "proc:99-100:rss_mb" for sample in agprof._samples)


def test_native_spans_report_the_process_identity_needed_by_the_host():
    import os
    from types import SimpleNamespace
    from agency.native_harness.profiling import NativeProfiler

    payloads = []
    bridge = SimpleNamespace(
        profiler_settings=lambda: {"enabled": True, "automatic": None},
        record_profiler_span=lambda payload: payloads.append(payload) or {"ok": True},
    )
    with NativeProfiler(bridge) as native:
        with native.span("example"):
            pass
    assert len(payloads) == 2
    for payload in payloads:
        identity = payload["attributes"]
        assert identity["agency.namespace_pid"] == os.getpid()
        assert identity["agency.process_start_ticks"] > 0
        assert identity["agency.pid_namespace"] == os.readlink("/proc/self/ns/pid")


def test_parallel_reporters_in_shared_sandbox_keep_separate_lanes_and_no_inferred_edge():
    records = []
    for i, agent in enumerate(["a", "b"]):
        meta = {
            "agency.agent_id": agent,
            "agency.sandbox_id": "shared",
            "agency.execution_side": "sandbox",
            "agency.reporter_id": -2 - i,
        }
        records.append((-1, agent, 1000, 1000, None, None, meta, i + 1, None))
    document = trace(records)
    events = slices(document)
    # Different unresolved attempts also remain distinct process paths. Once
    # mapped, source reporter IDs must still prevent inferring parentage.
    assert events["a"]["tid"] != events["b"]["tid"]
    assert not any(e["ph"] in ("s", "f") for e in document["traceEvents"])
