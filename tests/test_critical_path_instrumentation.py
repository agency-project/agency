"""Recorded causal evidence must survive scheduler completion and PID reuse."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agency.agent import agent
from agency.agdata import agdata
from agency.agskill import agskill
from agency.configs.agconfig import agconfig, agentconfig, llmconfig
from agency.engine import AgentEngine
from agency.native_harness.profiling import NativeProfiler, run_process, tool_execution
from agency.observability.profiler import agprof


def make_agent(tmp_path):
    sandbox = MagicMock()
    sandbox._lock = threading.RLock()
    return agent(
        sandbox=sandbox,
        agconfig=agconfig(agentconfig(log_dir=str(tmp_path)), llmconfig(api_key="test", model="m")),
    )


def dependency_events(orchestrator):
    orchestrator.data_logger.flush()
    with sqlite3.connect(orchestrator.data_logger.db_path) as connection:
        return [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT payload FROM events WHERE type='request_dependencies' ORDER BY id"
            )
        ]


def test_completed_input_and_context_keep_explicit_dependencies(monkeypatch, tmp_path):
    monkeypatch.setattr(AgentEngine, "execute", lambda self, **kwargs: agdata(ok=True))
    producer, consumer = make_agent(tmp_path), make_agent(tmp_path)
    skill = agskill("task", "")
    with agprof.session(tmp_path / "profile", sample_hz=0, auto_functions=False):
        first = producer.run(skill, agdata()).wait()
        first_id = object.__getattribute__(first, "_execution_id")
        second = producer.run(skill, agdata()).wait()
        third = consumer.run(skill, agdata(result=first)).wait()
    events = {event["request_id"]: event for event in dependency_events(producer._orchestrator)}
    second_id = object.__getattribute__(second, "_execution_id")
    third_id = object.__getattribute__(third, "_execution_id")
    assert first_id not in events
    assert events[second_id]["dependencies"] == [
        {
            "request_id": first_id,
            "agent_id": str(producer.agname),
            "kind": "context",
            "evidence": "context_chain",
        }
    ]
    assert events[third_id]["dependencies"] == [
        {
            "request_id": first_id,
            "agent_id": str(producer.agname),
            "kind": "input",
            "evidence": "result_handle",
        }
    ]
    request_spans = {
        record[6]["request_id"]: record
        for record in agprof.profile_records()
        if record[1].startswith("run")
    }
    assert request_spans[second_id][6]["dependency_request_ids"] == [first_id]
    assert request_spans[third_id][6]["dependency_agent_ids"] == [str(producer.agname)]


def test_parallel_workers_have_no_edges_but_join_records_each_source(monkeypatch, tmp_path):
    started = [threading.Event() for _ in range(3)]
    release = [threading.Event() for _ in range(3)]
    join_started = threading.Event()

    def execute(self, *, skill_input, **kwargs):
        if "worker" in skill_input._data:
            worker = skill_input.worker
            started[worker].set()
            assert release[worker].wait(3)
        else:
            join_started.set()
        return agdata(ok=True)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    workers = [make_agent(tmp_path) for _ in range(3)]
    consumer = make_agent(tmp_path)
    skill = agskill("task", "")
    results = [worker.run(skill, agdata(worker=index)) for index, worker in enumerate(workers)]
    try:
        assert all(event.wait(2) for event in started)
        joined = consumer.run(skill, agdata(results=results))
        events = dependency_events(consumer._orchestrator)
        assert len(events) == 1
        assert events[0]["request_id"] == object.__getattribute__(joined, "_execution_id")
        assert events[0]["dependency_request_ids"] == sorted(
            object.__getattribute__(result, "_execution_id") for result in results
        )
        assert not join_started.is_set()
        release[0].set()
        release[2].set()
        results[0].wait(2)
        results[2].wait(2)
        assert not join_started.is_set()
        release[1].set()
        assert joined.wait(2).ok
        consumer._orchestrator.data_logger.flush()
        with sqlite3.connect(consumer._orchestrator.data_logger.db_path) as connection:
            blocked = [
                json.loads(row[0])
                for row in connection.execute(
                    "SELECT payload FROM events WHERE type='request_blocked'"
                )
            ]
        assert blocked
        assert all(
            "dependencies" not in payload and "dependency_request_ids" not in payload
            for payload in blocked
        )
        assert len(dependency_events(consumer._orchestrator)) == 1
    finally:
        for event in release:
            event.set()


def test_wait_all_carries_every_producer_and_consumer_execution(tmp_path):
    futures = [Future(), Future(), Future()]
    handles = [agdata(_future=future) for future in futures]
    for index, (future, handle) in enumerate(zip(futures, handles)):
        object.__setattr__(handle, "_execution_id", f"run{index}")
        object.__setattr__(handle, "_execution_agent", SimpleNamespace(agname=f"worker{index}"))
        future.set_result(agdata(ok=True))
    with agprof.session(tmp_path, sample_hz=0, auto_functions=False):
        with agprof.execution_context({"agency.run_id": "run-join", "agency.agent_id": "parent"}):
            agdata.wait_all(handles)
    record = next(record for record in agprof.profile_records() if record[1] == "sync:wait_all")
    assert record[6]["request_id"] == "run-join"
    assert record[6]["dependency_request_ids"] == ["run0", "run1", "run2"]
    assert record[6]["dependency_agent_ids"] == ["worker0", "worker1", "worker2"]
    assert not any("_execution_id" in handle.to_dict() for handle in handles)


def test_delegation_records_runtime_caller_without_a_completion_dependency(monkeypatch, tmp_path):
    parent, child = make_agent(tmp_path), make_agent(tmp_path)
    child_results = []
    skill = agskill("task", "")

    def execute(self, **kwargs):
        if self._agent is parent:
            child_results.append(child.run(skill, agdata()))
        return agdata(ok=True)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    with agprof.session(tmp_path / "profile", sample_hz=0, auto_functions=False):
        parent_result = parent.run(skill, agdata()).wait()
        child_results[0].wait()
    child_id = object.__getattribute__(child_results[0], "_execution_id")
    child_span = next(
        record
        for record in agprof.profile_records()
        if record[1].startswith("run") and record[6]["request_id"] == child_id
    )
    assert child_span[6]["submitted_by_request_id"] == object.__getattribute__(
        parent_result, "_execution_id"
    )
    assert child_span[6]["submitted_by_agent_id"] == str(parent.agname)
    assert dependency_events(parent._orchestrator) == []


def test_unidentified_wait_is_disclosed_without_a_fabricated_source(tmp_path):
    future = Future()
    unknown = agdata(_future=future)
    future.set_result(agdata(ok=True))
    with agprof.session(tmp_path, sample_hz=0, auto_functions=False):
        agdata.wait_all([unknown])
    record = next(record for record in agprof.profile_records() if record[1] == "sync:wait_all")
    assert record[6]["dependency_request_ids"] == []
    assert record[6]["unidentified_dependency_count"] == 1


def test_llm_request_ownership_survives_an_unrelated_server_thread(tmp_path):
    from agency.engine.host_servers.llm_handler_server import LlmHandlerServer
    from agency.llm.usage_tracker import LlmUsageTracker

    class Backend:
        model = "test"

        def output_token_cap(self):
            return None

        def dispatch(self, request):
            return {
                "message": {
                    "role": "assistant",
                    "blocks": [{"type": "text", "index": 0, "text": "done"}],
                },
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                "stop_reason": "stop",
            }

    with agprof.session(tmp_path, sample_hz=0, auto_functions=False):
        with agprof.execution_context({"agency.agent_id": "worker", "agency.run_id": "run4"}):
            server = LlmHandlerServer(
                agconfig(
                    llmconfig(
                        api_key="test",
                        model="m",
                        base_url="http://x/v1",
                        unusable_response_backoff_s=0,
                    )
                ),
                MagicMock(),
                LlmUsageTracker(),
                request_id="run4",
            )
            server._backend = Backend()
        responses = []
        thread = threading.Thread(
            target=lambda: responses.append(
                server.dispatch(
                    {
                        "messages": [
                            {"role": "user", "blocks": [{"type": "text", "index": 0, "text": "go"}]}
                        ],
                    }
                )
            )
        )
        thread.start()
        thread.join(3)
        assert not thread.is_alive() and len(responses) == 1
    record = next(record for record in agprof.profile_records() if record[1] == "llm:attempt[0]")
    assert record[6]["request_id"] == "run4"
    assert record[6]["agency.agent_id"] == "worker"
    assert record[6]["agency.run_id"] == "run4"


@pytest.mark.skipif(os.name != "posix", reason="Linux subprocess identity")
@pytest.mark.parametrize(
    "command,timeout,exit_code",
    [
        ("printf evidence; exit 0", 2, 0),
        ("exit 7", 2, 7),
        ("sleep 2", 0.02, None),
    ],
)
def test_native_subprocess_records_explicit_tool_owner_and_actual_identity(
    command, timeout, exit_code
):
    payloads = []
    bridge = SimpleNamespace(
        profiler_settings=lambda: {"enabled": True, "automatic": None},
        record_profiler_span=lambda payload: payloads.append(payload) or {"ok": True},
    )
    with NativeProfiler(bridge):
        with tool_execution(bridge, "admission-id", "model-call-id"):
            if exit_code is None:
                with pytest.raises(subprocess.TimeoutExpired):
                    run_process(
                        ["bash", "-c", command], capture_output=True, text=True, timeout=timeout
                    )
            else:
                result = run_process(
                    ["bash", "-c", command], capture_output=True, text=True, timeout=timeout
                )
                assert result.returncode == exit_code
    opened, completed = payloads
    attrs = completed["attributes"]
    assert attrs["tool_call_id"] == "admission-id"
    assert attrs["model_tool_call_id"] == "model-call-id"
    assert attrs["agency.namespace_pid"] != os.getpid()
    assert attrs["agency.process_start_ticks"] > 0
    assert attrs["agency.pid_namespace"] == os.readlink("/proc/self/ns/pid")
    assert attrs["namespace_ppid"] == os.getpid()
    assert opened["span_id"] == completed["span_id"]
    assert completed["end_ts"] >= completed["start_ts"]
    assert command not in json.dumps(payloads)


def test_resource_process_identity_and_ancestry_are_reuse_safe(monkeypatch, tmp_path):
    proc_root = tmp_path / "proc"

    def proc(pid, start_ticks, parent):
        directory = proc_root / str(pid)
        (directory / "ns").mkdir(parents=True, exist_ok=True)
        fields = ["R"] + ["0"] * 21
        fields[1], fields[19:22] = str(parent), [str(start_ticks), "1048576", "2"]
        (directory / "stat").write_text(f"{pid} (process) " + " ".join(fields))
        (directory / "status").write_text(f"NSpid:\t{pid}\n")
        namespace = directory / "ns/pid"
        if not namespace.exists():
            try:
                namespace.symlink_to("pid:[123]")
            except FileExistsError:
                pass
        (directory / "schedstat").write_text("100 200 1")

    proc(8, 10, 1)
    proc(9, 20, 8)
    monkeypatch.setattr(agprof, "_process_info", {})
    monkeypatch.setattr(agprof, "_samples", [])
    sampler = agprof._Sampler.__new__(agprof._Sampler)
    sampler._proc_root, sampler._clock_ticks, sampler._page_mb = proc_root, 100, 4096 / 2**20
    sampler._sample_process(1000, 9, "/no-cgroup")
    assert agprof._process_info["9-20"]["parent_identity"] == "8-10"
    proc(9, 30, 8)
    sampler._sample_process(2000, 9, "/no-cgroup")
    assert set(agprof._process_info) == {"9-20", "9-30"}
    assert agprof._process_info["9-20"]["last_seen_ns"] == 1000
    assert agprof._process_info["9-30"]["parent_identity"] == "8-10"
    assert agprof._process_info["9-20"]["trace_pid"] != agprof._process_info["9-30"]["trace_pid"]


def test_saved_real_profiler_session_preserves_fan_in_and_model_tool_evidence(
    monkeypatch, tmp_path
):
    from agency.observability.agwebui.critical_path import build_profile
    from agency.observability.agwebui.critical_path_store import load_compact_run

    started, release = [threading.Event() for _ in range(3)], [threading.Event() for _ in range(3)]

    def model(label, iteration, messages, response):
        with agprof.span("llm:attempt[0]"):
            agprof.annotate(
                call_label=f"exchange-{label}-{iteration}",
                model="recorded-test-model",
                input_tokens=10,
                output_tokens=2,
                outcome="success",
                **agprof.detail_metadata("llm.messages", messages),
                **agprof.detail_metadata("llm.response", response),
            )
            time.sleep(0.002)

    def execute(self, *, skill_input, **kwargs):
        label = skill_input._data.get("worker", "join")
        identifier = f"model-tool-{label}"
        initial = [{"role": "user", "blocks": [{"type": "text", "text": "task", "index": 0}]}]
        response = {
            "role": "assistant",
            "blocks": [
                {"type": "tool_use", "id": identifier, "name": "Bash", "index": 0, "input": {}},
            ],
        }
        if label == "join":
            model(
                label,
                0,
                initial,
                {"role": "assistant", "blocks": [{"type": "text", "text": "done", "index": 0}]},
            )
        else:
            model(label, 0, initial, response)
            with agprof.span("tool:Bash"):
                agprof.annotate(
                    call_id=f"admission-{label}", model_tool_call_id=identifier, outcome="success"
                )
                started[label].set()
                assert release[label].wait(3)
            model(
                label,
                1,
                [
                    *initial,
                    response,
                    {
                        "role": "user",
                        "blocks": [
                            {
                                "type": "tool_result",
                                "tool_call_id": identifier,
                                "index": 0,
                                "content": "done",
                            },
                        ],
                    },
                ],
                {"role": "assistant", "blocks": [{"type": "text", "text": "done", "index": 0}]},
            )
        return agdata(ok=True)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    workers = [make_agent(tmp_path / "logs") for _ in range(3)]
    join = make_agent(tmp_path / "logs")
    launch = workers[0]._orchestrator._launch_request_locked

    def launch_with_observed_startup(request):
        # The blocking first submit keeps the caller from launching B/C.
        # This reproduces real cold ThreadPoolExecutor/engine startup without
        # treating worker A's completion as a prerequisite of worker B.
        if request.agent is workers[0]:
            time.sleep(0.03)
        return launch(request)

    monkeypatch.setattr(
        workers[0]._orchestrator, "_launch_request_locked", launch_with_observed_startup
    )
    profile_path = tmp_path / "profiler"
    skill = agskill("task", "")
    with agprof.session(profile_path, sample_hz=0, auto_functions=False):
        results = [worker.run(skill, agdata(worker=index)) for index, worker in enumerate(workers)]
        try:
            assert all(event.wait(2) for event in started)
            joined = join.run(skill, agdata(results=results))
            release[0].set()
            release[2].set()
            results[0].wait(2)
            results[2].wait(2)
            # Preserve a visibly later branch in real recorded timestamps;
            # assertions concern topology and bounds rather than a clock exact value.
            time.sleep(0.01)
            release[1].set()
            joined.wait(2)
        finally:
            for event in release:
                event.set()
    run = load_compact_run(
        profile_path / "agprof.trace.json", {"id": "runtime-fan-in", "source": "recorded"}
    )
    profile = build_profile(run)
    join_id = object.__getattribute__(joined, "_execution_id")
    worker_ids = {object.__getattribute__(result, "_execution_id") for result in results}
    join_spans = [
        action for action in run["actions"] if action["metadata"].get("request_id") == join_id
    ]
    assert any(
        set(action["metadata"].get("dependency_request_ids", [])) == worker_ids
        for action in join_spans
    )
    assert profile["coverage"]["cross_agent_dependencies"] >= 3
    assert not profile["coverage"]["malformed_edges"]
    assert profile["summary"]["critical_path_duration"] <= profile["summary"]["wall_time"] + 1e-6
    assert profile["summary"]["aggregate_work"] > profile["summary"]["critical_path_measured"]
    assert any(edge["kind"] == "issued_tool" for edge in profile["edges"])
    assert any(edge["kind"] == "submission_sequence" for edge in profile["edges"])
    critical_tools = [
        operation
        for operation in profile["operations"]
        if operation["critical"] and operation["label"] == "Bash"
    ]
    assert critical_tools and all(
        operation["agent"] == str(workers[1].agname) for operation in critical_tools
    )


def test_submission_predecessors_do_not_cross_profile_sessions_or_caller_requests(
    monkeypatch, tmp_path
):
    parent, child = make_agent(tmp_path / "logs"), make_agent(tmp_path / "logs")
    skill = agskill("task", "")
    child_results = []

    def execute(self, **kwargs):
        if self._agent is parent:
            child_results.append(child.run(skill, agdata()))
        return agdata(ok=True)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    for recording in ("one", "two"):
        with agprof.session(tmp_path / recording, sample_hz=0, auto_functions=False):
            parent.run(skill, agdata()).wait(2)
            child_results[-1].wait(2)
        submissions = [
            record[6] for record in agprof.profile_records() if record[1] == "sync:submission"
        ]
        assert len(submissions) == 2
        # Separate caller contexts are both roots even when a worker thread
        # or the workflow caller is reused for another recording.
        assert all(not metadata.get("preceding_submission_span_id") for metadata in submissions)
        envelopes = [
            record[6] for record in agprof.profile_records() if record[1].startswith("run")
        ]
        submission_ids = {
            record[7] for record in agprof.profile_records() if record[1] == "sync:submission"
        }
        assert all(
            int(metadata["launcher_span_id"], 16) in submission_ids for metadata in envelopes
        )


def test_reused_host_service_thread_scopes_launcher_identity_to_recorded_request(
    monkeypatch, tmp_path
):
    worker = make_agent(tmp_path / "logs")
    monkeypatch.setattr(AgentEngine, "execute", lambda self, **kwargs: agdata(ok=True))
    skill = agskill("task", "")
    with agprof.session(tmp_path / "profile", sample_hz=0, auto_functions=False):
        with agprof.execution_context({"agency.run_id": "service-a", "agency.agent_id": "a"}):
            worker.run(skill, agdata()).wait(2)
        with agprof.execution_context({"agency.run_id": "service-b", "agency.agent_id": "b"}):
            worker.run(skill, agdata()).wait(2)
            worker.run(skill, agdata()).wait(2)
    submissions = sorted(
        [record for record in agprof.profile_records() if record[1] == "sync:submission"],
        key=lambda record: record[2],
    )
    assert not submissions[0][6].get("preceding_submission_span_id")
    assert not submissions[1][6].get("preceding_submission_span_id")
    assert submissions[2][6]["preceding_submission_span_id"] == f"{submissions[1][7]:016x}"
    requests = sorted(
        [record for record in agprof.profile_records() if record[1].startswith("run")],
        key=lambda record: record[2],
    )
    assert [record[6]["submitted_by_request_id"] for record in requests] == [
        "service-a",
        "service-b",
        "service-b",
    ]


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux /proc measurements")
def test_native_cpu_subprocess_resources_survive_saved_profiler_reload(tmp_path):
    from agency.agpolicy import agpolicy
    from agency.engine.host_servers.host_interaction_server import HostInteractionServer
    from agency.observability.agwebui.critical_path import build_profile
    from agency.observability.agwebui.critical_path_store import load_compact_run

    sampling_threads, stop = [], threading.Event()
    with agprof.session(tmp_path, sample_hz=0, auto_functions=False):
        sampler = agprof._Sampler.__new__(agprof._Sampler)
        sampler._proc_root = Path("/proc")
        sampler._clock_ticks, sampler._page_mb = (
            os.sysconf("SC_CLK_TCK"),
            os.sysconf("SC_PAGE_SIZE") / 2**20,
        )
        sampler._process_scope = lambda _: "test-sandbox"
        with agprof.execution_context({"agency.agent_id": "cpu-worker", "agency.run_id": "run0"}):
            server = HostInteractionServer(
                SimpleNamespace(policy=agpolicy()),
                MagicMock(),
                "cpu-worker",
                profile_attributes={
                    "agency.agent_id": "cpu-worker",
                    "agency.sandbox_id": "test-sandbox",
                    "request_id": "run0",
                },
            )

            def report(payload):
                server.record_span(
                    payload["name"],
                    payload.get("start_ts"),
                    payload.get("end_ts"),
                    payload["attributes"],
                    parent=payload.get("parent"),
                    span_id=payload["span_id"],
                )
                if payload["name"].startswith("process:") and payload.get("end_ts") is None:
                    pid = payload["attributes"]["agency.namespace_pid"]
                    sampler._sample_process(time.perf_counter_ns(), pid, "/test-scope")

                    def sample():
                        while not stop.wait(0.005):
                            sampler._sample_process(time.perf_counter_ns(), pid, "/test-scope")

                    thread = threading.Thread(target=sample)
                    sampling_threads.append(thread)
                    thread.start()
                return {"ok": True}

            bridge = SimpleNamespace(
                profiler_settings=lambda: {"enabled": True, "automatic": None},
                record_profiler_span=report,
            )
            with NativeProfiler(bridge) as native:
                with native.span("turn0"):
                    call_id = server.admit_tool_call("Bash", {}, model_tool_call_id="model-cpu")[
                        "call_id"
                    ]
                    start_perf, start_wall = time.perf_counter_ns(), time.time_ns()
                    try:
                        with tool_execution(bridge, call_id, "model-cpu"):
                            result = run_process(
                                [
                                    sys.executable,
                                    "-c",
                                    "import time\nend=time.perf_counter()+0.08\nwhile time.perf_counter()<end: pass\n",
                                ],
                                capture_output=True,
                                text=True,
                                timeout=3,
                            )
                        assert result.returncode == 0
                    finally:
                        stop.set()
                        for thread in sampling_threads:
                            thread.join(2)
                    server.complete_tool_call(
                        call_id,
                        result="completed",
                        duration_ns=time.perf_counter_ns() - start_perf,
                        started_wall_ns=start_wall,
                    )
    run = load_compact_run(
        tmp_path / "agprof.trace.json", {"id": "native-cpu", "source": "recorded"}
    )
    profile = build_profile(run)
    tool = next(operation for operation in profile["operations"] if operation["label"] == "Bash")
    process = next(
        process for process in profile["processes"] if tool["id"] in process["owner_ids"]
    )
    assert process["agent"] == "cpu-worker"
    assert process["attribution"] == "explicit_tool_owner"
    assert process["metrics"]["cpu_time_seconds"] > 0
    assert process["metrics"]["rss_peak_mb"] > 0
    assert tool["metrics"]["process_resources"][0]["sampled_window"]
    assert profile["summary"]["critical_path_duration"] <= profile["summary"]["wall_time"] + 1e-6
    assert process["duration"] <= tool["duration"] + 0.01


def test_real_native_loop_records_serial_tools_and_resets_batch_at_next_model_exchange(tmp_path):
    from agency.agpolicy import agpolicy
    from agency.engine.host_servers.host_interaction_server import HostInteractionServer
    from agency.native_harness.react_loop import run_react_loop
    from agency.observability.agwebui.critical_path import build_profile
    from agency.observability.agwebui.critical_path_store import load_compact_run

    def tool(name, call_id):
        return {"id": call_id, "function": {"name": name, "arguments": "{}"}}

    responses = [
        {
            "message": {
                "role": "assistant",
                "tool_calls": [tool("first", "reused"), tool("second", "second")],
            }
        },
        {"message": {"role": "assistant", "tool_calls": [tool("first", "reused")]}},
        {"message": {"role": "assistant", "content": "done"}},
    ]
    schemas = [
        {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
        for name in ("first", "second")
    ]
    with agprof.session(tmp_path, sample_hz=0, auto_functions=False):
        server = HostInteractionServer(
            SimpleNamespace(policy=agpolicy()),
            MagicMock(),
            "native-worker",
            profile_attributes={"agency.agent_id": "native-worker", "request_id": "run0"},
        )

        def report(payload):
            server.record_span(
                payload["name"],
                payload.get("start_ts"),
                payload.get("end_ts"),
                payload["attributes"],
                parent=payload.get("parent"),
                span_id=payload["span_id"],
            )
            return {"ok": True}

        def admit(name, arguments, **kwargs):
            decision = server.admit_tool_call(name, arguments, **kwargs)
            return {**decision, "decision": "allow" if decision["allowed"] else "deny"}

        bridge = SimpleNamespace(
            token="test",
            profiler_settings=lambda: {"enabled": True, "automatic": None},
            record_profiler_span=report,
            check_tool_policy=admit,
            complete_tool_policy=server.complete_tool_call,
        )
        with NativeProfiler(bridge):
            result = run_react_loop(
                [{"role": "user", "content": "go"}],
                "test",
                SimpleNamespace(dispatch=lambda *args: responses.pop(0)),
                bridge=bridge,
                toolset=(
                    schemas,
                    {"first": lambda _: '{"ok":true}', "second": lambda _: '{"ok":true}'},
                ),
                offload_dir=str(tmp_path / "offload"),
            )
            assert result.status == "done"
    tools = sorted(
        [record for record in agprof.profile_records() if record[1].startswith("tool:")],
        key=lambda record: record[2],
    )
    assert len(tools) == 3
    first, second, following_exchange = [record[6] for record in tools]
    assert "preceding_tool_call_id" not in first
    assert second["preceding_tool_call_id"] == first["call_id"]
    assert second["preceding_model_tool_call_id"] == "reused"
    assert second["tool_batch_id"] == first["tool_batch_id"]
    assert following_exchange["tool_batch_id"] != first["tool_batch_id"]
    assert "preceding_tool_call_id" not in following_exchange
    run = load_compact_run(
        tmp_path / "agprof.trace.json", {"id": "native-serial", "source": "recorded"}
    )
    profile = build_profile(run)
    links = [edge for edge in profile["edges"] if edge["kind"] == "serial_tool"]
    assert len(links) == 1 and links[0]["blocking"]
