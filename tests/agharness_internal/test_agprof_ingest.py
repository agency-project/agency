from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from agency.agharness_internal.agllm_terminus import agLLMTerminus
from agency.agharness_internal.agprof_ingest import agProfilerIngest
from agency.agharness_internal import agprof_ingest
from agency.profiler import agprof

from .test_agllm_terminus import _completion


def test_registry_parents_terminus_and_derived_spans_without_exporting_token(monkeypatch, tmp_path):
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    registry = agProfilerIngest()
    monkeypatch.setattr(agprof_ingest, "_shared_profiler_ingest", registry)
    token = "credential-that-must-not-be-exported"

    term = agLLMTerminus()
    fake_client = SimpleNamespace()
    fake_client.chat = SimpleNamespace()
    fake_client.chat.completions = SimpleNamespace(
        create=lambda **kwargs: _completion(content="ok")
    )
    fake_client.close = lambda: None
    backend = SimpleNamespace(make_client=lambda timeout: fake_client, model="m")
    parent_agent = SimpleNamespace(agname="parent")
    ag = SimpleNamespace(
        agname="child",
        _parent_agent_id=parent_agent.agname,
        llm=SimpleNamespace(backend=backend),
    )
    term.register(token, ag)

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        with agprof.span("run7:skill:child"):
            agprof.annotate(
                **{
                    "agency.run_id": "run7",
                    "agency.agent_id": "child",
                    "agency.parent_agent_id": "parent",
                }
            )
            registry.register(token, ag)
            response = TestClient(term._app).post(
                "/internal/dispatch",
                json={
                    "token": token,
                    "kwargs": {"model": "m", "messages": [], "stream": False},
                },
            )
            assert response.status_code == 200
            registry.unregister(token)

    records = {record[1]: record for record in agprof._records}
    run_record = records["run7:skill:child"]
    attempt_record = records["llm:attempt[0]"]
    turn_record = records["turn0"]
    assert attempt_record[8] == run_record[7]
    assert turn_record[8] == run_record[7]
    for record in (run_record, attempt_record, turn_record):
        assert record[6]["agency.run_id"] == "run7"
        assert record[6]["agency.agent_id"] == "child"
        assert record[6]["agency.parent_agent_id"] == "parent"
    assert token not in json.dumps(agprof.summary_metrics())


def test_terminus_drain_waits_for_active_stream_finalizer():
    term = agLLMTerminus()
    drained = threading.Event()
    term._stream_started()

    waiter = threading.Thread(target=lambda: (term.drain(), drained.set()), daemon=True)
    waiter.start()
    time.sleep(0.02)
    assert not drained.is_set()

    term._stream_finished()

    assert drained.wait(timeout=1)
    waiter.join(timeout=1)


def test_native_events_mint_exact_container_spans_under_registered_run(monkeypatch, tmp_path):
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    registry = agProfilerIngest()
    ag = SimpleNamespace(agname="native-child", _parent_agent_id="parent")
    token = "native-emitter-credential"

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        with agprof.span("run3:native:child"):
            agprof.annotate(
                **{
                    "agency.run_id": "run3",
                    "agency.agent_id": "native-child",
                    "agency.parent_agent_id": "parent",
                }
            )
            registry.register(token, ag, exact_events=True)
            start_wall = time.time_ns()
            start_perf = time.perf_counter_ns()
            assert registry._handle_event(
                {
                    "token": token,
                    "ev": "span_start",
                    "span_id": "turn:0",
                    "name": "turn0",
                    "wall_ns": start_wall,
                    "perf_ns": start_perf,
                    "metadata": {"turn_index": 0},
                }
            )["ok"]
            assert registry._handle_event(
                {
                    "token": token,
                    "ev": "span_end",
                    "span_id": "turn:0",
                    "wall_ns": start_wall + 2_000_000,
                    "perf_ns": start_perf + 2_000_000,
                    "metadata": {"outcome": "success"},
                }
            )["ok"]
            assert registry._handle_event(
                {
                    "token": token,
                    "ev": "span_start",
                    "span_id": "tool:tc_1",
                    "name": "tool:bash",
                    "wall_ns": start_wall + 500_000,
                    "perf_ns": start_perf + 500_000,
                    "metadata": {
                        "tool_call_id": "tc_1",
                        "arguments": '{"command":"pwd"}',
                    },
                }
            )["ok"]
            assert registry._handle_event(
                {
                    "token": token,
                    "ev": "span_end",
                    "span_id": "tool:tc_1",
                    "wall_ns": start_wall + 1_500_000,
                    "perf_ns": start_perf + 1_500_000,
                    "metadata": {"outcome": "success", "result": "/workspace"},
                }
            )["ok"]
            registry.unregister(token)

    records = {record[1]: record for record in agprof._records}
    run_record = records["run3:native:child"]
    for name in ("turn0", "tool:bash"):
        record = records[name]
        assert record[8] == run_record[7]
        assert record[6]["timing"] == "exact"
        assert record[6]["provenance"] == "container_asserted"
        assert record[6]["agency.run_id"] == "run3"
        assert token not in json.dumps(record)
    assert records["tool:bash"][6]["result"] == "/workspace"


def test_exact_event_registration_disables_transcript_fallback():
    registry = agProfilerIngest()
    ag = SimpleNamespace(agname="native", _parent_agent_id=None)
    registry.register("exact", ag, exact_events=True)
    registry.register("derived", ag)

    assert registry.has_exact_events("exact") is True
    assert registry.has_exact_events("derived") is False
    assert registry.has_exact_events("missing") is False
