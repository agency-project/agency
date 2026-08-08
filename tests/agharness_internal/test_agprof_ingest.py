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
