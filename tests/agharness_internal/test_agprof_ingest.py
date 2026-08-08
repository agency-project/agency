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
from agency.profiler import agprof_derive

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


def test_claude_hooks_keep_derived_turns_and_emit_one_exact_tool(monkeypatch, tmp_path):
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    registry = agProfilerIngest()
    ag = SimpleNamespace(agname="claude-child", _parent_agent_id="parent")
    token = "secret-claude-token"
    response_0 = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tool-1",
                "type": "function",
                "function": {"name": "Bash", "arguments": '{"command":"printf ok"}'},
            },
            {
                "id": "tool-without-hook",
                "type": "function",
                "function": {"name": "Read", "arguments": "{}"},
            },
        ],
    }
    request_0 = [{"role": "user", "content": "run it"}]
    request_1 = request_0 + [
        response_0,
        {"role": "tool", "tool_call_id": "tool-1", "content": "ok"},
        {"role": "tool", "tool_call_id": "tool-without-hook", "content": "fallback"},
    ]

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        with agprof.span("run4:claude_code:child"):
            registry.register(token, ag, exact_tool_events=True)
            dispatch_0_perf = time.perf_counter_ns()
            dispatch_0_wall = time.time_ns()
            agprof_derive.on_dispatch(
                token,
                request_0,
                response_0,
                start_perf_ns=dispatch_0_perf,
                start_wall_ns=dispatch_0_wall,
                end_perf_ns=dispatch_0_perf + 1,
                end_wall_ns=dispatch_0_wall + 1,
                parent_context=registry.context_for_token(token),
                skip_tool_call_ids=registry.exact_tool_call_ids(token),
            )
            assert registry.has_exact_tool_events(token) is False
            pre = {
                "token": token,
                "ev": "hook",
                "hook_event_name": "PreToolUse",
                "wall_ns": time.time_ns(),
                "perf_ns": time.perf_counter_ns(),
                "payload": {
                    "hook_event_name": "PreToolUse",
                    "tool_use_id": "tool-1",
                    "tool_name": "Bash",
                    "tool_input": {
                        "command": "printf ok",
                        "nested": {
                            "access_token": token,
                            "headers": {"Authorization": "Bearer other-secret"},
                        },
                    },
                },
            }
            assert registry._handle_event(pre)["ok"]
            assert registry.has_exact_tool_events(token) is True
            duplicate = registry._handle_event(pre)
            assert duplicate == {"ok": False, "error": "duplicate span_start"}
            post_wall_ns = time.time_ns()
            post_perf_ns = time.perf_counter_ns()
            duration_ms = 0.001
            assert registry._handle_event(
                {
                    **pre,
                    "hook_event_name": "PostToolUse",
                    "wall_ns": post_wall_ns,
                    "perf_ns": post_perf_ns,
                    "payload": {
                        **pre["payload"],
                        "hook_event_name": "PostToolUse",
                        "duration_ms": duration_ms,
                        "tool_response": {
                            "stdout": f"AGPROF_TOKEN={token}\nAuthorization: Bearer leaked-value"
                        },
                    },
                }
            )["ok"]
            incomplete_pre = {
                "token": token,
                "ev": "hook",
                "hook_event_name": "PreToolUse",
                "wall_ns": time.time_ns(),
                "perf_ns": time.perf_counter_ns(),
                "payload": {
                    "hook_event_name": "PreToolUse",
                    "tool_use_id": "tool-without-hook",
                    "tool_name": "Read",
                    "tool_input": {},
                },
            }
            assert registry._handle_event(incomplete_pre)["ok"]
            assert registry._handle_event(
                {
                    **incomplete_pre,
                    "hook_event_name": "PostToolUse",
                    "wall_ns": time.time_ns(),
                    "perf_ns": time.perf_counter_ns(),
                    "payload": {
                        **incomplete_pre["payload"],
                        "hook_event_name": "PostToolUse",
                        # A malformed supplied duration is rejected, leaving
                        # this ID eligible for transcript fallback.
                        "duration_ms": "not-a-duration",
                    },
                }
            ) == {"ok": False, "error": "missing or invalid duration_ms"}
            dispatch_1_perf = time.perf_counter_ns()
            dispatch_1_wall = time.time_ns()
            agprof_derive.on_dispatch(
                token,
                request_1,
                {"role": "assistant", "content": "done"},
                start_perf_ns=dispatch_1_perf,
                start_wall_ns=dispatch_1_wall,
                end_perf_ns=dispatch_1_perf + 1,
                end_wall_ns=dispatch_1_wall + 1,
                parent_context=registry.context_for_token(token),
                skip_tool_call_ids=registry.exact_tool_call_ids(token),
                before_derive_tools=lambda call_ids: registry.reconcile_derived_tool_call_ids(
                    token, call_ids
                ),
            )
            assert registry._handle_event(
                {
                    **incomplete_pre,
                    "hook_event_name": "PostToolUse",
                    "wall_ns": time.time_ns(),
                    "perf_ns": time.perf_counter_ns(),
                    "payload": {
                        **incomplete_pre["payload"],
                        "hook_event_name": "PostToolUse",
                        "duration_ms": 0,
                    },
                }
            ) == {
                "ok": False,
                "error": "span_end without matching span_start",
            }
            registry.unregister(token)

    labels = [record[1] for record in agprof._records]
    assert labels.count("turn0") == 1
    assert labels.count("turn1") == 1
    assert labels.count("tool:Bash") == 1
    assert labels.count("tool:Read") == 1
    tool_record = next(record for record in agprof._records if record[1] == "tool:Bash")
    run_record = next(record for record in agprof._records if record[1] == "run4:claude_code:child")
    assert tool_record[6]["timing"] == "exact"
    assert "[REDACTED]" in tool_record[6]["arguments"]
    assert "[REDACTED]" in tool_record[6]["result"]
    assert "other-secret" not in tool_record[6]["arguments"]
    assert "leaked-value" not in tool_record[6]["result"]
    assert tool_record[2] == post_perf_ns - 1_000
    assert tool_record[3] == 1_000
    assert tool_record[6]["duration_ms"] == duration_ms
    assert tool_record[6]["timing_source"] == "claude_duration_ms"
    assert tool_record[8] == run_record[7]
    assert tool_record[2] >= run_record[2]
    assert token not in json.dumps(tool_record)
    fallback_record = next(record for record in agprof._records if record[1] == "tool:Read")
    assert fallback_record[6]["timing"] == "derived"
    assert not any(span["label"] == "tool:Read" for span in agprof._interrupted_spans)


def test_hook_missing_id_and_duplicate_end_fail_without_cross_pairing():
    registry = agProfilerIngest()
    ag = SimpleNamespace(agname="claude", _parent_agent_id=None)
    registry.register("tok", ag, exact_tool_events=True)
    base = {
        "token": "tok",
        "ev": "hook",
        "hook_event_name": "PreToolUse",
        "wall_ns": 1,
        "perf_ns": 1,
        "payload": {"hook_event_name": "PreToolUse", "tool_name": "Read", "tool_input": {}},
    }
    assert registry._handle_event(base) == {"ok": False, "error": "missing tool_use_id"}
    start = {
        **base,
        "payload": {**base["payload"], "tool_use_id": "x"},
    }
    assert registry._handle_event(start)["ok"]
    end = {
        **base,
        "hook_event_name": "PostToolUse",
        "wall_ns": 2,
        "perf_ns": 2,
        "payload": {
            **base["payload"],
            "hook_event_name": "PostToolUse",
            "tool_use_id": "x",
            "duration_ms": 0.000001,
        },
    }
    assert registry._handle_event(end)["ok"]
    assert registry._handle_event(end) == {
        "ok": False,
        "error": "span_end without matching span_start",
    }
    registry.unregister("tok")


def test_exact_post_reservation_closes_stale_derived_snapshot_race(monkeypatch, tmp_path):
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    registry = agProfilerIngest()
    ag = SimpleNamespace(agname="claude", _parent_agent_id=None)
    token = "post-race-token"
    response_0 = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "racing-tool",
                "type": "function",
                "function": {"name": "Bash", "arguments": "{}"},
            }
        ],
    }
    request_0 = [{"role": "user", "content": "run"}]
    request_1 = request_0 + [
        response_0,
        {"role": "tool", "tool_call_id": "racing-tool", "content": "ok"},
    ]
    entered_end = threading.Event()
    release_end = threading.Event()
    post_result = []

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        with agprof.span("run0:claude_code:claude"):
            registry.register(token, ag, exact_tool_events=True)
            first_perf_ns = time.perf_counter_ns()
            first_wall_ns = time.time_ns()
            agprof_derive.on_dispatch(
                token,
                request_0,
                response_0,
                start_perf_ns=first_perf_ns,
                start_wall_ns=first_wall_ns,
                end_perf_ns=first_perf_ns + 1,
                end_wall_ns=first_wall_ns + 1,
                parent_context=registry.context_for_token(token),
            )
            pre = {
                "token": token,
                "ev": "hook",
                "hook_event_name": "PreToolUse",
                "wall_ns": time.time_ns(),
                "perf_ns": time.perf_counter_ns(),
                "payload": {
                    "hook_event_name": "PreToolUse",
                    "tool_use_id": "racing-tool",
                    "tool_name": "Bash",
                    "tool_input": {},
                },
            }
            assert registry._handle_event(pre)["ok"]
            stale_exact_ids = registry.exact_tool_call_ids(token)
            assert stale_exact_ids == set()

            handle = registry._open_remote_spans[
                (token, agprof_ingest._claude_tool_span_id("racing-tool"))
            ].handle
            handle_type = type(handle)
            real_end = handle_type.end

            def blocking_end(self, **kwargs):
                entered_end.set()
                assert release_end.wait(timeout=2)
                return real_end(self, **kwargs)

            monkeypatch.setattr(handle_type, "end", blocking_end)
            post = {
                **pre,
                "hook_event_name": "PostToolUse",
                "wall_ns": time.time_ns(),
                "perf_ns": time.perf_counter_ns(),
                "payload": {
                    **pre["payload"],
                    "hook_event_name": "PostToolUse",
                    "duration_ms": 0,
                },
            }
            post_thread = threading.Thread(
                target=lambda: post_result.append(registry._handle_event(post)),
                daemon=True,
            )
            post_thread.start()
            try:
                assert entered_end.wait(timeout=2)
                # Post has removed the Pre and is blocked before materializing
                # its record. The marker must already be visible here.
                assert registry.exact_tool_call_ids(token) == {"racing-tool"}
                second_perf_ns = time.perf_counter_ns()
                second_wall_ns = time.time_ns()
                agprof_derive.on_dispatch(
                    token,
                    request_1,
                    {"role": "assistant", "content": "done"},
                    start_perf_ns=second_perf_ns,
                    start_wall_ns=second_wall_ns,
                    end_perf_ns=second_perf_ns + 1,
                    end_wall_ns=second_wall_ns + 1,
                    parent_context=registry.context_for_token(token),
                    skip_tool_call_ids=stale_exact_ids,
                    before_derive_tools=lambda call_ids: registry.reconcile_derived_tool_call_ids(
                        token, call_ids
                    ),
                )
            finally:
                release_end.set()
                post_thread.join(timeout=2)
            assert not post_thread.is_alive()
            assert post_result == [{"ok": True}]
            registry.unregister(token)

    tool_records = [record for record in agprof._records if record[1] == "tool:Bash"]
    assert len(tool_records) == 1
    assert tool_records[0][6]["timing"] == "exact"


def test_reconciliation_grace_prefers_late_exact_post(monkeypatch, tmp_path):
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    registry = agProfilerIngest()
    ag = SimpleNamespace(agname="claude", _parent_agent_id=None)
    token = "late-post-token"
    post_result = []

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        with agprof.span("run0:claude_code:claude"):
            registry.register(token, ag, exact_tool_events=True)
            pre = {
                "token": token,
                "ev": "hook",
                "hook_event_name": "PreToolUse",
                "wall_ns": time.time_ns(),
                "perf_ns": time.perf_counter_ns(),
                "payload": {
                    "hook_event_name": "PreToolUse",
                    "tool_use_id": "late-tool",
                    "tool_name": "Read",
                    "tool_input": {},
                },
            }
            assert registry._handle_event(pre)["ok"]

            def finish_post():
                time.sleep(0.02)
                post_result.append(
                    registry._handle_event(
                        {
                            **pre,
                            "hook_event_name": "PostToolUse",
                            "wall_ns": time.time_ns(),
                            "perf_ns": time.perf_counter_ns(),
                            "payload": {
                                **pre["payload"],
                                "hook_event_name": "PostToolUse",
                                "duration_ms": 0,
                            },
                        }
                    )
                )

            post_thread = threading.Thread(target=finish_post, daemon=True)
            post_thread.start()
            derive_ids = registry.reconcile_derived_tool_call_ids(token, {"late-tool"})
            post_thread.join(timeout=1)
            assert not post_thread.is_alive()
            assert post_result == [{"ok": True}]
            assert derive_ids == set()
            registry.unregister(token)

    tool_records = [record for record in agprof._records if record[1] == "tool:Read"]
    assert len(tool_records) == 1
    assert tool_records[0][6]["timing"] == "exact"


def test_reconciliation_grace_keeps_missing_post_fallback_bounded(monkeypatch, tmp_path):
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    monkeypatch.setattr(agprof_ingest, "_EXACT_POST_GRACE_S", 0.02)
    registry = agProfilerIngest()
    ag = SimpleNamespace(agname="claude", _parent_agent_id=None)
    token = "missing-post-token"

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        with agprof.span("run0:claude_code:claude"):
            registry.register(token, ag, exact_tool_events=True)
            pre = {
                "token": token,
                "ev": "hook",
                "hook_event_name": "PreToolUse",
                "wall_ns": time.time_ns(),
                "perf_ns": time.perf_counter_ns(),
                "payload": {
                    "hook_event_name": "PreToolUse",
                    "tool_use_id": "missing-tool",
                    "tool_name": "Read",
                    "tool_input": {},
                },
            }
            assert registry._handle_event(pre)["ok"]
            started = time.monotonic()
            derive_ids = registry.reconcile_derived_tool_call_ids(token, {"missing-tool"})
            elapsed = time.monotonic() - started
            assert derive_ids == {"missing-tool"}
            assert 0.015 <= elapsed < 0.25
            registry.unregister(token)

    assert not any(span["label"] == "tool:Read" for span in agprof._interrupted_spans)


def test_missing_post_duration_closes_honest_hook_boundary_span(monkeypatch, tmp_path):
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    registry = agProfilerIngest()
    ag = SimpleNamespace(agname="claude", _parent_agent_id=None)
    token = "hook-boundary-token"
    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        with agprof.span("run0:claude_code:claude"):
            registry.register(token, ag, exact_tool_events=True)
            pre_perf_ns = time.perf_counter_ns()
            pre_wall_ns = time.time_ns()
            pre = {
                "token": token,
                "ev": "hook",
                "hook_event_name": "PreToolUse",
                "wall_ns": pre_wall_ns,
                "perf_ns": pre_perf_ns,
                "payload": {
                    "hook_event_name": "PreToolUse",
                    "tool_use_id": "optional-duration",
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/tmp/example"},
                },
            }
            assert registry._handle_event(pre)["ok"]
            post_perf_ns = time.perf_counter_ns()
            post_wall_ns = time.time_ns()
            assert registry._handle_event(
                {
                    **pre,
                    "hook_event_name": "PostToolUse",
                    "wall_ns": post_wall_ns,
                    "perf_ns": post_perf_ns,
                    "payload": {
                        **pre["payload"],
                        "hook_event_name": "PostToolUse",
                        "tool_response": "ok",
                    },
                }
            )["ok"]
            assert registry.exact_tool_call_ids(token) == {"optional-duration"}
            registry.unregister(token)

    tool_record = next(record for record in agprof._records if record[1] == "tool:Read")
    assert tool_record[2] == pre_perf_ns
    assert tool_record[3] == post_perf_ns - pre_perf_ns
    assert tool_record[6]["timing"] == "hook_boundary"
    assert tool_record[6]["timing_source"] == "pre_post_hooks"
    assert not any(span["label"] == "tool:Read" for span in agprof._interrupted_spans)


def test_invalid_post_duration_falls_back_and_unmatched_span_is_interrupted(monkeypatch, tmp_path):
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    registry = agProfilerIngest()
    ag = SimpleNamespace(agname="claude", _parent_agent_id=None)
    token = "unmatched-hook-token"
    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        with agprof.span("run0:claude_code:claude"):
            registry.register(token, ag, exact_tool_events=True)
            assert registry._handle_event(
                {
                    "token": token,
                    "ev": "hook",
                    "hook_event_name": "PreToolUse",
                    "wall_ns": time.time_ns(),
                    "perf_ns": time.perf_counter_ns(),
                    "payload": {
                        "hook_event_name": "PreToolUse",
                        "tool_use_id": "never-finished",
                        "tool_name": "Bash",
                        "tool_input": {"command": "sleep 10"},
                    },
                }
            )["ok"]
            assert registry._handle_event(
                {
                    "token": token,
                    "ev": "hook",
                    "hook_event_name": "PostToolUse",
                    "wall_ns": time.time_ns(),
                    "perf_ns": time.perf_counter_ns(),
                    "payload": {
                        "hook_event_name": "PostToolUse",
                        "tool_use_id": "never-finished",
                        "tool_name": "Bash",
                        "tool_response": "terminated before an authoritative duration",
                        "duration_ms": "invalid",
                    },
                }
            ) == {"ok": False, "error": "missing or invalid duration_ms"}
            assert registry._handle_event(
                {
                    "token": token,
                    "ev": "hook",
                    "hook_event_name": "PostToolUse",
                    "wall_ns": time.time_ns(),
                    "perf_ns": time.perf_counter_ns(),
                    "payload": {
                        "hook_event_name": "PostToolUse",
                        "tool_use_id": "never-finished",
                        "tool_name": "Bash",
                        "tool_response": "forged timing",
                        "duration_ms": 10**12,
                    },
                }
            ) == {
                "ok": False,
                "error": "duration_ms exceeds observed PreToolUse/PostToolUse interval",
            }
            assert registry.exact_tool_call_ids(token) == set()
            registry.unregister(token)

    interrupted = next(span for span in agprof._interrupted_spans if span["label"] == "tool:Bash")
    assert sum(span["label"] == "tool:Bash" for span in agprof._interrupted_spans) == 1
    assert interrupted["outcome"] == "interrupted"
    assert token not in json.dumps(interrupted)


def test_hook_tool_name_and_remote_span_name_are_redacted(monkeypatch, tmp_path):
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    registry = agProfilerIngest()
    ag = SimpleNamespace(agname="claude", _parent_agent_id=None)
    token = "registered-tool-name-secret"
    bearer_secret = "unregistered-bearer-secret"

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        with agprof.span("run0:claude_code:claude"):
            registry.register(token, ag, exact_tool_events=True)
            pre_wall_ns = time.time_ns()
            pre_perf_ns = time.perf_counter_ns()
            assert registry._handle_event(
                {
                    "token": token,
                    "ev": "hook",
                    "hook_event_name": "PreToolUse",
                    "wall_ns": pre_wall_ns,
                    "perf_ns": pre_perf_ns,
                    "payload": {
                        "hook_event_name": "PreToolUse",
                        "tool_use_id": "redacted-name",
                        "tool_name": token,
                        "tool_input": {},
                    },
                }
            )["ok"]
            assert registry._handle_event(
                {
                    "token": token,
                    "ev": "hook",
                    "hook_event_name": "PostToolUse",
                    "wall_ns": time.time_ns(),
                    "perf_ns": time.perf_counter_ns(),
                    "payload": {
                        "hook_event_name": "PostToolUse",
                        "tool_use_id": "redacted-name",
                        "duration_ms": 0,
                    },
                }
            )["ok"]

            raw_wall_ns = time.time_ns()
            raw_perf_ns = time.perf_counter_ns()
            assert registry._handle_event(
                {
                    "token": token,
                    "ev": "span_start",
                    "span_id": "redacted-raw-name",
                    "name": f"process:Bearer {bearer_secret}",
                    "wall_ns": raw_wall_ns,
                    "perf_ns": raw_perf_ns,
                }
            )["ok"]
            assert registry._handle_event(
                {
                    "token": token,
                    "ev": "span_end",
                    "span_id": "redacted-raw-name",
                    "wall_ns": raw_wall_ns + 1,
                    "perf_ns": raw_perf_ns + 1,
                }
            )["ok"]
            registry.unregister(token)

    labels = [record[1] for record in agprof._records]
    assert "tool:[REDACTED]" in labels
    assert "process:Bearer [REDACTED]" in labels
    assert token not in json.dumps(labels)
    assert bearer_secret not in json.dumps(labels)


def test_unregister_race_cannot_resurrect_remote_span(monkeypatch):
    registry = agProfilerIngest()
    ag = SimpleNamespace(agname="race", _parent_agent_id=None)
    token = "teardown-race-token"
    registry.register(token, ag)
    real_bounded_metadata = agprof_ingest._bounded_metadata

    def unregister_while_parsing(value, *, secrets=()):
        registry.unregister(token)
        return real_bounded_metadata(value, secrets=secrets)

    monkeypatch.setattr(agprof_ingest, "_bounded_metadata", unregister_while_parsing)
    result = registry._handle_event(
        {
            "token": token,
            "ev": "span_start",
            "span_id": "racing-start",
            "name": "tool:Read",
            "wall_ns": time.time_ns(),
            "perf_ns": time.perf_counter_ns(),
        }
    )

    assert result == {"ok": False, "error": "unknown or missing token"}
    assert registry._open_remote_spans == {}
