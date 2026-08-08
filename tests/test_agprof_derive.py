"""Tests for agprof_derive -- turning a terminus token's sequence of
dispatch transcripts into turn{i}/tool:{name} profiler spans (M3,
docs/Design_profiler_harness_integration.md §5.2)."""

from __future__ import annotations


from agency.profiler import agprof, agprof_derive


def _labels(records) -> "list[str]":
    return [record[1] for record in records]


def _by_label(records) -> "dict[str, tuple]":
    return {record[1]: record for record in records}


def test_on_dispatch_is_a_noop_when_profiling_is_off():
    before = len(agprof._records)
    agprof_derive.on_dispatch(
        "tok-off",
        [{"role": "user", "content": "hi"}],
        {"role": "assistant", "content": "hi back"},
        start_perf_ns=0,
        start_wall_ns=0,
        end_perf_ns=100,
        end_wall_ns=100,
    )
    assert len(agprof._records) == before


def test_first_dispatch_emits_turn0_with_no_tool_calls(monkeypatch, tmp_path):
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        agprof_derive.on_dispatch(
            "tok-first",
            [{"role": "user", "content": "hi"}],
            {"role": "assistant", "content": "hi back"},
            start_perf_ns=1_000,
            start_wall_ns=1_000,
            end_perf_ns=2_000,
            end_wall_ns=2_000,
        )
    assert _labels(agprof._records) == ["turn0"]
    turn0 = _by_label(agprof._records)["turn0"]
    assert turn0[2] == 1_000  # start_perf_ns
    assert turn0[3] == 1_000  # duration
    assert turn0[4] is None  # cpu -- nothing measured live for a derived span
    assert turn0[5] is None  # runq
    assert turn0[6]["timing"] == "derived"
    assert turn0[6]["outcome"] == "success"


def test_second_dispatch_derives_tool_span_from_message_diff(monkeypatch, tmp_path):
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    response_0 = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        ],
    }
    request_0 = [{"role": "user", "content": "run ls"}]
    # What every harness resends: everything from dispatch 0's transcript
    # (request + its own response) plus the tool's result message.
    request_1 = request_0 + [
        response_0,
        {"role": "tool", "tool_call_id": "call_1", "content": "a\nb"},
    ]

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        agprof_derive.on_dispatch(
            "tok-diff",
            request_0,
            response_0,
            start_perf_ns=0,
            start_wall_ns=0,
            end_perf_ns=1_000,
            end_wall_ns=1_000,
        )
        agprof_derive.on_dispatch(
            "tok-diff",
            request_1,
            {"role": "assistant", "content": "done"},
            start_perf_ns=5_000,
            start_wall_ns=5_000,
            end_perf_ns=6_000,
            end_wall_ns=6_000,
        )

    records = _by_label(agprof._records)
    assert set(records) == {"turn0", "turn1", "tool:bash"}
    tool_span = records["tool:bash"]
    # Duration is the gap between dispatch 0 ending and dispatch 1 starting
    # (§5.2: "(dispatch_{n+1}.start - dispatch_n.end)"), not a real
    # measurement of the tool's own execution.
    assert tool_span[2] == 1_000  # starts where dispatch 0 ended
    assert tool_span[3] == 4_000  # 5_000 - 1_000
    assert tool_span[4] is None
    assert tool_span[6]["tool_call_id"] == "call_1"
    assert tool_span[6]["timing"] == "derived"
    assert records["turn1"][2] == 5_000
    assert records["turn1"][3] == 1_000


def test_parallel_tool_calls_each_get_the_full_gap_as_an_upper_bound(monkeypatch, tmp_path):
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    response_0 = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
            {"id": "call_2", "type": "function", "function": {"name": "read", "arguments": "{}"}},
        ],
    }
    request_0 = [{"role": "user", "content": "do two things"}]
    request_1 = request_0 + [
        response_0,
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        {"role": "tool", "tool_call_id": "call_2", "content": "ok"},
    ]

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        agprof_derive.on_dispatch(
            "tok-parallel",
            request_0,
            response_0,
            start_perf_ns=0,
            start_wall_ns=0,
            end_perf_ns=1_000,
            end_wall_ns=1_000,
        )
        agprof_derive.on_dispatch(
            "tok-parallel",
            request_1,
            {"role": "assistant", "content": "done"},
            start_perf_ns=3_000,
            start_wall_ns=3_000,
            end_perf_ns=4_000,
            end_wall_ns=4_000,
        )

    tool_records = [r for r in agprof._records if r[1].startswith("tool:")]
    assert {r[1] for r in tool_records} == {"tool:bash", "tool:read"}
    assert all(r[3] == 2_000 for r in tool_records)  # same gap, charged to both


def test_compaction_shrink_resets_baseline_without_crashing(monkeypatch, tmp_path):
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    response_0 = {
        "role": "assistant",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        ],
    }
    request_0 = [{"role": "user", "content": "a"}] * 5

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        agprof_derive.on_dispatch(
            "tok-compaction",
            request_0,
            response_0,
            start_perf_ns=0,
            start_wall_ns=0,
            end_perf_ns=1_000,
            end_wall_ns=1_000,
        )
        # A compacted follow-up: shorter than dispatch 0's own transcript,
        # e.g. after the harness's own context compaction.
        compacted_request = [{"role": "system", "content": "summary"}]
        agprof_derive.on_dispatch(
            "tok-compaction",
            compacted_request,
            {"role": "assistant", "content": "done"},
            start_perf_ns=5_000,
            start_wall_ns=5_000,
            end_perf_ns=6_000,
            end_wall_ns=6_000,
        )

    labels = _labels(agprof._records)
    assert labels == ["turn0", "turn1"]  # no spurious tool span from the shrink


def test_forget_resets_turn_index_for_a_reused_token(monkeypatch, tmp_path):
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    response = {"role": "assistant", "content": "hi"}
    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        agprof_derive.on_dispatch(
            "tok-forget",
            [],
            response,
            start_perf_ns=0,
            start_wall_ns=0,
            end_perf_ns=100,
            end_wall_ns=100,
        )
        agprof_derive.forget("tok-forget")
        agprof_derive.on_dispatch(
            "tok-forget",
            [],
            response,
            start_perf_ns=200,
            start_wall_ns=200,
            end_perf_ns=300,
            end_wall_ns=300,
        )

    assert _labels(agprof._records) == ["turn0", "turn0"]


def test_missing_or_unmatched_tool_call_id_falls_back_to_unknown(monkeypatch, tmp_path):
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    response_0 = {"role": "assistant", "content": None, "tool_calls": []}
    request_0 = [{"role": "user", "content": "hi"}]
    request_1 = request_0 + [
        response_0,
        {"role": "tool", "tool_call_id": "call_unmatched", "content": "?"},
    ]

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        agprof_derive.on_dispatch(
            "tok-unmatched",
            request_0,
            response_0,
            start_perf_ns=0,
            start_wall_ns=0,
            end_perf_ns=1_000,
            end_wall_ns=1_000,
        )
        agprof_derive.on_dispatch(
            "tok-unmatched",
            request_1,
            {"role": "assistant", "content": "done"},
            start_perf_ns=2_000,
            start_wall_ns=2_000,
            end_perf_ns=3_000,
            end_wall_ns=3_000,
        )

    assert "tool:unknown" in _labels(agprof._records)
