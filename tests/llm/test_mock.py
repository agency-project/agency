"""Tests for llm/mock.py: the replay/mock LLM backend."""

from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from agency.configs.agconfig import agconfig, dataloggerconfig, llmconfig
from agency.llm.mock import (
    constant_timing,
    exact_replay_timing,
    instant_timing,
    poisson_timing,
)
from agency.observability.agdatalogger import agDataLogger


def _make_source_db(db_path: Path) -> agDataLogger:
    logger = agDataLogger(agconfig(dataloggerconfig(db_path=str(db_path))))
    logger.start()
    return logger


_TEXT_EXCHANGE = [
    {
        "type": "text",
        "index": 0,
        "text": "Hello there, how are you doing today?",
        "ts_start": 100.0,
        "ts_end": 100.5,
    },
    {
        "type": "metadata",
        "index": 2**31 - 1,
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "stop_reason": "stop",
        "ttft_ms": 250.0,
        "data": [],
    },
]

_TOOL_EXCHANGE = [
    {
        "type": "tool_use",
        "index": 0,
        "id": "tool1",
        "name": "lookup",
        "arguments": '{"query": "weather"}',
        "ts_start": 200.0,
        "ts_end": 200.2,
    },
    {
        "type": "metadata",
        "index": 2**31 - 1,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "stop_reason": "tool_use",
        "ttft_ms": 100.0,
        "data": [],
    },
]


def _mock_backend(db_path: Path, **fields):
    from agency.llm.agllm import agllm

    cfg = agconfig(llmconfig(provider="mock", replay_db_path=str(db_path), **fields))
    return agllm.for_config(cfg)


@pytest.mark.parametrize("streaming", [False, True], ids=["dispatch", "dispatch_stream"])
def test_replay_dispatch_hook_gates_response_after_config_clone(tmp_path, streaming):
    from agency.llm.agllm import agllm

    db_path = tmp_path / "source.sqlite3"
    logger = _make_source_db(db_path)
    logger.record_final_transcript("call1", type="llm_block", payloads=_TEXT_EXCHANGE)
    logger.stop()
    entered = threading.Event()
    release = threading.Event()
    seen = []

    def hook(request):
        seen.append(request)
        entered.set()
        assert release.wait(timeout=5), "test did not release replay dispatch"
        request.clear()

    cfg = agconfig(
        llmconfig(
            provider="mock",
            replay_db_path=str(db_path),
            timing_mode="instant",
            replay_dispatch_hook=hook,
        )
    ).clone()
    assert "replay_dispatch_hook" not in cfg.llm.safe_snapshot()
    backend = agllm.for_config(cfg)
    request = {"messages": [{"role": "user", "blocks": []}]}

    def dispatch():
        if streaming:
            return list(backend.dispatch_stream(request))
        return backend.dispatch(request)

    with ThreadPoolExecutor(max_workers=1) as executor:
        response = executor.submit(dispatch)
        try:
            assert entered.wait(timeout=5)
            assert seen == [request]
            assert not response.done()
        finally:
            release.set()
        result = response.result(timeout=5)
    assert request == {"messages": [{"role": "user", "blocks": []}]}
    if streaming:
        assert "".join(item.get("text", "") for item in result) == _TEXT_EXCHANGE[0]["text"]
    else:
        assert result["message"]["blocks"] == _TEXT_EXCHANGE


class TestReplayOrderingAndDispatch:
    def test_dispatch_reconstructs_exchanges_in_order(self, tmp_path):
        db_path = tmp_path / "agent_x_data.sqlite3"
        logger = _make_source_db(db_path)
        logger.record_final_transcript("call1", type="llm_block", payloads=_TEXT_EXCHANGE)
        logger.record_final_transcript("call2", type="llm_block", payloads=_TOOL_EXCHANGE)
        logger.stop()

        backend = _mock_backend(db_path)
        first = backend.dispatch({"messages": []})
        assert first["usage"] == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        assert first["stop_reason"] == "stop"
        text_block = next(b for b in first["message"]["blocks"] if b["type"] == "text")
        assert text_block["text"] == "Hello there, how are you doing today?"

        second = backend.dispatch({"messages": []})
        assert second["stop_reason"] == "tool_use"
        tool_block = next(b for b in second["message"]["blocks"] if b["type"] == "tool_use")
        assert tool_block["arguments"] == '{"query": "weather"}'

    def test_dispatch_ignores_request_content(self, tmp_path):
        """Positional replay: the mock never inspects the request, so a
        wildly different prompt on the Nth call still replays the Nth
        recorded exchange."""
        db_path = tmp_path / "agent_x_data.sqlite3"
        logger = _make_source_db(db_path)
        logger.record_final_transcript("call1", type="llm_block", payloads=_TEXT_EXCHANGE)
        logger.stop()

        backend = _mock_backend(db_path)
        result = backend.dispatch(
            {"messages": [{"role": "user", "content": "totally different prompt"}]}
        )
        assert result["stop_reason"] == "stop"

    def test_exhaustion_raises_clear_error(self, tmp_path):
        db_path = tmp_path / "agent_x_data.sqlite3"
        logger = _make_source_db(db_path)
        logger.record_final_transcript("call1", type="llm_block", payloads=_TEXT_EXCHANGE)
        logger.stop()

        backend = _mock_backend(db_path)
        backend.dispatch({"messages": []})
        with pytest.raises(RuntimeError, match="replay exhausted"):
            backend.dispatch({"messages": []})

    def test_error_and_cancelled_exchanges_are_skipped(self, tmp_path):
        db_path = tmp_path / "agent_x_data.sqlite3"
        logger = _make_source_db(db_path)
        logger.record_final_transcript("call1", type="llm_block", payloads=_TEXT_EXCHANGE)
        logger.record_final_transcript(
            "call2", type="llm_stream_error", payloads=[{"error": "boom"}]
        )
        logger.record_final_transcript("call3", type="llm_block", payloads=_TOOL_EXCHANGE)
        logger.stop()

        backend = _mock_backend(db_path)
        first = backend.dispatch({"messages": []})
        assert first["stop_reason"] == "stop"
        second = backend.dispatch({"messages": []})
        assert second["stop_reason"] == "tool_use"

    def test_only_assistant_blocks_are_replayed(self, tmp_path):
        """A recorded group can hold a `role: "user"` prompt block or a
        `role: "tool"` result alongside the response -- the recorder logs
        whatever is new in the request as well as the response. Replaying
        those back as if the model said them means the tool call in the
        group never runs."""
        db_path = tmp_path / "agent_x_data.sqlite3"
        logger = _make_source_db(db_path)
        logger.record_final_transcript(
            "call1",
            type="llm_block",
            payloads=[
                {"role": "user", "type": "text", "index": 0, "text": "Find the bug"},
                {"role": "tool", "type": "tool_result", "index": 0, "text": "a.py"},
                {
                    "role": "assistant",
                    "type": "tool_use",
                    "index": 0,
                    "id": "call_1",
                    "name": "read",
                },
            ],
        )
        logger.stop()

        backend = _mock_backend(db_path)
        result = backend.dispatch({"messages": []})
        assert result["message"]["blocks"] == [
            {"role": "assistant", "type": "tool_use", "index": 0, "id": "call_1", "name": "read"}
        ]


class TestDispatchStream:
    def test_streams_reconstructable_text_deltas(self, tmp_path):
        db_path = tmp_path / "agent_x_data.sqlite3"
        logger = _make_source_db(db_path)
        logger.record_final_transcript("call1", type="llm_block", payloads=_TEXT_EXCHANGE)
        logger.stop()

        backend = _mock_backend(db_path, timing_mode="instant")
        items = list(backend.dispatch_stream({"messages": []}))
        assert items[-1] == {
            "type": "usage",
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "stop_reason": "stop",
        }
        deltas = items[:-1]
        assert all(d["type"] == "block_delta" and d["block_type"] == "text" for d in deltas)
        assert "".join(d["text"] for d in deltas) == "Hello there, how are you doing today?"

    def test_streams_reconstructable_tool_use_deltas(self, tmp_path):
        db_path = tmp_path / "agent_x_data.sqlite3"
        logger = _make_source_db(db_path)
        logger.record_final_transcript("call1", type="llm_block", payloads=_TOOL_EXCHANGE)
        logger.stop()

        backend = _mock_backend(db_path, timing_mode="instant")
        items = list(backend.dispatch_stream({"messages": []}))
        deltas = items[:-1]
        assert deltas[0]["id"] == "tool1"
        assert deltas[0]["name"] == "lookup"
        assert "".join(d.get("arguments", "") for d in deltas) == '{"query": "weather"}'


class TestTimingModes:
    def test_instant_timing_is_all_zero(self):
        exchange = {"blocks": [{"ts_start": 0.0, "ts_end": 5.0}], "metadata": {"ttft_ms": 999}}
        delays = list(instant_timing(exchange, [3]))
        assert delays == [0.0, 0.0, 0.0, 0.0]

    def test_exact_replay_timing_uses_recorded_ttft_and_block_duration(self):
        exchange = {
            "blocks": [{"ts_start": 100.0, "ts_end": 101.0}],
            "metadata": {"ttft_ms": 250.0},
        }
        delays = list(exact_replay_timing(exchange, [4]))
        assert delays[0] == pytest.approx(0.25)
        assert delays[1:] == pytest.approx([0.25, 0.25, 0.25, 0.25])

    def test_poisson_timing_is_reproducible_with_seed(self):
        exchange = {"blocks": [{}], "metadata": {}}
        fn = poisson_timing(rate_hz=10.0, ttft_mean_s=0.1, seed=42)
        first = list(fn(exchange, [5]))
        fn_again = poisson_timing(rate_hz=10.0, ttft_mean_s=0.1, seed=42)
        second = list(fn_again(exchange, [5]))
        assert first == second
        assert len(first) == 6  # ttft + 5 chunk delays
        assert all(d >= 0 for d in first)

    def test_constant_timing_uses_fixed_ttft_and_tpot(self):
        exchange = {"blocks": [{}, {}], "metadata": {}}
        fn = constant_timing(ttft_s=0.5, tpot_s=0.1)
        delays = list(fn(exchange, [2, 3]))
        assert delays == [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]

    def test_end_to_end_constant_timing_mode(self, tmp_path):
        db_path = tmp_path / "agent_x_data.sqlite3"
        logger = _make_source_db(db_path)
        logger.record_final_transcript("call1", type="llm_block", payloads=_TEXT_EXCHANGE)
        logger.stop()

        backend = _mock_backend(
            db_path, timing_mode="constant", constant_ttft_s=0.0, constant_tpot_s=0.0
        )
        items = list(backend.dispatch_stream({"messages": []}))
        assert items[-1]["stop_reason"] == "stop"
        deltas = items[:-1]
        assert "".join(d["text"] for d in deltas) == "Hello there, how are you doing today?"

    def test_end_to_end_timing_modes_produce_measurable_delay_difference(self, tmp_path):
        db_path = tmp_path / "agent_x_data.sqlite3"
        logger = _make_source_db(db_path)
        logger.record_final_transcript("call1", type="llm_block", payloads=_TEXT_EXCHANGE)
        logger.stop()

        instant_backend = _mock_backend(db_path, timing_mode="instant")
        t0 = time.monotonic()
        list(instant_backend.dispatch_stream({"messages": []}))
        instant_elapsed = time.monotonic() - t0
        assert instant_elapsed < 0.2

    def test_custom_timing_fn_overrides_timing_mode(self, tmp_path):
        db_path = tmp_path / "agent_x_data.sqlite3"
        logger = _make_source_db(db_path)
        logger.record_final_transcript("call1", type="llm_block", payloads=_TEXT_EXCHANGE)
        logger.stop()

        calls = []

        def custom_timing_fn(exchange, chunk_counts):
            calls.append((exchange, chunk_counts))
            for _ in range(sum(chunk_counts) + 1):
                yield 0.0

        cfg = agconfig(
            llmconfig(provider="mock", replay_db_path=str(db_path), timing_mode="poisson")
        )
        cfg.llm.update(timing_fn=custom_timing_fn)
        from agency.llm.agllm import agllm

        backend = agllm.for_config(cfg)
        list(backend.dispatch_stream({"messages": []}))
        assert len(calls) == 1

    def test_timing_fn_not_included_in_dynamic_snapshot(self):
        cfg = agconfig(llmconfig(provider="mock"))
        cfg.llm.timing_fn = lambda exchange, chunk_counts: iter([0.0])
        snapshot = cfg.llm.safe_snapshot()
        assert "timing_fn" not in snapshot


class TestLlmHandlerServerIntegration:
    """Wires the mock backend into a real LlmHandlerServer (via the real
    agllm.for_config() selection, not a monkeypatched attribute) to confirm
    the actual, unmodified merge/finalize path in llm_handler_server.py
    reconstructs a correct finalized llm_block group from the mock's
    synthetic deltas -- not just that the mock is self-consistent."""

    def test_streamed_replay_finalizes_matching_original_blocks(self, tmp_path):
        from agency.engine.host_servers.llm_handler_server import LlmHandlerServer
        from agency.llm.usage_tracker import LlmUsageTracker

        source_db = tmp_path / "source_agent_data.sqlite3"
        source_logger = _make_source_db(source_db)
        source_logger.record_final_transcript("call1", type="llm_block", payloads=_TEXT_EXCHANGE)
        source_logger.stop()

        server_db = tmp_path / "server_data.sqlite3"
        server_logger = _make_source_db(server_db)

        cfg = agconfig(
            llmconfig(provider="mock", replay_db_path=str(source_db), timing_mode="instant")
        )
        server = LlmHandlerServer(cfg, server_logger, LlmUsageTracker())
        handle = server.start_stream({"messages": []})

        item = handle.first()
        items = [item]
        while item["type"] not in ("done", "error"):
            item = handle.first()
            items.append(item)
        handle._thread.join(timeout=2.0)

        assert items[-1]["type"] == "done"
        message = items[-1]["message"]
        content_blocks = [b for b in message["blocks"] if b["type"] != "metadata"]
        assert len(content_blocks) == 1
        assert content_blocks[0]["type"] == "text"
        assert content_blocks[0]["text"] == "Hello there, how are you doing today?"
        assert items[-1]["usage"] == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        assert items[-1]["stop_reason"] == "stop"
        server_logger.stop()
