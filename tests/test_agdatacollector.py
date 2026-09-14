"""Tests for agdatalogger.py -- the per-agent, write-side event/span store."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

from agency.observability.agdatalogger import agDataLogger
from agency.configs.agconfig import agconfig as agconfig_cls, dataloggerconfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agconfig(db_path, **overrides):
    return agconfig_cls(dataloggerconfig(db_path=db_path, **overrides))


def _make_logger(tmp_path, **overrides):
    db_path = str(tmp_path / "agent.db")
    return agDataLogger(_make_agconfig(db_path, **overrides)), db_path


def _select_all(db_path, table):
    con = sqlite3.connect(db_path)
    try:
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
        rows = con.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
        return [dict(zip(cols, row)) for row in rows]
    finally:
        con.close()


# ---------------------------------------------------------------------------
# agconfig data_logger_* fields
# ---------------------------------------------------------------------------


def test_configs_defaults():
    cfg = agconfig_cls(dataloggerconfig(db_path="/tmp/does-not-matter.db"))
    assert cfg.data_logger.flush_batch_size == 20
    assert cfg.data_logger.flush_interval_s == 0.2


def test_configs_explicit_overrides():
    cfg = agconfig_cls(
        dataloggerconfig(
            db_path="/tmp/x.db",
            flush_batch_size=5,
            flush_interval_s=0.1,
        )
    )
    assert cfg.data_logger.flush_batch_size == 5
    assert cfg.data_logger.flush_interval_s == 0.1


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_reads_configs_from_agconfig(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=7, flush_interval_s=1.5)
    assert dc.agconfig.data_logger.db_path == db_path
    assert dc.agconfig.data_logger.flush_batch_size == 7
    assert dc.agconfig.data_logger.flush_interval_s == 1.5
    assert dc._conn is None
    assert dc._event_rows == []
    assert dc._span_rows == []
    assert dc._latest_value_rows == []
    assert dc._pending_count == 0


# ---------------------------------------------------------------------------
# change_config()
# ---------------------------------------------------------------------------


def test_change_config_replaces_the_stored_configs_object(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=7)
    new_agconfig = _make_agconfig(db_path, flush_batch_size=42, flush_interval_s=9.0)
    dc.change_config(new_agconfig)
    assert dc.agconfig.data_logger.flush_batch_size == 42
    assert dc.agconfig.data_logger.flush_interval_s == 9.0


def test_change_config_changes_flush_threshold_at_runtime(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=1000)
    dc.start()
    dc.record_event("tool", {"n": 1})
    assert _select_all(db_path, "events") == []  # buffered under the old, large threshold

    dc.change_config(_make_agconfig(db_path, flush_batch_size=2, flush_interval_s=1000))
    dc.record_event("tool", {"n": 2})  # now 2 pending -- new threshold triggers immediately
    assert len(_select_all(db_path, "events")) == 2
    dc.stop()


def test_change_config_changing_db_path_does_not_move_an_open_connection(tmp_path):
    # Known, accepted limitation: swapping db_path while already started does
    # NOT reopen the connection -- flush() keeps writing to whatever file
    # start() originally opened, even though agconfig.data_logger_db_path now
    # points elsewhere. Documented here rather than silently assumed.
    dc, original_path = _make_logger(tmp_path)
    dc.start()
    other_path = str(tmp_path / "other.db")
    dc.change_config(_make_agconfig(other_path))
    assert dc.agconfig.data_logger.db_path == other_path

    dc.record_event("tool", {"n": 1})
    dc.flush()
    assert len(_select_all(original_path, "events")) == 1
    assert not os.path.exists(other_path)
    dc.stop()


# ---------------------------------------------------------------------------
# start()
# ---------------------------------------------------------------------------


def test_start_creates_parent_directory(tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    dc = agDataLogger(agconfig_cls(dataloggerconfig(db_path=str(nested / "agent.db"))))
    assert not nested.exists()
    dc.start()
    try:
        assert nested.is_dir()
    finally:
        dc.stop()


def test_start_enables_wal_mode(tmp_path):
    dc, _ = _make_logger(tmp_path)
    dc.start()
    try:
        mode = dc._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        dc.stop()


def test_start_creates_schema_tables(tmp_path):
    dc, _ = _make_logger(tmp_path)
    dc.start()
    try:
        tables = {
            r[0] for r in dc._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"events", "spans", "latest_values", "stream_deltas"} <= tables
    finally:
        dc.stop()


def test_start_is_safe_against_a_pre_existing_db_file(tmp_path):
    dc1, db_path = _make_logger(tmp_path)
    dc1.start()
    dc1.stop()

    dc2 = agDataLogger(agconfig_cls(dataloggerconfig(db_path=db_path)))
    dc2.start()  # CREATE TABLE IF NOT EXISTS must not raise on a reused file
    dc2.stop()


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------


def test_stop_without_start_does_not_raise(tmp_path):
    dc, _ = _make_logger(tmp_path)
    dc.stop()
    assert dc._conn is None


def test_stop_is_idempotent(tmp_path):
    dc, _ = _make_logger(tmp_path)
    dc.start()
    dc.stop()
    dc.stop()
    assert dc._conn is None


def test_stop_flushes_pending_records(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=1000)
    dc.start()
    dc.record_event("tool", {"n": 1})
    assert _select_all(db_path, "events") == []
    dc.stop()
    rows = _select_all(db_path, "events")
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# flush()
# ---------------------------------------------------------------------------


def test_flush_writes_buffered_rows_immediately(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=1000)
    dc.start()
    dc.record_event("tool", {"n": 1})
    dc.record_span("op", 0.0, 1.0, {})
    assert _select_all(db_path, "events") == []
    assert _select_all(db_path, "spans") == []
    dc.flush()
    assert len(_select_all(db_path, "events")) == 1
    assert len(_select_all(db_path, "spans")) == 1
    dc.stop()


def test_flush_on_empty_buffers_is_a_noop(tmp_path):
    dc, db_path = _make_logger(tmp_path)
    dc.start()
    dc.flush()  # nothing buffered -- must not raise or write anything
    assert _select_all(db_path, "events") == []
    dc.stop()


def test_flush_updates_last_flush_ts_even_when_empty(tmp_path):
    dc, _ = _make_logger(tmp_path)
    dc.start()
    before = dc._last_flush_ts
    time.sleep(0.01)
    dc.flush()
    assert dc._last_flush_ts > before
    dc.stop()


# ---------------------------------------------------------------------------
# record_event()
# ---------------------------------------------------------------------------


def test_record_event_minimal_args_buffers_only(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=1000)
    dc.start()
    dc.record_event("tool", {"a": 1})
    assert len(dc._event_rows) == 1
    assert _select_all(db_path, "events") == []
    dc.stop()

    rows = _select_all(db_path, "events")
    assert len(rows) == 1
    assert rows[0]["type"] == "tool"
    assert rows[0]["call_label"] is None
    assert json.loads(rows[0]["payload"]) == {"a": 1}
    assert isinstance(rows[0]["timestamp"], float)


def test_record_event_with_call_label(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=1000)
    dc.start()
    dc.record_event("tool", {"a": 1}, call_label="call-42")
    dc.flush()
    rows = _select_all(db_path, "events")
    assert rows[0]["call_label"] == "call-42"
    dc.stop()


def test_record_event_timestamps_itself(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=1000)
    dc.start()
    before = time.time()
    dc.record_event("tool", {})
    after = time.time()
    dc.flush()
    rows = _select_all(db_path, "events")
    assert before <= rows[0]["timestamp"] <= after
    dc.stop()


def test_record_event_do_update_false_does_not_touch_latest_values(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=1000)
    dc.start()
    dc.record_event("agent_state", {"s": "running"}, update_latest_snapshot=False)
    dc.flush()
    assert _select_all(db_path, "latest_values") == []
    dc.stop()


def test_record_event_do_update_true_upserts_latest_values(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=1000)
    dc.start()
    dc.record_event("agent_state", {"s": "running"}, call_label="c1", update_latest_snapshot=True)
    dc.flush()

    rows = _select_all(db_path, "latest_values")
    assert len(rows) == 1
    assert rows[0]["type"] == "agent_state"
    assert rows[0]["call_label"] == "c1"
    assert json.loads(rows[0]["payload"]) == {"s": "running"}
    # update_latest_snapshot doesn't replace the append-only record -- it's in addition to it.
    assert len(_select_all(db_path, "events")) == 1
    dc.stop()


def test_record_event_do_update_overwrites_same_type(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=1000)
    dc.start()
    dc.record_event("agent_state", {"s": "running"}, update_latest_snapshot=True)
    dc.flush()
    dc.record_event("agent_state", {"s": "done"}, update_latest_snapshot=True)
    dc.flush()

    rows = _select_all(db_path, "latest_values")
    assert len(rows) == 1
    assert json.loads(rows[0]["payload"]) == {"s": "done"}
    # the append-only events table keeps both historical entries though.
    assert len(_select_all(db_path, "events")) == 2
    dc.stop()


def test_record_event_do_update_different_types_get_separate_slots(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=1000)
    dc.start()
    dc.record_event("agent_state", {"s": "running"}, update_latest_snapshot=True)
    dc.record_event("token_update", {"t": 10}, update_latest_snapshot=True)
    dc.flush()

    types = {r["type"] for r in _select_all(db_path, "latest_values")}
    assert types == {"agent_state", "token_update"}
    dc.stop()


# ---------------------------------------------------------------------------
# record_span()
# ---------------------------------------------------------------------------


def test_record_span_minimal_args(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=1000)
    dc.start()
    dc.record_span("llm:attempt", 10.0, 12.5, {"model": "x"})
    dc.flush()

    rows = _select_all(db_path, "spans")
    assert len(rows) == 1
    r = rows[0]
    assert r["span_name"] == "llm:attempt"
    assert r["start_ts"] == 10.0
    assert r["end_ts"] == 12.5
    assert r["cpu_ms"] is None
    assert r["runqueue_ms"] is None
    assert r["blocked_ms"] is None
    assert r["parent"] is None
    assert r["call_label"] is None
    assert json.loads(r["attributes"]) == {"model": "x"}
    dc.stop()


def test_record_span_full_args(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=1000)
    dc.start()
    dc.record_span(
        "tool:read_file",
        1.0,
        2.0,
        {"path": "/a"},
        cpu_ms=5.0,
        runqueue_ms=1.0,
        blocked_ms=0.5,
        parent="turn3",
        call_label="call1",
    )
    dc.flush()

    r = _select_all(db_path, "spans")[0]
    assert r["cpu_ms"] == 5.0
    assert r["runqueue_ms"] == 1.0
    assert r["blocked_ms"] == 0.5
    assert r["parent"] == "turn3"
    assert r["call_label"] == "call1"
    dc.stop()


def test_record_span_redirects_incomplete_span_to_events_instead_of_corrupting_the_batch(
    tmp_path, capsys
):
    """start_ts/end_ts are NOT NULL columns -- a caller reporting an
    incomplete span (e.g. a harness closing a span id the host never
    actually opened) must not crash the whole flush and take every other
    pending row down with it (see the real failure this guards against:
    sqlite3.IntegrityError from a batched executemany). It's still recorded,
    just as a `span_dropped` events row (no NOT NULL constraint to violate
    there) rather than silently lost -- so the run's own db shows this
    happened, not just a stderr line."""
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=1000)
    dc.start()
    dc.record_span("orphaned", None, 5.0, {"tool": "write"}, call_label="call-1")
    dc.record_span("also-orphaned", 5.0, None, {})
    dc.record_event("kept", {"ok": True})
    dc.flush()

    assert _select_all(db_path, "spans") == []
    events = _select_all(db_path, "events")
    assert [e["type"] for e in events] == ["span_dropped", "span_dropped", "kept"]

    first = json.loads(events[0]["payload"])
    assert first["span_name"] == "orphaned"
    assert first["start_ts"] is None
    assert first["end_ts"] == 5.0
    assert first["attributes"] == {"tool": "write"}
    assert first["reason"] == "missing start_ts"
    assert events[0]["call_label"] == "call-1"

    second = json.loads(events[1]["payload"])
    assert second["reason"] == "missing end_ts"

    assert "dropping span" in capsys.readouterr().err
    dc.stop()


def test_record_span_does_not_touch_latest_values(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=1000)
    dc.start()
    dc.record_span("op", 0.0, 1.0, {})
    dc.flush()
    assert _select_all(db_path, "latest_values") == []
    dc.stop()


# ---------------------------------------------------------------------------
# record_stream_delta() / record_final_transcript()
# ---------------------------------------------------------------------------


def test_record_stream_delta_only_touches_stream_deltas(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=1000)
    dc.start()
    dc.record_stream_delta("llm_stream_delta", {"text": "hi"}, call_label="c1")
    dc.flush()

    rows = _select_all(db_path, "stream_deltas")
    assert len(rows) == 1
    assert rows[0]["call_label"] == "c1"
    assert json.loads(rows[0]["payload"]) == {"text": "hi"}
    assert _select_all(db_path, "events") == []
    assert _select_all(db_path, "latest_values") == []
    dc.stop()


def test_record_final_transcript_deletes_flushed_deltas_and_appends_events(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=1000)
    dc.start()
    dc.record_stream_delta("llm_stream_delta", {"text": "h"}, call_label="c1")
    dc.record_stream_delta("llm_stream_delta", {"text": "i"}, call_label="c1")
    dc.flush()
    assert len(_select_all(db_path, "stream_deltas")) == 2

    dc.record_final_transcript("c1", type="llm_block", payloads=[{"type": "text", "text": "hi"}])

    assert _select_all(db_path, "stream_deltas") == []
    events = _select_all(db_path, "events")
    assert len(events) == 1
    assert events[0]["type"] == "llm_block"
    assert events[0]["call_label"] == "c1"
    assert json.loads(events[0]["payload"]) == {"type": "text", "text": "hi"}
    dc.stop()


def test_record_final_transcript_clears_not_yet_flushed_pending_deltas(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=1000)
    dc.start()
    dc.record_stream_delta("llm_stream_delta", {"text": "h"}, call_label="c1")
    assert len(dc._stream_delta_rows) == 1

    dc.record_final_transcript("c1", type="llm_block", payloads=[])

    assert dc._stream_delta_rows == []
    assert _select_all(db_path, "stream_deltas") == []
    dc.stop()


def test_record_final_transcript_only_clears_matching_call_label(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=1000)
    dc.start()
    dc.record_stream_delta("llm_stream_delta", {"text": "h"}, call_label="c1")
    dc.record_stream_delta("llm_stream_delta", {"text": "x"}, call_label="c2")
    dc.flush()

    dc.record_final_transcript("c1", type="llm_block", payloads=[{"text": "h"}])

    remaining = _select_all(db_path, "stream_deltas")
    assert len(remaining) == 1
    assert remaining[0]["call_label"] == "c2"
    dc.stop()


def test_record_final_transcript_writes_one_event_row_per_payload(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=1000)
    dc.start()
    dc.record_final_transcript(
        "c1", type="llm_block", payloads=[{"type": "thinking"}, {"type": "text", "text": "hi"}]
    )
    events = _select_all(db_path, "events")
    assert len(events) == 2
    assert [json.loads(e["payload"])["type"] for e in events] == ["thinking", "text"]
    assert all(e["call_label"] == "c1" for e in events)
    dc.stop()


def test_record_final_transcript_safe_with_no_prior_deltas(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=1000)
    dc.start()
    dc.record_final_transcript("c1", type="llm_block", payloads=[{"type": "text", "text": "hi"}])
    assert len(_select_all(db_path, "events")) == 1
    dc.stop()


# ---------------------------------------------------------------------------
# Auto-flush thresholds
# ---------------------------------------------------------------------------


def test_auto_flush_on_count_threshold(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=3, flush_interval_s=1000)
    dc.start()
    dc.record_event("tool", {"n": 1})
    dc.record_event("tool", {"n": 2})
    assert _select_all(db_path, "events") == []  # below threshold, still buffered
    dc.record_event("tool", {"n": 3})  # hits the threshold -> auto-flush
    assert len(_select_all(db_path, "events")) == 3
    dc.stop()


def test_auto_flush_on_time_threshold(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=0)
    dc.start()
    dc.record_event("tool", {"n": 1})  # elapsed-since-start is always >= 0
    assert len(_select_all(db_path, "events")) == 1
    dc.stop()


def test_no_auto_flush_below_both_thresholds(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=1000)
    dc.start()
    for i in range(50):
        dc.record_event("tool", {"n": i})
    assert _select_all(db_path, "events") == []
    assert len(dc._event_rows) == 50
    dc.stop()
    assert len(_select_all(db_path, "events")) == 50


def test_mixed_event_and_span_count_toward_the_same_threshold(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=2, flush_interval_s=1000)
    dc.start()
    dc.record_event("tool", {})
    assert _select_all(db_path, "events") == []
    dc.record_span("op", 0.0, 1.0, {})  # 2nd pending record -> flushes both buffers
    assert len(_select_all(db_path, "events")) == 1
    assert len(_select_all(db_path, "spans")) == 1
    dc.stop()


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_record_calls_are_thread_safe(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=17, flush_interval_s=1000)
    dc.start()

    n_threads = 20
    per_thread = 25

    def worker(i):
        for j in range(per_thread):
            dc.record_event("tool", {"thread": i, "n": j})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    dc.stop()

    rows = _select_all(db_path, "events")
    assert len(rows) == n_threads * per_thread
    seen = {(json.loads(r["payload"])["thread"], json.loads(r["payload"])["n"]) for r in rows}
    assert len(seen) == n_threads * per_thread  # no lost or duplicated records


def test_second_connection_can_read_while_writer_stays_open(tmp_path):
    dc, db_path = _make_logger(tmp_path, flush_batch_size=1000, flush_interval_s=1000)
    dc.start()
    dc.record_event("tool", {"n": 1})
    dc.flush()

    reader = sqlite3.connect(db_path)
    try:
        rows = reader.execute("SELECT type FROM events").fetchall()
        assert rows == [("tool",)]
    finally:
        reader.close()
    dc.stop()
