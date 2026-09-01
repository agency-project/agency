from __future__ import annotations

import json
import sqlite3
import threading
import time

from agency.agcollector import GlobalDataCollector
from agency.agwebui.emitter import agwebui_emitter


def _rows(path, table):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
    finally:
        connection.close()


def test_global_collector_persists_scoped_events_and_spans(tmp_path):
    path = tmp_path / "agency.sqlite3"
    collector = GlobalDataCollector(path, flush_batch_size=100, flush_interval_s=60)
    scoped = collector.scoped(agname="researcher", request_id="run7", skill="search")

    scoped.record_event("turn", {"index": 2}, call_label="call-1", do_update=True)
    scoped.record_span("llm:attempt", 10.0, 12.0, {"model": "x"})
    collector.flush(timeout_s=2)

    event = _rows(path, "lifecycle_events")[0]
    assert (event["agname"], event["request_id"], event["skill"]) == (
        "researcher",
        "run7",
        "search",
    )
    assert json.loads(event["payload"]) == {"index": 2}
    span = _rows(path, "spans")[0]
    assert (span["agname"], span["request_id"], span["name"]) == (
        "researcher",
        "run7",
        "llm:attempt",
    )
    latest = (
        sqlite3.connect(path)
        .execute("SELECT scope_key, payload FROM latest_values WHERE type='turn'")
        .fetchone()
    )
    assert latest == ("run7", '{"index": 2}')
    collector.shutdown(timeout_s=2)


def test_global_collector_accepts_concurrent_publishers_and_orders_records(tmp_path):
    path = tmp_path / "agency.sqlite3"
    collector = GlobalDataCollector(path, flush_batch_size=17, flush_interval_s=60)

    def publish(worker: int):
        for index in range(25):
            collector.record_event("sample", {"worker": worker, "index": index})

    threads = [threading.Thread(target=publish, args=(worker,)) for worker in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    collector.flush(timeout_s=2)

    rows = _rows(path, "lifecycle_events")
    sequences = [row["sequence"] for row in rows]
    assert len(rows) == 200
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == 200
    collector.shutdown(timeout_s=2)


def test_scoped_stop_does_not_stop_process_wide_writer(tmp_path):
    path = tmp_path / "agency.sqlite3"
    collector = GlobalDataCollector(path)
    first = collector.scoped(agname="a", request_id="run0", skill="one")
    second = collector.scoped(agname="b", request_id="run1", skill="two")

    first.start()
    first.record_event("event", {"n": 1})
    first.stop()
    second.record_event("event", {"n": 2})
    collector.shutdown(timeout_s=2)

    assert len(_rows(path, "lifecycle_events")) == 2


def test_new_collector_resumes_sequence_in_existing_global_database(tmp_path):
    path = tmp_path / "agency.sqlite3"
    first = GlobalDataCollector(path, flush_interval_s=0)
    assert first.record_event("first", {}) == 1
    first.shutdown(timeout_s=2)

    second = GlobalDataCollector(path, flush_interval_s=0)
    assert second.record_event("second", {}) == 2
    second.shutdown(timeout_s=2)

    assert [row["sequence"] for row in _rows(path, "lifecycle_events")] == [1, 2]


def test_batch_and_time_thresholds_flush_without_publisher_disk_io(tmp_path):
    batch_path = tmp_path / "batch.sqlite3"
    batch_collector = GlobalDataCollector(
        batch_path,
        flush_batch_size=2,
        flush_interval_s=60,
    )
    batch_collector.record_event("event", {"n": 1})
    batch_collector.record_event("event", {"n": 2})

    deadline = time.time() + 2
    while time.time() < deadline and len(_rows(batch_path, "lifecycle_events")) < 2:
        time.sleep(0.01)
    assert len(_rows(batch_path, "lifecycle_events")) == 2
    batch_collector.shutdown(timeout_s=2)

    interval_path = tmp_path / "interval.sqlite3"
    interval_collector = GlobalDataCollector(
        interval_path,
        flush_batch_size=100,
        flush_interval_s=0.02,
    )
    interval_collector.record_event("event", {"n": 1})
    deadline = time.time() + 2
    while time.time() < deadline and not _rows(interval_path, "lifecycle_events"):
        time.sleep(0.01)
    assert len(_rows(interval_path, "lifecycle_events")) == 1
    interval_collector.shutdown(timeout_s=2)


def test_subscriber_failure_is_observable_and_non_fatal(tmp_path):
    collector = GlobalDataCollector(tmp_path / "agency.sqlite3")

    def fail(_event):
        raise RuntimeError("subscriber broke")

    collector.subscribe(fail)
    collector.record_event("event", {"ok": True})
    collector.flush(timeout_s=2)

    assert "subscriber broke" in collector.snapshot()["telemetry_error"]
    collector.shutdown(timeout_s=2)


def test_deferred_webui_facade_uses_global_writer_and_projection_tables(tmp_path):
    path = tmp_path / "agency.sqlite3"
    collector = GlobalDataCollector(path, flush_interval_s=60)
    emitter = agwebui_emitter(tmp_path, defer_to_global=True)

    emitter.agent_state("researcher", "queued", "search", None)
    assert not (tmp_path / "ui_events.db").exists()
    emitter.bind_collector(collector)
    emitter.agent_state("researcher", "skill", "search", None)
    collector.flush(timeout_s=2)

    connection = sqlite3.connect(path)
    try:
        events = connection.execute("SELECT data FROM events ORDER BY id").fetchall()
        projection = connection.execute(
            "SELECT data FROM agent_state WHERE agname='researcher'"
        ).fetchone()
        global_projection = connection.execute(
            "SELECT payload FROM latest_values WHERE type='agent_state' AND scope_key='researcher'"
        ).fetchone()
    finally:
        connection.close()
    assert [json.loads(row[0])["state"] for row in events] == ["queued", "skill"]
    assert json.loads(projection[0])["state"] == "skill"
    assert json.loads(global_projection[0])["state"] == "skill"
    assert (tmp_path / "global_data_path.txt").read_text(encoding="utf-8") == str(path.resolve())
    collector.shutdown(timeout_s=2)
