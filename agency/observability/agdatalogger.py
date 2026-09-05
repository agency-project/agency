from __future__ import annotations

import json
import sqlite3
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..configs.agconfig import agconfig as agconfig_cls


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class agDataLogger:
    """A logger instance is reused across the system: one per agent (holding
    that agent's own high-frequency execution data) and one shared global
    instance (holding low-frequency system-overview data from agteam,
    agResourcePool, and the orchestrator itself). Both are this same class,
    same schema -- only ``default_name``/``default_object`` differ."""

    def __init__(
        self,
        agconfig: "agconfig_cls",
        *,
        default_name: "str | None" = None,
        default_object: "str | None" = None,
    ) -> None:
        self.set_config(agconfig)
        self._default_name = default_name
        self._default_object = default_object

        self._conn: "sqlite3.Connection | None" = None
        self._lock = threading.Lock()
        self._sequence = 0
        self._event_rows: list[tuple] = []
        self._span_rows: list[tuple] = []
        self._latest_value_rows: list[tuple] = []
        self._stream_delta_rows: list[tuple] = []
        self._pending_count = 0
        self._last_flush_ts = 0.0

    @property
    def db_path(self) -> str:
        return self.agconfig.data_logger_db_path

    def _next_id_locked(self) -> str:
        """Caller must already hold self._lock. The zero-padded sequence
        prefix keeps ids sortable in local insertion order -- rows that share
        one `timestamp` (e.g. finalize_stream's loop, which computes it once
        for every payload) still resolve correctly by `id` -- while the
        uuid4 suffix keeps every id globally unique across independently
        constructed instances (one per agent, plus the shared global one),
        for a later merge into one physical table."""
        self._sequence += 1
        return f"{self._sequence:020d}{uuid.uuid4().hex}"

    def set_config(self, agconfig: "agconfig_cls") -> None:
        if agconfig.data_logger_db_path is None:
            existing = getattr(self, "agconfig", None)
            if existing is None or existing.data_logger_db_path is None:
                raise ValueError("agDataLogger requires agconfig.data_logger_db_path on first use")
            # A change_config() call rebuilding this agent's whole agconfig
            # (e.g. new LLM settings) has no reason to also know/repeat this
            # logger's own db path -- carry it over from what this instance
            # was already using rather than erroring or silently redirecting
            # to a new, empty database.
            agconfig.data_logger_db_path = existing.data_logger_db_path
        self.agconfig = agconfig

    def start(self) -> None:
        Path(self.agconfig.data_logger_db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.agconfig.data_logger_db_path, timeout=30, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()
        self._last_flush_ts = time.time()

    def stop(self) -> None:
        # flush-then-close must be one atomic critical section: this instance
        # can now be shared (the process-wide global logger), so two threads
        # -- e.g. the orchestrator's scheduler thread on shutdown and the
        # interpreter's atexit handler -- can call stop() concurrently.
        # Closing outside the lock let one thread's close() race a second
        # thread's in-flight flush() on the same connection.
        with self._lock:
            self._flush_locked()
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def read_profile_records(self, profile_session_id: str) -> list[tuple]:
        """Return agprof-compatible span records for one profiling session.

        Profiler-only clock and trace fields live in each span's attributes,
        keeping the shared span schema useful to non-profiler producers. The
        logger is flushed under the same lock before reading, so callers see
        every span accepted before this method acquired the lock.
        """
        with self._lock:
            self._flush_locked()
            connection = self._conn
            owns_connection = connection is None
            if owns_connection:
                if self.agconfig.data_logger_db_path == ":memory:":
                    return []
                connection = sqlite3.connect(self.agconfig.data_logger_db_path, timeout=30)
            assert connection is not None
            try:
                rows = connection.execute(
                    "SELECT span_name,attributes FROM spans ORDER BY id"
                ).fetchall()
            finally:
                if owns_connection:
                    connection.close()

        records = []
        for span_name, attributes_json in rows:
            try:
                attributes = json.loads(attributes_json)
            except (TypeError, json.JSONDecodeError):
                continue
            if attributes.get("agency.profile_session_id") != profile_session_id:
                continue
            attributes.pop("agency.profile_session_id", None)
            profile_span_name = attributes.pop("agency.profile_span_name", span_name)
            start_perf_ns = attributes.get("agency.perf_start_ns")
            wall_ns = attributes.get("agency.wall_ns")
            thread_id = attributes.get("agency.thread_id")
            if start_perf_ns is None or wall_ns is None or thread_id is None:
                continue
            span_id = attributes.get("agency.span_id")
            parent_span_id = attributes.get("agency.parent_span_id")
            records.append(
                (
                    int(thread_id),
                    profile_span_name,
                    int(start_perf_ns),
                    int(wall_ns),
                    int(attributes.get("agency.cpu_ns", 0)),
                    (
                        None
                        if attributes.get("agency.runq_ns") is None
                        else int(attributes["agency.runq_ns"])
                    ),
                    attributes,
                    None if span_id is None else int(span_id, 16),
                    None if parent_span_id is None else int(parent_span_id, 16),
                )
            )
        return records

    def record_event(
        self,
        type: str,
        payload: dict,
        *,
        name: "str | None" = None,
        object: "str | None" = None,
        call_label: "str | None" = None,
        update_latest_snapshot: bool = False,
        term_message: "str | None" = None,
        flush: bool = False,
    ) -> None:
        timestamp = time.time()
        payload_json = json.dumps(payload)
        name = self._default_name if name is None else name
        object = self._default_object if object is None else object
        if term_message is not None:
            print(term_message, file=sys.stderr)
        with self._lock:
            self._event_rows.append(
                (
                    self._next_id_locked(),
                    type,
                    timestamp,
                    name,
                    object,
                    call_label,
                    payload_json,
                    term_message,
                )
            )
            if update_latest_snapshot:
                self._latest_value_rows.append(
                    (type, name, timestamp, object, call_label, payload_json, term_message)
                )
            self._pending_count += 1
            if flush:
                self._flush_locked()
            else:
                self._maybe_flush_locked()

    def record_stream_delta(
        self,
        type: str,
        payload: dict,
        *,
        name: "str | None" = None,
        object: "str | None" = None,
        call_label: "str | None" = None,
        flush: bool = False,
    ) -> None:
        """Append one raw streaming fragment to the transient `stream_deltas`
        table (never `events`, never `latest_values`)"""
        timestamp = time.time()
        payload_json = json.dumps(payload)
        name = self._default_name if name is None else name
        object = self._default_object if object is None else object
        with self._lock:
            self._stream_delta_rows.append(
                (
                    self._next_id_locked(),
                    type,
                    timestamp,
                    name,
                    object,
                    call_label,
                    payload_json,
                    None,
                )
            )
            self._pending_count += 1
            if flush:
                self._flush_locked()
            else:
                self._maybe_flush_locked()

    def finalize_stream(
        self,
        call_label: str,
        type: str,
        payloads: "list[dict]",
        *,
        name: "str | None" = None,
        object: "str | None" = None,
        term_message: "str | None" = None,
    ) -> None:
        """Atomically clear every `stream_deltas` row for *call_label* (both
        already-flushed and still-pending) and append each of *payloads* as
        its own permanent row in `events`."""
        timestamp = time.time()
        name = self._default_name if name is None else name
        object = self._default_object if object is None else object
        if term_message is not None:
            print(term_message, file=sys.stderr)
        with self._lock:
            self._stream_delta_rows = [
                row for row in self._stream_delta_rows if row[5] != call_label
            ]
            for payload in payloads:
                self._event_rows.append(
                    (
                        self._next_id_locked(),
                        type,
                        timestamp,
                        name,
                        object,
                        call_label,
                        json.dumps(payload),
                        None,
                    )
                )
                self._pending_count += 1
            self._flush_locked()
            if self._conn is not None:
                self._conn.execute("DELETE FROM stream_deltas WHERE call_label = ?", (call_label,))
                self._conn.commit()

    def record_span(
        self,
        span_name: str,
        start_ts: float,
        end_ts: float,
        attributes: dict,
        *,
        name: "str | None" = None,
        object: "str | None" = None,
        cpu_ms: "float | None" = None,
        runqueue_ms: "float | None" = None,
        blocked_ms: "float | None" = None,
        parent: "str | None" = None,
        call_label: "str | None" = None,
        term_message: "str | None" = None,
        flush: bool = False,
    ) -> None:
        name = self._default_name if name is None else name
        object = self._default_object if object is None else object
        if term_message is not None:
            print(term_message, file=sys.stderr)
        with self._lock:
            self._span_rows.append(
                (
                    self._next_id_locked(),
                    span_name,
                    start_ts,
                    end_ts,
                    name,
                    object,
                    cpu_ms,
                    runqueue_ms,
                    blocked_ms,
                    parent,
                    call_label,
                    json.dumps(attributes),
                    term_message,
                )
            )
            self._pending_count += 1
            if flush:
                self._flush_locked()
            else:
                self._maybe_flush_locked()

    def _ensure_schema(self) -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                name TEXT,
                object TEXT,
                call_label TEXT,
                payload TEXT NOT NULL,
                term_message TEXT
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_name ON events(name)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_object ON events(object)")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spans (
                id TEXT PRIMARY KEY,
                span_name TEXT NOT NULL,
                start_ts REAL NOT NULL,
                end_ts REAL NOT NULL,
                name TEXT,
                object TEXT,
                cpu_ms REAL,
                runqueue_ms REAL,
                blocked_ms REAL,
                parent TEXT,
                call_label TEXT,
                attributes TEXT NOT NULL,
                term_message TEXT
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_name ON spans(name)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_object ON spans(object)")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS latest_values (
                type TEXT NOT NULL,
                name TEXT NOT NULL DEFAULT '',
                timestamp REAL NOT NULL,
                object TEXT,
                call_label TEXT,
                payload TEXT NOT NULL,
                term_message TEXT,
                PRIMARY KEY (type, name)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stream_deltas (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                name TEXT,
                object TEXT,
                call_label TEXT,
                payload TEXT NOT NULL,
                term_message TEXT
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_stream_deltas_call_label ON stream_deltas(call_label)"
        )
        self._conn.commit()

    def _maybe_flush_locked(self) -> None:
        elapsed = time.time() - self._last_flush_ts
        if (
            self._pending_count >= self.agconfig.data_logger_flush_batch_size
            or elapsed >= self.agconfig.data_logger_flush_interval_s
        ):
            self._flush_locked()

    def _flush_locked(self) -> None:
        if self._conn is None:
            return
        if (
            not self._event_rows
            and not self._span_rows
            and not self._latest_value_rows
            and not self._stream_delta_rows
        ):
            self._last_flush_ts = time.time()
            return
        with self._conn:
            if self._event_rows:
                self._conn.executemany(
                    "INSERT INTO events (id, type, timestamp, name, object, call_label, "
                    "payload, term_message) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    self._event_rows,
                )
            if self._span_rows:
                self._conn.executemany(
                    "INSERT INTO spans "
                    "(id, span_name, start_ts, end_ts, name, object, cpu_ms, runqueue_ms, "
                    "blocked_ms, parent, call_label, attributes, term_message) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    self._span_rows,
                )
            if self._latest_value_rows:
                self._conn.executemany(
                    "INSERT INTO latest_values (type, name, timestamp, object, call_label, "
                    "payload, term_message) VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(type, name) DO UPDATE SET "
                    "timestamp=excluded.timestamp, object=excluded.object, "
                    "call_label=excluded.call_label, payload=excluded.payload, "
                    "term_message=excluded.term_message",
                    [
                        (t, n or "", ts, o, cl, p, tm)
                        for (t, n, ts, o, cl, p, tm) in self._latest_value_rows
                    ],
                )
            if self._stream_delta_rows:
                self._conn.executemany(
                    "INSERT INTO stream_deltas (id, type, timestamp, name, object, call_label, "
                    "payload, term_message) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    self._stream_delta_rows,
                )
        self._event_rows.clear()
        self._span_rows.clear()
        self._latest_value_rows.clear()
        self._stream_delta_rows.clear()
        self._pending_count = 0
        self._last_flush_ts = time.time()


def resolve_global_db_path(log_dir: "str | Path") -> Path:
    return Path(log_dir) / "global_data.sqlite3"
