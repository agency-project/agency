from __future__ import annotations

import json
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agconfig import agConfig


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class agDataCollectorConfigs:
    db_path: str
    flush_batch_size: int = 20
    flush_interval_s: float = 1.0


class agDataCollector:
    def __init__(self, agconfig: "agConfig") -> None:
        self.set_config(agconfig)

        self._conn: "sqlite3.Connection | None" = None
        self._lock = threading.Lock()
        self._event_rows: list[tuple] = []
        self._span_rows: list[tuple] = []
        self._latest_value_rows: list[tuple] = []
        self._stream_delta_rows: list[tuple] = []
        self._pending_count = 0
        self._last_flush_ts = 0.0

    def set_config(self, agconfig: "agConfig") -> None:
        configs = agconfig.__dict__.get("agDataCollectorConfigs")
        if configs is None:
            configs = getattr(self, "_configs", None)
            if configs is None:
                raise ValueError(
                    "agDataCollector requires agconfig.agDataCollectorConfigs on first use"
                )
            agconfig.agDataCollectorConfigs = configs
        self._configs = configs

    def start(self) -> None:
        Path(self._configs.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._configs.db_path, timeout=30, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()
        self._last_flush_ts = time.time()

    def stop(self) -> None:
        self.flush()
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def record_event(
        self,
        type: str,
        payload: dict,
        *,
        call_label: "str | None" = None,
        do_update: bool = False,
        term_message: "str | None" = None,
        flush: bool = False,
    ) -> None:
        timestamp = time.time()
        payload_json = json.dumps(payload)
        if term_message is not None:
            print(term_message, file=sys.stderr)
        with self._lock:
            self._event_rows.append((type, timestamp, call_label, payload_json, term_message))
            if do_update:
                self._latest_value_rows.append(
                    (type, timestamp, call_label, payload_json, term_message)
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
        call_label: "str | None" = None,
        flush: bool = False,
    ) -> None:
        """Append one raw streaming fragment to the transient `stream_deltas`
        table (never `events`, never `latest_values`)"""
        timestamp = time.time()
        payload_json = json.dumps(payload)
        with self._lock:
            self._stream_delta_rows.append((type, timestamp, call_label, payload_json, None))
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
        term_message: "str | None" = None,
    ) -> None:
        """Atomically clear every `stream_deltas` row for *call_label* (both
        already-flushed and still-pending) and append each of *payloads* as
        its own permanent row in `events`."""
        timestamp = time.time()
        if term_message is not None:
            print(term_message, file=sys.stderr)
        with self._lock:
            self._stream_delta_rows = [
                row for row in self._stream_delta_rows if row[2] != call_label
            ]
            for payload in payloads:
                self._event_rows.append((type, timestamp, call_label, json.dumps(payload), None))
                self._pending_count += 1
            self._flush_locked()
            if self._conn is not None:
                self._conn.execute("DELETE FROM stream_deltas WHERE call_label = ?", (call_label,))
                self._conn.commit()

    def record_span(
        self,
        name: str,
        start_ts: float,
        end_ts: float,
        attributes: dict,
        *,
        cpu_ms: "float | None" = None,
        runqueue_ms: "float | None" = None,
        blocked_ms: "float | None" = None,
        parent: "str | None" = None,
        call_label: "str | None" = None,
        term_message: "str | None" = None,
        flush: bool = False,
    ) -> None:
        if term_message is not None:
            print(term_message, file=sys.stderr)
        row = (
            name,
            start_ts,
            end_ts,
            cpu_ms,
            runqueue_ms,
            blocked_ms,
            parent,
            call_label,
            json.dumps(attributes),
            term_message,
        )
        with self._lock:
            self._span_rows.append(row)
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
                id INTEGER PRIMARY KEY,
                type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                call_label TEXT,
                payload TEXT NOT NULL,
                term_message TEXT
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spans (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                start_ts REAL NOT NULL,
                end_ts REAL NOT NULL,
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
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS latest_values (
                type TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                call_label TEXT,
                payload TEXT NOT NULL,
                term_message TEXT
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stream_deltas (
                id INTEGER PRIMARY KEY,
                type TEXT NOT NULL,
                timestamp REAL NOT NULL,
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
            self._pending_count >= self._configs.flush_batch_size
            or elapsed >= self._configs.flush_interval_s
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
                    "INSERT INTO events (type, timestamp, call_label, payload, term_message) "
                    "VALUES (?, ?, ?, ?, ?)",
                    self._event_rows,
                )
            if self._span_rows:
                self._conn.executemany(
                    "INSERT INTO spans "
                    "(name, start_ts, end_ts, cpu_ms, runqueue_ms, blocked_ms, parent, call_label, "
                    "attributes, term_message) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    self._span_rows,
                )
            if self._latest_value_rows:
                self._conn.executemany(
                    "INSERT INTO latest_values (type, timestamp, call_label, payload, term_message) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(type) DO UPDATE SET "
                    "timestamp=excluded.timestamp, call_label=excluded.call_label, "
                    "payload=excluded.payload, term_message=excluded.term_message",
                    self._latest_value_rows,
                )
            if self._stream_delta_rows:
                self._conn.executemany(
                    "INSERT INTO stream_deltas (type, timestamp, call_label, payload, term_message) "
                    "VALUES (?, ?, ?, ?, ?)",
                    self._stream_delta_rows,
                )
        self._event_rows.clear()
        self._span_rows.clear()
        self._latest_value_rows.clear()
        self._stream_delta_rows.clear()
        self._pending_count = 0
        self._last_flush_ts = time.time()
