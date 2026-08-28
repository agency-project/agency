from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agconfig import agConfig


@dataclass
class agDataCollectorConfigs:
    db_path: str
    flush_batch_size: int = 500
    flush_interval_s: float = 1.0


class agDataCollector:
    def __init__(self, agconfig: "agConfig") -> None:
        self.set_config(agconfig)

        self._conn: "sqlite3.Connection | None" = None
        self._lock = threading.Lock()
        self._event_rows: list[tuple] = []
        self._span_rows: list[tuple] = []
        self._latest_value_rows: list[tuple] = []
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
    ) -> None:
        timestamp = time.time()
        payload_json = json.dumps(payload)
        with self._lock:
            self._event_rows.append((type, timestamp, call_label, payload_json))
            if do_update:
                self._latest_value_rows.append((type, timestamp, call_label, payload_json))
            self._pending_count += 1
            self._maybe_flush_locked()

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
    ) -> None:
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
        )
        with self._lock:
            self._span_rows.append(row)
            self._pending_count += 1
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
                payload TEXT NOT NULL
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
                attributes TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS latest_values (
                type TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                call_label TEXT,
                payload TEXT NOT NULL
            )
            """
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
        if not self._event_rows and not self._span_rows and not self._latest_value_rows:
            self._last_flush_ts = time.time()
            return
        with self._conn:
            if self._event_rows:
                self._conn.executemany(
                    "INSERT INTO events (type, timestamp, call_label, payload) VALUES (?, ?, ?, ?)",
                    self._event_rows,
                )
            if self._span_rows:
                self._conn.executemany(
                    "INSERT INTO spans "
                    "(name, start_ts, end_ts, cpu_ms, runqueue_ms, blocked_ms, parent, call_label, attributes) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    self._span_rows,
                )
            if self._latest_value_rows:
                self._conn.executemany(
                    "INSERT INTO latest_values (type, timestamp, call_label, payload) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(type) DO UPDATE SET "
                    "timestamp=excluded.timestamp, call_label=excluded.call_label, payload=excluded.payload",
                    self._latest_value_rows,
                )
        self._event_rows.clear()
        self._span_rows.clear()
        self._latest_value_rows.clear()
        self._pending_count = 0
        self._last_flush_ts = time.time()
