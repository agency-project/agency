from __future__ import annotations

import copy
import json
import queue
import sqlite3
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class _CollectorRecord:
    kind: str
    sequence: int
    data: dict


class GlobalDataCollector:
    """Process-wide live-state hub with one asynchronous SQLite writer.

    Publishers never execute SQLite statements. They update the small in-memory
    runtime projection, enqueue an immutable record, and return. The writer
    thread owns the connection for its complete lifetime.
    """

    def __init__(
        self,
        db_path: "str | Path",
        *,
        flush_batch_size: int = 500,
        flush_interval_s: float = 1.0,
    ) -> None:
        if flush_batch_size <= 0:
            raise ValueError("flush_batch_size must be positive")
        if flush_interval_s < 0:
            raise ValueError("flush_interval_s cannot be negative")
        self.db_path = Path(db_path)
        self.flush_batch_size = flush_batch_size
        self.flush_interval_s = flush_interval_s
        self._queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._state_lock = threading.RLock()
        self._publish_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._sequence = 0
        self._started = False
        self._stopping = False
        self._stopped = False
        self._shutdown_done: "Future[None] | None" = None
        self._writer: "threading.Thread | None" = None
        self._subscribers: list[Callable[[dict], None]] = []
        self._runtime: dict = {}
        self._persistence_error: "str | None" = None
        self._telemetry_error: "str | None" = None
        self._ui_insert_count = 0

    def start(self) -> None:
        with self._start_lock:
            if self._stopping or self._stopped:
                raise RuntimeError("global data collector is shut down")
            if self._started:
                return
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                self._set_persistence_error(exc)
            ready: Future[None] = Future()
            self._writer = threading.Thread(
                target=self._writer_main,
                args=(ready,),
                daemon=True,
                name="agency-data-writer",
            )
            self._writer.start()
            ready.result(timeout=10)
            self._started = True

    def subscribe(self, callback: Callable[[dict], None]) -> None:
        with self._state_lock:
            self._subscribers.append(callback)

    def scoped(self, *, agname: str, request_id: str, skill: str) -> "ScopedDataCollector":
        return ScopedDataCollector(self, agname=agname, request_id=request_id, skill=skill)

    def record_event(
        self,
        type: str,
        payload: dict,
        *,
        source: str = "host",
        agname: "str | None" = None,
        request_id: "str | None" = None,
        skill: "str | None" = None,
        call_label: "str | None" = None,
        scope_key: "str | None" = None,
        do_update: bool = False,
        term_message: "str | None" = None,
        flush: bool = False,
        timestamp: "float | None" = None,
    ) -> int:
        self.start()
        if term_message is not None:
            print(term_message, file=sys.stderr)
        timestamp = time.time() if timestamp is None else timestamp
        immutable_payload = copy.deepcopy(payload)
        with self._publish_lock:
            if self._stopping or self._stopped:
                raise RuntimeError("global data collector is shut down")
            with self._state_lock:
                self._sequence += 1
                sequence = self._sequence
            event = {
                "sequence": sequence,
                "type": type,
                "timestamp": timestamp,
                "source": source,
                "agname": agname,
                "request_id": request_id,
                "skill": skill,
                "call_label": call_label,
                "scope_key": scope_key,
                "payload": immutable_payload,
                "do_update": do_update,
            }
            self._queue.put(("record", _CollectorRecord("event", sequence, event)))
        return sequence

    def record_span(
        self,
        name: str,
        start_ts: float,
        end_ts: float,
        attributes: dict,
        *,
        source: str = "host",
        agname: "str | None" = None,
        request_id: "str | None" = None,
        skill: "str | None" = None,
        cpu_ms: "float | None" = None,
        runqueue_ms: "float | None" = None,
        blocked_ms: "float | None" = None,
        parent: "str | None" = None,
        call_label: "str | None" = None,
        term_message: "str | None" = None,
        flush: bool = False,
    ) -> int:
        self.start()
        if term_message is not None:
            print(term_message, file=sys.stderr)
        immutable_attributes = copy.deepcopy(attributes)
        with self._publish_lock:
            if self._stopping or self._stopped:
                raise RuntimeError("global data collector is shut down")
            with self._state_lock:
                self._sequence += 1
                sequence = self._sequence
            span = {
                "sequence": sequence,
                "name": name,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "cpu_ms": cpu_ms,
                "runqueue_ms": runqueue_ms,
                "blocked_ms": blocked_ms,
                "parent": parent,
                "source": source,
                "agname": agname,
                "request_id": request_id,
                "skill": skill,
                "call_label": call_label,
                "attributes": immutable_attributes,
            }
            self._queue.put(("record", _CollectorRecord("span", sequence, span)))
        return sequence

    def record_ui_event(self, event: dict) -> int:
        """Enqueue one Web UI event on the same process-wide writer.

        The UI keeps its wire-compatible append-only event stream and
        projection tables, but it no longer owns a SQLite connection.  The
        assigned sequence is also embedded in the JSON envelope so UI and
        lifecycle records can be correlated across their respective tables.
        """
        self.start()
        immutable_event = copy.deepcopy(event)
        with self._publish_lock:
            if self._stopping or self._stopped:
                raise RuntimeError("global data collector is shut down")
            with self._state_lock:
                self._sequence += 1
                sequence = self._sequence
            immutable_event["sequence"] = sequence
            self._queue.put(("record", _CollectorRecord("ui_event", sequence, immutable_event)))
        return sequence

    def update_runtime(self, snapshot: dict) -> None:
        with self._state_lock:
            self._runtime = copy.deepcopy(snapshot)

    def snapshot(self) -> dict:
        with self._state_lock:
            result = copy.deepcopy(self._runtime)
            result["db_path"] = str(self.db_path)
            result["persistence_error"] = self._persistence_error
            result["telemetry_error"] = self._telemetry_error
            result["last_sequence"] = self._sequence
            return result

    def flush(self, timeout_s: "float | None" = None) -> None:
        with self._publish_lock:
            if not self._started or self._stopping or self._stopped:
                return
            done: Future[None] = Future()
            self._queue.put(("flush", done))
        done.result(timeout=timeout_s)

    def shutdown(self, timeout_s: "float | None" = None) -> None:
        with self._start_lock:
            if self._stopped:
                return
            if self._stopping:
                done = self._shutdown_done
                assert done is not None
            elif not self._started:
                self._stopping = True
                self._stopped = True
                return
            else:
                done = Future()
                self._shutdown_done = done
                with self._publish_lock:
                    self._stopping = True
                    self._queue.put(("stop", done))
        done.result(timeout=timeout_s)
        if self._writer is not None:
            self._writer.join(timeout=timeout_s)
        with self._start_lock:
            self._stopped = True

    def _writer_main(self, ready: "Future[None]") -> None:
        conn: "sqlite3.Connection | None" = None
        try:
            try:
                conn = sqlite3.connect(str(self.db_path), timeout=30)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                self._ensure_schema(conn)
                self._restore_sequence(conn)
            except BaseException as exc:
                if conn is not None:
                    conn.close()
                    conn = None
                self._set_persistence_error(exc)
            ready.set_result(None)
            batch: list[_CollectorRecord] = []
            deadline = time.monotonic() + self.flush_interval_s
            while True:
                timeout = (
                    None if self.flush_interval_s == 0 else max(0.0, deadline - time.monotonic())
                )
                try:
                    command, value = self._queue.get(timeout=timeout)
                except queue.Empty:
                    command, value = "tick", None

                if command == "record":
                    record = value
                    assert isinstance(record, _CollectorRecord)
                    batch.append(record)
                    self._notify_subscribers(record)
                if (
                    command in ("tick", "flush", "stop")
                    or len(batch) >= self.flush_batch_size
                    or (command == "record" and self.flush_interval_s == 0)
                ):
                    if batch:
                        if conn is not None:
                            self._write_batch(conn, batch)
                        batch.clear()
                    deadline = time.monotonic() + self.flush_interval_s
                if command == "flush":
                    assert isinstance(value, Future)
                    value.set_result(None)
                elif command == "stop":
                    assert isinstance(value, Future)
                    value.set_result(None)
                    break
        except BaseException as exc:
            if not ready.done():
                self._set_persistence_error(exc)
                ready.set_result(None)
            else:
                self._set_persistence_error(exc)
            self._drain_after_writer_failure()
        finally:
            if conn is not None:
                conn.close()

    def _drain_after_writer_failure(self) -> None:
        """Keep telemetry calls non-fatal after an unexpected writer error."""
        while True:
            command, value = self._queue.get()
            if command == "record":
                record = value
                assert isinstance(record, _CollectorRecord)
                self._notify_subscribers(record)
            elif command == "flush":
                assert isinstance(value, Future)
                value.set_result(None)
            elif command == "stop":
                assert isinstance(value, Future)
                value.set_result(None)
                return

    def _notify_subscribers(self, record: _CollectorRecord) -> None:
        with self._state_lock:
            subscribers = list(self._subscribers)
        envelope = {"kind": record.kind, **copy.deepcopy(record.data)}
        for subscriber in subscribers:
            try:
                subscriber(envelope)
            except Exception as exc:
                with self._state_lock:
                    self._telemetry_error = (
                        f"subscriber {subscriber!r}: {type(exc).__name__}: {exc}"
                    )
                print(f"[agcollector] WARNING: subscriber failed: {exc}")

    def _write_batch(self, conn: sqlite3.Connection, records: list[_CollectorRecord]) -> None:
        try:
            with conn:
                for record in records:
                    if record.kind == "event":
                        self._insert_event(conn, record.data)
                    elif record.kind == "span":
                        self._insert_span(conn, record.data)
                    else:
                        self._insert_ui_event(conn, record.data)
                conn.execute(
                    "INSERT INTO collector_metadata(key,value) VALUES('global_sequence',?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(max(record.sequence for record in records)),),
                )
        except Exception as exc:
            self._set_persistence_error(exc)

    def _set_persistence_error(self, exc: BaseException) -> None:
        message = f"{type(exc).__name__}: {exc}"
        with self._state_lock:
            first = self._persistence_error is None
            self._persistence_error = message
        if first:
            print(f"[agcollector] WARNING: persistence failed: {message}")

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS lifecycle_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence INTEGER NOT NULL UNIQUE,
                type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                source TEXT NOT NULL,
                agname TEXT,
                request_id TEXT,
                skill TEXT,
                call_label TEXT,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_lifecycle_events_timestamp
                ON lifecycle_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_lifecycle_events_type
                ON lifecycle_events(type);
            CREATE INDEX IF NOT EXISTS idx_lifecycle_events_agent
                ON lifecycle_events(agname);
            CREATE INDEX IF NOT EXISTS idx_lifecycle_events_request
                ON lifecycle_events(request_id);

            CREATE TABLE IF NOT EXISTS spans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence INTEGER NOT NULL UNIQUE,
                name TEXT NOT NULL,
                start_ts REAL NOT NULL,
                end_ts REAL NOT NULL,
                cpu_ms REAL,
                runqueue_ms REAL,
                blocked_ms REAL,
                parent TEXT,
                source TEXT NOT NULL,
                agname TEXT,
                request_id TEXT,
                skill TEXT,
                call_label TEXT,
                attributes TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_spans_name ON spans(name);
            CREATE INDEX IF NOT EXISTS idx_spans_agent ON spans(agname);
            CREATE INDEX IF NOT EXISTS idx_spans_request ON spans(request_id);

            CREATE TABLE IF NOT EXISTS latest_values (
                type TEXT NOT NULL,
                scope_key TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                source TEXT NOT NULL,
                agname TEXT,
                request_id TEXT,
                skill TEXT,
                call_label TEXT,
                payload TEXT NOT NULL,
                PRIMARY KEY (type, scope_key)
            );
            CREATE TABLE IF NOT EXISTS collector_metadata (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            -- Web UI transport log and cold-start projections. These retain
            -- the established schema consumed by agwebui.server, but this
            -- connection is now their sole writer.
            CREATE TABLE IF NOT EXISTS events (
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                type   TEXT    NOT NULL,
                agname TEXT,
                ts     REAL    NOT NULL,
                data   TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
            CREATE INDEX IF NOT EXISTS idx_events_agname ON events(agname);
            CREATE TABLE IF NOT EXISTS agent_tokens (
                agname TEXT PRIMARY KEY,
                data   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_messages (
                agname TEXT PRIMARY KEY,
                data   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_state (
                agname TEXT PRIMARY KEY,
                data   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_config_state (
                agname TEXT PRIMARY KEY,
                data   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_registry (
                agname TEXT PRIMARY KEY,
                data   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS team_registry (
                team_name TEXT PRIMARY KEY,
                data      TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS resource_state (
                id   INTEGER PRIMARY KEY CHECK (id = 1),
                data TEXT NOT NULL
            );
            """
        )
        conn.commit()

    def _restore_sequence(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT value FROM collector_metadata WHERE key='global_sequence'"
        ).fetchone()
        if row is not None:
            persisted = int(row[0])
        else:
            # Compatibility with databases created by the first global-schema
            # implementation, before collector_metadata existed.
            row = conn.execute(
                "SELECT MAX(sequence) FROM ("
                "SELECT sequence FROM lifecycle_events UNION ALL "
                "SELECT sequence FROM spans UNION ALL "
                "SELECT sequence FROM latest_values)"
            ).fetchone()
            persisted = int(row[0] or 0)
        with self._state_lock:
            self._sequence = max(self._sequence, persisted)

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    def _insert_event(self, conn: sqlite3.Connection, event: dict) -> None:
        values = (
            event["sequence"],
            event["type"],
            event["timestamp"],
            event["source"],
            event["agname"],
            event["request_id"],
            event["skill"],
            event["call_label"],
            self._json(event["payload"]),
        )
        conn.execute(
            "INSERT INTO lifecycle_events "
            "(sequence,type,timestamp,source,agname,request_id,skill,call_label,payload) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            values,
        )
        if event["do_update"]:
            scope_key = event["scope_key"] or event["request_id"] or event["agname"] or "global"
            conn.execute(
                "INSERT INTO latest_values "
                "(type,scope_key,sequence,timestamp,source,agname,request_id,skill,call_label,payload) "
                "VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(type,scope_key) DO UPDATE SET "
                "sequence=excluded.sequence,timestamp=excluded.timestamp,source=excluded.source,"
                "agname=excluded.agname,request_id=excluded.request_id,skill=excluded.skill,"
                "call_label=excluded.call_label,payload=excluded.payload",
                (
                    event["type"],
                    scope_key,
                    event["sequence"],
                    event["timestamp"],
                    event["source"],
                    event["agname"],
                    event["request_id"],
                    event["skill"],
                    event["call_label"],
                    values[-1],
                ),
            )

    def _insert_span(self, conn: sqlite3.Connection, span: dict) -> None:
        conn.execute(
            "INSERT INTO spans "
            "(sequence,name,start_ts,end_ts,cpu_ms,runqueue_ms,blocked_ms,parent,source,agname,"
            "request_id,skill,call_label,attributes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                span["sequence"],
                span["name"],
                span["start_ts"],
                span["end_ts"],
                span["cpu_ms"],
                span["runqueue_ms"],
                span["blocked_ms"],
                span["parent"],
                span["source"],
                span["agname"],
                span["request_id"],
                span["skill"],
                span["call_label"],
                self._json(span["attributes"]),
            ),
        )

    def _insert_ui_event(self, conn: sqlite3.Connection, event: dict) -> None:
        data = self._json(event)
        timestamp = float(event.get("ts") or time.time())
        event_type = str(event.get("type", ""))
        agname = event.get("agname")
        conn.execute(
            "INSERT INTO events(type,agname,ts,data) VALUES(?,?,?,?)",
            (event_type, agname, timestamp, data),
        )

        table = {
            "token_update": "agent_tokens",
            "messages_snapshot": "agent_messages",
            "agent_state": "agent_state",
            "agent_config": "agent_config_state",
            "agent_registered": "agent_registry",
            "team_registered": "team_registry",
        }.get(event_type)
        if table == "team_registry":
            team_name = event.get("team_name")
            if team_name:
                conn.execute(
                    "INSERT INTO team_registry(team_name,data) VALUES(?,?) "
                    "ON CONFLICT(team_name) DO UPDATE SET data=excluded.data",
                    (team_name, data),
                )
        elif table and agname:
            conn.execute(
                f"INSERT INTO {table}(agname,data) VALUES(?,?) "
                "ON CONFLICT(agname) DO UPDATE SET data=excluded.data",
                (agname, data),
            )
        if event_type == "resource_update":
            conn.execute(
                "INSERT INTO resource_state(id,data) VALUES(1,?) "
                "ON CONFLICT(id) DO UPDATE SET data=excluded.data",
                (data,),
            )

        if event_type in {
            "agent_state",
            "agent_config",
            "agent_registered",
            "team_registered",
            "token_update",
            "messages_snapshot",
            "resource_update",
        }:
            scope_key = agname or event.get("team_name") or "global"
            conn.execute(
                "INSERT INTO latest_values "
                "(type,scope_key,sequence,timestamp,source,agname,request_id,skill,"
                "call_label,payload) VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(type,scope_key) DO UPDATE SET "
                "sequence=excluded.sequence,timestamp=excluded.timestamp,"
                "source=excluded.source,agname=excluded.agname,"
                "request_id=excluded.request_id,skill=excluded.skill,"
                "call_label=excluded.call_label,payload=excluded.payload",
                (
                    event_type,
                    scope_key,
                    event["sequence"],
                    timestamp,
                    "webui",
                    agname,
                    None,
                    event.get("skill"),
                    None,
                    data,
                ),
            )

        self._ui_insert_count += 1
        if self._ui_insert_count % 500 == 0:
            self._prune_ui_events(conn)

    @staticmethod
    def _prune_ui_events(conn: sqlite3.Connection) -> None:
        prune_types = (
            "token_update",
            "messages_snapshot",
            "resource_update",
            "agent_state",
        )
        max_id_row = conn.execute("SELECT MAX(id) FROM events").fetchone()
        max_id = max_id_row[0] if max_id_row else None
        if max_id is None:
            return
        cutoff = max_id - 500
        placeholders = ",".join("?" for _ in prune_types)
        conn.execute(
            f"""
            DELETE FROM events
            WHERE type IN ({placeholders})
              AND id <= ?
              AND id NOT IN (
                SELECT MAX(id) FROM events
                WHERE type IN ({placeholders})
                  AND id <= ?
                GROUP BY type, agname, CAST(ts / 60.0 AS INTEGER)
              )
            """,
            prune_types + (cutoff,) + prune_types + (cutoff,),
        )


class ScopedDataCollector:
    """Run-scoped facade used by one per-agent HostInteractionServer."""

    def __init__(
        self,
        collector: GlobalDataCollector,
        *,
        agname: str,
        request_id: str,
        skill: str,
    ) -> None:
        self._collector = collector
        self.agname = agname
        self.request_id = request_id
        self.skill = skill

    def start(self) -> None:
        self._collector.start()

    def stop(self) -> None:
        # The process-wide owner controls the underlying writer lifetime.
        return None

    def set_config(self, _agconfig) -> None:
        return None

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
        self._collector.record_event(
            type,
            payload,
            source="harness",
            agname=self.agname,
            request_id=self.request_id,
            skill=self.skill,
            call_label=call_label,
            do_update=do_update,
            term_message=term_message,
            flush=flush,
        )

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
        self._collector.record_span(
            name,
            start_ts,
            end_ts,
            attributes,
            source="harness",
            agname=self.agname,
            request_id=self.request_id,
            skill=self.skill,
            cpu_ms=cpu_ms,
            runqueue_ms=runqueue_ms,
            blocked_ms=blocked_ms,
            parent=parent,
            call_label=call_label,
            term_message=term_message,
            flush=flush,
        )


class AgentDataCollectorFacade:
    """Agent-scoped publisher that acquires no SQLite connection.

    Events emitted before the process-wide orchestrator exists are retained
    in a small startup buffer. The first submitted request binds the facade
    to the global collector and drains that buffer in publication order.
    """

    def __init__(self, *, agname: str, default_db_path: "str | Path") -> None:
        self.agname = agname
        self.db_path = Path(default_db_path)
        self._collector: "GlobalDataCollector | None" = None
        self._pending: list[tuple[str, tuple, dict]] = []
        self._lock = threading.Lock()
        try:
            from .orchestrator import peek_orchestrator

            orchestrator = peek_orchestrator()
            if orchestrator is not None:
                self.bind_collector(orchestrator.data_collector)
        except Exception as exc:
            print(f"[agcollector] WARNING: startup collector binding failed: {exc}")

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def _bound_collector(self) -> "GlobalDataCollector | None":
        with self._lock:
            collector = self._collector
        if collector is not None:
            return collector
        try:
            from .orchestrator import peek_orchestrator

            orchestrator = peek_orchestrator()
            if orchestrator is not None:
                self.bind_collector(orchestrator.data_collector)
        except Exception as exc:
            print(f"[agcollector] WARNING: deferred collector binding failed: {exc}")
        with self._lock:
            return self._collector

    def flush(self, timeout_s: "float | None" = None) -> None:
        collector = self._bound_collector()
        if collector is not None:
            try:
                collector.flush(timeout_s=timeout_s)
            except Exception as exc:
                print(f"[agcollector] WARNING: agent flush failed: {exc}")

    def set_config(self, _agconfig) -> None:
        return None

    def bind_collector(self, collector: GlobalDataCollector) -> None:
        with self._lock:
            if self._collector is collector:
                return
            if self._collector is not None:
                raise RuntimeError("agent data collector is already bound")
            self._collector = collector
            pending = self._pending
            self._pending = []
        for kind, args, kwargs in pending:
            getattr(self, kind)(*args, **kwargs)

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
        if term_message is not None:
            print(term_message, file=sys.stderr)
            term_message = None
        kwargs = {
            "call_label": call_label,
            "do_update": do_update,
            "term_message": term_message,
            "flush": flush,
        }
        collector = self._bound_collector()
        if collector is None:
            with self._lock:
                if self._collector is None:
                    self._pending.append(
                        ("record_event", (type, copy.deepcopy(payload)), kwargs)
                    )
                    return
                collector = self._collector
        try:
            collector.record_event(
                type,
                payload,
                source="agent",
                agname=self.agname,
                scope_key=self.agname if do_update else None,
                **kwargs,
            )
        except Exception as exc:
            print(f"[agcollector] WARNING: agent event collection failed: {exc}")

    def record_span(
        self,
        name: str,
        start_ts: float,
        end_ts: float,
        attributes: dict,
        **kwargs,
    ) -> None:
        term_message = kwargs.pop("term_message", None)
        if term_message is not None:
            print(term_message, file=sys.stderr)
        collector = self._bound_collector()
        if collector is None:
            with self._lock:
                if self._collector is None:
                    self._pending.append(
                        (
                            "record_span",
                            (name, start_ts, end_ts, copy.deepcopy(attributes)),
                            dict(kwargs),
                        )
                    )
                    return
                collector = self._collector
        try:
            collector.record_span(
                name,
                start_ts,
                end_ts,
                attributes,
                source="agent",
                agname=self.agname,
                **kwargs,
            )
        except Exception as exc:
            print(f"[agcollector] WARNING: agent span collection failed: {exc}")


__all__ = ["AgentDataCollectorFacade", "GlobalDataCollector", "ScopedDataCollector"]
