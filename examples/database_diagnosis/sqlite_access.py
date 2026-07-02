"""Transaction-managed SQLite access for the simplified diagnosis example."""
from __future__ import annotations

import sqlite3
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# SQL value helpers
# ---------------------------------------------------------------------------

def quote_ident(name: str) -> str:
    """Quote a SQLite identifier discovered from a database."""
    return '"' + name.replace('"', '""') + '"'


def _normalize_params(params: list[Any] | None) -> list[Any]:
    return [] if params is None else list(params)


# ---------------------------------------------------------------------------
# Connection and transaction management
# ---------------------------------------------------------------------------

def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(Path(db_path).expanduser()), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def database_session(
    db_path: str | Path,
    *,
    write_enabled: bool = False,
) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection with the requested persistence semantics.

    Read-only here means "writes are allowed to execute, but are never durable":
    the connection starts a transaction and always rolls it back. Write-enabled
    sessions commit successful statements.
    """
    conn = _connect(db_path)
    transaction_started = False
    try:
        if not write_enabled:
            conn.execute("BEGIN")
            transaction_started = True
        yield conn
        if write_enabled:
            conn.commit()
        elif transaction_started:
            conn.rollback()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _fetch_rows(cursor: sqlite3.Cursor, max_rows: int) -> tuple[list[dict[str, Any]], bool]:
    if cursor.description is None:
        return [], False
    fetched = cursor.fetchmany(max_rows + 1)
    return [dict(row) for row in fetched[:max_rows]], len(fetched) > max_rows


def _execution_metadata(
    *,
    sql: str,
    params: list[Any],
    write_enabled: bool,
) -> dict[str, Any]:
    return {
        "sql": sql,
        "params": params,
        "write_enabled": write_enabled,
        "rolled_back": not write_enabled,
    }


# ---------------------------------------------------------------------------
# SQL execution
# ---------------------------------------------------------------------------

def execute_sql(
    db_path: str | Path,
    sql: str,
    params: list[Any] | None = None,
    *,
    write_enabled: bool = False,
    max_rows: int = 100,
) -> dict[str, Any]:
    """Execute one SQL statement under the configured transaction policy."""
    params = _normalize_params(params)
    try:
        with database_session(db_path, write_enabled=write_enabled) as conn:
            cursor = conn.execute(sql, params)
            rows, truncated = _fetch_rows(cursor, max_rows)
            return {
                "ok": True,
                **_execution_metadata(sql=sql, params=params, write_enabled=write_enabled),
                "rows": rows,
                "row_count": cursor.rowcount,
                "last_row_id": cursor.lastrowid,
                "truncated": truncated,
            }
    except sqlite3.DatabaseError as exc:
        return {
            "ok": False,
            **_execution_metadata(sql=sql, params=params, write_enabled=write_enabled),
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Query plans
# ---------------------------------------------------------------------------

def _explain_on_connection(
    conn: sqlite3.Connection,
    sql: str,
    params: list[Any],
) -> list[dict[str, Any]]:
    rows = conn.execute(f"EXPLAIN QUERY PLAN {sql}", params).fetchall()
    return [
        {"id": row["id"], "parent": row["parent"], "detail": row["detail"]}
        for row in rows
    ]


def explain_query_plan(
    db_path: str | Path,
    sql: str,
    params: list[Any] | None = None,
) -> list[dict[str, Any]]:
    params = _normalize_params(params)
    with database_session(db_path, write_enabled=False) as conn:
        return _explain_on_connection(conn, sql, params)


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def _benchmark_on_connection(
    conn: sqlite3.Connection,
    sql: str,
    params: list[Any],
    *,
    iterations: int,
    warmups: int,
    max_rows: int,
) -> dict[str, Any]:
    times_ms: list[float] = []
    rows_returned = 0
    first_row: dict[str, Any] | None = None
    truncated = False

    for _ in range(warmups):
        cursor = conn.execute(sql, params)
        rows, was_truncated = _fetch_rows(cursor, max_rows)
        truncated = truncated or was_truncated

    for _ in range(iterations):
        start = time.perf_counter()
        cursor = conn.execute(sql, params)
        rows, was_truncated = _fetch_rows(cursor, max_rows)
        times_ms.append((time.perf_counter() - start) * 1000)
        rows_returned = len(rows)
        first_row = rows[0] if rows else None
        truncated = truncated or was_truncated

    return {
        "iterations": iterations,
        "rows_returned": rows_returned,
        "min_ms": round(min(times_ms), 3),
        "median_ms": round(statistics.median(times_ms), 3),
        "mean_ms": round(statistics.fmean(times_ms), 3),
        "max_ms": round(max(times_ms), 3),
        "first_row": first_row,
        "truncated": truncated,
    }


def _benchmark_speedup(
    before: dict[str, Any],
    after: dict[str, Any],
) -> float | None:
    before_ms = before.get("median_ms")
    after_ms = after.get("median_ms")
    if not isinstance(before_ms, (int, float)) or not isinstance(after_ms, (int, float)):
        return None
    if before_ms <= 0 or after_ms <= 0:
        return None
    return round(before_ms / after_ms, 3)


def _plan_signature(plan: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("detail", "")) for item in plan]


def benchmark_sql(
    db_path: str | Path,
    sql: str,
    params: list[Any] | None = None,
    *,
    write_enabled: bool = False,
    iterations: int = 5,
    warmups: int = 1,
    max_rows: int = 100,
) -> dict[str, Any]:
    params = _normalize_params(params)
    try:
        with database_session(db_path, write_enabled=write_enabled) as conn:
            result = _benchmark_on_connection(
                conn,
                sql,
                params,
                iterations=iterations,
                warmups=warmups,
                max_rows=max_rows,
            )
            result.update(
                {
                    "ok": True,
                    **_execution_metadata(sql=sql, params=params, write_enabled=write_enabled),
                }
            )
            return result
    except sqlite3.DatabaseError as exc:
        return {
            "ok": False,
            **_execution_metadata(sql=sql, params=params, write_enabled=write_enabled),
            "iterations": 0,
            "rows_returned": 0,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Schema inspection
# ---------------------------------------------------------------------------

def sample_value(db_path: str | Path, table: str, column: str) -> Any | None:
    sql = (
        f"SELECT {quote_ident(column)} AS sample_value "
        f"FROM {quote_ident(table)} "
        f"WHERE {quote_ident(column)} IS NOT NULL "
        "LIMIT 1"
    )
    result = execute_sql(db_path, sql, write_enabled=False, max_rows=1)
    if not result["ok"] or not result["rows"]:
        return None
    return result["rows"][0]["sample_value"]


def inspect_database(db_path: str | Path) -> dict[str, Any]:
    """Inspect user tables, columns, indexes, foreign keys, and row counts."""
    expanded = Path(db_path).expanduser()
    with database_session(expanded, write_enabled=False) as conn:
        application_id = int(conn.execute("PRAGMA application_id").fetchone()[0])
        table_rows = conn.execute(
            """
            SELECT name, sql
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        tables = []
        for table_row in table_rows:
            table_name = table_row["name"]
            tables.append(
                {
                    "name": table_name,
                    "sql": table_row["sql"],
                    "columns": _columns(conn, table_name),
                    "indexes": _indexes(conn, table_name),
                    "foreign_keys": _foreign_keys(conn, table_name),
                    "row_count": _row_count(conn, table_name),
                }
            )
    return {
        "db_path": str(expanded),
        "application_id": application_id,
        "tables": tables,
        "table_count": len(tables),
        "total_rows": sum(table["row_count"] or 0 for table in tables),
    }


def _columns(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    rows = conn.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
    return [
        {
            "cid": row["cid"],
            "name": row["name"],
            "type": row["type"],
            "notnull": bool(row["notnull"]),
            "default": row["dflt_value"],
            "primary_key_position": row["pk"],
            "is_primary_key": bool(row["pk"]),
        }
        for row in rows
    ]


def _indexes(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    indexes = []
    for row in conn.execute(f"PRAGMA index_list({quote_ident(table)})").fetchall():
        idx = dict(row)
        columns = []
        for xrow in conn.execute(f"PRAGMA index_xinfo({quote_ident(idx['name'])})").fetchall():
            if xrow["key"] == 0 or xrow["name"] is None:
                continue
            columns.append(
                {
                    "name": xrow["name"],
                    "desc": bool(xrow["desc"]),
                    "collation": xrow["coll"],
                }
            )
        indexes.append(
            {
                "name": idx["name"],
                "unique": bool(idx["unique"]),
                "origin": idx["origin"],
                "partial": bool(idx["partial"]),
                "columns": columns,
                "is_user_created": idx["origin"] == "c",
            }
        )
    return indexes


def _foreign_keys(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    rows = conn.execute(f"PRAGMA foreign_key_list({quote_ident(table)})").fetchall()
    return [
        {
            "id": row["id"],
            "seq": row["seq"],
            "table": row["table"],
            "from": row["from"],
            "to": row["to"],
            "on_update": row["on_update"],
            "on_delete": row["on_delete"],
            "match": row["match"],
        }
        for row in rows
    ]


def _row_count(conn: sqlite3.Connection, table: str) -> int | None:
    try:
        row = conn.execute(
            f"SELECT COUNT(*) AS row_count FROM {quote_ident(table)}"
        ).fetchone()
    except sqlite3.DatabaseError:
        return None
    return int(row["row_count"])


# ---------------------------------------------------------------------------
# Index evaluation
# ---------------------------------------------------------------------------

def evaluate_index_candidate(
    db_path: str | Path,
    index_sql: str,
    *,
    query: str | None = None,
    params: list[Any] | None = None,
    write_enabled: bool = False,
) -> dict[str, Any]:
    """Create a candidate index, optionally benchmark it, then commit or rollback."""
    params = _normalize_params(params)
    try:
        with database_session(db_path, write_enabled=write_enabled) as conn:
            before_plan = _explain_on_connection(conn, query, params) if query else []
            before_benchmark = (
                _benchmark_on_connection(
                    conn,
                    query,
                    params,
                    iterations=5,
                    warmups=1,
                    max_rows=100,
                )
                if query
                else {}
            )
            conn.execute(index_sql)
            conn.execute("ANALYZE")
            after_plan = _explain_on_connection(conn, query, params) if query else []
            after_benchmark = (
                _benchmark_on_connection(
                    conn,
                    query,
                    params,
                    iterations=5,
                    warmups=1,
                    max_rows=100,
                )
                if query
                else {}
            )
            improvement_ratio = _benchmark_speedup(before_benchmark, after_benchmark)
            return {
                "ok": True,
                "fix_sql": index_sql,
                "fix_applied": write_enabled,
                "rolled_back": not write_enabled,
                "write_enabled": write_enabled,
                "before_plan": before_plan,
                "after_plan": after_plan,
                "plan": after_plan,
                "plan_changed": _plan_signature(before_plan) != _plan_signature(after_plan),
                "before_benchmark": before_benchmark,
                "after_benchmark": after_benchmark,
                "benchmark": after_benchmark,
                "improvement_ratio": improvement_ratio,
                "benchmark_improved": improvement_ratio is not None and improvement_ratio > 1.0,
            }
    except sqlite3.DatabaseError as exc:
        return {
            "ok": False,
            "fix_sql": index_sql,
            "fix_applied": False,
            "rolled_back": not write_enabled,
            "write_enabled": write_enabled,
            "error": str(exc),
        }
