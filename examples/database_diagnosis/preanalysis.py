"""Schema-derived preanalysis for query candidates and index findings."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sqlite_access import (
    benchmark_sql,
    explain_query_plan,
    inspect_database,
    quote_ident,
    sample_value,
)


# ---------------------------------------------------------------------------
# Candidate generation limits
# ---------------------------------------------------------------------------

LARGE_TABLE_ROW_THRESHOLD = 1_000
MAX_CANDIDATE_QUERIES = 12
DEFAULT_CANDIDATE_LIMIT = 25


# ---------------------------------------------------------------------------
# Evidence data models
# ---------------------------------------------------------------------------

@dataclass
class QueryCandidate:
    candidate_id: str
    description: str
    query: str
    params: list[Any]
    table: str
    filter_columns: list[str]
    order_by: list[tuple[str, str]]
    limit: int
    reason: str
    recommended_index_columns: list[tuple[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "description": self.description,
            "query": self.query,
            "params": self.params,
            "table": self.table,
            "filter_columns": self.filter_columns,
            "order_by": [
                {"column": column, "direction": direction}
                for column, direction in self.order_by
            ],
            "limit": self.limit,
            "reason": self.reason,
            "recommended_index_columns": [
                {"column": column, "direction": direction}
                for column, direction in self.recommended_index_columns
            ],
        }


@dataclass
class AnalysisFinding:
    severity: str
    category: str
    table: str
    evidence: list[str]
    recommendation: str
    recommendation_sql: str | None = None
    candidate_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "table": self.table,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "recommendation_sql": self.recommendation_sql,
            "candidate_id": self.candidate_id,
        }


# ---------------------------------------------------------------------------
# Schema metadata helpers
# ---------------------------------------------------------------------------

def column_names(table: dict[str, Any]) -> list[str]:
    return [column["name"] for column in table["columns"]]


def primary_key_columns(table: dict[str, Any]) -> list[str]:
    return [
        column["name"]
        for column in sorted(table["columns"], key=lambda item: item["primary_key_position"])
        if column["is_primary_key"]
    ]


def index_column_names(index: dict[str, Any]) -> list[str]:
    return [column["name"] for column in index["columns"]]


def is_likely_filter_column(column: dict[str, Any]) -> bool:
    name = column["name"].lower()
    if column["is_primary_key"]:
        return False
    return (
        name.endswith("_id")
        or name in {"status", "type", "kind", "category", "code", "email", "region", "tenant"}
        or name.endswith("_code")
        or name.endswith("_type")
        or name.endswith("_status")
    )


def is_likely_order_column(column: dict[str, Any]) -> bool:
    name = column["name"].lower()
    return (
        name in {"created_at", "updated_at", "timestamp", "created_on", "updated_on"}
        or name.endswith("_at")
        or name.endswith("_date")
        or name.endswith("_time")
    )


def likely_filter_columns(table: dict[str, Any]) -> list[str]:
    return [
        column["name"]
        for column in table["columns"]
        if is_likely_filter_column(column)
    ]


def likely_order_columns(table: dict[str, Any]) -> list[str]:
    return [
        column["name"]
        for column in table["columns"]
        if is_likely_order_column(column)
    ]


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def table_lookup(schema_snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {table["name"]: table for table in schema_snapshot["tables"]}


def has_index_prefix(table: dict[str, Any], columns: list[str]) -> bool:
    if not columns:
        return False
    lower_columns = [column.lower() for column in columns]
    pk_columns = [column.lower() for column in primary_key_columns(table)]
    if pk_columns[: len(lower_columns)] == lower_columns:
        return True
    for index in table["indexes"]:
        indexed = [column.lower() for column in index_column_names(index)]
        if indexed[: len(lower_columns)] == lower_columns:
            return True
    return False


def has_composite_index(table: dict[str, Any], columns: list[str]) -> bool:
    if len(columns) <= 1:
        return has_index_prefix(table, columns)
    lower_columns = [column.lower() for column in columns]
    for index in table["indexes"]:
        indexed = [column.lower() for column in index_column_names(index)]
        if indexed[: len(lower_columns)] == lower_columns:
            return True
    return False


# ---------------------------------------------------------------------------
# Index recommendation SQL helpers
# ---------------------------------------------------------------------------

def safe_index_name(table: str, columns: list[str]) -> str:
    raw = "idx_dbbot_" + table + "_" + "_".join(columns)
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_").lower()
    return cleaned[:58] or "idx_dbbot_candidate"


def create_index_sql(table: str, columns: list[tuple[str, str]]) -> str:
    index_name = safe_index_name(table, [column for column, _direction in columns])
    rendered_columns = []
    for column, direction in columns:
        direction_sql = " DESC" if direction.upper() == "DESC" else ""
        rendered_columns.append(f"{quote_ident(column)}{direction_sql}")
    return (
        f"CREATE INDEX IF NOT EXISTS {quote_ident(index_name)} "
        f"ON {quote_ident(table)} ({', '.join(rendered_columns)})"
    )


def projection_sql(table: dict[str, Any], preferred: list[str]) -> str:
    seen: set[str] = set()
    selected: list[str] = []
    available = set(column_names(table))
    for column in preferred + primary_key_columns(table) + column_names(table):
        if column in available and column not in seen:
            selected.append(column)
            seen.add(column)
        if len(selected) >= 6:
            break
    return ", ".join(quote_ident(column) for column in selected) if selected else "*"


def aliased_projection_sql(alias: str, projection: str) -> str:
    return ", ".join(f"{alias}.{part}" for part in projection.split(", "))


def recommended_columns(filter_column: str, order_column: str | None) -> list[tuple[str, str]]:
    columns = [(filter_column, "ASC")]
    if order_column:
        columns.append((order_column, "DESC"))
    return columns


# ---------------------------------------------------------------------------
# Schema-derived workload candidates
# ---------------------------------------------------------------------------

def generate_candidate_queries(schema_snapshot: dict[str, Any]) -> list[QueryCandidate]:
    candidates: list[QueryCandidate] = []
    lookup = table_lookup(schema_snapshot)
    db_path = schema_snapshot["db_path"]

    for table in schema_snapshot["tables"]:
        table_name = table["name"]
        if not table["columns"] or not table["row_count"]:
            continue

        fk_columns = [fk["from"] for fk in table["foreign_keys"]]
        filter_columns = ordered_unique(fk_columns + likely_filter_columns(table))
        order_columns = likely_order_columns(table)

        for filter_column in filter_columns[:4]:
            sample = sample_value(db_path, table_name, filter_column)
            if sample is None:
                continue
            order_column = next(
                (column for column in order_columns if column != filter_column),
                None,
            )
            projection = projection_sql(
                table,
                [filter_column] + ([order_column] if order_column else []),
            )
            where_sql = f"{quote_ident(filter_column)} = ?"
            order_sql = (
                f" ORDER BY {quote_ident(order_column)} DESC"
                if order_column
                else ""
            )
            index_columns = recommended_columns(filter_column, order_column)
            order_by = [(order_column, "DESC")] if order_column else []
            candidates.append(
                QueryCandidate(
                    candidate_id=f"q{len(candidates) + 1}",
                    description=f"Probe {table_name}.{filter_column} lookup pattern",
                    query=(
                        f"SELECT {projection} FROM {quote_ident(table_name)} "
                        f"WHERE {where_sql}{order_sql} LIMIT {DEFAULT_CANDIDATE_LIMIT}"
                    ),
                    params=[sample],
                    table=table_name,
                    filter_columns=[filter_column],
                    order_by=order_by,
                    limit=DEFAULT_CANDIDATE_LIMIT,
                    reason=(
                        "Column name or foreign key shape suggests this column may be "
                        "used for joins or filters."
                    ),
                    recommended_index_columns=index_columns,
                )
            )
            if len(candidates) >= MAX_CANDIDATE_QUERIES:
                return candidates

        for fk in table["foreign_keys"][:2]:
            parent_table = lookup.get(fk["table"])
            if not parent_table:
                continue
            sample = sample_value(db_path, table_name, fk["from"])
            if sample is None:
                continue
            child_order = next(iter(order_columns), None)
            child_projection = projection_sql(
                table,
                [fk["from"]] + ([child_order] if child_order else []),
            )
            parent_projection = projection_sql(parent_table, [fk["to"]])
            order_sql = (
                f" ORDER BY t0.{quote_ident(child_order)} DESC"
                if child_order
                else ""
            )
            index_columns = recommended_columns(fk["from"], child_order)
            order_by = [(child_order, "DESC")] if child_order else []
            candidates.append(
                QueryCandidate(
                    candidate_id=f"q{len(candidates) + 1}",
                    description=(
                        f"Probe join from {table_name}.{fk['from']} "
                        f"to {fk['table']}.{fk['to']}"
                    ),
                    query=(
                        f"SELECT {aliased_projection_sql('t0', child_projection)}, "
                        f"{aliased_projection_sql('t1', parent_projection)} "
                        f"FROM {quote_ident(table_name)} AS t0 "
                        f"JOIN {quote_ident(fk['table'])} AS t1 "
                        f"ON t0.{quote_ident(fk['from'])} = t1.{quote_ident(fk['to'])} "
                        f"WHERE t1.{quote_ident(fk['to'])} = ?"
                        f"{order_sql} LIMIT {DEFAULT_CANDIDATE_LIMIT}"
                    ),
                    params=[sample],
                    table=table_name,
                    filter_columns=[fk["from"]],
                    order_by=order_by,
                    limit=DEFAULT_CANDIDATE_LIMIT,
                    reason="Foreign keys are common join predicates and often need child-side indexes.",
                    recommended_index_columns=index_columns,
                )
            )
            if len(candidates) >= MAX_CANDIDATE_QUERIES:
                return candidates

    return candidates


# ---------------------------------------------------------------------------
# Query plan signal extraction
# ---------------------------------------------------------------------------

def plan_flags(plan: list[dict[str, Any]]) -> list[str]:
    flags = []
    for item in plan:
        detail = item["detail"]
        upper = detail.upper()
        if "SCAN" in upper and "USING INDEX" not in upper and "USING COVERING INDEX" not in upper:
            flags.append(f"Full scan: {detail}")
        if "USE TEMP B-TREE" in upper:
            flags.append(f"Temp sort: {detail}")
    return flags


# ---------------------------------------------------------------------------
# Finding generation
# ---------------------------------------------------------------------------

def recommend_indexes(
    schema_snapshot: dict[str, Any],
    query_results: list[dict[str, Any]],
) -> list[AnalysisFinding]:
    lookup = table_lookup(schema_snapshot)
    findings: list[AnalysisFinding] = []
    seen_sql: set[str] = set()

    def add_finding(finding: AnalysisFinding) -> None:
        if finding.recommendation_sql:
            if finding.recommendation_sql in seen_sql:
                return
            seen_sql.add(finding.recommendation_sql)
        findings.append(finding)

    for result in query_results:
        candidate = result["candidate"]
        table = lookup.get(candidate["table"])
        if not table:
            continue
        row_count = table["row_count"] or 0
        if row_count < LARGE_TABLE_ROW_THRESHOLD:
            continue
        flags = result["plan_flags"]
        index_columns = [
            (item["column"], item["direction"])
            for item in candidate["recommended_index_columns"]
        ]
        plain_columns = [column for column, _direction in index_columns]
        if not index_columns or has_composite_index(table, plain_columns):
            continue
        if flags:
            benchmark = result["benchmark"]
            add_finding(
                AnalysisFinding(
                    severity="high",
                    category="query_plan",
                    table=table["name"],
                    evidence=flags
                    + [
                        (
                            f"mean={benchmark.get('mean_ms')}ms "
                            f"rows={benchmark.get('rows_returned')}"
                        )
                    ],
                    recommendation=(
                        "Add an index matching the generated diagnostic query's filter "
                        "and ordering columns."
                    ),
                    recommendation_sql=create_index_sql(table["name"], index_columns),
                    candidate_id=candidate["candidate_id"],
                )
            )

    for table in schema_snapshot["tables"]:
        row_count = table["row_count"] or 0
        user_indexes = [index for index in table["indexes"] if index["is_user_created"]]
        likely_columns = likely_filter_columns(table)
        fk_columns = [fk["from"] for fk in table["foreign_keys"]]

        if row_count >= LARGE_TABLE_ROW_THRESHOLD and not user_indexes and likely_columns:
            first_column = likely_columns[0]
            if not has_index_prefix(table, [first_column]):
                add_finding(
                    AnalysisFinding(
                        severity="medium",
                        category="large_table_few_indexes",
                        table=table["name"],
                        evidence=[
                            f"{row_count:,} rows",
                            "No user-created indexes were found on the table.",
                            f"{first_column} looks like a filter or join key.",
                        ],
                        recommendation=(
                            f"Consider indexing {first_column} if workload queries filter on it."
                        ),
                        recommendation_sql=create_index_sql(table["name"], [(first_column, "ASC")]),
                    )
                )

        for fk_column in fk_columns:
            if row_count >= LARGE_TABLE_ROW_THRESHOLD and not has_index_prefix(table, [fk_column]):
                add_finding(
                    AnalysisFinding(
                        severity="medium",
                        category="foreign_key_without_child_index",
                        table=table["name"],
                        evidence=[
                            f"{table['name']}.{fk_column} is a foreign-key column.",
                            "SQLite does not automatically index child foreign-key columns.",
                        ],
                        recommendation=f"Consider indexing {fk_column} for joins and parent lookups.",
                        recommendation_sql=create_index_sql(table["name"], [(fk_column, "ASC")]),
                    )
                )

    severity_order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda item: (severity_order.get(item.severity, 9), item.table, item.category))
    return findings


# ---------------------------------------------------------------------------
# Preanalysis entrypoint
# ---------------------------------------------------------------------------

def collect_database_evidence(db_path: str | Path) -> dict[str, Any]:
    schema_snapshot = inspect_database(db_path)
    candidates = generate_candidate_queries(schema_snapshot)
    query_results: list[dict[str, Any]] = []
    for candidate in candidates:
        plan = explain_query_plan(db_path, candidate.query, candidate.params)
        benchmark = benchmark_sql(db_path, candidate.query, candidate.params, write_enabled=False)
        query_results.append(
            {
                "candidate": candidate.to_dict(),
                "plan": plan,
                "plan_flags": plan_flags(plan),
                "benchmark": benchmark,
            }
        )

    findings = recommend_indexes(schema_snapshot, query_results)
    return {
        "db_path": str(Path(db_path).expanduser()),
        "schema": schema_snapshot,
        "candidate_queries": [candidate.to_dict() for candidate in candidates],
        "query_results": query_results,
        "findings": [finding.to_dict() for finding in findings],
        "limitations": [
            "Candidate workload patterns are inferred from schema names and foreign keys.",
            "Diagnostic queries are capped with LIMIT and lightweight benchmarks.",
            "Read-only runs execute inside rollback-only transactions; write runs persist successful fixes.",
        ],
    }
