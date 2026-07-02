"""Agent and skill wiring for the one-path database diagnosis flow."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from agency import agent, agdata, agerror, agskill, agteam, agtool

from .preanalysis import collect_database_evidence
from .sqlite_access import (
    benchmark_sql as run_benchmark_sql,
    evaluate_index_candidate as run_evaluate_index_candidate,
    execute_sql as run_execute_sql,
)


# ---------------------------------------------------------------------------
# LLM endpoint configuration
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:18000/v1")


# ---------------------------------------------------------------------------
# LLM configuration helpers
# ---------------------------------------------------------------------------

def is_openai_api(base_url: str | None) -> bool:
    return str(base_url or "").rstrip("/") == "https://api.openai.com/v1"


IS_OPENAI_API = is_openai_api(BASE_URL)

LLM_CONFIG: dict[str, Any] = {
    "base_url": BASE_URL,
    "api_key": os.environ.get("VLLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
    "model": os.environ.get("VLLM_MODEL", os.environ.get("OPENAI_MODEL", "")),
}

MAX_FIX_ATTEMPTS = max(1, int(os.environ.get("DB_BOT_MAX_FIX_ATTEMPTS", "3")))
MIN_FIX_SPEEDUP = float(os.environ.get("DB_BOT_MIN_FIX_SPEEDUP", "1.05"))
TOOL_STRING_PREVIEW_CHARS = 240

if not IS_OPENAI_API:
    LLM_CONFIG.update(
        {
            "temperature": 0.2,
            "max_tokens": 8000,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.05,
        }
    )


def missing_required_api_key(llm_config: dict[str, Any] | None = None) -> bool:
    config = LLM_CONFIG if llm_config is None else llm_config
    return is_openai_api(config.get("base_url")) and not str(config.get("api_key") or "").strip()


# ---------------------------------------------------------------------------
# Shared agent output schemas
# ---------------------------------------------------------------------------

def _hypothesis_output() -> agdata:
    return agdata(
        role=str,
        hypothesis=str,
        evidence=list,
        recommended_sql=str,
        confidence=float,
    )


# ---------------------------------------------------------------------------
# Database tool adapters
# ---------------------------------------------------------------------------

def _compact_tool_value(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) <= TOOL_STRING_PREVIEW_CHARS:
            return value
        omitted = len(value) - TOOL_STRING_PREVIEW_CHARS
        return f"{value[:TOOL_STRING_PREVIEW_CHARS]}... <truncated {omitted} chars>"
    if isinstance(value, list):
        return [_compact_tool_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _compact_tool_value(item) for key, item in value.items()}
    return value


def _compact_tool_benchmark(benchmark: dict[str, Any]) -> dict[str, Any]:
    compact = dict(benchmark)
    first_row = compact.pop("first_row", None)
    if isinstance(first_row, dict):
        compact["first_row_columns"] = list(first_row.keys())
    elif first_row is not None:
        compact["first_row_preview"] = _compact_tool_value(first_row)
    return compact


def _compact_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    compact = dict(result)
    if isinstance(compact.get("rows"), list):
        compact["rows"] = [
            _compact_tool_value(row)
            for row in compact["rows"]
        ]
    if isinstance(compact.get("first_row"), dict):
        compact["first_row_columns"] = list(compact["first_row"].keys())
        compact.pop("first_row", None)
    elif "first_row" in compact:
        compact["first_row_preview"] = _compact_tool_value(compact.pop("first_row"))
    for key in ("before_benchmark", "after_benchmark", "benchmark"):
        if isinstance(compact.get(key), dict):
            compact[key] = _compact_tool_benchmark(compact[key])
    return compact

class DatabaseToolset:
    """Host-side database tools shared by the diagnosis agents."""

    def __init__(self, db_path: str, *, write_enabled: bool):
        self.db_path = db_path
        self.write_enabled = write_enabled

    def tools(self) -> list[agtool]:
        return [
            self._execute_sql_tool(),
            self._benchmark_sql_tool(),
            self._evaluate_index_tool(),
        ]

    @staticmethod
    def _int_arg(arg: Any, name: str, default: int) -> int:
        return int(getattr(arg, name, default) or default)

    def _execute_sql(self, arg: Any) -> agdata:
        return agdata(
            **_compact_tool_result(
                run_execute_sql(
                    self.db_path,
                    arg.sql,
                    getattr(arg, "params", None),
                    write_enabled=self.write_enabled,
                    max_rows=self._int_arg(arg, "max_rows", 100),
                )
            )
        )

    def _benchmark_sql(self, arg: Any) -> agdata:
        return agdata(
            **_compact_tool_result(
                run_benchmark_sql(
                    self.db_path,
                    arg.sql,
                    getattr(arg, "params", None),
                    write_enabled=self.write_enabled,
                    iterations=self._int_arg(arg, "iterations", 5),
                    warmups=self._int_arg(arg, "warmups", 1),
                    max_rows=self._int_arg(arg, "max_rows", 100),
                )
            )
        )

    def _evaluate_index_candidate(self, arg: Any) -> agdata:
        return agdata(
            **_compact_tool_result(
                run_evaluate_index_candidate(
                    self.db_path,
                    arg.index_sql,
                    query=getattr(arg, "query", None),
                    params=getattr(arg, "params", None),
                    write_enabled=self.write_enabled,
                )
            )
        )

    def _execute_sql_tool(self) -> agtool:
        return agtool(
            name="execute_sql",
            description=(
                "Execute one SQLite statement against the target database. "
                "Read-only runs start a transaction and roll it back, so even accidental writes "
                "are not durable. Write-enabled runs commit successful statements."
            ),
            fn=self._execute_sql,
            params={
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "params": {"type": "array"},
                    "max_rows": {"type": "integer"},
                },
                "required": ["sql"],
            },
        )

    def _benchmark_sql_tool(self) -> agtool:
        return agtool(
            name="benchmark_sql",
            description=(
                "Benchmark one SQLite statement. In read-only runs, the benchmark transaction "
                "is rolled back. In write-enabled runs, mutating benchmark statements persist."
            ),
            fn=self._benchmark_sql,
            params={
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "params": {"type": "array"},
                    "iterations": {"type": "integer"},
                    "warmups": {"type": "integer"},
                    "max_rows": {"type": "integer"},
                },
                "required": ["sql"],
            },
        )

    def _evaluate_index_tool(self) -> agtool:
        return agtool(
            name="evaluate_index_candidate",
            description=(
                "Create a candidate SQLite index and optionally benchmark a query with it. "
                "Read-only runs roll the index back. Write-enabled runs commit the index."
            ),
            fn=self._evaluate_index_candidate,
            params={
                "type": "object",
                "properties": {
                    "index_sql": {"type": "string"},
                    "query": {"type": "string"},
                    "params": {"type": "array"},
                },
                "required": ["index_sql"],
            },
        )


# ---------------------------------------------------------------------------
# Specialist agent skills
# ---------------------------------------------------------------------------

class SchemaExplorerSkill(agskill):
    def __init__(self, tools: list[agtool], **kwargs: Any):
        super().__init__(
            name="schema_explorer",
            system_prompt=(
                "You are SchemaExplorer. You diagnose SQLite schema and indexing shape. "
                "First call execute_sql to inspect sqlite_schema or PRAGMA metadata. "
                "Use the supplied evidence, but verify anything important directly in the database."
            ),
            input_schema=agdata(db_path=str, schema=dict, findings=list, write_enabled=bool),
            output_schema=_hypothesis_output(),
            replace_tools=tools,
            **kwargs,
        )


class QueryPlanSkill(agskill):
    def __init__(self, tools: list[agtool], **kwargs: Any):
        super().__init__(
            name="query_plan_analyst",
            system_prompt=(
                "You are QueryPlanAnalyst. You inspect generated candidate queries, plans, "
                "and timings. First call execute_sql with EXPLAIN QUERY PLAN for the most "
                "important candidate query. Recommend the smallest useful SQL fix."
            ),
            input_schema=agdata(db_path=str, query_results=list, findings=list, write_enabled=bool),
            output_schema=_hypothesis_output(),
            replace_tools=tools,
            **kwargs,
        )


class IndexAdvisorSkill(agskill):
    def __init__(self, tools: list[agtool], **kwargs: Any):
        super().__init__(
            name="index_advisor",
            system_prompt=(
                "You are IndexAdvisor. Review the findings and decide whether an index should "
                "be tested or applied. First call execute_sql to inspect existing indexes for "
                "the table you care about. Return a concrete CREATE INDEX statement when justified."
            ),
            input_schema=agdata(db_path=str, schema=dict, findings=list, write_enabled=bool),
            output_schema=_hypothesis_output(),
            replace_tools=tools,
            **kwargs,
        )


class ReviewerSkill(agskill):
    def __init__(self, tools: list[agtool], **kwargs: Any):
        super().__init__(
            name="reviewer",
            system_prompt=(
                "You are ReviewerAgent. Cross-review the specialist hypotheses and choose the "
                "most likely root cause and safest SQL fix. You may call execute_sql if a fact "
                "needs direct verification. If attempt_history is non-empty, treat it as "
                "evaluation feedback: do not repeat SQL that failed to improve the plan or "
                "benchmark, revise the hypothesis when needed, and choose another justified "
                "candidate index or explicitly return no SQL if the evidence is insufficient."
            ),
            input_schema=agdata(
                hypotheses=list,
                findings=list,
                query_results=list,
                attempt_history=list,
                write_enabled=bool,
            ),
            output_schema=agdata(
                root_cause=str,
                consensus=list,
                uncertainty=list,
                chosen_sql=str,
                confidence=float,
            ),
            replace_tools=tools,
            **kwargs,
        )


class FixEvaluatorSkill(agskill):
    def __init__(self, tools: list[agtool], **kwargs: Any):
        super().__init__(
            name="fix_evaluator",
            system_prompt=(
                "You are FixEvaluator. Review the supplied raw_evaluations generated by "
                "evaluate_index_candidate. They include before/after query plans, before/after "
                "benchmarks, plan_changed, and improvement_ratio. Treat raw_evaluations and "
                "acceptance as the source of truth; only call evaluate_index_candidate if the "
                "supplied evaluation is missing or internally inconsistent. Report whether the "
                "index was committed or rolled back, whether the plan/benchmark supports the "
                "recommendation, and whether the reviewer should revise the hypothesis."
            ),
            input_schema=agdata(
                index_sql=str,
                query=str,
                params=list,
                raw_evaluations=list,
                acceptance=dict,
                write_enabled=bool,
            ),
            output_schema=agdata(
                fix_applied=bool,
                rolled_back=bool,
                fix_sql=str,
                verdict=str,
                evidence=list,
                accepted=bool,
                needs_revision=bool,
            ),
            replace_tools=tools,
            **kwargs,
        )


class ReportSkill(agskill):
    def __init__(self, **kwargs: Any):
        super().__init__(
            name="report_agent",
            system_prompt=(
                "You are ReportAgent. Generate a concise markdown database diagnosis report. "
                "Include database path, write policy, schema summary, candidates tested, "
                "expert collaboration, iterative revision attempts, root cause, fix evaluation, "
                "limitations, and assumptions."
            ),
            input_schema=agdata(
                db_path=str,
                write_enabled=bool,
                evidence=dict,
                hypotheses=list,
                attempt_history=list,
                review=dict,
                fix=dict,
            ),
            output_schema=agdata(title=str, report_markdown=str),
            replace_tools=[],
            **kwargs,
        )


# ---------------------------------------------------------------------------
# LLM payload compaction
# ---------------------------------------------------------------------------

def _compact_plan(plan: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("detail", "")) for item in plan]


def _compact_benchmark(benchmark: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(benchmark, dict):
        return {}
    keys = (
        "ok",
        "iterations",
        "rows_returned",
        "min_ms",
        "median_ms",
        "mean_ms",
        "max_ms",
        "truncated",
        "error",
    )
    return {key: benchmark[key] for key in keys if key in benchmark}


def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_id",
        "description",
        "query",
        "params",
        "table",
        "filter_columns",
        "order_by",
        "reason",
        "recommended_index_columns",
    )
    return {key: candidate[key] for key in keys if key in candidate}


def _compact_query_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate": _compact_candidate(result.get("candidate", {})),
        "plan": _compact_plan(result.get("plan", [])),
        "plan_flags": result.get("plan_flags", []),
        "benchmark": _compact_benchmark(result.get("benchmark", {})),
    }


def _compact_table(table: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": table.get("name"),
        "row_count": table.get("row_count"),
        "columns": [column.get("name") for column in table.get("columns", [])],
        "indexes": [
            {
                "name": index.get("name"),
                "unique": bool(index.get("unique")),
                "columns": [column.get("name") for column in index.get("columns", [])],
                "is_user_created": bool(index.get("is_user_created")),
            }
            for index in table.get("indexes", [])
        ],
        "foreign_keys": [
            {
                "from": fk.get("from"),
                "to_table": fk.get("table"),
                "to": fk.get("to"),
            }
            for fk in table.get("foreign_keys", [])
        ],
    }


def _compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "db_path": schema.get("db_path"),
        "application_id": schema.get("application_id"),
        "table_count": schema.get("table_count"),
        "total_rows": schema.get("total_rows"),
        "tables": [_compact_table(table) for table in schema.get("tables", [])],
    }


def _compact_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "db_path": evidence.get("db_path"),
        "schema": _compact_schema(evidence.get("schema", {})),
        "candidate_queries": [
            _compact_candidate(candidate)
            for candidate in evidence.get("candidate_queries", [])
        ],
        "query_results": [
            _compact_query_result(result)
            for result in evidence.get("query_results", [])
        ],
        "findings": evidence.get("findings", []),
        "limitations": evidence.get("limitations", []),
    }


def _compact_raw_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(evaluation.get("ok")),
        "fix_sql": evaluation.get("fix_sql", ""),
        "fix_applied": bool(evaluation.get("fix_applied")),
        "rolled_back": bool(evaluation.get("rolled_back")),
        "write_enabled": bool(evaluation.get("write_enabled")),
        "candidate_id": evaluation.get("candidate_id"),
        "query": evaluation.get("query", ""),
        "params": evaluation.get("params", []),
        "requested_write_enabled": bool(evaluation.get("requested_write_enabled")),
        "before_plan": _compact_plan(evaluation.get("before_plan", [])),
        "after_plan": _compact_plan(evaluation.get("after_plan", [])),
        "plan_changed": bool(evaluation.get("plan_changed")),
        "before_benchmark": _compact_benchmark(evaluation.get("before_benchmark", {})),
        "after_benchmark": _compact_benchmark(evaluation.get("after_benchmark", {})),
        "improvement_ratio": evaluation.get("improvement_ratio"),
        "benchmark_improved": bool(evaluation.get("benchmark_improved")),
        "error": evaluation.get("error", ""),
    }


# ---------------------------------------------------------------------------
# Fix target selection
# ---------------------------------------------------------------------------

def _matching_candidate(evidence: dict[str, Any], candidate_id: str | None) -> dict[str, Any] | None:
    if not candidate_id:
        return None
    for result in evidence["query_results"]:
        if result["candidate"]["candidate_id"] == candidate_id:
            return result["candidate"]
    return None


def _normalize_sql_for_match(sql: str | None) -> str:
    if not sql:
        return ""
    return " ".join(str(sql).strip().rstrip(";").split()).lower()


def _split_sql_statements(sql: str | None) -> list[str]:
    if not sql:
        return []

    statements = []
    buffer = []
    for char in str(sql):
        buffer.append(char)
        candidate = "".join(buffer).strip()
        if char == ";" and sqlite3.complete_statement(candidate):
            statements.append(candidate)
            buffer = []

    tail = "".join(buffer).strip()
    if tail:
        statements.append(tail)
    return statements


def _is_index_statement(sql: str) -> bool:
    normalized = _normalize_sql_for_match(sql)
    return normalized.startswith("create index ") or normalized.startswith("create unique index ")


def _matching_finding_for_sql(evidence: dict[str, Any], sql: str | None) -> dict[str, Any] | None:
    normalized = _normalize_sql_for_match(sql)
    if not normalized:
        return None
    for finding in evidence["findings"]:
        if _normalize_sql_for_match(finding.get("recommendation_sql")) == normalized:
            return finding
    return None


def _select_fix_evaluation_target(
    evidence: dict[str, Any],
    chosen_sql: str | None,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    targets = _select_fix_evaluation_targets(evidence, chosen_sql)
    if not targets:
        return "", None, None
    return targets[0]


def _select_fix_evaluation_targets(
    evidence: dict[str, Any],
    chosen_sql: str | None,
) -> list[tuple[str, dict[str, Any] | None, dict[str, Any] | None]]:
    statements = [
        statement
        for statement in _split_sql_statements(chosen_sql)
        if _is_index_statement(statement)
    ]

    if not statements:
        chosen_finding = next(
            (finding for finding in evidence["findings"] if finding.get("recommendation_sql")),
            None,
        )
        statements = [str(chosen_finding.get("recommendation_sql") or "")] if chosen_finding else []

    targets = []
    for fix_sql in statements:
        chosen_finding = _matching_finding_for_sql(evidence, fix_sql)
        candidate = _matching_candidate(
            evidence,
            chosen_finding.get("candidate_id") if chosen_finding else None,
        )
        targets.append((fix_sql, chosen_finding, candidate))
    return targets


# ---------------------------------------------------------------------------
# Fix result and report helpers
# ---------------------------------------------------------------------------

def _combine_fix_results(results: list[dict[str, Any]]) -> agdata:
    if not results:
        return agdata(
            fix_applied=False,
            rolled_back=False,
            fix_sql="",
            verdict="No SQL fix was recommended.",
            evidence=[],
            accepted=False,
            needs_revision=True,
            acceptance={
                "accepted": False,
                "needs_revision": True,
                "reason": "No SQL fix was recommended.",
                "details": [],
            },
        )

    fix_sql = "\n".join(str(result.get("fix_sql", "")).strip() for result in results if result.get("fix_sql"))
    evidence = []
    verdicts = []
    for index, result in enumerate(results, start=1):
        sql = str(result.get("fix_sql", "")).strip()
        verdict = str(result.get("verdict", "")).strip()
        verdicts.append(f"{index}. {sql}: {verdict}" if sql else f"{index}. {verdict}")
        evidence.append(
            {
                "fix_sql": sql,
                "fix_applied": bool(result.get("fix_applied")),
                "rolled_back": bool(result.get("rolled_back")),
                "verdict": verdict,
                "evidence": result.get("evidence", []),
                "accepted": bool(result.get("accepted")),
                "needs_revision": bool(result.get("needs_revision", True)),
                "acceptance": result.get("acceptance", {}),
                "raw_evaluation": result.get("raw_evaluation", {}),
            }
        )

    accepted = all(bool(result.get("accepted")) for result in results)
    return agdata(
        fix_applied=all(bool(result.get("fix_applied")) for result in results),
        rolled_back=any(bool(result.get("rolled_back")) for result in results),
        fix_sql=fix_sql,
        verdict="\n".join(verdicts),
        evidence=evidence,
        accepted=accepted,
        needs_revision=not accepted,
        acceptance={
            "accepted": accepted,
            "needs_revision": not accepted,
            "reason": "All candidate indexes met the acceptance threshold."
            if accepted
            else "At least one candidate index did not meet the acceptance threshold.",
            "details": [result.get("acceptance", {}) for result in results],
        },
    )


def _median_ms(benchmark: dict[str, Any]) -> float | None:
    value = benchmark.get("median_ms")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _assess_fix_evaluation(
    evaluation: dict[str, Any],
    *,
    min_speedup: float = MIN_FIX_SPEEDUP,
) -> dict[str, Any]:
    sql = str(evaluation.get("fix_sql", "")).strip()
    candidate_id = evaluation.get("candidate_id")
    before_ms = _median_ms(evaluation.get("before_benchmark", {}))
    after_ms = _median_ms(evaluation.get("after_benchmark", {}))
    speedup = evaluation.get("improvement_ratio")
    if isinstance(speedup, (int, float)):
        speedup = float(speedup)
    else:
        speedup = None

    if not evaluation.get("ok"):
        return {
            "sql": sql,
            "candidate_id": candidate_id,
            "accepted": False,
            "needs_revision": True,
            "reason": str(evaluation.get("error") or "Candidate evaluation failed."),
            "before_median_ms": before_ms,
            "after_median_ms": after_ms,
            "improvement_ratio": speedup,
            "plan_changed": bool(evaluation.get("plan_changed")),
            "min_speedup": min_speedup,
        }

    if before_ms is None or after_ms is None or speedup is None:
        return {
            "sql": sql,
            "candidate_id": candidate_id,
            "accepted": False,
            "needs_revision": True,
            "reason": "No benchmarkable query was matched to this SQL fix.",
            "before_median_ms": before_ms,
            "after_median_ms": after_ms,
            "improvement_ratio": speedup,
            "plan_changed": bool(evaluation.get("plan_changed")),
            "min_speedup": min_speedup,
        }

    accepted = speedup >= min_speedup
    return {
        "sql": sql,
        "candidate_id": candidate_id,
        "accepted": accepted,
        "needs_revision": not accepted,
        "reason": (
            f"Median benchmark improved by {speedup:.3g}x."
            if accepted
            else f"Median benchmark improved by only {speedup:.3g}x; expected at least {min_speedup:.3g}x."
        ),
        "before_median_ms": before_ms,
        "after_median_ms": after_ms,
        "improvement_ratio": speedup,
        "plan_changed": bool(evaluation.get("plan_changed")),
        "min_speedup": min_speedup,
    }


def _assess_fix_evaluations(
    evaluations: list[dict[str, Any]],
    *,
    min_speedup: float = MIN_FIX_SPEEDUP,
) -> dict[str, Any]:
    details = [
        _assess_fix_evaluation(evaluation, min_speedup=min_speedup)
        for evaluation in evaluations
    ]
    accepted = bool(details) and all(detail["accepted"] for detail in details)
    return {
        "accepted": accepted,
        "needs_revision": not accepted,
        "reason": (
            "All candidate indexes met the benchmark acceptance threshold."
            if accepted
            else "One or more candidate indexes need revision."
        ),
        "details": details,
        "min_speedup": min_speedup,
    }


def _attempt_history_entry(
    *,
    attempt: int,
    review: dict[str, Any],
    targets: list[tuple[str, dict[str, Any] | None, dict[str, Any] | None]],
    fix: dict[str, Any],
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "root_cause": review.get("root_cause", ""),
        "chosen_sql": review.get("chosen_sql", ""),
        "confidence": review.get("confidence", 0.0),
        "targets": [
            {
                "fix_sql": fix_sql,
                "finding": finding,
                "candidate_id": candidate.get("candidate_id") if candidate else None,
                "query": candidate.get("query") if candidate else "",
                "params": candidate.get("params") if candidate else [],
            }
            for fix_sql, finding, candidate in targets
        ],
        "accepted": bool(fix.get("accepted")),
        "needs_revision": bool(fix.get("needs_revision", True)),
        "verdict": fix.get("verdict", ""),
        "acceptance": fix.get("acceptance", {}),
    }


def _fallback_report(
    evidence: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    review: dict[str, Any],
    fix: dict[str, Any],
    *,
    attempt_history: list[dict[str, Any]],
    write_enabled: bool,
) -> str:
    findings = evidence["findings"]
    finding_lines = [
        f"- [{item['severity']}] {item['table']}: {item['recommendation']}"
        for item in findings
    ] or ["- No index findings were generated."]
    hypothesis_lines = [
        f"- {item['role']}: {item['hypothesis']}"
        for item in hypotheses
    ] or ["- No specialist hypotheses were produced."]
    attempt_lines = [
        (
            f"- Attempt {item['attempt']}: accepted={item['accepted']}; "
            f"SQL=`{item.get('chosen_sql', '')}`; verdict={item.get('verdict', '')}"
        )
        for item in attempt_history
    ] or ["- No fix attempts were evaluated."]
    return "\n".join(
        [
            "# Database Diagnosis Report",
            "",
            f"Database: `{evidence['db_path']}`",
            f"Write enabled: `{write_enabled}`",
            "",
            "## Schema Summary",
            f"- Tables: {evidence['schema']['table_count']}",
            f"- Rows: {evidence['schema']['total_rows']}",
            "",
            "## Findings",
            *finding_lines,
            "",
            "## Expert Collaboration",
            *hypothesis_lines,
            "",
            "## Iterative Fix Attempts",
            *attempt_lines,
            "",
            "## Root Cause",
            review.get("root_cause", "No root cause was selected."),
            "",
            "## Fix Evaluation",
            f"- SQL: `{fix.get('fix_sql')}`",
            f"- Applied: `{fix.get('fix_applied')}`",
            f"- Rolled back: `{fix.get('rolled_back')}`",
            f"- Accepted: `{fix.get('accepted')}`",
            f"- Verdict: {fix.get('verdict', 'No verdict')}",
            "",
            "## Limitations",
            *(f"- {item}" for item in evidence["limitations"]),
            "",
        ]
    )


# ---------------------------------------------------------------------------
# Diagnosis team entrypoint
# ---------------------------------------------------------------------------

class DatabaseDiagnosisTeam(agteam):
    """Single-path DB diagnosis team.

    The team always receives a concrete SQLite database path. If the CLI wants a
    demo, it creates that database first and then calls this same team.
    """

    llm_config = LLM_CONFIG

    def setup(self) -> None:
        self.write_enabled = bool(getattr(self, "write_enabled", False))
        tools = DatabaseToolset(str(self.db_path), write_enabled=self.write_enabled).tools()

        self.schema_explorer = SchemaExplorerSkill(tools)
        self.query_plan_analyst = QueryPlanSkill(tools)
        self.index_advisor = IndexAdvisorSkill(tools)
        self.reviewer = ReviewerSkill(tools)
        self.fix_evaluator = FixEvaluatorSkill(tools)
        self.report_agent = ReportSkill()
        self.coordinator = agent(agname="database_diagnosis_coordinator")

    def run(self) -> agdata:
        print("=" * 72)
        print("Database diagnosis")
        print("=" * 72)
        print(f"Database      : {self.db_path}")
        print(f"Report        : {self.report_path}")
        print(f"Write enabled : {self.write_enabled}")
        print()

        print("Step 1 - collecting schema, plan, and benchmark evidence...")
        evidence = collect_database_evidence(self.db_path)
        llm_evidence = _compact_evidence(evidence)
        print(
            f"  tables={evidence['schema']['table_count']} "
            f"candidates={len(evidence['candidate_queries'])} "
            f"findings={len(evidence['findings'])}"
        )

        schema_input = agdata(
            db_path=str(self.db_path),
            schema=llm_evidence["schema"],
            findings=llm_evidence["findings"],
            write_enabled=self.write_enabled,
        )
        query_plan_input = agdata(
            db_path=str(self.db_path),
            query_results=llm_evidence["query_results"],
            findings=llm_evidence["findings"],
            write_enabled=self.write_enabled,
        )

        print("Step 2 - launching database specialists in parallel...")
        specialist_runs = [
            ("SchemaExplorer", agent.fork(self.coordinator).run(self.schema_explorer, schema_input)),
            ("QueryPlanAnalyst", agent.fork(self.coordinator).run(self.query_plan_analyst, query_plan_input)),
            ("IndexAdvisor", agent.fork(self.coordinator).run(self.index_advisor, schema_input)),
        ]

        hypotheses = []
        for name, pending in specialist_runs:
            result = pending.wait()
            if isinstance(result, agerror):
                raise RuntimeError(f"{name} failed: {result.error}")
            result_dict = result.to_dict()
            hypotheses.append(result_dict)
            print(f"  {name}: {result_dict['hypothesis'][:96]}")

        attempt_history: list[dict[str, Any]] = []
        review: agdata | agerror | None = None
        fix = _combine_fix_results([])

        for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
            print(f"Step 3.{attempt} - reviewer selects or revises a root cause and SQL fix...")
            review = self.coordinator.run(
                self.reviewer,
                agdata(
                    hypotheses=hypotheses,
                    findings=llm_evidence["findings"],
                    query_results=llm_evidence["query_results"],
                    attempt_history=attempt_history,
                    write_enabled=self.write_enabled,
                ),
            ).wait()
            if isinstance(review, agerror):
                raise RuntimeError(f"Reviewer failed: {review.error}")
            print(f"  root cause: {review.root_cause}")

            fix_targets = _select_fix_evaluation_targets(
                evidence,
                getattr(review, "chosen_sql", ""),
            )

            if not fix_targets:
                fix = _combine_fix_results([])
                attempt_history.append(
                    _attempt_history_entry(
                        attempt=attempt,
                        review=review.to_dict(),
                        targets=[],
                        fix=fix.to_dict(),
                    )
                )
                print(f"Step 4.{attempt} - no SQL fix was recommended.")
                break

            print(f"Step 4.{attempt} - evaluating reviewer-selected index SQL...")
            fix_results = []
            for fix_sql, _chosen_finding, candidate in fix_targets:
                query = candidate["query"] if candidate else ""
                params = candidate["params"] if candidate else []
                raw_evaluation = run_evaluate_index_candidate(
                    self.db_path,
                    fix_sql,
                    query=query,
                    params=params,
                    write_enabled=False,
                )
                raw_evaluation.update(
                    {
                        "candidate_id": candidate.get("candidate_id") if candidate else None,
                        "query": query,
                        "params": params,
                        "requested_write_enabled": self.write_enabled,
                    }
                )
                acceptance = _assess_fix_evaluations([raw_evaluation])
                compact_raw_evaluation = _compact_raw_evaluation(raw_evaluation)
                fix_result = self.coordinator.run(
                    self.fix_evaluator,
                    agdata(
                        index_sql=fix_sql,
                        query=query,
                        params=params,
                        raw_evaluations=[compact_raw_evaluation],
                        acceptance=acceptance,
                        write_enabled=self.write_enabled,
                    ),
                ).wait()
                if isinstance(fix_result, agerror):
                    detail = acceptance["details"][0] if acceptance["details"] else {}
                    fix_result_dict = {
                        "fix_applied": bool(raw_evaluation.get("fix_applied")),
                        "rolled_back": bool(raw_evaluation.get("rolled_back")),
                        "fix_sql": fix_sql,
                        "verdict": detail.get("reason", fix_result.error),
                        "evidence": [compact_raw_evaluation],
                        "accepted": bool(acceptance["accepted"]),
                        "needs_revision": bool(acceptance["needs_revision"]),
                    }
                else:
                    fix_result_dict = fix_result.to_dict()

                fix_result_dict.update(
                    {
                        "fix_applied": bool(raw_evaluation.get("fix_applied")),
                        "rolled_back": bool(raw_evaluation.get("rolled_back")),
                        "fix_sql": fix_sql,
                        "accepted": bool(acceptance["accepted"]),
                        "needs_revision": bool(acceptance["needs_revision"]),
                        "acceptance": acceptance,
                        "raw_evaluation": compact_raw_evaluation,
                    }
                )
                fix_results.append(fix_result_dict)
                detail = acceptance["details"][0] if acceptance["details"] else {}
                speedup = detail.get("improvement_ratio")
                speedup_text = f" speedup={speedup:.3g}x" if isinstance(speedup, (int, float)) else ""
                print(
                    f"  evaluated={fix_sql[:80]} "
                    f"accepted={acceptance['accepted']}{speedup_text} "
                    f"test_applied={raw_evaluation.get('fix_applied')} rolled_back={raw_evaluation.get('rolled_back')}"
                )

            fix = _combine_fix_results(fix_results)
            if fix.accepted and self.write_enabled:
                print("  committing accepted SQL because --write is enabled...")
                commit_results = [
                    run_execute_sql(self.db_path, fix_sql, write_enabled=True)
                    for fix_sql, _finding, _candidate in fix_targets
                ]
                committed = all(bool(result.get("ok")) for result in commit_results)
                fix.fix_applied = committed
                fix.rolled_back = not committed
                fix.evidence.append(
                    {
                        "commit_results": commit_results,
                        "committed": committed,
                    }
                )
                fix.verdict = (
                    f"{fix.verdict}\nCommit result: accepted SQL persisted."
                    if committed
                    else f"{fix.verdict}\nCommit result: accepted SQL failed to persist."
                )
            attempt_history.append(
                _attempt_history_entry(
                    attempt=attempt,
                    review=review.to_dict(),
                    targets=fix_targets,
                    fix=fix.to_dict(),
                )
            )
            if fix.accepted:
                print(f"  accepted after attempt {attempt}.")
                break
            if attempt < MAX_FIX_ATTEMPTS:
                print("  feedback: result did not clear the threshold; asking reviewer to revise.")
            else:
                print("  maximum attempts reached; reporting the best evaluated candidate.")

        assert review is not None

        print("Step 5 - report agent writes the diagnosis...")
        report = self.coordinator.run(
            self.report_agent,
            agdata(
                db_path=str(self.db_path),
                write_enabled=self.write_enabled,
                evidence=llm_evidence,
                hypotheses=hypotheses,
                attempt_history=attempt_history,
                review=review.to_dict(),
                fix=fix.to_dict(),
            ),
        ).wait()
        if isinstance(report, agerror):
            report_markdown = _fallback_report(
                llm_evidence,
                hypotheses,
                review.to_dict(),
                fix.to_dict(),
                attempt_history=attempt_history,
                write_enabled=self.write_enabled,
            )
        else:
            report_markdown = report.report_markdown
        if str(self.db_path) not in report_markdown:
            report_markdown = _fallback_report(
                llm_evidence,
                hypotheses,
                review.to_dict(),
                fix.to_dict(),
                attempt_history=attempt_history,
                write_enabled=self.write_enabled,
            )
        Path(self.report_path).write_text(report_markdown, encoding="utf-8")
        print(f"  report saved: {self.report_path}")

        return agdata(
            report_path=str(self.report_path),
            db_path=str(self.db_path),
            root_cause=review.root_cause,
            fix_applied=bool(fix.fix_applied),
            rolled_back=bool(fix.rolled_back),
            accepted=bool(fix.accepted),
            attempts=len(attempt_history),
            write_enabled=self.write_enabled,
        )
