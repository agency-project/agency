"""Regression tests for the simplified database diagnosis example."""

import argparse
import sqlite3
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Import path bootstrap
# ---------------------------------------------------------------------------

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_PARENT.parent
for path in (PROJECT_ROOT, PACKAGE_PARENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from database_diagnosis.agents import (  # noqa: E402
    DatabaseToolset,
    _assess_fix_evaluation,
    _combine_fix_results,
    _compact_evidence,
    _compact_raw_evaluation,
    _select_fix_evaluation_target,
    _select_fix_evaluation_targets,
    missing_required_api_key,
)
from database_diagnosis.main import resolve_database_path  # noqa: E402
from database_diagnosis.preanalysis import collect_database_evidence  # noqa: E402
from database_diagnosis.sqlite_access import (  # noqa: E402
    evaluate_index_candidate,
    execute_sql,
    quote_ident,
)


# ---------------------------------------------------------------------------
# Test database helpers
# ---------------------------------------------------------------------------

def _index_names(db_path: Path, table: str) -> list[str]:
    with sqlite3.connect(str(db_path)) as conn:
        return sorted(row[1] for row in conn.execute(f"PRAGMA index_list({quote_ident(table)})"))


def _create_task_database(db_path: Path, rows: int = 1_500) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE projects (
                project_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
            CREATE TABLE tasks (
                task_id INTEGER PRIMARY KEY,
                project_id INTEGER NOT NULL,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(project_id)
            );
            """
        )
        conn.executemany(
            "INSERT INTO projects (project_id, name) VALUES (?, ?)",
            [(i, f"Project {i}") for i in range(1, 41)],
        )
        conn.executemany(
            """
            INSERT INTO tasks (task_id, project_id, state, created_at)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    i,
                    1 if i <= rows // 3 else (i % 40) + 1,
                    ["open", "blocked", "done"][i % 3],
                    f"2025-01-{(i % 28) + 1:02d}T12:00:00",
                )
                for i in range(1, rows + 1)
            ],
        )


# ---------------------------------------------------------------------------
# SQLite transaction policy
# ---------------------------------------------------------------------------

def test_execute_sql_read_only_rolls_back_writes(tmp_path):
    db_path = tmp_path / "rollback.sqlite"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")

    result = execute_sql(
        db_path,
        "INSERT INTO items (name) VALUES (?)",
        ["temporary"],
        write_enabled=False,
    )

    assert result["ok"] is True
    assert result["rolled_back"] is True
    with sqlite3.connect(str(db_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 0


def test_execute_sql_write_mode_commits_writes(tmp_path):
    db_path = tmp_path / "write.sqlite"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")

    result = execute_sql(
        db_path,
        "INSERT INTO items (name) VALUES (?)",
        ["durable"],
        write_enabled=True,
    )

    assert result["ok"] is True
    assert result["rolled_back"] is False
    with sqlite3.connect(str(db_path)) as conn:
        assert conn.execute("SELECT name FROM items").fetchone()[0] == "durable"


def test_evaluate_index_candidate_rolls_back_or_commits_by_policy(tmp_path):
    read_only_db = tmp_path / "readonly.sqlite"
    write_db = tmp_path / "write.sqlite"
    _create_task_database(read_only_db)
    _create_task_database(write_db)

    index_sql = (
        'CREATE INDEX IF NOT EXISTS "idx_tasks_project_created" '
        'ON "tasks" ("project_id", "created_at" DESC)'
    )
    query = (
        'SELECT "task_id", "project_id" FROM "tasks" '
        'WHERE "project_id" = ? ORDER BY "created_at" DESC LIMIT 25'
    )

    dry_run = evaluate_index_candidate(
        read_only_db,
        index_sql,
        query=query,
        params=[1],
        write_enabled=False,
    )
    committed = evaluate_index_candidate(
        write_db,
        index_sql,
        query=query,
        params=[1],
        write_enabled=True,
    )

    assert dry_run["ok"] is True
    assert dry_run["fix_applied"] is False
    assert dry_run["rolled_back"] is True
    assert dry_run["before_plan"]
    assert dry_run["after_plan"]
    assert dry_run["before_benchmark"]["median_ms"] >= 0
    assert dry_run["after_benchmark"]["median_ms"] >= 0
    assert dry_run["improvement_ratio"] is None or dry_run["improvement_ratio"] > 0
    assert "idx_tasks_project_created" not in _index_names(read_only_db, "tasks")

    assert committed["ok"] is True
    assert committed["fix_applied"] is True
    assert committed["rolled_back"] is False
    assert "idx_tasks_project_created" in _index_names(write_db, "tasks")


def test_fix_acceptance_requires_benchmark_speedup():
    accepted = _assess_fix_evaluation(
        {
            "ok": True,
            "fix_sql": "CREATE INDEX idx_tasks_project_id ON tasks(project_id)",
            "before_benchmark": {"median_ms": 10.0},
            "after_benchmark": {"median_ms": 4.0},
            "improvement_ratio": 2.5,
            "plan_changed": True,
        },
        min_speedup=1.05,
    )
    rejected = _assess_fix_evaluation(
        {
            "ok": True,
            "fix_sql": "CREATE INDEX idx_tasks_project_id ON tasks(project_id)",
            "before_benchmark": {"median_ms": 10.0},
            "after_benchmark": {"median_ms": 9.8},
            "improvement_ratio": 1.02,
            "plan_changed": False,
        },
        min_speedup=1.05,
    )

    assert accepted["accepted"] is True
    assert accepted["needs_revision"] is False
    assert rejected["accepted"] is False
    assert rejected["needs_revision"] is True


def test_combined_fix_result_tracks_revision_state():
    fix = _combine_fix_results(
        [
            {
                "fix_applied": False,
                "rolled_back": True,
                "fix_sql": "CREATE INDEX idx_tasks_project_id ON tasks(project_id)",
                "verdict": "Not enough improvement.",
                "evidence": [],
                "accepted": False,
                "needs_revision": True,
                "acceptance": {"accepted": False},
            }
        ]
    )

    assert fix.accepted is False
    assert fix.needs_revision is True
    assert fix.rolled_back is True


# ---------------------------------------------------------------------------
# CLI database selection
# ---------------------------------------------------------------------------

def test_resolve_database_path_creates_default_demo_when_omitted(tmp_path):
    args = argparse.Namespace(database_path=None, db_path=None, rows=300, seed=11)
    selection = resolve_database_path(args, tmp_path)

    assert selection.created_default_demo is True
    assert selection.db_path == tmp_path / "default_demo.sqlite"
    assert selection.db_path.exists()

    evidence = collect_database_evidence(selection.db_path)
    assert evidence["db_path"] == str(selection.db_path)
    assert evidence["schema"]["table_count"] >= 1
    assert "mode" not in evidence


def test_resolve_database_path_uses_provided_database_without_creating_demo(tmp_path):
    db_path = tmp_path / "provided.sqlite"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY)")
    args = argparse.Namespace(database_path=db_path, db_path=None, rows=300, seed=11)

    selection = resolve_database_path(args, tmp_path)

    assert selection.created_default_demo is False
    assert selection.db_path == db_path


# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------

def test_api_key_required_only_for_openai_api_endpoint():
    assert missing_required_api_key(
        {
            "base_url": "https://api.openai.com/v1",
            "api_key": "",
            "model": "gpt-4o",
        }
    )
    assert not missing_required_api_key(
        {
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-test",
            "model": "gpt-4o",
        }
    )
    assert not missing_required_api_key(
        {
            "base_url": "http://127.0.0.1:18000/v1",
            "api_key": "",
            "model": "Qwen/Qwen3.5-4B",
        }
    )
    assert not missing_required_api_key(
        {
            "base_url": "http://127.0.0.1:18000/v1",
            "api_key": "EMPTY",
            "model": "meta-llama/Llama-3.1-8B-Instruct",
        }
    )


# ---------------------------------------------------------------------------
# Fix target selection
# ---------------------------------------------------------------------------

def test_fix_evaluation_prefers_reviewer_chosen_sql_over_first_finding():
    customer_sql = (
        'CREATE INDEX IF NOT EXISTS "idx_dbbot_customers_email_created_at" '
        'ON "customers" ("email", "created_at" DESC)'
    )
    orders_sql = (
        'CREATE INDEX IF NOT EXISTS "idx_dbbot_orders_customer_id_created_at" '
        'ON "orders" ("customer_id", "created_at" DESC)'
    )
    evidence = {
        "findings": [
            {
                "recommendation_sql": customer_sql,
                "candidate_id": "q1",
            },
            {
                "recommendation_sql": orders_sql,
                "candidate_id": "q3",
            },
        ],
        "query_results": [
            {
                "candidate": {
                    "candidate_id": "q1",
                    "query": "SELECT * FROM customers WHERE email = ?",
                    "params": ["customer00001@example.test"],
                }
            },
            {
                "candidate": {
                    "candidate_id": "q3",
                    "query": "SELECT * FROM orders WHERE customer_id = ?",
                    "params": [1],
                }
            },
        ],
    }

    fix_sql, finding, candidate = _select_fix_evaluation_target(
        evidence,
        orders_sql + ";",
    )

    assert fix_sql == orders_sql + ";"
    assert finding["candidate_id"] == "q3"
    assert candidate["query"] == "SELECT * FROM orders WHERE customer_id = ?"


def test_fix_evaluation_splits_multiple_reviewer_chosen_statements():
    orders_sql = (
        'CREATE INDEX IF NOT EXISTS "idx_dbbot_orders_customer_id_created_at" '
        'ON "orders" ("customer_id", "created_at" DESC)'
    )
    email_sql = (
        'CREATE INDEX IF NOT EXISTS "idx_dbbot_customers_email_created_at" '
        'ON "customers" ("email", "created_at" DESC)'
    )
    region_sql = (
        'CREATE INDEX IF NOT EXISTS "idx_dbbot_customers_region_created_at" '
        'ON "customers" ("region", "created_at" DESC)'
    )
    evidence = {
        "findings": [
            {
                "recommendation_sql": email_sql,
                "candidate_id": "q1",
            },
            {
                "recommendation_sql": region_sql,
                "candidate_id": "q2",
            },
            {
                "recommendation_sql": orders_sql,
                "candidate_id": "q3",
            },
        ],
        "query_results": [
            {
                "candidate": {
                    "candidate_id": "q1",
                    "query": "SELECT * FROM customers WHERE email = ?",
                    "params": ["customer00001@example.test"],
                }
            },
            {
                "candidate": {
                    "candidate_id": "q2",
                    "query": "SELECT * FROM customers WHERE region = ?",
                    "params": ["east"],
                }
            },
            {
                "candidate": {
                    "candidate_id": "q3",
                    "query": "SELECT * FROM orders WHERE customer_id = ?",
                    "params": [1],
                }
            },
        ],
    }

    targets = _select_fix_evaluation_targets(
        evidence,
        f"{orders_sql};\n{email_sql};\n{region_sql};",
    )

    assert [target[0] for target in targets] == [
        orders_sql + ";",
        email_sql + ";",
        region_sql + ";",
    ]
    assert [target[1]["candidate_id"] for target in targets] == ["q3", "q1", "q2"]
    assert [target[2]["params"] for target in targets] == [
        [1],
        ["customer00001@example.test"],
        ["east"],
    ]


def test_fix_evaluation_ignores_non_index_statements():
    index_sql = 'CREATE INDEX IF NOT EXISTS "idx_orders_customer_id" ON "orders" ("customer_id")'
    evidence = {
        "findings": [
            {
                "recommendation_sql": index_sql,
                "candidate_id": "q1",
            }
        ],
        "query_results": [
            {
                "candidate": {
                    "candidate_id": "q1",
                    "query": "SELECT * FROM orders WHERE customer_id = ?",
                    "params": [1],
                }
            }
        ],
    }

    targets = _select_fix_evaluation_targets(
        evidence,
        f"{index_sql};\nANALYZE;",
    )

    assert len(targets) == 1
    assert targets[0][0] == index_sql + ";"


def test_compact_evidence_strips_large_schema_sql_and_sample_rows(tmp_path):
    db_path = tmp_path / "compact.sqlite"
    _create_task_database(db_path)
    evidence = collect_database_evidence(db_path)
    compact = _compact_evidence(evidence)

    assert compact["schema"]["table_count"] == evidence["schema"]["table_count"]
    assert "sql" not in compact["schema"]["tables"][0]
    assert isinstance(compact["schema"]["tables"][0]["columns"][0], str)
    assert "first_row" not in compact["query_results"][0]["benchmark"]


def test_compact_raw_evaluation_strips_first_rows(tmp_path):
    db_path = tmp_path / "compact_eval.sqlite"
    _create_task_database(db_path)
    evaluation = evaluate_index_candidate(
        db_path,
        'CREATE INDEX IF NOT EXISTS "idx_tasks_project_id" ON "tasks" ("project_id")',
        query='SELECT "task_id" FROM "tasks" WHERE "project_id" = ? LIMIT 25',
        params=[1],
        write_enabled=False,
    )
    compact = _compact_raw_evaluation(evaluation)

    assert compact["before_plan"]
    assert compact["after_plan"]
    assert "first_row" not in compact["before_benchmark"]
    assert "first_row" not in compact["after_benchmark"]


def test_database_toolset_compacts_large_tool_outputs(tmp_path):
    db_path = tmp_path / "tool_compact.sqlite"
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE documents (
                document_id INTEGER PRIMARY KEY,
                body TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO documents (document_id, body) VALUES (1, ?)",
            ("x" * 1_000,),
        )

    toolset = DatabaseToolset(str(db_path), write_enabled=False)
    query_arg = argparse.Namespace(
        sql="SELECT body FROM documents WHERE document_id = 1",
        max_rows=1,
    )
    bench_arg = argparse.Namespace(
        sql="SELECT body FROM documents WHERE document_id = 1",
        max_rows=1,
        iterations=1,
        warmups=0,
    )
    eval_arg = argparse.Namespace(
        index_sql='CREATE INDEX IF NOT EXISTS "idx_documents_id" ON "documents" ("document_id")',
        query="SELECT body FROM documents WHERE document_id = 1",
        params=[],
    )

    query_result = toolset._execute_sql(query_arg).to_dict()
    bench_result = toolset._benchmark_sql(bench_arg).to_dict()
    eval_result = toolset._evaluate_index_candidate(eval_arg).to_dict()

    assert len(query_result["rows"][0]["body"]) < 320
    assert "first_row" not in bench_result
    assert "first_row_columns" in bench_result
    assert "first_row" not in eval_result["before_benchmark"]
    assert "first_row" not in eval_result["after_benchmark"]
    assert "first_row_columns" in eval_result["after_benchmark"]
