"""Simplified DB diagnosis entry point.

Run against a database path:

    uv run python examples/database_diagnosis/main.py --database-path runs/my_database.sqlite

Run with no path to create the default demo database first:

    uv run python examples/database_diagnosis/main.py
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Import path bootstrap
# ---------------------------------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = CURRENT_DIR.parent
PROJECT_ROOT = EXAMPLES_DIR.parent
for path in (PROJECT_ROOT, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agency import AgError, agent, agsync  # noqa: E402
from database_diagnosis.agents import (  # noqa: E402
    LLM_CONFIG,
    DatabaseDiagnosisTeam,
    missing_required_api_key,
)
from database_diagnosis.default_demo import create_default_database  # noqa: E402


# ---------------------------------------------------------------------------
# CLI data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatabaseSelection:
    db_path: Path
    created_default_demo: bool


# ---------------------------------------------------------------------------
# Run directory management
# ---------------------------------------------------------------------------

DEFAULT_ROWS = int(os.environ.get("DB_BOT_SIMPLE_ROWS", "120000"))
DEFAULT_SEED = int(os.environ.get("DB_BOT_SIMPLE_SEED", "42"))
DEFAULT_WEBUI_PORT = int(os.environ.get("DB_BOT_SIMPLE_WEBUI_PORT", "7860"))


def _make_run_dir(name: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    run_dir = Path(__file__).parents[2] / "runs" / f"{ts}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the simplified database diagnosis team.")
    parser.add_argument(
        "db_path",
        nargs="?",
        type=Path,
        help="SQLite database to diagnose. If omitted, a default demo database is created.",
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        help="SQLite database to diagnose. Equivalent to the positional path.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist successful fix SQL. Without this flag, database writes are rolled back.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
        help="Rows to generate when no database path is supplied.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for default demo database generation.",
    )
    parser.add_argument(
        "--webui",
        action="store_true",
        help="Run the diagnosis with the agwebui dashboard active.",
    )
    parser.add_argument(
        "--webui-port",
        type=int,
        default=DEFAULT_WEBUI_PORT,
        help="Port for --webui.",
    )
    parser.add_argument(
        "--no-webui-linger",
        action="store_true",
        help="With --webui, stop the dashboard as soon as the run completes.",
    )
    args = parser.parse_args(argv)
    if args.db_path and args.database_path:
        parser.error("provide either positional db_path or --database-path, not both")
    return args


# ---------------------------------------------------------------------------
# Database selection
# ---------------------------------------------------------------------------

def resolve_database_path(args: argparse.Namespace, run_dir: Path) -> DatabaseSelection:
    provided = args.database_path or args.db_path
    if provided:
        return DatabaseSelection(db_path=Path(provided).expanduser(), created_default_demo=False)

    db_path = run_dir / "default_demo.sqlite"
    create_default_database(db_path, rows=args.rows, seed=args.seed)
    return DatabaseSelection(db_path=db_path, created_default_demo=True)


# ---------------------------------------------------------------------------
# Diagnosis execution
# ---------------------------------------------------------------------------

def run_diagnosis(args: argparse.Namespace, run_dir: Path) -> int:
    report_path = run_dir / "database_diagnosis_report.md"

    print(f"Endpoint : {LLM_CONFIG['base_url']}")
    print(f"Model    : {LLM_CONFIG.get('model') or '<unset>'}")
    print(f"Run dir  : {run_dir}\n")

    try:
        if missing_required_api_key():
            print(
                "ERROR: OPENAI-compatible hosted mode requires VLLM_API_KEY "
                "or OPENAI_API_KEY to be set."
            )
            return 1

        selection = resolve_database_path(args, run_dir)
        if selection.created_default_demo:
            print(f"Default demo database created: {selection.db_path}")

        team = DatabaseDiagnosisTeam(
            db_path=str(selection.db_path),
            report_path=str(report_path),
            write_enabled=bool(args.write),
        )
        result = team.run()
        agsync(team)

        print(f"\nReport path : {result.report_path}")
        print(f"Database    : {result.db_path}")
        print(f"Root cause  : {result.root_cause}")
        print(f"Fix applied : {result.fix_applied}")
        print(f"Rolled back : {result.rolled_back}")
        return 0
    except (AgError, RuntimeError, ValueError, sqlite3.DatabaseError, OSError) as exc:
        print(f"\nERROR: {exc}")
        return 1


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    run_dir = _make_run_dir("database_diagnosis")
    agent.log_dir = run_dir / "logs"
    agent.output_dir = run_dir / "agent_output"

    if args.webui:
        from agency.agwebui import agwebui

        status = 1

        def _run_with_status() -> None:
            nonlocal status
            status = run_diagnosis(args, run_dir)

        agwebui.run(
            _run_with_status,
            run_dir=run_dir,
            port=args.webui_port,
            linger=not args.no_webui_linger,
        )
        return status

    return run_diagnosis(args, run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
