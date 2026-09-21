"""Replays a real bug_localization recording through the full agent lifecycle using the Mock backend.

Runs against golden_image (python:3.12-slim), not the SWE-bench image the recordings were
made against. This works because replay matches by position, not by what a tool call returns
(agency/llm/mock.py), so a tool call against the wrong filesystem can fail, but replay
hands back the next recorded answer anyway.

Data needed: tests/fixtures/bug_localization/<harness>/{recording.sqlite3,result.json}, frozen
from agency-benchmarks/targeted/bug_localization's own output/, instance
django__django-11066.

To regenerate, see the README at agency/tests/fixtures/bug_localization/README.md
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from agency import Agent, agconfig, agdata, agskill
from agency.configs.agconfig import agentconfig, llmconfig, sandboxconfig

from ..test_golden_execution import CONTAINER_BACKEND  # noqa: F401 -- fixture

HARNESSES = ("native", "claude_code", "codex")
FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "bug_localization"


def _bug_localizer_skill() -> agskill:
    """Field names must match bug_localization's own build_localizer_skill(). The
    recording's submit_output calls name these fields; a mismatch fails schema
    validation here, not the replay."""
    return agskill(
        "bug_localizer",
        "Report which files need to change.",
        input_schema=agdata(problem_statement=str),
        output_schema=agdata(relevant_files=list[str], reasoning=str),
        max_output_schema_retries=0,
    )


def _tool_call_count(db_path: Path) -> int:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return conn.execute("SELECT count(*) FROM events WHERE type='tool_call'").fetchone()[0]
    finally:
        conn.close()


@pytest.mark.timeout(180)
@pytest.mark.parametrize("harness", HARNESSES)
def test_replays_a_real_recording_end_to_end(harness, golden_image, tmp_path) -> None:
    fixture_dir = FIXTURES_DIR / harness
    expected = json.loads((fixture_dir / "result.json").read_text())
    recording = fixture_dir / "recording.sqlite3"

    config = agconfig(
        agentconfig(harness=harness, log_dir=str(tmp_path / "logs")),
        sandboxconfig(backend=CONTAINER_BACKEND, base_image=golden_image),
        llmconfig(
            provider="mock",
            model="claude-sonnet-5",
            context_limit=196_000,
            replay_db_path=str(recording),
            timing_mode="instant",
        ),
    )
    ag = Agent(agconfig=config)
    result = ag.run(
        _bug_localizer_skill(),
        agdata(problem_statement="(replayed -- ignored)"),
        max_steps=60,
    ).wait()

    data = result.to_dict()
    assert "error" not in data, data.get("error")
    assert data["relevant_files"] == expected["reported_files"]

    own_log = next((tmp_path / "logs").glob("agent_*_data.sqlite3"))
    assert _tool_call_count(own_log) == _tool_call_count(recording)
