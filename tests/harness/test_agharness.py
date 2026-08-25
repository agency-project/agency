"""Tests for agharness.py -- the thin, engine-agnostic glue shared by every
agharness_backends/* concrete backend."""

from __future__ import annotations

from unittest.mock import MagicMock

from agency.harness import agharness
from agency.agdata import agdata
from agency.agskill import agskill


def _make_agent(agname="test-agent"):
    ag = MagicMock()
    ag.agname = agname
    return ag


def test_materialize_config_home_creates_isolated_directory():
    ag = _make_agent()
    d1 = agharness.materialize_config_home(ag, token="t1", base_url="http://x")
    d2 = agharness.materialize_config_home(ag, token="t2", base_url="http://x")
    assert d1.is_dir()
    assert d2.is_dir()
    assert d1 != d2  # each launch gets its own directory
    agharness.cleanup_config_home(d1)
    agharness.cleanup_config_home(d2)
    assert not d1.exists()
    assert not d2.exists()


def test_cleanup_config_home_is_idempotent(tmp_path):
    d = tmp_path / "nonexistent"
    agharness.cleanup_config_home(d)  # must not raise


def test_build_user_turn_prompt_delegates_to_skill():
    skill = agskill(name="s", system_prompt="do the thing")
    content = agharness.build_user_turn_prompt(skill, agdata(task="go"))
    assert "go" in content if isinstance(content, str) else True
    assert content == skill._build_user_content(agdata(task="go"))


def test_build_output_format_instruction_none_when_no_schema():
    skill = agskill(name="s", system_prompt="do the thing")
    assert agharness.build_output_format_instruction(skill) is None


def test_build_output_format_instruction_none_for_raw_string_schema():
    from agency.agtype import agrawstring

    skill = agskill(name="s", system_prompt="do the thing", output_schema=agdata(text=agrawstring))
    assert agharness.build_output_format_instruction(skill) is None


def test_build_output_format_instruction_present_for_structured_schema():
    skill = agskill(
        name="s", system_prompt="do the thing", output_schema=agdata(answer=str, count=int)
    )
    instruction = agharness.build_output_format_instruction(skill)
    assert instruction is not None
    assert "JSON object" in instruction
    assert "answer" in instruction
    assert "count" in instruction
