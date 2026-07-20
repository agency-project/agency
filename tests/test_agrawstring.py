"""Tests for agrawstring — raw string bypass mode."""

import json
from unittest.mock import MagicMock, patch

from agency.agdata import agdata
from agency.agcontext import agcontext
from agency.agtype import agtype, agrawstring
from agency.agskill import agskill


# ---------------------------------------------------------------------------
# agrawstring class
# ---------------------------------------------------------------------------


def test_agrawstring_is_agtype_subclass():
    assert issubclass(agrawstring, agtype)


def test_agrawstring_schema_type():
    assert agrawstring.schema_type() == "str"


def test_agrawstring_needs_no_sandbox():
    assert agrawstring.needs_sandbox() is False


def test_agrawstring_prepare_passthrough():
    val, paths = agrawstring.prepare("hello world", None, "sk", "content")
    assert val == "hello world"
    assert paths == []


def test_agrawstring_recover_passthrough():
    val, paths = agrawstring.recover("some output", None)
    assert val == "some output"
    assert paths == []


# ---------------------------------------------------------------------------
# _build_system_prompt — JSON blocks omitted for agrawstring
# ---------------------------------------------------------------------------


def test_system_prompt_no_input_json_when_raw_input():
    sk = agskill("t", "Write a story.", input_schema=agdata(prompt=agrawstring))
    prompt = sk._build_system_prompt()
    assert "Input JSON format" not in prompt


def test_system_prompt_no_output_json_when_raw_output():
    sk = agskill("t", "Write a story.", output_schema=agdata(story=agrawstring))
    prompt = sk._build_system_prompt()
    assert "Output JSON format" not in prompt
    assert "plain text" in prompt.lower()


def test_system_prompt_keeps_json_for_normal_output():
    sk = agskill("t", "Summarise.", output_schema=agdata(summary=str))
    prompt = sk._build_system_prompt()
    assert "return_summary" in prompt


def test_system_prompt_keeps_input_json_for_normal_input():
    sk = agskill("t", "Summarise.", input_schema=agdata(text=str))
    prompt = sk._build_system_prompt()
    assert "Input JSON format" in prompt


# ---------------------------------------------------------------------------
# _build_user_content — raw passthrough for agrawstring input
# ---------------------------------------------------------------------------


def test_build_user_content_raw_input_returns_plain_string():
    sk = agskill("t", "", input_schema=agdata(prompt=agrawstring))
    inp = agdata(prompt="Tell me a story about a robot.")
    content = sk._build_user_content(inp)
    assert content == "Tell me a story about a robot."


def test_build_user_content_raw_input_no_json_wrapping():
    sk = agskill("t", "", input_schema=agdata(prompt=agrawstring))
    inp = agdata(prompt="Hello!")
    content = sk._build_user_content(inp)
    assert not content.startswith("{")


def test_build_user_content_normal_input_still_json():
    sk = agskill("t", "", input_schema=agdata(text=str))
    inp = agdata(text="hello")
    content = sk._build_user_content(inp)
    assert isinstance(content, str)
    parsed = json.loads(content.split("\n", 1)[1])
    assert parsed["text"] == "hello"


# ---------------------------------------------------------------------------
# agskill.run — raw output captures full response, no JSON parsing
# ---------------------------------------------------------------------------


def _make_chunk(text: str, finish: str = "stop"):
    """Build a minimal streaming chunk mock."""
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = text
    chunk.choices[0].delta.tool_calls = None
    chunk.choices[0].finish_reason = finish
    chunk.usage = None
    return chunk


def _make_mock_agent(llm):
    from agency.agent import agent as _agent_cls

    class _Cls:
        agresource_pool = MagicMock()
        ping_interval_s = 300
        poll_interval_s = 5
        agconfig = None
        _drain_inbox = _agent_cls._drain_inbox
        _check_pause = _agent_cls._check_pause

    ag = _Cls()
    from agency.agent import agent as _agent_cls, agent_state as _agent_state_cls

    ag._state = _agent_state_cls("test")
    ag.llm = llm
    ag.sandbox = MagicMock()
    # A bare MagicMock()'s _has_pending_background_work() would otherwise
    # auto-mock to a truthy value, making agtool.py's dispatch_tools() defer
    # stop() forever -- default to "nothing pending" so tests get the common
    # case without configuring it.
    ag.sandbox._has_pending_background_work.return_value = False
    ag.terminal = MagicMock()
    ag.log = MagicMock()
    ag.log.token_usage = {}
    ag.agname = "test"
    ag._set_ui_state = MagicMock()
    ag._push_live_messages = MagicMock()
    ag._append_full_history = MagicMock()
    ag._next_inbox_msg = MagicMock(return_value=None)
    ag.push_token_count_update_to_ui = MagicMock()
    return ag


def _run_skill_with_mock_response(sk, inp, response_text):
    """Run a skill, mocking the LLM to return response_text as a single chunk."""
    chunks = [_make_chunk(response_text, "stop"), _make_chunk("", "stop")]
    with patch("agency.agllm.openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = iter(chunks)
        from agency.agllm import agllm as _agllm
        from agency.agconfig import agConfig as _agConfig

        _llm = _agllm(
            _agConfig({"agllm_backend": {"base_url": "http://x", "api_key": "", "model": "m"}}),
            context_limit=128_000,
        )
        result, *_ = sk.execute_react(_make_mock_agent(_llm), agcontext(), inp, max_steps=5)
    return result


def test_raw_output_captures_full_response():
    sk = agskill("t", "Write prose.", output_schema=agdata(story=agrawstring))
    inp = agdata()
    result = _run_skill_with_mock_response(sk, inp, "Once upon a time in a land far away.")
    assert result.story == "Once upon a time in a land far away."


def test_raw_output_does_not_parse_as_json():
    sk = agskill("t", "Write prose.", output_schema=agdata(story=agrawstring))
    inp = agdata()
    # A response that is not valid JSON at all
    result = _run_skill_with_mock_response(sk, inp, "This is plain prose, not JSON!")
    assert result.story == "This is plain prose, not JSON!"


def test_raw_output_with_raw_input():
    sk = agskill(
        "t",
        "Write prose.",
        input_schema=agdata(prompt=agrawstring),
        output_schema=agdata(story=agrawstring),
    )
    inp = agdata(prompt="Begin the adventure.")
    result = _run_skill_with_mock_response(sk, inp, "The hero stepped forward.")
    assert result.story == "The hero stepped forward."


def test_raw_output_preserves_newlines_and_quotes():
    sk = agskill("t", "Write.", output_schema=agdata(text=agrawstring))
    inp = agdata()
    prose = 'He said "hello"\nShe replied "goodbye"'
    result = _run_skill_with_mock_response(sk, inp, prose)
    assert result.text == prose
