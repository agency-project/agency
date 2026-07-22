"""Tests for agpath — path-only string agskill schema field.

Unlike agfile/agbinary, agpath never reads or writes the sandbox filesystem;
it only checks that a value looks like a path, on both the input and output
side. It exists because a plain `str` output field silently auto-resolves a
path-looking value to that file's contents (see agschema.make_field_handler),
which is wrong for a field whose value is meant to *stay* a path.
"""

import json
from unittest.mock import MagicMock, patch
from agency.agdata import agdata, agerror
from agency.agcontext import agcontext
from agency.agconfig import agConfig
from agency.agschema import agschema
from agency.agtype import agtype, agpath
from agency.agskill import agskill
from agency.agllm import agllm

LLM_CONFIG = {"api_key": "test", "model": ""}
LLM = agllm(agConfig({"agllm_backend": LLM_CONFIG}), context_limit=128_000)


def _make_mock_agent(llm=None, sandbox=None):
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
    ag.llm = llm or LLM
    if sandbox is not None:
        ag.sandbox = sandbox
    else:
        ag.sandbox = MagicMock()
        # A bare MagicMock()'s _has_pending_background_work() would
        # otherwise auto-mock to a truthy value, making agtool.py's
        # dispatch_tools() defer stop() forever -- default to "nothing
        # pending" so tests get the common case without configuring it.
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


# ---------------------------------------------------------------------------
# Streaming mock helpers (same pattern as test_agfile.py)
# ---------------------------------------------------------------------------


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.model_extra = {}
        self.reasoning_content = None


class _Choice:
    def __init__(self, delta):
        self.delta = delta


class _Usage:
    prompt_tokens = 5


class _Chunk:
    def __init__(self, content=None, tool_calls=None, usage=None):
        self.choices = (
            [_Choice(_Delta(content, tool_calls))] if (content is not None or tool_calls) else []
        )
        self.usage = usage


class _TCDelta:
    def __init__(self, name, args_json, call_id):
        self.id = call_id
        self.index = 0
        self.function = _TCFnDelta(name, args_json)


class _TCFnDelta:
    def __init__(self, name, args):
        self.name = name
        self.arguments = args


def _direct(content: str):
    return [_Chunk(content=content), _Chunk(usage=_Usage())]


def _tool_call(name: str, args: dict, call_id: str = "c1") -> list:
    tc = _TCDelta(name, json.dumps(args), call_id)
    return [_Chunk(tool_calls=[tc]), _Chunk(usage=_Usage())]


# ---------------------------------------------------------------------------
# agpath class — basic classmethods
# ---------------------------------------------------------------------------


def test_agpath_is_agtype_subclass():
    assert issubclass(agpath, agtype)


def test_agpath_schema_type():
    assert agpath.schema_type() == "path"


def test_agpath_needs_sandbox():
    assert agpath.needs_sandbox() is False


def test_agpath_prepare_passthrough():
    val, paths = agpath.prepare("/workspace/out.txt", None, "skill", "field")
    assert val == "/workspace/out.txt"
    assert paths == []


def test_agpath_recover_passthrough():
    val, paths = agpath.recover("/workspace/out.txt", None)
    assert val == "/workspace/out.txt"
    assert paths == []


def test_agpath_extra_input_prompt_mentions_path():
    assert "path" in agpath.extra_input_prompt("dest").lower()


def test_agpath_extra_output_prompt_warns_against_content():
    prompt = agpath.extra_output_prompt("dest", "skill")
    assert "path" in prompt.lower()
    assert "content" in prompt.lower()


def test_agpath_get_return_tool_value_description_warns_against_content():
    desc = agpath.get_return_tool_value_description("dest")
    assert "content" in desc.lower()


def test_agpath_validate_input_value_accepts_path():
    assert agpath.validate_input_value("/workspace/out.txt") is None


def test_agpath_validate_input_value_rejects_non_path_string():
    err = agpath.validate_input_value("this is not a path")
    assert err is not None
    assert "does not look like a path" in err


def test_agpath_validate_input_value_rejects_non_string():
    err = agpath.validate_input_value(123)
    assert err is not None
    assert "must be a string" in err


def test_agpath_validate_output_accepts_path():
    assert agpath.validate_output("dest", "/data/note.txt", None, 5) is None


def test_agpath_validate_output_rejects_non_path_value():
    err = agpath.validate_output("dest", "The quick brown fox jumps.", None, 5)
    assert err is not None
    assert "does not look like a path" in err


def test_agpath_validate_output_rejects_non_string_value():
    assert agpath.validate_output("dest", 123, None, 5) is not None


# ---------------------------------------------------------------------------
# agdata serialization
# ---------------------------------------------------------------------------


def test_agdata_serializes_agpath_as_path():
    d = agdata(dest=agpath)
    assert json.loads(d.to_json()) == {"dest": "path"}


# ---------------------------------------------------------------------------
# agschema.check — schema validation
# ---------------------------------------------------------------------------


def test_check_schema_agpath_hint_accepts_path_string():
    s = agschema(agdata(dest=agpath))
    assert s.check(agdata(dest="/workspace/out.txt")) == []


def test_check_schema_agpath_hint_rejects_non_path_string():
    s = agschema(agdata(dest=agpath))
    errors = s.check(agdata(dest="not a path"))
    assert errors
    assert "does not look like a path" in errors[0]


def test_check_schema_agpath_hint_rejects_non_string():
    s = agschema(agdata(dest=agpath))
    errors = s.check(agdata(dest=123))
    assert errors


# ---------------------------------------------------------------------------
# agskill._build_system_prompt — agpath prompts injected
# ---------------------------------------------------------------------------


def test_system_prompt_includes_agpath_input_instructions():
    sk = agskill(
        "move",
        "Do stuff.",
        input_schema=agdata(dest=agpath),
        output_schema=agdata(result=str),
    )
    prompt = sk._build_system_prompt()
    assert "dest" in prompt
    assert "path" in prompt.lower()


def test_system_prompt_includes_agpath_output_instructions():
    sk = agskill(
        "move",
        "Do stuff.",
        input_schema=agdata(theme=str),
        output_schema=agdata(moved_to=agpath),
    )
    prompt = sk._build_system_prompt()
    assert "moved_to" in prompt
    assert "content" in prompt.lower()


def test_system_prompt_agpath_type_shown_as_path():
    sk = agskill("t", "", input_schema=agdata(dest=agpath))
    prompt = sk._build_system_prompt()
    assert '"dest": "path"' in prompt


# ---------------------------------------------------------------------------
# return_<field> tool — agpath validation during the tool call
# ---------------------------------------------------------------------------


def _run_skill_with_sandbox(skill, responses, sandbox, skill_input=None):
    """Helper: run skill with mocked LLM and a provided sandbox."""
    sandbox._has_pending_background_work.return_value = False
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, *_ = skill.execute_react(
            _make_mock_agent(LLM, sandbox), agcontext(), skill_input or agdata()
        )
    return result


def test_return_agpath_valid_path_is_accepted():
    sk = agskill("move", "", output_schema=agdata(moved_to=agpath), max_output_schema_retries=0)
    sandbox = MagicMock()
    responses = [
        _tool_call("return_moved_to", {"value": "/data/out.txt"}),
        _direct(""),
    ]
    result = _run_skill_with_sandbox(sk, responses, sandbox)
    assert result.moved_to == "/data/out.txt"
    # agpath never reads the sandbox filesystem, unlike agfile.
    sandbox.read_file.assert_not_called()


def test_return_agpath_non_path_value_returns_error_and_reprompts():
    """A value that doesn't look like a path is rejected and the agent is reprompted,
    not immediately failed — mirrors agfile's live-validation reprompt behaviour."""
    sk = agskill("move", "", output_schema=agdata(moved_to=agpath), max_output_schema_retries=3)
    sandbox = MagicMock()
    responses = [
        _tool_call("return_moved_to", {"value": "here is the file content, not a path"}),
        _tool_call("return_moved_to", {"value": "/data/out.txt"}),
        _direct(""),
    ]
    result = _run_skill_with_sandbox(sk, responses, sandbox)
    assert result.moved_to == "/data/out.txt"


def test_return_agpath_non_path_value_exhausts_retries():
    sk = agskill("move", "", output_schema=agdata(moved_to=agpath), max_output_schema_retries=0)
    sandbox = MagicMock()
    responses = [
        _tool_call("return_moved_to", {"value": "here is the file content, not a path"}),
        _direct(""),
    ]
    result = _run_skill_with_sandbox(sk, responses, sandbox)
    assert isinstance(result, agerror) or result._data.get("moved_to") is None


def test_input_agpath_rejects_non_path_value_before_llm_call():
    sk = agskill("move", "", input_schema=agdata(dest=agpath), output_schema=agdata(result=str))
    sandbox = MagicMock()
    sandbox._has_pending_background_work.return_value = False
    with patch("openai.OpenAI") as MockClient:
        result, *_ = sk.execute_react(
            _make_mock_agent(LLM, sandbox), agcontext(), agdata(dest="not a path")
        )
        # No LLM call should have been made — input validation fails first.
        MockClient.return_value.chat.completions.create.assert_not_called()
    assert isinstance(result, agerror)
    assert "does not look like a path" in result._data["error"]


# ---------------------------------------------------------------------------
# agpath vs the plain-str auto-resolve fallback
# ---------------------------------------------------------------------------


def test_return_agpath_does_not_auto_resolve_to_file_contents():
    """The plain-str shortcut (path-looking value -> file contents) must NOT
    apply to agpath fields -- this is the exact bug agpath exists to avoid."""
    sk = agskill(
        "write", "", output_schema=agdata(path=agpath, content=str), max_output_schema_retries=0
    )
    sandbox = MagicMock()
    sandbox.read_file.return_value = "The quick brown fox jumps over the lazy dog."
    responses = [
        _tool_call("return_content", {"value": "The quick brown fox jumps over the lazy dog."}),
        _tool_call("return_path", {"value": "/data/note.txt"}),
        _direct(""),
    ]
    result = _run_skill_with_sandbox(sk, responses, sandbox)
    assert result.path == "/data/note.txt"
    assert result.content == "The quick brown fox jumps over the lazy dog."


def test_return_str_path_field_still_auto_resolves_and_warns(capsys):
    """A plain `str` field (not agpath) still gets the auto-resolve fallback,
    but now prints a warning telling the caller to use agpath instead."""
    sk = agskill("write", "", output_schema=agdata(path=str), max_output_schema_retries=0)
    sandbox = MagicMock()
    sandbox.read_file.return_value = "the note's actual content"
    responses = [
        _tool_call("return_path", {"value": "/data/note.txt"}),
        _direct(""),
    ]
    result = _run_skill_with_sandbox(sk, responses, sandbox)
    assert result.path == "the note's actual content"
    captured = capsys.readouterr()
    assert "[agschema] WARNING" in captured.out
    assert "agpath" in captured.out
