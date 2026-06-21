"""Tests for agfile — file-backed agskill schema field."""
import json
import pytest
from unittest.mock import MagicMock, patch
from agency.agdata import agdata
from agency.agtype import agtype, agfile
from agency.agskill import agskill

LLM_CONFIG = {"api_key": "test", "model": "gpt-4o"}


# ---------------------------------------------------------------------------
# Streaming mock helpers
# ---------------------------------------------------------------------------

class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.model_extra = {}
        self.reasoning_content = None

class _Choice:
    def __init__(self, delta): self.delta = delta

class _Usage:
    prompt_tokens = 5

class _Chunk:
    def __init__(self, content=None, tool_calls=None, usage=None):
        self.choices = [_Choice(_Delta(content, tool_calls))] if (content is not None or tool_calls) else []
        self.usage = usage

class _TCDelta:
    def __init__(self, name, args_json, call_id):
        self.id = call_id
        self.index = 0
        self.function = _TCFnDelta(name, args_json)

class _TCFnDelta:
    def __init__(self, name, args): self.name = name; self.arguments = args

def _direct(content: str):
    return [_Chunk(content=content), _Chunk(usage=_Usage())]

def _tool_call(name: str, args: dict, call_id: str = "c1") -> list:
    tc = _TCDelta(name, json.dumps(args), call_id)
    return [_Chunk(tool_calls=[tc]), _Chunk(usage=_Usage())]


# ---------------------------------------------------------------------------
# agfile class
# ---------------------------------------------------------------------------

def test_agfile_is_agtype_subclass():
    assert issubclass(agfile, agtype)

def test_agfile_schema_type():
    assert agfile.schema_type() == "file"

def test_agfile_needs_sandbox():
    assert agfile.needs_sandbox() is True

def test_agfile_extra_input_prompt_mentions_read_tool():
    prompt = agfile.extra_input_prompt("background")
    assert "background" in prompt
    assert "read tool" in prompt

def test_agfile_extra_output_prompt_mentions_write():
    prompt = agfile.extra_output_prompt("report", "design")
    assert "report" in prompt
    assert "write" in prompt.lower()
    assert "report.txt" in prompt

def test_agfile_extra_output_prompt_mentions_chunked_writes():
    prompt = agfile.extra_output_prompt("doc", "write")
    assert "multiple" in prompt.lower() or "append" in prompt.lower() or "chunk" in prompt.lower()

def test_agfile_prepare_writes_to_sandbox_and_returns_path():
    sandbox = MagicMock()
    val, paths = agfile.prepare("file content", sandbox, "miskill", "myfield")
    sandbox.write_file.assert_called_once_with(
        "/workspace/inputs/myfield.txt", "file content"
    )
    assert val == "/workspace/inputs/myfield.txt"
    assert paths == ["/workspace/inputs/myfield.txt"]

def test_agfile_prepare_non_string_passthrough():
    sandbox = MagicMock()
    val, paths = agfile.prepare(42, sandbox, "skill", "field")
    sandbox.write_file.assert_not_called()
    assert val == 42
    assert paths == []

def test_agfile_prepare_sandbox_failure_leaves_value_unchanged():
    sandbox = MagicMock()
    sandbox.write_file.side_effect = OSError("disk full")
    val, paths = agfile.prepare("content", sandbox, "skill", "field")
    assert val == "content"
    assert paths == []

def test_agfile_recover_reads_from_sandbox_and_returns_content():
    sandbox = MagicMock()
    sandbox.read_file.return_value = "recovered content"
    val, paths = agfile.recover("/workspace/outputs/skill_field.txt", sandbox)
    sandbox.read_file.assert_called_once_with("/workspace/outputs/skill_field.txt")
    assert val == "recovered content"
    assert paths == ["/workspace/outputs/skill_field.txt"]

def test_agfile_recover_non_string_passthrough():
    sandbox = MagicMock()
    val, paths = agfile.recover(None, sandbox)
    sandbox.read_file.assert_not_called()
    assert val is None
    assert paths == []

def test_agfile_recover_sandbox_failure_leaves_path_unchanged():
    sandbox = MagicMock()
    sandbox.read_file.side_effect = OSError("not found")
    val, paths = agfile.recover("/some/path.txt", sandbox)
    assert val == "/some/path.txt"
    assert paths == []


# ---------------------------------------------------------------------------
# agdata serialization
# ---------------------------------------------------------------------------

def test_agdata_serializes_agfile_as_file():
    d = agdata(doc=agfile)
    assert json.loads(d.to_json()) == {"doc": "file"}


# ---------------------------------------------------------------------------
# agskill._check_schema — agfile hints
# ---------------------------------------------------------------------------

def test_check_schema_agfile_hint_accepts_string():
    sk = agskill("t", "", output_schema=agdata(doc=agfile))
    assert sk._check_schema(agdata(doc="/workspace/out.txt"), sk.output_schema) == []

def test_check_schema_agfile_hint_rejects_non_string():
    sk = agskill("t", "", output_schema=agdata(doc=agfile))
    errors = sk._check_schema(agdata(doc=123), sk.output_schema)
    assert len(errors) == 1
    assert "doc" in errors[0]


# ---------------------------------------------------------------------------
# agskill._build_system_prompt — agfile prompts injected
# ---------------------------------------------------------------------------

def test_system_prompt_includes_agfile_input_instructions():
    sk = agskill(
        "design", "Do stuff.",
        input_schema=agdata(background=agfile),
        output_schema=agdata(result=str),
    )
    prompt = sk._build_system_prompt()
    assert "File-backed fields" in prompt
    assert "background" in prompt
    assert "read tool" in prompt

def test_system_prompt_includes_agfile_output_instructions():
    sk = agskill(
        "design", "Do stuff.",
        input_schema=agdata(theme=str),
        output_schema=agdata(report=agfile),
    )
    prompt = sk._build_system_prompt()
    assert "File-backed fields" in prompt
    assert "report" in prompt
    assert "write" in prompt.lower()

def test_system_prompt_no_agfile_fields_no_file_backed_warning():
    sk = agskill("t", "Do stuff.", input_schema=agdata(x=str), output_schema=agdata(y=int))
    prompt = sk._build_system_prompt()
    assert "File-backed" not in prompt

def test_system_prompt_agfile_type_shown_as_file():
    sk = agskill("t", "", input_schema=agdata(doc=agfile))
    prompt = sk._build_system_prompt()
    assert '"doc": "file"' in prompt


# ---------------------------------------------------------------------------
# agskill ReAct loop — agfile output schema accepts path string
# ---------------------------------------------------------------------------

def test_skill_with_agfile_output_schema_validates_path_string():
    sk = agskill("write", "", output_schema=agdata(doc=agfile), max_output_schema_retries=0)
    responses = [
        _tool_call("return_doc", {"value": "/workspace/outputs/write_doc.txt"}),
        _direct(""),
    ]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, *_ = sk.run(LLM_CONFIG, agdata(), agdata(messages=[]), sandbox=None)
    assert result.doc == "/workspace/outputs/write_doc.txt"


# ---------------------------------------------------------------------------
# _prepare_agtype_inputs / _recover_agtype_outputs
# ---------------------------------------------------------------------------

def test_prepare_agtype_inputs_calls_prepare_on_agfile_fields():
    from agency.agent import _prepare_agtype_inputs
    sandbox = MagicMock()
    inp = agdata(theme="space opera", background="long background text")
    schema = agdata(theme=str, background=agfile)
    paths = _prepare_agtype_inputs(inp, schema, sandbox, "design")
    sandbox.write_file.assert_called_once_with(
        "/workspace/inputs/background.txt", "long background text"
    )
    assert inp._data["background"] == "/workspace/inputs/background.txt"
    assert inp._data["theme"] == "space opera"
    assert len(paths) == 1

def test_prepare_agtype_inputs_no_schema_returns_empty():
    from agency.agent import _prepare_agtype_inputs
    sandbox = MagicMock()
    inp = agdata(x="hello")
    paths = _prepare_agtype_inputs(inp, None, sandbox, "skill")
    assert paths == []
    sandbox.write_file.assert_not_called()

def test_recover_agtype_outputs_reads_file_and_replaces_path():
    from agency.agent import _recover_agtype_outputs
    sandbox = MagicMock()
    sandbox.read_file.return_value = "report content"
    result = agdata(report="/workspace/outputs/design_report.txt")
    schema = agdata(report=agfile)
    paths = _recover_agtype_outputs(result, schema, sandbox)
    assert result._data["report"] == "report content"
    assert paths == ["/workspace/outputs/design_report.txt"]

def test_recover_agtype_outputs_skips_error_result():
    from agency.agent import _recover_agtype_outputs
    sandbox = MagicMock()
    result = agdata(error="something went wrong")
    schema = agdata(report=agfile)
    paths = _recover_agtype_outputs(result, schema, sandbox)
    sandbox.read_file.assert_not_called()
    assert paths == []

def test_recover_agtype_outputs_no_schema_returns_empty():
    from agency.agent import _recover_agtype_outputs
    sandbox = MagicMock()
    result = agdata(report="/some/path.txt")
    paths = _recover_agtype_outputs(result, None, sandbox)
    assert paths == []
    sandbox.read_file.assert_not_called()
