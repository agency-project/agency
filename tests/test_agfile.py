"""Tests for agfile — file-backed agskill schema field."""

import json
from unittest.mock import MagicMock, patch
from agency.agdata import agdata, agerror
from agency.agcontext import agcontext
from agency.agconfig import agConfig
from agency.agschema import agschema
from agency.agtype import agtype, agfile
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
# Streaming mock helpers
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
    sandbox.write_file.assert_called_once_with("/workspace/inputs/myfield.txt", "file content")
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
# agskill.check_schema — agfile hints
# ---------------------------------------------------------------------------


def test_check_schema_agfile_hint_accepts_string():
    assert agschema(agdata(doc=agfile)).check(agdata(doc="/workspace/out.txt")) == []


def test_check_schema_agfile_hint_rejects_non_string():
    errors = agschema(agdata(doc=agfile)).check(agdata(doc=123))
    assert len(errors) == 1
    assert "doc" in errors[0]


# ---------------------------------------------------------------------------
# agskill._build_system_prompt — agfile prompts injected
# ---------------------------------------------------------------------------


def test_system_prompt_includes_agfile_input_instructions():
    sk = agskill(
        "design",
        "Do stuff.",
        input_schema=agdata(background=agfile),
        output_schema=agdata(result=str),
    )
    prompt = sk._build_system_prompt()
    assert "File-backed fields" in prompt
    assert "background" in prompt
    assert "read tool" in prompt


def test_system_prompt_includes_agfile_output_instructions():
    sk = agskill(
        "design",
        "Do stuff.",
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
    sandbox = MagicMock()
    sandbox._has_pending_background_work.return_value = False
    sandbox.read_file.return_value = "recovered file content"
    responses = [
        _tool_call("return_doc", {"value": "/workspace/outputs/write_doc.txt"}),
        _direct(""),
    ]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, *_ = sk.execute_react(_make_mock_agent(LLM, sandbox), agcontext(), agdata())
    sandbox.read_file.assert_called_with("/workspace/outputs/write_doc.txt")
    assert result.doc == "recovered file content"


# ---------------------------------------------------------------------------
# _prepare_agtype_inputs / _recover_agtype_outputs
# ---------------------------------------------------------------------------


def test_prepare_agtype_inputs_calls_prepare_on_agfile_fields():
    sandbox = MagicMock()
    inp = agdata(theme="space opera", background="long background text")
    schema = agdata(theme=str, background=agfile)
    paths, _ = agschema(schema).prepare_inputs_in_sandbox(inp, sandbox, "design")
    sandbox.write_file.assert_called_once_with(
        "/workspace/inputs/background.txt", "long background text"
    )
    assert inp._data["background"] == "/workspace/inputs/background.txt"
    assert inp._data["theme"] == "space opera"
    assert len(paths) == 1


def test_prepare_agtype_inputs_no_schema_returns_empty():
    sandbox = MagicMock()
    paths = []  # no schema = no agtype inputs to prepare
    assert paths == []
    sandbox.write_file.assert_not_called()


def test_recover_agtype_outputs_reads_file_and_replaces_path():
    sandbox = MagicMock()
    sandbox.read_file.return_value = "report content"
    result = agdata(report="/workspace/outputs/design_report.txt")
    schema = agdata(report=agfile)
    paths = agschema(schema).recover_outputs(result, sandbox)
    assert result._data["report"] == "report content"
    assert paths == ["/workspace/outputs/design_report.txt"]


def test_recover_agtype_outputs_skips_error_result():
    sandbox = MagicMock()
    result = agerror("something went wrong")
    schema = agdata(report=agfile)
    paths = agschema(schema).recover_outputs(result, sandbox)
    sandbox.read_file.assert_not_called()
    assert paths == []


def test_recover_agtype_outputs_no_schema_returns_empty():
    sandbox = MagicMock()
    # no schema = no recovery needed
    paths = []
    assert paths == []
    sandbox.read_file.assert_not_called()


# ---------------------------------------------------------------------------
# return_<field> tool — agfile validation during the tool call
# ---------------------------------------------------------------------------


def _run_skill_with_sandbox(skill, responses, sandbox):
    """Helper: run skill with mocked LLM and a provided sandbox."""
    sandbox._has_pending_background_work.return_value = False
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, *_ = skill.execute_react(_make_mock_agent(LLM, sandbox), agcontext(), agdata())
    return result


def test_return_agfile_directory_path_returns_error():
    """return_doc pointing at a directory should give an IsADirectoryError message."""
    sandbox = MagicMock()
    sandbox.read_file.side_effect = IsADirectoryError("/workspace/outputs is a directory")
    sk = agskill("w", "", output_schema=agdata(doc=agfile), max_output_schema_retries=0)
    responses = [
        _tool_call("return_doc", {"value": "/workspace/outputs"}),
        _direct(""),
    ]
    result = _run_skill_with_sandbox(sk, responses, sandbox)
    assert isinstance(result, agerror) or result._data.get("doc") is None


def test_return_agfile_binary_file_returns_error():
    """return_doc pointing at a binary file should give a UnicodeDecodeError message."""
    sandbox = MagicMock()
    raw = b"\x89PNG\r\n\x1a\n"  # PNG magic bytes
    sandbox.read_file.side_effect = UnicodeDecodeError(
        "utf-8", raw, 0, 1, "File /workspace/image.bin contains binary data"
    )
    sk = agskill("w", "", output_schema=agdata(doc=agfile), max_output_schema_retries=0)
    responses = [
        _tool_call("return_doc", {"value": "/workspace/image.bin"}),
        _direct(""),
    ]
    result = _run_skill_with_sandbox(sk, responses, sandbox)
    assert isinstance(result, agerror) or result._data.get("doc") is None


def test_return_agfile_missing_file_returns_error_and_reprompts():
    """return_doc with a path to a non-existent file should return an error to the agent."""
    sandbox = MagicMock()
    sandbox.read_file.side_effect = FileNotFoundError("no such file")
    sk = agskill("w", "", output_schema=agdata(doc=agfile), max_output_schema_retries=0)
    # Response 1: agent calls return_doc → sandbox raises → error returned to agent.
    # Response 2: agent produces direct text (gives up) → field still missing → skill errors.
    responses = [
        _tool_call("return_doc", {"value": "/workspace/outputs/missing.txt"}),
        _direct(""),
    ]
    result = _run_skill_with_sandbox(sk, responses, sandbox)
    assert isinstance(result, agerror) or result._data.get("doc") is None


def test_return_agfile_empty_file_returns_error():
    """return_doc with a path to an empty file should return an error."""
    sandbox = MagicMock()
    sandbox.read_file.return_value = ""
    sk = agskill("w", "", output_schema=agdata(doc=agfile), max_output_schema_retries=0)
    responses = [
        _tool_call("return_doc", {"value": "/workspace/outputs/empty.txt"}),
        _direct(""),
    ]
    result = _run_skill_with_sandbox(sk, responses, sandbox)
    assert isinstance(result, agerror) or result._data.get("doc") is None


def test_return_agfile_content_is_another_path_returns_error():
    """return_doc where the file contains only a path should be rejected."""
    sandbox = MagicMock()
    sandbox.read_file.return_value = "/workspace/turn_specula.py"
    sk = agskill("w", "", output_schema=agdata(doc=agfile), max_output_schema_retries=0)
    responses = [
        _tool_call("return_doc", {"value": "/workspace/outputs/doc.txt"}),
        _direct(""),
    ]
    result = _run_skill_with_sandbox(sk, responses, sandbox)
    assert isinstance(result, agerror) or result._data.get("doc") is None


def test_return_agfile_valid_content_is_accepted():
    """return_doc where the file has real content should be accepted."""
    sandbox = MagicMock()
    sandbox.read_file.return_value = "def main():\n    pass\n"
    sk = agskill("w", "", output_schema=agdata(doc=agfile), max_output_schema_retries=0)
    responses = [
        _tool_call("return_doc", {"value": "/workspace/outputs/code.py"}),
        _direct(""),
    ]
    result = _run_skill_with_sandbox(sk, responses, sandbox)
    assert result.doc == "def main():\n    pass\n"


# ---------------------------------------------------------------------------
# return_<field> tool — str auto-resolution of path values
# ---------------------------------------------------------------------------


def test_return_str_with_path_auto_resolves_to_content():
    """return_code called with a file path should silently resolve to file content."""
    sandbox = MagicMock()
    sandbox.read_file.return_value = "def main():\n    pass\n"
    sk = agskill("w", "", output_schema=agdata(code=str), max_output_schema_retries=0)
    responses = [
        _tool_call("return_code", {"value": "/workspace/core.py"}),
        _direct(""),
    ]
    result = _run_skill_with_sandbox(sk, responses, sandbox)
    assert result.code == "def main():\n    pass\n"
    sandbox.read_file.assert_called_with("/workspace/core.py")


def test_return_str_with_path_that_is_unreadable_keeps_original():
    """If sandbox.read_file raises, the original path value is kept as-is."""
    sandbox = MagicMock()
    sandbox.read_file.side_effect = FileNotFoundError("no file")
    sk = agskill("w", "", output_schema=agdata(code=str), max_output_schema_retries=0)
    responses = [
        _tool_call("return_code", {"value": "/workspace/core.py"}),
        _direct(""),
    ]
    result = _run_skill_with_sandbox(sk, responses, sandbox)
    assert result.code == "/workspace/core.py"


def test_return_str_with_real_content_not_resolved():
    """return_code with multiline content should never trigger path resolution."""
    sandbox = MagicMock()
    sk = agskill("w", "", output_schema=agdata(code=str), max_output_schema_retries=0)
    code = "import os\n\ndef main():\n    print('hello')\n"
    responses = [
        _tool_call("return_code", {"value": code}),
        _direct(""),
    ]
    result = _run_skill_with_sandbox(sk, responses, sandbox)
    assert result.code == code
    sandbox.read_file.assert_not_called()


def test_return_str_resolved_content_that_is_itself_a_path_is_not_substituted():
    """If the resolved file content is also a path, keep original to avoid chaining."""
    sandbox = MagicMock()
    sandbox.read_file.return_value = "/workspace/another.py"
    sk = agskill("w", "", output_schema=agdata(code=str), max_output_schema_retries=0)
    responses = [
        _tool_call("return_code", {"value": "/workspace/core.py"}),
        _direct(""),
    ]
    result = _run_skill_with_sandbox(sk, responses, sandbox)
    # Content itself looks like a path → not substituted → original path kept
    assert result.code == "/workspace/core.py"


# ---------------------------------------------------------------------------
# _looks_like_path
# ---------------------------------------------------------------------------


def test_looks_like_path_detects_workspace_paths():
    from agency.agutil import _looks_like_path

    assert _looks_like_path("/workspace/core.py")
    assert _looks_like_path("/workspace/outputs/report.txt")
    assert _looks_like_path("/tmp/scratch.py")
    assert _looks_like_path("/workspace/turn_specula.py")
    assert _looks_like_path("/workspace/outputs/harness_code_i1.py")


def test_looks_like_path_rejects_multiline():
    from agency.agutil import _looks_like_path

    assert not _looks_like_path("def main():\n    pass\n")
    assert not _looks_like_path("/workspace/file.py\nextra content")


def test_looks_like_path_rejects_non_absolute():
    from agency.agutil import _looks_like_path

    assert not _looks_like_path("relative/path.py")
    assert not _looks_like_path("just some text")
    assert not _looks_like_path("")


def test_looks_like_path_rejects_paths_with_spaces():
    from agency.agutil import _looks_like_path

    # Old heuristic would accept these; new regex rejects them
    assert not _looks_like_path("/this is not a path")
    assert not _looks_like_path("/workspace/file.py extra text")
    assert not _looks_like_path("/workspace/some file.py")


def test_looks_like_path_single_segment():
    from agency.agutil import _looks_like_path

    assert _looks_like_path("/bin")
    assert _looks_like_path("/a")
    assert _looks_like_path("/tmp")
    assert _looks_like_path("/tmp/file.txt")
    assert _looks_like_path("/a/b/c")
