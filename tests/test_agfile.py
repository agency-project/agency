"""Tests for agfile — file-backed agskill schema field."""

import json
from unittest.mock import MagicMock
from agency.agdata import agdata, agerror
from agency.agschema import agschema
from agency.agtype import agtype, agfile
from agency.agskill import agskill


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
# agskill._build_prompt — agfile prompts injected
# ---------------------------------------------------------------------------


def test_prompt_includes_agfile_input_instructions():
    sk = agskill(
        "design",
        "Do stuff.",
        input_schema=agdata(background=agfile),
        output_schema=agdata(result=str),
    )
    prompt = sk._build_prompt()
    assert "File-backed fields" in prompt
    assert "background" in prompt
    assert "read tool" in prompt


def test_prompt_includes_agfile_output_instructions():
    sk = agskill(
        "design",
        "Do stuff.",
        input_schema=agdata(theme=str),
        output_schema=agdata(report=agfile),
    )
    prompt = sk._build_prompt()
    assert "File-backed fields" in prompt
    assert "report" in prompt
    assert "write" in prompt.lower()


def test_prompt_no_agfile_fields_no_file_backed_warning():
    sk = agskill("t", "Do stuff.", input_schema=agdata(x=str), output_schema=agdata(y=int))
    prompt = sk._build_prompt()
    assert "File-backed" not in prompt


def test_prompt_agfile_type_shown_as_file():
    sk = agskill("t", "", input_schema=agdata(doc=agfile))
    prompt = sk._build_prompt()
    assert '"doc": "file"' in prompt


# ---------------------------------------------------------------------------
# agskill ReAct loop — agfile output schema accepts path string
# ---------------------------------------------------------------------------


# test_skill_with_agfile_output_schema_validates_path_string was retired
# here: same retired return_<field> validation chain (see the larger note
# further down this file), just for the "well-formed path, successfully
# recovered" case instead of an error case.


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


# _run_skill_with_sandbox() helper and every test_return_agfile_*/
# test_return_str_* test were retired here: they tested agschema.py's
# make_field_handler()'s agtype-specific validation chain (directory/
# binary/missing/empty-file errors, path-auto-resolution for a plain str
# field) as invoked by the retired per-field `return_<field>` tool handler,
# only ever reachable via execute_react(). Native's `submit_output` MCP
# tool does NOT run this same validation chain today (only basic type/
# shape checking via output_schema.check_field()) -- a real, documented gap
# for agtype OUTPUT fields specifically, noted in agmcp_server.py's own
# module docstring rather than silently dropped.


# ---------------------------------------------------------------------------
# _looks_like_path
# ---------------------------------------------------------------------------


def test_looks_like_path_detects_workspace_paths():
    from agency.utils.agutil import _looks_like_path

    assert _looks_like_path("/workspace/core.py")
    assert _looks_like_path("/workspace/outputs/report.txt")
    assert _looks_like_path("/tmp/scratch.py")
    assert _looks_like_path("/workspace/turn_specula.py")
    assert _looks_like_path("/workspace/outputs/harness_code_i1.py")


def test_looks_like_path_rejects_multiline():
    from agency.utils.agutil import _looks_like_path

    assert not _looks_like_path("def main():\n    pass\n")
    assert not _looks_like_path("/workspace/file.py\nextra content")


def test_looks_like_path_rejects_non_absolute():
    from agency.utils.agutil import _looks_like_path

    assert not _looks_like_path("relative/path.py")
    assert not _looks_like_path("just some text")
    assert not _looks_like_path("")


def test_looks_like_path_rejects_paths_with_spaces():
    from agency.utils.agutil import _looks_like_path

    # Old heuristic would accept these; new regex rejects them
    assert not _looks_like_path("/this is not a path")
    assert not _looks_like_path("/workspace/file.py extra text")
    assert not _looks_like_path("/workspace/some file.py")


def test_looks_like_path_single_segment():
    from agency.utils.agutil import _looks_like_path

    assert _looks_like_path("/bin")
    assert _looks_like_path("/a")
    assert _looks_like_path("/tmp")
    assert _looks_like_path("/tmp/file.txt")
    assert _looks_like_path("/a/b/c")
