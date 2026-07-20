"""Tests for agbinary — binary file-backed agskill schema field."""

import base64
import json
import pytest
from unittest.mock import MagicMock, patch

from agency.agdata import agdata, agerror
from agency.agcontext import agcontext
from agency.agconfig import agConfig
from agency.agtype import agtype, agbinary
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


PNG_MAGIC = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


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


def _run_skill_with_sandbox(skill, responses, sandbox):
    sandbox._has_pending_background_work.return_value = False
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, *_ = skill.execute_react(_make_mock_agent(LLM, sandbox), agcontext(), agdata())
    return result


# ---------------------------------------------------------------------------
# agbinary class interface
# ---------------------------------------------------------------------------


def test_agbinary_is_agtype_subclass():
    assert issubclass(agbinary, agtype)


def test_agbinary_schema_type():
    assert agbinary.schema_type() == "binary_file"


def test_agbinary_needs_sandbox():
    assert agbinary.needs_sandbox() is True


# ---------------------------------------------------------------------------
# agbinary._to_bytes
# ---------------------------------------------------------------------------


def test_to_bytes_from_bytes():
    raw = b"\x89PNG"
    assert agbinary._to_bytes(raw) == raw


def test_to_bytes_from_data_url():
    raw = b"\x89PNG"
    b64 = base64.b64encode(raw).decode()
    data_url = f"data:image/png;base64,{b64}"
    assert agbinary._to_bytes(data_url) == raw


def test_to_bytes_from_local_path(tmp_path):
    f = tmp_path / "test.bin"
    f.write_bytes(PNG_MAGIC)
    assert agbinary._to_bytes(str(f)) == PNG_MAGIC


def test_to_bytes_invalid_type_raises():
    with pytest.raises(TypeError):
        agbinary._to_bytes(42)


# ---------------------------------------------------------------------------
# agbinary.prepare
# ---------------------------------------------------------------------------


def test_prepare_writes_bytes_to_sandbox():
    sandbox = MagicMock()
    val, paths = agbinary.prepare(PNG_MAGIC, sandbox, "process", "audio")
    sandbox.write_file_bytes.assert_called_once_with("/workspace/inputs/audio.bin", PNG_MAGIC)
    assert val == "/workspace/inputs/audio.bin"
    assert paths == ["/workspace/inputs/audio.bin"]


def test_prepare_accepts_data_url():
    sandbox = MagicMock()
    b64 = base64.b64encode(PNG_MAGIC).decode()
    data_url = f"data:image/png;base64,{b64}"
    val, paths = agbinary.prepare(data_url, sandbox, "skill", "img")
    sandbox.write_file_bytes.assert_called_once_with("/workspace/inputs/img.bin", PNG_MAGIC)
    assert val == "/workspace/inputs/img.bin"


def test_prepare_local_path(tmp_path):
    f = tmp_path / "clip.wav"
    f.write_bytes(PNG_MAGIC)
    sandbox = MagicMock()
    val, paths = agbinary.prepare(str(f), sandbox, "skill", "audio")
    sandbox.write_file_bytes.assert_called_once_with("/workspace/inputs/audio.bin", PNG_MAGIC)


def test_prepare_unsupported_type_passthrough():
    sandbox = MagicMock()
    val, paths = agbinary.prepare(42, sandbox, "skill", "field")
    sandbox.write_file_bytes.assert_not_called()
    assert val == 42
    assert paths == []


def test_prepare_sandbox_failure_leaves_value_unchanged():
    sandbox = MagicMock()
    sandbox.write_file_bytes.side_effect = OSError("disk full")
    val, paths = agbinary.prepare(PNG_MAGIC, sandbox, "skill", "field")
    assert val == PNG_MAGIC
    assert paths == []


# ---------------------------------------------------------------------------
# agbinary.recover
# ---------------------------------------------------------------------------


def test_recover_reads_bytes_from_sandbox():
    sandbox = MagicMock()
    sandbox.read_file_bytes.return_value = PNG_MAGIC
    val, paths = agbinary.recover("/workspace/outputs/trimmed.bin", sandbox)
    sandbox.read_file_bytes.assert_called_once_with("/workspace/outputs/trimmed.bin")
    assert val == PNG_MAGIC
    assert paths == ["/workspace/outputs/trimmed.bin"]


def test_recover_non_string_passthrough():
    sandbox = MagicMock()
    val, paths = agbinary.recover(None, sandbox)
    sandbox.read_file_bytes.assert_not_called()
    assert val is None
    assert paths == []


def test_recover_sandbox_failure_leaves_path_unchanged():
    sandbox = MagicMock()
    sandbox.read_file_bytes.side_effect = FileNotFoundError("gone")
    val, paths = agbinary.recover("/workspace/out.bin", sandbox)
    assert val == "/workspace/out.bin"
    assert paths == []


# ---------------------------------------------------------------------------
# System prompt — agbinary prompts injected
# ---------------------------------------------------------------------------


def test_system_prompt_includes_agbinary_input_instructions():
    sk = agskill(
        "process",
        "Do stuff.",
        input_schema=agdata(audio=agbinary),
        output_schema=agdata(result=str),
    )
    prompt = sk._build_system_prompt()
    assert "File-backed fields" in prompt
    assert "audio" in prompt
    assert "binary" in prompt.lower()


def test_system_prompt_includes_agbinary_output_instructions():
    sk = agskill(
        "process",
        "Do stuff.",
        output_schema=agdata(trimmed=agbinary),
    )
    prompt = sk._build_system_prompt()
    assert "File-backed fields" in prompt
    assert "trimmed" in prompt
    assert "binary" in prompt.lower()


def test_system_prompt_agbinary_type_shown_as_binary_file():
    sk = agskill("t", "", input_schema=agdata(data=agbinary))
    prompt = sk._build_system_prompt()
    assert '"data": "binary_file"' in prompt


def test_agdata_serializes_agbinary_as_binary_file():
    d = agdata(payload=agbinary)
    assert json.loads(d.to_json()) == {"payload": "binary_file"}


# ---------------------------------------------------------------------------
# return_<field> tool — agbinary return tool descriptions
# ---------------------------------------------------------------------------


def test_return_tool_description_mentions_binary():
    desc = agbinary.get_return_tool_description("audio")
    assert "audio" in desc
    assert "binary" in desc.lower()


def test_return_value_description_mentions_path_not_content():
    desc = agbinary.get_return_tool_value_description("audio")
    assert "path" in desc.lower()
    assert "audio" in desc


# ---------------------------------------------------------------------------
# return_<field> validation — sandbox existence checks
# ---------------------------------------------------------------------------


def _make_exec_side_effect(*responses):
    """Return a side_effect list for _container_exec calls."""
    return list(responses)


def _sandbox_with_exec(side_effects):
    sandbox = MagicMock()
    sandbox._container_exec = MagicMock(side_effect=side_effects)
    return sandbox


def test_return_agbinary_missing_file_returns_error():
    """Agent returns a path that doesn't exist → error, no file registered."""
    # test -d → 1 (not a dir), test -s → 1 (not found/empty), test -e → 1 (not found)
    sandbox = _sandbox_with_exec([("", 1), ("", 1), ("", 1)])
    sk = agskill("w", "", output_schema=agdata(out=agbinary), max_output_schema_retries=0)
    responses = [
        _tool_call("return_out", {"value": "/workspace/outputs/missing.bin"}),
        _direct(""),
    ]
    result = _run_skill_with_sandbox(sk, responses, sandbox)
    assert isinstance(result, agerror) or result._data.get("out") is None


def test_return_agbinary_empty_file_returns_error():
    """Agent returns a path to an empty file → error."""
    # test -d → 1 (not a dir), test -s → 1 (empty), test -e → 0 (exists but empty)
    sandbox = _sandbox_with_exec([("", 1), ("", 1), ("", 0)])
    sk = agskill("w", "", output_schema=agdata(out=agbinary), max_output_schema_retries=0)
    responses = [
        _tool_call("return_out", {"value": "/workspace/outputs/empty.bin"}),
        _direct(""),
    ]
    result = _run_skill_with_sandbox(sk, responses, sandbox)
    assert isinstance(result, agerror) or result._data.get("out") is None


def test_return_agbinary_directory_returns_error():
    """Agent returns a directory path → error."""
    # test -d → 0 (is a directory)
    sandbox = _sandbox_with_exec([("", 0)])
    sk = agskill("w", "", output_schema=agdata(out=agbinary), max_output_schema_retries=0)
    responses = [
        _tool_call("return_out", {"value": "/workspace/outputs"}),
        _direct(""),
    ]
    result = _run_skill_with_sandbox(sk, responses, sandbox)
    assert isinstance(result, agerror) or result._data.get("out") is None


def test_return_agbinary_valid_file_is_accepted():
    """Agent returns a non-empty binary file path → accepted, bytes recovered."""
    # test -d → 1 (not a dir), test -s → 0 (exists and non-empty)
    sandbox = _sandbox_with_exec([("", 1), ("", 0)])
    sandbox.read_file_bytes.return_value = PNG_MAGIC
    sk = agskill("w", "", output_schema=agdata(out=agbinary), max_output_schema_retries=0)
    responses = [
        _tool_call("return_out", {"value": "/workspace/outputs/clip.bin"}),
        _direct(""),
    ]
    result = _run_skill_with_sandbox(sk, responses, sandbox)
    assert result.out == PNG_MAGIC


# ---------------------------------------------------------------------------
# agsandbox.read_file_bytes and write_file_bytes (unit, no Docker)
# ---------------------------------------------------------------------------


class TestAgSandboxBinaryIO:
    def _make_sb(self):
        from agency.agsandbox_backends.container import _ContainerBackendBase

        sb = _ContainerBackendBase.__new__(_ContainerBackendBase)
        return sb

    def test_read_file_bytes_returns_raw_bytes(self):
        sb = self._make_sb()
        b64 = base64.b64encode(PNG_MAGIC).decode()
        with patch.object(sb, "_container_exec", return_value=(b64, 0)):
            result = sb.read_file_bytes("/workspace/image.png")
        assert result == PNG_MAGIC

    def test_read_file_bytes_missing_raises_file_not_found(self):
        sb = self._make_sb()
        sb._container_exec = MagicMock(side_effect=[("", 1), ("", 1)])
        with pytest.raises(FileNotFoundError):
            sb.read_file_bytes("/workspace/missing.bin")

    def test_read_file_bytes_directory_raises_is_a_directory_error(self):
        sb = self._make_sb()
        sb._container_exec = MagicMock(side_effect=[("", 1), ("", 0)])
        with pytest.raises(IsADirectoryError):
            sb.read_file_bytes("/workspace/outputs")

    def test_read_file_bytes_does_not_raise_for_non_utf8(self):
        sb = self._make_sb()
        b64 = base64.b64encode(PNG_MAGIC).decode()
        with patch.object(sb, "_container_exec", return_value=(b64, 0)):
            result = sb.read_file_bytes("/workspace/image.png")
        # Must NOT raise UnicodeDecodeError — binary is expected
        assert isinstance(result, bytes)

    def test_write_file_bytes_encodes_via_base64(self):
        sb = self._make_sb()
        with patch.object(sb, "_container_exec", return_value=("", 0)) as mock_exec:
            sb.write_file_bytes("/workspace/out.bin", PNG_MAGIC)
        cmd = mock_exec.call_args[0][0]
        # The base64-encoded payload must appear in the shell command
        expected_b64 = base64.b64encode(PNG_MAGIC).decode("ascii")
        assert expected_b64 in cmd

    def test_write_file_bytes_failure_raises_os_error(self):
        sb = self._make_sb()
        with patch.object(sb, "_container_exec", return_value=("", 1)):
            with pytest.raises(OSError):
                sb.write_file_bytes("/workspace/out.bin", PNG_MAGIC)
