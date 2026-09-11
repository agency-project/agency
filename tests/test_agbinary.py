"""Tests for agbinary — binary file-backed agskill schema field."""

import base64
import json
import pytest
from unittest.mock import MagicMock, patch

from agency.agdata import agdata
from agency.agtype import agtype, agbinary
from agency.agskill import agskill

PNG_MAGIC = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


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


def test_prompt_includes_agbinary_input_instructions():
    sk = agskill(
        "process",
        "Do stuff.",
        input_schema=agdata(audio=agbinary),
        output_schema=agdata(result=str),
    )
    prompt = sk._build_prompt()
    assert "File-backed fields" in prompt
    assert "audio" in prompt
    assert "binary" in prompt.lower()


def test_prompt_includes_agbinary_output_instructions():
    sk = agskill(
        "process",
        "Do stuff.",
        output_schema=agdata(trimmed=agbinary),
    )
    prompt = sk._build_prompt()
    assert "File-backed fields" in prompt
    assert "trimmed" in prompt
    assert "binary" in prompt.lower()


def test_prompt_agbinary_type_shown_as_binary_file():
    sk = agskill("t", "", input_schema=agdata(data=agbinary))
    prompt = sk._build_prompt()
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


# _make_exec_side_effect()/_sandbox_with_exec() helpers and every
# test_return_agbinary_* test were retired here: they tested agschema.py's
# make_field_handler()'s agtype-specific validation chain (missing/empty/
# directory-path errors, real bytes recovery) as invoked by the retired
# per-field `return_<field>` tool handler, only ever reachable via
# execute_react(). Native's `submit_output` MCP tool does NOT run this same
# validation chain today -- a real, documented gap for agtype OUTPUT fields
# specifically, noted in agmcp_server.py's own module docstring.

# ---------------------------------------------------------------------------
# agsandbox.read_file_bytes and write_file_bytes (unit, no Docker)
# ---------------------------------------------------------------------------


class TestAgSandboxBinaryIO:
    def _make_sb(self):
        from agency.sandbox.container import _ContainerBackendBase
        from agency.configs.agconfig import agconfig

        sb = _ContainerBackendBase.__new__(_ContainerBackendBase)
        sb._agconfig = agconfig()
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
