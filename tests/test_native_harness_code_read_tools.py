import copy
import json

import pytest

from agency.native_harness import tools
from agency.tandem_harness import tools as tandem_tools

SOURCE = """import functools


def compile_json_path(keys):
    return "$" + "".join(keys)


class VelocityVerlet:
    def __init__(self, mol):
        self.mol = mol

    @functools.lru_cache
    def _next(self):
        accel = 1
        return accel
"""


@pytest.fixture
def module_file(tmp_path):
    path = tmp_path / "integrators.py"
    path.write_text(SOURCE)
    return str(path)


def _read(**args):
    return json.loads(tools._run_read_tool(json.dumps(args)))


def test_symbol_read_returns_one_method_with_its_decorator(module_file):
    out = _read(file_path=module_file, symbol="VelocityVerlet._next")
    lines = out["content"].splitlines()
    assert lines[0] == "12:     @functools.lru_cache"
    assert lines[-1] == "15:         return accel"
    assert "compile_json_path" not in out["content"]


def test_symbol_read_matches_a_bare_name(module_file):
    out = _read(file_path=module_file, symbol="compile_json_path")
    assert out["content"].splitlines() == [
        "4: def compile_json_path(keys):",
        '5:     return "$" + "".join(keys)',
    ]


def test_unknown_symbol_lists_close_matches(module_file):
    out = _read(file_path=module_file, symbol="next_step")
    assert "error" in out and "VelocityVerlet._next" in out["close_matches"]


def test_read_without_symbol_is_unchanged(module_file):
    assert _read(file_path=module_file, want="anything")["type"] == "file"


def test_code_read_tools_extend_schemas_only_when_enabled(monkeypatch):
    original = copy.deepcopy(tools.BUILTIN_TOOL_SCHEMAS)
    monkeypatch.setattr(tools, "BUILTIN_TOOL_SCHEMAS", copy.deepcopy(original))
    assert "want" not in tools.BUILTIN_TOOL_SCHEMAS["bash"]["function"]["parameters"]["properties"]
    tools.enable_code_read_tools()
    props = {
        n: s["function"]["parameters"]["properties"] for n, s in tools.BUILTIN_TOOL_SCHEMAS.items()
    }
    assert "symbol" in props["read"] and "want" in props["read"]
    assert "want" in props["bash"] and "want" in props["grep"]
    assert "want" not in props["write"]
    assert original["bash"]["function"]["parameters"]["properties"].get("want") is None


def test_tandem_worker_bash_compacts_run_output_when_enabled(monkeypatch):
    command = """python -c 'print(chr(10).join("0 H 0.0 1.43 1.10" for _ in range(500))); print(-74.94)'"""
    monkeypatch.setattr(tandem_tools, "CANONICALIZE_RUN_OUTPUT", True)
    out = json.loads(tandem_tools._run_bash_tool(json.dumps({"command": command})))["output"]
    assert "-74.94" in out and "lines omitted" in out


def test_want_required_only_on_bash(monkeypatch):
    monkeypatch.setattr(tools, "BUILTIN_TOOL_SCHEMAS", copy.deepcopy(tools.BUILTIN_TOOL_SCHEMAS))
    tools.enable_code_read_tools(want_required=True)
    required = {
        n: s["function"]["parameters"]["required"] for n, s in tools.BUILTIN_TOOL_SCHEMAS.items()
    }
    assert "want" in required["bash"] and "want" not in required["read"]


def test_compact_code_read_keeps_line_accounting_exact():
    from agency.native_harness.canonicalize import compact_code_read

    header = [f"# Copyright 2014 Authors, licensed under Apache {i}" for i in range(6)]
    body = ["import numpy", "", "", "", "", "def f(x):   ", "    return x"]
    out, stats = compact_code_read("\n".join(header + body))
    lines = out.split("\n")
    assert lines[0] == "[6 lines of license header omitted]"
    assert "[4 blank lines]" in lines and "def f(x):" in lines
    counted = sum(int(line.split()[0][1:]) if line.startswith("[") else 1 for line in lines)
    assert counted == len(header) + len(body)
    assert stats["chars_out"] < stats["chars_in"]


def test_code_read_commands_are_told_apart():
    from agency.native_harness.canonicalize import is_code_read_command

    assert is_code_read_command("cd pyscf/md; sed -n 420,607p integrators.py")
    assert is_code_read_command("cat -n django/db/models/fields/json.py")
    assert not is_code_read_command("PYTHONPATH=. python /tmp/t.py")
    assert not is_code_read_command("grep -n HasKeyLookup -r django")
