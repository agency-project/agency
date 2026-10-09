"""T42 cache keep-alive, T43 search compaction and T44 reversible elision in the native harness."""

from __future__ import annotations

import copy
import json
import threading
import time

import pytest

from agency.native_harness import elide, react_loop, tools

PY_SRC = '''"""Module doc."""
import os
import sys

LIMIT = 10


class Config:
    """Settings."""

    def load(self, path: str) -> dict:
        data = {}
        for line in open(path):
            data[line] = len(line)
        return data

    def save(self, path):
        return path


def helper(a, b=2):
    return a + b
''' + "\n".join(f"# filler line {i} with some words to make the file long enough" for i in range(400))


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(elide, "REGISTRY", {})
    monkeypatch.setattr(elide, "ELIDE_MIN_TOKENS", 0)
    monkeypatch.setattr(elide, "SELECTOR", None)
    monkeypatch.setattr(elide, "_CONTEXT", {"task": "", "step": ""})


def test_group_search_hits_prints_each_path_once():
    out = "a/b.py:3:x = 1\na/b.py:9:y = 2\nc.py:1:z\nplain line"
    text, _ = elide.group_search_hits(out)
    assert text == "a/b.py:\n  3:x = 1\n  9:y = 2\nc.py:\n  1:z\nplain line"


def test_strip_licence_header():
    out = "// Copyright 2022 Acme\n// Licensed under the Apache License\n//   http://www.apache.org/licenses\n\npackage x\n"
    text, stats = elide.strip_licence_headers(out)
    assert "Apache" not in text and "package x" in text and stats["lines"] == 3


def test_python_outline_has_classes_methods_and_variables():
    names = [e["name"] for e in elide.outline(PY_SRC, "m.py")]
    assert names[:5] == ["imports", "LIMIT", "Config", "Config.load", "Config.save"]
    assert "helper" in names


def test_go_and_ts_outlines():
    go = "package x\n\nimport (\n\t\"fmt\"\n)\n\ntype T struct {\n\tA int\n}\n\nfunc (t *T) Run(n int) error {\n\treturn nil\n}\n\nfunc Free() {}\n"
    assert [e["name"] for e in elide.outline(go, "x.go")] == ["imports", "T", "T.Run", "Free"]
    ts = "import a from 'a';\n\nexport class Box {\n  open(x: number): void {\n    return;\n  }\n}\n\nexport const f = (y) => {\n  return y;\n};\n"
    assert [e["name"] for e in elide.outline(ts, "x.ts")] == ["imports", "Box", "Box.open", "f"]


def test_large_cat_becomes_outline_and_parts_come_back_in_one_call(tmp_path, monkeypatch):
    monkeypatch.setattr(elide, "ELIDE_MIN_TOKENS", 500)
    f = tmp_path / "m.py"
    f.write_text(PY_SRC)
    out, stats = elide.elide_output(f"cat {f}", None, "header\n" + PY_SRC + "\nfooter")
    assert stats["files"] == 1 and "E1:" in out and "L11-15" in out and "footer" in out
    assert out.count("filler line") == 0
    got = json.loads(elide.show_elided(json.dumps({"parts": ["E1:Config.load", "E1:helper", "E1:1-3"]})))
    assert "data[line] = len(line)" in got["output"] and "return a + b" in got["output"]
    assert got["output"].count("== E1") == 3 and "errors" not in got
    f.write_text(PY_SRC.replace("return a + b", "return a - b"))
    again = json.loads(elide.show_elided(json.dumps({"parts": ["E1:helper", "E9:x", "E1:nope"]})))
    assert "return a - b" in again["output"] and len(again["errors"]) == 2


def test_small_outputs_and_partial_sed_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(elide, "ELIDE_MIN_TOKENS", 500)
    f = tmp_path / "m.py"
    f.write_text(PY_SRC)
    small = "\n".join(PY_SRC.split("\n")[:30])
    assert elide.elide_output(f"cat {f}", None, small)[0] == small
    part = "\n".join(PY_SRC.split("\n")[5:300])
    out, stats = elide.elide_output(f"cat {f} | sed -n 6,300p", None, part)
    assert stats["files"] == 1 and "lines 6-300" in out


def test_grep_hit_runs_are_cut(monkeypatch):
    monkeypatch.setattr(elide, "ELIDE_MIN_TOKENS", 50)
    out = "\n".join(f"src/a.py:{i}:value_{i} = compute({i})" for i in range(40))
    text, stats = elide.elide_output("grep -rn value src", None, out)
    assert stats["hit_files"] == 1 and text.count("src/a.py:") == 3 and "37 more hits" in text
    got = json.loads(elide.show_elided(json.dumps({"parts": ["E1:*"]})))
    assert got["output"].count("value_") == 40


def test_bash_tool_applies_elision_and_registers_show_elided(tmp_path, monkeypatch):
    monkeypatch.setattr(elide, "ELIDE_MIN_TOKENS", 500)
    monkeypatch.setattr(elide, "COMPACT_SEARCH", True)
    f = tmp_path / "m.py"
    f.write_text(PY_SRC)
    res = json.loads(tools._run_bash_tool(json.dumps({"command": f"cat {f}"})))
    assert "show_elided" in res["output"]
    monkeypatch.setattr(tools, "TOOL_DISPATCH", dict(tools.TOOL_DISPATCH))
    monkeypatch.setattr(tools, "BUILTIN_TOOL_SCHEMAS", dict(tools.BUILTIN_TOOL_SCHEMAS))
    tools.enable_show_elided()
    assert "show_elided" in tools.TOOL_DISPATCH and "show_elided" in tools.BUILTIN_TOOL_SCHEMAS


class _Llm:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.requests = []
        self.lock = threading.Lock()

    def dispatch(self, model, messages, tools=None, **kwargs):
        with self.lock:
            self.requests.append((copy.deepcopy(messages), kwargs))
            if kwargs.get("internal_kind") == "keepalive":
                return {"message": {"role": "assistant", "content": "x"}, "usage": {}}
            return next(self._responses)


def _turn(cmd):
    return {"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c0", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}]},
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


@pytest.mark.parametrize("keepalive_s", [0, 1])
def test_keepalive_pings_with_the_last_request_while_a_tool_runs(monkeypatch, tmp_path, keepalive_s):
    monkeypatch.setattr(react_loop, "CACHE_KEEPALIVE_S", keepalive_s)
    llm = _Llm([_turn("sleep 2.5"), {"message": {"role": "assistant", "content": "done"}, "usage": {}}])
    result = react_loop.run_react_loop([{"role": "user", "content": "go"}], "m", llm, offload_dir=str(tmp_path))
    time.sleep(0.2)
    pings = [r for r in llm.requests if r[1].get("internal_kind") == "keepalive"]
    assert result.status == "done"
    if not keepalive_s:
        assert not pings
        return
    assert 1 <= len(pings) <= 3
    assert all(p[1]["max_tokens"] == 1 and p[0] == llm.requests[0][0] for p in pings)


def test_selector_lines_are_shown_in_full_and_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(elide, "ELIDE_MIN_TOKENS", 500)
    prompts = []

    def selector(prompt):
        prompts.append(prompt)
        return {"message": {"content": 'Sure: {"ranges": [[11, 15], [21, 22], [30, 400]]}'}}

    monkeypatch.setattr(elide, "SELECTOR", selector)
    elide.note_context([{"role": "user", "content": "fix Config.load"}], {"content": "look at load", "tool_calls": []})
    f = tmp_path / "m.py"
    f.write_text(PY_SRC)
    out, stats = elide.elide_output(f"cat {f}", None, PY_SRC)
    assert stats["files"] == 1 and "fix Config.load" in prompts[0] and "8: class Config:" in prompts[0]
    assert "data[line] = len(line)" in out and "return a + b" in out
    shown = out.split("[Shown in full as likely relevant:]")[1].count("\n")
    assert shown < 0.4 * len(PY_SRC.split("\n"))


def test_selector_failure_falls_back_to_the_outline(tmp_path, monkeypatch):
    monkeypatch.setattr(elide, "ELIDE_MIN_TOKENS", 500)
    monkeypatch.setattr(elide, "SELECTOR", lambda prompt: {"error": "down"})
    f = tmp_path / "m.py"
    f.write_text(PY_SRC)
    out, _ = elide.elide_output(f"cat {f}", None, PY_SRC)
    assert "E1:" in out and "Shown in full" not in out
