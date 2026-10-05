import json

from agency.native_harness import tools
from agency.native_harness.canonicalize import canonicalize_run, clean, compact_tables, is_run_command
from agency.tandem_harness import tools as tandem_tools


def _kept_lines(out: str) -> list[str]:
    return [line for line in out.split("\n") if not line.startswith("[... ")]


def _is_ordered_subset(kept: list[str], source: list[str]) -> bool:
    it = iter(source)
    return all(any(line == s for s in it) for line in kept)


def test_small_output_passes_through_after_cleanup():
    raw = "\x1b[32mOK\x1b[0m   \nRan 3 tests in 0.1s\n"
    out, stats = canonicalize_run(raw)
    assert out == "OK\nRan 3 tests in 0.1s\n"
    assert stats["omitted"] == 0


def test_carriage_return_progress_keeps_final_state():
    assert clean("10%\r50%\r100% done") == "100% done"


def test_repeated_numeric_dump_is_collapsed_and_tail_kept():
    block = ["         x                y                z"] + [
        f"{i} H     0.000000000{i}     1.430522676{i}     1.109269235{i}" for i in range(3)
    ]
    raw = "\n".join(block * 100 + ["-74.9391972013", " 1 file changed, 1 deletion(-)"])
    out, stats = canonicalize_run(raw)
    assert "-74.9391972013" in out and "1 file changed" in out
    assert stats["omitted"] > 300
    assert "lines omitted" in out
    assert _is_ordered_subset(_kept_lines(out), clean(raw).split("\n"))


def test_errors_and_test_summary_survive_in_large_output():
    noise = [f"step {i} energy {i * 0.001:.6f}" for i in range(400)]
    failure = [
        "Traceback (most recent call last):",
        '  File "/workspace/pkg/mod.py", line 12, in run',
        "    assert x == y",
        "AssertionError: 1 != 2",
    ]
    raw = "\n".join(
        noise[:200] + failure + noise[200:] + ["Ran 88 tests in 2.9s", "FAILED (failures=6)"]
    )
    out, _ = canonicalize_run(raw)
    for line in failure + ["Ran 88 tests in 2.9s", "FAILED (failures=6)"]:
        assert line in out.split("\n")


def test_kept_lines_are_verbatim_and_in_order():
    raw = "\n".join(f"row {i}: value {i * 3.14159:.5f} label_{i % 7}" for i in range(1000))
    out, _ = canonicalize_run(raw)
    kept = _kept_lines(out)
    assert kept and _is_ordered_subset(kept, raw.split("\n"))
    assert len(out) < len(raw) / 5


def test_run_commands_are_told_apart_from_reads_and_queries():
    assert is_run_command("cd /workspace/pyscf; PYTHONPATH=. python /tmp/t.py")
    assert is_run_command("python tests/runtests.py model_fields")
    assert not is_run_command("sed -n 1,420p integrators.py")
    assert not is_run_command("python -c \"import duckdb; print(duckdb.sql('select 1'))\"")


def test_bash_tool_compacts_run_output_only_when_enabled(monkeypatch):
    command = """python -c 'print(chr(10).join("0 H 0.0 1.43 1.10" for _ in range(500))); print(-74.94)'"""
    monkeypatch.setattr(tools, "CANONICALIZE_RUN_OUTPUT", False)
    raw = json.loads(tools._run_bash_tool(json.dumps({"command": command})))["output"]
    monkeypatch.setattr(tools, "CANONICALIZE_RUN_OUTPUT", True)
    compact = json.loads(tools._run_bash_tool(json.dumps({"command": command})))["output"]
    assert len(raw.splitlines()) == 501
    assert "-74.94" in compact and "lines omitted" in compact
    assert len(compact) < len(raw) / 10


def test_bash_tool_leaves_large_file_reads_alone(tmp_path, monkeypatch):
    path = tmp_path / "big.py"
    path.write_text(chr(10).join(f"x_{i} = {i}" for i in range(600)))
    monkeypatch.setattr(tools, "CANONICALIZE_RUN_OUTPUT", True)
    out = json.loads(tools._run_bash_tool(json.dumps({"command": f"cat {path}"})))["output"]
    assert len(out.splitlines()) == 600


BOX = """ 66% ▕█████████████████████████             ▏ (~1 second remaining)     
100% ▕██████████████████████████████████████▏ (00:00:04.06 elapsed)     
┌─────────────────────┬───────┐
│         hr          │   n   │
│      timestamp      │ int64 │
├─────────────────────┼───────┤
│ 2026-09-12 15:00:00 │ 82239 │
│ 2026-09-12 14:00:00 │  NULL │
│ 2026-09-12 10:00:00 │       │
└─────────────────────┴───────┘
"""

TRUNCATED = """┌───────┬───────┐
│   i   │   j   │
│ int64 │ int64 │
├───────┼───────┤
│     0 │     0 │
│     · │     · │
│     · │     · │
│  9999 │ 19998 │
└───────┴───────┘
  ? rows       2 columns
  (>9999 rows, 20 shown) 
"""


def test_box_table_becomes_plain_rows_with_types_and_no_progress():
    out, stats = compact_tables(BOX)
    assert out.split("\n")[:4] == [
        "hr (timestamp) | n (int64)",
        "2026-09-12 15:00:00 | 82239",
        "2026-09-12 14:00:00 | NULL",
        "2026-09-12 10:00:00 | ",
    ]
    assert stats["tables"] == 1 and stats["progress_lines"] == 2
    assert "▕" not in out and "│" not in out


def test_every_cell_value_survives_table_compaction():
    out, _ = compact_tables(BOX + TRUNCATED)
    cells = {
        c.strip() for line in (BOX + TRUNCATED).split("\n") if line.startswith("│")
        for c in line.strip("│").split("│") if c.strip()
    }
    assert all(cell in out for cell in cells)


def test_truncated_table_keeps_one_elision_row_and_its_footer():
    out, _ = compact_tables(TRUNCATED)
    assert out.split("\n") == [
        "i (int64) | j (int64)",
        "0 | 0",
        "· | ·",
        "9999 | 19998",
        "? rows 2 columns",
        "(>9999 rows, 20 shown)",
        "",
    ]


def test_text_without_tables_is_unchanged():
    text = "Ran 3 tests in 0.1s\n│ not a table\nOK\n"
    assert compact_tables(text)[0] == text


def test_bash_tool_compacts_tables_only_when_enabled(monkeypatch):
    command = "printf '%s\\n' '┌───┐' '│ a │' '│ int │' '├───┤' '│ 1 │' '└───┘'"
    monkeypatch.setattr(tools, "COMPACT_TABLES", False)
    assert "┌" in json.loads(tools._run_bash_tool(json.dumps({"command": command})))["output"]
    for module in (tools, tandem_tools):
        monkeypatch.setattr(module, "COMPACT_TABLES", True)
        out = json.loads(module._run_bash_tool(json.dumps({"command": command})))["output"]
        assert out == "a (int)\n1\n"
