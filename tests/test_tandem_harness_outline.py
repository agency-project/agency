import json

from agency.tandem_harness import outline
from agency.tandem_harness.outline import outline_report, paths_from_calls

SOURCE = "\n".join(
    ["import math", "", ""]
    + [
        line
        for name in ("alpha", "beta", "gamma", "delta")
        for line in (
            f"def {name}(x):",
            f"    a_{name} = x + 1",
            f"    b_{name} = a_{name} * 2",
            f"    c_{name} = b_{name} - 3",
            f"    return math.sqrt(c_{name})",
            "",
        )
    ]
)


def _report(names):
    lines = ["Here is the code:"]
    for name in names:
        lines += [
            f"def {name}(x):",
            f"    a_{name} = x + 1",
            f"    b_{name} = a_{name} * 2",
            f"    c_{name} = b_{name} - 3",
            f"    return math.sqrt(c_{name})",
        ]
    return "\n".join(lines + ["All four run without error."])


def test_unnamed_functions_become_outlines_named_ones_stay(tmp_path, monkeypatch):
    path = tmp_path / "mod.py"
    path.write_text(SOURCE)
    monkeypatch.setattr(outline, "CODE_LINE_LIMIT", 5)
    out, stats = outline_report(_report(["alpha", "beta", "gamma", "delta"]), [str(path)], "Show beta.")
    assert "    b_beta = a_beta * 2" in out
    assert "b_alpha" not in out and "b_gamma" not in out
    assert f"[not shown: {path}:4-8  alpha: def alpha(x):]" in out
    assert "All four run without error." in out and outline.NOTE in out
    assert stats["outlined"] == 3 and stats["kept"] == 1


def test_small_reports_pass_through(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text(SOURCE)
    text = _report(["alpha", "beta"])
    assert outline_report(text, [str(path)], "Show the code.")[0] == text


def test_paths_come_from_read_and_bash_calls(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(SOURCE)
    calls = [
        {"arguments": json.dumps({"file_path": str(tmp_path / "pkg" / "mod.py")})},
        {"arguments": json.dumps({"command": f"cd {tmp_path} && sed -n 1,20p pkg/mod.py"})},
    ]
    assert paths_from_calls(calls) == [str(tmp_path / "pkg" / "mod.py")]
