import json
import subprocess

from agency.native_harness import fork, react_loop, tools


class ScriptedLLM:
    """First call: one bash call that appends the branch name to f.txt; second call: final report."""

    def __init__(self):
        self.calls = 0

    def dispatch(self, model, messages, tools_=None, **kw):
        self.calls += 1
        prompt = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        name = prompt.split("branch '")[1].split("'")[0]
        if self.calls == 1:
            cmd = f"echo {name} >> {REPO}/f.txt && cat {REPO}/f.txt"
            return {"message": {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{name}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}]},
                "usage": {}}
        return {"message": {"role": "assistant", "content": f"report from {name}"}, "usage": {}}


REPO = ""


def test_fork_runs_branches_in_isolated_copies(tmp_path, monkeypatch):
    global REPO
    REPO = str(tmp_path / "repo")
    subprocess.run(["git", "init", "-q", REPO], check=True)
    (tmp_path / "repo" / "f.txt").write_text("base\n")
    subprocess.run(["git", "-C", REPO, "add", "-A"], check=True)
    subprocess.run(["git", "-C", REPO, "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "init"], check=True)
    (tmp_path / "repo" / "f.txt").write_text("base\nparent edit\n")
    monkeypatch.setattr(fork, "LLM_FACTORY", ScriptedLLM)
    monkeypatch.setattr(fork, "CONTEXT", [{"role": "user", "content": "task"}])
    monkeypatch.setattr(fork, "OFFLOAD_DIR", str(tmp_path / "off"))
    tool = fork.make_tool(dict(tools.TOOL_DISPATCH))

    out = json.loads(tool(json.dumps({"repo_path": REPO, "branches": [
        {"name": "a", "instruction": "try a"}, {"name": "b", "instruction": "try b"}]})))

    assert (tmp_path / "repo" / "f.txt").read_text() == "base\nparent edit\n"
    by = {b["name"]: b for b in out["branches"]}
    assert by["a"]["report"] == "report from a" and by["a"]["changed_files"] == ["f.txt"]
    patch = open(by["b"]["patch"]).read()
    assert "+b" in patch and "parent edit" in patch.split("@@")[-1]


def test_fork_rejects_single_branch(tmp_path):
    tool = fork.make_tool({})
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    out = json.loads(tool(json.dumps({"repo_path": str(tmp_path), "branches": [{"name": "a", "instruction": "x"}]})))
    assert "error" in out


class RecordingLLM:
    """Bash 'true' until the forced step, then a final answer; records tool_choice and messages per call."""

    def __init__(self, fork_at=None):
        self.choices, self.seen, self.fork_at = [], [], fork_at

    def dispatch(self, model, messages, tools_=None, **kw):
        self.choices.append(kw.get("tool_choice"))
        self.seen.append(list(messages))
        n = len(self.choices) - 1
        if n >= 4:
            return {"message": {"role": "assistant", "content": "done"}, "usage": {}}
        name = "fork" if n == self.fork_at else "bash"
        args = {"repo_path": "/nonexistent", "branches": []} if name == "fork" else {"command": "true"}
        return {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}, "usage": {}}


def _loop(monkeypatch, llm, force_at):
    monkeypatch.setattr(fork, "ENABLED", True)
    monkeypatch.setattr(fork, "FORCE_AT_STEP", force_at)
    return react_loop.run_react_loop([{"role": "user", "content": "task"}], "m", llm)


def test_fork_is_forced_at_the_given_step(monkeypatch):
    llm = RecordingLLM()
    assert _loop(monkeypatch, llm, 2).status == "done"
    assert llm.choices == [None, None, {"type": "function", "function": {"name": "fork"}}, None, None]
    assert llm.seen[2][-1] == {"role": "user", "content": fork.FORCE_NOTE}


def test_no_forcing_when_the_agent_already_forked_or_when_off(monkeypatch):
    llm = RecordingLLM(fork_at=1)
    _loop(monkeypatch, llm, 2)
    assert all(c is None for c in llm.choices)
    llm = RecordingLLM()
    _loop(monkeypatch, llm, 0)
    assert all(c is None for c in llm.choices)


class ScriptLLM:
    """Plays a fixed list of tool calls (one per turn), then answers; records tool_choice per call."""

    def __init__(self, script):
        self.script, self.choices, self.seen = script, [], []

    def dispatch(self, model, messages, tools_=None, **kw):
        self.choices.append(kw.get("tool_choice"))
        self.seen.append(list(messages))
        n = len(self.choices) - 1
        if n >= len(self.script):
            return {"message": {"role": "assistant", "content": "done"}, "usage": {}}
        name, args = self.script[n]
        return {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}, "usage": {}}


def _fail_loop(monkeypatch, tmp_path, script):
    monkeypatch.setattr(fork, "ENABLED", True)
    monkeypatch.setattr(fork, "FORCE_AT_STEP", 0)
    monkeypatch.setattr(fork, "FORCE_ON_FAIL", True)
    llm = ScriptLLM(script)
    react_loop.run_react_loop([{"role": "user", "content": "task"}], "m", llm)
    return llm


def test_fork_is_forced_after_a_failed_test_run_that_follows_an_edit(monkeypatch, tmp_path):
    f = str(tmp_path / "x.py")
    llm = _fail_loop(monkeypatch, tmp_path, [("write", {"file_path": f, "content": "x = 1\n"}),
                                             ("bash", {"command": "echo running pytest; exit 1"}),
                                             ("bash", {"command": "true"})])
    assert llm.choices[:2] == [None, None]
    assert llm.choices[2] == {"type": "function", "function": {"name": "fork"}}
    assert llm.seen[2][-1] == {"role": "user", "content": fork.FIX_NOTE}


def test_no_forced_fork_for_a_failed_test_before_any_edit(monkeypatch, tmp_path):
    llm = _fail_loop(monkeypatch, tmp_path, [("bash", {"command": "echo running pytest; exit 1"}),
                                             ("bash", {"command": "true"})])
    assert all(c is None for c in llm.choices)


def test_failed_check_reads_failures_from_piped_output():
    call = [{"function": {"name": "bash", "arguments": json.dumps({"command": "go test ./x/ 2>&1 | tail -5"})}}]
    for text, want in (("--- FAIL: TestX (0.01s)\nFAIL\n", True), ("FAILED (failures=2)", True),
                       ("=== 3 failed, 10 passed in 2.1s ===", True), ("ok  \tx\t0.3s\n", False),
                       ("Ran 12 tests\n\nOK\n", False)):
        res = [{"content": json.dumps({"output": text, "returncode": 0})}]
        assert fork.failed_check(call, res) is want, text
