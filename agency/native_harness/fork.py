"""fork tool: explore alternatives in parallel branches, each with a copy of the agent's context and repository.

A branch is a react loop of its own that starts from the parent's context (as it was before the fork call),
works in a git worktree holding the parent's current changes, and returns a short report plus a patch the
parent can apply. Branch tool paths that name the parent's repository are rewritten to the branch's copy.
Branches cannot fork again or submit outputs.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

ENABLED = False
LLM_FACTORY = None  # () -> LLMClient, set by cli
MODEL = ""
BRIDGE = None
OFFLOAD_DIR = "./long_tool_call_outputs"
CONTEXT: list = []  # parent's messages before the assistant turn that called fork (set by react_loop)
BRANCH_STEPS = 30
MAX_BRANCHES = 4
FORCE_AT_STEP = 0  # >0: at this step, if the agent has not forked yet, the harness makes it call fork
FORCE_ON_FAIL = False  # the harness makes the agent fork right after its first failed test run following an edit
FIX_NOTE = (
    "[harness] Your last check failed. Use the fork tool now: call fork with 2-4 branches, each implementing and "
    "testing one different candidate fix (a different location or approach), so the branches can be compared."
)
TEST_CMD = re.compile(r"\b(pytest|py\.test|unittest|go test|npm (run )?test|tox|runtests(\.py)?|make test|cargo test)\b")


# Agents pipe test output through grep/tail, which hides the exit code, so failures are also read from the output.
TEST_FAILED = re.compile(r"\b[1-9]\d* failed\b|^FAILED\b|FAILED \((failures|errors)=|--- FAIL|^FAIL\s|"
                         r"Traceback \(most recent call last\)", re.M)


def failed_check(tool_calls: list, results: list) -> bool:
    """True if one of these bash calls ran tests that exited non-zero or reported failures."""
    for tc, res in zip(tool_calls, results):
        fn = tc.get("function") or {}
        if fn.get("name") != "bash":
            continue
        try:
            cmd = json.loads(fn.get("arguments") or "{}").get("command", "")
            out = json.loads(res.get("content") or "{}")
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
        text = str(out.get("output") or "") + str(out.get("error") or "")
        if TEST_CMD.search(cmd) and (out.get("returncode") not in (0, None) or TEST_FAILED.search(text)):
            return True
    return False
FORCE_NOTE = (
    "[harness] Use the fork tool now: call fork with 2-4 branches, one per plausible root-cause location, hypothesis "
    "or candidate fix you have identified so far, each with a concrete instruction for what to check or try."
)

DESCRIPTION = (
    "Explore 2-4 alternatives at once. Each branch starts from a copy of your current context and its own copy "
    "of the repository (with your current changes), follows its instruction for a bounded number of steps, and "
    "reports back what it found and what it changed. Branches run in parallel; you get every report and can "
    "apply a branch's changes with `git apply <patch>`. Use it when you have several independent hypotheses, "
    "candidate locations, or candidate fixes to check; do not use it for a single line of work."
)
PARAMS = {
    "type": "object",
    "properties": {
        "repo_path": {"type": "string", "description": "Repository root to copy for each branch."},
        "branches": {
            "type": "array",
            "minItems": 2,
            "maxItems": MAX_BRANCHES,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short identifier, e.g. 'fix_in_parser'."},
                    "instruction": {"type": "string", "description": "What this branch should investigate or try."},
                },
                "required": ["name", "instruction"],
            },
        },
    },
    "required": ["repo_path", "branches"],
}

BRANCH_PROMPT = (
    "[fork] You are branch '{name}' of a parallel exploration. Pursue only this: {instruction}\n"
    "Your own copy of the repository is at {branch_root} (it replaces {repo_root}; paths under {repo_root} are "
    "redirected there). Work for at most {steps} steps. Do not call submit_output. If your instruction is a fix, "
    "implement it in your copy and run the relevant tests. Finish with a short report: what you found, whether this "
    "direction works (and how you checked), and which files you changed."
)


def _git(root, *args, **kw):
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, **kw)


def _make_worktree(repo_root: str, name: str) -> str:
    branch_root = tempfile.mkdtemp(prefix=f"fork_{re.sub(r'[^A-Za-z0-9_]', '_', name)[:30]}_")
    index = f"{branch_root}.index"
    env = {**os.environ, "GIT_INDEX_FILE": index}
    subprocess.run(["git", "-C", repo_root, "read-tree", "HEAD"], env=env, capture_output=True)
    subprocess.run(["git", "-C", repo_root, "add", "-A"], env=env, capture_output=True)
    patch = subprocess.run(["git", "-C", repo_root, "diff", "--cached", "--binary", "HEAD"], env=env,
                           capture_output=True, text=True).stdout
    os.remove(index) if os.path.exists(index) else None
    _git(repo_root, "worktree", "add", "-q", "--detach", branch_root, "HEAD", check=True)
    if patch.strip():
        subprocess.run(["git", "-C", branch_root, "apply", "--binary", "-"], input=patch, text=True, capture_output=True)
        _git(branch_root, "add", "-A")
        _git(branch_root, "-c", "user.email=fork@local", "-c", "user.name=fork", "commit", "-q", "-m", "parent state")
    return branch_root


def _remap(handler, repo_root: str, branch_root: str):
    rx = re.compile(re.escape(repo_root.rstrip("/")) + r"(?=/|\b|$)")

    def call(args_json: str) -> str:
        return handler(rx.sub(branch_root, args_json or ""))
    return call


def _run_branch(spec: dict, repo_root: str, dispatch: dict) -> dict:
    from . import react_loop

    name = str(spec.get("name") or "branch")
    started = time.time()
    try:
        branch_root = _make_worktree(repo_root, name)
    except subprocess.CalledProcessError as exc:
        return {"name": name, "status": "error", "report": f"could not copy repository: {exc.stderr[-300:]}"}
    table = {k: _remap(v, repo_root, branch_root) for k, v in dispatch.items()
             if k not in ("fork", "submit_output")}
    prompt = BRANCH_PROMPT.format(name=name, instruction=spec.get("instruction", ""), branch_root=branch_root,
                                  repo_root=repo_root, steps=BRANCH_STEPS)
    messages = list(CONTEXT) + [{"role": "user", "content": prompt}]
    result = react_loop.run_branch_loop(messages, MODEL, LLM_FACTORY(), table, bridge=BRIDGE,
                                        max_steps=BRANCH_STEPS, offload_dir=OFFLOAD_DIR)
    diff = _git(branch_root, "diff", "--binary", "HEAD").stdout
    patch_path = f"{branch_root}.patch"
    with open(patch_path, "w") as f:
        f.write(diff)
    files = [l[6:] for l in diff.splitlines() if l.startswith("+++ b/")]
    return {
        "name": name,
        "status": result.status,
        "steps": result.turn_count,
        "seconds": round(time.time() - started),
        "report": (result.final_text or result.message or "")[-3000:],
        "changed_files": files,
        "patch": patch_path if diff.strip() else None,
    }


def make_tool(dispatch: dict):
    def fork(args_json: str) -> str:
        try:
            args = json.loads(args_json or "{}")
        except ValueError:
            return json.dumps({"error": "fork arguments must be JSON"})
        repo_root = (args.get("repo_path") or "").rstrip("/")
        branches = args.get("branches") or []
        if not repo_root or _git(repo_root, "rev-parse", "--git-dir").returncode != 0:
            return json.dumps({"error": f"repo_path {repo_root!r} is not a git repository"})
        if not 2 <= len(branches) <= MAX_BRANCHES:
            return json.dumps({"error": f"give 2-{MAX_BRANCHES} branches"})
        results: list = [None] * len(branches)

        def work(i, spec):
            try:
                results[i] = _run_branch(spec, repo_root, dispatch)
            except Exception as exc:  # noqa: BLE001
                results[i] = {"name": spec.get("name"), "status": "error", "report": f"{type(exc).__name__}: {exc}"}

        threads = [threading.Thread(target=work, args=(i, s), daemon=True) for i, s in enumerate(branches)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print(f"[native_harness] fork: {len(branches)} branches, "
              f"{[(r['name'], r['status'], r.get('steps')) for r in results]}", file=sys.stderr)
        return json.dumps({"branches": results,
                           "note": "Apply a branch's changes to your repository with: git -C <repo_path> apply <patch>"})
    return fork
