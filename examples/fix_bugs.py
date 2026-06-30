"""
Example: bug-fixing workflow that copies the vllm codebase into the agent's
sandbox at /workspace/vllm, fixes a specific bug by calling a model up to N times, 
runs one of the test cases (full version would run the entire test suite), and reports results.

Run:
    VLLM_BASE_URL=url VLLM_MODEL=model uv run python examples/fix_bugs.py --max-attempts N
"""
import argparse
import os
import subprocess
from pathlib import Path

from agency import agent, agskill, agdata
from agency.agsandbox import agSandbox, _RUN_ID


def _make_run_dir(name: str):
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(__file__).parent.parent / "runs" / f"{ts}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


LLM_CONFIG = {
    "base_url": os.environ.get("VLLM_BASE_URL", ""),
    "api_key":  os.environ.get("VLLM_API_KEY", ""),
    "model":    os.environ.get("VLLM_MODEL",   ""),
    "temperature":       0.6,
    "max_tokens":        16000,
    "top_p":             0.95,
    "top_k":             50,
    "repetition_penalty": 1.1,
}

VLLM_SRC = Path(__file__).parent / "vllm"


def main(max_attempts: int = 5):
    run_dir = _make_run_dir("fix_bugs")
    agent.log_dir    = run_dir / "logs"
    agent.output_dir = run_dir / "agent_output"
    print(f"Run dir  : {run_dir}\n")

    ag = agent(llm_config=LLM_CONFIG)

    # -- Copy vllm source into sandbox directly via docker cp (no LLM) ----
    print(">> [setup] copying vllm source into sandbox at /workspace/vllm")
    sandbox = agSandbox(ag.agname)
    sandbox._ensure_started()
    runtime = sandbox._runtime
    container = sandbox._container_name()
    subprocess.run(
        [runtime, "cp", str(VLLM_SRC.resolve()) + "/.", f"{container}:/workspace/vllm"],
        check=True,
    )
    # Commit the container as the agent's checkpoint so subsequent skills
    # pick up the pre-populated /workspace/vllm.
    ckpt_tag = f"agency/ckpt-{_RUN_ID}-{ag.agname}"
    sandbox.commit(ckpt_tag)
    sandbox.destroy()
    ag._checkpoint = ckpt_tag
    print("   done.\n")
    # agent copy ctor allows clearing history 

    # -- Skill: read GitHub issue and fix the bug ----------------------------
    fix_skill = agskill(
        name="bug_fixer",
        system_prompt=(
            "You are an expert software engineer. Your job is to:\n"
            "1. Read a GitHub issue to understand the bug.\n"
            "2. Explore the codebase at /workspace/vllm to find the relevant code.\n"
            "3. Write a patch that fixes the bug.\n\n"
            "Use webfetch to read the GitHub issue. Use grep, glob, and read to "
            "explore the codebase. Use edit to apply your fix. "
            "Do NOT modify test files — only fix the source code."
        ),
        input_schema=agdata(
            issue_url=str,
            repo_path=str,
        ),
        output_schema=agdata(
            issue_summary=str,
            files_changed=str,
            patch_description=str,
        ),
    )

    # -- Skill: run tests/compile -------------------------------------------
    test_skill = agskill(
        name="test_runner",
        system_prompt=(
            "You are a test-runner assistant. Your ONLY job is to run the "
            "pytest command and capture its raw output. "
            "Install any missing dependencies with pip before running tests. "
            "Do NOT interpret or summarize the results — just run the tests "
            "and return the complete pytest output verbatim."
        ),
        input_schema=agdata(
            task=str,
            test_path=str,
        ),
        output_schema=agdata(
            raw_output=str,
        ),
    )

    # -- Skill: analyze test results ----------------------------------------
    analyze_skill = agskill(
        name="test_analyzer",
        system_prompt=(
            "You are a test-result analyst. You receive raw pytest output and "
            "extract structured information from it. Parse the output carefully "
            "and report the status, counts, and a brief summary of any failures. "
            "Do NOT run any commands — only analyze the text you are given."
        ),
        replace_tools=[],
        input_schema=agdata(
            raw_output=str,
        ),
        output_schema=agdata(
            status=str,
            total=str,
            passed=str,
            failed=str,
            errors=str,
            summary=str,
        ),
    )

    # -- Fix / test loop: repeat until all tests pass ----------------------
    for attempt in range(1, max_attempts + 1):
        print(f"=== Attempt {attempt}/{max_attempts} ===\n")

        # --- bug_fixer ---
        print(">> [bug_fixer] reading issue and applying fix")
        r_fix = ag.run(
            fix_skill,
            agdata(
                issue_url="https://github.com/vllm-project/vllm/issues/46088",
                repo_path="/workspace/vllm",
            ),
        )
        print(f"   raw_output: {r_fix}")
        print(f"   issue_summary    : {r_fix.issue_summary!r}")
        print(f"   files_changed    : {r_fix.files_changed!r}")
        print(f"   patch_description: {r_fix.patch_description!r}")
        print()

        # --- test_runner ---
        print(">> [test_runner] running tests/compile/test_decorator.py")
        r_run = ag.run(
            test_skill,
            agdata(
                task=(
                    "Run the pytest test suite at the given test_path. "
                    "First, cd to /workspace/vllm and install the project in "
                    "editable mode if needed (pip install -e . or the minimal "
                    "deps required). Then run: python -m pytest <test_path> -v "
                    "and return the complete raw output."
                ),
                test_path="tests/compile/test_decorator.py",
            ),
        )
        print(f"   raw_output: {r_run.raw_output}")
        print()

        # --- test_analyzer ---
        print(">> [test_analyzer] analyzing test results")
        r_test = ag.run(
            analyze_skill,
            agdata(raw_output=r_run.raw_output),
        )
        print(f"   status  : {r_test.status!r}")
        print(f"   total   : {r_test.total!r}")
        print(f"   passed  : {r_test.passed!r}")
        print(f"   failed  : {r_test.failed!r}")
        print(f"   errors  : {r_test.errors!r}")
        print(f"   summary : {r_test.summary!r}")
        print()

        if r_test.failed == "0" and r_test.errors == "0":
            print("All tests passed!\n")
            break
    else:
        print(f"Bug not fixed after {max_attempts} attempts.\n")

    print(f"Shared history : {len(ag.history.messages)} messages total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bug-fixing workflow")
    parser.add_argument("--max-attempts", type=int, default=5,
                        help="Maximum fix/test attempts (default: 5)")
    args = parser.parse_args()

    from agency.agwebui import agwebui
    agwebui.run(lambda: main(max_attempts=args.max_attempts))
