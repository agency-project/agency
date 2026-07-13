"""
sandbox_handoff.py — Sandbox access and handoff between agents and harness.

This example showcases three sandbox interaction patterns:

  1. Agent creates a file in its sandbox (first agent writes hello.py).
  2. Harness takes ownership of the sandbox — runs the file, then patches
     it with sed in a way that strips the string quotes, breaking the code.
  3. Second agent receives the same sandbox as an external dependency and
     fixes the syntax error introduced by the harness.
  4. Harness runs the fixed file to confirm the repair.

Key APIs demonstrated:
  - agent.sandbox            : read/assign an agent's sandbox directly (plain attribute)
  - sandbox.exec(cmd)        : run a shell command in the container from Python
  - sandbox.read_file(path)  : read a file out of the container

Note: agskill always stops+commits an agent's sandbox at the end of a skill
run, regardless of who created it. This isn't destructive — stop(commit=True)
commits the container filesystem to an image first, so the next access (by
the harness or another agent sharing the same agSandbox object) transparently
restarts the container from that checkpoint.

Run:
    uv run python examples/sandbox_handoff.py
    LLM_MODEL=gpt-4o uv run python examples/sandbox_handoff.py
"""

import os
from pathlib import Path
from datetime import datetime

from agency import agent, agskill, agdata
from agency.agconfig import agConfig
from agency.agllm_backends import agVLLMBackendConfig
from agency.agtype import agpath

# ---------------------------------------------------------------------------
# LLM config
# ---------------------------------------------------------------------------

# See ../README.md for OpenAI, Anthropic, or Bedrock agconfig examples.
cfg = agConfig(
    agVLLMBackendConfig(
        base_url=os.environ.get("LLM_BASE_URL"),
        model=os.environ.get("LLM_MODEL", ""),
        api_key=os.environ.get("LLM_API_KEY", ""),
        temperature=0.7,
        top_p=0.95,
        top_k=20,
    )
)
FILE_PATH = "/workspace/hello.py"

# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

write_skill = agskill(
    name="write_python",
    system_prompt=(
        "You are a code writing assistant. "
        "Use the write tool to create the requested Python file at the given path."
    ),
    input_schema=agdata(task=str, file_path=agpath),
    output_schema=agdata(status=str, path=agpath),
)

fix_skill = agskill(
    name="fix_bug",
    system_prompt=(
        "You are a debugging assistant. "
        "Read the file at the given path, identify the syntax error, fix it using "
        "the write tool, then verify your fix by running the file with bash. "
        "Report the corrected output."
    ),
    input_schema=agdata(task=str, file_path=agpath),
    output_schema=agdata(status=str, output=str),
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sep(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _run_file(sandbox, label: str) -> None:
    output, rc = sandbox.exec(f"python {FILE_PATH}")
    status = "✓" if rc == 0 else f"✗ (exit {rc})"
    print(f"  [{label}] python {FILE_PATH}  →  {status}")
    print(f"  output : {output.strip()!r}")


def main() -> None:
    run_dir = Path("runs") / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_sandbox_handoff"
    run_dir.mkdir(parents=True, exist_ok=True)
    agent.log_dir = run_dir / "logs"
    agent.output_dir = run_dir / "agent_output"

    print(f"Run dir : {run_dir}\n")

    # ── Step 1: first agent writes hello.py ─────────────────────────────────
    _sep("Step 1 — agent_a writes hello.py")

    agent_a = agent(agconfig=cfg)
    result_a = agent_a.run(
        write_skill,
        agdata(
            task="Write a Python file that prints 'Hello World' to stdout.",
            file_path=FILE_PATH,
        ),
    )
    # Resolving the pending result also ensures the skill (and sandbox setup) has finished.
    print(f"  agent_a status : {result_a.status!r}")
    print(f"  agent_a path   : {result_a.path!r}")

    # ── Step 2: harness gets the sandbox and runs the file ───────────────────
    _sep("Step 2 — harness takes the sandbox and runs hello.py")

    sandbox = agent_a.sandbox  # ← the sandbox agent_a's skill run just used
    assert sandbox is not None, "sandbox not started — did the skill run complete?"

    content_before = sandbox.read_file(FILE_PATH)
    print(f"  file contents  :\n{content_before.rstrip()}")

    _run_file(sandbox, "before patch")

    # ── Step 3: harness patches the file with sed, breaking the syntax ───────
    _sep("Step 3 — harness patches with sed (removes string quotes → SYNTEX ERROR EXPECTED)")

    # This sed command replaces  print("Hello World")  or  print('Hello World')
    # with                       print(Hello External World)
    # — the string is no longer quoted, which is a SyntaxError.
    patch_cmd = (
        r"""sed -i "s/print(['\"]Hello World['\"])/print(Hello External World)/g" """ + FILE_PATH
    )
    _, rc = sandbox.exec(patch_cmd)
    print(f"  sed exit code  : {rc}")

    content_after = sandbox.read_file(FILE_PATH)
    print(f"  patched file   :\n{content_after.rstrip()}")

    _run_file(sandbox, "after patch — broken")

    # ── Step 4: second agent receives the sandbox and fixes the bug ──────────
    _sep("Step 4 — agent_b receives the sandbox and fixes the bug")

    agent_b = agent(agconfig=cfg)
    agent_b.sandbox = sandbox

    result_b = agent_b.run(
        fix_skill,
        agdata(
            task=(
                "The file has a syntax error: the string in the print() call is missing "
                "its quotation marks. Fix it so the file prints 'Hello External World' "
                "and runs without errors."
            ),
            file_path=FILE_PATH,
        ),
    )
    print(f"  agent_b status : {result_b.status!r}")
    print(f"  agent_b output : {result_b.output!r}")

    # ── Step 5: harness runs the fixed file ──────────────────────────────────
    _sep("Step 5 — harness runs hello.py after agent_b's fix")

    content_fixed = sandbox.read_file(FILE_PATH)
    print(f"  fixed file     :\n{content_fixed.rstrip()}")

    _run_file(sandbox, "after fix")

    # Sandbox cleanup happens automatically: agSandbox.__del__ calls destroy() on
    # garbage collection, with an atexit handler as a backstop.
    print("\nDone.")


if __name__ == "__main__":
    from agency.agwebui import agwebui

    agwebui.run(main, port=8007)
