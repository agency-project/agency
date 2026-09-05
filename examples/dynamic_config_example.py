"""
Example: building one agconfig covering both LLM and sandbox namespaces,
and dynamically updating a field between two skill calls on the same agent.

  - agconfig(llmconfig(...)) plus cfg.sandbox.add_mount() builds one config
    carrying both the LLM fields (under cfg.llm) and an agSandbox "data"
    mount (under cfg.sandbox) -- each domain has its own one-level namespace,
    but they're all still one object handed to agent(agconfig=cfg).
  - max_completion_tokens is read fresh from ag.agconfig.llm on every LLM call,
    no caching or locking. Set it too low (32) and the vLLM server truncates
    the tool-call JSON mid-argument, so the skill can't complete its
    required output field within a few ReAct steps.
  - Every framework object clones whatever agconfig it's given at
    construction time, so `cfg` and `ag.agconfig` are independent copies --
    mutating `cfg` after `agent(agconfig=cfg)` no longer reaches `ag`.
    `ag.change_config(new_cfg)` replaces `ag.agconfig` and pushes a fresh
    clone down through `ag.log`, `ag.data_logger`, and `ag.sandbox` -- no
    new agent, no sandbox teardown. The LLM dispatch itself (inside the
    harness daemon) reads `ag.agconfig` fresh on every skill execution, so
    it picks up the change on the very next call with nothing further to push.

See ../README.md for OpenAI, Anthropic, or Bedrock agconfig examples.

Run:
    uv run python examples/config_example.py
"""

import os
import time
from datetime import datetime
from pathlib import Path

from agency import agent, agskill, agdata
from agency.configs.agconfig import agconfig, llmconfig
from agency.agtype import agpath

_NOTE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox jumps over the lazy dog. "
    "This sentence is repeated here so the write tool's call has enough "
    "content that a 32-token completion budget cannot finish it."
)


def _make_run_dir(name: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(__file__).parent.parent / "runs" / f"{ts}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    run_dir = _make_run_dir("config_example")
    data_dir = run_dir / "data"
    print(f"Run dir  : {run_dir}\n")

    # One flat agconfig: LLM fields (including a deliberately too-small
    # max_completion_tokens) plus an agSandbox "data" mount, set once here
    # and never changed.
    cfg = agconfig(
        llmconfig(
            provider="vllm",
            base_url=os.environ.get("LLM_BASE_URL"),
            model=os.environ.get("LLM_MODEL", ""),
            api_key=os.environ.get("LLM_API_KEY", ""),
            temperature=0.7,
            top_p=0.95,
            top_k=20,
            max_completion_tokens=32,
        )
    )
    cfg.sandbox.add_mount("data", data_dir, "/data")

    write_note = agskill(
        name="write_note",
        system_prompt=(
            "Write the exact text you're given to the given file path using "
            "the write tool, then read it back to confirm."
        ),
        input_schema=agdata(text=str, file_path=agpath),
        output_schema=agdata(path=agpath, content=str),
    )

    ag = agent(agconfig=cfg)

    print(">> [call 1] max_completion_tokens=32 (too small to finish the tool call)")
    print(">> Execution should fail.")
    time.sleep(3)
    try:
        r1 = ag.run(write_note, agdata(text=_NOTE_TEXT, file_path="/data/note.txt"), max_steps=10)
        print(f"Path    : {r1.path!r}\n")
        print(f"Content : {r1.content!r}\n")
        print("Execution succeeded.\n")
    except Exception as e:
        # Accessing an output field on the Invocation blocks until the task
        # finishes; if the ReAct loop exhausted max_steps without a complete
        # tool call, or the truncated JSON never parsed, that access raises.
        print(f"Failed as expected: {e}\n")

    # The harness daemon re-reads max_completion_tokens fresh on every call,
    # so ag.change_config(new_cfg) makes the bump visible on the very next
    # LLM call -- no new agent, no sandbox teardown needed.
    print(
        "Bumping max_completion_tokens: 32 -> 4096 (dynamic update via ag.change_config, same agent)\n"
    )
    new_cfg = agconfig(
        llmconfig(
            provider="vllm",
            base_url=os.environ.get("LLM_BASE_URL"),
            model=os.environ.get("LLM_MODEL", ""),
            api_key=os.environ.get("LLM_API_KEY", ""),
            temperature=0.7,
            top_p=0.95,
            top_k=20,
            max_completion_tokens=4096,
        )
    )
    new_cfg.sandbox.add_mount("data", data_dir, "/data")
    ag.change_config(new_cfg)

    print(">> [call 2] max_completion_tokens=4096")
    print(">> Execution should succeed.")
    try:
        r2 = ag.run(write_note, agdata(text=_NOTE_TEXT, file_path="/data/note.txt"), max_steps=10)
        print(f"Path    : {r2.path!r}\n")
        print(f"Content : {r2.content!r}\n")
        print("Execution succeeded.\n")
    except Exception as e:
        print(f"Failed unexpectedly: {e}\n")


if __name__ == "__main__":
    from agency.observability.agwebui import agwebui

    agwebui.run(main, port=8003)
