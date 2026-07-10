"""
Example: composing a consolidated agConfig from two owners, and dynamically
updating a DynamicConfigParam field between two skill calls on the same
agent.

  - agConfig(agVLLMBackendConfig(...), agSandboxConfig(...)) merges the
    agllm_backend fields and an agSandbox "data" mount into one agConfig in
    a single call. The mount is set once, in agSandboxConfig()'s
    constructor -- sandbox fields are all Static/GlobalConfigParam (locked
    the first time a sandbox resolves them), so they're not a good fit for
    changing between calls; see agAgentConfig/agLLMBackendConfig for fields
    that are.
  - agllm_backend.max_completion_tokens is a DynamicConfigParam: re-read
    fresh from the agConfig on every LLM call, no caching or locking. Set it
    too low (32) and the vLLM server truncates the tool-call JSON mid-
    argument, so the skill can't complete its required output field within
    a few ReAct steps.
  - Every framework object clones whatever agConfig it's given at
    construction time, so `cfg`, `ag.agconfig`, and `ag.llm._agconfig` are
    all independent copies -- mutating `cfg` (or even `ag.agconfig`) after
    `agent(agconfig=cfg)` no longer reaches `ag.llm`. To change the LLM
    config live, reach into `ag.llm._agconfig` directly -- no new agent, no
    sandbox teardown -- and the very next call sees it.

See ../README.md for OpenAI, Anthropic, or Bedrock agconfig examples.

Run:
    uv run python examples/config_example.py
"""
import os
import time
from datetime import datetime
from pathlib import Path

from agency import agent, agskill, agdata
from agency.agconfig import agConfig
from agency.agllm_backend import agVLLMBackendConfig
from agency.agsandbox import agSandboxConfig

_NOTE_TEXT = (
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

    # One agConfig, built from two owners' views: agllm_backend fields
    # (including a deliberately too-small max_completion_tokens) plus an
    # agSandbox "data" mount, set once here and never changed.
    cfg = agConfig(
        agVLLMBackendConfig(
            base_url=os.environ.get("LLM_BASE_URL"),
            model=os.environ.get("LLM_MODEL", ""),
            api_key=os.environ.get("LLM_API_KEY", ""),
            max_completion_tokens=32,
        ),
        agSandboxConfig().add_mount("data", data_dir, "/data"),
    )

    write_note = agskill(
        name="write_note",
        system_prompt=(
            "Write the exact text you're given to the given file path using "
            "the write tool, then read it back to confirm."
        ),
        input_schema=agdata(text=str, file_path=str),
        output_schema=agdata(path=str, content=str),
    )

    ag = agent(agconfig=cfg)

    print(">> [call 1] max_completion_tokens=32 (too small to finish the tool call)")
    print(">> Execution should fail.")
    time.sleep(3)
    try:
        r1 = ag.run(write_note, agdata(text=_NOTE_TEXT, file_path="/data/note.txt"), max_steps=3)
        print(f"   path    : {r1.path!r}")
        print(f"   content : {r1.content!r}\n")
    except Exception as e:
        # Accessing a field on a pending agdata blocks until the task
        # finishes; if the ReAct loop exhausted max_steps without a complete
        # tool call, or the truncated JSON never parsed, that access raises.
        print(f"   failed as expected: {e}\n")

    # ag.llm._agconfig is its own clone -- mutating cfg (or ag.agconfig) here
    # would NOT reach ag.llm. max_completion_tokens is a DynamicConfigParam,
    # so this update on ag.llm's own agconfig is visible on the very next
    # LLM call -- no new agent, no sandbox teardown needed.
    print("Bumping max_completion_tokens: 32 -> 4096 (dynamic update, ag.llm's own agconfig, same agent)\n")
    ag.llm._agconfig.agllm_backend.max_completion_tokens = 4096

    print(">> [call 2] max_completion_tokens=4096")
    print(">> Execution should succeed.")
    r2 = ag.run(write_note, agdata(text=_NOTE_TEXT, file_path="/data/note.txt"))
    print(f"   path    : {r2.path!r}")
    print(f"   content : {r2.content!r}\n")

    note = data_dir / "note.txt"
    content = note.read_text() if note.exists() else None
    if not note.exists():
        print(f"FAILURE:   note not found: {note!r}")
    else:
        print(f"SUCCESS:   note found: {note!r}  content={content!r}")


if __name__ == "__main__":
    from agency.agwebui import agwebui
    agwebui.run(main, port=8003)
