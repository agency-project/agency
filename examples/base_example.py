"""
Example: agent talking to a remote LLM server, with typed agskill schemas.

OpenAI-compatible API example::
  - base_url  → http://<host>:<port>/v1
  - api_key   → "EMPTY" for local endpoints with no auth, or a real API key
  - model     → The model to use.

    Launch vLLM (example):
        vllm serve google/gemma-4-E2B-it \
            --enable-auto-tool-choice \
            --tool-call-parser gemma4 \
            --reasoning-parser gemma4

    Run this script:
        uv run python example.py
        LLM_BASE_URL="http://localhost:8000/v1" uv run python example.py
"""

import os
from pathlib import Path
from agency import agent, agskill, agdata
from agency.agconfig import agConfig
from agency.agllm_backends import agVLLMBackendConfig
from agency.agtype import agpath

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


def _make_run_dir(name: str):
    from datetime import datetime

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(__file__).parent.parent / "runs" / f"{ts}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    run_dir = _make_run_dir("base_example")
    agent.log_dir = run_dir / "logs"
    agent.output_dir = run_dir / "agent_output"
    print(f"Run dir  : {run_dir}\n")

    file_skill = agskill(
        name="file_manager",
        system_prompt=(
            "You are a file management assistant. "
            "Use the write and read tools to complete the task. "
            "Always confirm what you wrote by reading the file back. "
            "Write files to /workspace."
        ),
        input_schema=agdata(
            task=str,
            file_path=agpath,  # datatype for passing path in the sandbox
        ),
        output_schema=agdata(
            status=str,
            path=agpath,  # datatype for passing path in the sandbox
            content=str,
        ),
    )

    qa_skill = agskill(
        name="qa",
        system_prompt=(
            "Answer the user's question directly and concisely. "
            "You have access to prior conversation context."
        ),
        input_schema=agdata(question=str),
        output_schema=agdata(answer=str),
        replace_tools=[],
    )

    # No tools= argument — uses the default sandboxed tool list
    ag = agent(agconfig=cfg)

    print(">> [file_manager] write and verify a note")
    r1 = ag.run(
        file_skill,
        agdata(
            task="Write 'Hello from the agent!' to the given file and verify it.",
            file_path="/workspace/note.txt",
        ),
    )
    print(f"   status  : {r1.status!r}")
    print(f"   path    : {r1.path!r}")
    print(f"   content : {r1.content!r}")
    print()

    print(">> [qa] ask about the note using shared history")
    r2 = ag.run(
        qa_skill,
        agdata(question="What was written to the note file, and where is it?"),
    )
    print(f"   answer : {r2.answer!r}")
    print()

    print(f"Shared history : {len(ag.history.messages)} messages total")


if __name__ == "__main__":
    from agency.agwebui import agwebui

    agwebui.run(main, port=8001)
