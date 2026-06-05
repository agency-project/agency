"""
Example: agent talking to a remote vLLM server, with typed agskill schemas.

vLLM exposes an OpenAI-compatible API:
  - base_url  → http://<host>:<port>/v1
  - api_key   → "EMPTY" (or whatever --api-key you set when launching vLLM)
  - model     → the model name exactly as vLLM registered it

Launch vLLM (example):
    vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --enable-auto-tool-choice \
        --tool-call-parser llama3 \
        --port 8000

Run this script:
    uv run python example.py
    VLLM_BASE_URL=http://127.0.0.1:18000/v1 VLLM_MODEL=kimi_k2.6 uv run python example.py
"""
import os
from pathlib import Path

from agency import agent, agskill, agdata

def _make_run_dir(name: str):
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(__file__).parent.parent / "runs" / f"{ts}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

LLM_CONFIG = {
    "base_url": os.environ.get("VLLM_BASE_URL", "https://kimi.js-park.info:18000/v1"),
    "api_key":  os.environ.get("VLLM_API_KEY", ""),
    "model":    os.environ.get("VLLM_MODEL",   "Qwen/Qwen3.5-397B-A17B-FP8"),
}

def main():
    run_dir = _make_run_dir("base_example")
    agent.log_dir    = run_dir / "logs"
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
            task="str",
            file_path="str",
        ),
        output_schema=agdata(
            status="str",
            path="str",
            content="str",
        ),
    )

    qa_skill = agskill(
        name="qa",
        system_prompt=(
            "Answer the user's question directly and concisely. "
            "You have access to prior conversation context."
        ),
        input_schema=agdata(question="str"),
        output_schema=agdata(answer="str"),
        tools=[],
    )

    # No tools= argument — uses the default sandboxed tool list
    ag = agent(
        llm_config=LLM_CONFIG,
        agskills=[file_skill, qa_skill],
    )

    print(f"Endpoint : {LLM_CONFIG['base_url']}")
    print(f"Model    : {LLM_CONFIG['model']}")
    print(f"Skills   : {[f.name for f in ag.agskills]}")
    print(f"Tools    : {[t.name for t in ag.tools]}")
    print()

    print(">> [file_manager] write and verify a note")
    r1 = ag.run(
        "file_manager",
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
        "qa",
        agdata(question="What was written to the note file, and where is it?"),
    )
    print(f"   answer : {r2.answer!r}")
    print()

    print(f"Shared history : {len(ag.history.messages)} messages total")

if __name__ == "__main__":
    from agency import agUI
    # agUI.run(main)
    main()
