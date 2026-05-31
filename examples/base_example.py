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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _run_dir import make_run_dir

from src import agent, agskill, agdata, default_tools

# ---------------------------------------------------------------------------
# LLM config — override via environment variables
# ---------------------------------------------------------------------------
LLM_CONFIG = {
    "base_url": os.environ.get("VLLM_BASE_URL", "http://localhost:18000/v1"),
    "api_key":  os.environ.get("VLLM_API_KEY",  "EMPTY"),
    "model":    os.environ.get("VLLM_MODEL",     "kimi_k2.6"),
}


def main():
    run_dir = make_run_dir("base_example")
    note_path = str(run_dir / "note.txt")
    agent.log_dir = run_dir / "logs"
    print(f"Run dir  : {run_dir}\n")

    # --- Define skills with typed input/output schemas -------------------

    file_skill = agskill(
        name="file_manager",
        system_prompt=(
            "You are a file management assistant. "
            "Use the write and read tools to complete the task. "
            "Always confirm what you wrote by reading the file back."
        ),
        input_schema=agdata(
            task="str",       # what the user wants done
            file_path="str",  # absolute path to operate on
        ),
        output_schema=agdata(
            status="str",     # "ok" or short description
            path="str",       # the file path that was written
            content="str",    # the content that was confirmed on disk
        ),
        max_retries=3,
    )

    qa_skill = agskill(
        name="qa",
        system_prompt=(
            "Answer the user's question directly and concisely. "
            "You have access to prior conversation context."
        ),
        input_schema=agdata(
            question="str",
        ),
        output_schema=agdata(
            answer="str",
        ),
        tools=[],       # no filesystem access needed
        max_retries=3,
    )

    # --- Build agent -----------------------------------------------------
    ag = agent(
        llm_config=LLM_CONFIG,
        agskills=[file_skill, qa_skill],
        tools=default_tools,
    )

    print(f"Endpoint : {LLM_CONFIG['base_url']}")
    print(f"Model    : {LLM_CONFIG['model']}")
    print(f"Skills   : {[f.name for f in ag.agskills]}")
    print(f"Tools    : {[t.name for t in ag.tools]}")
    print()

    # --- Turn 1: file_manager creates and verifies a note ----------------
    print(">> [file_manager] write and verify a note")
    r1 = ag.run(
        "file_manager",
        agdata(
            task="Write 'Hello from the agent!' to the given file and verify it.",
            file_path=note_path,
        ),
    )
    print(f"   status  : {r1.status!r}")
    print(f"   path    : {r1.path!r}")
    print(f"   content : {r1.content!r}")
    print()

    # --- Turn 2: qa answers using shared history -------------------------
    print(">> [qa] ask about the note using shared history")
    r2 = ag.run(
        "qa",
        agdata(question="What was written to the note file, and where is it?"),
    )
    print(f"   answer : {r2.answer!r}")
    print()

    print(f"Shared history : {len(ag.history.messages)} messages total")


if __name__ == "__main__":
    from src import AgError
    try:
        main()
    except AgError as e:
        print(f"\nERROR: {e}")
