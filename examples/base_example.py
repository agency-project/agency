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
from agency import agent, agskill, agdata
from agency.configs.agconfig import agconfig, llmconfig
from agency.agtype import agpath

cfg = agconfig(
    llmconfig(
        provider="OpenAI_Compatible",
        base_url=os.environ["LLM_BASE_URL"],
        model=os.environ["LLM_MODEL"],
        api_key=os.environ["LLM_API_KEY"],
    )
)


def main():
    file_skill = agskill(
        name="file_manager",
        prompt=(
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
            path=agpath,  # datatype for passing path in the sandbox
            content=str,
        ),
    )

    qa_skill = agskill(
        name="qa",
        prompt=(
            "Answer the user's question directly and concisely. "
            "You have access to prior conversation context."
        ),
        input_schema=agdata(question=str),
        output_schema=agdata(answer=str),
    )

    ag = agent(agconfig=cfg)

    print(">> [file_manager] write and verify a note")
    r1 = ag.run(
        file_skill,
        agdata(
            task="Write 'Hello from the agent!' to the given file and verify it.",
            file_path="/workspace/note.txt",
        ),
    )
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
    from agency.observability.agwebui import agwebui

    agwebui.run(main, port=8009)
