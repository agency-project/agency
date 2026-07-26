"""Manual probe (not pytest): same engine="claude_code" Agency harness
pipeline as claude_code_example.py, but running the same 6-step
multi-file task used in the direct-CLI native-Bedrock vs ANTHROPIC_BASE_URL
comparison, so the two can be compared apples-to-apples.

Set AGENCY_DEBUG_CAPTURE_LOG to a file path to have agproxy_llm dump raw
incoming /v1/messages bodies (roles, mid-array system messages, reminder
markers) before any adapter transformation.

Run:
    LLM_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0 \
    AGENCY_DEBUG_CAPTURE_LOG=/tmp/agproxy_v1msgs_pipeline.log \
    .venv/bin/python3 examples/_manual_v1messages_pipeline_probe.py
"""

import os
from pathlib import Path
from agency import agent, agskill, agdata
from agency.agconfig import agConfig
from agency.agllm_backends import agBedrockBackendConfig
from agency.agtype import agpath

cfg = agConfig(
    agBedrockBackendConfig(model=os.environ.get("LLM_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0"))
)


def _make_run_dir(name: str):
    from datetime import datetime

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(__file__).parent.parent / "runs" / f"{ts}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


TASK_PROMPT = (
    "Create files a.txt, b.txt, c.txt each with a different one-line fact about the ocean, "
    "in the same directory as file_path. Then read all three back and write summary.txt "
    "combining them in that directory. Then list the directory. Then read summary.txt once "
    "more and count the words in it, writing the count to wordcount.txt in that directory. "
    "Then create a subdirectory called archive in that directory and copy all four txt files "
    "into it. Finally read back each file in archive to confirm."
)


def main():
    run_dir = _make_run_dir("v1messages_pipeline_probe")
    agent.log_dir = run_dir / "logs"
    agent.output_dir = run_dir / "agent_output"
    print(f"Run dir  : {run_dir}\n")

    file_skill = agskill(
        name="file_manager",
        system_prompt=(
            "You are a file management assistant. "
            "Use the write and read tools to complete the task. "
            "Always confirm what you wrote by reading files back. "
            "Write files to /workspace."
        ),
        input_schema=agdata(
            task=str,
            file_path=agpath,
        ),
        output_schema=agdata(
            status=str,
            path=agpath,
            content=str,
        ),
    )

    ag = agent(agconfig=cfg, engine="claude_code")

    print(">> [file_manager] 6-step multi-file ocean-facts task")
    r1 = ag.run(
        file_skill,
        agdata(task=TASK_PROMPT, file_path="/workspace/a.txt"),
    )
    print(f"   status  : {r1.status!r}")
    print(f"   path    : {r1.path!r}")
    print(f"   content : {r1.content!r}")
    print()
    print(f"Shared history : {len(ag.history.messages)} messages total")


if __name__ == "__main__":
    main()
