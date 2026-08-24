"""Manual probe (not pytest): Agency's native in-container ReAct loop.

This is deliberately parallel to ``_manual_v1messages_pipeline_probe.py``:
it runs the same 6-step multi-file task with the same schemas and LLM
configuration, but selects ``harness="native"``. It therefore tests the
native backend through the same public ``agent.run()`` harness abstraction
that the Claude Code probe uses, without the Claude CLI or its /v1/messages
adapter in the path.

Run:
    LLM_BASE_URL=https://your-openai-compatible-endpoint/v1 \
    LLM_API_KEY=your-api-key \
    LLM_MODEL=your-tool-capable-model \
    .venv/bin/python3 examples/_manual_native_react_pipeline_probe.py

With no LLM_BASE_URL, this falls back to Bedrock and requires AWS credentials.
Set LLM_REGION to select the Bedrock region (default: us-east-1), and
LLM_CONTEXT_LIMIT when the backend's model metadata endpoint is unavailable.
"""

import os
from pathlib import Path

from agency import agent, agdata, agskill
from agency.agconfig import agConfig
from agency.llm import agBedrockBackendConfig, agVLLMBackendConfig
from agency.agtype import agpath


if os.environ.get("LLM_BASE_URL"):
    cfg = agConfig(
        agVLLMBackendConfig(
            base_url=os.environ["LLM_BASE_URL"],
            model=os.environ.get("LLM_MODEL", ""),
            api_key=os.environ.get("LLM_API_KEY", ""),
        )
    )
else:
    bedrock_kwargs = {
        "model": os.environ.get("LLM_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
        "region": os.environ.get("LLM_REGION", "us-east-1"),
    }
    if os.environ.get("LLM_CONTEXT_LIMIT"):
        bedrock_kwargs["context_limit"] = int(os.environ["LLM_CONTEXT_LIMIT"])
    cfg = agConfig(agBedrockBackendConfig(**bedrock_kwargs))


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
    run_dir = _make_run_dir("native_react_pipeline_probe")
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

    # ``native`` is an agharness backend too: its ReAct loop runs inside
    # the sandbox container and is reached through the same agent.run()
    # dispatch contract as an external harness engine.
    ag = agent(agconfig=cfg, harness="native")

    print(">> [file_manager] 6-step multi-file ocean-facts task")
    r1 = ag.run(
        file_skill,
        agdata(task=TASK_PROMPT, file_path="/workspace/a.txt"),
    )
    result = r1.to_dict()
    if "error" in result:
        print(f"   error   : {result['error']}")
        return
    print(f"   status  : {result['status']!r}")
    print(f"   path    : {result['path']!r}")
    print(f"   content : {result['content']!r}")
    print()
    print(f"Shared history : {len(ag.history.messages)} messages total")


if __name__ == "__main__":
    main()
