"""Smallest sandbox-side MCP tool example.

Run with an OpenAI key and an authenticated Claude Code CLI:
    OPENAI_API_KEY=... uv run python examples/sandbox_mcp_tools.py
"""

import os

from agency import agent, agdata, agskill, agtool
from agency.configs.agconfig import agconfig, llmconfig


def double(arg):
    return agdata(result=arg.number * 2)


cfg = agconfig(
    llmconfig(
        provider="openai",
        model="gpt-5.6-luna",
        api_key=os.environ["OPENAI_API_KEY"],
        reasoning_effort="none",
        max_completion_tokens=1024,
    )
)

skill = agskill(
    name="sandbox_tool",
    system_prompt="Call double exactly once, then return its result.",
    add_sandbox_mcp_tools=[
        agtool(
            "double",
            "Double a number inside the sandbox.",
            double,
            params={
                "type": "object",
                "properties": {"number": {"type": "integer"}},
                "required": ["number"],
            },
        )
    ],
    input_schema=agdata(number=int),
    output_schema=agdata(result=int),
)


if __name__ == "__main__":
    invocation = agent(agconfig=cfg, harness="claude_code").run(skill, agdata(number=21))
    invocation.wait()
    print(invocation.result.result)
