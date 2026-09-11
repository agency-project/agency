"""Smallest sandbox-side MCP tool example.

Run with an OpenAI key and an authenticated Claude Code CLI:
    OPENAI_API_KEY=... uv run python examples/sandbox_mcp_tools.py
"""

import os

from agency import agent, agdata, agskill, agtool
from agency.configs.agconfig import agconfig, llmconfig


def double(arg):
    return agdata(result=arg.number * 2)


# See ../README.md for Anthropic or Bedrock agconfig examples.
cfg = agconfig(
    llmconfig(
        provider="OpenAI_Compatible",
        base_url=os.environ["LLM_BASE_URL"],
        model=os.environ["LLM_MODEL"],
        api_key=os.environ["LLM_API_KEY"],
    )
)

skill = agskill(
    name="sandbox_tool",
    prompt="Call double exactly once, then return its result.",
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
    invocation = agent(agconfig=cfg).run(skill, agdata(number=21))
    invocation.wait()
    print(invocation.result)
