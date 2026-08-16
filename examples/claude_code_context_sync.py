"""Minimal Claude Code native-session synchronization example.

Requires the ``claude`` binary and ``AWS_BEARER_TOKEN_BEDROCK``. The first
call creates a native Claude session. The second call resumes it because its
saved Agency revision matches ``ag.ctx.revision``, so prior Agency history is
not sent again.
"""

import os

from agency import agdata, agent, agskill
from agency.agconfig import agConfig
from agency.agllm_backends import agBedrockBackendConfig


SECRET = "PURPLE-42-NARWHAL"


def main() -> None:
    cfg = agConfig(
        agBedrockBackendConfig(
            region=os.environ.get("AWS_REGION", "us-east-2"),
            model=os.environ.get("LLM_MODEL", "minimax.minimax-m2.5"),
            context_limit=196_000,
        )
    )
    remember = agskill(
        name="remember",
        system_prompt="Follow the instruction and answer concisely.",
        replace_tools=[],
    )
    ag = agent("claude_sync", agconfig=cfg, engine="claude_code")

    first = ag.run(
        remember,
        agdata(instruction=f"Remember the code {SECRET}. Reply only with OK."),
    )
    print("call 1:", first.result)

    ag.ctx.resolve_prev_dependencies()
    session = ag._harness_sessions["claude_code"]
    print(f"before call 2: Agency={ag.ctx.revision}, Claude={session['agcontext_revision']}")
    assert session["agcontext_revision"] == ag.ctx.revision

    second = ag.run(
        remember,
        agdata(instruction="What code did I ask you to remember? Reply only with the code."),
    )
    print("call 2:", second.result)
    assert SECRET in second.result


if __name__ == "__main__":
    main()
