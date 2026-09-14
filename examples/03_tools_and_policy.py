"""Lesson 3: direct tools, host MCP tools, sandbox MCP tools, and policy."""

from __future__ import annotations

import json
import os

from agency import Agent, agdata, agskill, agtool
from agency.agpolicy import agpolicy

from _common import close_sandboxes, run_example, tutorial_config


HOST_TOOL_CALLS: list[dict[str, object]] = []
SANDBOX_PROOF_PATH = "/workspace/sandbox-double-proof.json"


def host_identity(arguments: agdata) -> agdata:
    proof = {"label": str(arguments.label), "process_pid": os.getpid()}
    HOST_TOOL_CALLS.append(proof)
    return agdata(**proof)


def double(arguments: agdata) -> agdata:
    return agdata(result=int(arguments.number) * 2)


def sandbox_double(arguments: agdata) -> agdata:
    result = int(arguments.number) * 2
    # The host reads this file directly from the sandbox after the skill run.
    # That makes the process-boundary proof independent of anything the model
    # might copy, omit, or invent in its structured response.
    with open(SANDBOX_PROOF_PATH, "w", encoding="utf-8") as proof_file:
        json.dump({"result": result, "process_pid": os.getpid()}, proof_file)
    return agdata(result=result)


def allow_tutorial_double(arguments: dict) -> tuple[bool, str]:
    allowed = arguments.get("number") == 21
    if allowed:
        return True, "double(number=21) is allowed"
    return False, "tutorial permits only double(number=21)"


HOST_IDENTITY = agtool(
    name="host_identity",
    description="Return a label and the PID of the host-side tool process.",
    fn=host_identity,
    params={
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
    },
)

DOUBLE = agtool(
    name="double",
    description="Double an integer in the current Python process.",
    fn=double,
    params={
        "type": "object",
        "properties": {"number": {"type": "integer"}},
        "required": ["number"],
    },
)

SANDBOX_DOUBLE = agtool(
    name="sandbox_double",
    description="Double an integer and record proof of the sandbox process that ran it.",
    fn=sandbox_double,
    params={
        "type": "object",
        "properties": {"number": {"type": "integer"}},
        "required": ["number"],
    },
)


def main() -> None:
    direct = DOUBLE(agdata(number=6))
    print(f"direct agtool call: result={direct.result}, caller_pid={os.getpid()}")
    print(f"OpenAI tool schema name: {DOUBLE.to_openai_tool()['function']['name']}")

    tool_policy = agpolicy(
        tool_hooks={"sandbox_double": allow_tutorial_double},
        default_to_deny=False,
    )
    double_policy = tool_policy.tool_hooks["sandbox_double"]
    allowed, allow_reason = double_policy({"number": 21})
    denied, deny_reason = double_policy({"number": 13})
    assert allowed and not denied
    print(f"policy hook contract: 21=allow, 13=deny ({allow_reason}; {deny_reason})")

    cfg, run_dir = tutorial_config("03_tools_and_policy")
    host_skill = agskill(
        name="host_tool_example",
        prompt=("Call host_identity exactly once with label tutorial, then report only its label."),
        add_host_mcp_tools=[HOST_IDENTITY],
        input_schema=agdata(task=str),
        output_schema=agdata(label=str),
    )
    sandbox_skill = agskill(
        name="sandbox_tool_example",
        prompt=(
            "Call sandbox_double exactly once with the supplied number, then report its result."
        ),
        add_sandbox_mcp_tools=[SANDBOX_DOUBLE],
        input_schema=agdata(number=int),
        output_schema=agdata(result=int),
        policy=tool_policy,
    )

    HOST_TOOL_CALLS.clear()
    host_learner = Agent("host-tools", agconfig=cfg, harness="native")
    sandbox_learner = Agent("sandbox-tools", agconfig=cfg, harness="codex")
    host_result = host_learner.run(host_skill, agdata(task="Identify the host tool process."))
    sandbox_result = sandbox_learner.run(sandbox_skill, agdata(number=21))
    assert host_result.label == "tutorial"
    assert HOST_TOOL_CALLS
    host_proof = HOST_TOOL_CALLS[-1]
    assert host_proof["label"] == "tutorial"
    assert host_proof["process_pid"] == os.getpid()
    assert sandbox_result.result == 42
    assert sandbox_learner.sandbox is not None
    sandbox_proof = json.loads(sandbox_learner.sandbox.read_file(SANDBOX_PROOF_PATH))
    assert sandbox_proof["result"] == 42
    assert sandbox_proof["process_pid"] > 0
    assert sandbox_proof["process_pid"] != host_proof["process_pid"]
    print(f"host MCP agent result: {host_result.to_dict()}")
    print(f"sandbox MCP agent result: {sandbox_result.to_dict()}")
    print(
        "execution boundary proved independently of the agent responses: "
        f"host={host_proof['process_pid']}, sandbox={sandbox_proof['process_pid']}"
    )
    print(f"artifacts: {run_dir}")

    close_sandboxes([host_learner, sandbox_learner])


if __name__ == "__main__":
    run_example(main)
