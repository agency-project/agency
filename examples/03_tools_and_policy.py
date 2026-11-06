"""Lesson 3: call a Python tool directly, from the host, or in a sandbox."""

import os
import sys

from agency import Agent, agdata, agskill, agtool
from agency.agpolicy import agpolicy

from _common import close_sandboxes, run_example, tutorial_config


def double(arguments: agdata) -> agdata:
    return agdata(result=arguments.number * 2)


def allow_positive(arguments: dict) -> tuple[bool, str]:
    return arguments.get("number", 0) > 0, "Use a positive number."


DOUBLE = agtool(
    name="double",
    description="Double a positive integer.",
    fn=double,
    params={
        "type": "object",
        "properties": {"number": {"type": "integer"}},
        "required": ["number"],
    },
)


def main() -> None:
    print(f"direct tool result: {DOUBLE(agdata(number=6)).result}")
    print(f"policy for 21: {allow_positive({'number': 21})}")
    print(f"policy for -1: {allow_positive({'number': -1})}")

    cfg, run_dir = tutorial_config("03_tools_and_policy")
    # Serialized Python tools need the same Python version in both processes.
    cfg.sandbox.base_image = os.environ.get(
        "AGENCY_SANDBOX_IMAGE",
        f"docker.io/library/python:{sys.version_info.major}.{sys.version_info.minor}-slim",
    )
    host = Agent("host-tools", agconfig=cfg, harness="native")
    sandbox = Agent("sandbox-tools", agconfig=cfg, harness="codex")
    policy = agpolicy(tool_hooks={"double": allow_positive}, default_to_deny=False)
    try:
        for learner, tools in (
            (host, {"add_host_mcp_tools": [DOUBLE]}),
            (sandbox, {"add_sandbox_mcp_tools": [DOUBLE]}),
        ):
            skill = agskill(
                name="double_number",
                prompt=(
                    "Call double with the supplied number and return its result."
                    "\nAfter submitting all output fields, finish with a short confirmation."
                ),
                input_schema=agdata(number=int),
                output_schema=agdata(result=int),
                policy=policy,
                **tools,
            )
            result = learner.run(skill, agdata(number=21))
            print(f"{learner.agname}: {result.result}")
        print(f"artifacts: {run_dir}")
    finally:
        close_sandboxes([host, sandbox])


if __name__ == "__main__":
    run_example(main)
