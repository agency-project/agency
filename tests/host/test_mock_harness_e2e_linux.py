"""Opt-in public Agent.run example using real EC2 harnesses and a replayed LLM.

Run with AGENCY_HOST_MOCK_E2E=1 and AGENCY_TEST_HARNESS_IMAGE set to an
image containing all six CLI harnesses. No model credentials are required.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest

from agency import Agent, agDataLogger, agconfig, agdata, agrawstring, agskill
from agency.configs.agconfig import (
    agentconfig,
    dataloggerconfig,
    llmconfig,
    resourcesconfig,
    sandboxconfig,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("AGENCY_HOST_MOCK_E2E") != "1",
    reason="requires opt-in EC2 container runtime and installed six-harness image",
)

HARNESSES = ("native", "claude_code", "codex", "grok", "opencode", "kimi")
ANSWER = "MOCK_E2E_OK"


def _response_chain(payloads: "list[dict]") -> "list[tuple[str, dict]]":
    """Hash each block deterministically for storage -- the mock backend
    (agency/llm/mock.py) only ever reads an exchange's response chain, so
    this fixture never needs a prompt chain at all."""
    return [
        (hashlib.sha256(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest(), p)
        for p in payloads
    ]


def _write_replay(path: Path) -> None:
    logger = agDataLogger(agconfig(dataloggerconfig(db_path=str(path))))
    logger.start()
    try:
        for index in range(8):
            logger.record_llm_exchange(
                f"mock-e2e-{index}",
                exchange_type="llm_block",
                prompt_chain=[],
                response_chain=_response_chain(
                    [
                        {"type": "text", "index": 0, "text": ANSWER},
                        {
                            "type": "metadata",
                            "index": 2**31 - 1,
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 3,
                                "total_tokens": 13,
                            },
                            "stop_reason": "stop",
                            "data": [],
                        },
                    ]
                ),
            )
    finally:
        logger.stop()


@pytest.mark.timeout(300)
@pytest.mark.parametrize("harness", HARNESSES)
def test_two_public_invocations_with_mock_llm(harness: str, tmp_path: Path) -> None:
    image = os.environ["AGENCY_TEST_HARNESS_IMAGE"]
    replay_path = tmp_path / "replay.sqlite3"
    _write_replay(replay_path)
    config = agconfig(
        agentconfig(harness=harness, log_dir=str(tmp_path / "logs")),
        sandboxconfig(
            backend=os.environ.get("AGENCY_TEST_CONTAINER_BACKEND", "podman"),
            base_image=image,
            gpu_passthrough=False,
        ),
        resourcesconfig(idle_cpus=1, idle_memory="2g"),
        llmconfig(
            provider="mock",
            model="claude-sonnet-4-6",
            context_limit=200_000,
            replay_db_path=str(replay_path),
            timing_mode="instant",
        ),
    )
    skill = agskill(
        "mock_harness_example",
        "Return a short plain-text answer. No tools are needed.",
        input_schema=agdata(instruction=agrawstring),
        output_schema=agdata(answer=agrawstring),
        max_output_schema_retries=0,
    )
    agent = Agent(agconfig=config)
    answers = []
    try:
        for instruction in ("First mock example turn", "Second mock example turn"):
            result = agent.run(skill, agdata(instruction=instruction), max_steps=4)
            result.wait(timeout=180)
            answer = result.to_dict()
            assert answer == {"answer": ANSWER}, answer
            answers.append(answer)
        assert ANSWER in json.dumps(agent.context.get_resolved_transcript())
    finally:
        (tmp_path / "result.json").write_text(
            json.dumps({"harness": harness, "answers": answers}, indent=2) + "\n"
        )
        if agent.sandbox is not None:
            agent.sandbox.destroy()
