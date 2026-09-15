"""Paid, opt-in public run() E2E on a host provisioned by agency setup-host.

Launch via agency run with AGENCY_HOST_LUNA_E2E=1, AGENCY_TEST_API_KEY_FILE,
AGENCY_TEST_HARNESS_IMAGE and per-CLI binary paths. Never substitutes model replies.
"""

import json
import os
import shlex
import time
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("AGENCY_HOST_LUNA_E2E") != "1",
    reason="requires explicit live Luna calls and a provisioned rootful host",
)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("harness", ["native", "claude_code", "codex", "grok", "opencode", "kimi"])
def test_live_luna_continues_across_invocations(harness, tmp_path, monkeypatch):
    import agency
    from agency.llm.agllm import agllm

    exposed_tools = []
    build_kwargs = agllm.build_kwargs

    def record_tools(backend, messages, openai_tools=None):
        exposed_tools.append(
            [tool.get("function", tool).get("name") for tool in (openai_tools or [])]
        )
        return build_kwargs(backend, messages, openai_tools)

    monkeypatch.setattr(agllm, "build_kwargs", record_tools)

    assert os.environ.get("AGENCY_HOST_CONFIG"), "Launch with agency run to test the saved profile"
    cfg = agency.agconfig()
    assert cfg.sandbox.checkpoint_backend == "cow_zfs"
    cfg.sandbox.base_image = os.environ["AGENCY_TEST_HARNESS_IMAGE"]
    cfg.agent.harness = harness
    cfg.agent.log_dir = str(tmp_path / "logs")
    cfg.llm.api_key = Path(os.environ["AGENCY_TEST_API_KEY_FILE"]).read_text().strip()
    cfg.llm.provider = "openai"
    cfg.llm.model = "gpt-5.6-luna"
    cfg.llm.reasoning_effort = "none"
    cfg.llm.temperature = 0
    cfg.llm.context_limit = 64000
    cfg.llm.max_completion_tokens = 4096
    cfg.ptrace.file_access = True
    cfg.resources.idle_cpus = 1
    cfg.resources.idle_memory = "3g"
    if harness != "native":
        cfg.harness_adapter.binary_path = os.environ[f"AGENCY_TEST_{harness.upper()}_BINARY"]
    artifact = Path(os.environ.get("AGENCY_HOST_E2E_RESULTS", str(tmp_path))) / harness
    artifact.mkdir(parents=True, exist_ok=True)
    agent = agency.Agent("host-luna-" + harness + "-" + uuid.uuid4().hex[:6], agconfig=cfg)

    # This function ships into an image without pytest. Rewritten assertions
    # would pull pytest modules into its serialized globals.
    def checkpoint_probe(arg):
        from pathlib import Path
        from agency.agdata import agdata

        turn = int(arg._data["turn"])
        token = arg._data["token"]
        path = Path("/workspace/host-luna-state")
        expected = [token, *[f"turn-{n}" for n in range(turn)]]
        if turn == 0:
            if path.exists():
                raise RuntimeError("Initial state already exists")
            path.write_text(token + "\n")
        else:
            if path.read_text().splitlines() != expected:
                raise RuntimeError("Prior checkpoint state was lost")
        with path.open("a") as stream:
            stream.write(f"turn-{turn}\n")
        return agdata(summary=path.read_text())

    probe = agency.agtool(
        "checkpoint_probe",
        "Advance and read the checkpoint test file inside the sandbox.",
        checkpoint_probe,
        params={
            "type": "object",
            "properties": {
                "turn": {"type": "integer"},
                "token": {"type": "string"},
            },
            "required": ["turn", "token"],
        },
    )
    # These adapters expose their CLI's own terminal tools, rather than Agency's
    # sandbox MCP server. Exercise the same file transition through that interface.
    terminal_probe = harness in {"grok", "opencode", "kimi"}
    prompt = (
        "Call the checkpoint_probe sandbox tool exactly once with the turn and token "
        "specified in instruction. If it is not listed, use the harness's tool discovery "
        "or MCP wait tools to find and invoke it. Copy its summary verbatim to your "
        "output summary and finish. Do not use shell commands or other filesystem tools."
    )
    if terminal_probe:
        prompt = (
            "Execute the provided command exactly once using your built-in terminal tool. "
            "Copy its stdout verbatim to your output summary and finish."
        )
    skill = agency.agskill(
        name="host_checkpoint_probe",
        prompt=prompt,
        input_schema=agency.agdata(instruction=str),
        output_schema=agency.agdata(summary=agency.agrawstring),
        add_sandbox_mcp_tools=[] if terminal_probe else [probe],
    )
    token = uuid.uuid4().hex
    invocation_count = int(os.environ.get("AGENCY_HOST_E2E_INVOCATIONS", "3"))
    assert invocation_count >= 2
    checkpoints, identities, durations, answers = [], [], [], []
    try:
        with agency.agprof.session(
            artifact / "profile", sample_hz=2, sample_gpu=False, auto_functions=False
        ):
            for turn in range(invocation_count):
                instruction = f"Call checkpoint_probe with turn={turn} and token={token}."
                if terminal_probe:
                    expected = "\n".join([token, *[f"turn-{n}" for n in range(turn)]]) + "\n"
                    script = (
                        "from pathlib import Path\n"
                        "p = Path('/workspace/host-luna-state')\n"
                        + (
                            f"assert not p.exists()\np.write_text({token!r} + '\\n')\n"
                            if turn == 0
                            else f"assert p.read_text() == {expected!r}\n"
                        )
                        + f"with p.open('a') as f: f.write('turn-{turn}\\n')\n"
                        "print(p.read_text(), end='')\n"
                    )
                    instruction = "Run: python3 -c " + shlex.quote(script)
                started = time.monotonic()
                result = agent.run(skill, agency.agdata(instruction=instruction), max_steps=12)
                result.wait()
                durations.append(time.monotonic() - started)
                answer = result.to_dict()
                answers.append(answer)
                assert "error" not in answer, answer
                text = json.dumps(answer)
                assert token in text and f"turn-{turn}" in text, answer
                checkpoint = agent.sandbox._backend._checkpointer.latest
                assert checkpoint is not None and agent.sandbox._backend._checkpointer.hibernated
                checkpoints.append(checkpoint)
                if cfg.sandbox.checkpoint_fast_resume:
                    assert checkpoint.stats["fast_resume_available"] is True
                    assert checkpoint.stats["fast_resume_live_pty"] is (harness != "native")
                    session = checkpoint.stats["sessions"][0]
                    identities.append((session["daemon_pid"], session.get("root_pid")))
                    if turn:
                        assert checkpoints[-2].stats.get("fast_resume_used") is True
                else:
                    assert checkpoint.stats.get("fast_resume_available") is False
                    assert checkpoint.stats.get("fast_resume_live_pty") is False
                    assert not checkpoint.stats.get("fast_resume_used")
            assert agent.sandbox.read_file("/workspace/host-luna-state").splitlines() == [
                token,
                *[f"turn-{turn}" for turn in range(invocation_count)],
            ]
            if cfg.sandbox.checkpoint_fast_resume:
                assert len(set(identities)) == 1
                assert all(cp.stats.get("fast_resume_used") is True for cp in checkpoints)
    finally:
        (artifact / "result.json").write_text(
            json.dumps(
                {
                    "harness": harness,
                    "model": cfg.llm.model,
                    "runtime": cfg.sandbox.backend,
                    "fast_resume": cfg.sandbox.checkpoint_fast_resume,
                    "invocation_seconds": durations,
                    "identities": identities,
                    "checkpoints": [cp.stats for cp in checkpoints],
                    "answers": answers,
                    "exposed_tools": exposed_tools,
                },
                indent=2,
            )
        )
        try:
            agent.sandbox.destroy()
        finally:
            agency.get_orchestrator().shutdown()
