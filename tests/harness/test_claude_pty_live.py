"""Opt-in real Claude 2.1.x + ptrace + HTTP/UDS, with a synthetic model only."""

import base64
import copy
import json
import os
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from agency.agskill import agskill
from agency.configs.agconfig import agconfig, dataloggerconfig, hostserverconfig, llmconfig
from agency.engine.clients import HarnessInteractionClient
from agency.engine.host_servers.host_server_manager import HostServerManager
from agency.harness.daemon import HarnessManager
from agency.harness.protocol import HarnessAttemptRequest, PromptPayload
from agency.llm.usage_tracker import LlmUsageTracker
from agency.observability.agdatalogger import agDataLogger

pytestmark = pytest.mark.skipif(
    os.environ.get("AGENCY_TEST_CLAUDE_PTY") != "1",
    reason="opt-in real Linux Claude PTY test",
)


class SyntheticBackend:
    model = "claude-sonnet-4-6"

    def __init__(self, mode):
        self.mode = mode
        self.requests = []
        self.started = threading.Event()
        self.interrupted = threading.Event()

    def build_kwargs(self, messages, tools):
        return {"messages": messages, "tools": tools}

    def dispatch_stream(self, request, *, on_client=None):
        self.requests.append(copy.deepcopy(request))
        text = json.dumps(request["messages"])
        if "SLOW_GENERATION" in text and "REDIRECT_EVIDENCE" not in text:
            if self.mode == "tool":
                yield {
                    "type": "content",
                    "index": 0,
                    "block_type": "tool_use",
                    "id": "slow-bash",
                    "name": "Bash",
                    "arguments": json.dumps(
                        {"command": "sleep 30", "description": "interrupt probe"}
                    ),
                }
                yield {"type": "usage", "usage": None, "stop_reason": "tool_use"}
                return
            if on_client:
                on_client(SimpleNamespace(close=self.interrupted.set))
            self.started.set()
            assert self.interrupted.wait(30), (
                "Claude disconnect did not cancel the host model request"
            )
            return
        yield {"type": "content", "index": 0, "block_type": "text", "text": "NATIVE_REPLY"}
        yield {
            "type": "usage",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "stop_reason": "stop",
        }


@pytest.mark.timeout(150)
@pytest.mark.parametrize("mode", ["generation", "tool"])
def test_real_claude_interrupt_acknowledgment_and_resumed_transcript(mode):
    root = Path("/tmp") / ("agency-redirect-live-" + uuid.uuid4().hex[:8])
    root.mkdir()
    config = agconfig(
        llmconfig(model="claude-sonnet-4-6"),
        hostserverconfig(uds_path=str(root / "host.sock")),
        dataloggerconfig(db_path=str(root / "agent.db")),
    )
    config.harness_adapter.binary_path = "/usr/local/bin/claude"
    logger = agDataLogger(config)
    ag = SimpleNamespace(
        agname="redirect-live",
        harness="claude_code",
        agconfig=config,
        data_logger=logger,
        llm_usage_tracker=LlmUsageTracker(),
    )
    backend = SyntheticBackend(mode)
    skill = agskill("probe", "Follow the user instruction.")
    from agency.agpolicy import agpolicy

    def syscall(event):
        if event.argv and event.argv[0].endswith("sleep"):
            backend.started.set()
        return True

    skill.policy = agpolicy(syscall_hooks={"execve": syscall})
    daemon = HarnessManager(
        str(root / "sandbox.sock"),
        str(root / "host.sock"),
        ag.agname,
        "claude_code",
        agconfig=config,
        harness_api_port=0,
    )
    host = None
    worker = None
    results, errors = [], []
    try:
        daemon.start()
        prior = None
        for index, prompt in enumerate(("FIRST_RUN", "SLOW_GENERATION")):
            token = uuid.uuid4().hex
            host = HostServerManager(
                ag, SimpleNamespace(), skill, SimpleNamespace(), request_id=str(index)
            )
            host.llm_handler_server._backend = backend
            host.bind_attempt_token(token)
            host.start()
            request = HarnessAttemptRequest(
                harness="claude_code",
                prompt=PromptPayload("", prompt),
                request_id=str(index),
                attempt_token=token,
                resume_session_id=prior.session_id if prior else None,
                prior_session_blob_b64=prior.session_blob_b64 if prior else None,
            )

            def run():
                try:
                    with HarnessInteractionClient(
                        str(root / "sandbox.sock"), timeout_s=100
                    ) as client:
                        results.append(client.run_harness_attempt(request))
                except BaseException as exc:
                    errors.append(exc)

            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            if index == 1:
                assert backend.started.wait(40), (results, errors)
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    handler = daemon._redirect_handler
                    if handler is not None and handler.__self__._active:
                        break
                    time.sleep(0.01)
                with HarnessInteractionClient(str(root / "sandbox.sock"), timeout_s=50) as client:
                    assert client.redirect_harness("0", "MUST_NOT_REACH_SECOND_RUN") is False
                    assert client.redirect_harness("1", "REDIRECT_EVIDENCE") is True
            worker.join(65)
            assert not worker.is_alive(), "Claude attempt did not finish"
            assert not errors, errors
            result = results[-1]
            assert result.ok, result.error_message
            assert result.final_text == "NATIVE_REPLY"
            blob = base64.b64decode(result.session_blob_b64)
            (root / f"transcript-{index}.jsonl").write_bytes(blob)
            if prior:
                assert result.session_id == prior.session_id
                assert b"FIRST_RUN" in blob
                assert b"REDIRECT_EVIDENCE" in blob
                assert b"MUST_NOT_REACH_SECOND_RUN" not in blob
            prior = result
            host.clear_attempt_token(token)
            host.stop()
            host = None
        assert any("REDIRECT_EVIDENCE" in json.dumps(r) for r in backend.requests)
        if mode == "generation":
            assert backend.interrupted.is_set()
    finally:
        daemon.control("cancel", request_id=daemon._current_request_id)
        if worker is not None:
            worker.join(5)
        daemon.stop()
        if host is not None:
            host.stop()
        logger.stop()
        print(f"Claude PTY evidence: {root}")


@pytest.mark.timeout(240)
def test_public_agent_redirect_and_late_queue_in_docker(monkeypatch, tmp_path):
    from agency import agent, agdata
    from agency.configs.agconfig import agentconfig, sandboxconfig
    from agency.engine.host_servers import llm_handler_server

    backend = SyntheticBackend("generation")
    monkeypatch.setattr(llm_handler_server.agllm, "for_config", lambda config: backend)
    config = agconfig(
        agentconfig(log_dir=str(tmp_path / "logs")),
        llmconfig(model=backend.model, api_key="unused"),
        sandboxconfig(backend="docker", base_image="agency-sandbox:latest"),
    )
    owner = agent(harness="claude_code", agconfig=config)
    skill = agskill("public-redirect", "Respond to the user.")
    result = None
    try:
        first = owner.run(skill, agdata(instruction="FIRST_RUN"))
        assert first.wait(timeout=90).to_dict() == {"result": "NATIVE_REPLY"}
        result = owner.run(skill, agdata(instruction="SLOW_GENERATION"))
        assert backend.started.wait(60)
        # The model request follows the native submit hook; allow its filesystem
        # acknowledgment to be collected by the sandbox adapter polling thread.
        time.sleep(0.1)
        owner.redirect(result, "REDIRECT_EVIDENCE")
        assert result.wait(timeout=60).to_dict() == {"result": "NATIVE_REPLY"}
        assert backend.interrupted.is_set()
        # The successful run hibernated its sandbox. A late redirect must enqueue
        # without trying to reach that stopped daemon or waking a new execution.
        owner.redirect(first, "LATE_CONTEXT_EVIDENCE")
        final = owner.run(skill, agdata(instruction="NEXT_RUN"))
        assert final.wait(timeout=90).to_dict() == {"result": "NATIVE_REPLY"}
        assert "LATE_CONTEXT_EVIDENCE" in json.dumps(backend.requests[-1])
        assert "REDIRECT_EVIDENCE" in json.dumps(backend.requests[-1])
    finally:
        if result is not None and result.is_pending():
            owner.cancel(result)
        if owner.sandbox is not None:
            owner.sandbox.rm_container()
