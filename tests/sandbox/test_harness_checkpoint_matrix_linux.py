"""Opt-in real harness/container/checkpoint matrix with deterministic model replies.

Every CLI and Agency's model gateway are real; only the upstream model is
synthetic so session and filesystem continuity do not depend on model behavior.
"""

import copy
import json
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx

from agency.agskill import agskill
from agency.configs.agconfig import agconfig
from agency.engine.harness_daemon_launcher import ensure_harness_daemon
from agency.engine.clients.harness_interaction_client import HarnessInteractionClient
from agency.engine.host_servers.host_server_manager import HostServerManager
from agency.harness.protocol import HarnessAttemptRequest, PromptPayload
from agency.llm.usage_tracker import LlmUsageTracker
from agency.observability.agdatalogger import agDataLogger
from agency.sandbox.agsandbox import agSandbox
from agency.utils.agutil import new_uds_path

pytestmark = pytest.mark.skipif(
    os.environ.get("AGENCY_CHECKPOINT_HARNESS_MATRIX") != "1",
    reason="requires opt-in rootful EC2 ZFS/CRIU and installed harnesses",
)


class ReplyBackend:
    model = "claude-sonnet-4-6"

    def __init__(self):
        self.requests = []

    def fetch_context_limit(self):
        return 32000

    def build_kwargs(self, messages, tools):
        return {"messages": messages, "tools": tools}

    def dispatch_stream(self, request, *, on_client=None):
        self.requests.append(copy.deepcopy(request))
        yield {"type": "content", "index": 0, "block_type": "text", "text": "MATRIX_REPLY"}
        yield {
            "type": "usage",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "stop_reason": "stop",
        }


@pytest.mark.timeout(300)
@pytest.mark.parametrize("runtime,fast", [("podman", True), ("docker", False)])
@pytest.mark.parametrize("harness", ["native", "claude_code", "codex", "grok", "opencode", "kimi"])
def test_harness_continues_after_checkpoint(runtime, fast, harness, tmp_path, monkeypatch):
    lifecycle_errors = []
    prepare = HarnessInteractionClient.prepare_fast_checkpoint

    def prepare_with_diagnostics(client):
        try:
            return prepare(client)
        except httpx.HTTPStatusError as exc:
            lifecycle_errors.append(exc.response.json())
            raise

    monkeypatch.setattr(
        HarnessInteractionClient, "prepare_fast_checkpoint", prepare_with_diagnostics
    )
    cfg = agconfig()
    cfg.agent.harness = harness
    cfg.llm.model = ReplyBackend.model
    cfg.sandbox.backend = runtime
    cfg.sandbox.base_image = os.environ["AGENCY_MATRIX_IMAGE"]
    cfg.sandbox.checkpoint_backend = "cow_zfs"
    cfg.sandbox.checkpoint_fast_resume = fast
    cfg.sandbox.checkpoint_zfs_parent = os.environ[f"AGENCY_MATRIX_{runtime.upper()}_PARENT"]
    cfg.resources.idle_cpus = 2
    cfg.ptrace.file_access = True
    cfg.data_logger.db_path = str(tmp_path / "agent.db")
    cfg.host_server.uds_path = str(new_uds_path("matrix-host"))
    if harness not in {"native", "kimi"}:
        binary = "claude" if harness == "claude_code" else harness
        cfg.harness_adapter.binary_path = str(
            Path(os.environ["AGENCY_TEST_HARNESS_BIN_DIR"]) / binary
        )
    if harness == "kimi":
        cfg.harness_adapter.binary_path = os.environ["AGENCY_TEST_KIMI_BINARY"]
    if harness == "claude_code" and os.environ.get("AGENCY_TEST_CLAUDE_BINARY"):
        cfg.harness_adapter.binary_path = os.environ["AGENCY_TEST_CLAUDE_BINARY"]
    sandbox = agSandbox("matrix-" + harness, agconfig=cfg)
    logger = agDataLogger(cfg)
    agent = SimpleNamespace(
        agname="matrix-" + harness,
        harness=harness,
        agconfig=cfg,
        data_logger=logger,
        llm_usage_tracker=LlmUsageTracker(),
    )
    backend = ReplyBackend()
    skill = agskill("checkpoint_probe", "Reply to the user.")
    prior = None
    identities, checkpoints = [], []
    host = None
    try:
        turns = int(os.environ.get("AGENCY_MATRIX_TURNS", "3"))
        assert turns >= 3
        for turn in range(turns):
            if turn:
                sandbox.restore(checkpoints[-1])
                assert sandbox.read_file("/workspace/matrix-state") == str(turn - 1)
                if fast:
                    assert checkpoints[-1].stats["fast_resume_used"] is True
            host = HostServerManager(agent, sandbox, skill, SimpleNamespace(), request_id=str(turn))
            host.llm_handler_server._backend = backend
            token = uuid.uuid4().hex
            host.bind_attempt_token(token)
            host.start()
            daemon = ensure_harness_daemon(
                sandbox,
                cfg.host_server.uds_path,
                agent.agname,
                harness,
                agconfig=cfg,
            )
            with daemon.client(timeout_s=150) as client:
                identities.append(client.daemon_identity())
                result = client.run_harness_attempt(
                    HarnessAttemptRequest(
                        harness=harness,
                        prompt=PromptPayload("", f"CHECKPOINT_TURN_{turn}: say hello"),
                        request_id=str(turn),
                        attempt_token=token,
                        resume_session_id=prior.session_id if prior else None,
                        prior_session_blob_b64=prior.session_blob_b64 if prior else None,
                        max_steps=4,
                    )
                )
            assert result.ok, result.error_message
            assert result.final_text == "MATRIX_REPLY"
            if prior:
                assert result.session_id == prior.session_id
                assert "CHECKPOINT_TURN_0" in json.dumps(backend.requests[-1])
            prior = result
            host.clear_attempt_token(token)
            host.stop()
            host = None
            sandbox.write_file("/workspace/matrix-state", str(turn))
            checkpoint = sandbox.checkpoint()
            checkpoints.append(checkpoint)
            assert not sandbox._backend._container_running()
            assert checkpoint.stats["fast_resume_available"] is fast, lifecycle_errors
            if fast:
                assert checkpoint.stats["fast_resume_live_pty"] is (harness != "native")
        if fast:
            # CRIU preserves namespace PIDs; kernel start ticks change on restore.
            assert len({identity[0] for identity in identities}) == 1
            if harness != "native":
                roots = [cp.stats["sessions"][0]["root_pid"] for cp in checkpoints]
                assert len(set(roots)) == 1
        else:
            assert len({tuple(identity) for identity in identities}) == turns
    finally:
        destination = Path(os.environ.get("AGENCY_MATRIX_RESULTS", str(tmp_path)))
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"{harness}-{runtime}.json").write_text(
            json.dumps(
                {
                    "harness": harness,
                    "runtime": runtime,
                    "fast": fast,
                    "daemon_identities": identities,
                    "checkpoint_stats": [cp.stats for cp in checkpoints],
                    "model_requests": len(backend.requests),
                    "lifecycle_errors": lifecycle_errors,
                },
                indent=2,
            )
        )
        try:
            if host is not None:
                host.stop()
        finally:
            sandbox.destroy()
