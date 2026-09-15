from __future__ import annotations

import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

from agency.configs.agconfig import agconfig, dataloggerconfig, hostserverconfig, llmconfig
from agency.observability.agdatalogger import agDataLogger
from agency.agpolicy import agpolicy
from agency.agskill import agskill
from agency.engine.host_servers.host_server_manager import HostServerManager
from agency.engine.clients import HarnessInteractionClient
from agency.harness.clients import HostServicesClient
from agency.harness.daemon import HarnessManager
from agency.harness.protocol import HarnessAttemptRequest, HarnessAttemptResult, PromptPayload
from agency.llm.usage_tracker import LlmUsageTracker


def test_reverse_host_rpc_completes_while_harness_attempt_rpc_remains_open():
    socket_dir = Path(f"/tmp/agency-dual-uds-{uuid.uuid4().hex[:8]}")
    host_socket = socket_dir / "host.sock"
    sandbox_socket = socket_dir / "sandbox.sock"
    database = socket_dir / "agent.db"
    socket_dir.mkdir(parents=True)

    policy_called = threading.Event()
    attempt_received = threading.Event()
    reverse_rpc_completed = threading.Event()
    release_attempt_result = threading.Event()
    seen_requests = []
    result_holder = {}
    errors = []
    attempt_token = "dual-uds-attempt"

    def mock_policy(tool_input):
        assert tool_input == {"path": "/workspace/example.py"}
        policy_called.set()
        return True, "mock policy allowed"

    config = agconfig(
        llmconfig(model="test-model", base_url="http://x/v1"),
        hostserverconfig(uds_path=str(host_socket)),
        dataloggerconfig(db_path=str(database)),
    )
    agent = SimpleNamespace(
        agname="agent-1",
        agconfig=config,
        data_logger=agDataLogger(config),
        llm_usage_tracker=LlmUsageTracker(),
    )
    skill = agskill(
        name="mock-attempt",
        prompt="mock",
        policy=agpolicy(tool_hooks={"read_file": mock_policy}),
    )
    sandbox = SimpleNamespace(
        ensure_gpu_acquired=lambda agname, *, is_cancelled=None: None,
        current_gpu_ids=lambda: None,
    )
    host_manager = HostServerManager(agent, sandbox, skill, SimpleNamespace())
    sandbox_server = None
    host_client = None
    sandbox_client = None
    attempt_thread = None

    try:
        host_manager.bind_attempt_token(attempt_token)
        assert host_manager.start() == str(host_socket)
        host_client = HostServicesClient(str(host_socket), timeout_s=2.0)
        host_client.register_attempt_token(attempt_token)

        def mock_attempt_handler(request):
            seen_requests.append(request)
            attempt_received.set()
            decision = host_client.check_tool_policy(
                attempt_token,
                "read_file",
                {"path": "/workspace/example.py"},
            )
            assert decision["decision"] == "allow"
            assert decision["reason"] == "mock policy allowed"
            reverse_rpc_completed.set()
            assert release_attempt_result.wait(timeout=2.0)
            return HarnessAttemptResult(
                ok=True,
                final_text="mock harness completed",
                input_tokens=12,
                output_tokens=3,
                session_id="mock-session",
                session_blob_b64="bW9jay1ibG9i",
            )

        sandbox_server = HarnessManager(
            str(sandbox_socket),
            str(host_socket),
            "mock-agent",
            "claude_code",
            attempt_handler=mock_attempt_handler,
            harness_api_port=0,
        )
        assert sandbox_server.start() == str(sandbox_socket)
        sandbox_client = HarnessInteractionClient(str(sandbox_socket), timeout_s=3.0)

        request = HarnessAttemptRequest(
            harness="claude_code",
            max_steps=20,
            prompt=PromptPayload(
                system_instruction="system",
                user_content="fix the bug",
                output_instruction="return JSON",
            ),
            attempt_token=attempt_token,
        )

        def run_attempt():
            try:
                result_holder["result"] = sandbox_client.run_harness_attempt(request)
            except BaseException as exc:
                errors.append(exc)

        attempt_thread = threading.Thread(target=run_attempt, name="mock-host-attempt")
        attempt_thread.start()

        assert attempt_received.wait(timeout=2.0)
        assert policy_called.wait(timeout=2.0)
        assert reverse_rpc_completed.wait(timeout=2.0)
        assert attempt_thread.is_alive(), "outer host → sandbox RPC returned too early"
        assert host_socket.exists()
        assert sandbox_socket.exists()

        release_attempt_result.set()
        attempt_thread.join(timeout=2.0)

        assert not attempt_thread.is_alive()
        assert errors == []
        assert seen_requests == [request]
        assert result_holder["result"] == HarnessAttemptResult(
            ok=True,
            final_text="mock harness completed",
            input_tokens=12,
            output_tokens=3,
            session_id="mock-session",
            session_blob_b64="bW9jay1ibG9i",
        )
    finally:
        release_attempt_result.set()
        if attempt_thread is not None:
            attempt_thread.join(timeout=2.0)
        if sandbox_client is not None:
            sandbox_client.close()
        if host_client is not None:
            host_client.close()
        if sandbox_server is not None:
            sandbox_server.stop()
        host_manager.stop()
        host_socket.unlink(missing_ok=True)
        sandbox_socket.unlink(missing_ok=True)
        database.unlink(missing_ok=True)
        socket_dir.rmdir()
