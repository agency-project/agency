"""Opt-in real external TUI + ptrace + HTTP/UDS, with a synthetic model only."""

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
    os.environ.get("AGENCY_TEST_EXTERNAL_PTY") != "1",
    reason="opt-in real Linux external PTY test",
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
                names = [tool.get("function", tool).get("name") for tool in request["tools"]]
                name = next(
                    (
                        name
                        for name in [
                            "exec_command",
                            "shell_command",
                            "bash",
                            "run_terminal_command",
                            "Bash",
                        ]
                        if name in names
                    ),
                    None,
                )
                assert name is not None, names
                arguments = {"command": "sleep 30", "description": "interrupt probe"}
                if name == "exec_command":
                    arguments = {"cmd": "sleep 30", "yield_time_ms": 30000}
                yield {
                    "type": "content",
                    "index": 0,
                    "block_type": "tool_use",
                    "id": "slow-bash",
                    "name": name,
                    "arguments": json.dumps(arguments),
                }
                yield {"type": "usage", "usage": None, "stop_reason": "tool_use"}
                return
            if on_client:
                on_client(SimpleNamespace(close=self.interrupted.set))
            self.started.set()
            assert self.interrupted.wait(30), "TUI disconnect did not cancel the host model request"
            return
        yield {"type": "content", "index": 0, "block_type": "text", "text": "NATIVE_REPLY"}
        yield {
            "type": "usage",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "stop_reason": "stop",
        }


@pytest.mark.timeout(150)
@pytest.mark.parametrize("harness", ["codex", "grok", "opencode"])
@pytest.mark.parametrize("mode", ["generation", "tool"])
def test_real_external_interrupt_acknowledgment_and_resumed_session(
    harness, mode, monkeypatch, tmp_path
):
    from agency.harness.adapters.pty_drivers import PtyDriver
    from agency.harness.adapters.pty_session import PtyExecution

    executions = []
    original_run = PtyExecution.run

    def run_execution(execution, prompt):
        executions.append(execution)
        return original_run(execution, prompt)

    monkeypatch.setattr(PtyExecution, "run", run_execution)

    original_init = PtyDriver.__init__
    drivers = []

    def init(driver, *args, **kwargs):
        original_init(driver, *args, **kwargs)
        driver.cwd = str(tmp_path)
        if driver.name == "codex":
            driver.argv += ["-c", f'projects.{json.dumps(str(tmp_path))}.trust_level="trusted"']
        drivers.append(driver)

    monkeypatch.setattr(PtyDriver, "__init__", init)

    def config_home(*args):
        path = tmp_path / ("state-" + uuid.uuid4().hex[:8])
        path.mkdir()
        return path

    monkeypatch.setattr("agency.harness.agharness.materialize_config_home", config_home)
    root = Path("/tmp") / ("agency-redirect-live-" + uuid.uuid4().hex[:8])
    root.mkdir()
    config = agconfig(
        llmconfig(model="claude-sonnet-4-6"),
        hostserverconfig(uds_path=str(root / "host.sock")),
        dataloggerconfig(db_path=str(root / "agent.db")),
    )
    binary_root = Path(
        os.environ.get("AGENCY_TEST_HARNESS_BIN_DIR", Path.home() / ".cache/agency_harness_bin")
    )
    config.harness_adapter.binary_path = str(binary_root / harness)
    logger = agDataLogger(config)
    ag = SimpleNamespace(
        agname="redirect-live",
        harness=harness,
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
        harness,
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
                harness=harness,
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
                assert backend.started.wait(40), (results, errors, backend.requests)
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
            assert not worker.is_alive(), "TUI attempt did not finish"
            assert not errors, errors
            result = results[-1]
            assert result.ok, result.error_message
            assert not executions[-1].driver.root.exists()
            assert not executions[-1].handle._loop._known_pids
            assert result.final_text == "NATIVE_REPLY"
            bundle = json.loads(base64.b64decode(result.session_blob_b64))
            blob = b"\n".join(base64.b64decode(value) for value in bundle["files"].values())
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
        assert "FIRST_RUN" in json.dumps(backend.requests[-1])
        assert "REDIRECT_EVIDENCE" in json.dumps(backend.requests[-1])
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
        print(f"{harness} PTY evidence: {root}; state roots: {[str(d.root) for d in drivers]}")
        for execution in executions:
            if execution.handle:
                lines, x, y, _ = execution.handle.terminal_screen()
                print(
                    "FINAL SCREEN", x, y, "\n".join(line.rstrip() for line in lines if line.strip())
                )
                print(
                    "STATE",
                    execution._expected_prompt,
                    execution._turn_id,
                    execution._stop,
                    execution._failure,
                )
                print(
                    "PIDS",
                    execution.handle.root_pid,
                    execution.handle.returncode,
                    execution.handle.pids(),
                )


@pytest.mark.timeout(120)
def test_opencode_terminal_process_tree_is_reaped_repeatedly(tmp_path):
    from agency.harness.adapters.agharness_backend import AdapterRuntime, agharness_backend
    from agency.harness.adapters.pty_drivers import driver_for
    from agency.harness.ptrace.supervisor import agProxyPtrace

    config = agconfig()
    binary_root = Path(
        os.environ.get("AGENCY_TEST_HARNESS_BIN_DIR", Path.home() / ".cache/agency_harness_bin")
    )
    config.harness_adapter.binary_path = str(binary_root / "opencode")
    runtime = AdapterRuntime(
        config,
        "test-model",
        "reap-test",
        "http://127.0.0.1:1",
        "test-token",
        SimpleNamespace(check=lambda *args: True),
    )
    for index in range(10):
        root = tmp_path / str(index)
        root.mkdir()
        driver = driver_for(
            agharness_backend.for_config("opencode", config), runtime, root, None, None, None
        )
        handle = agProxyPtrace(config, allow_initial_exec=True).launch(
            driver.argv,
            driver.env,
            cwd=str(tmp_path),
            pty_size=(120, 36),
            policy=runtime.syscall_policy,
            ag=None,
        )
        try:
            deadline = time.monotonic() + 15
            while not driver.ready(handle) and time.monotonic() < deadline:
                time.sleep(0.025)
            assert driver.ready(handle)
            handle.close()
            assert handle.returncode is not None
            assert not handle.pids()
        finally:
            handle.close()
