"""Opt-in managed Codex parity probe, with native CLI and synthetic inference.

AGENCY_TEST_CODEX_PARITY=1 AGENCY_TEST_CODEX_BINARY=/path/to/codex \
python -m pytest tests/harness/test_codex_interactive_parity.py -v -s

This exercises the normal HarnessManager/host-services/ptrace/PTY route on
the host. It does not claim to exercise container checkpointing or make a
live provider request. Normal executable/version checks remain enabled.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from agency.agpolicy import agpolicy
from agency.agskill import agskill
from agency.configs.agconfig import agconfig, dataloggerconfig, hostserverconfig, llmconfig
from agency.engine.clients import HarnessInteractionClient
from agency.engine.host_servers.host_server_manager import HostServerManager
from agency.harness.daemon import HarnessManager
from agency.harness.protocol import HarnessAttemptRequest, PromptPayload
from agency.llm.usage_tracker import LlmUsageTracker
from agency.observability.agdatalogger import agDataLogger, read_llm_exchanges

pytestmark = pytest.mark.skipif(
    os.environ.get("AGENCY_TEST_CODEX_PARITY") != "1",
    reason="opt-in real managed Codex parity test",
)


class NativeParityBackend:
    model = "gpt-6-luna"

    def __init__(self):
        self.requests = []

    def build_kwargs(self, messages, tools):
        return {"messages": messages, "tools": tools}

    def fetch_context_limit(self):
        return 200_000

    def dispatch_stream(self, request, *, on_client=None):
        self.requests.append(copy.deepcopy(request))
        assert len(self.requests) <= 12, "native tool loop did not converge"
        messages = request["messages"]
        text = json.dumps(messages)
        done = any(
            b.get("type") == "tool_result" and "PARITY_TOOL_OK" in str(b.get("text", ""))
            for m in messages
            for b in m.get("blocks", [])
        )
        names = [t.get("function", t).get("name") for t in request.get("tools") or []]
        title = "Generate a concise, single-line task title" in text
        tool = next(
            (n for n in ("exec_command", "shell_command", "shell", "Bash") if n in names),
            next((n for n in names if n and n.endswith("__exec__custom")), None),
        )
        if tool and not done and not title:
            if tool.endswith("__exec__custom"):
                arguments = {
                    "input": 'text(await tools.exec_command({cmd:"printf PARITY_TOOL_OK"}));'
                }
            elif tool == "exec_command":
                arguments = {"cmd": "printf PARITY_TOOL_OK"}
            elif tool == "shell":
                arguments = {"command": ["sh", "-c", "printf PARITY_TOOL_OK"]}
            else:
                arguments = {"command": "printf PARITY_TOOL_OK"}
            yield {
                "type": "content",
                "index": 0,
                "block_type": "tool_use",
                "id": "parity-tool",
                "name": tool,
                "arguments": json.dumps(arguments),
            }
            stop = "tool_use"
        else:
            yield {
                "type": "content",
                "index": 0,
                "block_type": "text",
                "text": "NATIVE_PARITY_DONE",
            }
            stop = "stop"
        yield {
            "type": "usage",
            "usage": {"prompt_tokens": 2000, "completion_tokens": 20},
            "stop_reason": stop,
        }


@pytest.mark.timeout(120)
def test_managed_codex_large_native_prompt_tool_and_completion(monkeypatch, tmp_path):
    from agency.harness.adapters.pty.driver import PtyDriver
    from agency.harness.adapters.pty.execution import PtyExecution

    binary = Path(
        os.environ.get("AGENCY_TEST_CODEX_BINARY", "/tmp/agency-merge-validation/clis/codex/codex")
    )
    version = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    original_init = PtyDriver.__init__

    def init(driver, *args, **kwargs):
        original_init(driver, *args, **kwargs)
        driver.cwd = str(tmp_path)
        driver.argv += ["-c", f'projects.{json.dumps(str(tmp_path))}.trust_level="trusted"']

    monkeypatch.setattr(PtyDriver, "__init__", init)
    executions = []
    original_run = PtyExecution.run

    def run(execution, prompt, **kwargs):
        executions.append(execution)
        return original_run(execution, prompt, **kwargs)

    monkeypatch.setattr(PtyExecution, "run", run)
    root = Path("/tmp") / ("agency-codex-parity-" + uuid.uuid4().hex[:8])
    root.mkdir()
    config = agconfig(
        llmconfig(
            provider="openai_responses",
            model="gpt-6-luna",
            base_url="https://synthetic.invalid/v1",
            api_key="parity-fixture-unused",
        ),
        hostserverconfig(uds_path=str(root / "host.sock")),
        dataloggerconfig(db_path=str(root / "agent.sqlite3")),
    )
    config.harness_adapter.binary_path = str(binary)
    logger = agDataLogger(config)
    logger.start()
    agent = SimpleNamespace(
        agname="codex-parity",
        harness="codex",
        agconfig=config,
        data_logger=logger,
        llm_usage_tracker=LlmUsageTracker(),
    )
    skill = agskill("parity", "Follow the user's task.")
    skill.policy = agpolicy()
    backend = NativeParityBackend()
    daemon = HarnessManager(
        str(root / "daemon.sock"),
        str(root / "host.sock"),
        agent.agname,
        "codex",
        agconfig=config,
        harness_api_port=0,
    )
    host = None
    worker = None
    results, errors = [], []
    prompt = (
        "Run the requested shell command, then answer NATIVE_PARITY_DONE.\n"
        + ("This is harmless padding to verify native large-paste prompt acknowledgment.\n" * 32)
        + "Run printf PARITY_TOOL_OK now."
    )
    assert len(prompt) > 1000
    token = uuid.uuid4().hex
    try:
        daemon.start()
        host = HostServerManager(
            agent,
            SimpleNamespace(current_gpu_ids=lambda: None, ensure_gpu_acquired=lambda *a, **k: None),
            skill,
            SimpleNamespace(),
            request_id="parity",
        )
        host.llm_handler_server._backend = backend
        host.bind_attempt_token(token)
        host.start()
        request = HarnessAttemptRequest(
            harness="codex",
            prompt=PromptPayload("", prompt),
            request_id="parity",
            attempt_token=token,
        )

        def execute():
            try:
                with HarnessInteractionClient(str(root / "daemon.sock"), timeout_s=95) as client:
                    results.append(client.run_harness_attempt(request))
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=execute, daemon=True)
        worker.start()
        worker.join(100)
        assert not worker.is_alive(), "managed native prompt acknowledgment/completion timed out"
        assert not errors, errors
        assert results and results[-1].ok, results
        assert results[-1].final_text == "NATIVE_PARITY_DONE"
        assert len(backend.requests) >= 2, "model did not receive a tool result"
        assert any(
            "PARITY_TOOL_OK" in str(b.get("text", ""))
            for r in backend.requests
            for m in r["messages"]
            for b in m.get("blocks", [])
            if b.get("type") == "tool_result"
        )
        logger.flush()
        assert read_llm_exchanges(logger.db_path)
        print(
            f"{version}: managed >1000-char prompt, tool result, completion, SQLite passed; evidence={root}"
        )
    finally:
        daemon.control("cancel", request_id="parity")
        if worker:
            worker.join(5)
        daemon.stop()
        if host:
            host.stop()
        logger.stop()
        for execution in executions:
            if execution.handle:
                lines, _, _, _ = execution.handle.terminal_screen()
                (root / "terminal.txt").write_text("\n".join(lines))
        (root / "requests.json").write_text(json.dumps(backend.requests, indent=2))
