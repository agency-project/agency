"""Opt-in: the real Kimi Code CLI through the shared runner, synthetic model only."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agency.configs.agconfig import agconfig
from agency.harness.adapters.base import AdapterRuntime, HarnessAdapter
from agency.harness.adapters.pty.driver import driver_for
from agency.harness.adapters.pty.execution import PtyExecution

pytestmark = pytest.mark.skipif(
    os.environ.get("AGENCY_TEST_EXTERNAL_PTY") != "1",
    reason="opt-in real Linux external PTY test",
)

REPLY = "hello from the synthetic model"


def _binary():
    root = Path(
        os.environ.get("AGENCY_TEST_HARNESS_BIN_DIR", Path.home() / ".cache/agency_harness_bin")
    )
    return root / "kimi"


class _Synthetic(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        if self.path.startswith("/agpolicy/"):
            body = json.dumps({"decision": "allow", "call_id": "call-1"}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()

        def chunk(delta, finish=None):
            return (
                "data: "
                + json.dumps(
                    {
                        "id": "1",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "synthetic",
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                )
                + "\n\n"
            ).encode()

        try:
            self.wfile.write(chunk({"role": "assistant", "content": ""}))
            self.wfile.write(chunk({"content": REPLY}))
            self.wfile.write(chunk({}, "stop"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


@pytest.mark.timeout(180)
def test_real_kimi_resumes_from_the_portable_session_blob(tmp_path):
    binary = _binary()
    if not binary.exists():
        pytest.skip(f"kimi executable not installed at {binary}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Synthetic)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    config = agconfig()
    config.harness_adapter.binary_path = str(binary)
    runtime = AdapterRuntime(
        config,
        "synthetic-model",
        "engine",
        base_url,
        "attempt-token",
        SimpleNamespace(check=lambda *a, **k: True),
        register_control_handle=Mock(),
        register_redirect=Mock(),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = HarnessAdapter.for_config("kimi", config)

    first_root = tmp_path / "first-state"
    first_root.mkdir()
    first_driver = driver_for(adapter, runtime, first_root, None, None, 4)
    first_driver.cwd = str(workspace)
    first_execution = PtyExecution(first_driver, runtime)
    first_execution.START_TIMEOUT = 90
    try:
        first = first_execution.run("say hello")

        second_root = tmp_path / "second-state"
        second_root.mkdir()
        second_driver = driver_for(
            adapter,
            runtime,
            second_root,
            first.session_id,
            first.session_blob,
            4,
        )
        second_driver.cwd = str(workspace)
        second_execution = PtyExecution(second_driver, runtime)
        second_execution.START_TIMEOUT = 90
        second = second_execution.run("say hello again")
    finally:
        server.shutdown()

    assert first.ok
    assert second.ok
    # The answer comes from the persisted transcript, never from the screen.
    assert first.final_text == REPLY
    assert second.final_text == REPLY
    assert first.session_id.startswith("session_")
    assert second.session_id == first.session_id
    bundle = json.loads(second.session_blob)
    assert bundle["harness"] == "kimi"
    assert bundle["session_id"] == second.session_id
    assert "session_index.jsonl" in bundle["files"]
    assert any(name.startswith("workspace-trust/wd_") for name in bundle["files"])
