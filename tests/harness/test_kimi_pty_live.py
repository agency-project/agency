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
from agency.harness.adapters.agharness_backend import AdapterRuntime, agharness_backend
from agency.harness.adapters.pty_drivers import driver_for
from agency.harness.adapters.pty_session import PtyExecution

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
def test_real_kimi_completes_a_turn_through_the_shared_runner(tmp_path):
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
    root = tmp_path / "state"
    root.mkdir()
    driver = driver_for(agharness_backend.for_config("kimi", config), runtime, root, None, None, 4)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    driver.cwd = str(workspace)

    execution = PtyExecution(driver, runtime)
    execution.START_TIMEOUT = 90
    try:
        result = execution.run("say hello")
    finally:
        server.shutdown()

    assert result.ok
    # The answer comes from the persisted transcript, never from the screen.
    assert result.final_text == REPLY
    assert result.session_id.startswith("session_")
    bundle = json.loads(result.session_blob)
    assert bundle["harness"] == "kimi"
    assert bundle["session_id"] == result.session_id
    assert "session_index.jsonl" in bundle["files"]
