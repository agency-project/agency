"""Real TLS/proxy traffic; model responses and credentials are local fixtures."""

import gzip
import base64
import json
import os
import socket
import ssl
import stat
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization

from agency.observability.agdatalogger import read_llm_exchanges
from agency.sessions.capture_proxy import ClaudeNativeCaptureConfig
from agency.sessions.claude_code import ClaudeCodeSession
from agency.sessions import session as session_module

from .helpers import eventually, actions, trajectory, otlp
from .test_capture import sse


class NativeFixture(SimpleNamespace):
    def __repr__(self):
        return "NativeClaudeFixture()"


def wire():
    return sse(
        {
            "type": "message_start",
            "message": {
                "id": "fixture-msg",
                "type": "message",
                "role": "assistant",
                "model": "fixture-native-model",
                "content": [],
                "future": {"opaque": "provider-state"},
                "usage": {"input_tokens": 10},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "thinking",
                "thinking": "private-thinking",
                "signature": "opaque-signature",
            },
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "redacted_thinking", "data": "opaque-redacted-data"},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "text", "text": "native answer"},
        },
        {"type": "content_block_stop", "index": 2},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 5},
        },
        {"type": "message_stop"},
    )


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    state = tmp_path / "claude-state"
    state.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(state))
    for name in (
        "ANTHROPIC_BASE_URL",
        "NODE_EXTRA_CA_CERTS",
        "SSL_CERT_FILE",
        "NODE_TLS_REJECT_UNAUTHORIZED",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDE_CODE_CLIENT_CERT",
        "CLAUDE_CODE_CLIENT_KEY",
        "CLAUDE_CODE_CLIENT_KEY_PASSPHRASE",
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
        "LITE_LLM_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def native(make_session):
    state = NativeFixture(
        requests=[], wire=wire(), status=200, headers={}, cancelled=threading.Event(), hang=False
    )

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            state.requests.append((self.path, raw, httpx.Headers(list(self.headers.items()))))
            self.send_response(state.status)
            self.send_header(
                "content-type", "text/event-stream" if state.status == 200 else "application/json"
            )
            for k, v in state.headers.items():
                self.send_header(k, v)
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            self.wfile.write(f"{len(state.wire):x}\r\n".encode() + state.wire + b"\r\n")
            self.wfile.flush()
            if state.hang:
                self.connection.settimeout(3)
                try:
                    self.connection.recv(1)
                    state.cancelled.set()
                except ConnectionResetError:
                    state.cancelled.set()
                except socket.timeout:
                    pass
                return
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state.session = make_session(
        session_class=ClaudeCodeSession,
        capture=ClaudeNativeCaptureConfig(f"http://127.0.0.1:{server.server_port}"),
    )
    gateway = state.session._capture_gateway
    state.proxy = f"http://agency:{state.session.host_token}@127.0.0.1:{gateway.port}"
    state.tls = ssl.create_default_context(cafile=str(gateway.directory / "ca.pem"))
    yield state
    server.shutdown()
    server.server_close()


def client(state):
    return httpx.Client(proxy=state.proxy, verify=state.tls, trust_env=False, timeout=5)


def rows(state, count=1):
    def check():
        state.session._agent_logger.flush()
        result = read_llm_exchanges(state.session._agent_logger.db_path)
        assert len(result) == count
        return result

    return eventually(check)


def manifests(state):
    return [
        json.loads(p.read_text())
        for p in (state.session.run_dir / "capture").glob("*/manifest.json")
    ]


def test_native_tls_retains_complete_bodies_opaque_state_auth_and_keepalive(native):
    body = {
        "model": "fixture-native-model",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"name": "fixture", "input_schema": {"type": "object"}}],
        "unknown": {"opaque": "unchanged"},
    }
    raw = json.dumps(body, indent=2).encode()
    with client(native) as http:
        for token in ("fixture-oauth", "fixture-refreshed-oauth"):
            response = http.post(
                "https://api.anthropic.com/v1/messages?beta=true",
                content=raw,
                headers={"Authorization": "Bearer " + token, "anthropic-beta": "fixture-beta"},
            )
            assert response.status_code == 200 and response.content == native.wire
    assert len(native.requests) == 2
    assert native.requests[0][0] == "/v1/messages?beta=true" and native.requests[0][1] == raw
    assert native.requests[0][2]["Authorization"] == "Bearer fixture-oauth"
    assert native.requests[1][2]["Authorization"] == "Bearer fixture-refreshed-oauth"
    assert "x-agency-token" not in {k.lower() for k in native.requests[0][2]}
    result = rows(native, 2)
    assert all(r["type"] == "llm_block" for r in result)
    assert all(
        s in json.dumps(result)
        for s in ("private-thinking", "opaque-signature", "opaque-redacted-data", "provider-state")
    )
    for manifest in manifests(native):
        directory = native.session.run_dir / "capture" / manifest["call_label"]
        assert (directory / "request.bin").read_bytes() == raw
        assert (directory / "response.bin").read_bytes() == native.wire
        for file in directory.iterdir():
            assert b"fixture-oauth" not in file.read_bytes()
            assert b"fixture-refreshed-oauth" not in file.read_bytes()
            assert native.session.host_token.encode() not in file.read_bytes()
    model = trajectory(native.session)
    assert len(actions(model, "model")) == 2 and all(
        a["context"]["available"] for a in actions(model, "model")
    )
    otlp(native.session, "api_request", 1, model="fixture-native-model", duration_ms=1)
    assert len(rows(native, 2)) == 2


def test_native_proxy_retains_gzip_requests_and_auxiliary_count_tokens(native):
    raw = gzip.compress(b'{"model":"fixture-native-model","messages":[],"unknown":true}')
    with client(native) as http:
        assert (
            http.post(
                "https://api.anthropic.com/v1/messages",
                content=raw,
                headers={"content-encoding": "gzip"},
            ).status_code
            == 200
        )
        assert (
            http.post(
                "https://api.anthropic.com/v1/messages/count_tokens",
                json={"model": "fixture-native-model", "messages": []},
            ).status_code
            == 200
        )
    assert native.requests[0][1] == raw
    assert len(rows(native)) == 1 and len(manifests(native)) == 2


def test_native_api_key_and_chunked_request_are_forwarded_without_translation(native):
    raw = b'{"model":"fixture-native-model","messages":[],"future":"untouched"}'
    with client(native) as http:
        assert (
            http.post(
                "https://api.anthropic.com/v1/messages",
                content=iter([raw[:5], raw[5:]]),
                headers={"x-api-key": "fixture-api-key"},
            ).status_code
            == 200
        )
    assert native.requests[0][1] == raw
    assert native.requests[0][2]["x-api-key"] == "fixture-api-key"
    rows(native)  # Wait for the asynchronous manifest replacement to finish.
    for file in (native.session.run_dir / "capture").rglob("*"):
        if file.is_file():
            assert b"fixture-api-key" not in file.read_bytes()


def test_nonmodel_api_authentication_is_forwarded_without_archiving_secrets(native):
    native.wire = b'{"access_token":"fixture-token-response"}'
    with client(native) as http:
        response = http.post(
            "https://api.anthropic.com/oauth/fixture", content=b"fixture-refresh-secret"
        )
    assert response.content == native.wire
    assert len(native.requests) == 1
    assert not manifests(native)


def test_unrelated_connect_stays_opaque_and_preserves_original_proxy_routing(native):
    gateway = native.session._capture_gateway
    gateway.original_env.update(
        HTTPS_PROXY="http://parent:fixture-parent-secret@127.0.0.1:1",
        NO_PROXY="*.invalid example.invalid",
    )
    assert gateway._proxy_for("example.invalid") is None
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    gateway.original_env["HTTPS_PROXY"] = (
        f"http://parent:fixture-parent-secret@127.0.0.1:{listener.getsockname()[1]}"
    )
    seen = []
    payload = b"\x16\x03\x03OPAQUE_TLS_WITH_FIXTURE_SECRET"

    def upstream():
        with listener:
            sock, _ = listener.accept()
            with sock:
                head = b""
                while b"\r\n\r\n" not in head:
                    head += sock.recv(4096)
                seen.append(head)
                sock.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                sock.sendall(sock.recv(4096))

    thread = threading.Thread(target=upstream, daemon=True)
    thread.start()
    with socket.create_connection(("127.0.0.1", gateway.port), timeout=3) as sock:
        authorization = base64.b64encode(("agency:" + native.session.host_token).encode())
        sock.sendall(
            b"CONNECT other.invalid:443 HTTP/1.1\r\nHost: other.invalid:443\r\nProxy-Authorization: Basic "
            + authorization
            + b"\r\n\r\n"
        )
        head = b""
        while b"\r\n\r\n" not in head:
            head += sock.recv(4096)
        assert b"200" in head
        sock.sendall(payload)
        assert sock.recv(4096) == payload
    thread.join(3)
    assert seen and b"other.invalid:443" in seen[0]
    assert base64.b64encode(b"parent:fixture-parent-secret") in seen[0]
    assert native.session.host_token.encode() not in seen[0]
    assert not manifests(native)


def test_existing_ca_is_preserved_but_ephemeral_capture_ca_is_not_upstream_trust(
    native, make_session, monkeypatch
):
    original = native.session._capture_gateway.directory / "ca.pem"
    monkeypatch.setenv("NODE_EXTRA_CA_CERTS", str(original))
    second = make_session(capture=ClaudeNativeCaptureConfig())
    gateway = second._capture_gateway

    def der(path):
        return x509.load_pem_x509_certificate(path.read_bytes()).public_bytes(
            serialization.Encoding.DER
        )

    trusted = ssl.create_default_context(cafile=str(gateway.directory / "trust.pem")).get_ca_certs(
        binary_form=True
    )
    assert der(original) in trusted and der(gateway.directory / "ca.pem") in trusted
    assert der(original) in gateway._tls.get_ca_certs(binary_form=True)
    assert der(gateway.directory / "ca.pem") not in gateway._tls.get_ca_certs(binary_form=True)
    second.close()
    assert original.exists()


def test_settings_file_network_environment_is_retained_without_copying_provider_credentials(
    native, make_session
):
    original = native.session._capture_gateway.directory / "ca.pem"
    settings = {
        "env": {
            "NODE_EXTRA_CA_CERTS": str(original),
            "HTTPS_PROXY": "http://parent.invalid:8181",
            "NO_PROXY": "*",
            "ANTHROPIC_API_KEY": "fixture-settings-api-key",
        }
    }
    second = make_session(
        capture=ClaudeNativeCaptureConfig(), cli_args=["--settings", json.dumps(settings)]
    )
    gateway = second._capture_gateway
    assert gateway.original_env["NODE_EXTRA_CA_CERTS"] == str(original)
    assert gateway.original_env["HTTPS_PROXY"] == "http://parent.invalid:8181"
    assert gateway._proxy_for("api.anthropic.com") is None
    assert "ANTHROPIC_API_KEY" not in gateway.original_env
    assert (
        gateway.original_env["NODE_EXTRA_CA_CERTS"] != second._environment()["NODE_EXTRA_CA_CERTS"]
    )


@pytest.mark.parametrize("sources", ["project,local", ""])
def test_native_network_resolution_respects_disabled_settings_sources(make_session, sources):
    state = Path(os.environ["CLAUDE_CONFIG_DIR"])
    (state / "settings.json").write_text(json.dumps({"env": {"CLAUDE_CODE_USE_BEDROCK": "1"}}))
    session = make_session(
        capture=ClaudeNativeCaptureConfig(), cli_args=["--settings-sources=" + sources]
    )
    assert "--settings-sources=" + sources in session._cli_argv()


@pytest.mark.parametrize("status", [400, 401, 429, 500])
def test_native_provider_errors_are_exact_and_not_retried(native, status):
    native.status = status
    native.wire = b'{"error":{"type":"fixture_error","message":"original error","unknown":true}}'
    native.headers = {"retry-after": "7", "www-authenticate": "Bearer"}
    with client(native) as http:
        response = http.post(
            "https://api.anthropic.com/v1/messages",
            json={"model": "fixture-native-model", "messages": []},
        )
    assert response.status_code == status and response.content == native.wire
    assert response.headers["retry-after"] == "7" and len(native.requests) == 1
    assert rows(native)[0]["type"] == "llm_stream_error"


def test_native_cancellation_closes_silent_upstream_and_retains_partial(native):
    native.hang = True
    native.wire = sse(
        {"type": "message_start", "message": {"id": "partial", "content": []}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "partial"},
        },
    )
    with client(native) as http:
        with http.stream(
            "POST",
            "https://api.anthropic.com/v1/messages",
            json={"model": "fixture-native-model", "messages": []},
        ) as response:
            assert next(response.iter_raw()) == native.wire
    assert native.cancelled.wait(3)
    assert rows(native)[0]["type"] == "llm_stream_cancelled"


def test_native_proxy_requires_local_auth_and_valid_certificate(native):
    with socket.create_connection(
        ("127.0.0.1", native.session._capture_gateway.port), timeout=3
    ) as sock:
        sock.sendall(
            b"CONNECT api.anthropic.com:443 HTTP/1.1\r\nHost: api.anthropic.com:443\r\n\r\n"
        )
        assert b"407" in sock.recv(4096)
    with httpx.Client(proxy=native.proxy, trust_env=False, timeout=3) as http:
        with pytest.raises(httpx.ConnectError):
            http.post("https://api.anthropic.com/v1/messages", json={})
    assert not native.requests


def test_native_ca_is_private_ephemeral_and_not_exported(native):
    gateway = native.session._capture_gateway
    directory = gateway.directory
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in directory.iterdir())
    assert not directory.is_relative_to(native.session.run_dir)
    native.session.close()
    assert not directory.exists()


def test_shutdown_retires_active_tls_connection_and_finalizes_capture(native):
    native.hang = True
    native.wire = sse(
        {"type": "message_start", "message": {"id": "shutdown", "content": []}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "partial"},
        },
    )
    with client(native) as http:
        with http.stream(
            "POST",
            "https://api.anthropic.com/v1/messages",
            json={"model": "fixture-native-model", "messages": []},
        ) as response:
            assert next(response.iter_raw())
            gateway = native.session._capture_gateway
            gateway.close()
            assert not gateway._thread.is_alive()
            assert not gateway.directory.exists()
    assert rows(native)[0]["type"] == "llm_stream_cancelled"


def test_native_child_auth_model_settings_remain_native(native, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fixture-oauth")
    monkeypatch.setenv("ANTHROPIC_MODEL", "fixture-selected-model")
    env = native.session._environment()
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "fixture-oauth"
    assert env["ANTHROPIC_MODEL"] == "fixture-selected-model"
    assert env["ANTHROPIC_BASE_URL"] == "https://api.anthropic.com"
    assert env["HTTPS_PROXY"] == env["https_proxy"] == native.proxy
    assert env["NODE_EXTRA_CA_CERTS"].endswith("trust.pem")
    assert "NODE_TLS_REJECT_UNAUTHORIZED" not in env
    assert native.session.host_token not in " ".join(native.session._cli_argv())


@pytest.mark.parametrize(
    "variable,value",
    [
        ("ANTHROPIC_BASE_URL", "https://other.invalid"),
        ("CLAUDE_CODE_USE_BEDROCK", "1"),
        ("NODE_TLS_REJECT_UNAUTHORIZED", "0"),
    ],
)
def test_native_capture_refuses_provider_changes_and_disabled_tls(
    make_session, monkeypatch, variable, value
):
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match="native|endpoint|verification"):
        make_session(capture=ClaudeNativeCaptureConfig())


def test_native_claude_scope_reexec_keeps_mode_without_litellm(tmp_path, monkeypatch):
    monkeypatch.delenv(session_module._SCOPED_VARIABLE, raising=False)
    monkeypatch.setattr(session_module, "_user_scope_available", lambda: True)

    class Reexec(Exception):
        pass

    invocation = []

    def execvp(binary, args):
        invocation.extend(args)
        raise Reexec

    monkeypatch.setattr(session_module.os, "execvp", execvp)
    args = [
        "--terminal",
        "browser",
        "--cwd",
        str(tmp_path),
        "--claude",
        "/bin/true",
        "--capture-native",
    ]
    with pytest.raises(Reexec):
        session_module.main(ClaudeCodeSession, "claude", args)
    assert invocation[-len(args) :] == args
