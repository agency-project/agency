"""Native OpenAI authentication, original upstream routing and WS wire capture.

All credentials and model responses here are offline fixtures, never live calls.
"""

import json
import base64
import os
import signal
import sqlite3
import subprocess
import threading
import time
from contextlib import contextmanager

import httpx
import pytest
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.sync.client import connect
from websockets.sync.server import serve

from agency.observability.agdatalogger import read_llm_exchanges, reconstruct_llm_messages
from agency.sessions import capture as module
from agency.sessions import session as session_module
from agency.sessions.capture import NativeCaptureConfig
from agency.sessions.codex import CodexSession, _native_capture_config

from .helpers import FIXTURES, actions, eventually, otlp, trajectory
from .test_capture import Stream


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("LITE_LLM_KEY", raising=False)


@pytest.fixture
def native(make_session, monkeypatch):
    original = httpx.AsyncClient
    requests = []

    def make(handler):
        async def respond(request):
            requests.append(request)
            result = handler(request)
            if result.is_stream_consumed:
                result = httpx.Response(
                    result.status_code, headers=result.headers, stream=Stream(result.content)
                )
            return result

        monkeypatch.setattr(
            module.httpx,
            "AsyncClient",
            lambda **kw: original(transport=httpx.MockTransport(respond), **kw),
        )
        return make_session(
            session_class=CodexSession,
            binary=FIXTURES / "fake_codex_session.py",
            capture=NativeCaptureConfig(),
        ), requests

    return make


def headers(session, *, subscription=True, token="fixture-native-bearer"):
    result = {"x-agency-token": session.host_token, "Authorization": f"Bearer {token}"}
    if subscription:
        result["ChatGPT-Account-Id"] = "fixture-account"
    return result


def manifests(session):
    return [
        json.loads(p.read_text()) for p in (session.run_dir / "capture").glob("*/manifest.json")
    ]


def rows(session, count=1):
    def check():
        session._agent_logger.flush()
        result = read_llm_exchanges(session._agent_logger.db_path)
        assert len(result) == count
        return result

    return eventually(check)


def assert_no_credentials(session):
    for path in (session.run_dir / "capture").rglob("*"):
        if path.is_file():
            assert b"fixture-native-bearer" not in path.read_bytes()
            assert b"fixture-refreshed-bearer" not in path.read_bytes()
            assert session.host_token.encode() not in path.read_bytes()


@pytest.mark.parametrize("subscription", [True, False])
def test_native_http_preserves_auth_body_model_headers_and_original_route(native, subscription):
    response = {
        "id": "native-response",
        "status": "completed",
        "model": "fixture-selected-model",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "native answer"}],
            }
        ],
    }
    session, requests = native(lambda req: httpx.Response(200, json=response))
    body = {
        "model": "fixture-selected-model",
        "input": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "name": "fixture", "parameters": {"type": "object"}}],
        "unknown": {"opaque": "retained"},
    }
    raw = json.dumps(body, indent=2).encode()
    incoming = headers(session, subscription=subscription)
    incoming.update(
        {
            "OpenAI-Beta": "fixture-beta",
            "Accept-Encoding": "gzip",
            "Content-Type": "application/json",
        }
    )
    result = httpx.post(
        f"http://127.0.0.1:{session.port}/capture/v1/responses?beta=fixture",
        content=raw,
        headers=incoming,
    )
    assert result.status_code == 200 and result.json() == response
    [request] = requests
    expected = (
        "https://chatgpt.com/backend-api/codex/responses"
        if subscription
        else "https://api.openai.com/v1/responses"
    )
    assert str(request.url) == expected + "?beta=fixture"
    assert request.content == raw
    assert request.headers["Authorization"] == incoming["Authorization"]
    assert request.headers["OpenAI-Beta"] == "fixture-beta"
    assert request.headers["Accept-Encoding"] == "gzip"
    assert "x-agency-token" not in request.headers
    row = rows(session)[0]
    assert next(b for b in row["response"] if b.get("type") == "metadata")["model"] == body["model"]
    with sqlite3.connect(session._agent_logger.db_path) as db:
        assert reconstruct_llm_messages(db, row["call_label"])[0]
    assert actions(trajectory(session), "model")[0]["context"]["available"]
    assert_no_credentials(session)


def test_native_401_is_not_retried_and_refreshed_cli_token_is_forwarded(native):
    def reply(request):
        if request.headers["Authorization"] == "Bearer fixture-native-bearer":
            return httpx.Response(
                401,
                json={"error": {"message": "expired fixture"}},
                headers={"www-authenticate": "Bearer"},
            )
        return httpx.Response(200, json={"id": "refreshed", "status": "completed", "output": []})

    session, requests = native(reply)
    url = f"http://127.0.0.1:{session.port}/capture/v1/responses"
    body = {"model": "gpt-6-luna", "input": "x"}
    failed = httpx.post(url, json=body, headers=headers(session))
    assert failed.status_code == 401 and failed.headers["www-authenticate"] == "Bearer"
    assert len(requests) == 1
    assert rows(session)[0]["type"] == "llm_stream_error"
    assert (
        httpx.post(
            url, json=body, headers=headers(session, token="fixture-refreshed-bearer")
        ).status_code
        == 200
    )
    assert len(rows(session, 2)) == 2 and len(requests) == 2
    assert requests[-1].headers["Authorization"] == "Bearer fixture-refreshed-bearer"
    assert_no_credentials(session)


def test_native_models_compaction_and_server_references(native):
    session, requests = native(
        lambda req: httpx.Response(
            200,
            json={
                "id": "compact",
                "output": [{"type": "compaction", "encrypted_content": "opaque-state"}],
            },
        )
    )
    url = f"http://127.0.0.1:{session.port}/capture/v1"
    assert (
        httpx.get(url + "/models?client_version=fixture", headers=headers(session)).status_code
        == 200
    )
    body = {
        "model": "gpt-6-luna",
        "previous_response_id": "missing-parent",
        "input": [{"type": "compaction", "encrypted_content": "opaque-state"}],
    }
    assert (
        httpx.post(url + "/responses/compact", json=body, headers=headers(session)).status_code
        == 200
    )
    assert requests[-1].url.path == "/backend-api/codex/responses/compact"
    assert requests[-1].content == json.dumps(body, separators=(",", ":")).encode()
    row = rows(session)[0]
    capture = next(b for b in row["response"] if b.get("type") == "metadata")["capture"]
    assert capture["compaction"] and capture["server_state"] and not capture["complete"]
    assert len(manifests(session)) == 2


def test_native_local_token_is_separate_and_cannot_select_a_destination(native):
    session, requests = native(lambda req: httpx.Response(200, json={}))
    url = f"http://127.0.0.1:{session.port}/capture"
    for incoming in [
        {"Authorization": "Bearer " + session.host_token},
        {"Authorization": "Bearer fixture-native-bearer"},
    ]:
        assert httpx.post(url + "/v1/responses", headers=incoming, json={}).status_code == 401
    assert httpx.post(url + "/v1/messages", headers=headers(session), json={}).status_code == 404
    assert not requests


def test_native_child_retains_auth_environment_and_cli_model_selection(make_session, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-native-bearer")
    session = make_session(
        session_class=CodexSession,
        capture=NativeCaptureConfig(),
        cli_args=["--model", "fixture-selected-model"],
    )
    env, argv = session._environment(), session._cli_argv()
    assert env["OPENAI_API_KEY"] == "fixture-native-bearer"
    assert "requires_openai_auth=true" in " ".join(argv)
    assert "supports_websockets=true" in " ".join(argv)
    assert "env_http_headers" in " ".join(argv)
    assert "env_key=" not in " ".join(argv)
    assert "fixture-native-bearer" not in " ".join(argv) and session.host_token not in " ".join(
        argv
    )
    assert argv[argv.index("--model") + 1] == "fixture-selected-model"
    assert not any(arg.startswith("model=") for arg in argv)


@contextmanager
def upstream_ws(handler, **kwargs):
    with serve(handler, "127.0.0.1", 0, **kwargs) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.socket.getsockname()[1]}"
        finally:
            server.shutdown()
            thread.join(3)


def ws_session(make_session, base):
    return make_session(
        session_class=CodexSession,
        binary=FIXTURES / "fake_codex_session.py",
        capture=NativeCaptureConfig(api_base_url=base, chatgpt_base_url=base),
    )


def test_native_websocket_multiple_turns_raw_text_binary_context_and_otel_dedup(make_session):
    received, sent = [], []

    def provider(connection):
        assert connection.request.headers["Authorization"] == "Bearer fixture-native-bearer"
        assert connection.request.headers["ChatGPT-Account-Id"] == "fixture-account"
        assert "x-agency-token" not in connection.request.headers
        for turn in (1, 2):
            received.append(connection.recv())
            frames = [
                json.dumps(
                    {
                        "type": "response.created",
                        "response": {"id": f"r{turn}", "status": "in_progress"},
                    }
                ),
                b'{"type":"provider.future","opaque":"untouched"}',
                json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "id": f"r{turn}",
                            "status": "completed",
                            "output": [
                                {"type": "reasoning", "encrypted_content": "opaque-reasoning"},
                                {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [{"type": "output_text", "text": f"answer {turn}"}],
                                },
                            ],
                        },
                    }
                ),
            ]
            sent.append(frames)
            for frame in frames:
                connection.send(frame)

    with upstream_ws(provider) as base:
        session = ws_session(make_session, base)
        requests = []
        with connect(
            f"ws://127.0.0.1:{session.port}/capture/v1/responses",
            additional_headers=headers(session),
            proxy=None,
        ) as client:
            for turn in (1, 2):
                body = {
                    "type": "response.create",
                    "model": "gpt-6-luna",
                    "input": [{"role": "user", "content": f"turn {turn}"}],
                }
                if turn == 2:
                    body["previous_response_id"] = "r1"
                raw = json.dumps(body, indent=2)
                requests.append(raw)
                client.send(raw)
                assert [client.recv() for _ in range(3)] == sent[turn - 1]
        assert received == requests
        result = rows(session, 2)
        assert all(r["type"] == "llm_block" for r in result)
        saved = sorted(manifests(session), key=lambda m: m["started_at"])
        assert saved[0]["coverage"]["complete"] and not saved[1]["coverage"]["complete"]
        for index, manifest in enumerate(saved):
            directory = session.run_dir / "capture" / manifest["call_label"]
            assert (directory / "request.bin").read_bytes() == requests[index].encode()
            raw = (directory / "response.bin").read_bytes()
            restored = [
                raw[f["offset"] : f["offset"] + f["length"]] for f in manifest["response_frames"]
            ]
            assert restored == [d.encode() if isinstance(d, str) else d for d in sent[index]]
            assert [f["binary"] for f in manifest["response_frames"]] == [False, True, False]
        assert len(actions(trajectory(session), "model")) == 2
        otlp(session, "codex.sse_event", time.time(), **{"event.kind": "response.completed"})
        assert len(rows(session, 2)) == 2
        assert_no_credentials(session)


def test_native_websocket_interruption_retains_partial_and_closes_silent_provider(make_session):
    closed = threading.Event()

    def provider(connection):
        connection.recv()
        connection.send('{"type":"response.output_text.delta","delta":"partial response"}')
        for _ in connection:
            pass
        closed.set()

    with upstream_ws(provider) as base:
        session = ws_session(make_session, base)
        with connect(
            f"ws://127.0.0.1:{session.port}/capture/v1/responses",
            additional_headers=headers(session),
            proxy=None,
        ) as client:
            client.send('{"type":"response.create","model":"gpt-6-luna","input":"cancel"}')
            assert "partial response" in client.recv()
        assert closed.wait(5)
        row = rows(session)[0]
        assert row["type"] == "llm_stream_cancelled"
        assert any(b.get("text") == "partial response" for b in row["response"])


def test_native_websocket_401_preserved_for_cli_refresh_and_local_auth_checked(make_session):
    seen = []

    def reject(connection, request):
        seen.append(request.headers.get("Authorization"))
        return connection.respond(401, "fixture expired")

    with upstream_ws(lambda c: None, process_request=reject) as base:
        session = ws_session(make_session, base)
        url = f"ws://127.0.0.1:{session.port}/capture/v1/responses"
        with pytest.raises(InvalidStatus) as exc:
            connect(
                url,
                additional_headers={"Authorization": "Bearer fixture-native-bearer"},
                proxy=None,
            )
        assert exc.value.response.status_code == 401 and not seen
        for token in ("fixture-native-bearer", "fixture-refreshed-bearer"):
            with pytest.raises(InvalidStatus) as exc:
                connect(url, additional_headers=headers(session, token=token), proxy=None)
            assert exc.value.response.status_code == 401
        assert seen == ["Bearer fixture-native-bearer", "Bearer fixture-refreshed-bearer"]
        assert len(manifests(session)) == 2
        assert_no_credentials(session)


def test_native_websocket_preserves_handshake_state_and_large_payload(make_session):
    text = "x" * (2**20 + 1)

    def handshake(connection, request, response):
        response.headers["x-codex-turn-state"] = "opaque-handshake-state"
        return response

    def provider(connection):
        connection.recv()
        connection.send(
            json.dumps(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "large",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": text}],
                            }
                        ],
                    },
                }
            )
        )

    with upstream_ws(provider, process_response=handshake) as base:
        session = ws_session(make_session, base)
        with connect(
            f"ws://127.0.0.1:{session.port}/capture/v1/responses",
            additional_headers=headers(session),
            proxy=None,
            max_size=None,
        ) as client:
            assert client.response.headers["x-codex-turn-state"] == "opaque-handshake-state"
            client.send('{"type":"response.create","model":"gpt-6-luna","input":"large"}')
            assert text in client.recv()
        assert any(b.get("text") == text for b in rows(session)[0]["response"])
        [manifest] = manifests(session)
        directory = session.run_dir / "capture" / manifest["call_label"]
        index = [
            json.loads(line)
            for line in (directory / "response.frames.jsonl").read_text().splitlines()
        ]
        assert index == manifest["response_frames"]


@pytest.mark.parametrize("code", [1000, 1011])
def test_native_websocket_provider_close_mid_response_is_error_not_user_cancel(make_session, code):
    def provider(connection):
        connection.recv()
        connection.send('{"type":"response.output_text.delta","delta":"partial"}')
        connection.close(code=code)

    with upstream_ws(provider) as base:
        session = ws_session(make_session, base)
        with connect(
            f"ws://127.0.0.1:{session.port}/capture/v1/responses",
            additional_headers=headers(session),
            proxy=None,
        ) as client:
            client.send('{"type":"response.create","model":"gpt-6-luna","input":"fail"}')
            assert "partial" in client.recv()
            with pytest.raises(ConnectionClosed):
                client.recv()
        assert rows(session)[0]["type"] == "llm_stream_error"
        assert not manifests(session)[0]["cancelled"]


def test_native_websocket_prewarm_does_not_count_as_model_generation(make_session):
    def provider(connection):
        for label in ("prewarm", "model"):
            connection.recv()
            connection.send(
                json.dumps(
                    {
                        "type": "response.completed",
                        "response": {"id": label, "status": "completed", "output": []},
                    }
                )
            )

    with upstream_ws(provider) as base:
        session = ws_session(make_session, base)
        with connect(
            f"ws://127.0.0.1:{session.port}/capture/v1/responses",
            additional_headers=headers(session),
            proxy=None,
        ) as client:
            for generate in (False, True):
                client.send(
                    json.dumps(
                        {
                            "type": "response.create",
                            "model": "gpt-6-luna",
                            "generate": generate,
                            "input": [],
                        }
                    )
                )
                client.recv()
        assert len(rows(session)) == 1
        assert len(manifests(session)) == 2


def test_native_capture_reports_invalid_ca_bundle_without_disabling_verification(monkeypatch):
    monkeypatch.setenv("CODEX_CA_CERTIFICATE", "/missing/fixture-ca.pem")
    with pytest.raises(ValueError, match="CA bundle"):
        module.CaptureGateway(None, NativeCaptureConfig())


def test_native_websocket_interrupt_cancels_one_exchange_and_reuses_connection(make_session):
    def provider(connection):
        connection.recv()
        connection.send(
            '{"type":"response.created","response":{"id":"cancel-me","status":"in_progress"}}'
        )
        connection.send(
            '{"type":"response.output_text.delta","response_id":"cancel-me","delta":"partial"}'
        )
        assert json.loads(connection.recv())["type"] == "response.interrupt"
        connection.send('{"type":"response.cancelled","response_id":"cancel-me"}')
        connection.recv()
        connection.send(
            '{"type":"response.completed","response":{"id":"next","status":"completed","output":[]}}'
        )

    with upstream_ws(provider) as base:
        session = ws_session(make_session, base)
        with connect(
            f"ws://127.0.0.1:{session.port}/capture/v1/responses",
            additional_headers=headers(session),
            proxy=None,
        ) as client:
            request = '{"type":"response.create","model":"gpt-6-luna","input":"x"}'
            client.send(request)
            client.recv()
            client.recv()
            client.send(
                '{"type":"response.interrupt","response_id":"cancel-me","mode":"discard_partial_items"}'
            )
            assert json.loads(client.recv())["type"] == "response.cancelled"
            client.send(request)
            assert json.loads(client.recv())["type"] == "response.completed"
        result = rows(session, 2)
        assert [r["type"] for r in result] == ["llm_stream_cancelled", "llm_block"]
        assert len(manifests(session)) == 3
        cancelled = [m for m in manifests(session) if m["cancelled"]]
        assert len(cancelled) == 1 and cancelled[0]["error"] is None


@pytest.mark.parametrize(
    "args",
    [
        ["-c", 'model_provider="other"'],
        ["--config=model_provider=other"],
        ["-cmodel_provider=other"],
        ["--oss"],
        ["--local-provider=ollama"],
    ],
)
def test_native_capture_refuses_silently_switching_other_providers(args):
    with pytest.raises(ValueError, match="OpenAI|openai"):
        _native_capture_config(NativeCaptureConfig(), args)


def test_native_capture_resolves_profile_and_explicit_base_without_reading_auth(
    tmp_path, monkeypatch
):
    home = tmp_path / "codex-home"
    home.mkdir()
    (home / "auth.json").write_text("not json: capture must not open this")
    (home / "config.toml").write_text(
        '[profiles.local]\nopenai_base_url="http://127.0.0.1:8123/v1"\n'
    )
    config = _native_capture_config(NativeCaptureConfig(), ["--profile=local"])
    assert config.api_base_url == config.chatgpt_base_url == "http://127.0.0.1:8123/v1"
    (home / "blocked.config.toml").write_text('model_provider="other"')
    with pytest.raises(ValueError, match="openai"):
        _native_capture_config(NativeCaptureConfig(), ["-pblocked"])


@pytest.mark.parametrize(
    "url",
    [
        "https://user:secret@example.com/v1",
        "http://external.invalid/v1",
        "https://example.com/v1?token=secret",
        123,
    ],
)
def test_native_upstream_validation(url):
    with pytest.raises(ValueError):
        NativeCaptureConfig(api_base_url=url)


def test_native_scope_reexec_keeps_mode_without_litellm_key(tmp_path, monkeypatch):
    monkeypatch.delenv(session_module._SCOPED_VARIABLE, raising=False)
    monkeypatch.setattr(session_module, "_user_scope_available", lambda: True)
    invocation = []

    class Reexec(Exception):
        pass

    def execvp(binary, args):
        invocation.extend(args)
        raise Reexec

    monkeypatch.setattr(session_module.os, "execvp", execvp)
    args = [
        "--terminal",
        "browser",
        "--cwd",
        str(tmp_path),
        "--codex",
        "/bin/true",
        "--capture-native",
    ]
    with pytest.raises(Reexec):
        session_module.main(CodexSession, "codex", args)
    assert invocation[-len(args) :] == args


@pytest.mark.parametrize(
    "extra",
    [["--capture", "--capture-native"], ["--capture-native", "--capture-model", "gpt-6-luna"]],
)
def test_native_capture_rejects_mixed_modes(extra, tmp_path):
    with pytest.raises(SystemExit) as exc:
        session_module.main(
            CodexSession,
            "codex",
            ["--terminal", "browser", "--cwd", str(tmp_path), "--codex", "/bin/true", *extra],
        )
    assert exc.value.code == 2


def test_real_codex_native_account_capture(make_session, monkeypatch, tmp_path):
    """Opt-in real auth owner, synthetic account, no actual subscription call."""
    from .test_capture_interactive import NativeUpstream

    binary = os.environ.get("AGENCY_CAPTURE_TEST_CODEX")
    if not binary:
        pytest.skip("set AGENCY_CAPTURE_TEST_CODEX for the offline real CLI probe")
    home = tmp_path / "codex-home"
    home.mkdir(mode=0o700)

    def encoded(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    token = (
        encoded({"alg": "RS256", "typ": "JWT"})
        + "."
        + encoded(
            {
                "email": "fixture@example.invalid",
                "exp": int(time.time()) + 3600,
                "https://api.openai.com/auth": {
                    "chatgpt_account_id": "fixture-account",
                    "chatgpt_plan_type": "plus",
                },
            }
        )
        + ".fixture"
    )
    auth = {
        "auth_mode": "chatgpt",
        "OPENAI_API_KEY": None,
        "tokens": {
            "id_token": token,
            "access_token": "fixture-native-bearer",
            "refresh_token": "fixture-refresh",
            "account_id": "fixture-account",
        },
        "last_refresh": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with os.fdopen(
        os.open(home / "auth.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
    ) as file:
        json.dump(auth, file)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    upstream = NativeUpstream("responses", "gpt-6-luna")
    upstream.enable_websocket()
    try:
        session = make_session(
            session_class=CodexSession,
            binary=binary,
            capture=NativeCaptureConfig(api_base_url=upstream.url, chatgpt_base_url=upstream.url),
            cli_args=[
                "-a",
                "never",
                "--dangerously-bypass-hook-trust",
                "-c",
                'cli_auth_credentials_store="file"',
                "-c",
                "features.apps=false",
                "-c",
                "features.plugins=false",
                "-c",
                "features.remote_plugin=false",
                "-c",
                "analytics.enabled=false",
                "exec",
                "--skip-git-repo-check",
                "--json",
                "--sandbox",
                "danger-full-access",
                "--model",
                "gpt-6-luna",
                "CAPTURE_FIRST_PROMPT: Run printf CAPTURE_TOOL_1 then reply CAPTURE_TURN_1_DONE.",
            ],
        )
        with (
            (tmp_path / "stdout.txt").open("w") as stdout,
            (tmp_path / "stderr.txt").open("w") as stderr,
        ):
            process = subprocess.Popen(
                session._cli_argv(),
                env=session._environment(),
                cwd=tmp_path,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                # This probe verifies native request construction and capture.
                # Offline account/plugin services can delay CLI shutdown, so
                # retire the fixture after its completed turn rather than
                # pretending to validate online account-service lifecycle.
                deadline = time.monotonic() + 45
                while '"type":"turn.completed"' not in (tmp_path / "stdout.txt").read_text():
                    assert process.poll() is None, (tmp_path / "stderr.txt").read_text()
                    assert time.monotonic() < deadline, "native account turn did not complete"
                    time.sleep(0.1)
            finally:
                # Helpers may retain inherited stdout after the CLI exits.
                # Retire this fixture's own process group, never user daemons.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        assert "CAPTURE_TURN_1_DONE" in (tmp_path / "stdout.txt").read_text()
        assert upstream.native_auth_seen and all(
            all(fields) for fields in upstream.native_auth_seen
        )
        assert len(rows(session, 2)) == 2
        assert_no_credentials(session)
    finally:
        upstream.close()
