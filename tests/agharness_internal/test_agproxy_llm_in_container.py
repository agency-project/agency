"""Tests for agproxy_llm_in_container -- launching agproxy_llm.py's FastAPI
app as a persistent, in-container process (Phase 2b-ii).

Real, docker-backed only -- like test_native.py, this module's entire
purpose is proving the mechanism works against a genuine container, so
there's no meaningful mocked-only tier. Skipped automatically when
Docker/Podman is unavailable.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import uuid
from unittest.mock import MagicMock, patch

import pytest

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice


def _docker_available() -> bool:
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


docker = pytest.mark.skipif(not _docker_available(), reason="Docker daemon not reachable")


def _make_sandbox(**kwargs):
    from agency.agconfig import agConfig
    from agency.agsandbox import agSandbox
    from agency.agsandbox_backends import agSandboxBackendConfig

    uid = str(uuid.uuid4())
    agconfig = kwargs.pop("agconfig", None)
    cfg = agConfig(agSandboxBackendConfig(backend="docker"), agconfig)
    return agSandbox(uid, agconfig=cfg, **kwargs)


def _completion(content: str) -> ChatCompletion:
    message = ChatCompletionMessage(role="assistant", content=content)
    choice = Choice(index=0, finish_reason="stop", message=message)
    return ChatCompletion(id="x", object="chat.completion", created=0, model="m", choices=[choice])


def test_launcher_uses_profiler_sentinel_when_profiling_is_off():
    from agency.agharness_internal.agproxy_llm_in_container import (
        ensure_agproxy_llm_in_container,
    )

    sandbox = MagicMock()
    terminus = MagicMock()
    terminus.ensure_uds_started.return_value = "/host/agllm-terminus.sock"
    with (
        patch(
            "agency.agharness_internal.agproxy_llm_in_container._is_reachable",
            side_effect=[False, True],
        ),
        patch("agency.agutil.ensure_python_packages_in_container"),
        patch(
            "agency.agharness_internal.agllm_terminus.get_shared_terminus",
            return_value=terminus,
        ),
        patch("agency.profiler.agprof.enabled", return_value=False),
        patch("agency.agharness_internal.agprof_ingest.get_shared_profiler_ingest") as ingest,
    ):
        ensure_agproxy_llm_in_container(sandbox, MagicMock())

    ingest.assert_not_called()
    command = sandbox.exec_detached.call_args.args[0]
    assert "agllm-terminus.sock - 8765" in command


def test_launcher_passes_bind_mounted_profiler_uds_when_profiling_is_on():
    from agency.agharness_internal.agproxy_llm_in_container import (
        ensure_agproxy_llm_in_container,
    )

    sandbox = MagicMock()
    terminus = MagicMock()
    terminus.ensure_uds_started.return_value = "/host/agllm-terminus.sock"
    ingest = MagicMock()
    ingest.ensure_uds_started.return_value = "/host/agprof-ingest-abc.sock"
    with (
        patch(
            "agency.agharness_internal.agproxy_llm_in_container._is_reachable",
            side_effect=[False, True],
        ),
        patch("agency.agutil.ensure_python_packages_in_container"),
        patch(
            "agency.agharness_internal.agllm_terminus.get_shared_terminus",
            return_value=terminus,
        ),
        patch("agency.profiler.agprof.enabled", return_value=True),
        patch(
            "agency.agharness_internal.agprof_ingest.get_shared_profiler_ingest",
            return_value=ingest,
        ),
    ):
        ensure_agproxy_llm_in_container(sandbox, MagicMock())

    command = sandbox.exec_detached.call_args.args[0]
    assert "/var/run/agency_llm_gateway/agprof-ingest-abc.sock 8765" in command


def test_entrypoint_maps_profiler_sentinel_to_none():
    from agency.agharness_internal import _agproxy_llm_in_container_entrypoint as entrypoint

    proxy = MagicMock()
    with (
        patch("agency.agharness_internal.agproxy_llm.agProxyLLM", return_value=proxy) as proxy_cls,
        patch.object(entrypoint.threading.Event, "wait", return_value=None),
    ):
        entrypoint.main(["/bridge/terminus.sock", "-", "8765"])

    assert proxy_cls.call_args.kwargs["terminus_uds_path"] == "/bridge/terminus.sock"
    assert proxy_cls.call_args.kwargs["profiler_uds_path"] is None


def test_profiled_run_rejects_already_running_unprofiled_proxy():
    from agency.agharness_internal.agproxy_llm_in_container import (
        ensure_agproxy_llm_in_container,
    )

    with (
        patch(
            "agency.agharness_internal.agproxy_llm_in_container._is_reachable",
            return_value=True,
        ),
        patch(
            "agency.agharness_internal.agproxy_llm_in_container._profiler_bridge_configured",
            return_value=False,
        ),
        patch("agency.profiler.agprof.enabled", return_value=True),
    ):
        with pytest.raises(RuntimeError, match="started without a profiler UDS bridge"):
            ensure_agproxy_llm_in_container(MagicMock(), MagicMock())


def _curl_from_container(sandbox, base_url: str, token: str, body: dict) -> "tuple[dict, int]":
    """Exercise the in-container agproxy_llm exactly the way a harness
    binary launched in the same container would -- an HTTP request
    originating from INSIDE the container, not from the host process
    (which can't reach this port directly at all -- see
    agproxy_llm_in_container.py's module docstring)."""
    # body_json is embedded via !r (a valid Python string literal) and used
    # as-is -- embedding json.dumps(body)'s OUTPUT directly as if it were
    # Python dict-literal syntax (an earlier version of this helper) is
    # broken: JSON's `false`/`true`/`null` aren't valid Python literals.
    body_json = json.dumps(body)
    script = (
        "import urllib.request as u, json, sys\n"
        f"req = u.Request({base_url!r} + '/v1/chat/completions', "
        f"data={body_json!r}.encode(), method='POST', "
        f"headers={{'Authorization': 'Bearer {token}', 'Content-Type': 'application/json'}})\n"
        "resp = u.urlopen(req, timeout=10)\n"
        "print(resp.status)\n"
        "print(resp.read().decode())\n"
    )
    cmd = f"python3 -c {shlex.quote(script)}"
    out, rc = sandbox.exec(cmd, timeout=15)
    assert rc == 0, f"in-container curl script failed: {out}{_proxy_log_tail(sandbox)}"
    lines = out.strip().split("\n")
    status = int(lines[0])
    payload = json.loads(lines[1])
    return payload, status


def _proxy_log_tail(sandbox) -> str:
    """The in-container proxy's own log, for failure messages.

    `ensure_agproxy_llm_in_container()` already redirects the detached
    process's stdout/stderr to this file precisely because a fire-and-forget
    launch has nowhere else to report -- but only its own readiness-timeout
    path ever reads it back. A request that fails *after* a successful launch
    (any 5xx: the route raised, so uvicorn logged a traceback here and
    returned a body with no detail in it) otherwise reports only the status
    code, which names neither the failing step nor the reason. Since every
    request begins with a token-validation POST across the bridge to the
    host-side terminus, "500 on every token, valid or not" and "the bridge is
    broken" look identical from outside -- this is what tells them apart.
    """
    log_path = "/tmp/.agproxy_llm_in_container.log"
    try:
        tail, _ = sandbox.exec(f"tail -c 4000 {shlex.quote(log_path)} 2>/dev/null", timeout=15)
    except Exception as exc:  # the log is a diagnostic, never the assertion
        return f"\n[could not read {log_path}: {type(exc).__name__}: {exc}]"
    return f"\n--- in-container proxy log ({log_path}) ---\n{tail}"


@docker
class TestEnsureAgproxyLlmInContainer:
    """One shared sandbox + launch across all tests in this class -- the
    `pip install fastapi uvicorn openai` cost (plus their own dependency
    trees) is real and shouldn't be paid once per test method. The
    idempotent-launch test still gets genuine coverage: launching once in
    setup_class and pinging it again in the test itself is exactly the
    same "already running" code path a fresh sandbox would exercise."""

    @classmethod
    def setup_class(cls):
        from agency.agconfig import agConfig
        from agency.agharness_internal.agproxy_llm_in_container import (
            ensure_agproxy_llm_in_container,
        )

        cls.cfg = agConfig()
        cls.sb = _make_sandbox(agconfig=cls.cfg)
        cls.base_url = ensure_agproxy_llm_in_container(cls.sb, cls.cfg, timeout_s=180)

    @classmethod
    def teardown_class(cls):
        cls.sb.destroy()

    def test_launches_and_is_idempotent(self):
        import time

        from agency.agharness_internal.agproxy_llm_in_container import (
            ensure_agproxy_llm_in_container,
        )

        assert self.base_url == "http://127.0.0.1:8765"

        # A second call (server already up from setup_class) must be fast --
        # it must NOT attempt to relaunch or reinstall dependencies.
        start = time.monotonic()
        base_url_2 = ensure_agproxy_llm_in_container(self.sb, self.cfg, timeout_s=120)
        elapsed = time.monotonic() - start
        assert base_url_2 == self.base_url
        assert elapsed < 5, f"idempotent call took {elapsed:.2f}s -- looks like it relaunched"

    def test_real_dispatch_round_trip_from_inside_the_container(self):
        """The full path: a request originating INSIDE the container
        (exactly like a harness binary would issue), through the
        in-container agproxy_llm, to the host-side agllm_terminus, to a
        (mocked) backend client -- proving the whole bridge, not just that
        the process is up."""
        from agency.agharness_internal.agllm_terminus import get_shared_terminus

        token = uuid.uuid4().hex
        terminus = get_shared_terminus(self.cfg)
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _completion(
            "hello from in-container proxy"
        )
        fake_ag = MagicMock()
        fake_ag.llm.backend.make_client.return_value = fake_client
        terminus.register(token, fake_ag)

        try:
            body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False}
            payload, status = _curl_from_container(self.sb, self.base_url, token, body)
            assert status == 200
            assert payload["choices"][0]["message"]["content"] == "hello from in-container proxy"
        finally:
            terminus.unregister(token)

    def test_unknown_token_rejected_from_inside_the_container(self):
        script = (
            "import urllib.request as u, urllib.error as e, json\n"
            f"req = u.Request({self.base_url!r} + '/v1/chat/completions', "
            "data=json.dumps({'model': 'm', 'messages': []}).encode(), method='POST', "
            "headers={'Authorization': 'Bearer totally-unknown-token'})\n"
            "try:\n"
            "    u.urlopen(req, timeout=10)\n"
            "    print('NO ERROR RAISED')\n"
            "except e.HTTPError as err:\n"
            "    print(err.code)\n"
        )
        cmd = f"python3 -c {shlex.quote(script)}"
        out, rc = self.sb.exec(cmd, timeout=15)
        assert rc == 0, f"{out}{_proxy_log_tail(self.sb)}"
        assert out.strip() == "401", f"{out}{_proxy_log_tail(self.sb)}"
