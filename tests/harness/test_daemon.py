from __future__ import annotations

import base64
import threading
import uuid
from pathlib import Path

import pytest

from agency.configs.agconfig import agconfig
from agency.engine.clients import HarnessInteractionClient
from agency.harness import daemon
from agency.harness.adapters.base import (
    AdapterRuntime,
    AttemptResult,
    HarnessAdapter,
)
from agency.harness.daemon import HarnessManager
from agency.harness.protocol import HarnessAttemptRequest, HarnessAttemptResult, PromptPayload


def test_render_attempt_prompt_includes_system_instruction_for_a_fresh_session():
    request = HarnessAttemptRequest(
        prompt=PromptPayload("system prompt", "user turn"),
        harness="fake",
    )
    assert daemon._render_attempt_prompt(request) == "system prompt\n\nuser turn"


def test_render_attempt_prompt_drops_system_instruction_when_resuming_a_session():
    # A resumed harness session already has the system instruction from the
    # attempt that established it -- retyping it here would duplicate it in
    # the harness's own transcript. Covers both a schema retry (which resumes
    # the attempt it's retrying) and any later turn on a persistent agent.
    request = HarnessAttemptRequest(
        prompt=PromptPayload("system prompt", "user turn"),
        harness="fake",
        resume_session_id="session-1",
    )
    assert daemon._render_attempt_prompt(request) == "user turn"


def test_render_attempt_prompt_keeps_output_instruction_when_resuming():
    request = HarnessAttemptRequest(
        prompt=PromptPayload("system prompt", "user turn", "output instruction"),
        harness="fake",
        resume_session_id="session-1",
    )
    assert daemon._render_attempt_prompt(request) == "user turn\n\noutput instruction"


def test_render_attempt_prompt_includes_output_instruction_for_non_tandem_harness():
    request = HarnessAttemptRequest(
        prompt=PromptPayload("system prompt", "user turn", "output instruction"),
        harness="fake",
    )
    assert (
        daemon._render_attempt_prompt(request) == "system prompt\n\nuser turn\n\noutput instruction"
    )


def test_render_attempt_prompt_excludes_output_instruction_for_tandem():
    # tandem routes output_instruction to the worker directly (see
    # _run_adapter_attempt) instead of folding it into the supervisor's
    # own task prompt -- the supervisor never calls submit_output itself.
    request = HarnessAttemptRequest(
        prompt=PromptPayload("system prompt", "user turn", "output instruction"),
        harness="tandem",
    )
    assert daemon._render_attempt_prompt(request) == "system prompt\n\nuser turn"


class _FakePingClient:
    """Records every POST made through it in the shared `posted` list --
    stands in for httpx.Client so _ping_daemon_lifecycle's tests don't need
    a real UDS listener."""

    def __init__(self, posted, *, raise_on_post=False, **_kwargs):
        self._posted = posted
        self._raise_on_post = raise_on_post

    def post(self, path, json):
        if self._raise_on_post:
            raise ConnectionError("daemon-side socket already gone")
        self._posted.append((path, json))

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def test_ping_daemon_lifecycle_posts_a_started_event(monkeypatch):
    posted = []
    monkeypatch.setattr(daemon.httpx, "Client", lambda **kw: _FakePingClient(posted, **kw))
    daemon._ping_daemon_lifecycle("/tmp/fake.sock", "harness_daemon_started", "agent-1")

    assert len(posted) == 1
    path, body = posted[0]
    assert path == "/record_event"
    assert body["type"] == "harness_daemon_started"
    assert body["payload"]["engine"] == "agent-1"
    assert "reason" not in body["payload"]
    assert body["term_message"] == "[agent-1] DAEMON started"


def test_ping_daemon_lifecycle_posts_an_exiting_event_with_its_reason(monkeypatch):
    posted = []
    monkeypatch.setattr(daemon.httpx, "Client", lambda **kw: _FakePingClient(posted, **kw))
    daemon._ping_daemon_lifecycle(
        "/tmp/fake.sock", "harness_daemon_exiting", "agent-1", reason="signal 15"
    )

    assert len(posted) == 1
    _path, body = posted[0]
    assert body["type"] == "harness_daemon_exiting"
    assert body["payload"]["reason"] == "signal 15"
    assert body["term_message"] == "[agent-1] DAEMON exiting: signal 15"


def test_ping_daemon_lifecycle_never_raises_when_the_post_fails(monkeypatch):
    monkeypatch.setattr(
        daemon.httpx, "Client", lambda **kw: _FakePingClient([], raise_on_post=True, **kw)
    )
    daemon._ping_daemon_lifecycle("/tmp/fake.sock", "harness_daemon_started", "agent-1")


@pytest.mark.parametrize(
    "event_type,label",
    [
        ("harness_daemon_setting_up_env", "setting up env"),
        ("harness_daemon_starting", "starting"),
        ("harness_daemon_dispatching_attempt", "dispatching attempt"),
        ("harness_daemon_attempt_completed", "attempt completed"),
        ("harness_daemon_signal_received", "signal received"),
    ],
)
def test_ping_daemon_lifecycle_derives_a_readable_label_from_any_event_type(
    monkeypatch, event_type, label
):
    posted = []
    monkeypatch.setattr(daemon.httpx, "Client", lambda **kw: _FakePingClient(posted, **kw))
    daemon._ping_daemon_lifecycle("/tmp/fake.sock", event_type, "agent-1")

    assert len(posted) == 1
    _path, body = posted[0]
    assert body["type"] == event_type
    assert body["term_message"] == f"[agent-1] DAEMON {label}"


def test_ping_daemon_lifecycle_passes_extra_fields_into_the_payload_only(monkeypatch):
    posted = []
    monkeypatch.setattr(daemon.httpx, "Client", lambda **kw: _FakePingClient(posted, **kw))
    daemon._ping_daemon_lifecycle(
        "/tmp/fake.sock",
        "harness_daemon_attempt_completed",
        "agent-1",
        request_id="req-1",
        ok=False,
        error="boom",
    )

    assert len(posted) == 1
    _path, body = posted[0]
    assert body["payload"]["request_id"] == "req-1"
    assert body["payload"]["ok"] is False
    assert body["payload"]["error"] == "boom"
    # Extra fields don't leak into the printed line -- only `reason` does.
    assert body["term_message"] == "[agent-1] DAEMON attempt completed"


@pytest.mark.parametrize("sandbox_payload", [None, "serialized-tools"])
def test_adapter_session_blob_crosses_daemon_protocol(monkeypatch, sandbox_payload):
    seen = {}

    class FakeAdapter(HarnessAdapter):
        engine_key = "fake"

        def run_daemon_attempt(self, runtime, **kwargs):
            seen["runtime"] = runtime
            seen.update(kwargs)
            return AttemptResult(
                ok=True,
                final_text="done",
                session_id="session-2",
                session_blob=b"updated session state",
            )

    monkeypatch.setattr(
        HarnessAdapter,
        "for_config",
        classmethod(lambda cls, name, config: FakeAdapter(config)),
    )
    request = HarnessAttemptRequest(
        prompt=PromptPayload("system", "user"),
        harness="fake",
        max_steps=7,
        resume_session_id="session-1",
        prior_session_blob_b64=base64.b64encode(b"prior session state").decode("ascii"),
        attempt_token="attempt-one",
        sandbox_mcp_tools_b64=sandbox_payload,
    )

    result = daemon._run_adapter_attempt(
        request,
        agconfig(),
        "http://127.0.0.1:8766",
        "model",
        "agent-1",
        object(),
        lambda handle: None,
        lambda handler: None,
    )

    assert seen["resume_session_id"] == "session-1"
    assert seen["prior_session_blob"] == b"prior session state"
    assert seen["max_steps"] == 7
    assert isinstance(seen["runtime"], AdapterRuntime)
    assert seen["runtime"].engine_name == "agent-1"
    assert seen["runtime"].model == "model"
    assert seen["runtime"].token == "attempt-one"
    assert seen["runtime"].has_sandbox_mcp_tools is (sandbox_payload is not None)
    assert result.session_id == "session-2"
    assert base64.b64decode(result.session_blob_b64) == b"updated session state"


@pytest.mark.parametrize(
    ("harness", "expect_kwarg"),
    [("tandem", True), ("fake", False)],
)
def test_run_adapter_attempt_passes_output_instruction_only_for_tandem(
    monkeypatch, harness, expect_kwarg
):
    seen = {}

    class FakeAdapter(HarnessAdapter):
        engine_key = harness

        def run_daemon_attempt(self, runtime, **kwargs):
            seen.update(kwargs)
            return AttemptResult(ok=True, final_text="done")

    monkeypatch.setattr(
        HarnessAdapter,
        "for_config",
        classmethod(lambda cls, name, config: FakeAdapter(config)),
    )
    request = HarnessAttemptRequest(
        prompt=PromptPayload("system", "user", "output instruction"),
        harness=harness,
        attempt_token="attempt-one",
    )

    daemon._run_adapter_attempt(
        request,
        agconfig(),
        "http://127.0.0.1:8766",
        "model",
        "agent-1",
        object(),
        lambda handle: None,
        lambda handler: None,
    )

    assert ("output_instruction" in seen) is expect_kwarg
    if expect_kwarg:
        assert seen["output_instruction"] == "output instruction"


def test_daemon_dispatch_selects_adapter_from_request(monkeypatch):
    request = HarnessAttemptRequest(
        prompt=PromptPayload("system", "user"),
        harness="claude_code",
        max_steps=4,
        attempt_token="attempt-one",
    )
    expected = HarnessAttemptResult(ok=True, final_text="done")
    seen = []
    manager = HarnessManager.__new__(HarnessManager)
    manager._agconfig = agconfig()
    manager._harness = "claude_code"
    manager._bootstrapped = True
    manager._engine_name = "agent-1"
    manager._attempt_lock = threading.Lock()
    manager._current_attempt_token = None
    manager._control_lock = threading.Lock()
    manager._current_control_handle = None
    manager._agent_paused = False
    manager._persistent = False
    manager._live_control_handle = None
    policy = object()

    class HarnessApi:
        base_url = "http://127.0.0.1:8766"

        def __init__(self):
            self.events = []

        def register_attempt_token(self, token):
            self.events.append(("register", token))

        def clear_attempt_token(self, token):
            self.events.append(("clear", token))
            return True

        def resolve_model(self, token):
            self.events.append(("resolve", token))
            return "model"

        def syscall_policy(self, token, **_kwargs):
            self.events.append(("policy", token))
            return policy

    manager._harness_api = HarnessApi()
    manager._attempt_handler = manager._run_adapter_request

    def run_adapter(
        got_request,
        config,
        base_url,
        model,
        engine_name,
        syscall_policy,
        register_control_handle,
        register_redirect,
    ):
        seen.append((got_request, config, base_url, model, engine_name, syscall_policy))
        return expected

    monkeypatch.setattr(daemon, "_run_adapter_attempt", run_adapter)

    assert manager._dispatch_attempt(request) is expected
    assert seen == [
        (
            request,
            manager._agconfig,
            "http://127.0.0.1:8766",
            "model",
            "agent-1",
            policy,
        )
    ]
    assert manager._harness_api.events == [
        ("register", "attempt-one"),
        ("resolve", "attempt-one"),
        ("policy", "attempt-one"),
        ("clear", "attempt-one"),
    ]
    assert manager._current_attempt_token is None


def test_daemon_dispatch_pings_dispatching_then_completed(monkeypatch):
    request = HarnessAttemptRequest(
        prompt=PromptPayload("system", "user"),
        harness="claude_code",
        request_id="req-1",
        attempt_token="attempt-one",
    )
    expected = HarnessAttemptResult(ok=True, final_text="done")
    manager = HarnessManager.__new__(HarnessManager)
    manager._agconfig = agconfig()
    manager._harness = "claude_code"
    manager._bootstrapped = True
    manager._engine_name = "agent-1"
    manager._host_uds_path = "/tmp/fake-host.sock"
    manager._attempt_lock = threading.Lock()
    manager._current_attempt_token = None
    manager._control_lock = threading.Lock()
    manager._current_control_handle = None
    manager._agent_paused = False
    manager._persistent = False
    manager._live_control_handle = None
    manager._harness_api = type(
        "HarnessApi",
        (),
        {
            "register_attempt_token": lambda self, token: None,
            "clear_attempt_token": lambda self, token: True,
        },
    )()
    manager._attempt_handler = lambda _request: expected

    pings = []
    monkeypatch.setattr(
        daemon,
        "_ping_daemon_lifecycle",
        lambda uds, event_type, engine, **extra: pings.append((uds, event_type, engine, extra)),
    )

    assert manager._dispatch_attempt(request) is expected
    assert [p[1] for p in pings] == [
        "harness_daemon_dispatching_attempt",
        "harness_daemon_attempt_completed",
    ]
    for uds, _event_type, engine, _extra in pings:
        assert uds == "/tmp/fake-host.sock"
        assert engine == "agent-1"
    assert pings[0][3] == {"request_id": "req-1"}
    assert pings[1][3] == {"request_id": "req-1", "ok": True, "error": None}


def test_daemon_dispatch_does_not_ping_completed_when_the_handler_raises(monkeypatch):
    request = HarnessAttemptRequest(
        prompt=PromptPayload("system", "user"),
        harness="claude_code",
        attempt_token="attempt-one",
    )
    manager = HarnessManager.__new__(HarnessManager)
    manager._agconfig = agconfig()
    manager._harness = "claude_code"
    manager._engine_name = "agent-1"
    manager._host_uds_path = "/tmp/fake-host.sock"
    manager._attempt_lock = threading.Lock()
    manager._current_attempt_token = None
    manager._control_lock = threading.Lock()
    manager._current_control_handle = None
    manager._agent_paused = False
    manager._persistent = False
    manager._live_control_handle = None
    manager._harness_api = type(
        "HarnessApi",
        (),
        {
            "register_attempt_token": lambda self, token: None,
            "clear_attempt_token": lambda self, token: True,
        },
    )()

    def fail(_request):
        raise RuntimeError("boom")

    manager._attempt_handler = fail

    pings = []
    monkeypatch.setattr(
        daemon,
        "_ping_daemon_lifecycle",
        lambda uds, event_type, engine, **extra: pings.append(event_type),
    )

    with pytest.raises(RuntimeError, match="boom"):
        manager._dispatch_attempt(request)

    assert pings == ["harness_daemon_dispatching_attempt"]


def test_daemon_rejects_missing_attempt_token_without_registering():
    manager = HarnessManager.__new__(HarnessManager)
    manager._attempt_lock = threading.Lock()
    manager._current_attempt_token = None
    manager._control_lock = threading.Lock()
    manager._current_control_handle = None
    manager._agent_paused = False
    manager._persistent = False
    manager._live_control_handle = None
    manager._harness_api = type(
        "HarnessApi",
        (),
        {"register_attempt_token": lambda self, token: (_ for _ in ()).throw(AssertionError())},
    )()
    manager._attempt_handler = lambda _request: (_ for _ in ()).throw(AssertionError())

    result = manager._dispatch_attempt(
        HarnessAttemptRequest(
            prompt=PromptPayload("system", "user"),
            harness="claude_code",
        )
    )

    assert result.ok is False
    assert result.error_message == "missing harness attempt token"


def test_daemon_revokes_attempt_token_when_handler_raises():
    events = []
    manager = HarnessManager.__new__(HarnessManager)
    manager._attempt_lock = threading.Lock()
    manager._current_attempt_token = None
    manager._control_lock = threading.Lock()
    manager._current_control_handle = None
    manager._agent_paused = False
    manager._persistent = False
    manager._live_control_handle = None
    manager._harness_api = type(
        "HarnessApi",
        (),
        {
            "register_attempt_token": lambda self, token: events.append(("register", token)),
            "clear_attempt_token": lambda self, token: events.append(("clear", token)),
        },
    )()

    def fail(_request):
        raise RuntimeError("adapter failed")

    manager._attempt_handler = fail
    request = HarnessAttemptRequest(
        prompt=PromptPayload("system", "user"),
        harness="claude_code",
        attempt_token="attempt-one",
    )

    try:
        manager._dispatch_attempt(request)
    except RuntimeError as exc:
        assert str(exc) == "adapter failed"
    else:
        raise AssertionError("handler failure was swallowed")

    assert events == [("register", "attempt-one"), ("clear", "attempt-one")]
    assert manager._current_attempt_token is None


def _adapter_request_manager(*, bootstrapped: bool):
    """A HarnessManager stripped to just what _run_adapter_request() reads,
    with a _harness_api fake that fails loudly if reached before bootstrap
    should have already happened."""
    manager = HarnessManager.__new__(HarnessManager)
    manager._agconfig = agconfig()
    manager._harness = "codex"
    manager._bootstrapped = bootstrapped
    manager._current_attempt_token = "attempt-one"
    manager._live_local_token = None
    manager._persistent = False
    manager._engine_name = "agent-1"

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("must not run before bootstrap completes")

    manager._harness_api = type(
        "HarnessApi",
        (),
        {
            "base_url": "http://127.0.0.1:8766",
            "resolve_model": (lambda self, token: "model") if bootstrapped else _unexpected,
            "syscall_policy": (lambda self, token, **kw: object()) if bootstrapped else _unexpected,
        },
    )()
    return manager


def test_run_adapter_request_reports_bootstrap_failure_without_running_the_harness(monkeypatch):
    manager = _adapter_request_manager(bootstrapped=False)
    monkeypatch.setattr(
        daemon,
        "prepare_harness_executable_local",
        lambda harness, config: (_ for _ in ()).throw(RuntimeError("codex executable missing")),
    )
    monkeypatch.setattr(
        daemon,
        "_run_adapter_attempt",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not run the harness")),
    )
    request = HarnessAttemptRequest(
        prompt=PromptPayload("system", "user"), harness="codex", attempt_token="attempt-one"
    )

    result = manager._run_adapter_request(request)

    assert result.ok is False
    assert "codex executable missing" in result.error_message
    # Not sticky on failure -- a transient bootstrap problem (e.g. pip
    # install briefly lacking network) should be retried on the next attempt.
    assert manager._bootstrapped is False


def test_run_adapter_request_skips_bootstrap_once_already_done(monkeypatch):
    manager = _adapter_request_manager(bootstrapped=True)
    calls = []
    monkeypatch.setattr(
        daemon,
        "prepare_harness_executable_local",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("bootstrap must not re-run")),
    )
    expected = HarnessAttemptResult(ok=True, final_text="done")
    monkeypatch.setattr(
        daemon, "_run_adapter_attempt", lambda *a, **kw: calls.append(a) or expected
    )
    request = HarnessAttemptRequest(
        prompt=PromptPayload("system", "user"), harness="codex", attempt_token="attempt-one"
    )

    result = manager._run_adapter_request(request)

    assert result == expected
    assert len(calls) == 1


def test_harness_manager_start_pings_starting_before_bringing_up_servers(monkeypatch):
    manager = HarnessManager.__new__(HarnessManager)
    manager._engine_name = "agent-1"
    manager._host_uds_path = "/tmp/fake-host.sock"
    calls = []
    manager._harness_api = type("HarnessApi", (), {"start": lambda self: calls.append("api")})()
    manager._interaction_server = type(
        "InteractionServer", (), {"start": lambda self: calls.append("interaction") or "started"}
    )()

    monkeypatch.setattr(
        daemon,
        "_ping_daemon_lifecycle",
        lambda uds, event_type, engine, **extra: calls.append(("ping", event_type, uds, engine)),
    )

    assert manager.start() == "started"
    assert calls == [
        ("ping", "harness_daemon_starting", "/tmp/fake-host.sock", "agent-1"),
        "api",
        "interaction",
    ]


def test_harness_manager_returns_attempt_result_on_original_rpc():
    socket_dir = Path(f"/tmp/agency-daemon-{uuid.uuid4().hex[:8]}")
    socket_dir.mkdir()
    socket_path = socket_dir / "sandbox.sock"
    request = HarnessAttemptRequest(
        prompt=PromptPayload("system", "user", "output"),
        harness="claude_code",
        max_steps=8,
        attempt_token="attempt-one",
    )
    expected = HarnessAttemptResult(
        ok=True,
        final_text="mock daemon result",
        input_tokens=5,
        output_tokens=2,
        session_id="session-1",
        session_blob_b64="c2Vzc2lvbiBzdGF0ZQ==",
    )
    seen = []
    manager = HarnessManager(
        str(socket_path),
        str(socket_dir / "host.sock"),
        "agent-1",
        "claude_code",
        attempt_handler=lambda got_request: seen.append(got_request) or expected,
        harness_api_port=0,
    )

    try:
        manager.start()
        with HarnessInteractionClient(str(socket_path), timeout_s=2.0) as client:
            assert client.is_ready()
            result = client.run_harness_attempt(request)
    finally:
        manager.stop()
        socket_path.unlink(missing_ok=True)
        socket_dir.rmdir()

    assert seen == [request]
    assert result == expected


def test_host_syscall_policy_check_forwards_hooked_syscalls_to_host_services():
    from types import SimpleNamespace

    from agency.harness.daemon import _HostSyscallPolicy

    class _FakeHostServices:
        def __init__(self):
            self.check_calls = []
            self.complete_calls = []

        def check_syscall_policy(self, token, syscall):
            self.check_calls.append((token, syscall))
            return (True, None, "call-1")

        def complete_syscall_policy(self, token, call_id, return_value):
            self.complete_calls.append((token, call_id, return_value))

    host_services = _FakeHostServices()
    policy = _HostSyscallPolicy(
        host_services, "attempt-token", hooked_syscalls=frozenset({"openat"})
    )
    event = SimpleNamespace(syscall="openat")

    decision = policy.check(None, event)
    assert decision == (True, None, "call-1")
    assert host_services.check_calls == [("attempt-token", event)]

    policy.check_completion(None, "call-1", 3)
    assert host_services.complete_calls == [("attempt-token", "call-1", 3)]


def test_host_syscall_policy_check_completion_is_a_noop_without_a_call_id():
    from agency.harness.daemon import _HostSyscallPolicy

    class _FakeHostServices:
        def __init__(self):
            self.complete_calls = []

        def complete_syscall_policy(self, token, call_id, return_value):
            self.complete_calls.append((token, call_id, return_value))

    host_services = _FakeHostServices()
    policy = _HostSyscallPolicy(host_services, "attempt-token")

    policy.check_completion(None, None, 3)
    assert host_services.complete_calls == []


@pytest.mark.parametrize("completion", [False, True])
@pytest.mark.parametrize("retired", [False, True])
def test_syscall_racing_attempt_retirement_preserves_the_tracer(monkeypatch, completion, retired):
    from agency.harness._syscall_event import agsyscallevent
    from agency.harness.clients.host_services_client import HostServicesClient
    from agency.harness.daemon import _HostSyscallPolicy

    bridge = HostServicesClient("/unused/host.sock")
    bridge.register_attempt_token("attempt-token")
    policy = _HostSyscallPolicy(bridge, "attempt-token")

    def racing_request(*args, **kwargs):
        if retired:
            bridge.clear_attempt_token("attempt-token")
        raise RuntimeError("request failed")

    monkeypatch.setattr(bridge.client, "post", racing_request)
    event = agsyscallevent(
        syscall="execve", pid=1, tid=1, argv=[], envp={}, path="/bin/true", timestamp=0
    )
    invoke = (
        (lambda: policy.check_completion(None, "admitted-call", 0))
        if completion
        else (lambda: policy.check(None, event))
    )
    try:
        if retired:
            result = invoke()
            assert (
                result is None
                if completion
                else result == (False, "inactive harness attempt", None, None)
            )
        else:
            with pytest.raises(RuntimeError, match="request failed"):
                invoke()
    finally:
        bridge.close()


def test_pause_finishes_before_a_concurrent_redirect_can_start():
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    manager = HarnessManager("/unused/sandbox", "/unused/host", "agent", "claude_code")
    pause_entered, release_pause = threading.Event(), threading.Event()
    redirect_requested, redirect_entered = threading.Event(), threading.Event()
    order = []

    def pause():
        pause_entered.set()
        assert release_pause.wait(3)
        order.append("paused")

    def deliver(message):
        redirect_entered.set()
        order.append("redirect")
        return True

    def redirect():
        redirect_requested.set()
        return manager.redirect("run", "message")

    manager._current_control_handle = SimpleNamespace(pause=pause)
    manager._current_request_id = "run"
    manager._register_redirect(deliver)
    with ThreadPoolExecutor(max_workers=2) as pool:
        paused = pool.submit(manager.control, "pause")
        assert pause_entered.wait(2)
        redirected = pool.submit(redirect)
        try:
            assert redirect_requested.wait(2)
            assert not redirect_entered.wait(0.1)
        finally:
            release_pause.set()
        paused.result(timeout=2)
        assert redirected.result(timeout=2)
    assert order == ["paused", "redirect"]


def test_host_syscall_policy_short_circuits_unhooked_syscalls_by_default():
    """No hook for this syscall -- the decision must be `not default_to_deny`,
    computed locally, without ever *waiting* on the host RPC (proven here by
    blocking that RPC until after the assertion on `decision` already ran).
    The same admission RPC still fires in the background for logging, and
    its result must never gate the syscall."""
    import threading
    from types import SimpleNamespace

    from agency.harness.daemon import _HostSyscallPolicy

    entered_rpc = threading.Event()
    release_rpc = threading.Event()
    completed = threading.Event()

    class _FakeHostServices:
        def __init__(self):
            self.check_calls = []
            self.complete_calls = []

        def check_syscall_policy(self, token, syscall):
            entered_rpc.set()
            assert release_rpc.wait(timeout=5), "test never released the blocked RPC"
            self.check_calls.append((token, syscall))
            return (True, None, "call-1")

        def complete_syscall_policy(self, token, call_id, return_value):
            self.complete_calls.append((token, call_id, return_value))
            completed.set()

    host_services = _FakeHostServices()
    policy = _HostSyscallPolicy(
        host_services, "attempt-token", hooked_syscalls=frozenset({"openat"})
    )
    # Not "execve"/"execveat" -- those are always forced onto the
    # synchronous path (see _HostSyscallPolicy._ALWAYS_SYNCHRONOUS), so
    # they'd defeat the point of this short-circuit test.
    event = SimpleNamespace(syscall="open")

    decision = policy.check(None, event)
    # check() already returned even though the background RPC is still
    # blocked (or hasn't even started) -- proves it never waited on it.
    assert decision == (True, None, None, None)

    release_rpc.set()
    assert entered_rpc.wait(timeout=5)
    assert completed.wait(timeout=5)
    assert host_services.check_calls == [("attempt-token", event)]
    assert host_services.complete_calls == [("attempt-token", "call-1", None)]


def test_host_syscall_policy_short_circuits_to_deny_when_default_to_deny_set():
    from types import SimpleNamespace

    from agency.harness.daemon import _HostSyscallPolicy

    class _FakeHostServices:
        def check_syscall_policy(self, token, syscall):
            raise AssertionError("must not synchronously reach the host")

        def complete_syscall_policy(self, token, call_id, return_value):
            pass

    policy = _HostSyscallPolicy(_FakeHostServices(), "attempt-token", default_to_deny=True)
    event = SimpleNamespace(syscall="open")

    assert policy.check(None, event) == (False, None, None, None)


def test_host_syscall_policy_logging_failure_never_raises():
    import threading
    from types import SimpleNamespace

    from agency.harness.daemon import _HostSyscallPolicy

    attempted = threading.Event()

    class _FakeHostServices:
        def check_syscall_policy(self, token, syscall):
            attempted.set()
            raise RuntimeError("host unreachable")

    policy = _HostSyscallPolicy(_FakeHostServices(), "attempt-token")
    event = SimpleNamespace(syscall="open")

    assert policy.check(None, event) == (True, None, None, None)
    assert attempted.wait(timeout=5)
