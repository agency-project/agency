from __future__ import annotations

import io
import json
import urllib.error

import pytest

from agency.harness import _harness_permission_hook as hook


class _Response:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self._body


def _payload(event="PreToolUse", tool_use_id="tool-1"):
    return {
        "hook_event_name": event,
        "tool_use_id": tool_use_id,
        "tool_name": "Bash",
        "tool_input": {"command": "printf ok"},
    }


def test_pretooluse_allow_checks_policy_then_posts_profiler_header_only(monkeypatch, capsys):
    requests = []
    timeouts = []

    def urlopen(request, timeout):
        requests.append(request)
        timeouts.append(timeout)
        if request.full_url.endswith("/agpolicy/check_tool"):
            return _Response({"decision": "allow", "reason": "safe"})
        return _Response({"ok": True})

    monkeypatch.setenv("AGPOLICY_BASE_URL", "http://gateway")
    monkeypatch.setenv("AGPOLICY_TOKEN", "secret-token")
    monkeypatch.setenv("AGPROF_BASE_URL", "http://gateway")
    monkeypatch.setenv("AGPROF_TOKEN", "secret-token")
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(_payload())))
    monkeypatch.setattr(hook.urllib.request, "urlopen", urlopen)

    assert hook.main() == 0

    assert [request.full_url for request in requests] == [
        "http://gateway/agpolicy/check_tool",
        "http://gateway/agprof/hook",
    ]
    assert timeouts == [hook._POLICY_TIMEOUT_S, hook._PROFILER_TIMEOUT_S]
    profiler_request = requests[1]
    profiler_body = json.loads(profiler_request.data)
    assert profiler_request.get_header("Authorization") == "Bearer secret-token"
    assert "token" not in profiler_body
    assert "token" not in profiler_body["payload"]
    assert profiler_body["hook_event_name"] == "PreToolUse"
    assert profiler_body["payload"]["tool_use_id"] == "tool-1"
    output = json.loads(capsys.readouterr().out)
    assert output["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_denied_pretooluse_does_not_open_profiler_span(monkeypatch, capsys):
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        return _Response({"decision": "deny", "reason": "blocked"})

    monkeypatch.setenv("AGPOLICY_BASE_URL", "http://gateway")
    monkeypatch.setenv("AGPOLICY_TOKEN", "policy-token")
    monkeypatch.setenv("AGPROF_BASE_URL", "http://gateway")
    monkeypatch.setenv("AGPROF_TOKEN", "profiler-token")
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(_payload())))
    monkeypatch.setattr(hook.urllib.request, "urlopen", urlopen)

    assert hook.main() == 0

    assert [request.full_url for request in requests] == ["http://gateway/agpolicy/check_tool"]
    output = json.loads(capsys.readouterr().out)
    assert output["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_posttooluse_only_posts_profiler_and_emits_no_hook_response(monkeypatch, capsys):
    requests = []
    payload = _payload("PostToolUse")
    payload["tool_response"] = {"stdout": "ok"}
    payload["duration_ms"] = 12.5

    def urlopen(request, timeout):
        requests.append(request)
        return _Response({"ok": True})

    monkeypatch.setenv("AGPROF_BASE_URL", "http://gateway")
    monkeypatch.setenv("AGPROF_TOKEN", "profiler-token")
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(hook.urllib.request, "urlopen", urlopen)

    assert hook.main() == 0

    assert [request.full_url for request in requests] == ["http://gateway/agprof/hook"]
    posted_payload = json.loads(requests[0].data)["payload"]
    assert posted_payload["tool_response"] == {"stdout": "ok"}
    assert posted_payload["duration_ms"] == 12.5
    assert capsys.readouterr().out == ""


def test_missing_tool_use_id_skips_profiler_without_blocking(monkeypatch, capsys):
    monkeypatch.setenv("AGPROF_BASE_URL", "http://gateway")
    monkeypatch.setenv("AGPROF_TOKEN", "profiler-token")
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(_payload("PostToolUse", ""))))
    monkeypatch.setattr(
        hook.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not POST")),
    )

    assert hook.main() == 0
    assert "missing tool_use_id" in capsys.readouterr().err


def test_profiler_timeout_budget_stays_at_or_below_200ms_per_tool():
    # Pre and Post each make at most one synchronous telemetry request.
    assert hook._PROFILER_TIMEOUT_S * 2 <= 0.2


@pytest.fixture
def pretool(monkeypatch):
    monkeypatch.setenv("AGPOLICY_BASE_URL", "http://gateway")
    monkeypatch.setenv("AGPOLICY_TOKEN", "policy-token")
    monkeypatch.setenv("AGPROF_BASE_URL", "http://gateway")
    monkeypatch.setenv("AGPROF_TOKEN", "profiler-token")
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(_payload())))
    return monkeypatch


def _assert_denied(capsys):
    assert hook.main() == 0
    output = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
    assert output["hookEventName"] == "PreToolUse"
    assert output["permissionDecision"] == "deny"
    assert output["permissionDecisionReason"]


@pytest.mark.parametrize("missing", ["AGPOLICY_BASE_URL", "AGPOLICY_TOKEN"])
def test_missing_policy_configuration_denies_without_profiler(pretool, capsys, missing):
    pretool.delenv(missing)
    requests = []
    pretool.setattr(hook.urllib.request, "urlopen", lambda *a, **kw: requests.append(a))
    _assert_denied(capsys)
    assert requests == []


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("timeout"),
        urllib.error.URLError("connection refused"),
        urllib.error.HTTPError("http://gateway", 401, "unauthorized", {}, None),
        urllib.error.HTTPError("http://gateway", 503, "unavailable", {}, None),
    ],
)
def test_policy_transport_failure_denies_without_profiler(pretool, capsys, failure):
    requests = []

    def urlopen(request, timeout):
        requests.append(request.full_url)
        raise failure

    pretool.setattr(hook.urllib.request, "urlopen", urlopen)
    _assert_denied(capsys)
    assert requests == ["http://gateway/agpolicy/check_tool"]


@pytest.mark.parametrize(
    "response",
    [
        None,
        [],
        {},
        {"decision": None},
        {"decision": "rewrite"},
        {"decision": "ask"},
        {"decision": True},
        {"decision": "allow", "reason": {}},
        "invalid-json",
    ],
)
def test_malformed_policy_response_denies_without_profiler(pretool, capsys, response):
    requests = []

    def urlopen(request, timeout):
        requests.append(request.full_url)
        result = _Response(response)
        if response == "invalid-json":
            result._body = b"not JSON"
        return result

    pretool.setattr(hook.urllib.request, "urlopen", urlopen)
    _assert_denied(capsys)
    assert requests == ["http://gateway/agpolicy/check_tool"]


@pytest.mark.parametrize(
    "payload",
    ["not JSON", "null", "[]", "{}", '{"tool_name": 3}', '{"tool_name": "Bash", "tool_input": []}'],
)
def test_malformed_hook_input_denies_before_request(pretool, capsys, payload):
    pretool.setattr(hook.sys, "stdin", io.StringIO(payload))
    requests = []
    pretool.setattr(hook.urllib.request, "urlopen", lambda *a, **kw: requests.append(a))
    _assert_denied(capsys)
    assert requests == []


def test_bad_gateway_url_denies_instead_of_crashing(pretool, capsys):
    pretool.setenv("AGPOLICY_BASE_URL", "not-a-url")
    _assert_denied(capsys)


def test_profiler_failure_does_not_revoke_explicit_admission(pretool, capsys):
    def urlopen(request, timeout):
        if request.full_url.endswith("/agpolicy/check_tool"):
            return _Response({"decision": "allow"})
        raise TimeoutError("profiler down")

    pretool.setattr(hook.urllib.request, "urlopen", urlopen)
    assert hook.main() == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert "profiler request failed" in output.err


def test_pending_redirect_denies_claude_tool_until_acknowledged(pretool, capsys):
    from types import SimpleNamespace

    from agency._agent_control import AgentControl
    from agency.agpolicy import agpolicy
    from agency.engine.host_servers.host_interaction_server import HostInteractionServer

    invocation = AgentControl().begin_invocation("claude")
    invocation.redirect("reconsider this action")
    server = HostInteractionServer(
        SimpleNamespace(policy=agpolicy(default_to_deny=False)),
        None,
        invocation=invocation,
    )
    requests = []

    def urlopen(request, timeout):
        requests.append(request.full_url)
        if request.full_url.endswith("/agprof/hook"):
            return _Response({"ok": True})
        payload = json.loads(request.data)
        allowed, reason = server.check_tool(payload["tool_name"], payload["tool_input"])
        return _Response({"decision": "allow" if allowed else "deny", "reason": reason})

    pretool.setattr(hook.urllib.request, "urlopen", urlopen)
    _assert_denied(capsys)
    snapshot = invocation._checkpoint("model", allow_messages=True, phase="model")
    pretool.setattr(hook.sys, "stdin", io.StringIO(json.dumps(_payload())))
    _assert_denied(capsys)
    assert requests == ["http://gateway/agpolicy/check_tool"] * 2
    invocation._acknowledge_redirects(snapshot.invocation_messages)
    pretool.setattr(hook.sys, "stdin", io.StringIO(json.dumps(_payload())))
    assert hook.main() == 0
    assert (
        json.loads(capsys.readouterr().out)["hookSpecificOutput"]["permissionDecision"] == "allow"
    )
