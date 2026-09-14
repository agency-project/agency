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


def _grok_payload(event="PreToolUse", tool_use_id="grok-call-1"):
    return {
        "hook_event_name": event,
        "toolName": "run_terminal_command",
        "toolInput": {"command": "printf ok"},
        "toolUseId": tool_use_id,
    }


def test_pretooluse_allow_checks_policy_and_persists_call_id(monkeypatch, capsys, tmp_path):
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        return _Response({"decision": "allow", "reason": "safe", "call_id": "call-abc"})

    monkeypatch.setenv("AGPOLICY_BASE_URL", "http://gateway")
    monkeypatch.setenv("AGPOLICY_TOKEN", "secret-token")
    monkeypatch.setenv("AGPOLICY_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(_payload())))
    monkeypatch.setattr(hook.urllib.request, "urlopen", urlopen)

    assert hook.main() == 0

    assert [request.full_url for request in requests] == ["http://gateway/agpolicy/check_tool"]
    request = requests[0]
    assert request.get_header("Authorization") == "Bearer secret-token"
    body = json.loads(request.data)
    assert body == {"tool_name": "Bash", "tool_input": {"command": "printf ok"}}
    output = json.loads(capsys.readouterr().out)
    assert output["hookSpecificOutput"]["permissionDecision"] == "allow"

    state_file = tmp_path / ".agpolicy_call_tool-1.json"
    assert json.loads(state_file.read_text()) == {"call_id": "call-abc"}


@pytest.mark.parametrize("camel_case", [False, True])
def test_grok_pretooluse_and_posttooluse_use_canonical_policy_and_shared_call_id(
    monkeypatch, capsys, tmp_path, camel_case
):
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        if request.full_url.endswith("/agpolicy/check_tool"):
            return _Response({"decision": "allow", "reason": "safe", "call_id": "call-grok"})
        return _Response({"ok": True})

    monkeypatch.setenv("AGPOLICY_BASE_URL", "http://gateway")
    monkeypatch.setenv("AGPOLICY_TOKEN", "secret-token")
    monkeypatch.setenv("AGPOLICY_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(hook.urllib.request, "urlopen", urlopen)
    pre_payload = _grok_payload()
    if camel_case:
        del pre_payload["hook_event_name"]
        pre_payload["hookEventName"] = "pre_tool_use"
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(pre_payload)))

    assert hook.main() == 0
    assert json.loads(requests[0].data) == {
        "tool_name": "run_terminal_command",
        "tool_input": {"command": "printf ok"},
    }
    state_file = tmp_path / ".agpolicy_call_grok-call-1.json"
    assert json.loads(state_file.read_text()) == {"call_id": "call-grok"}
    assert (
        json.loads(capsys.readouterr().out)["hookSpecificOutput"]["permissionDecision"] == "allow"
    )

    post_payload = _grok_payload("PostToolUse")
    if camel_case:
        del post_payload["hook_event_name"]
        post_payload["hookEventName"] = "post_tool_use"
    post_payload["tool_response"] = {"stdout": "ok"}
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(post_payload)))

    assert hook.main() == 0
    assert [request.full_url for request in requests] == [
        "http://gateway/agpolicy/check_tool",
        "http://gateway/agpolicy/complete_tool",
    ]
    assert json.loads(requests[1].data) == {
        "call_id": "call-grok",
        "result": {"stdout": "ok"},
        "error": None,
    }
    assert not state_file.exists()


def test_denied_pretooluse_persists_no_call_id(monkeypatch, capsys, tmp_path):
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        return _Response({"decision": "deny", "reason": "blocked"})

    monkeypatch.setenv("AGPOLICY_BASE_URL", "http://gateway")
    monkeypatch.setenv("AGPOLICY_TOKEN", "policy-token")
    monkeypatch.setenv("AGPOLICY_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(_payload())))
    monkeypatch.setattr(hook.urllib.request, "urlopen", urlopen)

    assert hook.main() == 0

    assert [request.full_url for request in requests] == ["http://gateway/agpolicy/check_tool"]
    output = json.loads(capsys.readouterr().out)
    assert output["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert list(tmp_path.iterdir()) == []


def test_posttooluse_reads_call_id_and_posts_completion(monkeypatch, capsys, tmp_path):
    (tmp_path / ".agpolicy_call_tool-1.json").write_text(json.dumps({"call_id": "call-abc"}))
    requests = []
    payload = _payload("PostToolUse")
    payload["tool_response"] = {"stdout": "ok"}

    def urlopen(request, timeout):
        requests.append(request)
        return _Response({"ok": True})

    monkeypatch.setenv("AGPOLICY_BASE_URL", "http://gateway")
    monkeypatch.setenv("AGPOLICY_TOKEN", "policy-token")
    monkeypatch.setenv("AGPOLICY_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(hook.urllib.request, "urlopen", urlopen)

    assert hook.main() == 0

    assert [request.full_url for request in requests] == ["http://gateway/agpolicy/complete_tool"]
    posted = json.loads(requests[0].data)
    assert posted == {"call_id": "call-abc", "result": {"stdout": "ok"}, "error": None}
    assert requests[0].get_header("Authorization") == "Bearer policy-token"
    assert capsys.readouterr().out == ""
    # The state file is consumed exactly once.
    assert not (tmp_path / ".agpolicy_call_tool-1.json").exists()


def test_posttoolusefailure_forwards_the_error_field(monkeypatch, tmp_path):
    (tmp_path / ".agpolicy_call_tool-1.json").write_text(json.dumps({"call_id": "call-abc"}))
    requests = []
    payload = _payload("PostToolUseFailure")
    payload["error"] = "boom"

    def urlopen(request, timeout):
        requests.append(request)
        return _Response({"ok": True})

    monkeypatch.setenv("AGPOLICY_BASE_URL", "http://gateway")
    monkeypatch.setenv("AGPOLICY_TOKEN", "policy-token")
    monkeypatch.setenv("AGPOLICY_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(hook.urllib.request, "urlopen", urlopen)

    assert hook.main() == 0
    posted = json.loads(requests[0].data)
    assert posted["error"] == "boom"


def test_posttooluse_with_no_matching_admission_is_a_noop(monkeypatch, tmp_path):
    """No PreToolUse ever admitted this tool_use_id (denied, or the hook
    process never got as far as persisting a call_id) -- completion has
    nothing to report and must not guess at one."""
    requests = []
    payload = _payload("PostToolUse")
    payload["tool_response"] = {"stdout": "ok"}

    monkeypatch.setenv("AGPOLICY_BASE_URL", "http://gateway")
    monkeypatch.setenv("AGPOLICY_TOKEN", "policy-token")
    monkeypatch.setenv("AGPOLICY_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(
        hook.urllib.request,
        "urlopen",
        lambda *a, **kw: requests.append(a) or _Response({"ok": True}),
    )

    assert hook.main() == 0
    assert requests == []


def test_posttooluse_without_state_dir_configured_is_a_noop(monkeypatch):
    requests = []
    payload = _payload("PostToolUse")
    payload["tool_response"] = {"stdout": "ok"}

    monkeypatch.delenv("AGPOLICY_STATE_DIR", raising=False)
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(
        hook.urllib.request,
        "urlopen",
        lambda *a, **kw: requests.append(a) or _Response({"ok": True}),
    )

    assert hook.main() == 0
    assert requests == []


def test_missing_tool_use_id_skips_completion_without_blocking(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("AGPOLICY_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(_payload("PostToolUse", ""))))
    monkeypatch.setattr(
        hook.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not POST")),
    )

    assert hook.main() == 0
    assert "missing tool_use_id" in capsys.readouterr().err


@pytest.fixture
def pretool(monkeypatch, tmp_path):
    monkeypatch.setenv("AGPOLICY_BASE_URL", "http://gateway")
    monkeypatch.setenv("AGPOLICY_TOKEN", "policy-token")
    monkeypatch.setenv("AGPOLICY_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(_payload())))
    return monkeypatch


def _assert_denied(capsys):
    assert hook.main() == 0
    output = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
    assert output["hookEventName"] == "PreToolUse"
    assert output["permissionDecision"] == "deny"
    assert output["permissionDecisionReason"]


@pytest.mark.parametrize("missing", ["AGPOLICY_BASE_URL", "AGPOLICY_TOKEN"])
def test_missing_policy_configuration_denies(pretool, capsys, missing):
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
def test_policy_transport_failure_denies(pretool, capsys, failure):
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
def test_malformed_policy_response_denies(pretool, capsys, response):
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


@pytest.mark.parametrize(
    "payload",
    [
        {"hook_event_name": "PreToolUse", "toolName": 3, "toolInput": {}},
        {"hook_event_name": "PreToolUse", "toolName": "run_terminal_command", "toolInput": []},
        {"hook_event_name": "PreToolUse", "toolInput": {"command": "printf ok"}},
    ],
)
def test_malformed_grok_hook_input_denies_before_request(pretool, capsys, payload):
    pretool.setattr(hook.sys, "stdin", io.StringIO(json.dumps(payload)))
    requests = []
    pretool.setattr(hook.urllib.request, "urlopen", lambda *a, **kw: requests.append(a))
    _assert_denied(capsys)
    assert requests == []


def test_bad_gateway_url_denies_instead_of_crashing(pretool, capsys):
    pretool.setenv("AGPOLICY_BASE_URL", "not-a-url")
    _assert_denied(capsys)


def test_completion_failure_is_swallowed(monkeypatch, capsys, tmp_path):
    (tmp_path / ".agpolicy_call_tool-1.json").write_text(json.dumps({"call_id": "call-abc"}))
    payload = _payload("PostToolUse")
    payload["tool_response"] = {"stdout": "ok"}

    monkeypatch.setenv("AGPOLICY_BASE_URL", "http://gateway")
    monkeypatch.setenv("AGPOLICY_TOKEN", "policy-token")
    monkeypatch.setenv("AGPOLICY_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(
        hook.urllib.request,
        "urlopen",
        lambda *a, **kw: (_ for _ in ()).throw(TimeoutError("gateway down")),
    )

    assert hook.main() == 0
    assert "complete_tool request failed" in capsys.readouterr().err


def _kimi_payload(event="PreToolUse"):
    """Captured from Kimi Code CLI 0.42.0: snake_case like Claude's, but the
    call identity is `tool_call_id` and the result is `tool_output`."""
    return {
        "hook_event_name": event,
        "session_id": "session_abc",
        "client_type": "kimi_code_cli",
        "tool_name": "Read",
        "tool_input": {"path": "hello.txt"},
        "tool_call_id": "toolcall_kimi_1",
    }


def test_kimi_tool_call_id_completes_its_span(monkeypatch, capsys, tmp_path):
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        return _Response({"decision": "allow", "call_id": "call-kimi"})

    monkeypatch.setenv("AGPOLICY_BASE_URL", "http://gateway")
    monkeypatch.setenv("AGPOLICY_TOKEN", "token")
    monkeypatch.setenv("AGPOLICY_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(hook.urllib.request, "urlopen", urlopen)

    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(_kimi_payload())))
    assert hook.main() == 0
    assert (
        json.loads(capsys.readouterr().out)["hookSpecificOutput"]["permissionDecision"] == "allow"
    )

    post = _kimi_payload("PostToolUse")
    post["tool_output"] = "1\tfile contents here"
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(post)))
    assert hook.main() == 0

    assert [request.full_url for request in requests] == [
        "http://gateway/agpolicy/check_tool",
        "http://gateway/agpolicy/complete_tool",
    ]
    assert json.loads(requests[1].data) == {
        "call_id": "call-kimi",
        "result": "1\tfile contents here",
        "error": None,
    }


def test_explicit_null_tool_response_is_not_overridden_by_tool_output(monkeypatch, tmp_path):
    """A harness that reports `tool_response: null` means null, not missing."""
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        return _Response({"decision": "allow", "call_id": "call-x"})

    monkeypatch.setenv("AGPOLICY_BASE_URL", "http://gateway")
    monkeypatch.setenv("AGPOLICY_TOKEN", "token")
    monkeypatch.setenv("AGPOLICY_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(hook.urllib.request, "urlopen", urlopen)

    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(_payload())))
    assert hook.main() == 0

    post = _payload("PostToolUse")
    post["tool_response"] = None
    post["tool_output"] = "ignored"
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(post)))
    assert hook.main() == 0
    assert json.loads(requests[1].data)["result"] is None
