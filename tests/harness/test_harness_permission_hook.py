from __future__ import annotations

import io
import json

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
