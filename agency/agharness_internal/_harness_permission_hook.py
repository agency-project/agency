"""Standalone PreToolUse hook bridging a harness's own native permission
check to `agpolicy` over HTTP.

Deliberately self-contained -- stdlib only (`json`, `os`, `sys`,
`urllib.request`), ZERO imports from the `agency` package -- because it
runs as a subprocess of the harness binary itself (e.g. `claude` invoking
its own `PreToolUse` hook), not inside this process, and can't reach into
this process's Python state directly. `claude_code.py` writes this file
into the launch's own `config_home` and registers it via `--settings`'
`hooks.PreToolUse` block; `AGPOLICY_BASE_URL`/`AGPOLICY_TOKEN` (set on the
harness's own env, the same values ANTHROPIC_BASE_URL's bearer token
already uses) tell it where to reach `agproxy_llm.py`'s
`/agpolicy/check_tool` route -- the one thing this subprocess and that
route share is the per-run bearer token, so that's the auth.

Reads a harness's PreToolUse hook payload on stdin (Claude Code's shape:
`{"tool_name": ..., "tool_input": ..., ...}`), POSTs `{tool_name,
tool_input}` to `/agpolicy/check_tool`, and translates the returned
`agdecision` kind into Claude Code's own hook output shape. Fails open
(allow) on any network/parse error -- a transient gateway hiccup blocking
every tool call in every harness-driven agent would be worse than the
(already-default) allow-all policy this replaces; the failure is still
visible on stderr for debugging.
"""
import json
import os
import sys
import urllib.request

_TIMEOUT_S = 5.0


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        print(f"[agpolicy hook] could not parse hook input: {exc}", file=sys.stderr)
        _emit("allow", "hook input was not valid JSON, failing open")
        return 0

    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}

    base_url = os.environ.get("AGPOLICY_BASE_URL")
    token = os.environ.get("AGPOLICY_TOKEN")
    if not base_url or not token:
        print("[agpolicy hook] AGPOLICY_BASE_URL/AGPOLICY_TOKEN not set, failing open", file=sys.stderr)
        _emit("allow", "agpolicy gateway not configured, failing open")
        return 0

    body = json.dumps({"tool_name": tool_name, "tool_input": tool_input}).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/agpolicy/check_tool",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            result = json.loads(resp.read())
    except Exception as exc:
        print(f"[agpolicy hook] check_tool request failed: {exc!r}, failing open", file=sys.stderr)
        _emit("allow", f"agpolicy gateway unreachable ({exc}), failing open")
        return 0

    kind = result.get("decision", "allow")
    reason = result.get("reason")
    if kind == "allow":
        _emit("allow", reason)
    elif kind == "deny":
        _emit("deny", reason or "denied by agpolicy")
    else:
        # "rewrite" has no equivalent in Claude Code's PreToolUse
        # allow/deny/ask model -- deny rather than silently allow an
        # unrewritten call the policy didn't actually approve as-is.
        _emit("deny", f"agpolicy returned {kind!r}, not supported via this hook -- denying")
    return 0


def _emit(decision: str, reason: "str | None") -> None:
    out = {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": decision}}
    if reason:
        out["hookSpecificOutput"]["permissionDecisionReason"] = reason
    print(json.dumps(out))


if __name__ == "__main__":
    sys.exit(main())
