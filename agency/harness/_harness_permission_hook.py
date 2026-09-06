"""Standalone hook bridging policy admission and tool-call telemetry to
Agency's host-side endpoints.

Deliberately self-contained -- stdlib only (`json`, `os`, `sys`,
`urllib.request`), ZERO imports from the `agency` package -- because it
runs as a subprocess of the harness binary itself (e.g. `claude`/`codex`/
`grok` invoking their own hook), not inside this process, and can't reach
into this process's Python state directly. Each harness adapter that uses
it (`claude_code.py`, `codex.py`, `grok.py` -- their PreToolUse/PostToolUse
hook payload shapes are near-identical) writes this file into the launch's
own `config_home` and registers it via that CLI's own hook-config format.
`AGPOLICY_BASE_URL`/`AGPOLICY_TOKEN` (set on the harness's own env, the same
values ANTHROPIC_BASE_URL's bearer token already uses) tell it where to
reach the daemon's `/agpolicy/check_tool`/`/agpolicy/complete_tool` routes
-- the one thing this subprocess and those routes share is the per-run
bearer token, so that's the auth.

For ``PreToolUse`` it posts to `/agpolicy/check_tool`, which both decides
allow/deny AND opens a host-side pending span, returning a `call_id`. For
``PostToolUse``/``PostToolUseFailure`` it posts to `/agpolicy/complete_tool`
to close that span and record the tool's result. Since each hook firing is
a fresh subprocess with no memory of the last one, the `call_id` returned
by PreToolUse is persisted to a small per-`tool_use_id` file under
`AGPOLICY_STATE_DIR` (each adapter points this at its own per-launch
`config_home`, so it's cleaned up the same way the rest of that directory
is) and read back (then deleted) by PostToolUse.

Tool admission fails closed: only an explicit allow from Agency may start a
new tool. This preserves redirect, pause, and cancellation fences when the
policy gateway is unavailable. Completion telemetry is best-effort; its
failure never revokes an already-admitted tool call.
"""

import json
import os
import sys
import urllib.request

_POLICY_TIMEOUT_S = 5.0
# Telemetry is best-effort and synchronous in the hook process. Keep the
# PostToolUse completion POST bounded well under Claude Code's own hook
# timeout even when the local gateway is wedged; healthy requests still
# return immediately.
_COMPLETION_TIMEOUT_S = 2.0
_MAX_RESULT_FIELD_CHARS = 64 * 1024


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        # DATACOLLECTOR: append -- this subprocess has zero agency imports and no transport
        # to agDataCollector today; folding needs a new HTTP call here, not just a type name.
        print(f"[agpolicy hook] could not parse hook input: {exc}", file=sys.stderr)
        _emit("deny", "Cannot check invocation admission: hook input was not valid JSON")
        return 0

    if not isinstance(payload, dict):
        _emit("deny", "Cannot check invocation admission: hook input must be an object")
        return 0

    hook_event_name = payload.get("hook_event_name") or "PreToolUse"
    if hook_event_name != "PreToolUse":
        if hook_event_name in ("PostToolUse", "PostToolUseFailure"):
            _report_completion(payload, hook_event_name)
        else:
            _emit("deny", "Cannot check invocation admission: unknown hook event")
        return 0

    kind, reason, call_id = _check_tool_policy(payload)
    if kind == "allow":
        _remember_call_id(payload, call_id)
        _emit("allow", reason)
    elif kind == "deny":
        _emit("deny", reason or "denied by agpolicy")
    else:
        # "rewrite" has no equivalent in Claude Code's PreToolUse
        # allow/deny/ask model -- deny rather than silently allow an
        # unrewritten call the policy didn't actually approve as-is.
        _emit("deny", f"agpolicy returned {kind!r}, not supported via this hook -- denying")
    return 0


def _check_tool_policy(payload: dict) -> "tuple[str, str | None, str | None]":
    tool_name = payload.get("tool_name")
    tool_input = payload.get("tool_input", {})
    if not isinstance(tool_name, str) or not tool_name.strip() or not isinstance(tool_input, dict):
        return "deny", "Cannot check invocation admission: invalid tool name or input", None

    base_url = os.environ.get("AGPOLICY_BASE_URL")
    token = os.environ.get("AGPOLICY_TOKEN")
    if not base_url or not token:
        print(
            "[agpolicy hook] AGPOLICY_BASE_URL/AGPOLICY_TOKEN not set, denying tool",
            file=sys.stderr,
        )
        return "deny", "Cannot check invocation admission: agpolicy gateway not configured", None

    try:
        body = json.dumps({"tool_name": tool_name, "tool_input": tool_input}).encode()
        req = urllib.request.Request(
            base_url.rstrip("/") + "/agpolicy/check_tool",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_POLICY_TIMEOUT_S) as resp:
            result = json.loads(resp.read())
        if not isinstance(result, dict) or result.get("decision") not in ("allow", "deny"):
            return "deny", "Cannot check invocation admission: invalid agpolicy decision", None
        reason = result.get("reason")
        if reason is not None and not isinstance(reason, str):
            return "deny", "Cannot check invocation admission: invalid agpolicy reason", None
    except Exception as exc:
        print(f"[agpolicy hook] check_tool request failed: {exc!r}, denying tool", file=sys.stderr)
        return (
            "deny",
            "Cannot check invocation admission: agpolicy request failed; return to the model",
            None,
        )

    return result["decision"], reason, result.get("call_id")


def _call_state_path(tool_use_id: str) -> "str | None":
    state_dir = os.environ.get("AGPOLICY_STATE_DIR")
    if not state_dir:
        return None
    return os.path.join(state_dir, f".agpolicy_call_{tool_use_id}.json")


def _remember_call_id(payload: dict, call_id: "str | None") -> None:
    """Persist *call_id* so the separate PostToolUse subprocess can find it
    -- best-effort: a missing/unwritable state dir just means completion
    telemetry is skipped later, never that admission itself fails."""
    if not call_id:
        return
    tool_use_id = payload.get("tool_use_id")
    if not isinstance(tool_use_id, str) or not tool_use_id:
        return
    path = _call_state_path(tool_use_id)
    if path is None:
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"call_id": call_id}, f)
    except OSError as exc:
        print(f"[agpolicy hook] could not persist call_id: {exc!r}", file=sys.stderr)


def _report_completion(payload: dict, hook_event_name: str) -> None:
    tool_use_id = payload.get("tool_use_id")
    if not isinstance(tool_use_id, str) or not tool_use_id:
        print(
            f"[agpolicy hook] {hook_event_name} missing tool_use_id; telemetry skipped",
            file=sys.stderr,
        )
        return
    path = _call_state_path(tool_use_id)
    if path is None:
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            call_id = json.load(f).get("call_id")
    except (OSError, json.JSONDecodeError):
        # No matching PreToolUse admission (denied, or state dir wasn't
        # configured) -- nothing to complete.
        return
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    if not call_id:
        return

    base_url = os.environ.get("AGPOLICY_BASE_URL")
    token = os.environ.get("AGPOLICY_TOKEN")
    if not base_url or not token:
        return

    result = payload.get("tool_response")
    error = payload.get("error")
    if hook_event_name == "PostToolUseFailure" and error is None:
        error = "Tool reported failure"
    encoded_result = json.dumps(result, default=str)
    if len(encoded_result) > _MAX_RESULT_FIELD_CHARS:
        result = {"_truncated_json": encoded_result[:_MAX_RESULT_FIELD_CHARS] + "…[truncated]"}
    if isinstance(error, str) and len(error) > _MAX_RESULT_FIELD_CHARS:
        error = error[:_MAX_RESULT_FIELD_CHARS] + "…[truncated]"

    body = json.dumps({"call_id": call_id, "result": result, "error": error}).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/agpolicy/complete_tool",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        # A gateway outage must add a small bounded hiccup, never block the
        # harness's own PostToolUse handling on Agency's telemetry.
        with urllib.request.urlopen(req, timeout=_COMPLETION_TIMEOUT_S) as resp:
            resp.read()
    except Exception as exc:
        # DATACOLLECTOR: append, correlate (tool_use_id) -- this subprocess has no transport
        # to agDataCollector today; folding needs a new HTTP call here, not just a type name.
        print(f"[agpolicy hook] complete_tool request failed: {exc!r}", file=sys.stderr)


def _emit(decision: str, reason: "str | None") -> None:
    out = {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": decision}}
    if reason:
        out["hookSpecificOutput"]["permissionDecisionReason"] = reason
    print(json.dumps(out))


if __name__ == "__main__":
    sys.exit(main())
