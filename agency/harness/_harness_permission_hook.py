"""Standalone Claude Code hook bridging policy and tool telemetry.

Deliberately self-contained -- stdlib only (`json`, `os`, `sys`,
`urllib.request`), ZERO imports from the `agency` package -- because it
runs as a subprocess of the harness binary itself (e.g. `claude` invoking
its own hook), not inside this process, and can't reach into
this process's Python state directly. `claude_code.py` writes this file
into the launch's own `config_home` and registers it via `--settings`'
`hooks.PreToolUse`/`hooks.PostToolUse` blocks;
`AGPOLICY_BASE_URL`/`AGPOLICY_TOKEN` (set on the
harness's own env, the same values ANTHROPIC_BASE_URL's bearer token
already uses) tell it where to reach `agproxy_llm.py`'s
`/agpolicy/check_tool` route -- the one thing this subprocess and that
route share is the per-run bearer token, so that's the auth.

For ``PreToolUse`` it performs the existing policy POST first and only opens
a profiler span when the call will be allowed.  For ``PostToolUse``
(and ``PostToolUseFailure``) it closes the span with the matching
``tool_use_id``. Profiler POSTs go to the container-reachable agproxy HTTP
bridge, which forwards them to agProfilerIngest's separate host UDS; the
bearer token is header-only and is never included in the semantic payload.

Tool admission fails closed: only an explicit allow from Agency may start a
new tool. This preserves redirect, pause, and cancellation fences when the
policy gateway is unavailable. Profiler delivery remains best-effort;
telemetry failures do not revoke an admitted tool.
"""

import json
import os
import sys
import time
import urllib.request

_POLICY_TIMEOUT_S = 5.0
# Telemetry is best-effort and synchronous in Claude Code's hook process.
# Keep the combined Pre + Post profiler budget at or below 200 ms when the
# local gateway is wedged.  Healthy requests still return immediately.  A
# 50 ms boundary proved too short for two concurrent Claude hook processes on
# EC2 (interpreter startup plus the local HTTP -> host UDS hop), causing an
# otherwise valid PostToolUse event to be discarded and honestly downgraded
# to transcript-derived timing.
_PROFILER_TIMEOUT_S = 0.1
_MAX_PROFILER_FIELD_CHARS = 64 * 1024


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
            _post_profiler_event(payload, hook_event_name)
        else:
            _emit("deny", "Cannot check invocation admission: unknown hook event")
        return 0

    kind, reason = _check_tool_policy(payload)
    if kind == "allow":
        # A denied PreToolUse does not receive a matching PostToolUse. Open
        # only after policy explicitly allows so no denied call leaves
        # an unmatched profiler span behind.
        _post_profiler_event(payload, hook_event_name)
        _emit("allow", reason)
    elif kind == "deny":
        _emit("deny", reason or "denied by agpolicy")
    else:
        # "rewrite" has no equivalent in Claude Code's PreToolUse
        # allow/deny/ask model -- deny rather than silently allow an
        # unrewritten call the policy didn't actually approve as-is.
        _emit("deny", f"agpolicy returned {kind!r}, not supported via this hook -- denying")
    return 0


def _check_tool_policy(payload: dict) -> "tuple[str, str | None]":
    tool_name = payload.get("tool_name")
    tool_input = payload.get("tool_input", {})
    if not isinstance(tool_name, str) or not tool_name.strip() or not isinstance(tool_input, dict):
        return "deny", "Cannot check invocation admission: invalid tool name or input"

    base_url = os.environ.get("AGPOLICY_BASE_URL")
    token = os.environ.get("AGPOLICY_TOKEN")
    if not base_url or not token:
        print(
            "[agpolicy hook] AGPOLICY_BASE_URL/AGPOLICY_TOKEN not set, denying tool",
            file=sys.stderr,
        )
        return "deny", "Cannot check invocation admission: agpolicy gateway not configured"

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
            return "deny", "Cannot check invocation admission: invalid agpolicy decision"
        reason = result.get("reason")
        if reason is not None and not isinstance(reason, str):
            return "deny", "Cannot check invocation admission: invalid agpolicy reason"
    except Exception as exc:
        print(f"[agpolicy hook] check_tool request failed: {exc!r}, denying tool", file=sys.stderr)
        return (
            "deny",
            "Cannot check invocation admission: agpolicy request failed; return to the model",
        )

    return result["decision"], reason


def _post_profiler_event(payload: dict, hook_event_name: str) -> None:
    base_url = os.environ.get("AGPROF_BASE_URL")
    token = os.environ.get("AGPROF_TOKEN")
    if not base_url or not token:
        return
    tool_use_id = payload.get("tool_use_id")
    if not isinstance(tool_use_id, str) or not tool_use_id:
        # DATACOLLECTOR: append, correlate (tool_use_id, missing here) -- same no-transport gap as above.
        print(
            f"[agprof hook] {hook_event_name} missing tool_use_id; telemetry skipped",
            file=sys.stderr,
        )
        return

    # Capture the hook boundary before the HTTP round trip.  The token stays
    # exclusively in Authorization; even a caller-supplied top-level token is
    # stripped from the semantic payload before forwarding.
    safe_payload = {
        key: payload[key]
        for key in (
            "hook_event_name",
            "tool_use_id",
            "tool_name",
            "tool_input",
            "tool_response",
            "error",
            "duration_ms",
            "pid",
        )
        if key in payload
    }
    for key in ("tool_input", "tool_response", "error"):
        if key not in safe_payload:
            continue
        encoded = json.dumps(safe_payload[key], default=str)
        if len(encoded) <= _MAX_PROFILER_FIELD_CHARS:
            continue
        preview = encoded[:_MAX_PROFILER_FIELD_CHARS] + "…[truncated]"
        safe_payload[key] = {"_truncated_json": preview} if key == "tool_input" else preview
    body = json.dumps(
        {
            "hook_event_name": hook_event_name,
            "wall_ns": time.time_ns(),
            "perf_ns": time.perf_counter_ns(),
            "payload": safe_payload,
        }
    ).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/agprof/hook",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        # A profiler outage must add a small bounded hiccup, never the five
        # second policy budget, at both boundaries of every tool call.
        with urllib.request.urlopen(req, timeout=_PROFILER_TIMEOUT_S) as resp:
            result = json.loads(resp.read() or b"{}")
        if not result.get("ok"):
            # DATACOLLECTOR: append, correlate (tool_use_id) -- same no-transport gap as above.
            print(f"[agprof hook] ingest rejected event: {result!r}", file=sys.stderr)
    except Exception as exc:
        # DATACOLLECTOR: append, correlate (tool_use_id) -- same no-transport gap as above.
        print(f"[agprof hook] profiler request failed: {exc!r}, failing open", file=sys.stderr)


def _emit(decision: str, reason: "str | None") -> None:
    out = {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": decision}}
    if reason:
        out["hookSpecificOutput"]["permissionDecisionReason"] = reason
    print(json.dumps(out))


if __name__ == "__main__":
    sys.exit(main())
