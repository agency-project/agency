"""Forward one CLI hook payload to the Agency session that launched the CLI.

Used by Codex hooks and Claude Code's explicit-policy PreToolUse hook.
Deliberately stdlib-only: an empty successful reply leaves the CLI's native
permission flow in charge; a policy denial is forwarded as native hook JSON.
An unavailable policy service blocks enforced PreToolUse calls with exit 2.

The command line that runs this file must be identical in every session --
Codex trusts a hook by the hash of its definition -- so everything
per-session (endpoint, token) comes from the environment the CLI inherited.
Outside an Agency session those variables are absent and it does nothing.
"""

import os
import json
import sys
import urllib.request

# Telemetry must add at most a short, bounded delay to the user's tool call,
# and finish inside the CLI's own hook timeout (3 s for Codex).
_TIMEOUT_S = 2.0


def main() -> int:
    url = os.environ.get("AGENCY_SESSION_HOOK_URL")
    token = os.environ.get("AGENCY_SESSION_TOKEN")
    payload = sys.stdin.buffer.read()
    if not url or not token:
        return 0
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            decision = response.read()
        if decision:
            sys.stdout.buffer.write(decision)
            sys.stdout.buffer.flush()
    except Exception as exc:
        print(f"[agency hook relay] {type(exc).__name__}: {exc}", file=sys.stderr)
        if os.environ.get("AGENCY_SESSION_ENFORCE_POLICY") == "1":
            try:
                event = json.loads(payload).get("hook_event_name")
            except (ValueError, AttributeError):
                event = "PreToolUse"
            if event == "PreToolUse":
                return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
