"""Manual probe: real two-call continuity test for the new native-session
history mechanism (docs/Design_harness_history.md) -- call 1 tells the
agent a fact, a FRESH sandbox is created for call 2 (a different
container instance, not the same one), and call 2 asks the agent to recall
that fact using --resume against the captured session blob. NOT a pytest
file on purpose while hand-verifying.
"""

import subprocess
import sys

sys.path.insert(0, "agency")

from agency.agconfig import agConfig
from agency.agent import agent
from agency.agskill import agskill
from agency.agdata import agdata
from agency.agsandbox import agSandbox
from agency.agsandbox_backends.base import agSandboxBackendConfig

CLAUDE_HOST_BINARY = "/home/eecs/js_park/.local/share/claude/versions/2.1.220"


def install_claude(sandbox):
    sandbox._backend._ensure_started()
    runtime, name = sandbox._backend._runtime, sandbox._backend._container_name()
    subprocess.run(
        [runtime, "cp", CLAUDE_HOST_BINARY, f"{name}:/usr/local/bin/claude"], capture_output=True
    )
    subprocess.run([runtime, "exec", name, "chmod", "+x", "/usr/local/bin/claude"], check=True)


def main():
    cfg = agConfig(
        {
            "agllm_backend": {
                "provider": "bedrock",
                "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                "region": "us-east-1",
            },
        }
    )
    backend_cfg = agSandboxBackendConfig(backend="docker").agconfig
    cfg = cfg.clone()
    cfg.set("agsandbox_backend", "backend", backend_cfg.get("agsandbox_backend", "backend", None))

    skill = agskill(name="continuity_test_skill", system_prompt="You are a test assistant.")

    sandbox1 = agSandbox("historyprobe1", agconfig=cfg)
    ag = agent("history_probe_agent", agconfig=cfg, sandbox=sandbox1, engine="claude_code")
    try:
        install_claude(sandbox1)

        print("=== Call 1: tell the agent a secret code ===")
        r1 = ag.run(
            skill,
            agdata(
                instruction=("My project's deployment codename is PURPLE-42-NARWHAL. Just say OK.")
            ),
        )
        r1._resolve()
        print("call 1 result:", r1._data)
        assert not r1.is_pending() and not r1._data.get("error")

        stored = ag._harness_sessions.get("claude_code")
        print(
            f"\ncaptured session state: session_id={stored.get('session_id') if stored else None}, "
            f"blob_bytes={len(stored['blob_b64']) if stored else 0}"
        )
        assert stored and stored.get("session_id"), "no session captured after call 1!"

    finally:
        try:
            sandbox1.rm_container()
        except Exception as e:
            print(f"sandbox1 cleanup warning: {e}")

    # Swap to a completely fresh sandbox/container for call 2 -- proving
    # continuity travels with the AGENT (ag._harness_sessions), not with
    # any particular container.
    sandbox2 = agSandbox("historyprobe2", agconfig=cfg)
    ag.sandbox = sandbox2
    try:
        install_claude(sandbox2)

        print("\n=== Call 2 (FRESH container): ask it to recall the secret code ===")
        r2 = ag.run(
            skill,
            agdata(
                instruction="What's my project's deployment codename? Reply with just the codename."
            ),
        )
        r2._resolve()
        print("call 2 result:", r2._data)

        assert not r2.is_pending() and not r2._data.get("error")
        text = r2._data.get("result", "")
        assert "PURPLE-42-NARWHAL" in text, (
            f"continuity FAILED -- agent didn't recall the code: {text!r}"
        )

        print("\nHISTORY CONTINUITY PROBE (real --resume, across a fresh container): PASS")
    finally:
        try:
            sandbox2.rm_container()
        except Exception as e:
            print(f"sandbox2 cleanup warning: {e}")


if __name__ == "__main__":
    main()
