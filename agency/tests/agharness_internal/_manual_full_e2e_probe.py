"""Full end-to-end probe: real agent, real agskill, real Bedrock-backed LLM,
real docker-backed agSandbox (with the real `claude` binary copied in),
engine="claude_code" -- exercises the actual new in-container code path in
claude_code.py through the real public agent.run() API. NOT a pytest file
on purpose while hand-verifying.
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


def main():
    cfg = agConfig({
        "agllm_backend": {
            "provider": "bedrock",
            "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "region": "us-east-1",
        },
    })
    backend_cfg = agSandboxBackendConfig(backend="docker").agconfig
    cfg = cfg.clone()
    for owner, key in [("agsandbox_backend", "backend")]:
        cfg.set(owner, key, backend_cfg.get(owner, key, None))

    sandbox = agSandbox("e2eprobe", agconfig=cfg)
    ag = agent("e2e_claude_agent", agconfig=cfg, sandbox=sandbox, engine="claude_code")

    try:
        sandbox._backend._ensure_started()
        runtime, container_name = sandbox._backend._runtime, sandbox._backend._container_name()
        print(f"container: {container_name}")

        print("copying real claude binary into the container (264MB)...")
        subprocess.run([runtime, "cp", CLAUDE_HOST_BINARY, f"{container_name}:/usr/local/bin/claude"],
                       capture_output=True)
        subprocess.run([runtime, "exec", container_name, "chmod", "+x", "/usr/local/bin/claude"],
                       check=True)
        ver = subprocess.run([runtime, "exec", container_name, "/usr/local/bin/claude", "--version"],
                              capture_output=True, text=True)
        print(f"claude --version inside container: {ver.stdout.strip()}")

        skill = agskill(
            name="e2e_test_skill",
            system_prompt="You are a test assistant running inside a sandboxed container.",
        )

        print("\nrunning ag.run(skill, ...) -- this launches the real claude CLI "
              "inside the real container via the new in-container ptrace + UDS-relay path...")
        result = ag.run(skill, agdata(instruction=(
            "Run the shell command `echo HELLO_FROM_CONTAINER && pwd && echo $$` "
            "using your Bash tool, then tell me exactly what it printed."
        )))
        result._resolve()

        print(f"\nresult pending: {result.is_pending()}")
        print(f"result data: {result._data}")
        print(f"agent ctx messages: {ag.ctx.messages}")

        assert not result.is_pending()
        assert "error" not in result._data or not result._data.get("error"), result._data
        text = result._data.get("result", "")
        assert "HELLO_FROM_CONTAINER" in text, f"unexpected result: {result._data!r}"

        print("\nFULL END-TO-END PROBE (real claude CLI, inside real container): PASS")
    finally:
        try:
            sandbox.rm_container()
        except Exception as e:
            print(f"cleanup warning: {e}")


if __name__ == "__main__":
    main()
