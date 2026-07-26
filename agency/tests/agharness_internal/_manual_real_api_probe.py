"""Manual probe driving the REAL public API end to end: agSandbox +
agProxyPtrace.launch(sandbox=...) + wire_to_sandbox + agpolicy -- not the
hand-rolled test drivers used earlier to validate the entrypoint mechanism
in isolation. NOT a pytest file on purpose while hand-verifying.
"""
import sys

sys.path.insert(0, "agency")

from agency.agsandbox import agSandbox
from agency.agsandbox_backends.base import agSandboxBackendConfig
from agency.agpolicy import agpolicy, agdecision
from agency.agharness_internal.agproxy_ptrace import agProxyPtrace, wire_to_sandbox


class _DenySpecificPathPolicy(agpolicy):
    def __init__(self, deny_path):
        self._deny_path = deny_path

    def check(self, ag, event):
        if event.path == self._deny_path:
            return agdecision.deny(f"blocked {event.path}")
        return agdecision.allow()


def main():
    agconfig = agSandboxBackendConfig(backend="docker").agconfig
    sandbox = agSandbox("realapiprobe", agconfig=agconfig)
    try:
        px = agProxyPtrace(agconfig)
        policy = _DenySpecificPathPolicy("/bin/true")

        handle = px.launch(
            ["/bin/sh", "-c", "echo REAL_API_PID=$$ && pwd && /bin/true; echo exit_was=$?"],
            {"PATH": "/usr/bin:/bin"},
            cwd="/workspace",
            policy=policy,
            ag=None,
            sandbox=sandbox,
        )
        wire_to_sandbox(handle, sandbox)

        stdout, stderr, rc = handle.wait(timeout=30)
        print(f"stdout={stdout!r}")
        print(f"stderr={stderr!r}")
        print(f"returncode={rc}")

        assert "REAL_API_PID=" in stdout, stdout
        assert "/workspace" in stdout, stdout
        assert "exit_was=126" in stdout, stdout  # denied exec -> shell reports 126
        assert rc == 0, rc

        live = sandbox.get_live_pids() if hasattr(sandbox, "get_live_pids") else None
        print(f"sandbox.get_live_pids() after completion: {live}")

        print("\nREAL-API PROBE: PASS")
    finally:
        try:
            sandbox.rm_container()
        except Exception as e:
            print(f"cleanup warning: {e}")


if __name__ == "__main__":
    main()
