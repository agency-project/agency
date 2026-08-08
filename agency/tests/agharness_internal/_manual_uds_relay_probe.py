"""Manual probe: real agProxyLLM UDS listener + real agSandbox (with its new
default gateway-socket mount) + the in-container TCP-to-UDS relay, all
wired together for real. NOT a pytest file on purpose while hand-verifying.
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "agency")

from agency.agsandbox import agSandbox
from agency.agsandbox_backends.base import agSandboxBackendConfig
from agency.agharness_internal.agproxy_llm import agProxyLLM

RELAY_SOURCE = Path(
    "agency/agharness_internal/agproxy_ptrace_internal/_tcp_to_uds_relay.py"
).read_bytes()


def main():
    gateway = agProxyLLM()
    sock_path = gateway.ensure_uds_started()
    print(f"host UDS socket: {sock_path}")

    agconfig = agSandboxBackendConfig(backend="docker").agconfig
    sandbox = agSandbox("udsrelayprobe", agconfig=agconfig)
    relay_proc = None
    try:
        sandbox._backend._ensure_started()
        runtime, container_name = sandbox._backend._runtime, sandbox._backend._container_name()

        # Confirm the default mount actually landed inside the container.
        out, rc = sandbox._backend._container_exec("ls /var/run/agency_llm_gateway/")
        print(f"container's view of the mounted gateway dir (rc={rc}): {out.strip()!r}")
        assert rc == 0, f"mount did not land: {out}"
        assert Path(sock_path).name in out, f"socket file not visible in container: {out!r}"

        container_sock_path = f"/var/run/agency_llm_gateway/{Path(sock_path).name}"
        sandbox.write_file_bytes("/tmp/_tcp_to_uds_relay.py", RELAY_SOURCE)

        relay_proc = subprocess.Popen(
            [runtime, "exec", "-i", container_name, "python3", "/tmp/_tcp_to_uds_relay.py",
             container_sock_path, "58500"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        ready_line = relay_proc.stdout.readline()
        print(f"relay startup line: {ready_line.strip()!r}")
        assert ready_line.strip() == "READY", f"relay did not report ready: {ready_line!r}"

        # Now: from INSIDE the container, hit the relay's local port and
        # confirm it actually reaches the real host-side gateway (a 401
        # for a missing bearer token is a perfectly good proof of reaching
        # the real FastAPI app -- it's the same app `_build_app` builds for
        # the TCP listener, just reached via UDS this time).
        check_cmd = (
            "import urllib.request, json\n"
            "req = urllib.request.Request('http://127.0.0.1:58500/v1/chat/completions', "
            "data=b'{}', method='POST', headers={'Content-Type':'application/json'})\n"
            "try:\n"
            "    urllib.request.urlopen(req, timeout=5)\n"
            "except urllib.error.HTTPError as e:\n"
            "    print('HTTP', e.code, e.read().decode())\n"
        )
        result, rc = sandbox._backend._container_exec(f"python3 -c \"{check_cmd}\"")
        print(f"in-container request via relay (rc={rc}): {result.strip()}")
        assert "HTTP 401" in result, f"expected a 401 (reached the real app, no token), got: {result!r}"

        print("\nUDS RELAY PROBE: PASS")
    finally:
        if relay_proc is not None:
            relay_proc.terminate()
        try:
            sandbox.rm_container()
        except Exception as e:
            print(f"cleanup warning: {e}")
        gateway.stop()


if __name__ == "__main__":
    main()
