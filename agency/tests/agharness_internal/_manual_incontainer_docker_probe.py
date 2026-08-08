"""Manual probe: drives _in_container_entrypoint.py via `docker exec -i`
against a real running container -- the actual deployment shape, not just
a bare host subprocess. NOT a pytest file on purpose while hand-verifying.
"""
import json
import subprocess
import sys

CONTAINER = "test-ptrace-container"
ENTRYPOINT_IN_CONTAINER = "/tmp/_in_container_entrypoint.py"


def run_probe(argv, decide=lambda ev: "allow", expect_returncode=None, expect_stdout_contains=None, label=""):
    proc = subprocess.Popen(
        ["docker", "exec", "-i", CONTAINER, "python3", ENTRYPOINT_IN_CONTAINER],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    spec = {"argv": argv, "envp": {"PATH": "/usr/bin:/bin"}, "cwd": "/workspace", "syscalls": ["execve", "execveat"]}
    proc.stdin.write(json.dumps(spec) + "\n")
    proc.stdin.flush()

    events = []
    result = None
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        msg = json.loads(line)
        if msg["type"] == "event":
            events.append(msg)
            kind = decide(msg)
            proc.stdin.write(json.dumps({"type": "decision", "kind": kind}) + "\n")
            proc.stdin.flush()
        elif msg["type"] in ("result", "error"):
            result = msg
            break

    proc.wait(timeout=15)
    stderr = proc.stderr.read()

    print(f"[{label}] argv={argv}")
    print(f"  events: {[(e['syscall'], e.get('path'), e.get('pid')) for e in events]}")
    print(f"  result: {result}")
    if stderr.strip():
        print(f"  stderr: {stderr.strip()}")

    assert result is not None and result["type"] == "result", f"unexpected: {result}"
    if expect_returncode is not None:
        assert result["returncode"] == expect_returncode, f"expected rc={expect_returncode}, got {result['returncode']}"
    if expect_stdout_contains:
        assert expect_stdout_contains in result["stdout"], f"missing {expect_stdout_contains!r} in {result['stdout']!r}"
    print("  PASS")
    return result, events


if __name__ == "__main__":
    # 1. Baseline: allow everything, confirm real container-namespace execution + cwd.
    run_probe(
        ["/bin/sh", "-c", "echo IN_CONTAINER_PID=$$ && pwd && echo hi > /workspace/probe_marker.txt"],
        expect_returncode=0, expect_stdout_contains="IN_CONTAINER_PID=", label="allow-baseline",
    )

    # 2. Confirm the write actually landed in the CONTAINER's /workspace (proves
    #    this ran in the container's own mount namespace, not the host's).
    check = subprocess.run(["docker", "exec", CONTAINER, "cat", "/workspace/probe_marker.txt"],
                            capture_output=True, text=True)
    assert check.stdout.strip() == "hi", f"marker file not found/wrong in container: {check!r}"
    print("[filesystem-check] /workspace/probe_marker.txt correctly written inside the container: PASS")

    # 3. Deny path: reject the execve of /bin/true specifically, allow /bin/sh itself.
    def deny_true(ev):
        return "deny" if ev.get("path") == "/bin/true" else "allow"

    run_probe(
        ["/bin/sh", "-c", "/bin/true; echo exit_was=$?"],
        decide=deny_true, expect_stdout_contains="exit_was=", label="deny-specific-exec",
    )

    print("\nALL DOCKER-EXEC PROBES PASSED")
