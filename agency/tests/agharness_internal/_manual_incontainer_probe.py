"""Manual, standalone probe for _in_container_entrypoint.py -- NOT a pytest
file (no test_ prefix on purpose while this is being hand-verified). Drives
the entrypoint exactly the way the host-side launcher will: spawn it as a
subprocess, send a launch spec on stdin, read JSON event lines, reply
"allow" to each, and check the final result. Run directly: `python3
_manual_incontainer_probe.py`.
"""

import json
import subprocess
import sys

ENTRYPOINT = "agency/agharness_internal/agproxy_ptrace_internal/_in_container_entrypoint.py"


def run_probe(argv, expect_stdout_contains=None, expect_returncode=0):
    proc = subprocess.Popen(
        [sys.executable, ENTRYPOINT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    spec = {
        "argv": argv,
        "envp": {"PATH": "/usr/bin:/bin"},
        "cwd": "",
        "syscalls": ["execve", "execveat"],
    }
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
            proc.stdin.write(json.dumps({"type": "decision", "kind": "allow"}) + "\n")
            proc.stdin.flush()
        elif msg["type"] in ("result", "error"):
            result = msg
            break
        # spawn/exit messages: just observe

    proc.wait(timeout=10)
    stderr = proc.stderr.read()

    print(f"argv={argv}")
    print(f"  events seen: {[(e['syscall'], e.get('path')) for e in events]}")
    print(f"  result: {result}")
    if stderr.strip():
        print(f"  entrypoint stderr: {stderr.strip()}")

    assert result is not None, "no result/error message received"
    assert result["type"] == "result", f"got error instead: {result}"
    assert result["returncode"] == expect_returncode, (
        f"expected rc={expect_returncode}, got {result['returncode']}"
    )
    if expect_stdout_contains:
        assert expect_stdout_contains in result["stdout"], (
            f"stdout missing {expect_stdout_contains!r}: {result['stdout']!r}"
        )
    assert len(events) >= 1, "expected at least one intercepted execve event"
    print("  PASS")
    return result, events


if __name__ == "__main__":
    run_probe(
        ["/bin/sh", "-c", "echo hello-from-traced-child && /bin/true"],
        expect_stdout_contains="hello-from-traced-child",
    )
    print()
    run_probe(["/bin/sh", "-c", "exit 7"], expect_returncode=7)
    print("\nALL PROBES PASSED")
