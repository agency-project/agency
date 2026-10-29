"""Interactive Claude Code controller. Does not import Agency."""

import fcntl
import json
import os
from pathlib import Path
import pty
import select
import shlex
import signal
import struct
import subprocess
import sys
import termios
import time

import pyte


def main():
    settings = json.load(sys.stdin)
    root = Path("/tmp/claude-fidelity-state")
    root.mkdir()
    out = Path("/smoke-output")
    (root / ".claude.json").write_text(
        json.dumps(
            {
                "hasCompletedOnboarding": True,
                "theme": "dark",
                "bypassPermissionsModeAccepted": True,
                "projects": {str(root): {"hasTrustDialogAccepted": True}},
            }
        )
    )
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "TERM": "xterm-256color",
        "HOME": "/root",
        "IS_SANDBOX": "1",
        "CLAUDE_CONFIG_DIR": str(root),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_AUTOUPDATER": "1",
        "CLAUDE_CODE_USE_BEDROCK": "0",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-haiku-5-5",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-haiku-5-5",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-haiku-5-5",
        "ANTHROPIC_BASE_URL": settings["base_url"],
        "ANTHROPIC_AUTH_TOKEN": settings["proxy_token"],
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000",
    }
    protected = set(env) - {"PATH", "HOME"}
    for name, value in settings.get("task_environment", {}).items():
        if name not in protected and not name.startswith(("ANTHROPIC_", "AGENCY_", "AGPOLICY_")):
            env[name] = value
    # This observer writes lifecycle events and never modifies or blocks a turn.
    hook = root / "observer.py"
    hook.write_text(
        "import json,sys,time\nfrom pathlib import Path\np=json.load(sys.stdin)\n"
        "Path('/smoke-output/events').mkdir(exist_ok=True)\n"
        "f=Path('/smoke-output/events',str(time.time_ns())+'.json')\n"
        "t=f.with_suffix('.tmp');t.write_text(json.dumps(p));t.replace(f)\n"
    )
    hook_command = "env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH " + shlex.join(
        [settings.get("hook_python", "python3"), str(hook)]
    )
    hooks = {
        name: [{"hooks": [{"type": "command", "command": hook_command}]}]
        for name in ["SessionStart", "UserPromptSubmit", "Stop", "StopFailure"]
    }
    if settings.get("stop_gate"):
        stop_hook = hooks["Stop"][0]["hooks"][0]
        stop_hook["command"] = "env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH " + shlex.join(
            [settings["hook_python"], "/smoke-driver/stop_gate.py", hook_command]
        )
        stop_hook["timeout"] = settings["gate_timeout_s"] + 30
    argv = [
        settings["binary"],
        "--permission-mode",
        "bypassPermissions",
        "--setting-sources",
        "",
        "--settings",
        json.dumps({"hooks": hooks}),
        "--mcp-config",
        '{"mcpServers":{}}',
        "--strict-mcp-config",
        "--model",
        "claude-haiku-5-5",
        "--effort",
        "medium",
        "--disallowedTools",
        "Agent",
        "--autocompact",
        "900000",
    ]
    if settings.get("max_turns", None) is not None:
        argv += ["--max-turns", str(settings.get("max_turns", None))]
    (out / "launch.json").write_text(
        json.dumps({"argv": argv, "cwd": str(root), "mode": "interactive-pty"}, indent=2)
    )
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 36, 120, 0, 0))
    proc = subprocess.Popen(
        argv, cwd=root, env=env, stdin=slave, stdout=slave, stderr=slave, start_new_session=True
    )
    os.close(slave)
    screen = pyte.Screen(120, 36)
    stream = pyte.Stream(screen)
    start = time.monotonic()
    submitted = None
    result = {"status": "failed", "mode": "interactive-pty"}
    try:
        with (out / "terminal.log").open("wb") as log:
            while time.monotonic() - start < settings.get("timeout", 900):
                readable, _, _ = select.select([master], [], [], 0.1)
                if readable:
                    try:
                        data = os.read(master, 65536)
                    except OSError:
                        break
                    log.write(data)
                    log.flush()
                    stream.feed(data.decode(errors="replace"))
                    if b"\x1b[6n" in data:
                        os.write(master, b"\x1b[1;1R")
                    if b"\x1b[c" in data:
                        os.write(master, b"\x1b[?1;2c")
                events = []
                for path in sorted((out / "events").glob("*.json")):
                    events.append(json.loads(path.read_text()))
                lines = screen.display
                if submitted is None and any(
                    e.get("hook_event_name") == "SessionStart" for e in events
                ):
                    if any(line.lstrip().startswith("❯") for line in lines):
                        os.write(master, b"\x1b[200~" + settings["prompt"].encode() + b"\x1b[201~")
                        time.sleep(0.2)
                        os.write(master, b"\r")
                        submitted = time.monotonic()
                stops = [e for e in events if e.get("hook_event_name") in {"Stop", "StopFailure"}]
                if submitted is not None and stops:
                    event = stops[-1]
                    result.update(
                        status="completed" if event["hook_event_name"] == "Stop" else "failed",
                        final_text=event.get("last_assistant_message", ""),
                        event=event,
                        wall_s=time.monotonic() - submitted,
                    )
                    break
                if proc.poll() is not None:
                    result["error"] = f"Claude exited with {proc.returncode}"
                    break
            else:
                result.update(
                    status="timeout",
                    error=f"interactive task exceeded {settings.get('timeout', 900)} seconds",
                )
        (out / "screen.txt").write_text("\n".join(screen.display))
        result["startup_and_task_wall_s"] = time.monotonic() - start
        result["submitted"] = submitted is not None
        for path in root.rglob("*.jsonl"):
            if "projects" in path.parts:
                (out / "transcript.jsonl").write_bytes(path.read_bytes())
        (out / "result.json").write_text(json.dumps(result, indent=2))
        print(
            json.dumps({k: v for k, v in result.items() if k not in {"event", "final_text"}}),
            flush=True,
        )
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        os.close(master)


if __name__ == "__main__":
    main()
