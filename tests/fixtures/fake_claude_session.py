#!/usr/bin/env python3
"""A stand-in for the interactive `claude` CLI, driven by single keystrokes.

It speaks the same integration surfaces as the real CLI under `agency claude`:
it reads the hook settings file passed with `--settings`, posts hook JSON to
those HTTP hooks, and exports OpenTelemetry log records as OTLP/JSON to the
endpoint in its environment. Keys:

  t  a tool call that succeeds        p  a tool call awaiting permission
  y  approve the pending tool         n  reject it (telemetry only, no hook)
  m  one completed model request      s  end the turn (Stop hook)
  b  start a background process that leaves the process group
  q  exit                             anything else is echoed
"""

import json
import os
import signal
import subprocess
import sys
import termios
import time
import tty
import urllib.request
import uuid

args = sys.argv[1:]
settings = json.loads(open(args[args.index("--settings") + 1]).read())
session_id = args[args.index("--session-id") + 1] if "--session-id" in args else str(uuid.uuid4())
hook = settings["hooks"]["PreToolUse"][0]["hooks"][0]
token = os.environ[hook["allowedEnvVars"][0]]


def post(url, body, headers):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, response.read()


def hook_event(name, **payload):
    return post(
        hook["url"],
        {"hook_event_name": name, "session_id": session_id, "cwd": os.getcwd(), **payload},
        {"Authorization": f"Bearer {token}"},
    )


def log_event(name, **attributes):
    key, value = os.environ["OTEL_EXPORTER_OTLP_LOGS_HEADERS"].split("=", 1)
    values = [{"key": "event.name", "value": {"stringValue": name}}]
    for attribute, raw in attributes.items():
        if isinstance(raw, bool):
            encoded = {"boolValue": raw}
        elif isinstance(raw, int):
            encoded = {"intValue": str(raw)}
        elif isinstance(raw, float):
            encoded = {"doubleValue": raw}
        else:
            encoded = {"stringValue": str(raw)}
        values.append({"key": attribute, "value": encoded})
    record = {
        "timeUnixNano": str(time.time_ns()),
        "body": {"stringValue": f"claude_code.{name}"},
        "attributes": values,
    }
    body = {"resourceLogs": [{"scopeLogs": [{"logRecords": [record]}]}]}
    post(os.environ["OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"], body, {key: value})


def say(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def on_resize(_signum, _frame):
    rows, columns = os.get_terminal_size(0).lines, os.get_terminal_size(0).columns
    say(f"\r\nsize {columns}x{rows}\r\n")


signal.signal(signal.SIGWINCH, on_resize)
original = termios.tcgetattr(0)
tty.setraw(0)
background = None
try:
    status, reply = hook_event("SessionStart", source="startup", model="fake-model")
    say("\x1b[?2004h\x1b[1;32mfake claude\x1b[0m ready\r\n")
    say(f"hook reply {status} {reply!r}\r\n")
    say(f"session leader {os.getsid(0) == os.getpid()} tty {os.ttyname(0)}\r\n")
    while True:
        key = os.read(0, 1)
        if key == b"q" or not key:
            break
        if key == b"t":
            hook_event("UserPromptSubmit", prompt="run a tool", prompt_id="p1")
            hook_event(
                "PreToolUse",
                tool_name="Bash",
                tool_input={"command": "echo hi"},
                tool_use_id="tool-ok",
            )
            time.sleep(0.2)
            hook_event(
                "PostToolUse",
                tool_name="Bash",
                tool_input={"command": "echo hi"},
                tool_use_id="tool-ok",
                tool_response={"stdout": "hi"},
                duration_ms=150,
            )
            say("tool done\r\n")
        elif key == b"p":
            hook_event(
                "PreToolUse",
                tool_name="Write",
                tool_input={"file_path": "x"},
                tool_use_id="tool-ask",
            )
            hook_event("PermissionRequest", tool_name="Write", tool_input={"file_path": "x"})
            say("permission asked\r\n")
        elif key == b"y":
            hook_event(
                "PostToolUse",
                tool_name="Write",
                tool_input={"file_path": "x"},
                tool_use_id="tool-ask",
                tool_response={"ok": True},
                duration_ms=20,
            )
            say("approved\r\n")
        elif key == b"n":
            log_event(
                "tool_decision",
                tool_name="Write",
                tool_use_id="tool-ask",
                decision="reject",
                source="user_reject",
            )
            say("rejected\r\n")
        elif key == b"m":
            log_event(
                "api_request",
                model="fake-model",
                duration_ms=1200,
                input_tokens=10,
                cache_read_tokens=5,
                output_tokens=7,
                request_id="req_1",
                cost_usd=0.001,
            )
            say("model done\r\n")
        elif key == b"s":
            hook_event("Stop", last_assistant_message="done")
            say("turn over\r\n")
        elif key == b"b":
            background = subprocess.Popen(["sleep", "300"], start_new_session=True)
            say(f"background {background.pid}\r\n")
        else:
            say(key.decode(errors="replace"))
finally:
    termios.tcsetattr(0, termios.TCSADRAIN, original)
