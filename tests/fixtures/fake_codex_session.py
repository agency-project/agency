#!/usr/bin/env python3
"""A stand-in for the interactive `codex` CLI, driven by single keystrokes.

It reads Agency's `-c hooks.<Event>=...` and `-c otel.exporter=...` overrides
the way Codex does (TOML values), runs each hook command through the shell
with the hook JSON on stdin, and exports OTLP/JSON log records shaped like
Codex 0.162.1's (time in `event.timestamp`, timeUnixNano 0). Keys:

  t  a tool call that succeeds        p  a tool call awaiting approval
  y  approve the pending tool         n  deny it (telemetry only, no hook)
  m  one model request + response     i  interrupt the turn (Interrupt hook)
  s  end the turn (Stop hook)         q  exit
"""

import json
import os
import signal
import subprocess
import sys
import termios
import time
import tomllib
import tty
import urllib.request
from datetime import datetime, timezone

hooks = {}
endpoint = None
arguments = sys.argv[1:]
for flag, value in zip(arguments, arguments[1:]):
    if flag != "-c":
        continue
    key, _, raw = value.partition("=")
    parsed = tomllib.loads(f"value = {raw}")["value"]
    if key.startswith("hooks."):
        hooks[key.removeprefix("hooks.")] = parsed[0]["hooks"][0]
    elif key == "otel.exporter":
        endpoint = parsed["otlp-http"]["endpoint"]
session_id = "conv-fake"
turn_id = "turn-1"


def run_hook(name, **payload):
    handler = hooks[name]
    body = {
        "hook_event_name": name,
        "session_id": session_id,
        "cwd": os.getcwd(),
        "model": "fake-codex-model",
        **payload,
    }
    finished = subprocess.run(
        handler["command"],
        shell=True,
        input=json.dumps(body).encode(),
        capture_output=True,
        timeout=handler["timeout"],
    )
    return finished.returncode, finished.stdout


def log_event(name, **attributes):
    stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    values = [
        {"key": "event.name", "value": {"stringValue": f"codex.{name}"}},
        {"key": "event.timestamp", "value": {"stringValue": stamp}},
        {"key": "conversation.id", "value": {"stringValue": session_id}},
        {"key": "model", "value": {"stringValue": "fake-codex-model"}},
    ]
    for attribute, raw in attributes.items():
        values.append({"key": attribute, "value": {"stringValue": str(raw)}})
    record = {"timeUnixNano": "0", "attributes": values}
    header, _, token = os.environ["OTEL_EXPORTER_OTLP_LOGS_HEADERS"].partition("=")
    request = urllib.request.Request(
        endpoint,
        data=json.dumps({"resourceLogs": [{"scopeLogs": [{"logRecords": [record]}]}]}).encode(),
        headers={"Content-Type": "application/json", header: token},
        method="POST",
    )
    urllib.request.urlopen(request, timeout=5).read()


def say(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def on_resize(_signum, _frame):
    size = os.get_terminal_size(0)
    say(f"\r\nsize {size.columns}x{size.lines}\r\n")


signal.signal(signal.SIGWINCH, on_resize)
original = termios.tcgetattr(0)
tty.setraw(0)
try:
    log_event("conversation_starts")
    code, output = run_hook("SessionStart", source="startup")
    say("\x1b[?2004h\x1b[1;36mfake codex\x1b[0m ready\r\n")
    say(f"hook exit {code} output {output!r}\r\n")
    while True:
        key = os.read(0, 1)
        if key == b"q" or not key:
            break
        if key == b"t":
            run_hook("UserPromptSubmit", turn_id=turn_id, prompt="run a tool")
            run_hook(
                "PreToolUse",
                turn_id=turn_id,
                tool_name="Bash",
                tool_input={"command": "echo hi"},
                tool_use_id="exec-ok",
            )
            run_hook(
                "PostToolUse",
                turn_id=turn_id,
                tool_name="Bash",
                tool_input={"command": "echo hi"},
                tool_use_id="exec-ok",
                tool_response="hi\n",
            )
            say("tool done\r\n")
        elif key == b"p":
            run_hook(
                "PreToolUse",
                turn_id=turn_id,
                tool_name="Bash",
                tool_input={"command": "rm -rf build"},
                tool_use_id="exec-ask",
            )
            run_hook(
                "PermissionRequest",
                turn_id=turn_id,
                tool_name="Bash",
                tool_input={"command": "rm -rf build"},
            )
            say("approval asked\r\n")
        elif key == b"y":
            run_hook(
                "PostToolUse",
                turn_id=turn_id,
                tool_name="Bash",
                tool_input={"command": "rm -rf build"},
                tool_use_id="exec-ask",
                tool_response="",
            )
            say("approved\r\n")
        elif key == b"n":
            log_event(
                "tool_decision",
                call_id="exec-ask",
                tool_name="exec_command",
                decision="denied",
                source="User",
            )
            say("denied\r\n")
        elif key == b"m":
            log_event(
                "api_request",
                duration_ms=500,
                attempt=0,
                endpoint="/responses",
                **{"http.response.status_code": 200},
            )
            time.sleep(0.3)
            log_event(
                "sse_event",
                **{"event.kind": "response.completed"},
                input_token_count=100,
                output_token_count=7,
                cached_token_count=60,
                reasoning_token_count=0,
                ttft_ms=420,
            )
            say("model done\r\n")
        elif key == b"i":
            run_hook("Interrupt", turn_id=turn_id)
            say("interrupted\r\n")
        elif key == b"s":
            run_hook("Stop", turn_id=turn_id, last_assistant_message="done")
            say("turn over\r\n")
        else:
            say(key.decode(errors="replace"))
finally:
    termios.tcsetattr(0, termios.TCSADRAIN, original)
