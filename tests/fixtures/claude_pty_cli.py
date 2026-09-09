"""Scripted raw-terminal harness speaking Claude's input and hook conventions."""

import json
import os
import re
import sys
import termios
import time
import tty
import uuid
from pathlib import Path

root = Path(os.environ["AGENCY_CLAUDE_STATE"])
mode = os.environ["TEST_PTY_MODE"]
transcript = root / "projects" / re.sub(r"[^a-zA-Z0-9]", "-", str(root)) / "native.jsonl"
transcript.parent.mkdir(parents=True)


def event(kind, **payload):
    state = json.loads((root / "agency-turn.json").read_text())
    path = root / "events" / f"{time.time_ns()}-{uuid.uuid4().hex}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "turn_id": state["turn_id"],
                "payload": {
                    "hook_event_name": kind,
                    "session_id": "native",
                    **payload,
                },
            }
        )
    )
    temporary.replace(path)


def row(role, text):
    with transcript.open("a") as out:
        out.write(
            json.dumps(
                {
                    "type": role,
                    "message": {
                        "content": [
                            {"type": "text", "text": text},
                        ]
                    },
                }
            )
            + "\n"
        )


def editor(draft=""):
    # A three-line Claude input box, with cursor on the editable second line.
    sys.stdout.write("\x1b[2J\x1b[H──────────\r\n❯ " + draft + "\r\n──────────\x1b[2;3H")
    sys.stdout.flush()


tty.setraw(0, termios.TCSANOW)
event("SessionStart")
editor()
buffer = b""
while True:
    chunk = os.read(0, 65536)
    with (root / "input.bin").open("ab") as out:
        out.write(chunk)
    buffer += chunk
    if buffer == b"\x03":
        buffer = b""
        if mode == "exit":
            sys.exit(0)
        if mode == "completion_race":
            row("assistant", "scripted final")
            event("Stop", last_assistant_message="scripted final")
            continue
        if mode == "restored":
            editor("Restored prompt")
        else:
            row("user", "[Request interrupted by user]")
            editor()
        continue
    if buffer == b"\x1b\x1b":
        buffer = b""
        editor()
        continue
    if not buffer.endswith(b"\x1b[201~\r"):
        continue
    assert buffer.startswith(b"\x1b[200~")
    prompt = buffer[len(b"\x1b[200~") : -len(b"\x1b[201~\r")].decode()
    buffer = b""
    row("user", prompt)
    if mode == "unacknowledged" and prompt.startswith("[Agency redirect]"):
        continue
    event("UserPromptSubmit", prompt=prompt)
    if prompt.startswith("[Agency redirect]") or mode == "finished":
        row("assistant", "scripted final")
        event("Stop", last_assistant_message="scripted final")
