"""Drive a running interactive session the way its CLI and browser would."""

import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pyte
import pytest

from agency.sessions.terminal import _Screen

FIXTURES = Path(__file__).parents[1] / "fixtures"
linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="interactive sessions are Linux-only"
)


def post(session, path, body, headers):
    request = urllib.request.Request(
        f"http://127.0.0.1:{session.port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def hook(session, name, **payload):
    return post(
        session,
        "/hooks",
        {"hook_event_name": name, **payload},
        {"Authorization": f"Bearer {session.host_token}"},
    )


def otlp(session, name, at, *, time_in_attribute=False, **attributes):
    """Export one log record. Codex sends timeUnixNano 0 and the time as an
    `event.timestamp` attribute; *time_in_attribute* reproduces that."""
    values = [{"key": "event.name", "value": {"stringValue": name}}]
    if time_in_attribute:
        stamp = datetime.fromtimestamp(at, timezone.utc).isoformat(timespec="milliseconds")
        values.append({"key": "event.timestamp", "value": {"stringValue": stamp}})
    for key, value in attributes.items():
        kind = "intValue" if isinstance(value, int) else "stringValue"
        values.append({"key": key, "value": {kind: str(value)}})
    unix_nano = "0" if time_in_attribute else str(int(at * 1e9))
    record = {"timeUnixNano": unix_nano, "attributes": values}
    body = {"resourceLogs": [{"scopeLogs": [{"logRecords": [record]}]}]}
    return post(session, "/v1/logs", body, {"x-agency-token": session.host_token})


def trajectory(session):
    from agency.observability.agwebui.trajectory import LiveSource

    session._agent_logger.flush()
    source = LiveSource(session.run_dir)
    source.refresh(force=True)
    return source.model


def actions(model, kind=None):
    return [a for a in model.actions.values() if kind is None or a["kind"] == kind]


def eventually(check, timeout=10):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return check()
        except AssertionError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)


class TerminalClient:
    def __init__(self, session, columns=100, rows=30):
        from websockets.sync.client import connect

        self.socket = connect(
            f"ws://127.0.0.1:{session.port}/terminal",
            additional_headers={"X-Agency-Token": session.host_token},
        )
        self.socket.send(json.dumps({"cols": columns, "rows": rows}))
        self.screen = _Screen(columns, rows)
        self.stream = pyte.ByteStream(self.screen)
        self.status = []

    def pump(self, timeout=0.2):
        from websockets.exceptions import ConnectionClosed

        try:
            while True:
                message = self.socket.recv(timeout=timeout)
                if isinstance(message, bytes):
                    self.stream.feed(message)
                else:
                    self.status.append(json.loads(message))
        except (TimeoutError, ConnectionClosed):
            return

    def text(self):
        return "\n".join(line.rstrip() for line in self.screen.display)

    def wait_for(self, needle, timeout=10):
        deadline = time.monotonic() + timeout
        while needle not in self.text():
            if time.monotonic() > deadline:
                raise AssertionError(f"{needle!r} not on screen:\n{self.text()}")
            self.pump()

    def type(self, data):
        self.socket.send(data)

    def close(self):
        self.socket.close()


def with_model_call(model):
    assert actions(model, "model")
    return model


def require(condition):
    assert condition


def rows(session, query):
    session._agent_logger.flush()
    with sqlite3.connect(session._agent_logger.db_path) as connection:
        return connection.execute(query).fetchall()


def process_state(pid):
    stat = Path(f"/proc/{pid}/stat").read_text()
    return stat[stat.rfind(")") + 2]
