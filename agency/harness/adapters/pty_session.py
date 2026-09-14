"""Interactive external-harness attempts, never a headless/pipe fallback.

The daemon still owns process control and policy. Drivers own CLI dialects;
this runner serializes submission, redirection, completion, and retirement.
Native deliberately does not use this module.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from contextlib import nullcontext
from pathlib import PurePosixPath

from .agharness_backend import AttemptResult


MAX_SESSION_BYTES = 64 * 1024 * 1024


def restore_session(root, harness, session_id, blob):
    if (session_id is None) != (blob is None):
        raise ValueError("native resume requires both session id and session data")
    if blob is None:
        return
    if len(blob) > MAX_SESSION_BYTES:
        raise ValueError("native session exceeds size limit")
    bundle = json.loads(blob)
    if not isinstance(bundle, dict):
        raise ValueError("invalid native session bundle")
    if (bundle.get("version"), bundle.get("harness"), bundle.get("session_id")) != (
        1,
        harness,
        session_id,
    ):
        raise ValueError("native session bundle identity mismatch")
    files = bundle.get("files")
    if not isinstance(files, dict) or not files or len(files) > 10000:
        raise ValueError("invalid native session file inventory")
    size = 0
    decoded = []
    for name, encoded in files.items():
        path = PurePosixPath(name)
        if str(path) != name or not session_file_allowed(harness, path):
            raise ValueError(f"invalid native session path: {name}")
        data = base64.b64decode(encoded, validate=True)
        size += len(data)
        if size > MAX_SESSION_BYTES:
            raise ValueError("native session exceeds size limit")
        target = root / path
        if not target.resolve().is_relative_to(root.resolve()) or target.is_symlink():
            raise ValueError("unsafe native session destination")
        decoded.append((target, data))
    # Validate the entire envelope before writing any files.
    for target, data in decoded:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def session_file_allowed(harness, path):
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return False
    if harness in {"codex", "grok"}:
        return path.parts[0] == "sessions" and path.suffix in {".jsonl", ".json"}
    if harness == "kimi":
        # The index names each session directory; both are needed to resume.
        if str(path) == "session_index.jsonl":
            return True
        return path.parts[0] == "sessions" and path.suffix in {".jsonl", ".json"}
    return harness == "opencode" and str(path) == "data/opencode/opencode.db"


def snapshot_session(root, harness, session_id):
    files = {}
    size = 0
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not session_file_allowed(harness, PurePosixPath(relative.as_posix())):
            continue
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError("unsafe native session file")
        if path.stat().st_size > MAX_SESSION_BYTES - size:
            raise ValueError("native session exceeds size limit")
        data = path.read_bytes()
        size += len(data)
        if size > MAX_SESSION_BYTES:
            raise ValueError("native session exceeds size limit")
        files[relative.as_posix()] = base64.b64encode(data).decode()
    if not files:
        raise RuntimeError(f"{harness} did not persist a native session")
    blob = json.dumps(
        {"version": 1, "harness": harness, "session_id": session_id, "files": files}
    ).encode()
    if len(blob) > MAX_SESSION_BYTES:
        raise ValueError("native session exceeds size limit")
    return blob


async def stream_response(router, token, context, model, formatter):
    """A TUI interrupt must close the upstream request, not just its HTTP socket."""
    import anyio

    stream = router.dispatch_stream_async(token, context)
    try:
        async for item in stream:
            if item.get("type") == "done":
                for frame in formatter([item], model):
                    yield frame
    finally:
        with anyio.CancelScope(shield=True):
            await stream.aclose()


class PtyExecution:
    INPUT_TIMEOUT = 60.0
    START_TIMEOUT = 45.0
    ATTEMPT_TIMEOUT = 600.0

    def __init__(self, driver, runtime):
        self.driver = driver
        self.runtime = runtime
        self.handle = None
        self._lock = threading.RLock()
        self._active = False
        self._expected_prompt = None
        self._turn_id = None
        self._turn_started = False
        self._acknowledged = False
        self._stop = None
        self._interrupted = False
        self._failure = None
        self._deadline = 0
        self._last_activity_generation = None
        self.INPUT_TIMEOUT = driver.INPUT_TIMEOUT
        self.START_TIMEOUT = driver.START_TIMEOUT
        self.ATTEMPT_TIMEOUT = driver.ATTEMPT_TIMEOUT

    @staticmethod
    def validate_prompt(prompt):
        if not prompt.strip():
            raise ValueError("native prompt must not be empty")
        if any((ord(c) < 32 and c not in "\n\t") or ord(c) == 127 for c in prompt):
            raise ValueError("native prompt contains terminal control characters")

    def _poll(self):
        for event in self.driver.events():
            if event.get("kind") == "submit" and self.driver.prompt_matches(
                event.get("prompt"), self._expected_prompt
            ):
                # A CLI that owns its turn identity reports it here; a driver
                # that mints its own has already set it in _submit.
                if event.get("turn_id"):
                    self._turn_id = event["turn_id"]
                self._acknowledged = True
            if self._turn_id is None:
                # Nothing can complete or be interrupted before a turn exists,
                # but a CLI that fails during startup must not be waited out to
                # the deadline. Once any turn has run, a turn-less event is
                # stale and cannot fail its replacement.
                if not self._turn_started and event.get("kind") == "error":
                    self._failure = event.get("error", "native turn failed")
                continue
            if event.get("turn_id") != self._turn_id:
                continue
            if event["kind"] == "stop":
                self._stop = event
            elif event["kind"] == "interrupt":
                self._interrupted = True
            elif event["kind"] == "error":
                self._failure = event.get("error", "native turn failed")

    def _check_alive(self):
        if self._failure:
            raise RuntimeError(self._failure)
        if self.handle.returncode is not None:
            raise RuntimeError(f"{self.driver.name} process exited ({self.handle.returncode})")

    def _wait_until(self, predicate, description, timeout=None):
        deadline = time.monotonic() + (self.INPUT_TIMEOUT if timeout is None else timeout)
        last_poll = time.monotonic()
        while True:
            now = time.monotonic()
            if self.handle.is_paused():
                deadline += now - last_poll
            last_poll = now
            self._poll()
            if predicate():
                return
            self._check_alive()
            if now >= deadline:
                raise RuntimeError(f"{self.driver.name} timed out waiting for {description}")
            time.sleep(0.025)

    def _submit(self, text, label):
        self.validate_prompt(text)
        self._turn_started = True
        self._expected_prompt = f"{self.driver.submission_marker(label)}\n{text}"
        # A driver that fences turns itself returns the identity it just
        # committed; the rest learn it from the CLI's own acknowledgment.
        self._turn_id = self.driver.begin_turn(self._expected_prompt)
        self._acknowledged = False
        self._stop = None
        self._interrupted = False
        self.handle.write_terminal(b"\x1b[200~" + self._expected_prompt.encode() + b"\x1b[201~")
        self.handle.write_terminal(b"\r")
        self._wait_until(lambda: self._acknowledged, "native prompt acknowledgment")
        self._deadline = time.monotonic() + self.ATTEMPT_TIMEOUT

    def redirect(self, text):
        with self._lock:
            if not self._active or self.handle.is_paused():
                return False
            self._poll()
            if self._stop is not None:
                return False
            try:
                self.validate_prompt(text)
            except ValueError:
                return False
            try:
                self._check_alive()
                self._interrupted = False
                self.driver.begin_interrupt(self.handle)
                self.handle.write_terminal(self.driver.interrupt_key)
                if self.driver.confirm_interrupt:
                    self._wait_until(
                        lambda: (
                            self.driver.interrupt_pending(self.handle) or self._stop is not None
                        ),
                        "native interrupt confirmation",
                    )
                    if self._stop is not None:
                        return False
                    self.handle.write_terminal(self.driver.interrupt_key)
                self._wait_until(
                    lambda: (
                        self._interrupted
                        or self._stop is not None
                        or self.driver.interrupted(self.handle)
                    ),
                    "native interruption",
                )
                if self._stop is not None:
                    return False
                self.driver.clear_input(self.handle, self._wait_until)
                self._wait_until(lambda: self.driver.ready(self.handle), "empty native input")
                self._submit(text, "redirect")
                return True
            except (OSError, RuntimeError) as exc:
                self._failure = str(exc)
                self._active = False
                # Reap before reporting failure: a queued fallback must not race
                # a partially accepted redirect in a still-running CLI.
                self.driver.reap(self.handle)
                return False

    def run(self, prompt):
        from ..agharness import cleanup_config_home
        from ..ptrace.supervisor import agProxyPtrace

        phase = getattr(self.driver, "profile_span", lambda name: nullcontext())
        try:
            self.validate_prompt(prompt)
            with phase("harness:launch"):
                self.handle = agProxyPtrace(self.runtime.agconfig, allow_initial_exec=True).launch(
                    self.driver.argv,
                    self.driver.env,
                    cwd=self.driver.cwd,
                    pty_size=(120, 36),
                    policy=self.runtime.syscall_policy,
                    ag=None,
                )
            self.runtime.register_control_handle(self.handle)
            self.runtime.register_redirect(self.redirect)
            with phase("harness:startup_ready"):
                self._wait_until(
                    lambda: self.driver.ready(self.handle), "startup input", self.START_TIMEOUT
                )
            with phase("harness:submit"):
                self._submit(prompt, "run")
            with self._lock:
                self._active = True
            last_poll = time.monotonic()
            # Explicit envelope, not a claim of idle CPU: the CLI may be
            # working while this controller awaits its terminal event.
            with phase("harness:await_cli"):
                while True:
                    with self._lock:
                        now = time.monotonic()
                        if self.handle.is_paused():
                            self._deadline += now - last_poll
                        elif self.driver.activity_extends_deadline:
                            # Terminal output is the only progress signal some
                            # CLIs emit during a long tool call.
                            generation = self.handle.terminal_screen()[3]
                            if generation != self._last_activity_generation:
                                self._last_activity_generation = generation
                                self._deadline = now + self.ATTEMPT_TIMEOUT
                        last_poll = now
                        self._poll()
                        self._check_alive()
                        if self._stop is not None and self.driver.completed(self._stop):
                            self._active = False
                            with phase("harness:snapshot"):
                                blob = self.driver.snapshot()
                            return AttemptResult(
                                ok=True,
                                final_text=self._stop["text"],
                                session_id=self.driver.session_id,
                                session_blob=blob,
                                input_tokens=self._stop.get("input_tokens", 0),
                                output_tokens=self._stop.get("output_tokens", 0),
                            )
                        if now > self._deadline:
                            raise RuntimeError(f"{self.driver.name} attempt timed out")
                    time.sleep(0.025)
        finally:
            try:
                with self._lock:
                    self._active = False
                    if self.handle is not None:
                        with phase("harness:retire"):
                            self.handle.close()
            finally:
                cleanup_config_home(self.driver.root)
