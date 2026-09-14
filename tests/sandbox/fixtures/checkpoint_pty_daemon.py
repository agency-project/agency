"""Linux test driver using Agency's real manager, PTY controller and tracer.

Bash is a deterministic command dialect for this transport test. The separate
LLM test exercises the unmodified Codex adapter and Agent.run() end to end.
"""

import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from agency.configs.agconfig import agconfig
from agency.harness.adapters.agharness_backend import AdapterRuntime
from agency.harness.adapters.pty_session import PtyExecution
from agency.harness.adapters.pty_drivers import PtyDriver
from agency.harness.daemon import HarnessManager
from agency.harness.protocol import HarnessAttemptResult

cfg = agconfig()
cfg.sandbox.checkpoint_backend = "cow_zfs"
cfg.sandbox.checkpoint_fast_resume = True


class ShellDriver(PtyDriver):
    def __init__(self):
        pass

    name = "shell"
    argv = [
        "/bin/bash",
        "-c",
        "stty -echo; export PS1='AGENCY_READY>'; exec /bin/bash --noprofile --norc -i",
    ]
    env = dict(os.environ, PS1="AGENCY_READY>", TERM="xterm-256color")
    cwd = "/workspace"
    session_id = uuid.uuid4().hex
    root = Path("/workspace/pty-home")
    offset = 0
    marker = None

    def ready(self, handle):
        output = handle.terminal_output()
        if output != getattr(self, "last_output", None):
            print("PTY output:", repr(output), flush=True)
            self.last_output = output
        return "AGENCY_READY>" in output

    def events(self):
        output = execution.handle.terminal_output()
        match = re.search(
            r"\r?\nRESULT_" + str(self.marker) + r"=([^\r\n]+)\r?\n", output[self.offset :]
        )
        if match:
            self.offset = len(output)
            return [{"kind": "stop", "turn_id": self.marker, "text": match[1]}]
        return []

    def completed(self, event):
        return True

    def snapshot(self):
        return b"shell transport test; actual state remains in process memory"


class ShellExecution(PtyExecution):
    def _submit(self, text, label):
        self.validate_prompt(text)
        self.driver.marker = self._turn_id = uuid.uuid4().hex
        self._stop = None
        self._deadline = time.monotonic() + 20
        command = (
            text
            + "; /bin/true; "
            + (
                "printf '\\nRESULT_%s=%s|%s|%s|%s|%s\\n' "
                f'{self._turn_id} $$ "$SESSION_RAM" "$COUNT" "$PWD" "$RAM_TTY" >&9'
            )
        )
        self.handle.write_terminal(command.encode() + b"\r")


execution = None


def attempt(request):
    global execution
    admissions = []
    policy = SimpleNamespace(check=lambda *args: admissions.append(args[-1]) or True)
    runtime = AdapterRuntime(
        agconfig=cfg,
        model="shell-test",
        engine_name="pty-test",
        harness_base_url=manager._harness_api.base_url,
        token=request.attempt_token,
        syscall_policy=policy,
        register_control_handle=manager._register_control_handle,
        register_redirect=manager._register_redirect,
    )
    if execution is None:
        ShellDriver.root.mkdir(exist_ok=True)
        execution = ShellExecution(ShellDriver(), runtime)
        manager._live_execution = execution
    result = execution.run(request.prompt.user_content, keep_alive=True)
    if execution.handle is None:
        raise RuntimeError("The current invocation's policy did not observe /bin/true")
    return HarnessAttemptResult(
        ok=result.ok, final_text=result.final_text, session_id=result.session_id
    )


manager = HarnessManager(
    sys.argv[1],
    sys.argv[2],
    "pty-test",
    "native",
    agconfig=cfg,
    attempt_handler=attempt,
    harness_api_port=0,
)
manager.start()
threading.Event().wait()
