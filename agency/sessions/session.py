"""A normal interactive coding-agent CLI, observed and controlled by Agency.

`agency claude` / `agency codex` inherit the user's terminal and native CLI
settings. The shared tracer, collectors and control service stay outside the
CLI workload scope. An opt-in gateway archives native model traffic without
a workflow or Agency sandbox. An explicit browser mode retains the existing
PTY and terminal page. Both modes reuse `agwebui.run()` for the dashboard,
profiler sampling and control relay.

What Agency adds to the CLI, and why each is needed:

* Lifecycle hooks (merged with the user's own). They give live tool starts
  and ends, permission prompts and turn boundaries, recorded through the same
  HostInteractionServer that external harnesses report to. Optional Agency
  policy can deny a tool; an allow returns no forced permission decision so
  the CLI's own permission flow still applies.
* OpenTelemetry log export to this process. No hook reports model requests;
  the CLIs' own telemetry carries each request's measured timing and tokens,
  delivered in batches shortly after the request ends. With --capture, native
  HTTP requests/responses supply model records instead; model OTel is ignored
  to avoid duplicates, while tool and lifecycle telemetry is retained.
* Its own systemd scope (cgroup). The profiler samples that scope's CPU,
  memory and processes, pause/resume freezes it, and stop kills everything
  the session started, including background processes outside its process
  group.
* Optional shared ptrace supervision follows local child processes and applies
  selected syscall policy without reading or replacing terminal input/output.

Each CLI's module (claude_code.py, codex.py) supplies how to launch it and
how to read its telemetry; everything else lives here. Linux only.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import os
import re
import runpy
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from collections import deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

# Module level, not inside _build_app(): with postponed annotations FastAPI
# resolves handler parameter types against this module's globals.
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

# Read by the web UI server (agwebui/server.py) to relay its terminal page.
SESSION_FILE = "terminal_session.json"
SESSION_INFO_FILE = "interactive_session.json"
TOKEN_VARIABLE = "AGENCY_SESSION_TOKEN"
_SCOPED_VARIABLE = "AGENCY_SESSION_SCOPED"


def hook_command() -> str:
    """Stable command hook; per-session credentials stay in the environment."""
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).with_name('_hook_relay.py')))}"


# Set by a coding-agent process for its own children. When Agency itself is
# launched from inside one they describe that *other* session, and the CLI
# started here would otherwise believe it is nested inside it.
_PARENT_SESSION_VARIABLES = (
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_SESSION_ATTENDED",
    "CLAUDE_CODE_HOST_SESSION_ID",
    "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_CODE_EXECPATH",
    "CLAUDE_PID",
    # Desktop Codex injects its current chat's tool pipe and identity. A new
    # interactive CLI must create its own tools/session, rather than inherit
    # this process's desktop tool routing and permission profile.
    "CODEX_APP_TOOLS_PIPE_PATH",
    "CODEX_APP_RUNTIME_CLI_PATH",
    "CODEX_SESSION_ID",
    "CODEX_THREAD_ID",
    "CODEX_TOOL_CALL_ID",
    "CODEX_PERMISSION_PROFILE",
    "CODEX_TASK_WORKSPACE_VERIFYING_IDENTITY",
    "CODEX_INTERNAL_ORIGINATOR_OVERRIDE",
    "AGENCY_SESSION_ENFORCE_POLICY",
)

# PreToolUse blocks the tool until answered. Keep a wedged Agency from
# stalling the user's session for the CLIs' 600 s default.
HOOK_TIMEOUT_S = 5
# Same bound the harness hook applies to tool results it forwards.
_MAX_RESULT_CHARS = 64 * 1024
# A tool that a user rejects or interrupts may fire no completion hook; its
# fate arrives with the next telemetry batch. After a turn ends, wait this
# long for that evidence before recording the call as unreported.
_UNREPORTED_TOOL_GRACE_S = 5.0
_STOP_GRACE_S = 5.0


# A subagent's run is over in one of these states; Agency's controls then
# drop it, as they drop a finished agent.
_FINISHED_STATES = {"completed", "cancelled", "agent_exit"}


class _Actor:
    """One trajectory column: the CLI's main agent or a subagent it spawned.

    Each has its own logger and telemetry server, as each Agency agent does,
    so the trajectory shows a subagent as its own agent with a recorded
    parent rather than folding its calls into the parent's column."""

    def __init__(self, name: str, logger, host) -> None:
        self.name = name
        self.logger = logger
        self.host = host
        self.state = "agent_idle"
        self.subagent_type: "str | None" = None
        self.started_at = time.time()
        self.stopped_at: "float | None" = None


class _SubagentControls:
    """Relay target for a subagent. It runs inside its parent CLI's process,
    so there is nothing of its own to pause or stop."""

    def __init__(self, parent: str) -> None:
        self.parent = parent

    def _refuse(self, *_args, **_kwargs) -> None:
        raise ValueError(
            f"This subagent runs inside {self.parent}'s process; pause or stop {self.parent}."
        )

    pause = resume = stop = _refuse


class CliSession:
    # Set by each CLI's subclass.
    agent_name = ""  # actor name in the trajectory and relay
    harness = ""  # Agency harness identifier
    label = ""  # shown to people
    engine_coverage: "dict[str, str]" = {}

    def __init__(
        self,
        *,
        cwd: Path,
        binary: str,
        cli_args: "list[str]",
        run_dir: Path,
        ui_port: int,
        use_scope: bool,
        capture=None,
        terminal_mode: str = "native",
        trace: bool = True,
        policy=None,
        ptrace_config=None,
    ) -> None:
        self.cwd = cwd
        self.binary = binary
        self.cli_args = list(cli_args)
        self.run_dir = run_dir
        self.ui_port = ui_port
        self.use_scope = use_scope
        self.capture = capture
        if capture is not None and capture.native and self.harness != capture.harness:
            raise ValueError("native capture configuration does not match this CLI")
        if terminal_mode not in {"native", "browser"}:
            raise ValueError("terminal_mode must be native or browser")
        from ..agpolicy import agpolicy
        from ..configs.agconfig import agconfig, ptraceconfig

        self.terminal_mode = terminal_mode
        self.trace = trace
        self.policy = policy if policy is not None else agpolicy()
        self.runtime_config = agconfig(ptrace_config or ptraceconfig())
        self.runtime_config.ptrace.syscalls = tuple(
            dict.fromkeys(
                (
                    *self.runtime_config.ptrace.syscalls,
                    *(self.policy.syscall_hooks or {}),
                )
            )
        )
        if (self.policy.syscall_hooks or self.policy.default_to_deny) and not trace:
            raise ValueError("syscall policy requires tracing")
        if terminal_mode == "browser" and trace:
            raise ValueError("tracing requires the native terminal mode")
        self._capture_gateway = None
        if capture is not None:
            from .capture import CaptureGateway

            if capture.native and self.harness == "claude_code":
                from .capture_proxy import ClaudeCaptureGateway

                self._capture_gateway = ClaudeCaptureGateway(self, capture)
            else:
                self._capture_gateway = CaptureGateway(self, capture)
        self.session_id = str(uuid.uuid4())
        self.host_token = secrets.token_urlsafe(32)
        self.ui_token = secrets.token_urlsafe(32)
        self.scope_unit = f"agency-{self.agent_name}-{self.session_id[:8]}"
        self.scope_dir: "Path | None" = None
        self.port: "int | None" = None
        self.terminal = None
        self._watcher: "threading.Thread | None" = None
        self.ended = threading.Event()
        self._spawn_lock = threading.Lock()
        self._telemetry_lock = threading.Lock()
        # CLI tool-call id -> (the column that made the call, Agency call_id)
        self._pending_tools: "dict[str, tuple[_Actor, str]]" = {}
        self._waiting_for_input: "set[str]" = set()
        self._main: "_Actor | None" = None
        self._subagents: "dict[str, _Actor]" = {}  # the CLI's agent_id -> column
        self._subagents_lock = threading.Lock()
        # Labels of spawn calls still waiting for their SubagentStart, in order.
        self._spawn_labels: "deque[str]" = deque()
        self._paused = False
        self._server = None
        self._server_thread = None
        self._agent_logger = None
        self._global_logger = None
        self._host = None

    # -- What each CLI supplies ------------------------------------------------

    def _cli_argv(self) -> "list[str]":
        """The CLI's command line, once the endpoint port is known."""
        raise NotImplementedError

    def _cli_environment(self, env: dict) -> None:
        """Add the CLI's telemetry settings to *env* (the user's own)."""

    def _prepare(self) -> None:
        """Write any per-session files the CLI reads, once the port is known."""

    def handle_log_records(self, body: dict) -> None:
        """Map one OTLP/JSON log export from the CLI onto Agency events."""
        raise NotImplementedError

    # -- Lifecycle -----------------------------------------------------------

    def run(self) -> None:
        """agwebui.run() workload: lasts exactly as long as the CLI."""
        try:
            self.start()
            if self.terminal_mode == "native":
                print(
                    f"[agency] Dashboard: http://127.0.0.1:{self.ui_port}/profiler?run=live",
                    flush=True,
                )
                self.launch_native()
            else:
                print(
                    f"[agency] {self.label} session ready: http://127.0.0.1:{self.ui_port}/terminal"
                    f"#token={self.ui_token}",
                    flush=True,
                )
                print(
                    f"[agency] {self.label} starts when the terminal page first connects.",
                    flush=True,
                )
            while not self.ended.wait(0.05 if self.terminal_mode == "native" else 0.5):
                if self.terminal_mode == "native" and self.terminal is not None:
                    self.terminal.poll_job_control()
            if getattr(self, "_process_error", None) is not None:
                raise RuntimeError("CLI supervision failed") from self._process_error
        finally:
            self.close()

    def start(self, global_logger=None) -> None:
        """*global_logger* defaults to the orchestrator's run-wide logger."""
        from ..observability.agwebui.relay import register_standalone_target
        from ..observability.profiler import agprof
        from ..orchestrator import get_orchestrator

        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.capture is not None:
            # The SQLite projection contains the same complete context as
            # the private wire archive, including opaque provider state.
            self.run_dir.chmod(0o700)
        self._global_logger = global_logger or get_orchestrator().data_logger
        self._main = self._open_actor(
            self.agent_name,
            f"{self.agent_name}_data.sqlite3",
            {
                "cwd": str(self.cwd),
                "interactive_session_id": self.session_id,
                "terminal_mode": self.terminal_mode,
                "trace": self.trace,
            },
        )
        self._agent_logger = self._main.logger
        self._host = self._main.host
        self._set_state("agent_idle")
        coverage = dict(self.engine_coverage)
        coverage.update(
            terminal=self.terminal_mode,
            syscalls="ptrace_selected" if self.trace else "unavailable",
            processes="ptrace_lifecycle" if self.trace else "sampled",
        )
        if self.capture is not None:
            coverage.update(
                llm="wire_capture",
                reason="Native model requests and responses archived through the opt-in capture gateway; process/tool coverage reported separately.",
            )
        agprof.register_engine(self.harness, **coverage)
        self._start_server()
        if self.capture is not None and self.capture.native and self.harness == "claude_code":
            self._capture_gateway.start()
        self._prepare()
        if self.terminal_mode == "browser":
            self._write_private_json(
                SESSION_FILE,
                {
                    "host_port": self.port,
                    "host_token": self.host_token,
                    "ui_token": self.ui_token,
                    "label": self.label,
                    "session_id": self.session_id,
                    "cwd": str(self.cwd),
                    "capture": {
                        "model": self.capture.model,
                        "authentication": "native" if self.capture.native else "litellm",
                        "gateway": getattr(
                            self._capture_gateway,
                            "endpoint",
                            f"http://127.0.0.1:{self.port}/capture",
                        ),
                    }
                    if self.capture is not None
                    else None,
                },
            )
        register_standalone_target(self.agent_name, self)
        self._write_session_info()

    def _write_session_info(self) -> None:
        """Durable, credential-free identity for the dashboard and run export."""
        self._write_private_json(
            SESSION_INFO_FILE,
            {
                "session_id": self.session_id,
                "cli": self.agent_name,
                "terminal_mode": self.terminal_mode,
                "trace": self.trace,
                "runtime_pid": os.getpid(),
                "cli_pid": getattr(self.terminal, "pid", None),
                "scope": self.scope_unit if self.scope_dir is not None else None,
                "returncode": getattr(self.terminal, "returncode", None),
                "capture_model": self.capture.model if self.capture else None,
                "capture_authentication": ("native" if self.capture.native else "litellm")
                if self.capture
                else None,
            },
        )

    def close(self) -> None:
        from ..observability.agwebui.relay import unregister_standalone_target

        unregister_standalone_target(self.agent_name)
        for actor in self._subagent_actors():
            unregister_standalone_target(actor.name)
        terminal = self.terminal
        if terminal is not None and not terminal.exited.is_set():
            self._kill_tree()
            terminal.wait(10)
        if self._watcher is not None:
            # It records the exit; the logger must still be open for that.
            self._watcher.join(timeout=10)
        self._finish_scope()
        if self.capture is not None and self.capture.native and self.harness == "claude_code":
            self._capture_gateway.close()
        # The tokens stop meaning anything once the server is gone; do not
        # leave them lying in the run directory.
        (self.run_dir / SESSION_FILE).unlink(missing_ok=True)
        if self._server is not None:
            self._server.should_exit = True
            if self._server_thread is not None and self._server_thread.ident is not None:
                self._server_thread.join(timeout=5)
        for actor in [*self._subagent_actors(), self._main]:
            if actor is None:
                continue
            actor.host.finalize_profile()
            actor.logger.flush()
            actor.logger.stop()
        self.ended.set()

    def _open_actor(self, name: str, db_name: str, created: dict, parent=None) -> _Actor:
        """A new column: its own logger and telemetry server, registered with
        the run (and with its parent, for a subagent)."""
        from ..configs.agconfig import agconfig, dataloggerconfig
        from ..engine.host_servers.host_interaction_server import HostInteractionServer
        from ..observability.agdatalogger import agDataLogger

        logger = agDataLogger(
            agconfig(dataloggerconfig(db_path=str(self.run_dir / db_name))),
            default_name=name,
            default_object="agent",
        )
        logger.start()
        attributes = {"harness": self.harness, "agency.interactive_session_id": self.session_id}
        if parent is not None:
            attributes["agency.parent_agent_id"] = parent
        host = HostInteractionServer(
            SimpleNamespace(policy=self.policy),
            logger,
            name,
            profile_attributes=attributes,
            is_cancelled=self.ended.is_set,
        )
        actor = _Actor(name, logger, host)
        logger.record_event(
            "agent_created",
            {
                "agname": name,
                "harness": self.harness,
                **({"parent_agname": parent} if parent else {}),
                **created,
            },
            term_message=f"[{name}] CREATED interactive {self.label} "
            + ("subagent" if parent else "session"),
            print_to_terminal=False,
            flush=True,
        )
        self._register(actor, parent=parent)
        return actor

    def _register(self, actor: _Actor, model: "str | None" = None, parent=None) -> None:
        payload = {
            "db_path": str(Path(actor.logger.db_path).resolve()),
            "team": None,
            "harness": self.harness,
        }
        if parent:
            payload["parent_agname"] = parent
        if model:
            payload["llm_config"] = {"model": model}
        self._global_logger.record_event(
            "agent_registered",
            payload,
            name=actor.name,
            object="agent",
            update_latest_snapshot=True,
            flush=True,
        )

    def _subagent_actors(self) -> "list[_Actor]":
        with self._subagents_lock:
            return list(self._subagents.values())

    def _subagent(self, agent_id: str, agent_type: "str | None") -> _Actor:
        """The column of the CLI's subagent *agent_id*, opened on first sight.

        Normally that is its SubagentStart hook, but a tool hook from inside
        the subagent may arrive first."""
        from ..observability.agwebui.relay import register_standalone_target

        with self._subagents_lock:
            actor = self._subagents.get(agent_id)
            if actor is not None:
                return actor
            label = self._spawn_labels.popleft() if self._spawn_labels else None
            name = f"{self.agent_name}/{agent_type or 'subagent'}-{agent_id[-8:]}"
            actor = self._open_actor(
                name,
                f"{self.agent_name}_subagent_{re.sub(r'[^A-Za-z0-9_-]', '_', agent_id)}"
                "_data.sqlite3",
                {"subagent_id": agent_id, "subagent_type": agent_type},
                parent=self.agent_name,
            )
            actor.subagent_type = agent_type
            self._subagents[agent_id] = actor
        self._main.logger.record_event(
            "delegation",
            {"to": name, "task": label or agent_type or "subagent", "subagent_id": agent_id},
            flush=True,
        )
        register_standalone_target(name, _SubagentControls(self.agent_name))
        self._set_state("running_harness", actor)
        return actor

    def _actor_for(self, payload: dict) -> _Actor:
        """Hooks fired inside a subagent carry its agent_id; others are the
        main agent's."""
        agent_id = payload.get("agent_id")
        if not agent_id:
            return self._main
        return self._subagent(str(agent_id), payload.get("agent_type"))

    def _known_subagent(self, agent_id) -> _Actor:
        """A subagent already seen through hooks, else the main agent."""
        with self._subagents_lock:
            return self._subagents.get(agent_id) or self._main

    def _write_private_json(self, name: str, value: dict) -> None:
        path = self.run_dir / name
        temporary = path.with_name(path.name + ".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream)
        temporary.replace(path)

    # -- The CLI process -----------------------------------------------------

    def launch_native(self) -> None:
        """Use the caller's actual terminal; tracing owns no extra PTY."""
        from .process import NativeProcess
        from ..engine.host_servers.interaction_policy import InteractionSyscallPolicy

        with self._spawn_lock:
            if self.terminal is not None:
                raise RuntimeError("session process already started")
            self.terminal = NativeProcess(
                self._cli_argv(),
                self._environment(),
                str(self.cwd),
                trace=self.trace,
                config=self.runtime_config,
                policy=InteractionSyscallPolicy(self._host),
                on_spawn=self._create_scope if self.use_scope else None,
                agent=SimpleNamespace(agname=self.agent_name),
            )
            self._watcher = threading.Thread(
                target=self._watch_process, name="cli-session-exit", daemon=True
            )
            self._write_session_info()
            self._watcher.start()

    def _create_scope(self, pid: int) -> None:
        """Move the stopped launch child into its own transient scope."""
        subprocess.run(
            [
                "busctl",
                "--user",
                "call",
                "org.freedesktop.systemd1",
                "/org/freedesktop/systemd1",
                "org.freedesktop.systemd1.Manager",
                "StartTransientUnit",
                "ssa(sv)a(sa(sv))",
                f"{self.scope_unit}.scope",
                "fail",
                "2",
                "PIDs",
                "au",
                "1",
                str(pid),
                "CollectMode",
                "s",
                "inactive-or-failed",
                "0",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        self._register_scope(pid)

    def attach_terminal(self, columns: int, rows: int, listener):
        """Start the CLI on first attach, at the first viewer's size, so its
        startup terminal queries reach a real terminal that can answer them."""
        if self.terminal_mode != "browser":
            raise ValueError("This CLI uses the local terminal; open the dashboard to inspect it.")
        from .terminal import Terminal

        with self._spawn_lock:
            if self.terminal is None:
                argv = self._cli_argv()
                if self.use_scope:
                    argv = [
                        "systemd-run",
                        "--user",
                        "--scope",
                        "--collect",
                        "--quiet",
                        f"--unit={self.scope_unit}",
                        "--",
                        *argv,
                    ]
                self.terminal = Terminal(argv, self._environment(), str(self.cwd), columns, rows)
                self._watcher = threading.Thread(
                    target=self._watch_process, name="cli-session-exit", daemon=True
                )
                self._watcher.start()
            elif not self.terminal.exited.is_set():
                self.terminal.resize(columns, rows)
            return self.terminal, self.terminal.attach(listener)

    def _environment(self) -> dict:
        env = dict(os.environ)
        for name in (*_PARENT_SESSION_VARIABLES, _SCOPED_VARIABLE):
            env.pop(name, None)
        if self.terminal_mode == "browser":
            env.update(TERM="xterm-256color", COLORTERM="truecolor")
        env[TOKEN_VARIABLE] = self.host_token
        env["AGENCY_SESSION_HOOK_URL"] = f"http://127.0.0.1:{self.port}/hooks"
        if self.policy.tool_hooks or self.policy.default_to_deny:
            env["AGENCY_SESSION_ENFORCE_POLICY"] = "1"
        self._cli_environment(env)
        if self.capture is not None and self.capture.key_env:
            # The upstream credential belongs only to the gateway process.
            env.pop(self.capture.key_env, None)
        return env

    def _watch_process(self) -> None:
        terminal = self.terminal
        returncode = None
        try:
            if self.use_scope and self.terminal_mode == "browser":
                self._register_scope(terminal.pid)
            returncode = terminal.wait()
        except Exception as exc:
            self._process_error = exc
        finally:
            try:
                self._write_session_info()
                self._agent_logger.record_event(
                    "agent_state",
                    {
                        "state": "agent_exit",
                        "returncode": returncode,
                        **(
                            {"error": type(self._process_error).__name__}
                            if getattr(self, "_process_error", None) is not None
                            else {}
                        ),
                    },
                    update_latest_snapshot=True,
                    term_message=f"[{self.agent_name}] EXITED ({returncode})",
                    print_to_terminal=False,
                    flush=True,
                )
                self._finish_subagents("cancelled")
            except Exception as exc:
                self._process_error = exc
            finally:
                try:
                    # Retire background work even if waiting/logging failed.
                    self._kill_tree()
                except Exception as exc:
                    self._process_error = exc
                finally:
                    self.ended.set()

    def _register_scope(self, pid: int) -> None:
        """Find the scope systemd moved the CLI into, then start sampling it."""
        from ..observability.profiler import agprof

        suffix = f"/{self.scope_unit}.scope"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                line = Path(f"/proc/{pid}/cgroup").read_text().strip()
            except OSError:
                return
            relative = line.split("::", 1)[-1]
            if relative.endswith(suffix):
                self.scope_dir = Path("/sys/fs/cgroup") / relative.lstrip("/")
                break
            time.sleep(0.05)
        else:
            print(f"[agency] WARNING: {self.scope_unit} never appeared; pause/stop use signals")
            return
        try:
            agprof.container_started(self.scope_unit, str(self.scope_dir), host_network=True)
        except RuntimeError as exc:
            print(f"[agency] WARNING: profiler cannot sample the session scope: {exc}")
        # The trajectory links an agent to its resource series by this label.
        self._set_state(self._main.state, sandbox=self.scope_unit)

    def _finish_scope(self) -> None:
        from ..observability.profiler import agprof

        agprof.container_stopped(self.scope_unit)

    # -- Controls (the relay's standalone-target interface) ------------------

    def is_paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        if self._paused or not self._process_running():
            return
        self._pause_via_process = not self._write_scope("cgroup.freeze", "1")
        if self._pause_via_process:
            self.terminal.signal_group(signal.SIGSTOP)
        self._paused = True
        self._agent_logger.record_event("agent_paused", {}, update_latest_snapshot=True, flush=True)

    def resume(self) -> None:
        if not self._paused:
            return
        thawed = self._write_scope("cgroup.freeze", "0")
        if getattr(self, "_pause_via_process", False) and self.terminal is not None:
            self.terminal.signal_group(signal.SIGCONT)
        elif not thawed:
            # SIGCONT cannot thaw a frozen cgroup. Keep the paused state so
            # the user can retry rather than falsely reporting a resume.
            raise RuntimeError("Cannot unfreeze the CLI workload scope")
        self._pause_via_process = False
        self._paused = False
        self._agent_logger.record_event(
            "agent_resumed", {}, update_latest_snapshot=True, flush=True
        )

    def stop(self, force: bool = False) -> None:
        self._agent_logger.record_event(
            "agent_stop_requested", {"force": force}, update_latest_snapshot=True, flush=True
        )
        if not self._process_running():
            self.ended.set()
            return
        if force:
            self._kill_tree()
            return
        # A frozen process cannot handle SIGTERM until it is thawed.
        self.resume()
        self.terminal.signal_group(signal.SIGTERM)
        timer = threading.Timer(_STOP_GRACE_S, self._kill_tree)
        timer.daemon = True
        timer.start()

    def interrupt(self) -> None:
        """What pressing Escape in the CLI's own terminal does."""
        if self.terminal_mode != "browser":
            raise ValueError(
                "Interrupt the CLI from its local terminal; Agency cannot inject keys into it."
            )
        if self._process_running():
            self.terminal.write(b"\x1b")

    def _process_running(self) -> bool:
        return self.terminal is not None and not self.terminal.exited.is_set()

    def _kill_tree(self) -> None:
        if self._write_scope("cgroup.kill", "1"):
            return
        if self.terminal is not None:
            self.terminal.signal_group(signal.SIGKILL)

    def _write_scope(self, name: str, value: str) -> bool:
        if self.scope_dir is None:
            return False
        try:
            (self.scope_dir / name).write_text(value)
            return True
        except OSError:
            return False

    # -- Telemetry -------------------------------------------------------------

    def _set_state(self, state: str, actor: "_Actor | None" = None, **extra) -> None:
        actor = actor or self._main
        actor.state = state
        actor.logger.record_event(
            "agent_state", {"state": state, **extra}, update_latest_snapshot=True, flush=True
        )

    def _await_input(self, message: str, actor: _Actor) -> None:
        with self._telemetry_lock:
            self._waiting_for_input.add(actor.name)
        actor.logger.record_event(
            "input_required", {"message": message}, update_latest_snapshot=True, flush=True
        )
        self._set_state("waiting_input", actor)

    def _input_resolved(self, actor: "_Actor | None" = None) -> None:
        """Close *actor*'s pending input request, or every one when None."""
        actors = [actor] if actor is not None else [self._main, *self._subagent_actors()]
        for candidate in actors:
            with self._telemetry_lock:
                if candidate.name not in self._waiting_for_input:
                    continue
                self._waiting_for_input.discard(candidate.name)
            candidate.logger.record_event("input_resolved", {}, flush=True)

    def _finish_subagents(self, state: str) -> None:
        for actor in self._subagent_actors():
            if actor.state not in _FINISHED_STATES:
                actor.stopped_at = time.time()
                self._input_resolved(actor)
                self._set_state(state, actor)

    def _subagent_running_at(self, agent_type, at: float) -> "_Actor | None":
        """The one subagent of *agent_type* that was running at *at*, if
        exactly one was. For telemetry that names a subagent's type but not
        which subagent; a second candidate makes the answer unknowable."""
        # Hook arrival and the CLI's own clock reading differ slightly.
        slack = 1.0
        candidates = [
            actor
            for actor in self._subagent_actors()
            if actor.subagent_type == agent_type
            and actor.started_at - slack <= at
            and (actor.stopped_at is None or at <= actor.stopped_at + slack)
        ]
        return candidates[0] if len(candidates) == 1 else None

    def handle_hook(self, payload: dict) -> "dict | None":
        """Claude Code and Codex share these hook names and field names; an
        event one CLI never sends is simply never matched."""
        event = payload.get("hook_event_name")
        actor = self._actor_for(payload)
        if event == "SessionStart":
            if actor is self._main:
                self._register(self._main, model=payload.get("model"))
                self._set_state("agent_idle")
        elif event == "UserPromptSubmit":
            self._input_resolved(actor)
            actor.logger.record_event(
                "user_prompt",
                {
                    "prompt": str(payload.get("prompt", ""))[:_MAX_RESULT_CHARS],
                    "prompt_id": payload.get("prompt_id") or payload.get("turn_id"),
                },
                flush=True,
            )
            self._set_state("running_harness", actor)
        elif event == "PreToolUse":
            admission = self._tool_started(actor, payload)
            if not admission["allowed"]:
                from ..harness._native_hooks import decision_to_hook_response

                return decision_to_hook_response((False, admission["reason"]))
        elif event == "PermissionRequest":
            self._await_input(
                f"{self.label} asks permission to use {payload.get('tool_name')}", actor
            )
        elif event == "PostToolUse":
            self._tool_finished(
                payload.get("tool_use_id"),
                result=payload.get("tool_response"),
                duration_ms=payload.get("duration_ms"),
            )
        elif event == "PostToolUseFailure":
            error = str(payload.get("error") or "Tool failed")
            if payload.get("is_interrupt"):
                error = f"Interrupted: {error}"
            self._tool_finished(
                payload.get("tool_use_id"), error=error, duration_ms=payload.get("duration_ms")
            )
        elif event == "Notification":
            kind = payload.get("notification_type")
            if kind == "permission_prompt":
                self._await_input(
                    str(payload.get("message") or f"{self.label} needs permission"), actor
                )
            elif kind == "idle_prompt":
                self._input_resolved()
                self._set_state("agent_idle")
        elif event == "SubagentStart":
            pass  # _actor_for() above opened its column
        elif event == "SubagentStop" and actor is not self._main:
            actor.stopped_at = time.time()
            self._input_resolved(actor)
            self._set_state("completed", actor)
        elif event == "Interrupt":
            # The user interrupted the turn; anything still running is over.
            for tool_use_id in self._pending_tool_ids():
                self._tool_finished(tool_use_id, error="Interrupted by the user")
            self._input_resolved()
            self._finish_subagents("cancelled")
            self._set_state("agent_idle")
        elif event in {"Stop", "StopFailure"} and actor is not self._main:
            # A subagent's own turn ended; SubagentStop marks it finished.
            self._input_resolved(actor)
            self._set_state("agent_idle", actor)
        elif event in {"Stop", "StopFailure"}:
            self._input_resolved()
            if event == "StopFailure":
                self._main.logger.record_event(
                    "turn_failed", {"error": payload.get("error")}, flush=True
                )
            self._set_state("agent_idle")
            self._expire_unreported_tools()
        elif event == "SessionEnd":
            self._main.logger.record_event(
                "session_ended", {"reason": payload.get("reason")}, flush=True
            )

    def _pending_tool_ids(self) -> "list[str]":
        with self._telemetry_lock:
            return list(self._pending_tools)

    def _tool_started(self, actor: _Actor, payload: dict) -> dict:
        tool_use_id = payload.get("tool_use_id")
        tool_name = str(payload.get("tool_name") or "unknown")
        tool_input = payload.get("tool_input")
        if not isinstance(tool_input, dict):
            tool_input = {"input": tool_input}
        if _spawns_subagent(tool_name):
            # Its SubagentStart names only the subagent's type; the parent's
            # spawn call is where the task it was given is described.
            with self._subagents_lock:
                self._spawn_labels.append(_spawn_label(tool_input))
        # The tool-call id is the model's own id for this call, which the
        # trajectory uses to link a tool to the model response requesting it.
        admission = actor.host.admit_tool_call(
            tool_name, tool_input, {"model_tool_call_id": tool_use_id}
        )
        if tool_use_id and admission["allowed"]:
            with self._telemetry_lock:
                self._pending_tools[tool_use_id] = (actor, admission["call_id"])
        return admission

    def _tool_finished(self, tool_use_id, *, result=None, error=None, duration_ms=None) -> bool:
        """Close a pending call; False when none was pending for this id."""
        with self._telemetry_lock:
            pending = self._pending_tools.pop(tool_use_id, None) if tool_use_id else None
        if pending is None:
            return False
        actor, call_id = pending
        encoded = json.dumps(result, default=str)
        if len(encoded) > _MAX_RESULT_CHARS:
            result = {"_truncated_json": encoded[:_MAX_RESULT_CHARS] + "…[truncated]"}
        timing = {}
        duration_ms = number(duration_ms)
        if duration_ms is not None:
            # The CLI's own execution time, which excludes any permission wait
            # that the PreToolUse boundary includes.
            duration_ns = max(0, int(duration_ms * 1_000_000))
            timing = {"duration_ns": duration_ns, "started_wall_ns": time.time_ns() - duration_ns}
        actor.host.complete_tool_call(call_id, result, error, **timing)
        self._input_resolved(actor)
        if actor.state != "agent_idle" and actor.state not in _FINISHED_STATES:
            self._set_state("running_harness", actor)
        return True

    def _expire_unreported_tools(self) -> None:
        at_turn_end = set(self._pending_tool_ids())

        def expire():
            for tool_use_id in at_turn_end:
                self._tool_finished(
                    tool_use_id,
                    error="Turn ended with no completion reported (rejected or interrupted)",
                )

        timer = threading.Timer(_UNREPORTED_TOOL_GRACE_S, expire)
        timer.daemon = True
        timer.start()

    def _record_model_request(
        self,
        started_at: float,
        ended_at: float,
        *,
        model,
        actor: "_Actor | None" = None,
        input_tokens=None,
        output_tokens=None,
        request_id=None,
        error=None,
        details: "dict | None" = None,
    ) -> None:
        """One completed model request, placed where it actually ran. The
        CLIs report a request only after it ends, so this is always late."""
        if self.capture is not None:
            return  # The gateway is authoritative; retain OTel tool/lifecycle events.
        actor = actor or self._main
        call_label = uuid.uuid4().hex[:12]
        metadata = {
            "type": "metadata",
            "model": model,
            "request_id": request_id,
            "usage": {
                "prompt_tokens": int(number(input_tokens) or 0),
                "completion_tokens": int(number(output_tokens) or 0),
            },
            "source": f"{self.harness}_otel",
            **(details or {}),
        }
        from agency.llm.message_store import MessageStore

        store = MessageStore()
        response = [metadata]
        if error is not None:
            marker = {"error": error}
            response.append(marker)
        response_hashes = store.add_messages([{"role": "assistant", "blocks": response}])
        # A backdated start must not overwrite the agent's live state.
        actor.logger.record_event(
            "agent_state",
            {"state": "waiting_llm", "model": model, "request_id": request_id},
            call_label=call_label,
            timestamp=started_at,
            flush=True,
        )
        actor.logger.record_llm_exchange(
            call_label,
            exchange_type="llm_stream_error" if error is not None else "llm_block",
            message_store=store,
            harness_message_hashes=[],
            patched_message_hashes=[],
            response_message_hashes=response_hashes,
            term_message=f"[{actor.name}] LLM    {'✗' if error else '✓'}  {model}",
            print_to_terminal=False,
            timestamp=ended_at,
        )
        actor.host.record_span(
            f"llm:{model}",
            started_at,
            ended_at,
            {
                "model": model,
                "request_id": request_id,
                "outcome": "failure" if error is not None else "success",
                "timing": "reported_duration",
                "provenance": f"{self.harness}_otel",
            },
            call_label=call_label,
            span_id=call_label,
        )

    # -- Local HTTP/WebSocket endpoint -----------------------------------------

    def _authorized(self, presented: "str | None") -> bool:
        return isinstance(presented, str) and hmac.compare_digest(presented, self.host_token)

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        if self._capture_gateway is not None and not (
            self.capture.native and self.harness == "claude_code"
        ):

            @app.api_route("/capture/{path:path}", methods=["GET", "POST", "DELETE"])
            async def capture_request(request: Request, path: str):
                return await self._capture_gateway.forward(request, path)

            @app.websocket("/capture/{path:path}")
            async def capture_websocket(websocket: WebSocket, path: str):
                await self._capture_gateway.websocket(websocket, path)

        @app.post("/hooks")
        def hooks(payload: dict, request: Request):
            header = request.headers.get("authorization", "")
            if not self._authorized(header.removeprefix("Bearer ")):
                return Response(status_code=401)
            try:
                decision = self.handle_hook(payload)
            except Exception as exc:
                print(f"[agency] WARNING: hook failed: {type(exc).__name__}")
                if payload.get("hook_event_name") == "PreToolUse" and (
                    self.policy.tool_hooks or self.policy.default_to_deny
                ):
                    from ..harness._native_hooks import decision_to_hook_response

                    return JSONResponse(
                        decision_to_hook_response((False, "Agency policy check failed"))
                    )
                decision = None
            if decision:
                return JSONResponse(decision)
            # An empty 2xx body is "no decision": the CLI continues with its
            # normal permission flow.
            return Response(status_code=200)

        @app.post("/v1/logs")
        def logs(payload: dict, request: Request):
            if not self._authorized(request.headers.get("x-agency-token")):
                return Response(status_code=401)
            try:
                self.handle_log_records(payload)
            except Exception as exc:
                print(f"[agency] WARNING: log telemetry dropped: {type(exc).__name__}: {exc}")
            return JSONResponse({})

        @app.websocket("/terminal")
        async def terminal(websocket: WebSocket):
            if self.terminal_mode != "browser":
                await websocket.close(code=4404)
                return
            if not self._authorized(websocket.headers.get("x-agency-token")):
                await websocket.close(code=4401)
                return
            await websocket.accept()
            try:
                await self._serve_terminal(websocket)
            except WebSocketDisconnect:
                pass

        return app

    async def _serve_terminal(self, websocket) -> None:
        hello = json.loads(await websocket.receive_text())
        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue" = asyncio.Queue()
        # A viewer that cannot keep up is disconnected rather than buffered
        # without bound; its reconnect starts again from a screen snapshot.
        limit = 4096

        def deliver(chunk):
            if queue.qsize() < limit:
                queue.put_nowait(chunk)
            else:
                queue.put_nowait(OverflowError)

        def listener(chunk):
            loop.call_soon_threadsafe(deliver, chunk)

        terminal, snapshot = await asyncio.to_thread(
            self.attach_terminal,
            _dimension(hello.get("cols"), 80),
            _dimension(hello.get("rows"), 24),
            listener,
        )
        try:
            await websocket.send_bytes(snapshot)
            if terminal.exited.is_set():
                await websocket.send_json({"type": "exited", "returncode": terminal.returncode})
                return
            await websocket.send_json({"type": "attached", "paused": self._paused})

            async def to_viewer():
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        await websocket.send_json(
                            {"type": "exited", "returncode": terminal.returncode}
                        )
                        return
                    if chunk is OverflowError:
                        await websocket.close(code=1013)
                        return
                    await websocket.send_bytes(chunk)

            async def from_viewer():
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    if message.get("bytes"):
                        if not terminal.write(message["bytes"]):
                            await websocket.send_json(
                                {"type": "input_dropped", "paused": self._paused}
                            )
                    elif message.get("text"):
                        control = json.loads(message["text"])
                        if control.get("type") == "resize":
                            terminal.resize(
                                _dimension(control.get("cols"), terminal.size[0]),
                                _dimension(control.get("rows"), terminal.size[1]),
                            )

            tasks = [asyncio.create_task(to_viewer()), asyncio.create_task(from_viewer())]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                task.result()
        finally:
            terminal.detach(listener)

    def _start_server(self) -> None:
        import uvicorn

        server = uvicorn.Server(
            uvicorn.Config(
                self._build_app(),
                host="127.0.0.1",
                port=0,
                log_level="warning",
                timeout_graceful_shutdown=3,
                # Capture relays upstream handshake headers. Adding another
                # Date/Server header can invalidate a WebSocket handshake.
                date_header=False,
                server_header=False,
                ws_max_size=None,
            )
        )
        self._server = server
        self._server_thread = threading.Thread(
            target=server.run, name="cli-session-endpoint", daemon=True
        )
        self._server_thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if time.monotonic() > deadline:
                raise RuntimeError(f"{self.label} session endpoint did not start")
            time.sleep(0.01)
        self.port = server.servers[0].sockets[0].getsockname()[1]


def _spawns_subagent(tool_name: str) -> bool:
    """Claude Code's Agent (formerly Task) tool; Codex's collaboration.spawn_agent."""
    return tool_name in {"Agent", "Task"} or tool_name.endswith("spawn_agent")


def _spawn_label(tool_input: dict) -> str:
    for key in ("description", "task_name", "subagent_type", "agent_type"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:200]
    return "subagent"


def _dimension(value, fallback: int) -> int:
    """A viewer's terminal size, clamped: a page measured before layout can
    report one row or column (or none), which must not keep the session from
    starting. The viewer resizes again once it has laid out."""
    try:
        return min(max(int(value), 2), 1000)
    except (TypeError, ValueError, OverflowError):
        return fallback


def number(value):
    """A telemetry attribute as a number; OTLP/JSON sends int64 as strings."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _otlp_value(value: dict):
    """One OTLP/JSON AnyValue. int64 values arrive as strings by spec."""
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "arrayValue" in value:
        return [_otlp_value(item) for item in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {
            item["key"]: _otlp_value(item.get("value", {}))
            for item in value["kvlistValue"].get("values", [])
        }
    return None


def otlp_log_records(body: dict, prefix: str):
    """Each log record as {name, attributes, time}, with the CLI's event-name
    *prefix* (e.g. "codex.") removed from the name."""
    for resource in body.get("resourceLogs") or []:
        for scope in resource.get("scopeLogs") or []:
            for record in scope.get("logRecords") or []:
                attributes = {
                    item["key"]: _otlp_value(item.get("value") or {})
                    for item in record.get("attributes") or []
                    if "key" in item
                }
                name = attributes.get("event.name") or _otlp_value(record.get("body") or {})
                if not isinstance(name, str):
                    continue
                yield {
                    "name": name.removeprefix(prefix),
                    "attributes": attributes,
                    "time": _record_time(record, attributes),
                }


def _record_time(record: dict, attributes: dict) -> float:
    # Codex leaves timeUnixNano at 0 and carries the time as event.timestamp.
    nanoseconds = int(record.get("timeUnixNano") or 0)
    if nanoseconds:
        return nanoseconds / 1e9
    stamp = attributes.get("event.timestamp")
    if isinstance(stamp, str):
        try:
            return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return time.time()


def _user_scope_available() -> bool:
    if shutil.which("systemd-run") is None:
        return False
    from ..observability.profiler.agprof import _user_scope_available as probe

    return probe()


@contextmanager
def _post_cli_sigint(session, *, enabled: bool):
    """Finish native CLI cleanup if another Ctrl-C arrives after it exits."""
    if not enabled or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGINT)

    def interrupt(signum, frame):
        if getattr(session.terminal, "returncode", None) is not None:
            return
        if callable(previous):
            previous(signum, frame)
        elif previous == signal.SIG_DFL:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGINT)

    signal.signal(signal.SIGINT, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def main(session_class, command: str, argv) -> int:
    """`agency <command>`: run a native CLI with the shared Agency runtime."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog=f"agency {command}",
        description=f"Run {session_class.label} in your terminal with Agency observation and controls.",
    )
    parser.add_argument("--cwd", default=".", help=f"Directory {session_class.label} works in")
    parser.add_argument("--port", type=int, default=7860, help="Web UI port (127.0.0.1 only)")
    parser.add_argument(
        "--terminal",
        choices=["native", "browser"],
        default="native",
        help="Use your current terminal (default), or the optional browser terminal",
    )
    parser.add_argument(
        "--trace",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Trace local CLI processes/syscalls (default: on for native terminal)",
    )
    parser.add_argument(
        "--trace-file-access",
        action="store_true",
        help="Record file-access metadata using the shared tracer",
    )
    parser.add_argument(
        "--policy", type=Path, help="Python configuration file exporting policy = agpolicy(...)"
    )
    parser.add_argument(
        f"--{command}",
        dest="binary",
        default=None,
        help=f"{session_class.label} executable (default: `{command}` on PATH)",
    )
    parser.add_argument(
        "--no-scope",
        action="store_true",
        help="Do not use systemd user scopes (pause/stop fall back to signals)",
    )
    parser.add_argument(
        "--no-linger",
        action="store_true",
        help=f"Exit when {session_class.label} exits instead of keeping the UI",
    )
    capture_modes = parser.add_mutually_exclusive_group()
    capture_modes.add_argument(
        "--capture",
        action="store_true",
        help="Route native model HTTP requests through Agency and archive full context (opt-in)",
    )
    capture_modes.add_argument(
        "--capture-native",
        action="store_true",
        help=f"Capture {session_class.label} model traffic while retaining native login and model selection",
    )
    parser.add_argument(
        "--capture-model",
        choices=["gpt-6-luna", "claude-haiku-5-5"],
        help="Capture model on LiteLLM (default: gpt-6-luna, validated with both CLIs)",
    )
    parser.add_argument(
        "cli_args", nargs=argparse.REMAINDER, help=f"Arguments after -- go to {command}"
    )
    args = parser.parse_args(argv)
    if args.cli_args[:1] == ["--"]:
        args.cli_args = args.cli_args[1:]
    if not sys.platform.startswith("linux"):
        print(f"agency {command}: Linux only", file=sys.stderr)
        return 1
    trace = args.trace if args.trace is not None else args.terminal == "native"
    if args.terminal == "native" and not (sys.stdin.isatty() and sys.stdout.isatty()):
        parser.error(
            "native mode requires your terminal on stdin/stdout; use --terminal browser for remote viewing"
        )
    if args.terminal == "browser" and trace:
        parser.error("--trace requires --terminal native")
    if args.trace_file_access and not trace:
        parser.error("--trace-file-access requires --trace")
    from ..agpolicy import agpolicy
    from ..configs.agconfig import ptraceconfig

    policy = agpolicy()
    if args.policy:
        try:
            policy = runpy.run_path(str(args.policy.resolve()))["policy"]
        except (OSError, KeyError) as exc:
            parser.error(f"cannot load policy: {exc}")
        if not isinstance(policy, agpolicy):
            parser.error("policy file must export an agpolicy instance named policy")
    if not trace and (policy.syscall_hooks or policy.default_to_deny):
        parser.error("syscall policy requires --trace")
    binary = args.binary or shutil.which(command)
    if not binary:
        print(f"agency {command}: `{command}` not found on PATH (use --{command})", file=sys.stderr)
        return 1
    cwd = Path(args.cwd).resolve()
    if not cwd.is_dir():
        print(f"agency {command}: {cwd} is not a directory", file=sys.stderr)
        return 1
    capture = None
    if args.capture_model and not args.capture:
        parser.error("--capture-model requires --capture")
    if args.capture:
        from .capture import CaptureConfig

        if not os.environ.get("LITE_LLM_KEY"):
            parser.error("--capture requires LITE_LLM_KEY in the environment")
        capture = CaptureConfig(args.capture_model or "gpt-6-luna")
    elif getattr(args, "capture_native", False):
        if command == "claude":
            from .capture_proxy import ClaudeNativeCaptureConfig

            capture = ClaudeNativeCaptureConfig()
        else:
            from .capture import NativeCaptureConfig

            capture = NativeCaptureConfig()
    use_scope = not args.no_scope and _user_scope_available()
    if use_scope and os.environ.get(_SCOPED_VARIABLE) != "1":
        # Run this process in its own scope too, so the profiler's workload
        # counters cover Agency's side of the session and nothing else in the
        # user's login session.
        os.execvp(
            "systemd-run",
            [
                "systemd-run",
                "--user",
                "--scope",
                "--collect",
                "--quiet",
                f"--unit=agency-session-{uuid.uuid4().hex[:8]}",
                "env",
                f"{_SCOPED_VARIABLE}=1",
                sys.executable,
                "-m",
                "agency.cli",
                command,
                *argv,
            ],
        )
    if not use_scope:
        print(
            "[agency] WARNING: systemd user scopes unavailable; the profiler cannot separate "
            "this session's resources and pause/stop use process-group signals.",
            flush=True,
        )

    from ..observability.agwebui import agwebui
    from ..utils.agutil import _DEFAULT_LOG_DIR

    try:
        session = session_class(
            cwd=cwd,
            binary=binary,
            cli_args=args.cli_args,
            run_dir=_DEFAULT_LOG_DIR,
            ui_port=args.port,
            use_scope=use_scope,
            capture=capture,
            terminal_mode=args.terminal,
            trace=trace,
            policy=policy,
            ptrace_config=ptraceconfig(file_access=args.trace_file_access),
        )
    except ValueError as exc:
        parser.error(str(exc))

    def interactive_session():
        try:
            session.run()
        except Exception as exc:
            # agwebui records workload failures before returning to its
            # caller; preserve a failing exit status for the CLI wrapper.
            session._process_error = exc
            raise

    with _post_cli_sigint(session, enabled=args.terminal == "native"):
        agwebui.run(
            interactive_session,
            port=args.port,
            host="127.0.0.1",
            linger=args.terminal == "browser" and not args.no_linger,
        )
    if getattr(session, "_process_error", None) is not None:
        return 1
    returncode = getattr(session.terminal, "returncode", None)
    return (128 - returncode if returncode < 0 else returncode) if returncode is not None else 0
