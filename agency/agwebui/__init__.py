"""agwebui — web-based UI for monitoring agency runs.

Starts a standalone FastAPI server in a separate process and serves a
browser dashboard.  The execution process writes structured events to a
SQLite database (ui_events.db); the server polls it and pushes updates
over WebSocket.  The execution script runs directly in the main thread —
no asyncio conflicts.

Usage::

    from agency.agwebui import agwebui

    agwebui.run(main_fn)            # opens http://localhost:7860
    agwebui.run(main_fn, port=8080)
    agwebui.run(main_fn, linger=False)  # exit immediately when done
"""
from __future__ import annotations

import atexit
import json
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from .emitter import agwebui_emitter

# Module-level singleton — set while agwebui.run() is active.
_active: "agwebui | None" = None


def _dispatch_command(cmd: dict) -> None:
    """Apply one pause/resume command written by the webui server process.

    Mirrors the ask_human file-drop pattern (agwebui/emitter.py's
    _reply_dir), but in the opposite direction: the (isolated, no-agency-
    imports) server process can only write a plain file describing what it
    wants; this side -- running inside the execution process, with real
    agent objects -- is what actually applies it."""
    from ..agent import agent as _agent_cls
    from ..agconfig import agConfig as _agConfig_cls

    ctype  = cmd.get("type")
    agname = cmd.get("agname")
    if ctype in ("pause", "resume"):
        for a in _agent_cls.all():
            if a.agname == agname:
                (a.pause if ctype == "pause" else a.resume)()
                break
    elif ctype in ("pause_all", "resume_all"):
        for a in _agent_cls.all():
            (a.pause if ctype == "pause_all" else a.resume)()
    elif ctype == "update_config":
        new_cfg = _agConfig_cls(cmd.get("config") or {})
        for a in _agent_cls.all():
            if a.agname == agname:
                a.change_config(new_cfg)
                break
    elif ctype == "update_config_all":
        new_cfg = _agConfig_cls(cmd.get("config") or {})
        for a in _agent_cls.all():
            a.change_config(new_cfg)


def _poll_commands(command_dir: Path, stop_event: threading.Event) -> None:
    command_dir.mkdir(parents=True, exist_ok=True)
    while not stop_event.is_set():
        for f in sorted(command_dir.glob("*.json")):
            try:
                cmd = json.loads(f.read_text(encoding="utf-8"))
                _dispatch_command(cmd)
            except Exception as _e:
                print(f"[agwebui] WARNING: failed to apply command {f.name}: {_e}")
            finally:
                try:
                    f.unlink()
                except Exception:
                    pass
        stop_event.wait(0.2)


class agwebui:
    """Web-based UI for monitoring agency runs.

    Primary usage — ``agwebui.run(fn)`` starts the web server subprocess,
    registers the event emitter, runs *fn()* in the calling thread, then
    optionally lingers until Ctrl+C.
    """

    def __init__(self, run_dir: Path, port: int) -> None:
        self.emitter = agwebui_emitter(run_dir)
        self._run_dir = run_dir
        self._port    = port
        self._server_proc: subprocess.Popen | None = None
        self._server_log = None

    @classmethod
    def run(
        cls,
        fn: Any,
        *args,
        run_dir: Path | None = None,
        port: int = 7860,
        linger: bool = True,
        **kwargs,
    ) -> None:
        """Run *fn(\\*args, \\*\\*kwargs)* with the web UI active.

        Starts the server subprocess first, waits for it to be ready, then
        calls *fn* in the current thread.  Blocks until *fn* completes and
        (if *linger=True*) until the user presses Ctrl+C.
        """
        global _active

        # Fail fast if the port is already occupied (before creating any dirs).
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
            _s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if _s.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(
                    f"[agwebui] Port {port} is already in use. "
                    f"Stop the existing server before starting a new run."
                )

        if run_dir is None:
            ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = Path("runs") / f"webui_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        ui = cls(run_dir=run_dir, port=port)

        server_log = open(run_dir / "server.log", "w")
        ui._server_log = server_log
        ui._server_proc = subprocess.Popen(
            [
                sys.executable, "-m", "agency.agwebui.server",
                "--run-dir", str(run_dir),
                "--port",    str(port),
            ],
            stdout=server_log,
            stderr=server_log,
        )

        # Wait up to 10 s for the server to be ready.
        health_url = f"http://localhost:{port}/health"
        for _ in range(50):
            try:
                urllib.request.urlopen(health_url, timeout=1)
                break
            except Exception:
                time.sleep(0.2)
        else:
            print(f"[agwebui] Warning: server may not be ready at http://localhost:{port}", flush=True)

        _active = ui
        print(f"[agwebui] Web UI: http://localhost:{port}", flush=True)

        # Emit initial resource pool state so the dashboard shows GPU/CPU
        # capacity immediately without waiting for the first acquire/release.
        try:
            from ..agent import agent as _agent_cls
            _pool = _agent_cls.agresource_pool
            if _pool is not None:
                _pool._emit_resource()
        except Exception:
            pass

        def _kill_server() -> None:
            proc = ui._server_proc
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()

        atexit.register(_kill_server)

        # Background thread applying pause/resume (and future) commands the
        # webui server process writes to run_dir/ui_commands -- the server
        # process itself has no agency imports and can't call agent.pause()
        # directly, so this is the execution-process side of that relay.
        command_stop = threading.Event()
        threading.Thread(
            target=_poll_commands, args=(run_dir / "ui_commands", command_stop),
            daemon=True, name="agwebui-commands",
        ).start()

        try:
            fn(*args, **kwargs)
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            command_stop.set()
            ui.emitter.done()
            _active = None

            if linger:
                print(
                    f"[agwebui] Done — dashboard still at http://localhost:{port}  (Ctrl+C to exit)",
                    flush=True,
                )
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass

            if ui._server_proc is not None:
                ui._server_proc.terminate()
                try:
                    ui._server_proc.wait(timeout=5)
                except Exception:
                    pass
                if ui._server_log is not None:
                    ui._server_log.close()
