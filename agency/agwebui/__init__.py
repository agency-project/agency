"""agwebui — web-based UI for monitoring agency runs.

Drop-in replacement for agUI.  Starts a standalone FastAPI server in a
separate process and serves a browser dashboard.  The execution process
writes structured events to a JSONL file; the server tails it and pushes
updates over WebSocket.  The execution script runs directly in the main
thread — no asyncio, no Textual, no spawn conflict.

Usage::

    from agency.agwebui import agwebui

    agwebui.run(main_fn)            # opens http://localhost:7860
    agwebui.run(main_fn, port=8080)
    agwebui.run(main_fn, linger=False)  # exit immediately when done
"""
from __future__ import annotations

import atexit
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from .emitter import agwebui_emitter

# Module-level singleton — set while agwebui.run() is active.
_active: "agwebui | None" = None


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

        try:
            fn(*args, **kwargs)
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
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
