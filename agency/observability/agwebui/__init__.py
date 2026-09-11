"""agwebui — web-based UI for monitoring agency runs.

Starts a standalone FastAPI server in a separate process and serves a
browser dashboard. The server process has no agency imports of its own; it
reads the same log_dir the execution process's agents already write to
(each agent's own agDataLogger database, plus the orchestrator's shared
global_data.sqlite3) directly off disk and pushes updates over WebSocket.
The execution script runs directly in the main thread — no asyncio
conflicts.

Usage::

    from agency.observability.agwebui import agwebui

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
from pathlib import Path
from typing import Any

from ...utils.agutil import _DEFAULT_LOG_DIR, sigterm_as_exit
from ..profiler import agprof

# Module-level singleton — set while agwebui.run() is active.
_active: "agwebui | None" = None


def _merge_config_fields(agconfig: Any, config: dict) -> None:
    """Mutate *agconfig*'s own namespaces in place, field by field, rather
    than replacing it with a new object. *config* is shaped like
    ``agconfig.safe_snapshot()``'s output: ``{"llm": {...}, "sandbox":
    {...}}``. agconfig.clone() (called by every agent()/agteam()
    construction) just snapshots whatever is currently set -- so anything
    that hasn't cloned this exact object yet will pick up the change on its
    next construction, with no cooperation needed from whatever code holds
    another reference to it (e.g. a user script's own module-level config
    variable)."""
    agconfig.update(**config)


def _apply_config_update(target: Any, config: dict, agconfig_cls: Any) -> None:
    """Merge *config* into *target*'s existing agconfig, then push via
    ``change_config``. Falls back to a fresh agconfig when the target has
    none yet. Never replaces a live agconfig with only the editor payload
    -- that payload is a safe_snapshot() and omits static fields such as
    sandbox mounts."""
    if target.agconfig is not None:
        merged = target.agconfig.clone()
        _merge_config_fields(merged, config)
        target.change_config(merged)
    else:
        fresh = agconfig_cls()
        _merge_config_fields(fresh, config)
        target.change_config(fresh)


def _all_agteam_subclasses(cls):
    """Every agteam subclass currently defined, at any depth -- found via
    Python's own subclass tracking (__subclasses__()), not a framework
    registry. This is how a team class's own agconfig class attribute (e.g.
    `agconfig = LLM_CONFIG` in a user script) gets reached without the
    framework needing to know that attribute, or the script, exists."""
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_agteam_subclasses(sub)


def _dispatch_command(cmd: dict) -> None:
    """Apply one pause/resume command written by the webui server process.

    The (isolated, no-agency-imports) server process can only write a plain
    file describing what it wants; this side -- running inside the
    execution process, with real agent objects -- is what actually applies
    it."""
    from ...agent import agent as _agent_cls
    from ...configs.agconfig import agconfig as _agconfig_cls

    ctype = cmd.get("type")
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
        # Merge into the agent's existing agconfig. The webui editor only
        # ships a safe_snapshot() (LLM knobs etc.) -- replacing the whole
        # object would drop static fields the editor never sees, notably
        # agSandbox.mounts / base_image. Forks after a wipe then recreate
        # sandboxes without the shared HF cache bind and bake model weights
        # into lifecycle image layers.
        config = cmd.get("config") or {}
        for a in _agent_cls.all():
            if a.agname == agname:
                _apply_config_update(a, config, _agconfig_cls)
                break
    elif ctype == "update_config_all":
        from ...agteam import agteam as _agteam_cls

        config = cmd.get("config") or {}

        # 1. Agents that already exist -- each already cloned its own
        #    agconfig at construction time, so it needs a direct push.
        #    Merge (not replace) so sandbox mounts / base_image survive.
        for a in _agent_cls.all():
            _apply_config_update(a, config, _agconfig_cls)

        # 2. Team instances that already exist -- change_config() replaces
        #    the team's own live agconfig *and* cascades to every agent it
        #    tracks, covering agents added to this team from here on.
        #    Same merge rule as agents: the editor payload is partial.
        for t in _agteam_cls.all():
            _apply_config_update(t, config, _agconfig_cls)

        # 3. Every agteam subclass's class-level agconfig, mutated in place
        #    (not replaced) -- so a team constructed *after* this point,
        #    whose __init__ clones type(self).agconfig fresh, sees the
        #    update. Reaches a user script's own shared config object (e.g.
        #    `agconfig = LLM_CONFIG`) without needing to know it exists.
        for team_cls in _all_agteam_subclasses(_agteam_cls):
            if team_cls.agconfig is not None:
                _merge_config_fields(team_cls.agconfig, config)

        # 4. The framework-wide fallback for a bare agent() call made with
        #    no active team context.
        if _agent_cls.default_agconfig is not None:
            _merge_config_fields(_agent_cls.default_agconfig, config)


def _flush_loop(stop_event: threading.Event, interval: float = 0.5) -> None:
    """Force periodic disk flushes of every live agDataLogger (the
    orchestrator's global one, plus each agent's own) while the webui is
    active.

    agDataLogger only flushes as a side effect of a record_event() call
    landing >= flush_interval_s after the last flush -- there's no
    background timer of its own. During any quiet stretch (e.g. one long
    LLM call, or simply nothing happening for a moment) nothing triggers
    that, so the separate webui server process reading the same files
    directly off disk would otherwise see stale or empty data until traffic
    resumes."""
    from ...agent import agent as _agent_cls
    from ...orchestrator import peek_orchestrator

    while not stop_event.wait(interval):
        orchestrator = peek_orchestrator()
        if orchestrator is not None:
            try:
                orchestrator.data_logger.flush()
            except Exception as _e:
                print(f"[agwebui] WARNING: failed to flush orchestrator data logger: {_e}")
        for a in _agent_cls.all():
            try:
                a.data_logger.flush()
            except Exception as _e:
                print(f"[agwebui] WARNING: failed to flush data logger for {a.agname}: {_e}")


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
                except FileNotFoundError:
                    # Command file may already be removed by another actor;
                    # this cleanup is best-effort.
                    pass
                except Exception as _e:
                    print(f"[agwebui] WARNING: failed to remove command file {f.name}: {_e}")
        stop_event.wait(0.2)


class agwebui:
    """Web-based UI for monitoring agency runs.

    Primary usage — ``agwebui.run(fn)`` starts the web server subprocess,
    runs *fn()* in the calling thread, then optionally lingers until
    Ctrl+C.
    """

    def __init__(self, run_dir: Path, port: int) -> None:
        self._run_dir = run_dir
        self._port = port
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

        Wraps the call in ``agutil.sigterm_as_exit()`` -- see its docstring
        for why a plain `kill <pid>` needs that at all (short version: with
        no handler, SIGTERM skips every `atexit` cleanup hook the framework
        relies on, exactly like SIGKILL). A script that builds/runs agents
        directly without going through ``agwebui.run()``/``graphui.run()``
        should wrap itself in ``agutil.sigterm_as_exit()`` (or the
        ``agency.sigterm_as_exit`` re-export) the same way, since that
        protection doesn't otherwise exist anywhere more central.
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

        from .build_perfetto import ensure_viewer

        ensure_viewer()

        if run_dir is None:
            # Same directory every agent()/agteam() already writes its own
            # agDataLogger database to (unless overridden via `log_dir` on
            # their agconfig) -- the webui reads that data directly, so it
            # has to point at the same place rather than a throwaway of its
            # own.
            run_dir = _DEFAULT_LOG_DIR
        run_dir.mkdir(parents=True, exist_ok=True)

        ui = cls(run_dir=run_dir, port=port)

        server_log = open(run_dir / "server.log", "w")
        ui._server_log = server_log
        ui._server_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "agency.observability.agwebui.server",
                "--run-dir",
                str(run_dir),
                "--port",
                str(port),
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
            print(
                f"[agwebui] Warning: server may not be ready at http://localhost:{port}", flush=True
            )

        _active = ui
        print(f"[agwebui] Web UI: http://localhost:{port}", flush=True)

        # Emit initial resource pool state so the dashboard shows GPU/CPU
        # capacity immediately without waiting for the first acquire/release.
        # get_orchestrator() is a process-wide singleton: if this is the
        # first call anywhere (no agent constructed yet), passing the same
        # run_dir the server subprocess was pointed at keeps its global
        # agDataLogger database where the webui is actually reading from --
        # otherwise it would fall back to _DEFAULT_LOG_DIR and silently
        # diverge from an explicitly-passed run_dir.
        try:
            from ...orchestrator import get_orchestrator
            from ..agdatalogger import resolve_global_db_path

            _pool = get_orchestrator(
                default_db_path=resolve_global_db_path(run_dir)
            ).agresource_pool
            if _pool is not None:
                _pool._emit_resource()
        except Exception as _e:
            print(f"[agwebui] WARNING: failed to emit initial resource pool state: {_e}")

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
            target=_poll_commands,
            args=(run_dir / "ui_commands", command_stop),
            daemon=True,
            name="agwebui-commands",
        ).start()

        # Keeps the dashboard's view of disk reasonably fresh (see
        # _flush_loop's docstring) -- stops alongside the command relay.
        threading.Thread(
            target=_flush_loop,
            args=(command_stop,),
            daemon=True,
            name="agwebui-flush",
        ).start()

        with sigterm_as_exit("agwebui") as sigterm_received:
            try:
                with agprof.workload():
                    fn(*args, **kwargs)
            except Exception:
                import traceback

                traceback.print_exc()
            finally:
                command_stop.set()
                try:
                    from ...orchestrator import get_orchestrator

                    get_orchestrator().data_logger.record_event(
                        "done",
                        {},
                        name="webui",
                        object="agwebui",
                        update_latest_snapshot=True,
                        flush=True,
                    )
                except Exception as _e:
                    print(f"[agwebui] WARNING: failed to emit done event: {_e}")
                _active = None

                # Skip lingering if we're already unwinding from a SIGTERM --
                # the caller asked this process to exit, not to keep serving
                # the dashboard until a *second* signal (Ctrl+C) arrives.
                if linger and not sigterm_received.is_set():
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
                    except Exception as _e:
                        print(f"[agwebui] WARNING: server process did not exit cleanly: {_e}")
                    if ui._server_log is not None:
                        ui._server_log.close()
