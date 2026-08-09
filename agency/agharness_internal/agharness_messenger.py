"""Host-side messenger bridging an agent's inbox/pause state into a
container-resident harness loop.

Design (from the discussion that produced it): treat this as "the user" as
far as the harness loop is concerned -- when there's a pending inbox
message, it's delivered as an ordinary user-role message, not a
side-channel the loop needs to know about specially. Mirrors what
`execute_react()` already does host-side, in-process, at the top of every
ReAct iteration (`ag._check_pause(self.name)` then `ag._drain_inbox(
messages)`, see agent.py/agskill.py) -- this bridges the exact same two
calls for a loop that no longer runs in the same process.

**Only meaningfully helps native.py's own loop.** An external harness's
internal loop (Claude Code, etc.) is opaque to us -- there is no hook to
inject a message mid-turn into a running third-party CLI process, so
inbox/pause for those engines stays at the coarser boundary they already
have: `agskill.py`'s `_task()` checks `ag._check_pause()` once before a
launch even starts. Native's loop, by contrast, calls this before every
turn, since it's a real long-lived process we fully control.

Deliberately its own class with its own token->agent registry, not
piggy-backed onto `agLLMTerminus`'s -- same reasoning as that module's own
docstring: each bridged service should own its own registry rather than
reaching into a sibling's in-process state.

**Known limitation, not hidden**: `/internal/check_in`'s route handler
calls `ag._check_pause()`, which blocks for as long as the agent stays
paused -- potentially a very long time. It's declared as a plain (non-
`async`) route so FastAPI/Starlette runs it in its worker thread pool
rather than the event loop, so one paused agent's check-in doesn't stall
every other agent's -- but that thread pool is still bounded (~40 threads
by default), so a large number of simultaneously-paused agents could
exhaust it. Acceptable for now (pausing many agents at once is not the
common case); revisit if it proves to be a real constraint.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..agconfig import GlobalConfigParam, _AgConfigViewBase

if TYPE_CHECKING:
    from ..agconfig import agConfig
    from ..agent import agent


class _AgHarnessMessengerFields:
    bind_host = GlobalConfigParam("agharness_messenger", default="127.0.0.1")


class agHarnessMessengerConfig(_AgConfigViewBase):
    _OWNER = "agharness_messenger"


class _CheckInRequest(BaseModel):
    token: str


class agHarnessMessenger:
    """One process-wide messenger, shared across every engine the same
    way `agllm_terminus`/`agmcp_server`'s shared instances are."""

    def __init__(self, agconfig: "agConfig | None" = None) -> None:
        self._agconfig = agconfig
        self._lock = threading.Lock()
        self._agents_by_token: "dict[str, agent]" = {}
        self._app = self._build_app()
        self._server = None
        self._thread: "threading.Thread | None" = None
        self.base_url: "str | None" = None
        self._uds_server = None
        self._uds_thread: "threading.Thread | None" = None
        self.uds_path: "str | None" = None
        # Survives stop_uds() (which clears uds_path) so a restart rebinds the
        # SAME path -- see agutil.reserve_uds_path.
        self._uds_reserved_path: "str | None" = None

    # -- token <-> agent registry ---------------------------------------

    def register(self, token: str, ag: "agent") -> None:
        with self._lock:
            self._agents_by_token[token] = ag

    def unregister(self, token: str) -> None:
        with self._lock:
            self._agents_by_token.pop(token, None)

    def _agent_for_token(self, token: "str | None"):
        if token is None:
            return None
        with self._lock:
            return self._agents_by_token.get(token)

    # -- app / route ------------------------------------------------------

    def _build_app(self):
        app = FastAPI()

        # Deliberately a plain `def`, not `async def` -- see this class's
        # own docstring for why: `ag._check_pause()` blocks for as long as
        # the agent stays paused, and FastAPI/Starlette runs a sync route
        # in a worker thread pool rather than the single event loop, so
        # this doesn't stall every other agent's own check-in.
        @app.post("/internal/check_in")
        def check_in(payload: _CheckInRequest):
            ag = self._agent_for_token(payload.token)
            if ag is None:
                return JSONResponse({"error": "unknown or missing token"}, status_code=401)
            ag._check_pause()
            drained: "list[dict]" = []
            ag._drain_inbox(drained)
            return {"messages": drained}

        return app

    # -- lifecycle ------------------------------------------------------------
    # Identical shape to agLLMTerminus's own start()/ensure_uds_started()/
    # stop() -- see that module for the reasoning (uvicorn on a daemon
    # thread, poll `server.started` rather than assume a fixed delay, UDS
    # as the container-boundary-crossing mechanism via agsandbox's
    # existing bind-mount primitive).

    def start(self, timeout_s: float = 10) -> str:
        if self.base_url is not None:
            return self.base_url

        fields = _AgHarnessMessengerFields()
        config = uvicorn.Config(self._app, host=fields.bind_host, port=0, log_level="warning")
        server = uvicorn.Server(config)
        self._server = server

        self._thread = threading.Thread(target=server.run, daemon=True, name="agharness_messenger")
        self._thread.start()

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not server.started:
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("agHarnessMessenger server did not start within timeout")

        port = server.servers[0].sockets[0].getsockname()[1]
        self.base_url = f"http://{fields.bind_host}:{port}"
        return self.base_url

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._server = None
        self._thread = None
        self.base_url = None
        self.stop_uds()

    def ensure_uds_started(self, timeout_s: float = 10) -> str:
        from ..agutil import reserve_uds_path, uds_listener_is_live

        # "Idempotently" must mean *still working*, not merely *started once*:
        # an external cleanup can delete a live socket file (a socket's mtime
        # never updates, so age-based reapers see every long-lived one as
        # stale) and a dead server thread takes its socket with it, since
        # uvicorn unlinks on shutdown. Either leaves this method handing out a
        # path nothing listens on. Rebuild at the same reserved path instead.
        if self.uds_path is not None:
            if uds_listener_is_live(self.uds_path, self._uds_thread):
                return self.uds_path
            self.stop_uds()

        sock_path = reserve_uds_path(self._uds_reserved_path, "agharness_messenger")
        self._uds_reserved_path = sock_path
        config = uvicorn.Config(self._app, uds=sock_path, log_level="warning")
        server = uvicorn.Server(config)
        self._uds_server = server

        self._uds_thread = threading.Thread(
            target=server.run, daemon=True, name="agharness_messenger-uds"
        )
        self._uds_thread.start()

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not server.started:
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("agHarnessMessenger UDS server did not start within timeout")

        self.uds_path = sock_path
        return sock_path

    def stop_uds(self) -> None:
        if self._uds_server is not None:
            self._uds_server.should_exit = True
        if self._uds_thread is not None:
            self._uds_thread.join(timeout=10)
        self._uds_server = None
        self._uds_thread = None
        self.uds_path = None


_shared_messenger: "agHarnessMessenger | None" = None
_shared_messenger_lock = threading.Lock()


def get_shared_messenger(agconfig: "agConfig | None" = None) -> agHarnessMessenger:
    """One `agHarnessMessenger` per process, mirroring
    `agllm_terminus.get_shared_terminus`/`agmcp_server.get_shared_mcp_server`."""
    global _shared_messenger
    if _shared_messenger is not None:
        return _shared_messenger
    with _shared_messenger_lock:
        if _shared_messenger is None:
            _shared_messenger = agHarnessMessenger(agconfig)
        return _shared_messenger


__all__ = ["agHarnessMessenger", "agHarnessMessengerConfig", "get_shared_messenger"]
