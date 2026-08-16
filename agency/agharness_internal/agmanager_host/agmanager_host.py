"""New: one host-side agent manager PER AGENT (`agHostAgentManager`).

Status: `native.py` and `claude_code.py` are migrated onto this design.
`codex.py`/`opencode.py`/`grok.py` are not (deprioritized for now) --
they're currently non-functional, since the two of the five old shared
singletons they still depended on (`agproxy_llm.py`, `agprof_ingest.py`)
were removed once `agllm_terminus.py`/`agmcp_server.py`/
`agharness_messenger.py` (the other three, which only native.py/
claude_code.py's OLD design used) were retired. This is the host half of a
two-server redesign (the other half is `agmanager_harness/
agmanager_harness.py`, a per-agent process that runs INSIDE the sandbox
container, or in-process on the host for a bare-host/chroot launch).

This module is a thin COMPOSITION ROOT -- each concern lives in its own
sibling module, all sharing one `launch_state.LaunchRegistry`:
- `launch_state.py` -- per-launch state/registry (`LaunchRegistry`,
  `LaunchHandle`), shared by every module below.
- `llm_dispatch.py` -- the actual credentialed LLM call (`LLMDispatcher`),
  model/context-limit resolution.
- `control_routes.py` -- token validation, warning logging, tool-policy
  checks, pause/inbox check-in.
- `mcp_tools.py` -- resource control, `submit_output`, `ask_human`.
- `profiler_ingest.py` -- profiler correlation/ingestion (`ProfilerIngest`),
  its own separate UDS listener.
- `serialization.py` -- duck-typed chat-completion response/chunk helpers,
  used only by `llm_dispatch.py`.
- `config.py` -- the two `GlobalConfigParam`s (`bind_host`,
  `request_timeout_s`) both this module and `llm_dispatch.py` need.

**Why one instance per agent instead of one shared singleton per service:**
today, `agLLMTerminus`/`agMCPServer`/`agHarnessMessenger`/`agProfilerIngest`
are each a single process-wide object multiplexing every concurrently
running agent by a `token -> agent` dict lookup, on one FastAPI app, one
uvicorn event loop, one thread. That has a real concurrency cost: every one
of those services' `async def` routes calls a *synchronous* blocking
operation in-line (`client.chat.completions.create()` most notably, a plain
`openai.OpenAI` client, not `AsyncOpenAI`) with no thread-pool offload,
which blocks that single shared event loop for every other agent's request
for the duration of the call. Giving each agent its own manager instance,
its own background thread, and (for non-streaming dispatch in particular)
its own blocking call, means one agent's slow LLM turn no longer serializes
against any other agent's.

**What moved here, from which old module, and why:**
- LLM dispatch, response (re)serialization, transcript recording (from
  `agllm_terminus.py`'s `/internal/dispatch`, now `llm_dispatch.py`) -- the
  actual credentialed client construction (`ag.llm.backend.make_client()`)
  only ever happens here; this is the one place real per-agent backend
  credentials exist.
- Model/context-limit resolution, warning logging, tool-policy checks (from
  `agllm_terminus.py`'s `/internal/resolve_model`, `/internal/context_limit`,
  `/internal/log_warning`, `/internal/check_tool_policy`, now split across
  `llm_dispatch.py`/`control_routes.py`) -- these existed there only
  because a container-side caller has no live `ag`; a per-agent manager
  already knows exactly which `ag` every request is about.
- Resource control (`reserve_cpu`/`cpu_release`/`daemon_release`), output
  submission (`submit_output`), human-in-the-loop (`ask_human`) (from
  `agmcp_server.py`, now `mcp_tools.py`) -- all mutate/read host-only
  objects (`ag.sandbox`, `ag.agresource_pool`, `skill.output_schema`), so
  they stay host-side.
- Pause/inbox-drain (`/internal/check_in`, from `agharness_messenger.py`,
  now `control_routes.py`) -- `ag._check_pause()`/`ag._drain_inbox()` are
  host-only methods on the live agent object, so the actual check has to
  happen here. The container-side manager (`agmanager_harness`) exposes
  the endpoint a harness/loop actually polls and forwards it here; see
  that module's docstring for why the polling endpoint itself needs to be
  container-local.
- Profiler correlation/ingestion (from `agprof_ingest.py`, now
  `profiler_ingest.py`) -- run/span context and host wall-clock are
  host-only; kept as a SEPARATE listener (its own socket/thread) from the
  main dispatch app, for the same reason the original design split it out:
  a burst of profiler events must never be able to stall a streaming
  dispatch and corrupt its TTFT measurement.

**What did NOT move here** (out of scope for this pass, see conversation):
- LLM wire-format translation (Anthropic Messages / OpenAI Responses <->
  chat-completions, from `agproxy_llm.py`) -- no host-only state, purely a
  translation layer; belongs in `agmanager_harness`, container-side.
- UDS<->TCP bridging, the policy/profiler hook bridge endpoints a
  subprocess-based permission hook can actually reach -- also
  `agmanager_harness`, since they only need to be reachable FROM inside the
  container, not host-only state.
- Native's own ReAct loop (compaction, built-in tool execution, dispatch
  retry) -- engine-specific, not part of either per-agent manager; would
  sit on top of `agmanager_harness` as a client once native itself becomes
  a standalone harness (a separate, later refactor).
- Claude Code's `--resume` session-blob capture/restore -- `agskill`
  backend-specific persistence on `ag._harness_sessions`, orthogonal to
  this split.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI

from .config import AgHostAgentManagerFields
from .control_routes import build_router as build_control_router
from .launch_state import LaunchHandle, LaunchRegistry
from .llm_dispatch import LLMDispatcher
from .mcp_tools import build_mcp_server
from .profiler_ingest import ProfilerIngest

if TYPE_CHECKING:
    from ...agconfig import agConfig
    from ...agent import agent
    from ...agskill import agskill


_live_managers_lock = threading.Lock()


class agHostAgentManager:
    """One instance per agent. See module docstring for the full design and
    what moved here from which of the five old shared singletons.

    **Self-registers on creation** into `_live_managers` (this class's own
    registry, keyed by `agent_id`, so any code that needs to find/enumerate
    live managers can do so directly, e.g. `agHostAgentManager.get(agent_id)`/
    `all_live()`). Deliberately done HERE, in `__init__`, not by whoever
    constructs a manager (`agharness.get_or_create_host_manager()`), so
    registration can't be forgotten by a caller."""

    #: Every currently-live manager, keyed by `agent_id` (== `ag.agname`).
    #: A plain dict, not a weak-value one -- membership here is an
    #: explicit part of this class's own lifecycle (added in `__init__`,
    #: removed in `stop()`), not an opportunistic cache that should
    #: silently evict itself via GC.
    _live_managers: "dict[str, agHostAgentManager]" = {}

    def __init__(self, ag: "agent", agconfig: "agConfig | None" = None) -> None:
        self._ag = ag
        self._agconfig = agconfig
        self.agent_id = str(ag.agname)

        self._registry = LaunchRegistry(ag)
        self._dispatcher = LLMDispatcher(ag, self._registry, agconfig)
        self._profiler = ProfilerIngest(ag, self._registry)
        self._mcp = build_mcp_server(ag, self._registry)
        self._app = self._build_app()

        self._server = None
        self._thread: "threading.Thread | None" = None
        self.base_url: "str | None" = None

        self._uds_server = None
        self._uds_thread: "threading.Thread | None" = None
        self.uds_path: "str | None" = None
        self._uds_reserved_path: "str | None" = None

        with _live_managers_lock:
            agHostAgentManager._live_managers[self.agent_id] = self

    @classmethod
    def get(cls, agent_id: str) -> "agHostAgentManager | None":
        with _live_managers_lock:
            return cls._live_managers.get(agent_id)

    @classmethod
    def all_live(cls) -> "list[agHostAgentManager]":
        with _live_managers_lock:
            return list(cls._live_managers.values())

    # -- per-launch registry (thin delegation to launch_state.LaunchRegistry) -

    def register_launch(
        self,
        token: "str | None" = None,
        *,
        skill: "agskill | None" = None,
        exact_events: bool = False,
        exact_tool_events: bool = False,
    ) -> LaunchHandle:
        return self._registry.register(
            token, skill=skill, exact_events=exact_events, exact_tool_events=exact_tool_events
        )

    def unregister_launch(self, token: str) -> None:
        self._registry.unregister(token)

    def collected_output(self, token: str) -> dict:
        return self._registry.collected_output(token)

    def transcript_for_token(self, token: "str | None") -> "list[dict] | None":
        return self._registry.transcript_for_token(token)

    @property
    def request_log(self) -> "list[dict]":
        return self._dispatcher.request_log

    # -- app assembly ---------------------------------------------------------

    def _build_app(self) -> FastAPI:
        from contextlib import asynccontextmanager, AsyncExitStack
        from mcp.server.transport_security import TransportSecuritySettings

        # Built before the FastAPI() call below (not inline at the mount
        # site) so its lifespan can be wired into the PARENT app's own
        # lifespan -- see the `lifespan()` function below for why that step
        # is required at all: a mounted ASGI sub-app's lifespan is NOT
        # started automatically just because it's `app.mount()`ed onto a
        # parent whose own lifespan uvicorn does run. Confirmed the hard
        # way: without this, every /mcp request 500s with "Task group is
        # not initialized. Make sure to use run()." -- the MCP session
        # manager's own background task group is only entered by its
        # lifespan context, which nothing was invoking.
        mcp_app = self._mcp.streamable_http_app(
            transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False)
        )

        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            async with AsyncExitStack() as stack:
                await stack.enter_async_context(mcp_app.router.lifespan_context(mcp_app))
                yield

        app = FastAPI(lifespan=lifespan)
        app.include_router(self._dispatcher.build_router())
        app.include_router(build_control_router(self._ag, self._registry))

        # Mounted at "/" (NOT "/mcp") -- `streamable_http_app()` already
        # defines its own route at "/mcp" internally (confirmed directly:
        # `MCPServer(...).streamable_http_app().routes` has exactly one
        # route, path "/mcp"), so mounting it AT "/mcp" would double up to
        # "/mcp/mcp". Mounting at "/" instead means a request to "/mcp"
        # resolves to the sub-app's own route, while every other path this
        # app already declared above still matches THIS app's own routes
        # first -- Starlette tries a FastAPI app's explicitly-declared
        # routes before falling through to a mount, and this mount is
        # added last.
        #
        # Reuses the SAME `mcp_app` object built above (not a second
        # `streamable_http_app()` call) -- that's the one instance whose
        # session-manager lifespan was actually entered by `lifespan()`
        # above; a second call would build an unrelated instance with its
        # own never-started task group, reintroducing the exact bug that
        # comment describes.
        #
        # Bridged into one shared socket rather than served as a separate
        # listener, so `agmanager_harness`'s UDS<->TCP bridge only needs to
        # know about ONE bridged socket for this agent's whole request
        # surface (dispatch/policy/logging/check-in AND the MCP tool
        # protocol), not two. DNS-rebinding protection is disabled
        # unconditionally (not just for the UDS listener, unlike
        # agmcp_server.py's split) -- this app is meant to be reached
        # primarily over the UDS bridge, where that middleware is
        # meaningless (see agmcp_server.py's own reasoning for the same
        # call); the bare-host TCP listener is already loopback-bound.
        app.mount("/", mcp_app)

        return app

    # -- lifecycle: main app (TCP for bare-host callers, UDS for container-
    # crossing) -- identical shape to the old singletons' start()/
    # ensure_uds_started()/stop() (see agllm_terminus.py for the reasoning:
    # uvicorn on a daemon thread, poll `server.started`, UDS survives a
    # restart at the same reserved path). ------------------------------------

    def start(self) -> str:
        if self.base_url is not None:
            return self.base_url
        fields = AgHostAgentManagerFields(self._agconfig)
        config = uvicorn.Config(self._app, host=fields.bind_host, port=0, log_level="warning")
        server = uvicorn.Server(config)
        self._server = server
        self._thread = threading.Thread(
            target=server.run, daemon=True, name=f"agmanager_host[{self._ag.agname}]"
        )
        self._thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not server.started:
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("agHostAgentManager server did not start within 10s")
        port = server.servers[0].sockets[0].getsockname()[1]
        self.base_url = f"http://{fields.bind_host}:{port}"
        return self.base_url

    def stop(self) -> None:
        with _live_managers_lock:
            agHostAgentManager._live_managers.pop(self.agent_id, None)
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._server = None
        self._thread = None
        self.base_url = None
        self.stop_uds()
        self._profiler.stop_uds()

    def ensure_uds_started(self) -> str:
        from ...agutil import reserve_uds_path, uds_listener_is_live

        if self.uds_path is not None:
            if uds_listener_is_live(self.uds_path, self._uds_thread):
                return self.uds_path
            self.stop_uds()
        sock_path = reserve_uds_path(self._uds_reserved_path, f"agmanager-host-{self._ag.agname}")
        self._uds_reserved_path = sock_path
        config = uvicorn.Config(self._app, uds=sock_path, log_level="warning")
        server = uvicorn.Server(config)
        self._uds_server = server
        self._uds_thread = threading.Thread(
            target=server.run,
            daemon=True,
            name=f"agmanager_host-uds[{self._ag.agname}]",
        )
        self._uds_thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not server.started:
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("agHostAgentManager UDS server did not start within 10s")
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

    # -- lifecycle: profiler ingest, delegated to ProfilerIngest (a SEPARATE
    # listener on purpose -- see module docstring). --------------------------

    def ensure_profiler_uds_started(self, timeout_s: float = 10) -> str:
        return self._profiler.ensure_uds_started(timeout_s)

    def stop_profiler_uds(self) -> None:
        self._profiler.stop_uds()

    @property
    def profiler_uds_path(self) -> "str | None":
        return self._profiler.uds_path


__all__ = ["agHostAgentManager"]
