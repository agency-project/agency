"""New: one container-side agent manager PER AGENT, assembled by
`build_app()` and run via `main()`.

Status: parallel scaffolding, not wired into any backend yet -- see
`agmanager_host/agmanager_host.py`'s module docstring for the full
two-server design this is half of, and for why NONE of the existing five
shared singletons (`agllm_terminus.py`, `agproxy_llm.py`, `agmcp_server.py`,
`agharness_messenger.py`, `agprof_ingest.py`) or backends (`native.py`,
`claude_code.py`, `codex.py`, `opencode.py`, `grok.py`) are touched by this
module.

**Runs as a standalone script inside a sandbox container, one instance per
agent** (unlike native.py's in-container entrypoint, which today only
serves native's own engine, this is meant to be reachable by ANY harness --
Claude Code, Codex, opencode, Grok, and eventually native itself once it
becomes a standalone harness binary, a separate later refactor). Launched
via `launcher.py`'s `launch_in_container()`, mirroring
`agharness_backends/native.py`'s `launch_in_container_entrypoint` foundation
(`exec_detached`, the bind-mounted `agency` package, a captured PID released
from monitoring) -- reused as a *pattern*, not imported, since this process
needs `fastapi`/`uvicorn`/`openai` on its own `PYTHONPATH` the way native's
deliberately stdlib-only entrypoint does not.

**Why this exists as a SEPARATE process from `agmanager_host`, not just a
client library the harness backend loads:** everything here either (a) is
only reachable from inside the container at all (a subprocess-based
permission hook, or a harness's `--mcp-config`/`ANTHROPIC_BASE_URL`, which
only understand a local `http://host:port`, never a host-side address or a
bind-mounted UDS path directly), or (b) is pure wire-format translation with
no host-only state, so there is no reason to pay a host-process credential
boundary for it.

**Module layout** -- each concern lives in its own sibling module, all
sharing one `_HostBridge`:
- `host_bridge.py` -- `_HostBridge`, the one bridged connection to
  `agmanager_host` (dispatch, model resolution, logging, policy, check-in,
  profiler-event forwarding). Everything else in this package calls
  through it, never a second path.
- `llm_routing.py` -- `/v1/chat/completions`, `/v1/messages`,
  `/v1/messages/count_tokens`, `/v1/responses` -- replaces the old
  `agproxy_llm.py`'s three-route translation layer (and the separate
  `agproxy_llm_in_container.py` launcher that stood up a second copy of
  that class inside the container -- not needed here, since this whole
  process already runs in-container unconditionally).
- `hooks_bridge.py` -- `/agpolicy/check_tool`, `/agprof/hook`,
  `/agprof/status`, `/internal/check_in` -- local endpoints a subprocess-
  based hook (or a future harness-independent pause-check loop) can reach;
  each just forwards to `agmanager_host`, replacing the identical routes
  on the old `agproxy_llm.py`'s gateway and `agharness_messenger.py`.
- `mcp_proxy.py` -- `/mcp` reverse proxy, replacing
  `agproxy_ptrace_internal`'s `start_tcp_relay`/`stop_tcp_relay` (a separate
  byte-relay subprocess) with an ordinary HTTP proxy route on this same
  process.
- `common.py` -- `extract_bearer_token`, shared by `llm_routing.py` and
  `hooks_bridge.py`.

**What is reused, unmodified, from the modules this is meant to eventually
replace, and why that's not a contradiction:** `agproxy_llm_adapters.py`'s
wire-format conversion functions (`anthropic_messages_to_openai`,
`openai_response_to_anthropic_message`, etc.) are a pure, stateless leaf
module -- no class state, no registry, nothing tying them to
`agproxy_llm.py`'s specific class. They were hardened against real CLI
behavior (see that module and `agproxy_llm.py`'s own docstrings) and are
exactly the kind of logic this redesign should keep, not rederive. Likewise
`_native_hooks.py`'s `hook_payload_to_syscallevent` (used
by `agmanager_host`'s `profiler_ingest.py`, not this module) is a pure hook-
JSON parser with no ties to any of the five old singletons. Nothing from
`agllm_terminus.py`/`agproxy_llm.py`/`agmcp_server.py`/
`agharness_messenger.py`/`agprof_ingest.py` themselves (the actual classes
being replaced) is imported anywhere in this design.

**What did NOT move here** (see `agmanager_host.py`'s docstring for the
full list): native's own ReAct loop, custom-tool (`add_tools`/
`replace_tools`) execution, Claude Code's `--resume` session continuity --
all engine-specific, out of scope for this harness-independent layer.

**Known, real gap, not hidden:** nothing outside native's own loop today
actually polls `/internal/check_in` -- making Claude Code/Codex/opencode/
Grok's own opaque internal loops check in mid-turn is a real, unsolved
problem (there is no hook point inside a third-party CLI's own loop to
inject this), not something this endpoint's mere existence fixes.
"""

from __future__ import annotations

import sys

import uvicorn
from fastapi import FastAPI

from . import hooks_bridge, llm_routing, mcp_proxy
from .host_bridge import _HostBridge


def build_app(bridge: "_HostBridge") -> FastAPI:
    app = FastAPI()
    app.include_router(llm_routing.build_router(bridge))
    app.include_router(hooks_bridge.build_router(bridge))
    app.include_router(mcp_proxy.build_router(bridge))
    return app


def main(argv: "list[str] | None" = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 3:
        raise SystemExit(
            "usage: python3 agmanager_harness.py <host-bridge-uds> "
            "<profiler-bridge-uds-or-dash> <port>"
        )
    bridge_uds, profiler_bridge, port_s = argv[0], argv[1], argv[2]
    profiler_uds = None if profiler_bridge == "-" else profiler_bridge
    bridge = _HostBridge(bridge_uds, profiler_uds)
    app = build_app(bridge)
    uvicorn.run(app, host="127.0.0.1", port=int(port_s), log_level="warning")


if __name__ == "__main__":
    main()
