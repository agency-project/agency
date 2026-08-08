"""Shared host-side MCP server (Phase 4 -- see docs/Design_harness_
integration.md's extension unifying native and harness-driven execution).

Exposes three kinds of tool, reached identically by every engine (external
harnesses via their own native MCP client support -- the whole reason MCP
is the right protocol here, since it's the one sanctioned "additive tool"
seam every one of them already speaks -- and native's own in-container
loop, wrapping the same calls): resource control (`reserve_cpu`/
`cpu_release`/`daemon_release`, mirroring `agency/tools/resource.py`'s
existing native-only closures), output submission (`submit_output`,
replacing per-engine output-schema handling with one mechanism), and
`ask_human` (mirroring `agency/tools/human.py`'s host-side tool -- shares
its actual blocking implementation via `ask_human_and_wait`, since a human
operator is inherently at the host, never inside the container).

Per-launch bearer token, same pattern as `agllm_terminus`/`agproxy_llm`:
`register(token, ag, skill)` before launch, read from the request's
`Authorization` header inside each tool call via MCP's `Context.headers`.
Deliberately a SEPARATE registry from `agllm_terminus`'s (not reusing it)
since this one keys on `(ag, skill)`, not just `ag` -- `submit_output`
needs the skill's own `output_schema` to validate against, which isn't a
property of the agent.

**Known, real gap, not hidden**: `reserve_cpu`/`cpu_release` (container-wide
docker/podman limit changes) and `daemon_release` (PID bookkeeping already
shared between ptrace-observed and native-observed processes) work
identically regardless of which engine calls them. GPU reservation does
NOT work end-to-end yet for either engine through this server -- the
existing native mechanism (`agency/tools/resource.py`'s `make_gpu_reserve`)
lazily injects `CUDA_VISIBLE_DEVICES` inside `agsandbox_backends/base.py`'s
`exec()`, a path only the HOST-side `agtool.py` dispatch goes through.
Neither `_native_in_container_entrypoint.py`'s own bash tool nor any
harness engine's own bash execution (which never calls `sandbox.exec()` at
all -- the harness/native-in-container process IS already inside the
container) consults that injection point. Making GPU reservation actually
visible to an in-container process's own subsequent commands needs a
different mechanism (e.g. a well-known env file the tool instructs the
model to source) -- not implemented here; `reserve_gpu` is deliberately
NOT exposed by this server yet rather than shipping a tool that appears to
work but silently doesn't.

**Second known gap, found while retiring `execute_react()`**: `submit_output`
only calls `output_schema.check_field()` (basic type/shape validation) --
unlike the old per-field `return_<field>` tool handler
(`agschema.make_field_handler`'s `_handle`), it does NOT call
`agtype_cls.validate_output(field_name, value, sandbox, exec_timeout)`
afterward. For a plain-typed field this is a no-op difference, but for an
agtype OUTPUT field (`agfile`/`agpath`/`agbinary`), that extra step is what
actually resolves a submitted path into real file content/validates it
exists -- so an agtype output field submitted via `submit_output` today
is NOT recovered the way it would be via the retired mechanism. Not fixed
here; the fields most affected are covered only by the (still execute_
react-only) input-side offload/recovery tests in test_agfile.py/
test_agpath.py/test_agbinary.py, not by any submit_output-based test.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, TYPE_CHECKING

from mcp.server.mcpserver import Context, MCPServer

from ..agconfig import GlobalConfigParam, _AgConfigViewBase

if TYPE_CHECKING:
    from ..agconfig import agConfig
    from ..agent import agent
    from ..agskill import agskill


class _AgMCPServerFields:
    bind_host = GlobalConfigParam("agmcp_server", default="127.0.0.1")


class agMCPServerConfig(_AgConfigViewBase):
    _OWNER = "agmcp_server"


def _extract_bearer_token(headers) -> "str | None":
    if not headers:
        return None
    auth = headers.get("authorization") or headers.get("Authorization")
    if not auth:
        return None
    if auth.lower().startswith("bearer "):
        return auth[len("Bearer ") :].strip()
    return auth.strip()


class agMCPServer:
    """One process-wide MCP server, shared across every engine (native and
    harness-driven alike) the same way `agllm_terminus`/`agproxy_llm`'s
    shared instances are. See module docstring for the tool set and the
    known GPU-reservation gap."""

    def __init__(self, agconfig: "agConfig | None" = None) -> None:
        self._agconfig = agconfig
        self._lock = threading.Lock()
        self._sessions_by_token: "dict[str, tuple[agent, agskill]]" = {}
        self._collected_outputs: "dict[str, dict[str, object]]" = {}
        self._server = MCPServer(name="agency-mcp")
        self._register_tools()
        self._uvicorn_server = None
        self._thread: "threading.Thread | None" = None
        self.base_url: "str | None" = None
        # UDS listener -- reachable from inside a container the same way
        # agllm_terminus's is (a bind-mounted directory every
        # container-backed sandbox already carries), for native.py's
        # in-container entrypoint and any harness whose own MCP client
        # supports a Unix socket transport. Independent of start()/base_url,
        # started lazily.
        self._uds_server = None
        self._uds_thread: "threading.Thread | None" = None
        self.uds_path: "str | None" = None

    # -- token <-> (agent, skill) registry ---------------------------------

    def register(self, token: str, ag: "agent", skill: "agskill") -> None:
        with self._lock:
            self._sessions_by_token[token] = (ag, skill)
            self._collected_outputs[token] = {}

    def unregister(self, token: str) -> None:
        with self._lock:
            self._sessions_by_token.pop(token, None)
            self._collected_outputs.pop(token, None)

    def _session_for_token(self, token: "str | None"):
        if token is None:
            return None
        with self._lock:
            return self._sessions_by_token.get(token)

    def collected_output(self, token: str) -> dict:
        """Whatever fields have been submitted so far for this token's
        launch -- read by the caller (e.g. a harness backend's execute())
        once its process exits, replacing `agschema.validate_and_recover`'s
        post-hoc free-text parse of the harness's own final message."""
        with self._lock:
            return dict(self._collected_outputs.get(token, {}))

    # -- tools --------------------------------------------------------------

    def _register_tools(self) -> None:
        server = self._server

        @server.tool(structured_output=True)
        def reserve_cpu(
            cpus: "float | None" = None, memory: "str | None" = None, *, ctx: Context
        ) -> dict[str, Any]:
            """Boost CPU and/or memory limits for the current sandbox container
            before running compute-intensive work. Always call cpu_release when done."""
            token = _extract_bearer_token(ctx.headers)
            session = self._session_for_token(token)
            if session is None:
                return {"error": "unknown or missing token"}
            ag, _skill = session
            sandbox = ag.sandbox
            pool = ag.agresource_pool
            try:
                sandbox.update_limits(
                    cpus=float(cpus) if cpus is not None else None,
                    memory=memory,
                )
                acquired_cpus = float(cpus) if cpus is not None else 0.0
                acquired_mb = _parse_memory_mb(memory)
                sandbox._cpu_acquired += acquired_cpus
                sandbox._memory_acquired_mb += acquired_mb
                pool.notify_cpu_acquired(acquired_cpus, acquired_mb)
                return {"message": f"Resource limits updated: cpus={cpus}, memory={memory}"}
            except Exception as e:
                return {"error": str(e)}

        @server.tool(structured_output=True)
        def cpu_release(*, ctx: Context) -> dict[str, Any]:
            """Reset CPU and memory limits back to idle defaults after
            compute-intensive work."""
            token = _extract_bearer_token(ctx.headers)
            session = self._session_for_token(token)
            if session is None:
                return {"error": "unknown or missing token"}
            ag, _skill = session
            sandbox = ag.sandbox
            pool = ag.agresource_pool
            try:
                held_cpus = sandbox._cpu_acquired
                held_mb = sandbox._memory_acquired_mb
                sandbox.update_limits(cpus=pool.idle_cpus, memory=pool.idle_memory)
                sandbox._cpu_acquired = 0.0
                sandbox._memory_acquired_mb = 0
                pool.notify_cpu_released(held_cpus, held_mb)
                return {
                    "message": f"CPU/memory reset to idle: cpus={pool.idle_cpus}, "
                    f"memory={pool.idle_memory}"
                }
            except Exception as e:
                return {"error": str(e)}

        @server.tool(structured_output=True)
        def daemon_release(pid: int, *, ctx: Context) -> dict[str, Any]:
            """Release a background process (PID) from monitoring so the skill
            can complete without waiting for it -- for intentionally long-lived
            services (servers, monitors) that should keep running."""
            token = _extract_bearer_token(ctx.headers)
            session = self._session_for_token(token)
            if session is None:
                return {"error": "unknown or missing token"}
            ag, _skill = session
            ag.sandbox.release_daemon(pid)
            return {"message": f"PID {pid} released as daemon -- will not block skill completion"}

        @server.tool(structured_output=True)
        def submit_output(field: str, value: str, *, ctx: Context) -> dict[str, Any]:
            """Submit one required output field's value. `value` is the
            field's value encoded as a JSON literal (a quoted string for a
            string field, a bare number for int/float, `true`/`false` for
            bool) -- call once per required field. Call this for every
            required field before finishing."""
            token = _extract_bearer_token(ctx.headers)
            session = self._session_for_token(token)
            if session is None:
                return {"error": "unknown or missing token"}
            ag, skill = session
            output_schema = skill.output_schema
            if output_schema is None:
                return {"error": "this skill declares no output_schema -- nothing to submit"}
            if field not in output_schema._data:
                return {"error": f"unknown output field {field!r}"}

            try:
                parsed_value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                parsed_value = value  # a bare unquoted string is a common model mistake

            err = output_schema.check_field(field, parsed_value)
            if err is not None:
                return {"error": err}

            with self._lock:
                self._collected_outputs.setdefault(token, {})[field] = parsed_value
                collected = self._collected_outputs[token]

            required = set(output_schema._data.keys())
            still_missing = sorted(required - set(collected.keys()))
            return {"result": f"field {field!r} recorded", "still_missing": still_missing}

        @server.tool(structured_output=True)
        def ask_human(question: str, *, ctx: Context) -> dict[str, Any]:
            """Ask the human operator a question and wait for their reply.
            Use when you need information, a decision, or clarification
            that only a human can provide. Prefer autonomous action; only
            ask when genuinely blocked."""
            token = _extract_bearer_token(ctx.headers)
            session = self._session_for_token(token)
            if session is None:
                return {"error": "unknown or missing token"}
            ag, _skill = session
            # Unlike the host-side `ask_human` agtool (agency/tools/
            # human.py), this already has a live `ag` reference from the
            # token registry -- no Agent.all()-by-name lookup needed, that
            # convention only exists there because the tool object was
            # built before any specific agent/token was known.
            from ..tools.human import ask_human_and_wait, DEFAULT_TIMEOUT_S

            reply = ask_human_and_wait(ag.agname, question, DEFAULT_TIMEOUT_S, ag=ag)
            return {"reply": reply}

    # -- lifecycle ------------------------------------------------------------

    def start(self, timeout_s: float = 10) -> str:
        """Start the server on a background thread (streamable-HTTP
        transport, since every one of these harnesses' own MCP clients
        supports it); returns its base URL. Idempotent."""
        if self.base_url is not None:
            return self.base_url

        import uvicorn

        fields = _AgMCPServerFields()
        app = self._server.streamable_http_app(host=fields.bind_host)
        config = uvicorn.Config(app, host=fields.bind_host, port=0, log_level="warning")
        server = uvicorn.Server(config)
        self._uvicorn_server = server

        self._thread = threading.Thread(target=server.run, daemon=True, name="agmcp_server")
        self._thread.start()

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not server.started:
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("agMCPServer did not start within timeout")

        port = server.servers[0].sockets[0].getsockname()[1]
        self.base_url = f"http://{fields.bind_host}:{port}"
        return self.base_url

    def stop(self) -> None:
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._uvicorn_server = None
        self._thread = None
        self.base_url = None
        self.stop_uds()

    def ensure_uds_started(self, timeout_s: float = 10) -> str:
        """Start (idempotently) a second listener for the same MCP server
        bound to a Unix domain socket instead of TCP -- see class docstring."""
        if self.uds_path is not None:
            return self.uds_path

        import uuid

        import uvicorn

        from ..agutil import agharness_llm_gateway_dir

        sock_path = str(agharness_llm_gateway_dir() / f"agmcp_server-{uuid.uuid4().hex}.sock")
        # DNS-rebinding protection validates the Host header against an
        # allowed-hosts list built from the TCP `host=`/bound port (host
        # header including PORT, confirmed by trial against the real
        # middleware) -- meaningless for a Unix domain socket, which has no
        # TCP port at all and isn't reachable by an arbitrary network
        # client/browser in the first place (only a process with filesystem
        # access to the bind-mounted socket path can connect at all, the
        # actual protection a UDS-only channel already gets for free).
        # Disabling this middleware here, not just choosing a matching Host
        # value, is the correct fix rather than a workaround.
        from mcp.server.transport_security import TransportSecuritySettings

        app = self._server.streamable_http_app(
            transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False)
        )
        config = uvicorn.Config(app, uds=sock_path, log_level="warning")
        server = uvicorn.Server(config)
        self._uds_server = server

        self._uds_thread = threading.Thread(target=server.run, daemon=True, name="agmcp_server-uds")
        self._uds_thread.start()

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not server.started:
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("agMCPServer UDS listener did not start within timeout")

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


def _parse_memory_mb(mem: "str | None") -> int:
    """Parse a Docker-style memory string to MB -- copied from
    agency/tools/resource.py's identical helper rather than imported, since
    that module's own version is wired to the native in-process tool
    factory pattern this server intentionally does not depend on."""
    if not mem:
        return 0
    s = str(mem).lower().strip()
    try:
        if s.endswith("g"):
            return int(float(s[:-1]) * 1024)
        if s.endswith("m"):
            return int(float(s[:-1]))
        return int(s) // (1024 * 1024)
    except (ValueError, AttributeError):
        return 0


_shared_server: "agMCPServer | None" = None
_shared_server_lock = threading.Lock()


def get_shared_mcp_server(agconfig: "agConfig | None" = None) -> agMCPServer:
    """One `agMCPServer` per process, mirroring `agllm_terminus.get_shared_terminus`."""
    global _shared_server
    if _shared_server is not None:
        return _shared_server
    with _shared_server_lock:
        if _shared_server is None:
            _shared_server = agMCPServer(agconfig)
        return _shared_server


__all__ = ["agMCPServer", "agMCPServerConfig", "get_shared_mcp_server"]
