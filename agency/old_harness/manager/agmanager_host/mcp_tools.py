"""Resource control (`reserve_cpu`/`cpu_release`/`daemon_release`), output
submission (`submit_output`), and human-in-the-loop (`ask_human`) MCP tools
for `agmanager_host`.

Same tool set/semantics as the old `agmcp_server.py`'s shared server -- see
that module's docstring for the known `reserve_gpu` / `submit_output`-vs-
`validate_output` gaps, unchanged here. See `agmanager_host.py`'s
module docstring for the full two-server design."""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from mcp.server.mcpserver import Context, MCPServer

if TYPE_CHECKING:
    from ...agent import agent
    from .launch_state import LaunchRegistry


def _extract_bearer_token(headers) -> "str | None":
    if not headers:
        return None
    auth = headers.get("authorization") or headers.get("Authorization") or headers.get("x-api-key")
    if not auth:
        return None
    if auth.lower().startswith("bearer "):
        return auth[len("Bearer ") :].strip()
    return auth.strip()


def _parse_memory_mb(mem: "str | None") -> int:
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


def build_mcp_server(ag: "agent", registry: "LaunchRegistry") -> MCPServer:
    server = MCPServer(name="agency-agent")

    @server.tool(structured_output=True)
    def reserve_cpu(
        cpus: "float | None" = None, memory: "str | None" = None, *, ctx: Context
    ) -> dict[str, Any]:
        """Boost CPU and/or memory limits for the current sandbox container
        before running compute-intensive work. Always call cpu_release when done."""
        token = _extract_bearer_token(ctx.headers)
        if registry.get(token) is None:
            return {"error": "unknown or missing token"}
        sandbox = ag.sandbox
        pool = ag.agresource_pool
        try:
            sandbox.update_limits(cpus=float(cpus) if cpus is not None else None, memory=memory)
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
        if registry.get(token) is None:
            return {"error": "unknown or missing token"}
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
        if registry.get(token) is None:
            return {"error": "unknown or missing token"}
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
        launch = registry.get(token)
        if launch is None:
            return {"error": "unknown or missing token"}
        output_schema = launch.skill.output_schema if launch.skill is not None else None
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

        with registry.lock:
            launch.collected_output[field] = parsed_value
            collected = dict(launch.collected_output)

        required = set(output_schema._data.keys())
        still_missing = sorted(required - set(collected.keys()))
        return {"result": f"field {field!r} recorded", "still_missing": still_missing}

    @server.tool(structured_output=True)
    def ask_human(question: str, *, ctx: Context) -> dict[str, Any]:
        """Ask the human operator a question and wait for their reply.
        Use when you need information, a decision, or clarification that
        only a human can provide. Prefer autonomous action; only ask when
        genuinely blocked."""
        token = _extract_bearer_token(ctx.headers)
        if registry.get(token) is None:
            return {"error": "unknown or missing token"}
        from ...tools.human import ask_human_and_wait, DEFAULT_TIMEOUT_S

        reply = ask_human_and_wait(ag.agname, question, DEFAULT_TIMEOUT_S, ag=ag)
        return {"reply": reply}

    return server


__all__ = ["build_mcp_server"]
