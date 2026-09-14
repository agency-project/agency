"""Local hot-path measurements: python -m benchmarks.runtime_hotpaths.

No model, container, ptrace launch, or remote host is used. The measurement
calls the production registered MCP wrapper with a JSON-backed state factory.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

from agency import agdata, agtool
from agency.engine.host_servers.host_mcp_server import HostMcpServer


def persistent_state(iterations=1000):
    payload = json.dumps({"values": list(range(10000))})
    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return json.loads(payload)

    tool = agtool(
        "state",
        "",
        lambda arg, state: agdata(size=len(state["values"])),
        persistent_vars={"state": factory},
    )
    manager = HostMcpServer(None, SimpleNamespace(output_schema=None), None, None)
    callbacks = []
    manager._register_tool(
        SimpleNamespace(add_tool=lambda fn, **kwargs: callbacks.append(fn)), tool
    )
    started = time.perf_counter()
    for _ in range(iterations):
        assert callbacks[0]() == {"size": 10000}
    return {
        "calls": iterations,
        "factory_calls": factory_calls,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


if __name__ == "__main__":
    print(json.dumps(persistent_state(), indent=2))
