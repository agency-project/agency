"""Entrypoint executed INSIDE a container-backed sandbox to run
`agproxy_llm.py`'s FastAPI app as a persistent, local process -- the actual
relocation described in docs/Design_harness_integration.md, on top of the
launch+bridge foundation proven in agharness_backends/native.py.

Unlike `agharness_backends/_native_in_container_entrypoint.py`, this file
DOES import `agency` (via `-m` or direct import, doesn't matter which) --
`agproxy_llm.py` needs the full package (`agconfig`, `agllm_terminus`,
`agproxy_llm_adapters`), which transitively needs `openai`/`fastapi`/
`uvicorn` at import time. That only works because the launcher
(`agproxy_llm_in_container.ensure_agproxy_llm_in_container`) calls
`agutil.ensure_python_packages_in_container` first -- this file assumes
those are already installed, it doesn't install anything itself.

Binds a FIXED, well-known local port (see `_AGPROXY_LLM_IN_CONTAINER_PORT`
in agproxy_llm_in_container.py) rather than an OS-assigned one: this
process is launched detached (`exec_detached`, fire-and-forget, no stdout
the launcher can read back), and a fixed port is entirely safe since each
container has its own network namespace -- no cross-container collision
risk, unlike binding a fixed port on the host.
"""

from __future__ import annotations

import sys
import threading


def main(argv: "list[str] | None" = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 3:
        raise SystemExit(
            "usage: python3 _agproxy_llm_in_container_entrypoint.py "
            "<terminus-uds-path> <profiler-uds-path> <port>"
        )
    terminus_uds_path, profiler_uds_path, port = argv[0], argv[1], int(argv[2])
    if profiler_uds_path == "-":
        profiler_uds_path = None

    from agency.agconfig import agConfig
    from agency.agharness_internal.agproxy_llm import agProxyLLM, agProxyLLMConfig

    cfg = agConfig(agProxyLLMConfig(port=port))
    px = agProxyLLM(
        cfg,
        terminus_uds_path=terminus_uds_path,
        profiler_uds_path=profiler_uds_path,
    )
    px.start()

    # This process's only job is to keep that server alive -- start() itself
    # runs it on a background thread, so block forever here rather than
    # exiting (which would tear the process, and the server with it, down).
    threading.Event().wait()


if __name__ == "__main__":
    main()
