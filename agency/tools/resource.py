from __future__ import annotations

from typing import TYPE_CHECKING
from ..agdata import agdata, _fmt_exc
from ..agtool import agtool

if TYPE_CHECKING:
    from ..agsandbox import agSandbox
    from ..agresources import agResourcePool


def make_gpu_acquire(sandbox: "agSandbox", pool: "agResourcePool") -> agtool:
    """Return a tool that acquires exclusive GPU access for the sandbox."""
    def _run(arg: agdata) -> agdata:
        timeout: float | None = getattr(arg, "timeout", None)
        if timeout is not None:
            timeout = float(timeout)
        if sandbox._gpu_id is not None:
            return agdata(
                gpu_id=sandbox._gpu_id,
                message=f"GPU {sandbox._gpu_id} already held",
            )
        try:
            gpu_id = pool.acquire_gpu(timeout=timeout)
            sandbox._gpu_id = gpu_id
            return agdata(gpu_id=gpu_id, message=f"GPU {gpu_id} acquired")
        except TimeoutError as e:
            return agdata(error=_fmt_exc(e))

    return agtool(
        name="gpu_acquire",
        fn=_run,
        description=(
            "Acquire exclusive access to a GPU before running commands with GPU acceleration. "
            "Returns the assigned GPU ID. CUDA_VISIBLE_DEVICES is set automatically for "
            "all subsequent bash calls. Always call gpu_release when finished."
        ),
        params={
            "type": "object",
            "properties": {
                "timeout": {
                    "type": "number",
                    "description": "Max seconds to wait for a free GPU (default: wait indefinitely)",
                },
            },
        },
        need_sandbox=True,
    )


def make_gpu_release(sandbox: "agSandbox", pool: "agResourcePool") -> agtool:
    def _run(arg: agdata) -> agdata:
        if sandbox._gpu_id is None:
            return agdata(message="no GPU currently held")
        released = sandbox._gpu_id
        pool.release_gpu(released)
        sandbox._gpu_id = None
        return agdata(message=f"GPU {released} released")

    return agtool(
        name="gpu_release",
        fn=_run,
        description=(
            "Release the GPU acquired by gpu_acquire back to the shared pool. "
            "Call this as soon as GPU-intensive work is complete."
        ),
        params={"type": "object", "properties": {}},
        need_sandbox=True,
    )


def make_cpu_acquire(sandbox: "agSandbox") -> agtool:
    def _run(arg: agdata) -> agdata:
        cpus: float | None = getattr(arg, "cpus", None)
        memory: str | None = getattr(arg, "memory", None)
        if cpus is None and memory is None:
            return agdata(error="specify at least one of: cpus, memory")

        try:
            sandbox.update_limits(
                cpus=float(cpus) if cpus is not None else None,
                memory=memory,
            )
            return agdata(
                message=f"Resource limits updated: cpus={cpus}, memory={memory}",
                cpus=cpus,
                memory=memory,
            )
        except Exception as e:
            return agdata(error=_fmt_exc(e))

    return agtool(
        name="cpu_acquire",
        fn=_run,
        description=(
            "Boost CPU and/or memory limits for the current sandbox container "
            "before running compute-intensive work. Always call cpu_release when done."
        ),
        params={
            "type": "object",
            "properties": {
                "cpus": {
                    "type": "number",
                    "description": "Number of CPUs to allocate (e.g. 4.0)",
                },
                "memory": {
                    "type": "string",
                    "description": "Memory limit (e.g. '8g', '4096m')",
                },
            },
        },
        need_sandbox=True,
    )


def make_cpu_release(sandbox: "agSandbox", pool: "agResourcePool") -> agtool:
    def _run(arg: agdata) -> agdata:
        try:
            sandbox.update_limits(cpus=pool.idle_cpus, memory=pool.idle_memory)
            return agdata(
                message=f"CPU/memory reset to idle: cpus={pool.idle_cpus}, memory={pool.idle_memory}"
            )
        except Exception as e:
            return agdata(error=_fmt_exc(e))

    return agtool(
        name="cpu_release",
        fn=_run,
        description=(
            f"Reset CPU and memory limits back to idle defaults after compute-intensive work "
            f"(idle: {pool.idle_cpus} CPUs, {pool.idle_memory} memory)."
        ),
        params={"type": "object", "properties": {}},
        need_sandbox=True,
    )


def make_daemon_release(sandbox: "agSandbox") -> agtool:
    """Return a tool that releases a PID from monitoring as a daemon.

    Call this for intentionally long-lived services (servers, monitors) that
    should keep running after the skill finishes.  The process and all its
    descendants will continue running in the sandbox but will never block the
    outer monitoring loop.
    """
    def _run(arg: agdata) -> agdata:
        pid_val = getattr(arg, "pid", None)
        if pid_val is None:
            return agdata(error="pid is required")
        try:
            pid = int(pid_val)
        except (TypeError, ValueError):
            return agdata(error=f"invalid pid: {pid_val!r}")
        sandbox.release_daemon(pid)
        return agdata(message=f"PID {pid} released as daemon — will not block skill completion")

    return agtool(
        name="daemon_release",
        fn=_run,
        description=(
            "Release a background process as a daemon so the skill can complete "
            "without waiting for it. Use this for intentionally long-lived services "
            "(servers, monitors) that should keep running after the skill finishes. "
            "The process and all its descendants will continue running in the sandbox."
        ),
        params={
            "type": "object",
            "properties": {
                "pid": {"type": "integer", "description": "PID of the process to release as a daemon"},
            },
            "required": ["pid"],
        },
        need_sandbox=True,
    )
