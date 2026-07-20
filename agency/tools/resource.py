from __future__ import annotations

from typing import TYPE_CHECKING
from ..agdata import agdata, agerror
from ..agutil import format_exception
from ..agtool import agtool

if TYPE_CHECKING:
    from ..agsandbox import agSandbox
    from ..agresources import agResourcePool


def _parse_memory_mb(mem: "str | None") -> int:
    """Parse a Docker-style memory string to MB (e.g. '8g' → 8192, '512m' → 512)."""
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


def make_gpu_reserve(sandbox: "agSandbox", pool: "agResourcePool") -> agtool:
    """Return a tool that reserves GPU access for the sandbox.

    No physical GPU is claimed here. The physical allocation happens lazily
    inside sandbox.exec() the moment a bash command is actually run, and is
    held for the rest of the sandbox's lifetime -- there is no agent-facing
    release tool. It is freed automatically when the sandbox stops (container
    exit / no live chroot PIDs), tying the real GPU semaphore's lifetime to
    the sandbox's own lifetime rather than to the agent remembering to call
    a release tool mid-skill.
    """

    def _run(arg: agdata) -> agdata:
        if not pool.gpus:
            return agdata(
                warning=(
                    "No GPUs are available on this machine. "
                    "Bash commands will run without GPU acceleration."
                )
            )
        if sandbox._gpu_virtual:
            return agdata(message="GPU already acquired — use cuda:0 in your scripts")
        sandbox._gpu_virtual = True
        sandbox._gpu_acquire_fn = pool.acquire_gpu
        sandbox._gpu_release_fn = pool.release_gpu
        return agdata(message="GPU acquired — use cuda:0 in your scripts")

    return agtool(
        name="reserve_gpu",
        fn=_run,
        description=(
            "Reserve GPU access before running commands with GPU acceleration. "
            "CUDA_VISIBLE_DEVICES is set automatically — always use cuda:0 inside your scripts. "
            "The GPU is held for the rest of this sandbox's lifetime and freed "
            "automatically when the sandbox finishes -- there is no separate release step."
        ),
        params={"type": "object", "properties": {}},
        run_in_subprocess=False,
    )


def make_cpu_reserve(sandbox: "agSandbox", pool: "agResourcePool") -> agtool:
    def _run(arg: agdata) -> agdata:
        cpus: float | None = getattr(arg, "cpus", None)
        memory: str | None = getattr(arg, "memory", None)
        if cpus is None and memory is None:
            return agerror("specify at least one of: cpus, memory")

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
            return agdata(
                message=f"Resource limits updated: cpus={cpus}, memory={memory}",
                cpus=cpus,
                memory=memory,
            )
        except Exception as e:
            return agerror(format_exception(e))

    return agtool(
        name="reserve_cpu",
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
        run_in_subprocess=False,
    )


def make_cpu_release(sandbox: "agSandbox", pool: "agResourcePool") -> agtool:
    def _run(arg: agdata) -> agdata:
        try:
            held_cpus = sandbox._cpu_acquired
            held_mb = sandbox._memory_acquired_mb
            sandbox.update_limits(cpus=pool.idle_cpus, memory=pool.idle_memory)
            sandbox._cpu_acquired = 0.0
            sandbox._memory_acquired_mb = 0
            pool.notify_cpu_released(held_cpus, held_mb)
            return agdata(
                message=f"CPU/memory reset to idle: cpus={pool.idle_cpus}, memory={pool.idle_memory}"
            )
        except Exception as e:
            return agerror(format_exception(e))

    return agtool(
        name="cpu_release",
        fn=_run,
        description=(
            f"Reset CPU and memory limits back to idle defaults after compute-intensive work "
            f"(idle: {pool.idle_cpus} CPUs, {pool.idle_memory} memory)."
        ),
        params={"type": "object", "properties": {}},
        run_in_subprocess=False,
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
            return agerror("pid is required")
        try:
            pid = int(pid_val)
        except (TypeError, ValueError):
            return agerror(f"invalid pid: {pid_val!r}")
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
                "pid": {
                    "type": "integer",
                    "description": "PID of the process to release as a daemon",
                },
            },
            "required": ["pid"],
        },
        run_in_subprocess=False,
    )
