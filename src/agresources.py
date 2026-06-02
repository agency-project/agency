from __future__ import annotations

import os
import subprocess
import threading
import time


def detect_gpus() -> list[int]:
    """Return GPU IDs visible to nvidia-smi, or [] if none are found."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        pass
    return []


def detect_cpus() -> int:
    """Return the number of logical CPU cores on this host."""
    return os.cpu_count() or 1


def detect_memory_mb() -> int:
    """Return total host RAM in MB, read from /proc/meminfo (Linux) or via sysctl (macOS)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024   # kB → MB
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) // (1024 * 1024)
    except Exception:
        pass
    return 4096  # safe fallback


class agResourcePool:
    """Manages shared GPU tokens and CPU/memory limits for all sandboxes.

    All parameters are optional — call ``agResourcePool()`` with no arguments
    and GPUs, CPU count, and total memory are detected from the host
    automatically.  ``agent.agresource_pool`` is pre-set to a default instance,
    so you only need to construct one explicitly when you want to override the
    detected values.

    Usage::

        # Fully automatic — no configuration required:
        agent.agresource_pool = agResourcePool()

        # Override specific values:
        agent.agresource_pool = agResourcePool(gpus=[0], total_cpus=8, total_memory_mb=16384)

    ``total_cpus`` and ``total_memory_mb`` set the ceiling for ``cpu_acquire``
    (what an agent may request). ``idle_cpus`` / ``idle_memory`` are the limits
    applied when no work is running (restored by ``cpu_release``).
    """

    def __init__(
        self,
        gpus: list[int] | None = None,
        total_cpus: int | None = None,
        total_memory_mb: int | None = None,
        idle_cpus: float = 0.5,
        idle_memory: str = "512m",
    ) -> None:
        self.gpus = list(gpus) if gpus is not None else detect_gpus()
        self.total_cpus = total_cpus if total_cpus is not None else detect_cpus()
        self.total_memory_mb = total_memory_mb if total_memory_mb is not None else detect_memory_mb()
        self.idle_cpus = idle_cpus
        self.idle_memory = idle_memory
        self._gpu_locks: dict[int, threading.Semaphore] = {
            gpu_id: threading.Semaphore(1) for gpu_id in self.gpus
        }

    def acquire_gpu(self, timeout: float | None = None) -> int:
        """Block until any GPU is free; return its id."""
        deadline = None if timeout is None else time.monotonic() + timeout
        poll = 0.25

        while True:
            for gpu_id, sem in self._gpu_locks.items():
                if sem.acquire(blocking=False):
                    return gpu_id
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"No GPU available within {timeout}s "
                    f"(pool: {self.gpus})"
                )
            time.sleep(poll)

    def release_gpu(self, gpu_id: int) -> None:
        sem = self._gpu_locks.get(gpu_id)
        if sem is not None:
            try:
                sem.release()
            except ValueError:
                pass

    def __repr__(self) -> str:
        return (
            f"agResourcePool(gpus={self.gpus!r}, "
            f"total_cpus={self.total_cpus}, total_memory_mb={self.total_memory_mb}, "
            f"idle_cpus={self.idle_cpus}, idle_memory={self.idle_memory!r})"
        )
