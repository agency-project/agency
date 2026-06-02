from __future__ import annotations

import threading
import time


class agResourcePool:
    """Manages shared GPU tokens and idle resource defaults for all sandboxes.

    Usage::

        pool = agResourcePool(gpus=[0, 1], idle_cpus=0.5, idle_memory="512m")
        agent.agresource_pool = pool

    Agents then get ``gpu_acquire`` / ``gpu_release`` / ``cpu_acquire`` /
    ``cpu_release`` tools automatically added to their tool list.
    """

    def __init__(
        self,
        gpus: list[int],
        idle_cpus: float = 0.5,
        idle_memory: str = "512m",
    ) -> None:
        self.gpus = list(gpus)
        self.idle_cpus = idle_cpus
        self.idle_memory = idle_memory
        self._gpu_locks: dict[int, threading.Semaphore] = {
            gpu_id: threading.Semaphore(1) for gpu_id in gpus
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
            f"idle_cpus={self.idle_cpus}, idle_memory={self.idle_memory!r})"
        )
