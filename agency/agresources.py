from __future__ import annotations

import ctypes
import os
import subprocess
import threading
import time


# VRAM held per GPU as a framework presence marker (visible in nvidia-smi).
_MARKER_MB = 128
_MARKER_BYTES = _MARKER_MB * 1024 * 1024


def _allocate_gpu_markers(gpu_ids: list[int]) -> None:
    """Allocate _MARKER_MB of VRAM on each GPU directly in the calling process.

    Uses the CUDA driver API via ctypes — no torch dependency required.
    Allocations live for the process lifetime, which is fine: the memory is
    tiny (128 MB per GPU) and there is no need to release it mid-run.
    Runs silently if CUDA is unavailable.
    """
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return
    if cuda.cuInit(0) != 0:
        return

    # When CUDA_VISIBLE_DEVICES is set (e.g. "0,3,5,7"), the CUDA driver
    # remaps physical GPUs to indices 0..N-1.  gpu_ids are physical IDs, so
    # we must convert to the remapped index before calling CUDA APIs.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cuda_index: dict[int, int] = {}
    if cvd and cvd.lower() not in ("nodevfiles", "none"):
        try:
            cvd_list = [int(x.strip()) for x in cvd.split(",") if x.strip().lstrip("-").isdigit()]
            cuda_index = {phys: idx for idx, phys in enumerate(cvd_list)}
        except Exception:
            pass

    for gpu_id in gpu_ids:
        try:
            dev = cuda_index.get(gpu_id, gpu_id)
            ctx = ctypes.c_void_p()
            ptr = ctypes.c_void_p()
            if cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) != 0:
                continue
            cuda.cuMemAlloc_v2(ctypes.byref(ptr), _MARKER_BYTES)
            # Leave context current; allocation persists for the process lifetime.
        except Exception:
            pass


def _cvd_filter(gpu_ids: list[int]) -> list[int]:
    """Filter gpu_ids to the subset allowed by CUDA_VISIBLE_DEVICES (if set)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cvd or cvd.lower() in ("nodevfiles", "none"):
        return gpu_ids
    try:
        allowed = {int(x.strip()) for x in cvd.split(",") if x.strip().lstrip("-").isdigit()}
        if allowed:
            return [g for g in gpu_ids if g in allowed]
    except Exception:
        pass
    return gpu_ids


def detect_gpus() -> list[int]:
    """Return GPU IDs visible to nvidia-smi or rocm-smi, filtered by CUDA_VISIBLE_DEVICES."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            ids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
            return _cvd_filter(ids)
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["rocm-smi", "--showid", "--csv"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            ids = []
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line or line.lower().startswith("device"):
                    continue
                # First column is "cardN" — extract the numeric suffix.
                col = line.split(",")[0].strip().lower()
                if col.startswith("card"):
                    try:
                        ids.append(int(col[4:]))
                    except ValueError:
                        pass
                else:
                    try:
                        ids.append(int(col))
                    except ValueError:
                        pass
            if ids:
                return _cvd_filter(ids)
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

    ``total_cpus`` and ``total_memory_mb`` set the ceiling for ``reserve_cpu``
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
        mark_gpus: bool = False,
    ) -> None:
        self.gpus = list(gpus) if gpus is not None else detect_gpus()
        self.total_cpus = total_cpus if total_cpus is not None else detect_cpus()
        self.total_memory_mb = total_memory_mb if total_memory_mb is not None else detect_memory_mb()
        self.idle_cpus = idle_cpus
        self.idle_memory = idle_memory
        self._gpu_locks: dict[int, threading.Semaphore] = {
            gpu_id: threading.Semaphore(1) for gpu_id in self.gpus
        }
        self._res_lock = threading.Lock()
        self._gpus_acquired: int = 0
        self.cpus_acquired: float = 0.0
        self.memory_acquired_mb: int = 0
        if mark_gpus and self.gpus:
            import multiprocessing
            if multiprocessing.current_process().name == "MainProcess":
                _allocate_gpu_markers(self.gpus)

    def acquire_gpu(self, timeout: float | None = None) -> int:
        """Block until any GPU is free; return its id."""
        deadline = None if timeout is None else time.monotonic() + timeout
        poll = 0.25

        while True:
            for gpu_id, sem in self._gpu_locks.items():
                if sem.acquire(blocking=False):
                    with self._res_lock:
                        self._gpus_acquired += 1
                    self._emit_resource()
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
            with self._res_lock:
                self._gpus_acquired = max(0, self._gpus_acquired - 1)
            self._emit_resource()

    def notify_cpu_acquired(self, cpus: float, memory_mb: int) -> None:
        """Record that a sandbox boosted its CPU/memory limits."""
        with self._res_lock:
            self.cpus_acquired += cpus
            self.memory_acquired_mb += memory_mb
        self._emit_resource()

    def notify_cpu_released(self, cpus: float, memory_mb: int) -> None:
        """Record that a sandbox reset its CPU/memory limits to idle."""
        with self._res_lock:
            self.cpus_acquired = max(0.0, self.cpus_acquired - cpus)
            self.memory_acquired_mb = max(0, self.memory_acquired_mb - memory_mb)
        self._emit_resource()

    def _emit_resource(self) -> None:
        try:
            from . import agwebui as _agwebui
            if _agwebui._active is not None:
                _agwebui._active.emitter.resource_update(
                    gpus_acquired=self._gpus_acquired,
                    gpus_total=len(self.gpus),
                    cpus_acquired=self.cpus_acquired,
                    cpus_total=self.total_cpus,
                    memory_acquired_mb=self.memory_acquired_mb,
                    memory_total_mb=self.total_memory_mb,
                )
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"agResourcePool(gpus={self.gpus!r}, "
            f"total_cpus={self.total_cpus}, total_memory_mb={self.total_memory_mb}, "
            f"idle_cpus={self.idle_cpus}, idle_memory={self.idle_memory!r})"
        )
