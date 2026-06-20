from __future__ import annotations

import atexit
import os
import subprocess
import sys
import threading
import time

# VRAM held per GPU as a framework presence marker (visible in nvidia-smi).
_MARKER_MB = 128
_MARKER_BYTES = _MARKER_MB * 1024 * 1024

# Inline script run as a subprocess per GPU.  Sets its own process name to
# "agency-gpu" so it is identifiable in nvidia-smi and ps.  Uses the CUDA
# driver API (libcuda.so.1) directly — no torch dependency required.
_MARKER_SCRIPT = f"""\
import ctypes, time, sys

try:
    ctypes.CDLL(None).prctl(15, b'agency-gpu', 0, 0, 0)
except Exception:
    pass

try:
    cuda = ctypes.CDLL('libcuda.so.1')
    ctx  = ctypes.c_void_p()
    ptr  = ctypes.c_void_p()
    if cuda.cuInit(0) != 0:
        sys.exit(0)
    if cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, 0) != 0:
        sys.exit(0)
    if cuda.cuMemAlloc_v2(ctypes.byref(ptr), {_MARKER_BYTES}) != 0:
        sys.exit(0)
    time.sleep(1e9)
except Exception:
    sys.exit(0)
"""


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
        self._marker_procs: list[subprocess.Popen] = []
        if mark_gpus and self.gpus:
            # Only run markers in the main process.  agtool uses a
            # ProcessPoolExecutor whose workers also import agent.py, which
            # re-evaluates the class-level agresource_pool and would otherwise
            # spawn a full set of marker subprocesses in every worker.
            import multiprocessing
            if multiprocessing.current_process().name == "MainProcess":
                self._start_gpu_markers()

    def _start_gpu_markers(self) -> None:
        """Launch one background process per GPU that holds _MARKER_MB of VRAM.

        Each process names itself 'agency-gpu' via prctl so it appears clearly
        in nvidia-smi and ps.  Processes exit silently if torch or CUDA is
        unavailable.  All markers are terminated when the pool is garbage-
        collected or the process exits.
        """
        for gpu_id in self.gpus:
            try:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"]  = str(gpu_id)
                env["HIP_VISIBLE_DEVICES"]   = str(gpu_id)
                proc = subprocess.Popen(
                    [sys.executable, "-c", _MARKER_SCRIPT],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._marker_procs.append(proc)
            except Exception:
                pass
        atexit.register(self._stop_gpu_markers)

    def _stop_gpu_markers(self) -> None:
        """Terminate all GPU marker processes and reap them to avoid zombies."""
        for proc in self._marker_procs:
            try:
                proc.terminate()
            except Exception:
                pass
        for proc in self._marker_procs:
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        self._marker_procs.clear()

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
