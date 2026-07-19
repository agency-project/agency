from __future__ import annotations

import ctypes
import os
import re
import subprocess
import threading
import time

from .agconfig import agConfig, GlobalConfigParam, DynamicConfigParam, _AgConfigViewBase


# Exists to register agResourcePool's config fields (via __set_name__ at
# import time). The detection/gating tunables are tier 1 (global): read once
# at process-wide resource-detection time, shared regardless of which
# agconfig (if any) is in play. idle_cpus/idle_memory are tier 3 (dynamic):
# a per-agent sandbox resource footprint, not a process-wide constant, so
# they're read fresh from whichever agConfig the consumer holds
# (agResourcePool itself, or a throwaway instance reading a sandbox's own
# agconfig -- see agsandbox.py's _ensure_started(), which also applies these
# as the container's starting limits, not just its idle-reset limits).
class _AgResourcePoolFields:
    gpu_detect_timeout_s = GlobalConfigParam(
        "agResourcePool", default=10
    )  # Seconds to wait for nvidia-smi/rocm-smi before giving up
    sysctl_detect_timeout_s = GlobalConfigParam(
        "agResourcePool", default=5
    )  # Seconds to wait for sysctl hw.memsize on macOS
    memory_detect_fallback_mb = GlobalConfigParam(
        "agResourcePool", default=4096
    )  # Safe fallback total RAM in MB when detection fails on both Linux and macOS
    gpu_acquire_poll_interval_s = GlobalConfigParam(
        "agResourcePool", default=0.25
    )  # Seconds between polls waiting for a free GPU semaphore
    marker_mb = GlobalConfigParam(
        "agResourcePool", default=128
    )  # VRAM held per GPU as a framework presence marker (visible in nvidia-smi)
    gpu_release_wait_poll_s = GlobalConfigParam(
        "agResourcePool", default=1.0
    )  # Seconds between nvidia-smi polls waiting for a GPU's compute processes to exit before release
    gpu_release_wait_timeout_s = GlobalConfigParam(
        "agResourcePool", default=30
    )  # Max seconds to wait for stragglers to exit before releasing anyway (with a warning)

    # CPU limit applied both when a sandbox container is first created (docker
    # run) and whenever it's reset to idle (docker update, via cpu_release) --
    # the same footprint at rest either way. idle_memory has no such fixed
    # cap by default (None): both container.py's creation path and
    # update_limits() treat None as "omit --memory", which is Docker's own
    # native unlimited behavior (cgroup memory.max="max") -- correct here
    # since sandboxes are torn down after use rather than reset-and-reused
    # indefinitely, so there's no multi-tenant idle container to bound.
    idle_cpus = DynamicConfigParam("agResourcePool", default=8.0)
    idle_memory = DynamicConfigParam("agResourcePool", default=None)

    def __init__(self, agconfig=None) -> None:
        self._agconfig = agconfig


class agResourcePoolConfig(_AgConfigViewBase):
    """View over an agConfig for pre-setting agResourcePool tunables in one call::

        cfg = agConfig(agResourcePoolConfig(gpu_detect_timeout_s=20))

    See `_AgConfigViewBase` in agconfig.py for the shared mechanics.
    """

    _OWNER = "agResourcePool"


def _visible_device_remap(env_names: "tuple[str, ...]") -> dict[int, int]:
    """Build a physical-id -> remapped-index map from whichever of env_names
    is set (e.g. CUDA_VISIBLE_DEVICES="0,3,5,7" -> {0:0, 3:1, 5:2, 7:3}),
    mirroring how the driver itself remaps physical GPUs to indices 0..N-1
    inside a process that only sees a restricted device list."""
    for name in env_names:
        val = os.environ.get(name, "")
        if not val or val.lower() in ("nodevfiles", "none"):
            continue
        try:
            ids = [int(x.strip()) for x in val.split(",") if x.strip().lstrip("-").isdigit()]
            if ids:
                return {phys: idx for idx, phys in enumerate(ids)}
        except Exception:
            pass
    return {}


def _allocate_gpu_markers_cuda(gpu_ids: list[int], marker_bytes: int) -> bool:
    """Try the CUDA driver API. Returns True if libcuda was found at all
    (regardless of whether individual per-GPU allocations went on to
    succeed) so the caller knows not to also try the ROCm/HIP path -- a
    cuInit failure means a broken/inaccessible NVIDIA driver, not "try AMD
    instead," since there's no AMD hardware to fall back to on an NVIDIA
    host anyway. Returns False only when libcuda.so.1 isn't present at all."""
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return False
    if cuda.cuInit(0) != 0:
        return True

    # When CUDA_VISIBLE_DEVICES is set (e.g. "0,3,5,7"), the CUDA driver
    # remaps physical GPUs to indices 0..N-1.  gpu_ids are physical IDs, so
    # we must convert to the remapped index before calling CUDA APIs.
    cuda_index = _visible_device_remap(("CUDA_VISIBLE_DEVICES",))

    for gpu_id in gpu_ids:
        try:
            dev = cuda_index.get(gpu_id, gpu_id)
            ctx = ctypes.c_void_p()
            ptr = ctypes.c_void_p()
            if cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) != 0:
                continue
            cuda.cuMemAlloc_v2(ctypes.byref(ptr), marker_bytes)
            # Leave context current; allocation persists for the process lifetime.
        except Exception:
            pass
    return True


def _allocate_gpu_markers_rocm(gpu_ids: list[int], marker_bytes: int) -> None:
    """ROCm/HIP equivalent of _allocate_gpu_markers_cuda. HIP's runtime API
    manages a context implicitly per device (hipSetDevice + hipMalloc)
    rather than CUDA driver API's explicit per-device context object, so
    there's no analogue of cuCtxCreate to call here."""
    try:
        hip = ctypes.CDLL("libamdhip64.so")
    except OSError:
        return
    if hip.hipInit(0) != 0:
        return

    hip_index = _visible_device_remap(("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))

    for gpu_id in gpu_ids:
        try:
            dev = hip_index.get(gpu_id, gpu_id)
            if hip.hipSetDevice(dev) != 0:
                continue
            ptr = ctypes.c_void_p()
            hip.hipMalloc(ctypes.byref(ptr), marker_bytes)
            # Leave allocated; persists for the process lifetime, same as the
            # CUDA path above.
        except Exception:
            pass


def _allocate_gpu_markers(gpu_ids: list[int]) -> None:
    """Allocate marker_mb of VRAM on each GPU directly in the calling process.

    Tries the CUDA driver API first, falling back to ROCm/HIP — no torch
    dependency required either way. Allocations live for the process
    lifetime, which is fine: the memory is tiny (128 MB per GPU by default)
    and there is no need to release it mid-run. Runs silently if neither
    CUDA nor ROCm is available.
    """
    marker_bytes = _AgResourcePoolFields().marker_mb * 1024 * 1024
    if _allocate_gpu_markers_cuda(gpu_ids, marker_bytes):
        return
    _allocate_gpu_markers_rocm(gpu_ids, marker_bytes)


def _nvidia_gpu_compute_pids(gpu_id: int) -> "set[int] | None":
    """Return host PIDs with an active CUDA context on physical GPU *gpu_id*,
    or None if nvidia-smi is unavailable/failed (including on non-NVIDIA
    hardware, where the binary doesn't exist at all)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_id),
            ],
            capture_output=True,
            text=True,
            timeout=_AgResourcePoolFields().gpu_detect_timeout_s,
        )
        if result.returncode != 0:
            return None
        return {int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()}
    except Exception:
        return None


_ROCM_PIDGPUS_HEADER_RE = re.compile(r"^PID (\d+) is using (\d+) DRM device\(s\)")


def _rocm_gpu_compute_pids(gpu_id: int) -> "set[int] | None":
    """Return host PIDs with an active KFD compute context on physical GPU
    *gpu_id*, or None if rocm-smi is unavailable/failed.

    ``rocm-smi --showpidgpus``'s "DRM device" number is rocm-smi's own
    device index -- the exact same index space as ``--showid``'s cardN (both
    ultimately come from the same `range(numberOfDevices)` list inside
    rocm-smi's own implementation), so it lines up directly with detect_gpus()'s
    gpu_id numbering with no render-node/topology-node/PCI translation needed.
    Deliberately not ``--json``/``--csv``: rocm-smi silently omits PID data
    under structured-output modes for this particular query.
    """
    try:
        result = subprocess.run(
            ["rocm-smi", "--showpidgpus"],
            capture_output=True,
            text=True,
            timeout=_AgResourcePoolFields().gpu_detect_timeout_s,
        )
        if result.returncode != 0:
            return None
    except Exception:
        return None

    pids: set[int] = set()
    lines = result.stdout.splitlines()
    i = 0
    while i < len(lines):
        m = _ROCM_PIDGPUS_HEADER_RE.match(lines[i].strip())
        if m is None:
            i += 1
            continue
        pid, n = int(m.group(1)), int(m.group(2))
        i += 1
        devices: list[int] = []
        while len(devices) < n and i < len(lines):
            devices.extend(int(tok) for tok in lines[i].split() if tok.lstrip("-").isdigit())
            i += 1
        if gpu_id in devices:
            pids.add(pid)
    return pids


def _gpu_compute_pids(gpu_id: int) -> "set[int] | None":
    """Return host PIDs with an active compute context on physical GPU
    *gpu_id*, trying nvidia-smi then falling back to rocm-smi (mirroring
    detect_gpus()'s own nvidia-then-rocm dispatch).

    Returns None (rather than an empty set) when neither query could be
    performed (no nvidia-smi/rocm-smi, unsupported hardware, timeout) so
    callers can tell "confirmed empty" apart from "couldn't check" and avoid
    blocking release on hardware where this check simply isn't possible.
    """
    pids = _nvidia_gpu_compute_pids(gpu_id)
    if pids is not None:
        return pids
    return _rocm_gpu_compute_pids(gpu_id)


def _cvd_filter(gpu_ids: list[int]) -> list[int]:
    """Filter gpu_ids to the subset allowed by CUDA_VISIBLE_DEVICES,
    HIP_VISIBLE_DEVICES, or ROCR_VISIBLE_DEVICES (whichever is set; checked
    in that order so a CUDA restriction always wins if somehow more than one
    is set at once)."""
    for _env in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        cvd = os.environ.get(_env, "")
        if not cvd or cvd.lower() in ("nodevfiles", "none"):
            continue
        try:
            allowed = {int(x.strip()) for x in cvd.split(",") if x.strip().lstrip("-").isdigit()}
            if allowed:
                return [g for g in gpu_ids if g in allowed]
        except Exception:
            pass
    return gpu_ids


def detect_gpus() -> list[int]:
    """Return GPU IDs visible to nvidia-smi or rocm-smi, filtered by CUDA_VISIBLE_DEVICES."""
    _timeout = _AgResourcePoolFields().gpu_detect_timeout_s
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=_timeout,
        )
        if result.returncode == 0 and result.stdout.strip():
            ids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
            return _cvd_filter(ids)
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["rocm-smi", "--showid", "--csv"],
            capture_output=True,
            text=True,
            timeout=_timeout,
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
                    return int(line.split()[1]) // 1024  # kB → MB
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=_AgResourcePoolFields().sysctl_detect_timeout_s,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) // (1024 * 1024)
    except Exception:
        pass
    return _AgResourcePoolFields().memory_detect_fallback_mb


class agResourcePool(_AgResourcePoolFields):
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
    (what an agent may request). ``idle_cpus``/``idle_memory`` are the
    resting-state limits -- applied both when a sandbox container is first
    created (see ``agsandbox.py``'s ``_ensure_started()``) and whenever it's
    reset to idle afterward (restored by ``cpu_release``). Both are
    ``DynamicConfigParam`` -- inherited from ``_AgResourcePoolFields``, so
    they're re-read live from whichever ``agconfig`` this pool holds; the
    keyword arguments below are just a convenience for setting them at
    construction without building an ``agResourcePoolConfig`` separately.
    """

    def __init__(
        self,
        gpus: list[int] | None = None,
        total_cpus: int | None = None,
        total_memory_mb: int | None = None,
        idle_cpus: float | None = None,
        idle_memory: str | None = None,
        mark_gpus: bool = False,
        agconfig: "agConfig | None" = None,
    ) -> None:
        self._agconfig = agconfig.clone() if agconfig is not None else agConfig()
        for _name, _value in (
            ("idle_cpus", idle_cpus),
            ("idle_memory", idle_memory),
        ):
            if _value is not None:
                self._agconfig.set("agResourcePool", _name, _value)
        self.gpus = list(gpus) if gpus is not None else detect_gpus()
        self.total_cpus = total_cpus if total_cpus is not None else detect_cpus()
        self.total_memory_mb = (
            total_memory_mb if total_memory_mb is not None else detect_memory_mb()
        )
        self._gpu_locks: dict[int, threading.BoundedSemaphore] = {
            gpu_id: threading.BoundedSemaphore(1) for gpu_id in self.gpus
        }
        self._res_lock = threading.Lock()
        self._gpus_acquired: int = 0
        self.cpus_acquired: float = 0.0
        self.memory_acquired_mb: int = 0
        if mark_gpus and self.gpus:
            import multiprocessing

            if multiprocessing.current_process().name == "MainProcess":
                _allocate_gpu_markers(self.gpus)

    def change_config(self, agconfig: "agConfig | None") -> None:
        """Replace this pool's agconfig with a clone of the given one."""
        self._agconfig = agconfig.clone() if agconfig is not None else agConfig()

    def get_config_copy(self) -> "agConfig":
        """Return a clone of this pool's agconfig."""
        return self._agconfig.clone()

    def acquire_gpu(self, timeout: float | None = None) -> int:
        """Block until any GPU is free; return its id."""
        deadline = None if timeout is None else time.monotonic() + timeout
        poll = _AgResourcePoolFields().gpu_acquire_poll_interval_s

        while True:
            for gpu_id, sem in self._gpu_locks.items():
                if sem.acquire(blocking=False):
                    with self._res_lock:
                        self._gpus_acquired += 1
                    self._emit_resource()
                    return gpu_id
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"No GPU available within {timeout}s (pool: {self.gpus})")
            time.sleep(poll)

    def _wait_for_gpu_clear(self, gpu_id: int, own_pids: "set[int] | None" = None) -> None:
        """Block until nvidia-smi/rocm-smi reports no relevant compute
        processes left on *gpu_id*. Guards against handing a "released" GPU
        to a new acquirer while a background job the releasing sandbox
        spawned (or a just-killed process's CUDA/HIP context) is still
        actually resident on the device -- see release_gpu()'s docstring.

        *own_pids*, when given, is the exact set of host PIDs the releasing
        sandbox itself spawned (see _ContainerBackendBase._own_host_pids()/
        _ChrootBackend._own_host_pids()) -- only THOSE PIDs are waited on, so
        an unrelated process sharing the same physical GPU (another tenant,
        another harness run entirely -- this pool has no cross-process
        visibility into those and isn't trying to coordinate with them)
        never counts as a straggler and never triggers a wait or a false
        warning.

        When own_pids is None (no sandbox context -- e.g. a caller other
        than a container/chroot-backed sandbox), falls back to the coarser
        "anything other than our own orchestrator PID" check.

        Also serves as the gap between release calls: a just-freed GPU can't
        be re-acquired any sooner than this check completes.

        Gives up and returns (so release still proceeds, with a warning)
        after gpu_release_wait_timeout_s -- a wedged/never-exiting straggler
        must not permanently strand the GPU as unreleasable. Returns
        immediately if the query itself isn't possible (no nvidia-smi /
        rocm-smi) since there's nothing to poll on.
        """
        exclude = {os.getpid()}
        poll = self.gpu_release_wait_poll_s
        deadline = time.monotonic() + self.gpu_release_wait_timeout_s
        while True:
            pids = _gpu_compute_pids(gpu_id)
            if pids is None:
                return
            stragglers = (pids & own_pids) if own_pids is not None else (pids - exclude)
            if not stragglers:
                return
            if time.monotonic() >= deadline:
                print(
                    f"[agresources] WARNING: gpu_id={gpu_id} still shows compute "
                    f"processes {sorted(stragglers)} after {self.gpu_release_wait_timeout_s}s; "
                    "releasing anyway"
                )
                return
            time.sleep(poll)

    def release_gpu(self, gpu_id: int, own_pids: "set[int] | None" = None) -> None:
        sem = self._gpu_locks.get(gpu_id)
        if sem is not None:
            self._wait_for_gpu_clear(gpu_id, own_pids)
            try:
                sem.release()
            except ValueError as _e:
                print(
                    f"[agresources] WARNING: GPU semaphore double-release for gpu_id={gpu_id}: {_e}"
                )
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
        except Exception as _e:
            print(f"[agresources] WARNING: resource_update push failed: {_e}")

    def __repr__(self) -> str:
        return (
            f"agResourcePool(gpus={self.gpus!r}, "
            f"total_cpus={self.total_cpus}, total_memory_mb={self.total_memory_mb}, "
            f"idle_cpus={self.idle_cpus}, idle_memory={self.idle_memory!r})"
        )
