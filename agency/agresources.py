from __future__ import annotations

import ctypes
import os
import re
import subprocess
import threading
import time

from .profiler import agprof
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
    marker_mb = GlobalConfigParam(
        "agResourcePool", default=128
    )  # VRAM held per GPU as a framework presence marker (visible in nvidia-smi)

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
        except Exception as _e:
            print(f"[agresources] WARNING: could not parse {name}={val!r}: {_e}")
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
        except Exception as _e:
            print(f"[agresources] WARNING: CUDA marker allocation failed for GPU {gpu_id}: {_e}")
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
        except Exception as _e:
            print(f"[agresources] WARNING: ROCm marker allocation failed for GPU {gpu_id}: {_e}")


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
        except Exception as _e:
            print(f"[agresources] WARNING: could not parse {_env}={cvd!r}: {_e}")
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
    except Exception as _e:
        # Expected on any host without an NVIDIA driver/nvidia-smi installed --
        # falls through to the ROCm probe below.
        print(f"[agresources] nvidia-smi probe failed, trying rocm-smi: {_e}")
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
    except Exception as _e:
        # Expected on any host without an AMD driver/rocm-smi installed.
        print(f"[agresources] rocm-smi probe failed, no GPUs detected: {_e}")
    return []


# Matches the trailing PCI bus address segment of a resolved sysfs device
# path (e.g. ".../0000:75:00.0" -> "0000:75:00.0").
_PCI_BUS_RE = re.compile(r"([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f])$")


def _amd_render_node_pci_bus(name: str) -> "str | None":
    """Resolve the real PCI bus address (e.g. "0000:75:00.0") backing a
    /dev/dri render node by name (e.g. "renderD128"), by following
    /sys/class/drm/<name>/device. Returns None for XCD/compute-partition
    sibling nodes: MI300/MI350-class GPUs expose one render node per
    accelerator-complex-die under a "amdgpu_xcp_N" platform device even
    while the GPU itself is in unpartitioned (SPX) mode, and only the one
    node with a real PCI parent maps 1:1 to a physical GPU."""
    target = os.path.realpath(f"/sys/class/drm/{name}/device")
    match = _PCI_BUS_RE.search(target)
    return match.group(1).lower() if match else None


def amd_render_node_paths_by_pci_bus(candidates: "list[str]") -> "list[str] | None":
    """Reorder *candidates* (absolute /dev/dri/renderD* paths) so index N is
    the render node for rocm-smi's GPU N, by cross-referencing `rocm-smi
    --showbus` (GPU index -> PCI bus) against each candidate's own resolved
    PCI bus -- NOT by assuming sorted order already matches GPU index.

    Confirmed necessary on real 8x MI350X hardware: each GPU there exposes
    itself plus 7 XCD/compute-partition sibling render nodes (64 nodes total
    for 8 GPUs), and even the primary node's number doesn't sort in the same
    order as rocm-smi's GPU index -- e.g. GPU 3's real node was the
    numerically LOWEST of the 64 present, not the 4th. Naively picking
    sorted-index N (the old behavior) scoped several GPU IDs to nodes
    belonging to a different physical GPU entirely.

    Returns None (caller should fall back to naive sorted order) if
    `rocm-smi --showbus` fails/is unavailable, or any GPU's bus can't be
    matched to exactly one candidate -- a partial mapping is never applied,
    since trusting it for some GPUs and not others would be worse than the
    naive fallback it's meant to replace.
    """
    try:
        result = subprocess.run(
            ["rocm-smi", "--showbus", "--csv"],
            capture_output=True,
            text=True,
            timeout=_AgResourcePoolFields().gpu_detect_timeout_s,
        )
    except Exception:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None

    bus_by_gpu_id: "dict[int, str]" = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("device"):
            continue
        parts = [c.strip() for c in line.split(",")]
        if len(parts) < 2 or not parts[0].lower().startswith("card"):
            continue
        try:
            gpu_id = int(parts[0][4:])
        except ValueError:
            continue
        bus_by_gpu_id[gpu_id] = parts[1].lower()
    if not bus_by_gpu_id:
        return None

    render_by_bus: "dict[str, str]" = {}
    for path in candidates:
        bus = _amd_render_node_pci_bus(os.path.basename(path))
        if bus is not None:
            render_by_bus.setdefault(bus, path)

    ordered = []
    for gpu_id in range(max(bus_by_gpu_id) + 1):
        bus = bus_by_gpu_id.get(gpu_id)
        path = render_by_bus.get(bus) if bus is not None else None
        if path is None:
            return None
        ordered.append(path)
    return ordered


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
    except Exception as _e:
        # Expected on non-Linux hosts (e.g. macOS has no /proc) -- falls
        # through to the sysctl probe below.
        print(f"[agresources] /proc/meminfo read failed, trying sysctl: {_e}")
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=_AgResourcePoolFields().sysctl_detect_timeout_s,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) // (1024 * 1024)
    except Exception as _e:
        print(f"[agresources] sysctl memory probe failed, using configured fallback: {_e}")
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
        # A single Condition (rather than one BoundedSemaphore per GPU) so
        # release_gpu() can directly wake a waiter instead of every
        # acquire_gpu() call polling every GPU's own lock in a loop -- see
        # acquire_gpu()/release_gpu() docstrings for the full reasoning.
        # Guards only _free_gpus/_gpus_acquired -- NOT cpus_acquired/
        # memory_acquired_mb, which have no invariant linking them to GPU
        # state (a thread can reserve CPU while another concurrently
        # acquires a GPU with no interaction between the two), so they get
        # their own _cpu_mem_cond instead of sharing this one.
        self._gpu_cond = threading.Condition()
        self._free_gpus: set[int] = set(self.gpus)
        self._gpus_acquired: int = 0
        # Plain mutex for cpus_acquired/memory_acquired_mb -- a Condition
        # rather than a bare Lock only for consistency with _gpu_cond above;
        # nothing here ever calls wait()/notify().
        self._cpu_mem_cond = threading.Condition()
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
        """Block until any GPU is free; return its id.

        Waits on a single shared Condition rather than polling every GPU's
        own lock in a loop -- release_gpu() notifies exactly one waiter the
        moment a GPU frees up, instead of every waiter re-checking on a
        fixed timer (which wasted CPU/context-switches under contention and
        added up to one poll interval of latency before a freed GPU was
        even noticed).

        The `while not self._free_gpus` re-check after `wait()` returns is
        required, not defensive style: `notify()` only guarantees the
        woken thread gets a chance to recheck the condition, not that what
        it was waiting for is still there by the time it reacquires the
        lock -- another thread (a waiter woken earlier, or a fresh caller
        that never waited at all) can win the race and take the last free
        GPU first. `remaining` is recomputed from the original deadline on
        each iteration (not reset to a fresh `timeout`) so a caller that
        gets repeatedly out-raced still times out after its original
        budget, not a fresh one per iteration.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with agprof.span("sync:gpu_wait"), self._gpu_cond:
            while not self._free_gpus:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"No GPU available within {timeout}s (pool: {self.gpus})")
                if not self._gpu_cond.wait(timeout=remaining):
                    raise TimeoutError(f"No GPU available within {timeout}s (pool: {self.gpus})")
            gpu_id = self._free_gpus.pop()
            self._gpus_acquired += 1
        # Lease-interval event for the profiler: exclusive ownership of this
        # device starts now (device-scope; joined to spans at analysis time).
        agprof.gpu_lease_begin(gpu_id)
        self._emit_resource()
        return gpu_id

    def release_gpu(self, gpu_id: int) -> None:
        """Release *gpu_id* back to the pool.

        No separate "is this actually idle yet" wait: callers (backend
        `stop()`/`destroy()`) already run their own teardown -- kill
        watched processes, then remove the container / rmtree the chroot
        jail -- synchronously, BEFORE calling this. That ordering is
        already the confirmation that the sandbox has exited; a poll here
        would just be re-checking, via the exact same tracked state the
        teardown already acted on, something the sequencing already
        guarantees. (A prior version of this threaded an `is_clear`
        predicate through here for exactly that re-check; removed as
        redundant -- see agsandbox_backends.chroot's module docstring for
        the reasoning that led here.)

        Explicitly guards against gpu_id not being one of this pool's GPUs,
        and against double-releasing a gpu_id already in `_free_gpus` --
        neither is caught for free by a plain set the way a
        BoundedSemaphore used to reject an over-release on its own. The
        second check matters even though a set can't hold two copies of
        the same id: without it, a double-release (or releasing a gpu_id
        another sandbox still legitimately holds) would silently mark an
        in-use GPU as free, letting two sandboxes acquire the same physical
        GPU at once -- the actual hazard, not just a cosmetic duplicate
        entry.
        """
        with self._gpu_cond:
            if gpu_id not in self.gpus:
                print(f"[agresources] WARNING: release_gpu called with unknown gpu_id={gpu_id}")
                return
            if gpu_id in self._free_gpus:
                print(f"[agresources] WARNING: GPU double-release for gpu_id={gpu_id}")
                return
            self._free_gpus.add(gpu_id)
            self._gpus_acquired = max(0, self._gpus_acquired - 1)
            self._gpu_cond.notify()
        agprof.gpu_lease_end(gpu_id)
        self._emit_resource()

    def notify_cpu_acquired(self, cpus: float, memory_mb: int) -> None:
        """Record that a sandbox boosted its CPU/memory limits."""
        with self._cpu_mem_cond:
            self.cpus_acquired += cpus
            self.memory_acquired_mb += memory_mb
        self._emit_resource()

    def notify_cpu_released(self, cpus: float, memory_mb: int) -> None:
        """Record that a sandbox reset its CPU/memory limits to idle."""
        with self._cpu_mem_cond:
            self.cpus_acquired = max(0.0, self.cpus_acquired - cpus)
            self.memory_acquired_mb = max(0, self.memory_acquired_mb - memory_mb)
        self._emit_resource()

    def _emit_resource(self) -> None:
        """Push a resource_update to the webui dashboard, if active.

        Reads _gpus_acquired/cpus_acquired/memory_acquired_mb directly, with
        no lock -- this is a live status gauge for a dashboard badge, not a
        value anything computes with, so there's no invariant here worth
        guarding: a plain int/float attribute read is already atomic under
        the GIL (no torn reads), and the value is stale the instant it
        crosses into the emitter/websocket/browser regardless of whether a
        lock momentarily delayed a concurrent writer. Locking here would
        narrow that inevitable staleness window by nothing -- it would just
        relocate where the race happens, not remove it. (Contrast
        _free_gpus in acquire_gpu()/release_gpu(), which DOES need locking:
        two threads racing there can make two sandboxes believe they hold
        the same physical GPU -- an actual invariant, not a status number.)
        """
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
