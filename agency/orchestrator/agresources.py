from __future__ import annotations

import heapq
import itertools
import threading
import time
from typing import TYPE_CHECKING

from ..utils.agutil import (
    _AgResourcePoolFields,
    _allocate_gpu_markers,
    detect_cpus,
    detect_gpus,
    detect_memory_mb,
)
from ..profiler import agprof
from ..agconfig import agConfig, _AgConfigViewBase

if TYPE_CHECKING:
    from ..agdatacollector import agDataCollector


class agResourcePoolConfig(_AgConfigViewBase):
    """View over an agConfig for pre-setting agResourcePool tunables in one call::

        cfg = agConfig(agResourcePoolConfig(gpu_detect_timeout_s=20))

    See `_AgConfigViewBase` in agconfig.py for the shared mechanics.
    """

    _OWNER = "agResourcePool"


def _memory_mb_to_docker_str(memory_mb: "float | None") -> "str | None":
    if memory_mb is None:
        return None
    return f"{int(memory_mb)}m"


def _floor(value: "float | None", minimum: float) -> "float | None":
    """None (no cap) passes through unchanged; a real value is never let
    below *minimum*."""
    return None if value is None else max(value, minimum)


class _GpuRequest:
    __slots__ = ("count", "granted_ids")

    def __init__(self, count: int) -> None:
        self.count = count
        self.granted_ids: "list[int] | None" = None


class agResourcePool(_AgResourcePoolFields):
    """Manages shared GPU tokens and CPU/memory limits for all sandboxes.

    All parameters are optional — call ``agResourcePool()`` with no arguments
    and GPUs, CPU count, and total memory are detected from the host
    automatically. Owned by the process-wide orchestrator (constructed
    eagerly, see ``GlobalAgentOrchestrator.__init__``), so you only need to
    construct one explicitly when you want to override the detected values.

    Usage::

        # Fully automatic -- no configuration required, this is what the
        # orchestrator does by default:
        get_orchestrator().agresource_pool = agResourcePool()

        # Override specific values:
        get_orchestrator().agresource_pool = agResourcePool(
            gpus=[0], total_cpus=8, total_memory_mb=16384
        )

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
        data_collector: "agDataCollector | None" = None,
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
        self._gpu_request_seq = itertools.count()
        self._gpu_queue: "list[tuple[int, int, _GpuRequest]]" = []
        # Plain mutex for cpus_acquired/memory_acquired_mb -- a Condition
        # rather than a bare Lock only for consistency with _gpu_cond above;
        # nothing here ever calls wait()/notify().
        self._cpu_mem_cond = threading.Condition()
        self.cpus_acquired: float = 0.0
        self.memory_acquired_mb: int = 0
        # Composed in by the owner (GlobalAgentOrchestrator constructs both
        # eagerly and wires this one in) -- optional so a standalone pool
        # (tests, ad-hoc scripts) can skip resource-usage logging entirely
        # rather than needing a real collector just to exercise allocation
        # logic.
        self._data_collector = data_collector
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

    def acquire_gpus(self, sandbox, count: int, timeout: "float | None" = None) -> "list[int]":
        """Block until *count* GPUs are free; grant them to *sandbox* (setting
        its `_gpu_ids`) and return their ids.

        Queued (not just waited-on) so multiple concurrent requests for
        different counts get served smallest-count-first rather than
        strictly in arrival order: `_dispatch_gpu_queue_locked()` walks the
        queue in ascending count order and grants whichever prefix of it
        currently fits in `_free_gpus`, so a request for 1 GPU behind a
        queued request for 5 doesn't wait on the 5 to be satisfiable first.

        Raises ValueError immediately if `count` exceeds the pool's total
        size -- that's never satisfiable, so there's no reason to queue it.
        """
        if count <= 0:
            return []
        if count > len(self.gpus):
            raise ValueError(
                f"requested {count} GPUs but pool only has {len(self.gpus)} (pool: {self.gpus})"
            )
        deadline = None if timeout is None else time.monotonic() + timeout
        request = _GpuRequest(count)
        with agprof.span("sync:gpu_wait"), self._gpu_cond:
            seq = next(self._gpu_request_seq)
            heapq.heappush(self._gpu_queue, (count, seq, request))
            self._dispatch_gpu_queue_locked()
            while request.granted_ids is None:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    self._dequeue_gpu_request_locked(request)
                    raise TimeoutError(
                        f"No {count} GPU(s) available within {timeout}s (pool: {self.gpus})"
                    )
                if not self._gpu_cond.wait(timeout=remaining):
                    self._dequeue_gpu_request_locked(request)
                    raise TimeoutError(
                        f"No {count} GPU(s) available within {timeout}s (pool: {self.gpus})"
                    )
        for gpu_id in request.granted_ids:
            agprof.gpu_lease_begin(gpu_id)
        sandbox._gpu_ids = request.granted_ids
        self._update_resource_log(
            who=sandbox._name, action="acquire_gpu", count=count, gpu_ids=request.granted_ids
        )
        return request.granted_ids

    def _dequeue_gpu_request_locked(self, request: "_GpuRequest") -> None:
        if request.granted_ids is not None:
            return
        self._gpu_queue = [entry for entry in self._gpu_queue if entry[2] is not request]
        heapq.heapify(self._gpu_queue)

    def _dispatch_gpu_queue_locked(self) -> None:
        granted_any = False
        while self._gpu_queue:
            count, _seq, request = self._gpu_queue[0]
            if len(self._free_gpus) < count:
                break
            heapq.heappop(self._gpu_queue)
            request.granted_ids = [self._free_gpus.pop() for _ in range(count)]
            self._gpus_acquired += count
            granted_any = True
        if granted_any:
            self._gpu_cond.notify_all()

    def release_gpus(self, sandbox, gpu_ids: "list[int]") -> None:
        """Release *gpu_ids* (held by *sandbox*) back to the pool, clearing
        *sandbox*'s own `_gpu_ids`.

        No separate "is this actually idle yet" wait: callers (backend
        `stop()`/`destroy()`) already run their own teardown -- kill
        watched processes, then remove the container / rmtree the chroot
        jail -- synchronously, BEFORE calling this. That ordering is
        already the confirmation that the sandbox has exited; a poll here
        would just be re-checking, via the exact same tracked state the
        teardown already acted on, something the sequencing already
        guarantees.

        Explicitly guards against a gpu_id not being one of this pool's
        GPUs, and against double-releasing a gpu_id already in
        `_free_gpus` -- neither is caught for free by a plain set. The
        second check matters even though a set can't hold two copies of
        the same id: without it, a double-release (or releasing a gpu_id
        another sandbox still legitimately holds) would silently mark an
        in-use GPU as free, letting two sandboxes acquire the same physical
        GPU at once -- the actual hazard, not just a cosmetic duplicate
        entry.
        """
        if not gpu_ids:
            return
        with self._gpu_cond:
            for gpu_id in gpu_ids:
                if gpu_id not in self.gpus:
                    # DATACOLLECTOR: append -- process-level, real usage-bug signal.
                    print(
                        f"[agresources] WARNING: release_gpus called with unknown gpu_id={gpu_id}"
                    )
                    continue
                if gpu_id in self._free_gpus:
                    # DATACOLLECTOR: append -- process-level, real invariant-violation signal.
                    print(f"[agresources] WARNING: GPU double-release for gpu_id={gpu_id}")
                    continue
                self._free_gpus.add(gpu_id)
                self._gpus_acquired = max(0, self._gpus_acquired - 1)
            self._dispatch_gpu_queue_locked()
        for gpu_id in gpu_ids:
            agprof.gpu_lease_end(gpu_id)
        sandbox._gpu_ids = []
        self._update_resource_log(who=sandbox._name, action="release_gpu", gpu_ids=list(gpu_ids))

    def acquire_cpu_mem(
        self, sandbox, cpus: "float | None" = None, memory_mb: "float | None" = None
    ) -> None:
        """Boost *sandbox*'s CPU/memory limits (applying the min_cpus/
        min_memory_mb floor), update its `_cpu_acquired`/
        `_memory_acquired_mb` bookkeeping, and record the acquisition.

        A running container throttled to 0 cpu shares or 0 memory can't
        make forward progress, so a requested value is never let below the
        floor -- silently raised rather than rejected, since a too-small
        request is a caller mistake to correct for, not a real capacity
        constraint (see the total_cpus/total_memory_mb ceiling check, which
        IS a real rejection, in agskill.py's _reserve_resource).
        """
        if cpus is None and memory_mb is None:
            return
        applied_cpus = _floor(cpus, self.min_cpus)
        applied_memory_mb = _floor(memory_mb, self.min_memory_mb)
        sandbox.update_limits(cpus=applied_cpus, memory=_memory_mb_to_docker_str(applied_memory_mb))
        if applied_cpus is not None:
            sandbox._cpu_acquired += applied_cpus
        if applied_memory_mb is not None:
            sandbox._memory_acquired_mb += applied_memory_mb
        with self._cpu_mem_cond:
            self.cpus_acquired += applied_cpus or 0.0
            self.memory_acquired_mb += applied_memory_mb or 0
        self._update_resource_log(
            who=sandbox._name,
            action="acquire_cpu_mem",
            cpus=applied_cpus,
            memory_mb=applied_memory_mb,
        )

    def release_cpu_mem(self, sandbox, cpu: bool = False, memory: bool = False) -> None:
        """Reset *sandbox*'s CPU and/or memory limits to idle (flooring
        idle_cpus at min_cpus), update its `_cpu_acquired`/
        `_memory_acquired_mb` bookkeeping, and record the release.

        idle_memory is a pre-formatted docker string (e.g. "1g") or None
        (unlimited) -- unlike memory_mb on the acquire side, it's not a raw
        MB number, so it's passed straight through rather than floored."""
        if not (cpu or memory):
            return
        held_cpus = sandbox._cpu_acquired if cpu else 0.0
        held_mb = sandbox._memory_acquired_mb if memory else 0
        sandbox.update_limits(
            cpus=_floor(self.idle_cpus, self.min_cpus) if cpu else None,
            memory=self.idle_memory if memory else None,
        )
        if cpu:
            sandbox._cpu_acquired = 0.0
        if memory:
            sandbox._memory_acquired_mb = 0
        with self._cpu_mem_cond:
            self.cpus_acquired = max(0.0, self.cpus_acquired - held_cpus)
            self.memory_acquired_mb = max(0, self.memory_acquired_mb - held_mb)
        self._update_resource_log(
            who=sandbox._name, action="release_cpu_mem", cpus=held_cpus, memory_mb=held_mb
        )

    def _update_resource_log(
        self, *, who: "str | None" = None, action: "str | None" = None, **request
    ) -> None:
        dc = self._data_collector
        if dc is None:
            return
        dc.record_event(
            type="resource_update",
            payload={
                "gpus_acquired": self._gpus_acquired,
                "gpus_total": len(self.gpus),
                "cpus_acquired": self.cpus_acquired,
                "cpus_total": self.total_cpus,
                "memory_acquired_mb": self.memory_acquired_mb,
                "memory_total_mb": self.total_memory_mb,
            },
            overwrite=True,
        )
        dc.record_event(
            type="resource_request",
            payload={"who": who, "action": action, "request": request or None},
            overwrite=False,
            flush=True,
        )

    def __repr__(self) -> str:
        return (
            f"agResourcePool(gpus={self.gpus!r}, "
            f"total_cpus={self.total_cpus}, total_memory_mb={self.total_memory_mb}, "
            f"idle_cpus={self.idle_cpus}, idle_memory={self.idle_memory!r})"
        )
