from __future__ import annotations

import atexit
import threading
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ..profiler import agprof
from ..agconfig import agConfig, StaticConfigParam, DynamicConfigParam, _AgConfigViewBase
from ..agname import agname as _agname
from .base import agsandbox_backend, backend_for_image_kind
from .container import _RUN_ID

if TYPE_CHECKING:
    from ..orchestrator.agresources import agResourcePool


# Exists to register agSandbox's config fields (via __set_name__ at import
# time) and hold their hardcoded defaults as plain class attributes --
# agSandbox inherits from this below, so self.base_image works via the
# inherited ConfigParam descriptor exactly as if declared directly on it.
# Only facade-level vocabulary lives here now (the image to run, and the
# mounts to attach) -- every backend-mechanics tunable (timeouts, retries,
# the docker/podman semaphore, ...) moved to agsandbox_backend.py's
# AgSandboxBackendFields, since those are meaningful only to a container-
# based backend, not to sandboxing in general.
class _AgSandboxFields:
    base_image = StaticConfigParam("agSandbox", default="agency-sandbox:latest")
    persistent = DynamicConfigParam(
        "agSandbox", default=False
    )  # If True, agtool.py:dispatch_tools() never hibernates (sandbox.stop())
    # this sandbox between tool calls within a skill run -- it only
    # stops/commits at the skill-run boundary (agskill.py). This trades away
    # GPU/keyring-slot release during the LLM "think" turn between tool
    # calls (today's whole reason dispatch_tools() hibernates) for a
    # container that stays warm for the entire skill call -- required once
    # an agent's LLM calls or harness process themselves run inside the
    # container rather than hopping in only for each tool call, since there
    # is then no safe point to stop the container without killing whatever
    # is making that call. Default False preserves today's per-tool-call
    # hibernate behavior unchanged.


class agSandboxConfig(_AgConfigViewBase):
    """View over an ``agConfig`` exposing ``agSandbox``'s own vocabulary
    (image, mounts) scoped to the ``"agSandbox"`` namespace.

    ``base_image`` is a flat registered field, so it's just sugar over the
    inherited ``update()``. ``mounts`` is a ``dict[str, tuple[str, str, str]]``
    built incrementally (``add_mount``/``remove_mount`` read-modify-write it)
    -- that doesn't fit ``update()``'s one-field-at-a-time-overwrite model, so
    it's layered on top here directly via the raw ``agConfig`` get/set calls,
    same as before this class subclassed ``_AgConfigViewBase``.
    """

    _OWNER = "agSandbox"

    @property
    def base_image(self) -> str:
        return self._agconfig.get_static(
            self._OWNER, "base_image", _AgSandboxFields.base_image.default
        )

    def set_base_image(self, image: str) -> "agSandboxConfig":
        return self.update(base_image=image)

    @property
    def mounts(self) -> dict[str, tuple[str, str, str]]:
        return self._agconfig.get_static(self._OWNER, "mounts", {})

    def add_mount(
        self, name: str, host_path, container_path: str, mode: str = "rw"
    ) -> "agSandboxConfig":
        # Raw (non-locking) read: this is a builder mutating a not-yet-consumed
        # config, not a consumer resolving it — going through the `mounts`
        # property here would lock the key via get_static() on its own first
        # call and then immediately fail the .set() below.
        current = self._agconfig.get(self._OWNER, "mounts", {})
        mounts = {**current, name: (str(host_path), container_path, mode)}
        self._agconfig.set(self._OWNER, "mounts", mounts)
        return self

    def remove_mount(self, name: str) -> "agSandboxConfig":
        current = self._agconfig.get(self._OWNER, "mounts", {})
        mounts = dict(current)
        mounts.pop(name, None)
        self._agconfig.set(self._OWNER, "mounts", mounts)
        return self


# Global registry of live sandboxes for atexit cleanup.
_live_sandboxes: weakref.WeakSet["agSandbox"] = weakref.WeakSet()


def _cleanup_all_sandboxes() -> None:
    """Destroy all live sandboxes on process exit."""
    for sandbox in list(_live_sandboxes):
        try:
            sandbox.destroy()
        except Exception as _e:
            # DATACOLLECTOR: append, agname=sandbox._agname -- exceptional (atexit teardown failure).
            print(f"[agsandbox] WARNING: atexit destroy failed for {sandbox._agname}: {_e}")


atexit.register(_cleanup_all_sandboxes)


class agSandbox(_AgSandboxFields):
    """Facade over one ``agsandbox_backend`` instance for one agent.

    Owns nothing about *how* isolation is achieved -- that's entirely the
    backend's job (see agsandbox_backend.py: a container today, potentially
    a chroot jail or another mechanism in the future, selected via
    ``agSandboxBackendConfig.backend``). This class resolves the image/mounts
    vocabulary that's meaningful regardless of backend, builds the backend via
    ``agsandbox_backend.for_config()``, and forwards every sandboxing
    operation to it.
    """

    def __init__(
        self,
        agname: str,
        checkpoint_image: str | None = None,
        agconfig: "agConfig | None" = None,
    ) -> None:
        with agprof.span("sandbox:create"):
            self._initialize(agname, checkpoint_image, agconfig)

    def _initialize(
        self,
        agname: str,
        checkpoint_image: str | None,
        agconfig: "agConfig | None",
    ) -> None:
        # Claimed from the SAME shared registry agent() uses (agname.py)
        # -- the "sandbox_" prefix guarantees this claim can never collide
        # with an agent's own agname. Without it, an agent named e.g.
        # "alex_0000" and a directly-constructed agSandbox("alex_0000", ...)
        # (bypassing agent entirely) would compute the IDENTICAL container
        # name and lifecycle tag within the same process -- _RUN_ID is a
        # process-wide constant, not object-specific, so nothing else
        # disambiguates them. allocate_agname() (not claim_unique_agname())
        # so this never raises, even when the same base *agname* is used to
        # construct multiple sandboxes -- each gets its own auto-suffixed
        # claim instead.
        self._agname = _agname.allocate_agname(f"sandbox_{agname}")
        # AgentEngine holds this for a complete execution so two agents sharing
        # this facade cannot interleave harness or teardown operations.
        self._lock = threading.RLock()
        self._destroyed = False
        # Cloned so this sandbox's own agconfig is independent of the
        # caller's -- mutating the caller's original agConfig afterward does
        # not affect this sandbox. To change it live, mutate
        # sandbox._agconfig (or one of its owner views) directly.
        self._agconfig: "agConfig | None" = agconfig.clone() if agconfig is not None else None

        # Container name is fixed at creation time so any explicitly serialized
        # or cross-process view still uses the same sandbox identity.
        self._name = f"sandbox-{_RUN_ID}-{self._agname}"

        # Resolve image/mounts once, here, rather than lazily -- a running
        # backend is physically fixed once created, so this is a tier-2
        # (object-static) read: it locks these keys on *this* agconfig
        # instance against further mutation. Mounts are handed to the backend
        # raw (host, container, mode) -- formatting them into a CLI flag (`-v
        # host:container:mode` for docker/podman, a bind mount for chroot) is
        # each backend's own business, not the facade's.
        base_image = self.base_image
        mounts: dict[str, tuple[str, str, str]] = {}
        for mount_name, (host, container, mode) in agSandboxConfig(self._agconfig).mounts.items():
            host_path = Path(host)
            host_path.mkdir(parents=True, exist_ok=True)
            mounts[mount_name] = (str(host_path.resolve()), container, mode)

        # Unconditional, harmless-if-unused default mount for a
        # docker/podman-backed harness launch's Unix-domain-socket host
        # services bridge (see agutil.agharness_llm_gateway_dir's docstring).
        # Added here rather than requiring each
        # harness backend to configure it per-agent, for the same reason
        # GPU passthrough flags are attached to every container
        # unconditionally (container.py's _gpu_flags): neither runtime
        # supports adding a bind mount to an already-created container, so
        # this must be present at creation time regardless of whether this
        # particular agent ever actually runs a harness. A bind mount is a
        # live view, not a copy -- the socket file inside this directory
        # doesn't need to exist yet.
        from ..utils.agutil import (
            AGENCY_PACKAGE_CONTAINER_MOUNT,
            agency_package_dir,
            agharness_binary_cache_dir,
            agharness_llm_gateway_dir,
        )

        mounts.setdefault(
            "_agharness_llm_gateway",
            (str(agharness_llm_gateway_dir()), "/var/run/agency_llm_gateway", "rw"),
        )
        # Same rationale, for the harness binary cache (see
        # agutil.agharness_binary_cache_dir's docstring): read-only,
        # since a container should never be able to write back into a
        # cache shared across every sandbox on this host.
        mounts.setdefault(
            "_agharness_bin_cache",
            (str(agharness_binary_cache_dir()), "/opt/agency_harness_bin", "ro"),
        )
        # Same rationale again, for the `agency` package itself (see
        # agutil.agency_package_dir's docstring) -- needed by a persistent
        # in-container entrypoint (agharness_backends/native.py's
        # react-loop process, or a container-relocated agproxy_llm) to
        # `import agency` and run the EXACT same code as the host process,
        # not a second copy baked into the sandbox's base image. Read-only,
        # same reasoning as the binary cache.
        mounts.setdefault(
            "_agency_package",
            (str(agency_package_dir()), AGENCY_PACKAGE_CONTAINER_MOUNT, "ro"),
        )

        self._backend = agsandbox_backend.for_config(
            self._agconfig,
            agname=self._agname,
            name=self._name,
            checkpoint_image=checkpoint_image,
            base_image=base_image,
            mounts=mounts,
        )
        _live_sandboxes.add(self)

    # ------------------------------------------------------------------
    # GPU/CPU bookkeeping -- pool coordination, not sandboxing mechanics, but
    # the backend's own exec() reads/writes these live (to gate GPU env-var
    # injection and release), so they're properties over the backend's state
    # rather than independent facade state. tools/resource.py sets these
    # directly on the sandbox instance (e.g. ``sandbox._gpu_count_requested = 1``).
    # ------------------------------------------------------------------

    @property
    def _gpu_ids(self) -> "list[int]":
        return self._backend._gpu_ids

    @_gpu_ids.setter
    def _gpu_ids(self, value: "list[int]") -> None:
        self._backend._gpu_ids = value

    @property
    def _gpu_count_requested(self) -> int:
        return self._backend._gpu_count_requested

    @_gpu_count_requested.setter
    def _gpu_count_requested(self, value: int) -> None:
        self._backend._gpu_count_requested = value

    @property
    def _gpu_acquire_fn(self):
        return self._backend._gpu_acquire_fn

    @_gpu_acquire_fn.setter
    def _gpu_acquire_fn(self, value) -> None:
        self._backend._gpu_acquire_fn = value

    @property
    def _gpu_release_fn(self):
        return self._backend._gpu_release_fn

    @_gpu_release_fn.setter
    def _gpu_release_fn(self, value) -> None:
        self._backend._gpu_release_fn = value

    def _own_host_pids(self) -> "set[int]":
        return self._backend._own_host_pids()

    def _has_pending_background_work(self) -> bool:
        return self._backend._has_pending_background_work()

    @property
    def _cpu_acquired(self) -> float:
        return self._backend._cpu_acquired

    @_cpu_acquired.setter
    def _cpu_acquired(self, value: float) -> None:
        self._backend._cpu_acquired = value

    @property
    def _memory_acquired_mb(self) -> int:
        return self._backend._memory_acquired_mb

    @_memory_acquired_mb.setter
    def _memory_acquired_mb(self, value: int) -> None:
        self._backend._memory_acquired_mb = value

    @property
    def _checkpoint_image(self) -> "str | None":
        return self._backend._checkpoint_image

    @_checkpoint_image.setter
    def _checkpoint_image(self, value: "str | None") -> None:
        self._backend._checkpoint_image = value

    @property
    def _watched_pids(self) -> dict[int, float]:
        return self._backend._watched_pids

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def change_config(self, agconfig: "agConfig | None") -> None:
        """Replace this sandbox's agconfig with a clone of the given one.

        Only affects fields read live going forward -- the backend's image
        and mounts were resolved once at construction (tier-2, physically
        fixed once the backend exists) and are not re-resolved here."""
        self._agconfig = agconfig.clone() if agconfig is not None else None
        self._backend.change_config(self._agconfig)

    def get_config_copy(self) -> "agConfig | None":
        """Return a clone of this sandbox's agconfig, or None if it has none."""
        return self._agconfig.clone() if self._agconfig is not None else None

    def __getstate__(self) -> dict:
        # threading.RLock is not picklable. A lock is process-local, so there is
        # nothing meaningful to carry if a caller explicitly serializes a sandbox.
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Sandboxing operations -- all forwarded straight to the backend.
    # ------------------------------------------------------------------

    def _container_exec(self, *args, **kwargs):
        with self._lock:
            return self._backend._container_exec(*args, **kwargs)

    def exec(self, *args, **kwargs):
        with self._lock:
            with agprof.span("sandbox:exec"):
                return self._backend.exec(*args, **kwargs)

    def exec_detached(self, *args, **kwargs) -> None:
        with self._lock:
            return self._backend.exec_detached(*args, **kwargs)

    def read_file(self, *args, **kwargs):
        with self._lock:
            with agprof.span("sandbox:read_file"):
                return self._backend.read_file(*args, **kwargs)

    def read_file_bytes(self, *args, **kwargs):
        with self._lock:
            with agprof.span("sandbox:read_file_bytes"):
                return self._backend.read_file_bytes(*args, **kwargs)

    def write_file(self, *args, **kwargs):
        with self._lock:
            with agprof.span("sandbox:write_file"):
                return self._backend.write_file(*args, **kwargs)

    def write_file_bytes(self, *args, **kwargs):
        with self._lock:
            with agprof.span("sandbox:write_file_bytes"):
                return self._backend.write_file_bytes(*args, **kwargs)

    def update_limits(self, *args, **kwargs) -> None:
        with self._lock:
            with agprof.span("sandbox:update_limits"):
                self._backend.update_limits(*args, **kwargs)

    def commit(self, *args, **kwargs) -> bool:
        with self._lock:
            with agprof.span("sandbox:commit"):
                return self._backend.commit(*args, **kwargs)

    def stop(self, *args, **kwargs) -> None:
        with self._lock:
            with agprof.span("sandbox:stop"):
                self._backend.stop(*args, **kwargs)

    def rm_container(self, *args, **kwargs) -> None:
        with self._lock:
            with agprof.span("sandbox:rm"):
                self._backend.rm_container(*args, **kwargs)

    def restore(self, *args, **kwargs) -> None:
        with self._lock:
            self._backend.restore(*args, **kwargs)

    def release_daemon(self, pid: int) -> None:
        self._backend.release_daemon(pid)

    def get_live_pids(self) -> set[int]:
        return self._backend.get_live_pids()

    def pid_status_summary(self) -> str:
        return self._backend.pid_status_summary()

    def release_resources(self, pool: "agResourcePool | None" = None) -> None:
        self._backend.release_resources(pool)

    def remove_files(self, paths: list[str]) -> None:
        self._backend.remove_files(paths)

    def __del__(self) -> None:
        # Python silently discards any exception raised out of __del__
        # anyway (printed as "Exception ignored in..." with no way for a
        # caller to observe it, since nothing is running a call stack that
        # could catch it) -- so this print is the only way this failure is
        # ever surfaced at all.
        try:
            self.destroy()
        except Exception as _e:
            # DATACOLLECTOR: append, agname=self._agname -- exceptional, and __del__'s only
            # surfacing mechanism (Python otherwise swallows the exception silently).
            print(f"[agsandbox] WARNING: destroy() failed during __del__ for {self._agname}: {_e}")

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        with agprof.span("sandbox:destroy"):
            _live_sandboxes.discard(self)
            self._backend.destroy()

    def fork(self, new_agname: str, agconfig: "agConfig | None" = None) -> "agSandbox":
        """Return a new agSandbox for *new_agname* starting from this sandbox's
        current checkpoint image.  If no checkpoint exists the fork starts fresh.

        *new_agname* doesn't need to already be unique -- like every
        agSandbox construction, it's automatically deduplicated (see
        __init__'s docstring below).

        When *agconfig* is not given, the fork inherits this sandbox's own
        agconfig unchanged (rather than silently re-reading whatever
        base_image happens to be at fork time).

        The caller owns the returned sandbox and is responsible for calling
        destroy() on it when done.
        """
        with agprof.span("sandbox:fork"):
            cfg = agconfig if agconfig is not None else self._agconfig
            fork_sb = agSandbox(new_agname, agconfig=cfg)
            checkpoint_image = self._backend._checkpoint_image
            if checkpoint_image:
                # type(self._backend), not the docker-only agSandbox.tag_image
                # static forwarder -- a chroot-backed sandbox's checkpoint is a
                # snapshot directory, not a docker/podman image tag, so it must
                # be retagged by the same backend class that created it.
                type(self._backend).tag_image(checkpoint_image, fork_sb._backend._lifecycle_tag())
            fork_sb._backend._checkpoint_image = fork_sb._backend._lifecycle_tag()
        return fork_sb

    @property
    def image_kind(self) -> str:
        """Identifies the checkpoint format this sandbox's backend uses
        ("container" or "chroot") -- see agsandbox_backend.backend_for_image_kind().
        agent.py's save() records this alongside a checkpoint so load() knows
        which backend's tag_image/export_image/import_image/delete_image can
        make sense of it."""
        return type(self._backend).IMAGE_KIND

    # ------------------------------------------------------------------
    # Static helpers — image-level operations used for checkpointing.
    # These bare forwarders always route to the container backend (they have
    # no instance/agconfig to resolve a backend from) -- callers that know
    # which backend a checkpoint came from should use backend_for_image_kind()
    # (agent.py's save()/load() do, via self.sandbox.image_kind), or, with a
    # live sandbox instance at hand, `type(sandbox._backend).tag_image(...)`
    # like fork() above.
    # ------------------------------------------------------------------

    @staticmethod
    def backend_for_image_kind(kind: str) -> type:
        return backend_for_image_kind(kind)

    @staticmethod
    def tag_image(source: str, dest: str) -> None:
        agsandbox_backend.tag_image(source, dest)

    @staticmethod
    def delete_image(tag: str, *, force: bool = False) -> None:
        agsandbox_backend.delete_image(tag, force=force)

    @staticmethod
    def export_image(tag: str, timeout: int) -> bytes:
        return agsandbox_backend.export_image(tag, timeout)

    @staticmethod
    def import_image(image_bytes: bytes, timeout: int) -> None:
        agsandbox_backend.import_image(image_bytes, timeout)

    def wait_for_processes(
        self,
        skill_name: str,
        agname: str = "",
        ping_interval_s: float = agsandbox_backend.WAIT_PING_INTERVAL_S,
        poll_interval_s: float = agsandbox_backend.WAIT_POLL_INTERVAL_S,
        state_fn: "Callable | None" = None,
    ) -> "str | None":
        """Wait for sandbox background processes after the LLM produces a final answer.

        Returns None if the sandbox is already clean (no action needed).
        Otherwise polls until all PIDs exit or ping_interval_s elapses, then
        returns a user-facing message to inject into the conversation so the
        LLM can act on the outcome.
        """
        _ping_interval_s = ping_interval_s
        _poll_interval_s = poll_interval_s
        _state = state_fn

        if self is None or not self._has_pending_background_work():
            return None

        summary = self.pid_status_summary()
        if _state:
            _state("proc_wait", skill=skill_name)

        import time

        # Poll _has_pending_background_work(), not just get_live() -- for
        # backends whose live-PID tracking can under-count (see
        # _ChrootBackend._has_pending_background_work()'s docstring), a
        # process invisible to get_live() can still be genuinely running;
        # relying on get_live() alone here would let this loop -- and the
        # "completed" determination right after it -- falsely conclude
        # nothing is left before it's actually finished.
        deadline = time.monotonic() + _ping_interval_s
        while time.monotonic() < deadline:
            time.sleep(_poll_interval_s)
            if not self._has_pending_background_work():
                break

        if not self._has_pending_background_work():
            return "Background processes have completed. Read their output and act on the results."

        summary = self.pid_status_summary()
        return (
            f"Background processes are still running: {summary}. "
            f"You may check their output, wait, or proceed if appropriate. "
            f"If any of these processes are intentional long-running services "
            f"(daemons, servers, monitors) that should not block completion, "
            f"call daemon_release(pid) for each such PID to release it from monitoring."
        )


# Re-exported for callers that imported these directly from agsandbox before
# the docker/podman mechanics moved to sandbox/.
from .container import (  # noqa: E402,F401
    get_container_runtime,
    seed_cache_from_image,
    keyring_quota,
)
