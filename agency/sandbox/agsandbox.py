from __future__ import annotations

import atexit
import threading
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ..observability.profiler import agprof
from ..configs.agconfig import agconfig as agconfig_cls
from ..agname import agname as _agname
from .base import agsandbox_backend, backend_for_image_kind
from .container import _RUN_ID

if TYPE_CHECKING:
    from ..engine.clients import HarnessInteractionClient
    from ..orchestrator.agresources import agResourcePool


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


class agSandbox:
    """Facade over one ``agsandbox_backend`` instance for one agent.

    Owns nothing about *how* isolation is achieved -- that's entirely the
    backend's job (see agsandbox_backend.py: a container today, potentially
    a chroot jail or another mechanism in the future, selected via
    ``agconfig.sandbox.backend``). This class resolves the image/mounts vocabulary
    that's meaningful regardless of backend, builds the backend via
    ``agsandbox_backend.for_config()``, and forwards every sandboxing
    operation to it.
    """

    def __init__(
        self,
        name: str,
        checkpoint_image: str | None = None,
        agconfig: "agconfig_cls | None" = None,
    ) -> None:
        with agprof.span("sandbox:create"):
            self._initialize(name, checkpoint_image, agconfig)

    def _initialize(
        self,
        name: str,
        checkpoint_image: str | None,
        agconfig: "agconfig_cls | None",
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
        self._agname = _agname.allocate_agname(name, prefix="sandbox")
        # AgentEngine holds this for a complete execution so two agents sharing
        # this facade cannot interleave harness or teardown operations.
        self._lock = threading.RLock()
        self._destroyed = False
        # Cloned so this sandbox's own agconfig is independent of the
        # caller's -- mutating the caller's original agconfig afterward does
        # not affect this sandbox. To change it live, use change_config().
        self.agconfig: "agconfig_cls" = agconfig.clone() if agconfig is not None else agconfig_cls()

        # Container name is fixed at creation time so any explicitly serialized
        # or cross-process view still uses the same sandbox identity.
        self._name = f"sandbox-{_RUN_ID}-{self._agname}"

        # Resolve image/mounts once, here, rather than lazily -- a running
        # backend is physically fixed once created; agconfig.sandbox.base_image/
        # mounts aren't re-read after this. Mounts are handed to the backend
        # raw (host, container, mode) -- formatting them into a CLI flag (`-v
        # host:container:mode` for docker/podman, a bind mount for chroot) is
        # each backend's own business, not the facade's.
        base_image = self.agconfig.sandbox.base_image
        mounts: dict[str, tuple[str, str, str]] = {}
        for mount_name, (host, container, mode) in self.agconfig.sandbox.mounts.items():
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
            _DEFAULT_LOG_DIR,
            AGENCY_LLM_GATEWAY_CONTAINER_MOUNT,
            AGENCY_LOGS_CONTAINER_MOUNT,
            AGENCY_PACKAGE_CONTAINER_MOUNT,
            agency_package_dir,
            agharness_llm_gateway_dir,
        )

        mounts.setdefault(
            "_agharness_llm_gateway",
            (str(agharness_llm_gateway_dir()), AGENCY_LLM_GATEWAY_CONTAINER_MOUNT, "rw"),
        )
        # Same rationale, for this agent's logs/ directory (the harness
        # daemon's crash log, see harness_daemon_launcher.py). Derived from
        # db_path's parent, not _DEFAULT_LOG_DIR, so it follows a caller-set
        # agconfig.agent.log_dir; falls back for a sandbox built outside
        # agent.py (e.g. a test).
        _log_dir = (
            Path(self.agconfig.data_logger.db_path).parent
            if self.agconfig.data_logger.db_path
            else _DEFAULT_LOG_DIR
        )
        _log_dir.mkdir(parents=True, exist_ok=True)
        mounts.setdefault(
            "_agency_logs",
            (str(_log_dir), AGENCY_LOGS_CONTAINER_MOUNT, "rw"),
        )
        from ..harness.executable import harness_installation_mounts

        for name, mount in harness_installation_mounts(self.agconfig).items():
            for existing in mounts.values():
                if existing[1] == mount[1] and existing != mount:
                    raise ValueError(f"Conflicting harness installation mount at {mount[1]}")
            mounts[name] = mount
        # Same rationale again, for the `agency` package itself (see
        # agutil.agency_package_dir's docstring) -- needed by a persistent
        # in-container entrypoint (agharness_backends/native.py's
        # react-loop process, or a container-relocated agproxy_llm) to
        # `import agency` and run the EXACT same code as the host process,
        # not a second copy baked into the sandbox's base image. Read-only,
        # same reasoning as the harness installation mounts.
        mounts.setdefault(
            "_agency_package",
            (str(agency_package_dir()), AGENCY_PACKAGE_CONTAINER_MOUNT, "ro"),
        )

        self._backend = agsandbox_backend.for_config(
            self.agconfig,
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

    def current_gpu_ids(self) -> "list[int] | None":
        """Currently-held GPU ids, or None if nothing was ever reserved."""
        if self._gpu_count_requested <= 0:
            return None
        return list(self._gpu_ids)

    def ensure_gpu_acquired(
        self, agname: str, *, is_cancelled: "Callable[[], bool] | None" = None
    ) -> None:
        """Physically acquire the reserved GPU(s), if not already held.
        Pauses agname's harness for the duration (best-effort)."""
        if self._gpu_count_requested <= 0 or self._gpu_ids or self._gpu_acquire_fn is None:
            return
        count = self._gpu_count_requested
        client = self._harness_daemon_client(agname)
        if client is None:
            self._gpu_ids = self._gpu_acquire_fn(count, is_cancelled=is_cancelled)
            return
        with client:
            try:
                client.pause_harness()
            except Exception as exc:
                print(f"[agsandbox] WARNING: pause_harness() failed for {agname}: {exc}")
            try:
                self._gpu_ids = self._gpu_acquire_fn(count, is_cancelled=is_cancelled)
            finally:
                try:
                    client.resume_harness()
                except Exception as exc:
                    print(f"[agsandbox] WARNING: resume_harness() failed for {agname}: {exc}")

    def _harness_daemon_client(self, agname: str) -> "HarnessInteractionClient | None":
        handles = getattr(self, "_agency_harness_daemon_handles", None)
        handle = handles.get(agname) if handles else None
        return handle.client(timeout_s=10) if handle is not None else None

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

    def change_config(self, agconfig: "agconfig_cls | None") -> None:
        """Replace this sandbox's agconfig with a clone of the given one.
        Only affects fields read live going forward -- image/mounts were
        resolved once at construction."""
        self.agconfig = agconfig.clone() if agconfig is not None else agconfig_cls()
        self._backend.change_config(self.agconfig)

    def get_config_copy(self) -> "agconfig_cls":
        """Return a clone of this sandbox's agconfig."""
        return self.agconfig.clone()

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

    def _register_harness_pid(self, pid: int, start_ticks: int) -> None:
        self._backend._register_harness_pid(pid, start_ticks)

    def get_live_pids(self) -> set[int]:
        return self._backend.get_live_pids()

    def pid_status_summary(self) -> str:
        return self._backend.pid_status_summary()

    def release_resources(self, pool: "agResourcePool | None" = None) -> None:
        self._backend.release_resources(pool)

    def remove_files(self, paths: list[str]) -> None:
        self._backend.remove_files(paths)

    def __del__(self) -> None:
        # __del__'s exceptions are silently swallowed by Python, so this
        # print is the only way a destroy() failure here ever surfaces.
        try:
            self.destroy()
        except Exception as _e:
            # DATACOLLECTOR: append, agname=self._agname -- exceptional, low priority.
            print(f"[agsandbox] WARNING: destroy() failed during __del__ for {self._agname}: {_e}")

    def destroy(self) -> None:
        with self._lock:
            if self._destroyed:
                return
            with agprof.span("sandbox:destroy"):
                self._backend.destroy()
                self._destroyed = True
                _live_sandboxes.discard(self)

    def fork(self, new_name: str, agconfig: "agconfig_cls | None" = None) -> "agSandbox":
        """Return a new agSandbox for *new_name* starting from this
        sandbox's checkpoint image (fresh if none exists; *new_name*
        auto-deduplicated). Without *agconfig*, inherits this sandbox's
        own config unchanged. Caller must destroy() the result."""
        with self._lock, agprof.span("sandbox:fork"):
            cfg = agconfig if agconfig is not None else self.agconfig
            fork_sb = agSandbox(new_name, agconfig=cfg)
            checkpoint_image = self._backend._checkpoint_image
            if checkpoint_image:
                # Backend's own class, not the docker-only forwarder: a chroot checkpoint isn't a docker/podman tag.
                type(self._backend).tag_image(checkpoint_image, fork_sb._backend._lifecycle_tag())
                fork_sb._backend._checkpoint_image = fork_sb._backend._lifecycle_tag()
        return fork_sb

    @property
    def image_kind(self) -> str:
        """Checkpoint format this backend uses ("container" or "chroot");
        agent.py's save()/load() use it to pick the matching backend."""
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
