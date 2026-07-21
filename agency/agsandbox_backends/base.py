"""Sandbox backend base class, shared config, and backend selection.

An `agSandbox` instance builds exactly one `agsandbox_backend` from its config
(via `agsandbox_backend.for_config()`) and delegates every sandboxing
operation (exec, file I/O, lifecycle, checkpointing) to it. Backend selection
logic (podman vs. docker vs. chroot, see `agSandboxBackendConfig.backend`)
lives here instead of being hardcoded into `agSandbox` itself.

Every backend exposes the same surface `agSandbox` uses: `exec()`,
`_container_exec()`, `read_file()`/`write_file()` (+ `_bytes` variants),
`commit()`/`stop()`/`restore()`/`destroy()`, `update_limits()`,
`remove_files()`, `get_live_pids()`/`pid_status_summary()`/`release_daemon()`,
`release_resources()`, `fork()`, and the static image-level helpers
`tag_image()`/`export_image()`/`import_image()`/`delete_image()`.

The concrete backends themselves live in sibling modules -- `.container`
(shared docker/podman plumbing), `.docker`, `.podman`, `.chroot` -- each
importing `agsandbox_backend` from here to subclass it. Every reference back
from *this* module to *those* is therefore a lazy, function-local import
(inside `for_config()`/`_auto_detect_runtime()`/`backend_for_image_kind()`
below) rather than a module-level one, to avoid a circular import.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as _FutureTimeoutError
from typing import TYPE_CHECKING, Callable

from ..agconfig import agConfig, GlobalConfigParam, DynamicConfigParam, _AgConfigViewBase

if TYPE_CHECKING:
    from ..agresources import agResourcePool


# ---------------------------------------------------------------------------
# Backend config -- every tunable used by any backend, as ConfigParam
# descriptors. `backend` (podman/docker/chroot/auto) is the analogue of
# agllm_backend's `provider`. The rest are tier-1 (global) tunables for the
# backend machinery itself (docker/podman daemon calls, the container
# semaphore, the keyring quota) -- unchanged in kind from when they lived on
# agsandbox.py's _AgSandboxFields, just moved to their own owner namespace.
# ---------------------------------------------------------------------------


class AgSandboxBackendFields:
    DEFAULT_EXEC_TIMEOUT_S = (
        120  # Fallback for exec/_container_exec when no explicit timeout is passed.
    )
    SEED_CACHE_TIMEOUT_S = (
        900  # seed_cache_from_image: copying a large pre-baked directory out of an image.
    )
    WAIT_PING_INTERVAL_S = (
        300  # wait_for_processes() fallback, overridden in practice by every real caller.
    )
    WAIT_POLL_INTERVAL_S = (
        5  # wait_for_processes() fallback, overridden in practice by every real caller.
    )

    backend = DynamicConfigParam("agsandbox_backend", default="auto")
    docker_semaphore_limit = GlobalConfigParam("agsandbox_backend", default=16)

    # Timeouts (seconds) for the various classes of docker/podman subprocess call.
    inspect_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=120
    )  # Fast metadata queries: docker inspect, docker ps, nvidia-smi, docker update.
    exec_quick_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=120
    )  # Quick in-container exec calls: kill <pids>, test -d, and similar.
    docker_run_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=120
    )  # docker run: GPU init via the NVIDIA container runtime can take 60+ s under load.
    docker_rm_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=120
    )  # docker rm -f: fast teardown; should complete in a few seconds.
    file_io_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=120
    )  # In-container file I/O via docker exec (base64 read/write, mkdir).
    image_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=120
    )  # docker images list / docker rmi.
    commit_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=120
    )  # docker commit: snapshots a full overlay layer; large workspaces need extra time.
    keyring_wait_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=120
    )  # Maximum time to wait for a keyring slot before abandoning a docker/podman run retry.
    unkillable_child_grace_s = GlobalConfigParam(
        "agsandbox_backend", default=10
    )  # Extra time _run() waits for subprocess.run's own kill()+wait() to finish
    # reaping a timed-out child before giving up and returning control to the
    # caller anyway (see _run()'s docstring for why timeout= alone isn't reliable).

    # _keyring_container_limit(): concurrent-container cap derived from the kernel
    # session-keyring quota, which docker and podman (rootless, via runc) both
    # consume identically (see that function's docstring for the full rationale).
    container_limit_floor = GlobalConfigParam(
        "agsandbox_backend", default=4
    )  # Never cap below this many concurrent containers.
    container_limit_buffer = GlobalConfigParam(
        "agsandbox_backend", default=5
    )  # Safety margin subtracted from the raw kernel/fallback quota.
    container_limit_fallback = GlobalConfigParam(
        "agsandbox_backend", default=200
    )  # Assumed kernel quota when /proc/sys/kernel/keys/maxkeys isn't readable.

    # _run_with_conflict_retry(): retrying `docker run` on name-conflict/keyring errors.
    conflict_retry_max_attempts = GlobalConfigParam("agsandbox_backend", default=8)
    keyring_poll_interval_s = GlobalConfigParam(
        "agsandbox_backend", default=5
    )  # Poll interval while waiting for a free keyring slot.
    container_removal_wait_s = GlobalConfigParam(
        "agsandbox_backend", default=10
    )  # Max time to wait for a conflicting container to finish being removed.
    container_removal_poll_interval_s = GlobalConfigParam("agsandbox_backend", default=0.5)
    conflict_retry_backoff_base_s = GlobalConfigParam(
        "agsandbox_backend", default=0.5
    )  # Multiplied by attempt number for linear backoff between retries.

    # stop(): hibernate (docker/podman stop, container kept).
    docker_stop_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=120
    )  # docker stop: should complete fast, see docker_stop_grace_s below.
    docker_stop_grace_s = GlobalConfigParam(
        "agsandbox_backend", default=0
    )  # `docker stop -t` grace period before SIGKILL. Default 0 (immediate
    # SIGKILL) because every sandbox container's entrypoint is `tail -f
    # /dev/null`, which never handles SIGTERM -- any nonzero grace period
    # is pure wasted wall-clock waiting for a timeout that always fires.
    docker_start_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=120
    )  # docker start: resuming a hibernating container.

    # commit(): checkpoint-in-place, container is not removed.
    stop_inspect_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=30
    )  # docker inspect (pre-commit image-id lookup).
    commit_retry_attempts = GlobalConfigParam("agsandbox_backend", default=3)
    commit_retry_backoff_s = GlobalConfigParam("agsandbox_backend", default=1)
    stop_ps_check_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=10
    )  # docker ps (checking whether the old image is still in use).

    # rm_container(): force-remove teardown, discarding all container state.
    rm_retry_attempts = GlobalConfigParam("agsandbox_backend", default=3)
    rm_retry_backoff_s = GlobalConfigParam("agsandbox_backend", default=1)

    # Every commit() is a diff layer on top of whatever the container was
    # restarted from, and restart always resumes FROM the last checkpoint --
    # so a long-running sandbox's layer chain grows by one every checkpoint
    # cycle with nothing to bound it, until it crosses the container
    # runtime's hard layer-depth cap ("max depth exceeded" on docker/moby).
    # checkpoint_squash_max_depth periodically flattens the chain back to a
    # single layer (export/import instead of commit) well before that cap,
    # triggered by the chain's actual current depth
    # (`len(_image_diff_ids(tag))`, a cheap `docker/podman inspect` --
    # not proportional to image size) rather than a fixed commit count: a
    # count can't account for how many layers the base image itself
    # already consumes (a real base image was observed at 80 layers on its
    # own), so a count-based interval could let the real depth cross the
    # runtime's actual cap before the interval ever fired -- exactly what
    # caused a real "max depth exceeded" failure on an ordinary plain commit.
    checkpoint_squash_max_depth = GlobalConfigParam(
        "agsandbox_backend", default=100
    )  # Squash once the chain's actual depth reaches this; keep well under
    # the real cap (~125 observed empirically on docker/moby) to leave
    # margin for the squash itself and any variance across storage
    # drivers/runtime versions.
    squash_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=600
    )  # export/import serializes the FULL filesystem, not a diff -- needs more headroom than commit_timeout_s.


class agSandboxBackendConfig(_AgConfigViewBase):
    """View over an agConfig for pre-selecting the sandbox backend::

    cfg = agSandboxBackendConfig(backend="docker").agconfig
    ag = agent(agconfig=cfg)
    """

    _OWNER = "agsandbox_backend"


def run_with_unkillable_child_grace(
    call: "Callable[[], subprocess.CompletedProcess[bytes]]",
    *,
    args: list[str],
    timeout: float,
    grace_s: float,
    on_give_up: "Callable[[], None] | None" = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run *call* (a zero-arg thunk wrapping a single ``subprocess.run(...,
    timeout=timeout)`` invocation) in a background thread, and give up
    waiting on it after ``timeout + grace_s`` total, regardless of whether it
    has actually finished.

    ``subprocess.run``'s own ``timeout=`` is not a reliable ceiling: on
    ``TimeoutExpired`` it SIGKILLs the child then calls an UNBOUNDED
    ``process.wait()`` to reap it, and SIGKILL cannot preempt a process stuck
    in uninterruptible kernel sleep (D-state -- e.g. blocked on a wedged
    daemon/mount/overlayfs syscall). Such a child can leave ``wait()``
    hanging for hours, blocking every caller -- and everything blocked
    transitively on it -- for just as long.

    Giving up does NOT kill the background thread -- Python cannot kill a
    thread -- it keeps trying to reap the real subprocess for as long as it
    takes, discarded once it eventually finishes. This function just stops
    blocking the caller once ``grace_s`` has also elapsed past *call*'s own
    timeout, converting the hang into the same ``subprocess.TimeoutExpired``
    every caller already handles for an ordinary (fast) timeout.

    *on_give_up*, if given, runs synchronously (still in the caller's thread)
    before ``subprocess.TimeoutExpired`` is raised -- e.g. to release a
    concurrency-limiting semaphore slot the caller held for this call, now
    that waiting on it is over.
    """
    future: "Future[subprocess.CompletedProcess[bytes]]" = Future()

    def _task() -> None:
        try:
            future.set_result(call())
        except BaseException as e:  # noqa: BLE001 - relayed verbatim to the waiter below
            future.set_exception(e)

    threading.Thread(target=_task, daemon=True).start()
    try:
        return future.result(timeout=timeout + grace_s)
    except _FutureTimeoutError:
        if on_give_up is not None:
            on_give_up()
        print(
            f"[agsandbox_backend] WARNING: {' '.join(args)} did not exit within "
            f"{timeout + grace_s}s even after SIGKILL (unkillable/D-state child?) -- "
            "giving up waiting; the underlying process is not killed and will keep "
            "being reaped in the background.",
            flush=True,
        )
        raise subprocess.TimeoutExpired(args, timeout) from None


_BGPIDS_MARKER = "__BGPIDS__:"


# ---------------------------------------------------------------------------
# Backend base class + factory
# ---------------------------------------------------------------------------


class agsandbox_backend(AgSandboxBackendFields):
    """One backend instance per agSandbox — owns the actual isolation
    mechanism (a running container, a chroot jail, ...) for that sandbox's
    whole lifetime. Use `agsandbox_backend.for_config()` to get the right
    subclass; don't instantiate a subclass directly.

    Concrete backends implement every method `agSandbox` (the facade in
    agsandbox.py) delegates to: exec/_container_exec, file I/O,
    commit/stop/restore/destroy, update_limits, remove_files, PID tracking,
    fork, and the static image-level helpers.
    """

    # Identifies the image/snapshot format a concrete backend's
    # tag_image/export_image/import_image/delete_image use -- checkpoints
    # produced by one backend kind are meaningless to another (a chroot
    # snapshot directory isn't a docker image tag, and vice versa), so
    # agent.py's save()/load() records this alongside a checkpoint and uses
    # backend_for_image_kind() to route to the matching backend class rather
    # than assuming the container backend unconditionally.
    IMAGE_KIND: "str" = ""

    def _own_host_pids(self) -> "set[int]":
        """Return the host PIDs of every process this sandbox currently has
        running. Overridden per-backend; default is empty (no sandbox
        process context)."""
        return set()

    @staticmethod
    def for_config(
        agconfig: "agConfig | None",
        *,
        agname: str,
        name: str,
        checkpoint_image: "str | None",
        base_image: str,
        mounts: "dict[str, tuple[str, str, str]]",
    ) -> "agsandbox_backend":
        from .chroot import chroot_available
        from .container import _runtime_works

        requested = (
            agconfig.get("agsandbox_backend", "backend", "auto") if agconfig else None
        ) or "auto"

        if requested == "auto":
            runtime = _auto_detect_runtime()
        elif requested in ("docker", "podman"):
            if not (shutil.which(requested) and _runtime_works(requested)):
                raise RuntimeError(
                    f"agsandbox_backend.backend={requested!r} was requested but {requested} "
                    f"is not installed or its daemon is not reachable."
                )
            runtime = requested
        elif requested == "chroot":
            if not chroot_available():
                raise RuntimeError(
                    "agsandbox_backend.backend='chroot' was requested but unprivileged user "
                    "namespaces are not usable on this host (checked "
                    "/proc/sys/kernel/unprivileged_userns_clone and a live "
                    "`unshare --user --map-root-user` smoke test, both bare and "
                    "rootlesskit-wrapped)."
                )
            runtime = "chroot"
        else:
            raise ValueError(
                f"Unknown agsandbox_backend.backend {requested!r} "
                f"(expected 'auto', 'docker', 'podman', or 'chroot')"
            )

        if runtime == "chroot":
            from .chroot import _ChrootBackend

            return _ChrootBackend(
                agname,
                name=name,
                checkpoint_image=checkpoint_image,
                mounts=mounts,
                agconfig=agconfig,
            )

        if runtime == "docker":
            from .docker import _DockerBackend

            backend_cls = _DockerBackend
        else:
            from .podman import _PodmanBackend

            backend_cls = _PodmanBackend

        return backend_cls(
            agname,
            name=name,
            checkpoint_image=checkpoint_image,
            base_image=base_image,
            mounts=mounts,
            agconfig=agconfig,
        )

    # ------------------------------------------------------------------
    # Static image-level helpers, forwarded to the container backend — the
    # only backend with an image concept so far. agSandbox (the facade)
    # calls these without an instance at hand (see agent.py's save()/load()),
    # so they're exposed here rather than requiring a live backend instance.
    # ------------------------------------------------------------------

    @staticmethod
    def tag_image(source: str, dest: str) -> None:
        from .container import _ContainerBackendBase

        _ContainerBackendBase.tag_image(source, dest)

    @staticmethod
    def delete_image(tag: str, *, force: bool = False) -> None:
        from .container import _ContainerBackendBase

        _ContainerBackendBase.delete_image(tag, force=force)

    @staticmethod
    def export_image(tag: str, timeout: int) -> bytes:
        from .container import _ContainerBackendBase

        return _ContainerBackendBase.export_image(tag, timeout)

    @staticmethod
    def import_image(image_bytes: bytes, timeout: int) -> None:
        from .container import _ContainerBackendBase

        _ContainerBackendBase.import_image(image_bytes, timeout)

    # ------------------------------------------------------------------
    # Shared implementation -- exec (GPU env injection + background-PID
    # tracking), file I/O, and PID bookkeeping are all implemented purely in
    # terms of the abstract _container_exec() primitive below, so every
    # concrete backend gets them for free rather than reimplementing the
    # same base64/proc-diffing logic per backend.
    # ------------------------------------------------------------------

    def _read_proc_table(self, script: str, timeout: int) -> "tuple[str, int]":
        """Run a pure /proc-reading *script* (no filesystem access, no user
        command involved) and return its ``(output, rc)``.

        Default: delegate to ``_container_exec()`` -- a container's own exec
        session runs inside its own procfs view, which IS the right process
        table for `_snapshot_pids()`/`get_live_pids()` to read for docker and
        podman. Overridden by `_ChrootBackend`, which has no isolated procfs
        of its own to exec into (see its module docstring) and must instead
        read the real host `/proc` directly, unchrooted.
        """
        return self._container_exec(script, timeout=timeout, shell="sh")

    def _snapshot_pids(self) -> set[int]:
        """Return the set of all live PIDs currently in the container, excluding
        the snapshot shell itself so that monitoring shells are not mistaken
        for user-spawned processes."""
        out, _ = self._read_proc_table(
            "__SELF=$$\n"
            "for __d in /proc/[0-9]*; do\n"
            '  [ -f "$__d/status" ] || continue\n'
            "  __p=${__d##*/}\n"
            '  [ "$__p" != "$__SELF" ] && echo "$__p"\n'
            "done",
            timeout=self.inspect_timeout_s,
        )
        pids: set[int] = set()
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.add(int(line))
        return pids

    def _exec_with_pid_tracking(
        self, env_export: str, cmd: str, workdir: str, timeout: int
    ) -> "tuple[str, int]":
        """Run *cmd* (with *env_export* prefixed) inside the container,
        wrapped in a before/after /proc diff that discovers any background or
        detached process *cmd* left running. Returns raw ``(output, rc)`` --
        the ``__BGPIDS__`` marker is parsed by the caller (`exec()`).

        Default: build the whole before/cmd/after script as ONE string and
        run it through `_container_exec()` -- for docker/podman that single
        exec session's own procfs view IS both the command's filesystem
        context and the right place to read /proc from, so before-snapshot,
        command, and after-diff can all happen in the same place.
        Overridden by `_ChrootBackend`, which has no procfs of its own (see
        its module docstring) and must run the before/after /proc reads
        against the real host /proc, outside the jail, while still running
        *cmd* itself chrooted for filesystem containment.
        """
        wrapped = (
            f"exec 2>&1\n"  # merge stderr into stdout so the BGPIDS marker is never split
            # Snapshot every live PID in the container before the command runs.
            # Filtering on /proc/<N>/status avoids races with short-lived kernel threads.
            f"__AGENCY_BEFORE=$(for __d in /proc/[0-9]*; do"
            f" [ -f \"$__d/status\" ] && echo \"${{__d##*/}}\"; done | tr '\\n' ' ')\n"
            f"__AGENCY_SHELL=$$\n"
            f"{env_export}{cmd}\n"
            f"__AGENCY_RC=$?\n"
            # Diff /proc after the command: any PID not in the before-snapshot
            # and not the shell itself was spawned by the command.
            f"__AGENCY_BGPIDS=''\n"
            f"for __d in /proc/[0-9]*; do\n"
            f'  [ -f "$__d/status" ] || continue\n'
            f"  __p=${{__d##*/}}\n"
            f'  case " $__AGENCY_BEFORE $__AGENCY_SHELL " in\n'
            f'    *" $__p "*) ;;\n'
            f'    *) __AGENCY_BGPIDS="$__AGENCY_BGPIDS $__p" ;;\n'
            f"  esac\n"
            f"done\n"
            f"printf '\\n{_BGPIDS_MARKER}%s' \"$__AGENCY_BGPIDS\"\n"
            f"exit $__AGENCY_RC"
        )
        return self._container_exec(wrapped, workdir=workdir, timeout=timeout)

    def exec(
        self,
        cmd: str,
        workdir: str = "/workspace",
        timeout: int = AgSandboxBackendFields.DEFAULT_EXEC_TIMEOUT_S,
    ) -> tuple[str, int]:
        """Run a user command inside the container."""
        # Lazily acquire a physical GPU now that we have a bash call to run.
        # Blocks (polling every 0.25 s) until any GPU in the pool is free.
        if self._gpu_virtual and self._gpu_id is None and self._gpu_acquire_fn is not None:
            self._gpu_id = self._gpu_acquire_fn()

        # Restrict GPU access to the acquired GPU ID. Set both CUDA_VISIBLE_DEVICES
        # (NVIDIA/CUDA) and HIP_VISIBLE_DEVICES (AMD/ROCm) so only the leased
        # device is accessible regardless of which runtime is present.
        # "NoDevFiles" hides all GPUs when no GPU has been acquired.
        # An empty string would leave CUDA_VISIBLE_DEVICES unset, making all GPUs visible.
        import os

        gpu_id = str(self._gpu_id) if self._gpu_id is not None else "NoDevFiles"
        hf_token = os.environ.get("HF_TOKEN", "")
        hf_export = f"export HF_TOKEN={hf_token}\n" if hf_token else ""
        # readonly (not just export) so a command that itself starts with
        # "CUDA_VISIBLE_DEVICES=<n> ..." cannot hijack the leased GPU. Bash rejects that inline reassignment ("readonly
        # variable", visible on stderr) and still runs the command with the
        # correct exported value, rather than quietly routing it onto whatever
        # GPU the command hardcoded.
        env_export = (
            f"export CUDA_VISIBLE_DEVICES={gpu_id}\n"
            f"export HIP_VISIBLE_DEVICES={gpu_id}\n"
            f"readonly CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES\n"
            f"{hf_export}"
        )

        output, rc = self._exec_with_pid_tracking(env_export, cmd, workdir, timeout)

        if _BGPIDS_MARKER in output:
            parts = output.rsplit(_BGPIDS_MARKER, 1)
            clean_output = parts[0].rstrip("\n")
            pids_str = parts[1].strip()
            if pids_str:
                now = time.monotonic()
                for pid_s in pids_str.split():
                    try:
                        self._watched_pids[int(pid_s)] = now
                    except ValueError:
                        # Ignore malformed PID tokens in __BGPIDS__ output.
                        # PID capture is best-effort and should not fail exec().
                        pass
        else:
            clean_output = output

        # Re-verify liveness when _watched_pids is non-empty — get_live_pids()
        # prunes dead entries. GPU release itself is deferred until stop(),
        # which runs after teardown (kill processes, then remove the
        # container / rmtree the chroot jail) has already completed
        # synchronously -- not tied to an empty watched set here while the
        # idle entrypoint still runs. There is no agent-facing release tool;
        # the real GPU semaphore's lifetime is tied to the sandbox's own
        # lifetime.
        if self._watched_pids and self._gpu_virtual and self._gpu_id is not None:
            self.get_live_pids()

        return clean_output, rc

    def read_file(self, path: str) -> str:
        """Read a text file from the container.

        Raises:
            IsADirectoryError: if the path exists but is a directory.
            UnicodeDecodeError: if the file exists but is not valid UTF-8.
            FileNotFoundError: if the path does not exist.
        """
        import base64

        b64, rc = self._container_exec(
            f"base64 {shlex.quote(path)}", timeout=self.file_io_timeout_s, shell="sh"
        )
        if rc != 0:
            _, dir_rc = self._container_exec(
                f"test -d {shlex.quote(path)}", timeout=self.exec_quick_timeout_s, shell="sh"
            )
            if dir_rc == 0:
                raise IsADirectoryError(f"Path is a directory, not a file: {path}")
            raise FileNotFoundError(f"Not found in container: {path}")
        raw = base64.b64decode(b64.strip())
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UnicodeDecodeError(
                exc.encoding,
                exc.object,
                exc.start,
                exc.end,
                f"File {path} contains binary data and is not UTF-8 text",
            ) from None

    def read_file_bytes(self, path: str) -> bytes:
        """Read raw bytes from a file in the container.

        Like read_file but returns bytes without any UTF-8 decode attempt.
        Use for binary files (images, audio, compiled artifacts, etc.).

        Raises:
            IsADirectoryError: if the path exists but is a directory.
            FileNotFoundError: if the path does not exist.
        """
        import base64

        b64, rc = self._container_exec(
            f"base64 {shlex.quote(path)}", timeout=self.file_io_timeout_s, shell="sh"
        )
        if rc != 0:
            _, dir_rc = self._container_exec(
                f"test -d {shlex.quote(path)}", timeout=self.exec_quick_timeout_s, shell="sh"
            )
            if dir_rc == 0:
                raise IsADirectoryError(f"Path is a directory, not a file: {path}")
            raise FileNotFoundError(f"Not found in container: {path}")
        return base64.b64decode(b64.strip())

    def write_file_bytes(self, path: str, data: bytes) -> None:
        """Write raw bytes to a file in the container.

        Use for binary files. The data is base64-encoded on the host and
        decoded inside the container, avoiding any shell-quoting issues with
        arbitrary byte sequences.
        """
        import base64

        b64 = base64.b64encode(data).decode("ascii")
        quoted = shlex.quote(path)
        sh_cmd = (
            f"mkdir -p $(dirname {quoted}) && printf '%s' {shlex.quote(b64)} | base64 -d > {quoted}"
        )
        _, rc = self._container_exec(sh_cmd, timeout=self.file_io_timeout_s, shell="sh")
        if rc != 0:
            raise OSError(f"Failed to write binary file {path} in container")

    def write_file(self, path: str, content: str) -> None:
        quoted = shlex.quote(path)
        sh_cmd = f"mkdir -p $(dirname {quoted}) && cat > {quoted}"
        _, rc = self._container_exec(
            sh_cmd, stdin=content.encode("utf-8"), timeout=self.file_io_timeout_s, shell="sh"
        )
        if rc != 0:
            raise OSError(f"Failed to write {path} in container")

    def release_daemon(self, pid: int) -> None:
        """Move *pid* out of the monitored set into the daemon set.

        The process and all its future descendants will continue running in the
        container but will never block the outer monitoring loop.
        """
        self._daemon_pids.add(pid)
        self._watched_pids.pop(pid, None)

    # When True (Docker/Podman), get_live_pids() adds newly discovered
    # non-baseline live PIDs into _watched_pids so children of backgrounded
    # work stay tracked. Chroot sets this False: its /proc scan is the whole
    # host, so adopting strangers would latch unrelated long-lived host
    # processes into the wait/GPU-clear path.
    _adopt_unwatched_live_pids: bool = True

    def _has_pending_background_work(self) -> bool:
        """Cheap fast-path check backing `agSandbox.wait_for_processes()`:
        is there ANYTHING this sandbox might still need to be waited on for?

        Default: just check `_watched_pids`. For docker/podman, an isolated
        container's own procfs means `_watched_pids` (with
        `_adopt_unwatched_live_pids` adopting anything new) accurately
        reflects everything that could possibly be running there, so an
        empty `_watched_pids` really does mean nothing to wait for.
        Overridden by `_ChrootBackend`, whose host-wide /proc means
        `_watched_pids` can under-count a process spawned after the
        `exec()` call that started its parent already returned (see its
        module docstring) -- that override also checks a fresh, live,
        non-mutating scan rather than trusting `_watched_pids` alone.
        """
        return bool(self._watched_pids)

    def get_live_pids(self) -> set[int]:
        if not self._watched_pids:
            return set()

        # Read pid, ppid, state, and comm name for every entry in /proc,
        # excluding the monitoring shell itself. Pure shell builtins only
        # (read/parameter-expansion/case) -- NOT awk per entry: forking 3
        # subprocesses per /proc entry across thousands of entries on a busy
        # host was confirmed empirically to take 10+ seconds (for chroot,
        # whose scan spans the whole host -- see chroot.py's module
        # docstring), long enough for a short-lived tracked process to exit
        # before the scan even reaches it. Reading /proc/<pid>/status
        # line-by-line via the `read` builtin is dramatically faster since
        # nothing forks per entry.
        script = (
            "__SELF=$$\n"
            "for __d in /proc/[0-9]*; do\n"
            '  [ -f "$__d/status" ] || continue\n'
            "  __p=${__d##*/}\n"
            '  [ "$__p" = "$__SELF" ] && continue\n'
            "  __ppid=''\n"
            "  __st=''\n"
            "  __nm=''\n"
            "  while IFS= read -r __line; do\n"
            '    case "$__line" in\n'
            "      PPid:*) set -- ${__line#PPid:}; __ppid=$1 ;;\n"
            "      State:*) set -- ${__line#State:}; __st=$1 ;;\n"
            "      Name:*) set -- ${__line#Name:}; __nm=$1 ;;\n"
            "    esac\n"
            '  done < "$__d/status"\n'
            '  echo "$__p $__ppid $__st $__nm"\n'
            "done"
        )
        output, _ = self._read_proc_table(script, timeout=self.inspect_timeout_s)

        proc_info: dict[int, tuple[int, str, str]] = {}  # pid → (ppid, state, name)
        for line in output.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
                ppid = int(parts[1])
                state = parts[2] if len(parts) > 2 else "?"
                name = parts[3] if len(parts) > 3 else ""
            except ValueError:
                continue
            proc_info[pid] = (ppid, state, name)

        # Mark NVIDIA container-runtime helper processes as system PIDs and
        # propagate that status to their children.  The NVIDIA toolkit entrypoint
        # (comm = "nvidia_entrypoi") periodically spawns short-lived GPU check
        # processes (cudaCheck, deviceQuery, …) that are not user processes and
        # must not be counted as live background work.
        #
        # Baseline PIDs are always excluded — PID 1 in GPU containers IS named
        # "nvidia_entrypoi" (it is the init process), so we must not add it to
        # system_pids or every process reparented to it after its parent exits
        # would be incorrectly filtered.
        system_pids: set[int] = {
            pid
            for pid, (_, _, name) in proc_info.items()
            if name == "nvidia_entrypoi" and pid not in self._baseline_pids
        }
        changed = True
        while changed:
            changed = False
            for pid, (ppid, _, _) in proc_info.items():
                if (
                    pid not in system_pids
                    and pid not in self._baseline_pids
                    and ppid in system_pids
                ):
                    system_pids.add(pid)
                    changed = True

        # Propagate daemon status down the tree: if a process's parent is a
        # daemon, the child inherits that status and is also excluded from
        # monitoring.  Repeat until no new daemons are discovered.
        changed = True
        while changed:
            changed = False
            for pid, (ppid, _, _) in proc_info.items():
                if pid not in self._daemon_pids and ppid in self._daemon_pids:
                    self._daemon_pids.add(pid)
                    self._watched_pids.pop(pid, None)
                    changed = True

        # A PID counts as live background work if it exists in /proc, is not
        # baseline / NVIDIA helper / daemon / zombie, AND is either already
        # watched or (when _adopt_unwatched_live_pids) newly discovered.
        # Chroot disables adoption so a non-baseline host process started by
        # something else cannot enter _watched_pids via this scan.
        alive: set[int] = set()
        now = time.monotonic()
        for pid, (_, state, _) in proc_info.items():
            if (
                pid in self._baseline_pids
                or pid in system_pids
                or pid in self._daemon_pids
                or state == "Z"
            ):
                continue
            if pid in self._watched_pids:
                alive.add(pid)
            elif self._adopt_unwatched_live_pids:
                self._watched_pids[pid] = now
                alive.add(pid)

        # Prune _watched_pids entries that are no longer alive.
        for pid in set(self._watched_pids):
            if pid not in alive:
                del self._watched_pids[pid]

        return alive

    def pid_status_summary(self) -> str:
        live = self.get_live_pids()
        if not live:
            # get_live_pids() can under-count for some backends (see
            # _ChrootBackend._has_pending_background_work()'s docstring) --
            # avoid flatly contradicting a caller (e.g. wait_for_processes())
            # that already determined via the authoritative check that
            # something is still running, just not individually listable here.
            if self._has_pending_background_work():
                return "background activity detected but not individually trackable right now"
            return "no background processes running"
        now = time.monotonic()
        parts = []
        for pid in sorted(live):
            elapsed = int(now - self._watched_pids.get(pid, now))
            mins, secs = divmod(elapsed, 60)
            parts.append(f"PID {pid} (running {mins}m {secs}s)")
        return ", ".join(parts)

    def release_resources(self, pool: "agResourcePool | None" = None) -> None:
        self._gpu_virtual = False
        if self._gpu_id is not None:
            if pool is not None:
                pool.release_gpu(self._gpu_id)
            elif self._gpu_release_fn is not None:
                self._gpu_release_fn(self._gpu_id)
            self._gpu_id = None
        if pool is not None and (self._cpu_acquired or self._memory_acquired_mb):
            pool.notify_cpu_released(self._cpu_acquired, self._memory_acquired_mb)
            self._cpu_acquired = 0.0
            self._memory_acquired_mb = 0
        if pool is not None:
            try:
                self.update_limits(cpus=pool.idle_cpus, memory=pool.idle_memory)
            except Exception as _e:
                print(f"[agsandbox_backend] WARNING: update_limits failed for {self._name}: {_e}")

    def remove_files(self, paths: list[str]) -> None:
        """Delete sandbox files previously written by offload/agfile helpers."""
        import shlex as _shlex

        for path in paths:
            try:
                self._container_exec(f"rm -f {_shlex.quote(path)}", shell="sh")
            except Exception as _e:
                print(f"[agsandbox_backend] WARNING: failed to remove offloaded file {path}: {_e}")


def _auto_detect_runtime() -> str:
    """Auto-detect which backend to use: podman, then docker, then chroot."""
    from .chroot import chroot_available
    from .container import get_container_runtime

    try:
        return get_container_runtime()
    except RuntimeError:
        if chroot_available():
            return "chroot"
        raise RuntimeError(
            "No usable sandbox backend found: neither docker nor podman is "
            "installed/reachable, and unprivileged user namespaces (required "
            "for the chroot backend) are not usable on this host."
        ) from None


def backend_for_image_kind(kind: str) -> type:
    """Return the backend class whose tag_image/export_image/import_image/
    delete_image understand a checkpoint of the given IMAGE_KIND.

    Used by agent.py's load() to route a saved checkpoint's image bytes to
    the same kind of backend that produced them in save() -- a chroot
    snapshot directory and a docker/podman image tag are different formats
    entirely, so this can't be assumed to always be the container backend.
    Docker and podman checkpoints are both IMAGE_KIND="container" (tag_image/
    export_image/import_image/delete_image are identical either way, both
    just auto-detecting the live runtime via get_container_runtime()), so
    both route to the same shared _ContainerBackendBase rather than needing
    to know which of the two originally produced the checkpoint.
    """
    if kind == "container":
        from .container import _ContainerBackendBase

        return _ContainerBackendBase
    if kind == "chroot":
        from .chroot import _ChrootBackend

        return _ChrootBackend
    raise ValueError(
        f"Unknown sandbox checkpoint image kind {kind!r} (expected one of ['chroot', 'container'])"
    )
