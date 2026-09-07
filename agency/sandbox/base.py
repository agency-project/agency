"""Sandbox backend base class, shared config, and backend selection.

An `agSandbox` instance builds exactly one `agsandbox_backend` from its config
(via `agsandbox_backend.for_config()`) and delegates every sandboxing
operation (exec, file I/O, lifecycle, checkpointing) to it. Backend selection
logic (podman vs. docker vs. chroot, see `agconfig.sandbox.backend`)
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
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as _FutureTimeoutError
from typing import TYPE_CHECKING, Callable, ClassVar

from . import pid_diagnostics

from ..configs.agconfig import agconfig as agconfig_cls

if TYPE_CHECKING:
    from ..orchestrator.agresources import agResourcePool


class AgSandboxBackendFields:
    """Plain constants unrelated to agconfig -- fallback defaults for
    keyword args at a couple of call sites, not tunable per-agent."""

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

    from ..observability.profiler import agprof

    agprof.spawn_traced(_task).start()
    try:
        return future.result(timeout=timeout + grace_s)
    except _FutureTimeoutError:
        if on_give_up is not None:
            on_give_up()
        # DATACOLLECTOR: append -- no agname in scope here (only argv); genuinely exceptional
        # (unkillable/D-state child), worth surfacing prominently once folded in.
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

    # Fields with no viable fallback for a given agconfig.sandbox.backend --
    # checked eagerly by _validate_config() on every construction/
    # change_config() call. Empty today: docker/podman/chroot's own timeout
    # and retry fields all ship sensible defaults, so nothing about them is
    # strictly required from agconfig alone (unlike agllm's per-provider
    # requirements). Kept as a real, populated mechanism -- not a stub --
    # for the day a backend-specific field with no safe default is added.
    _REQUIRED_FIELDS_BY_BACKEND: "ClassVar[dict[str, tuple[str, ...]]]" = {}

    def _validate_config(self, agconfig: "agconfig_cls") -> None:
        backend = agconfig.sandbox.backend
        required = self._REQUIRED_FIELDS_BY_BACKEND.get(backend, ())
        missing = [name for name in required if not getattr(agconfig.sandbox, name)]
        if missing:
            raise ValueError(
                f"agconfig.sandbox with backend={backend!r} is missing required "
                f"field(s): {', '.join(missing)}"
            )

    def _own_host_pids(self) -> "set[int]":
        """Return the host PIDs of every process this sandbox currently has
        running. Overridden per-backend; default is empty (no sandbox
        process context)."""
        return set()

    @staticmethod
    def for_config(
        agconfig: "agconfig_cls | None",
        *,
        agname: str,
        name: str,
        checkpoint_image: "str | None",
        base_image: str,
        mounts: "dict[str, tuple[str, str, str]]",
    ) -> "agsandbox_backend":
        from .chroot import chroot_available
        from .container import _runtime_works

        requested = (agconfig.sandbox.backend if agconfig else None) or "auto"

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
        table for `get_live_pids()` to read for docker and podman. Overridden
        by `_ChrootBackend`, which has no isolated procfs
        of its own to exec into (see its module docstring) and must instead
        read the real host `/proc` directly, unchrooted.
        """
        return self._container_exec(script, timeout=timeout, shell="sh")

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
        identity_capture = (
            pid_diagnostics.IDENTITY_SHELL
            if getattr(self._agconfig.sandbox, "hibernation_diagnostics", False)
            else ""
        )
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
            f'    *) __AGENCY_BGPIDS="$__AGENCY_BGPIDS $__p"\n'
            f"{identity_capture} ;;\n"
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
        # Lazily acquire the requested physical GPU(s) now that we have a
        # bash call to run. Blocks until enough GPUs in the pool are free.
        if self._gpu_count_requested > 0 and not self._gpu_ids and self._gpu_acquire_fn is not None:
            self._gpu_ids = self._gpu_acquire_fn(self._gpu_count_requested)

        # Restrict GPU access to the acquired GPU ID(s). Set both CUDA_VISIBLE_DEVICES
        # (NVIDIA/CUDA) and HIP_VISIBLE_DEVICES (AMD/ROCm) so only the leased
        # device(s) are accessible regardless of which runtime is present.
        # "NoDevFiles" hides all GPUs when none have been acquired.
        # An empty string would leave CUDA_VISIBLE_DEVICES unset, making all GPUs visible.
        import os

        gpu_id = ",".join(str(g) for g in self._gpu_ids) if self._gpu_ids else "NoDevFiles"
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
        identities = {}
        if getattr(self._agconfig.sandbox, "hibernation_diagnostics", False):
            output, identities = pid_diagnostics.extract_identities(output)

        if _BGPIDS_MARKER in output:
            parts = output.rsplit(_BGPIDS_MARKER, 1)
            clean_output = parts[0].rstrip("\n")
            pids_str = parts[1].strip()
            if pids_str:
                now = time.monotonic()
                for pid_s in pids_str.split():
                    try:
                        pid = int(pid_s)
                        self._watched_pids[pid] = now
                        pid_diagnostics.register(self, pid, "exec_proc_diff", identities.get(pid))
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
        if self._watched_pids and self._gpu_count_requested > 0 and self._gpu_ids:
            self.get_live_pids()

        return clean_output, rc

    def exec_detached(self, cmd: str, workdir: str = "/workspace") -> None:
        """Launch a long-lived process inside the container and return as
        soon as it's registered, without waiting for it to finish or
        tracking its output/exit code -- for a persistent in-container
        process the caller will reach afterward over its own bridge (e.g.
        agharness_backends/native.py's react-loop entrypoint, or a
        container-relocated agproxy_llm), not via this call's return value.
        No GPU/PID-tracking wiring here, unlike `exec()` -- a persistent
        process manages its own environment for the lifetime of the
        container, it isn't a single bounded command. Only implemented by
        container-backed backends (docker/podman) so far -- see
        `_container_exec_detached` in sandbox/container.py."""
        self._container_exec_detached(cmd, workdir=workdir)

    def read_file(self, path: str) -> str:
        """Read a text file from the container.

        Raises:
            IsADirectoryError: if the path exists but is a directory.
            UnicodeDecodeError: if the file exists but is not valid UTF-8.
            FileNotFoundError: if the path does not exist.
        """
        import base64

        b64, rc = self._container_exec(
            f"base64 {shlex.quote(path)}",
            timeout=self._agconfig.sandbox.file_io_timeout_s,
            shell="sh",
        )
        if rc != 0:
            _, dir_rc = self._container_exec(
                f"test -d {shlex.quote(path)}",
                timeout=self._agconfig.sandbox.exec_quick_timeout_s,
                shell="sh",
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
            f"base64 {shlex.quote(path)}",
            timeout=self._agconfig.sandbox.file_io_timeout_s,
            shell="sh",
        )
        if rc != 0:
            _, dir_rc = self._container_exec(
                f"test -d {shlex.quote(path)}",
                timeout=self._agconfig.sandbox.exec_quick_timeout_s,
                shell="sh",
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
        _, rc = self._container_exec(
            sh_cmd, timeout=self._agconfig.sandbox.file_io_timeout_s, shell="sh"
        )
        if rc != 0:
            raise OSError(f"Failed to write binary file {path} in container")

    def write_file(self, path: str, content: str) -> None:
        quoted = shlex.quote(path)
        sh_cmd = f"mkdir -p $(dirname {quoted}) && cat > {quoted}"
        _, rc = self._container_exec(
            sh_cmd,
            stdin=content.encode("utf-8"),
            timeout=self._agconfig.sandbox.file_io_timeout_s,
            shell="sh",
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

    def _register_harness_pid(self, pid: int, start_ticks: int) -> None:
        # Exempt only the manager itself. Unlike release_daemon(), its user
        # workload descendants must remain eligible for monitoring.
        if not hasattr(self, "_infrastructure_pids"):
            self._infrastructure_pids = {}
        self._infrastructure_pids[pid] = start_ticks
        self._watched_pids.pop(pid, None)

    def ingest_ptrace_pids(
        self, spawned: "set[int] | None" = None, exited: "set[int] | None" = None
    ) -> None:
        """Alternate population path for `_watched_pids`, fed by
        `agproxy_ptrace`'s fork/exit event stream (see
        `agProxyPtraceHandle.on_spawn`/`.on_exit` in agproxy_ptrace.py) instead
        of `exec()`'s `/proc`-diff + `__BGPIDS__` marker. `get_live_pids()`/
        `pid_status_summary()`/`wait_for_processes()`/`release_daemon()`'s
        external contracts are unchanged -- only the internal population
        mechanism differs for harness-driven agents versus native ones.

        Pids passed here are tracked in `_ptrace_managed_pids` in addition to
        `_watched_pids`, so `get_live_pids()` trusts *this* method's `exited`
        calls as the sole liveness signal for them rather than pruning them
        the moment its own `/proc` scan doesn't happen to show them --
        ptrace's fork/exit events are exact regardless of whether the traced
        pids are visible in whatever PID namespace `_container_exec()`
        queries, which they are NOT in general (a docker/podman container has
        its own separate PID namespace from a ptrace supervisor forked on the
        host; only a supervisor that itself runs inside the container's
        namespace, e.g. via `docker exec`, or the chroot backend, which
        shares the host namespace, would see them there too). Bridging that
        gap for the docker/podman backends -- running the supervisor inside
        the container plus an IPC channel back to the caller's `agpolicy` --
        is an open item.
        """
        now = time.monotonic()
        baseline_pids = self._baseline_pids or ()
        for pid in spawned or ():
            if pid in baseline_pids or pid in self._daemon_pids:
                continue
            self._watched_pids.setdefault(pid, now)
            pid_diagnostics.register(self, pid, "ptrace_spawn")
            self._ptrace_managed_pids.add(pid)
        for pid in exited or ():
            self._watched_pids.pop(pid, None)
            self._ptrace_managed_pids.discard(pid)

    # When True (Docker/Podman), get_live_pids() adds newly discovered
    # non-baseline live PIDs into _watched_pids so children of backgrounded
    # work stay tracked. Chroot sets this False: its /proc scan is the whole
    # host, so adopting strangers would latch unrelated long-lived host
    # processes into the wait/GPU-clear path.
    _adopt_unwatched_live_pids: bool = True

    def _has_pending_background_work(self) -> bool:
        """Refresh tracked work before deciding whether hibernation is safe.

        CPU-only execs can retain exited runtime helpers in the watched set.
        Use the existing liveness/descendant rules rather than treating a stale
        dictionary entry as work. Chroot retains its own PGID-based override.
        """
        return bool(self.get_live_pids()) if self._watched_pids else False

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
        identity_capture = (
            pid_diagnostics.IDENTITY_SHELL
            if (
                getattr(self._agconfig.sandbox, "hibernation_diagnostics", False)
                or getattr(self, "_infrastructure_pids", {})
            )
            else ""
        )
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
            f"{identity_capture}"
            "done"
        )
        output, rc = self._read_proc_table(script, timeout=self._agconfig.sandbox.inspect_timeout_s)
        if rc != 0:
            # An unreadable process table is not evidence that work finished.
            return set(self._watched_pids)
        identities = {}
        if identity_capture:
            output, identities = pid_diagnostics.extract_identities(output)

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
        # would be incorrectly filtered. `_baseline_pids` is None when this
        # backend's container/jail has never been through _ensure_started()
        # yet (e.g. a harness-driven agent whose only PID activity so far
        # came through ingest_ptrace_pids(), never a native exec()) --
        # treated as "nothing captured", not an error.
        baseline_pids = self._baseline_pids or ()
        system_pids: set[int] = {
            pid
            for pid, (_, _, name) in proc_info.items()
            if name == "nvidia_entrypoi" and pid not in baseline_pids
        }
        changed = True
        while changed:
            changed = False
            for pid, (ppid, _, _) in proc_info.items():
                if pid not in system_pids and pid not in baseline_pids and ppid in system_pids:
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
                pid in baseline_pids
                or (
                    pid in getattr(self, "_infrastructure_pids", {})
                    and identities.get(pid, {}).get("start_ticks") == self._infrastructure_pids[pid]
                )
                or pid in system_pids
                or pid in self._daemon_pids
                or state == "Z"
            ):
                continue
            if pid in self._watched_pids:
                alive.add(pid)
            elif self._adopt_unwatched_live_pids:
                self._watched_pids[pid] = now
                pid_diagnostics.register(self, pid, "proc_scan_adoption", identities.get(pid))
                alive.add(pid)

        # ptrace-managed pids (see ingest_ptrace_pids()) are trusted as alive
        # for as long as they remain in _watched_pids regardless of whether
        # this /proc scan happens to show them -- ptrace's own fork/exit
        # events are the authoritative liveness signal for them (exact, not
        # a poll), and in general they live in a different PID namespace
        # than whatever this method's _container_exec() scan just queried.
        alive |= self._ptrace_managed_pids & set(self._watched_pids)

        # Prune _watched_pids entries that are no longer alive -- except
        # ptrace-managed ones, which only ever leave _watched_pids via an
        # explicit ingest_ptrace_pids(exited=...) call, never because this
        # scan didn't happen to observe them.
        for pid in set(self._watched_pids):
            if pid not in alive and pid not in self._ptrace_managed_pids:
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
        self._gpu_count_requested = 0
        if self._gpu_ids:
            if pool is not None:
                pool.release_gpus(self, self._gpu_ids)
            elif self._gpu_release_fn is not None:
                self._gpu_release_fn(self._gpu_ids)
        if pool is not None:
            try:
                pool.release_cpu_mem(self, cpu=True, memory=True)
            except Exception as _e:
                # DATACOLLECTOR: append, agname=self._name -- best-effort failure, low priority.
                print(f"[agsandbox_backend] WARNING: update_limits failed for {self._name}: {_e}")

    def remove_files(self, paths: list[str]) -> None:
        """Delete sandbox files previously written by offload/agfile helpers."""
        import shlex as _shlex

        for path in paths:
            try:
                self._container_exec(f"rm -f {_shlex.quote(path)}", shell="sh")
            except Exception as _e:
                # DATACOLLECTOR: append, agname=self._name -- best-effort cleanup failure, low priority.
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
