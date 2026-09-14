"""Image-materialized and local ZFS filesystem checkpoint mechanisms."""

from __future__ import annotations

import json
import hashlib
import copy
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..observability.profiler import agprof


class CheckpointCapabilityError(RuntimeError):
    """The requested backend cannot preserve the sandbox's state on this host."""


@dataclass(frozen=True)
class CheckpointHandle:
    backend: str
    runtime: str
    container: str
    reference: str
    stats: dict = field(default_factory=dict, compare=False)


class CheckpointBackend(Protocol):
    def checkpoint(self, sandbox, tag: str | None = None) -> CheckpointHandle | None: ...
    def restore(self, sandbox, checkpoint: CheckpointHandle) -> None: ...
    def delete_checkpoint(self, sandbox, checkpoint: CheckpointHandle) -> None: ...


def _annotate(sandbox, backend):
    agprof.annotate(sandbox_runtime=sandbox._runtime, checkpoint_backend=backend)


class ImageCommitCheckpoint:
    def checkpoint(self, sandbox, tag=None):
        with agprof.span("checkpoint.total"):
            _annotate(sandbox, "image_commit")
            with agprof.span("checkpoint.fs_snapshot"):
                _annotate(sandbox, "image_commit")
                if not sandbox._commit_image(tag):
                    return None
            return CheckpointHandle(
                "image_commit", sandbox._runtime, sandbox._name, sandbox._checkpoint_image
            )

    def restore(self, sandbox, checkpoint):
        with agprof.span("restore.total"):
            _annotate(sandbox, "image_commit")
            with agprof.span("restore.fs"):
                sandbox._restore_image(checkpoint.reference)

    def delete_checkpoint(self, sandbox, checkpoint):
        sandbox._rmi(checkpoint.reference, force=True)


def _command(args, timeout=120):
    try:
        return subprocess.run(args, capture_output=True, check=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        stderr = getattr(exc, "stderr", b"") or b""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        raise CheckpointCapabilityError(
            f"Checkpoint command {args[0]} {args[1] if len(args) > 1 else ''} failed: "
            f"{stderr[-4000:] or type(exc).__name__}"
        ) from exc


class ZfsRuntimeStorage:
    """Shared immutable image seeds with a private ZFS clone per sandbox.

    Native OverlayFS keeps immutable image layers and the writable upper in
    one dataset; cloned seeds share base blocks and container creation adds an
    empty upper layer. FUSE helpers are rejected. Checkpoint and rollback are
    genuine ZFS operations over all layers and runtime state together.
    """

    def __init__(self, sandbox):
        self.parent = sandbox._agconfig.sandbox.checkpoint_zfs_parent
        self.irmap_paths = sandbox._agconfig.sandbox.checkpoint_irmap_paths
        if any(
            not Path(path).is_absolute() or ".." in Path(path).parts for path in self.irmap_paths
        ):
            raise CheckpointCapabilityError("checkpoint_irmap_paths must be absolute sandbox paths")
        if platform.system() != "Linux" or os.geteuid() != 0:
            raise CheckpointCapabilityError(
                "ZFS checkpoints require a local rootful runtime on Linux"
            )
        if sandbox._runtime not in {"docker", "podman"}:
            raise CheckpointCapabilityError("cow_zfs requires Docker or Podman")
        if not self.parent or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:/-]*", self.parent):
            raise CheckpointCapabilityError("Invalid checkpoint_zfs_parent dataset name")
        for binary in (sandbox._runtime, "zfs"):
            if not shutil.which(binary):
                raise CheckpointCapabilityError(
                    f"Missing {binary}; install it on the Linux runtime host"
                )
        if not Path("/dev/zfs").exists():
            raise CheckpointCapabilityError(
                "Missing /dev/zfs; load a compatible OpenZFS kernel module"
            )
        self.runtime = sandbox._runtime
        init_binary = shutil.which("catatonit") or "/usr/libexec/podman/catatonit"
        self.init_path = str(Path(init_binary).resolve())
        self.dataset = (
            f"{self.parent}/agency-{uuid.uuid4().hex}" if self.runtime == "podman" else None
        )
        self.path = None
        self.prepared = False
        self.created = False
        self.criu_ready = False
        self.criu_setup_error = None

    @property
    def prefix(self):
        if self.runtime == "docker":
            return ["docker"]
        if self.path is None:
            raise CheckpointCapabilityError("Private Podman storage has not been provisioned")
        return [
            "podman",
            "--remote=false",
            "--root",
            str(self.path / "graphroot"),
            "--runroot",
            str(self.path / "runroot"),
            "--tmpdir",
            str(self.path / "tmp"),
            "--storage-driver=overlay",
            "--storage-opt=overlay.mount_program=",
            "--runtime=runc",
            "--events-backend=file",
        ]

    def prepare(self, sandbox):
        if self.prepared:
            return
        with agprof.span("checkpoint.setup"):
            _annotate(sandbox, sandbox._agconfig.sandbox.checkpoint_backend)
            mount = (
                _command(["zfs", "get", "-H", "-o", "value", "mountpoint", self.parent])
                .decode()
                .strip()
            )
            if not mount.startswith("/"):
                raise CheckpointCapabilityError(
                    "checkpoint_zfs_parent needs an absolute ZFS mountpoint"
                )
            if self.runtime == "docker":
                info = json.loads(_command(["docker", "info", "--format", "{{json .}}"]))
                root = Path(info["DockerRootDir"]).resolve()
                if info.get("Driver") != "zfs":
                    raise CheckpointCapabilityError(
                        "cow_zfs with Docker requires the daemon's native zfs storage driver"
                    )
                if root != Path(mount).resolve():
                    raise CheckpointCapabilityError(
                        "DockerRootDir must be the configured checkpoint_zfs_parent mountpoint"
                    )
                self.path = root
                self.prepared = True
                return
            import fcntl

            image = sandbox._resolve_image(sandbox._base_image)
            original = json.loads(
                _command(["podman", "--remote=false", "image", "inspect", image])
            )[0]
            # Immutable image identity plus seed layout version. One shared
            # pool/parent, one base dataset per image, cheap clones per sandbox.
            key = hashlib.sha256(("overlay-v1:" + original["Id"]).encode()).hexdigest()[:32]
            base_dataset = f"{self.parent}/agency-base-{key}"
            if len(os.fsencode(f"{mount}/a-{'0' * 20}/runroot")) > 50:
                raise CheckpointCapabilityError(
                    "Podman requires runroot <= 50 bytes; use a short ZFS parent mountpoint (for example /agz/s)"
                )
            base_path = Path(mount) / f"b-{key[:20]}"
            base_snapshot = base_dataset + "@seed-v1"
            lock_path = Path(mount) / f".agency-base-{key}.lock"
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                listed = (
                    _command(
                        ["zfs", "list", "-H", "-o", "name", "-t", "snapshot", "-r", self.parent]
                    )
                    .decode()
                    .splitlines()
                )
                if base_snapshot not in listed:
                    self._seed_image(
                        sandbox, image, original, base_dataset, base_path, base_snapshot
                    )
                self.path = Path(mount) / ("a-" + self.dataset.rsplit("-", 1)[1][:20])
                _command(
                    [
                        "zfs",
                        "clone",
                        "-o",
                        f"mountpoint={self.path}",
                        "-o",
                        "readonly=off",
                        base_snapshot,
                        self.dataset,
                    ]
                )
                self.created = True
            try:
                self.path.chmod(0o700)
                self._reset_seed_runtime_metadata(self.path)
                loaded = json.loads(_command(self.prefix + ["image", "inspect", image]))[0]
                if loaded["Id"] != original["Id"]:
                    raise CheckpointCapabilityError("Cloned image identity changed")
                if sandbox._agconfig.sandbox.checkpoint_fast_resume:
                    try:
                        self._prepare_criu(sandbox)
                        self.criu_ready = True
                    except Exception as exc:
                        self.criu_setup_error = f"{type(exc).__name__}: {exc}"
                        logging.getLogger(__name__).warning(
                            "CRIU fast resume setup unavailable; ZFS checkpoints remain enabled: %s",
                            exc,
                        )
                self.prepared = True
            except BaseException:
                self.destroy()
                raise

    def _seed_image(self, sandbox, image, original, dataset, path, snapshot):
        # This one-time import is provisioning, never a checkpoint operation.
        # No container is launched in a seed. Erase only its unused libpod DB
        # and volatile paths before snapshotting so clone paths initialize their
        # own runtime metadata rather than inheriting absolute seed paths.
        _command(["zfs", "create", "-o", f"mountpoint={path}", dataset])
        self.path = path
        path.chmod(0o700)
        try:
            timeout = sandbox._agconfig.sandbox.checkpoint_setup_timeout_s
            _command(self.prefix + ["info"])
            helper_flag = path / "graphroot" / "overlay" / ".has-mount-program"
            if helper_flag.exists() and helper_flag.read_text().strip() == "true":
                raise CheckpointCapabilityError(
                    "Native OverlayFS on ZFS is required; FUSE overlay is unsupported"
                )
            archive = path / "seed-image.tar"
            _command(["podman", "--remote=false", "save", "-o", str(archive), image], timeout)
            _command(self.prefix + ["load", "-i", str(archive)], timeout)
            archive.unlink()
            loaded = json.loads(_command(self.prefix + ["image", "inspect", image]))[0]
            if loaded["Id"] != original["Id"]:
                raise CheckpointCapabilityError("Seeded image identity changed")
            if _command(self.prefix + ["ps", "-aq"]).strip():
                raise CheckpointCapabilityError("Base image seed unexpectedly contains containers")
            self._reset_seed_runtime_metadata(path)
            _command(["zfs", "snapshot", snapshot])
            _command(["zfs", "set", "readonly=on", dataset])
        except BaseException:
            # No clones can exist before publication under the seed lock.
            _command(["zfs", "destroy", "-r", dataset])
            raise

    @staticmethod
    def _reset_seed_runtime_metadata(path):
        # Seed preparation asserted that there are no containers. Podman can
        # use either SQLite (graphroot/db.sql) or BoltDB (graphroot/libpod).
        for unused in (path / "graphroot" / "libpod", path / "runroot", path / "tmp"):
            if unused.exists():
                shutil.rmtree(unused)
        for name in ("db.sql", "db.sql-shm", "db.sql-wal"):
            (path / "graphroot" / name).unlink(missing_ok=True)

    @property
    def criu_config(self):
        if self.path is None:
            raise CheckpointCapabilityError("Private Podman storage has not been provisioned")
        return self.path / "criu.conf"

    def _prepare_criu(self, sandbox):
        if self.runtime != "podman":
            return
        for binary in ("criu", "runc"):
            if not shutil.which(binary):
                raise CheckpointCapabilityError(
                    f"Missing {binary}; CRIU fast resume will be unavailable"
                )
        if not Path(self.init_path).is_file():
            raise CheckpointCapabilityError("Podman catatonit init binary is required")
        cgroups = (
            _command(
                ["podman", "info", "--format", "{{.Host.CgroupManager}} {{.Host.CgroupsVersion}}"]
            )
            .decode()
            .strip()
        )
        if cgroups != "systemd v2":
            raise CheckpointCapabilityError(
                "CRIU fast resume requires cgroup v2 and Podman systemd cgroup management"
            )
        _command(["criu", "check"])
        for operation in ("checkpoint", "restore"):
            help_text = _command(["podman", "container", operation, "--help"]).decode()
            for flag in ("--keep", "--file-locks", "--tcp-established", "--print-stats"):
                if flag not in help_text:
                    raise CheckpointCapabilityError(f"Podman {operation} lacks {flag}")
        script = self.path / "restore-runtime-mounts.py"
        script.write_bytes(Path(__file__).with_name("_criu_restore_mounts.py").read_bytes())
        launcher = self.path / "restore-runtime-mounts"
        launcher.write_text(
            f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(script))}\n"
        )
        launcher.chmod(0o700)
        self.criu_config.write_text(
            "action-script "
            + json.dumps(str(launcher))
            + "\n"
            + "".join("irmap-scan-path " + json.dumps(path) + "\n" for path in self.irmap_paths)
        )

    def prepare_restore_mounts(self, inspection, reference, sessions):
        if self.runtime != "podman" or self.path is None:
            raise CheckpointCapabilityError("CRIU restore hooks require private Podman storage")
        config_path = (
            self.path
            / "graphroot"
            / "overlay-containers"
            / inspection["Id"]
            / "userdata"
            / "config.json"
        )
        config = json.loads(config_path.read_text())
        files = []
        for mount in config["mounts"]:
            if mount["destination"] not in {
                "/etc/hosts",
                "/etc/hostname",
                "/etc/resolv.conf",
                "/run/.containerenv",
            }:
                continue
            source = Path(mount["source"]).resolve()
            if not source.is_relative_to(self.path.resolve()) or not source.is_file():
                raise CheckpointCapabilityError(
                    "Runtime bind file is outside the private ZFS dataset"
                )
            files.append(str(source.relative_to(self.path.resolve())))
        (self.path / "restore-mounts.json").write_text(
            json.dumps(
                {
                    "snapshot": reference.split("@", 1)[1],
                    "files": files,
                    "container_id": inspection["Id"],
                    "stopped_tasks": sorted(
                        {
                            pid
                            for session in sessions
                            for pid in session.get("pids", ())
                            if pid is not None
                        }
                    ),
                }
            )
        )

    def snapshot(self, reference):
        _command(["zfs", "snapshot", reference])

    def snapshot_stats(self, reference):
        raw = _command(
            [
                "zfs",
                "get",
                "-Hp",
                "-o",
                "property,value",
                "used,referenced,logicalreferenced",
                reference,
            ]
        ).decode()
        values = dict(line.split("\t", 1) for line in raw.splitlines())
        return {
            "zfs_dataset": reference.split("@", 1)[0],
            "zfs_snapshot_used_bytes": int(values["used"]),
            "zfs_referenced_bytes": int(values["referenced"]),
            "zfs_logical_referenced_bytes": int(values["logicalreferenced"]),
        }

    def rollback(self, reference):
        # No -r/-R: never discard a newer or unrelated snapshot implicitly.
        _command(["zfs", "rollback", reference])

    def delete(self, reference):
        if not reference.startswith(self.dataset + "@agency-"):
            raise ValueError("Checkpoint does not belong to this sandbox dataset")
        _command(["zfs", "destroy", reference])

    def capture_dataset(self, sandbox):
        if self.runtime == "podman":
            return self.dataset
        inspection = json.loads(
            sandbox._run(["docker", "inspect", sandbox._name], check=True).stdout
        )[0]
        dataset = inspection.get("GraphDriver", {}).get("Data", {}).get("Dataset")
        if not isinstance(dataset, str) or not dataset.startswith(self.parent + "/"):
            raise CheckpointCapabilityError(
                "Docker did not report a container-owned ZFS dataset under checkpoint_zfs_parent"
            )
        if self.dataset is not None and dataset != self.dataset:
            raise CheckpointCapabilityError("Docker writable dataset identity changed")
        self.dataset = dataset
        return dataset

    def destroy(self):
        if self.runtime == "docker":
            if self.dataset is not None:
                snapshots = (
                    _command(
                        ["zfs", "list", "-H", "-o", "name", "-t", "snapshot", "-r", self.dataset]
                    )
                    .decode()
                    .splitlines()
                )
                for snapshot in snapshots:
                    if snapshot.startswith(self.dataset + "@agency-"):
                        _command(["zfs", "destroy", snapshot])
            # Keep the runtime usable until _ContainerBackendBase.destroy()
            # removes the container.  Marking this unprepared here makes
            # _run() reject the subsequent `docker rm`, leaking the live
            # container and its writable dataset.
            return
        if self.created:
            # The generated dataset is private; no shared runtime store is removed.
            # Runtime teardown and short-lived host readers can retain a VFS
            # reference briefly after the container's mounts disappear. Retry
            # only EBUSY's diagnostic, without forcing mounts or killing readers.
            for attempt in range(10):
                try:
                    _command(["zfs", "destroy", "-r", self.dataset])
                    break
                except CheckpointCapabilityError as exc:
                    if "dataset is busy" not in str(exc) or attempt == 9:
                        raise
                    time.sleep(min(0.1 * 2**attempt, 1.0))
            self.created = False
            self.prepared = False


def _criu_stats(raw, key="podman_criu_stats"):
    try:
        result = json.loads(raw)
    except (ValueError, TypeError):
        return {key + "_unavailable": True}
    return {key: result}


def check_process_resources(sandbox):
    """Verify CRIU can own the detached tree and measure its resident memory."""
    pids = sandbox._own_host_pids()
    if not pids:
        raise CheckpointCapabilityError("Cannot enumerate the running container's host processes")
    memory = 0
    for pid in pids:
        try:
            status = Path(f"/proc/{pid}/status").read_text()
        except FileNotFoundError:
            continue
        fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
        if int(fields.get("TracerPid", "0")):
            raise CheckpointCapabilityError(
                f"PID {pid} is ptrace-owned after the checkpoint handoff "
                f"(TracerPid={fields['TracerPid'].strip()})"
            )
        memory += int(fields.get("VmRSS", "0 kB").split()[0]) * 1024
    return memory


class CowZfsCheckpoint:
    def __init__(self):
        self.latest = None
        self.hibernated = False
        self.manifest = None
        self.generation = 0
        self.fast_artifact = None
        self._pending_reference = None

    def _daemon_handles(self, sandbox):
        return list(getattr(sandbox, "_agency_harness_daemon_handles", {}).values())

    def _prepare_daemons(self, sandbox):
        handles = self._daemon_handles(sandbox)
        prepared = []
        sessions = []
        live_pty = False
        try:
            for handle in handles:
                with handle.client(timeout_s=15) as client:
                    result = client.prepare_fast_checkpoint()
                prepared.append(handle)
                sessions.append(result)
                live_pty = live_pty or bool(result.get("live_pty"))
        except BaseException:
            self._resume_daemons(prepared, abort=True)
            raise
        return prepared, sessions, live_pty

    @staticmethod
    def _resume_daemons(handles, *, abort=False):
        for handle in handles:
            with handle.client(timeout_s=15) as client:
                if abort:
                    client.abort_fast_checkpoint()
                else:
                    client.complete_fast_restore()

    def _delete_fast_artifact(self, sandbox, artifact):
        if artifact is None:
            return
        if sandbox._runtime == "docker":
            sandbox._run(
                ["docker", "checkpoint", "rm", sandbox._name, artifact["name"]],
                check=False,
                timeout=30,
            )

    def _runtime_checkpoint(self, sandbox):
        if sandbox._gpu_count_requested:
            raise CheckpointCapabilityError("CRIU fast resume is disabled for GPU sandboxes")
        storage = sandbox._checkpoint_storage
        if sandbox._runtime != "podman" or not storage.criu_ready:
            detail = (
                storage.criu_setup_error or "supported only with private rootful Podman storage"
            )
            raise CheckpointCapabilityError(f"CRIU fast resume is unavailable: {detail}")
        prepared, sessions, live_pty = self._prepare_daemons(sandbox)
        name = "agency-" + uuid.uuid4().hex
        try:
            memory = check_process_resources(sandbox)
            tracking = {
                key: copy.deepcopy(getattr(sandbox, key, None))
                for key in (
                    "_watched_pids",
                    "_baseline_pids",
                    "_daemon_pids",
                    "_ptrace_managed_pids",
                )
            }
            inspection = json.loads(
                sandbox._run(["podman", "inspect", sandbox._name], check=True).stdout
            )[0]
            if inspection.get("ExecIDs") or inspection.get("ExecSessions"):
                raise CheckpointCapabilityError(
                    "Active runtime exec sessions cannot be checkpointed; daemonize under container init"
                )
            if inspection.get("Config", {}).get("Tty"):
                raise CheckpointCapabilityError("Runtime-owned external terminals are unsupported")
            reference = self._pending_reference
            storage.prepare_restore_mounts(inspection, reference, sessions)
            command = [
                "podman",
                "container",
                "checkpoint",
                "--keep",
                "--file-locks",
                "--tcp-established",
                "--print-stats",
                sandbox._name,
            ]
            with agprof.span("checkpoint.process_dump"):
                raw = sandbox._run(
                    command,
                    check=True,
                    timeout=sandbox._agconfig.sandbox.checkpoint_setup_timeout_s,
                ).stdout
            sandbox._mark_runtime_checkpoint_stopped()
        except BaseException:
            if sandbox._container_running():
                self._resume_daemons(prepared, abort=True)
            else:
                sandbox._mark_runtime_checkpoint_stopped()
            raise
        return {
            "name": name,
            "daemon_count": len(prepared),
            "live_pty": live_pty,
            "sessions": sessions,
            "process_memory_rss_bytes": memory,
            **_criu_stats(raw),
            "pid_tracking": tracking,
        }

    def _runtime_restore(self, sandbox, artifact):
        handles = self._daemon_handles(sandbox)
        if len(handles) != artifact["daemon_count"]:
            raise CheckpointCapabilityError("harness daemon set changed before fast restore")
        for handle in handles:
            socket_path = Path(handle.sandbox_uds_path)
            if socket_path.is_socket():
                socket_path.unlink()
            elif socket_path.exists():
                raise CheckpointCapabilityError(f"Expected harness control socket at {socket_path}")
        command = [
            "podman",
            "container",
            "restore",
            "--keep",
            "--file-locks",
            "--tcp-established",
            "--print-stats",
            sandbox._name,
        ]
        sandbox._acquire_runtime_slot()
        acquired = True
        try:
            with agprof.span("restore.process"):
                raw = sandbox._run(
                    command,
                    check=True,
                    timeout=sandbox._agconfig.sandbox.checkpoint_setup_timeout_s,
                ).stdout
            if not sandbox._container_running():
                raise CheckpointCapabilityError(
                    "CRIU restore returned without a running process tree"
                )
            for handle in handles:
                with handle.client(timeout_s=15) as client:
                    client.seize_fast_restore()
            from ._criu_restore_mounts import thaw_retained_tasks

            thaw_retained_tasks(sandbox._checkpoint_storage.path)
            self._resume_daemons(handles)
            for key, value in artifact["pid_tracking"].items():
                setattr(sandbox, key, copy.deepcopy(value))
            sandbox._infrastructure_pids = {}
            artifact["restore_stats"] = _criu_stats(raw, "podman_criu_restore_stats")
            acquired = False  # Running container owns the acquired slot.
        finally:
            if acquired:
                if sandbox._container_running():
                    try:
                        sandbox._stop_image()
                    except Exception as stop_exc:
                        logging.getLogger(__name__).warning(
                            "Could not stop partial CRIU restore cleanly: %s", stop_exc
                        )
                else:
                    sandbox._release_runtime_slot()

    @staticmethod
    def _discard_daemon_handles(sandbox):
        getattr(sandbox, "_agency_harness_daemon_handles", {}).clear()

    def _check_handle(self, sandbox, checkpoint):
        if checkpoint != self.latest or checkpoint.container != sandbox._name:
            raise ValueError("Only this sandbox's latest COW checkpoint can be restored")

    def checkpoint(self, sandbox, tag=None):
        if self.hibernated or not sandbox._checkpoint_storage.prepared:
            return self._checkpoint(sandbox, tag)
        with agprof.span("checkpoint.total"):
            _annotate(sandbox, "cow_zfs")
            started = time.monotonic()
            handle = self._checkpoint(sandbox, tag)
            if handle is not None:
                handle.stats["checkpoint_total_seconds"] = time.monotonic() - started
                agprof.annotate(**handle.stats)
            return handle

    def _checkpoint(self, sandbox, tag=None):
        if tag is not None:
            raise CheckpointCapabilityError(
                "cow_zfs checkpoints are local handles, not portable image tags"
            )
        if self.hibernated:
            return self.latest
        storage = sandbox._checkpoint_storage
        if not storage.prepared:
            return None  # A sandbox that never started has no state to capture.
        manifest = sandbox._normalized_checkpoint_manifest()
        if self.manifest is None:
            self.manifest = manifest
        elif manifest != self.manifest:
            raise CheckpointCapabilityError(
                "Runtime launch configuration changed after the first local checkpoint"
            )
        previous = self.latest
        previous_fast = self.fast_artifact
        if previous_fast is not None:
            # The process image is only a cache.  Discarding it before the
            # next dump cannot compromise the still-published ZFS checkpoint.
            self._delete_fast_artifact(sandbox, previous_fast)
            self.fast_artifact = None
        dataset = storage.capture_dataset(sandbox)
        reference = f"{dataset}@agency-{uuid.uuid4().hex}"
        self._pending_reference = reference
        started = time.monotonic()
        fast_artifact = None
        fast_error = None
        with agprof.span("checkpoint.quiesce"):
            try:
                if sandbox._agconfig.sandbox.checkpoint_fast_resume:
                    fast_artifact = self._runtime_checkpoint(sandbox)
            except Exception as exc:
                fast_error = type(exc).__name__
                logging.getLogger(__name__).warning(
                    "CRIU fast checkpoint unavailable; using filesystem checkpoint: %s",
                    exc,
                )
            finally:
                self._pending_reference = None
            if fast_artifact is None and sandbox._container_running():
                sandbox._stop_image()
        fs_started = time.monotonic()
        with agprof.span("checkpoint.fs_snapshot"):
            _annotate(sandbox, "cow_zfs")
            storage.snapshot(reference)
        self.generation += 1
        stats = {
            "checkpoint_schema_version": 1,
            "checkpoint_generation": self.generation,
            "checkpoint_created_wall_ns": time.time_ns(),
            "fs_snapshot_seconds": time.monotonic() - fs_started,
            "checkpoint_total_seconds": time.monotonic() - started,
            "fast_resume_available": fast_artifact is not None,
            "fast_resume_live_pty": bool(fast_artifact and fast_artifact["live_pty"]),
            **(
                {
                    "sessions": fast_artifact.get("sessions", []),
                    "process_memory_rss_bytes": fast_artifact.get("process_memory_rss_bytes", 0),
                    **{
                        key: value
                        for key, value in fast_artifact.items()
                        if key in {"podman_criu_stats", "criu_stats_unavailable"}
                    },
                }
                if fast_artifact
                else {}
            ),
            **({"fast_resume_checkpoint_error": fast_error} if fast_error else {}),
            **storage.snapshot_stats(reference),
        }
        self.latest = CheckpointHandle("cow_zfs", sandbox._runtime, sandbox._name, reference, stats)
        self.hibernated = True
        self.fast_artifact = fast_artifact
        agprof.annotate(**stats)
        if previous is not None:
            with agprof.span("checkpoint.cleanup"):
                try:
                    storage.delete(previous.reference)
                except Exception as exc:
                    # A committed transaction must not become a failed agent
                    # invocation merely because old storage could not be freed.
                    agprof.annotate(cleanup_error_type=type(exc).__name__)
                    logging.getLogger(__name__).warning(
                        "Old COW checkpoint cleanup failed: %s", exc
                    )
        return self.latest

    def restore(self, sandbox, checkpoint):
        self._check_handle(sandbox, checkpoint)
        if not self.hibernated:
            raise CheckpointCapabilityError(
                "Restore requires hibernation/discard of the current live state"
            )
        with agprof.span("restore.total"):
            _annotate(sandbox, "cow_zfs")
            if sandbox._container_running():
                raise CheckpointCapabilityError("Refusing ZFS rollback while container is running")
            with agprof.span("restore.fs"):
                _annotate(sandbox, "cow_zfs")
                sandbox._checkpoint_storage.rollback(checkpoint.reference)
            self.hibernated = False
            artifact = self.fast_artifact
            if artifact is not None:
                try:
                    self._runtime_restore(sandbox, artifact)
                    checkpoint.stats["fast_resume_used"] = True
                    checkpoint.stats.update(artifact.get("restore_stats", {}))
                    agprof.annotate(fast_resume_used=True)
                    return
                except Exception as exc:
                    checkpoint.stats.update(
                        fast_resume_used=False,
                        fast_resume_restore_error=type(exc).__name__,
                    )
                    agprof.annotate(
                        fast_resume_used=False,
                        fast_resume_restore_error=type(exc).__name__,
                    )
                    logging.getLogger(__name__).warning(
                        "CRIU fast restore failed; starting from ZFS snapshot: %s", exc
                    )
                    self._discard_daemon_handles(sandbox)
                    if sandbox._container_running():
                        sandbox._stop_image()
                    # A partial CRIU restore may have dirtied the rootfs.
                    sandbox._checkpoint_storage.rollback(checkpoint.reference)
                    self.fast_artifact = None

    def delete_checkpoint(self, sandbox, checkpoint):
        self._check_handle(sandbox, checkpoint)
        if self.hibernated:
            raise CheckpointCapabilityError(
                "Cannot delete the active checkpoint of a hibernated sandbox"
            )
        sandbox._checkpoint_storage.delete(checkpoint.reference)
        self._delete_fast_artifact(sandbox, self.fast_artifact)
        self.fast_artifact = None
        self.latest = None


def checkpoint_backend(name):
    if name == "image_commit":
        return ImageCommitCheckpoint()
    if name == "cow_zfs":
        return CowZfsCheckpoint()
    raise ValueError(f"Unknown checkpoint_backend {name!r}")
