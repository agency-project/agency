"""Image-materialized and local ZFS filesystem checkpoint mechanisms."""

from __future__ import annotations

import json
import hashlib
import logging
import os
import platform
import re
import shutil
import subprocess
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
        self.dataset = (
            f"{self.parent}/agency-{uuid.uuid4().hex}" if self.runtime == "podman" else None
        )
        self.path = None
        self.prepared = False
        self.created = False

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


class CowZfsCheckpoint:
    def __init__(self):
        self.latest = None
        self.hibernated = False
        self.manifest = None
        self.generation = 0

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
        dataset = storage.capture_dataset(sandbox)
        reference = f"{dataset}@agency-{uuid.uuid4().hex}"
        started = time.monotonic()
        with agprof.span("checkpoint.quiesce"):
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
            **storage.snapshot_stats(reference),
        }
        self.latest = CheckpointHandle("cow_zfs", sandbox._runtime, sandbox._name, reference, stats)
        self.hibernated = True
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

    def delete_checkpoint(self, sandbox, checkpoint):
        self._check_handle(sandbox, checkpoint)
        if self.hibernated:
            raise CheckpointCapabilityError(
                "Cannot delete the active checkpoint of a hibernated sandbox"
            )
        sandbox._checkpoint_storage.delete(checkpoint.reference)
        self.latest = None


def checkpoint_backend(name):
    if name == "image_commit":
        return ImageCommitCheckpoint()
    if name == "cow_zfs":
        return CowZfsCheckpoint()
    raise ValueError(f"Unknown checkpoint_backend {name!r}")
