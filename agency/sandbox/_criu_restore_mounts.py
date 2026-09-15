"""CRIU action: restore runtime-owned bind files before tracees resume.

Podman may recreate/truncate hosts, hostname, resolv.conf and .containerenv on
restore. Read their original bytes/metadata directly from the ZFS snapshot;
this is not a filesystem checkpoint implementation or a rootfs copy.
"""

import json
import os
from pathlib import Path
import stat
import time


def restore_runtime_files(root):
    manifest = json.loads((root / "restore-mounts.json").read_text())
    snapshot = manifest["snapshot"]
    if not snapshot.startswith("agency-") or "/" in snapshot or ".." in snapshot:
        raise ValueError("Invalid snapshot name")
    runtime = manifest.get("runtime", "podman")
    for entry in manifest["files"]:
        if runtime == "podman":
            relative_path = Path(entry)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError("Invalid runtime bind path")
            target = root / relative_path
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError("Runtime bind escaped its private dataset")
            original = root / ".zfs" / "snapshot" / snapshot / relative_path
        elif runtime == "docker":
            runtime_root = Path(manifest["runtime_root"]).resolve()
            target = Path(entry["target"]).resolve()
            backup = Path(entry["backup"])
            if (
                not runtime_root.is_absolute()
                or not target.is_relative_to(runtime_root)
                or backup.is_absolute()
                or ".." in backup.parts
            ):
                raise ValueError("Invalid Docker runtime bind path")
            original = (root / backup).resolve()
            if not original.is_relative_to(root.resolve()):
                raise ValueError("Docker runtime backup escaped its private directory")
        else:
            raise ValueError("Invalid checkpoint runtime")
        metadata = original.stat()
        if not stat.S_ISREG(metadata.st_mode) or target.is_symlink():
            raise ValueError("Expected a regular runtime bind file")
        # Write the existing inode: restored FDs may already reference it.
        with target.open("wb") as stream:
            stream.write(original.read_bytes())
            stream.flush()
            os.fchown(stream.fileno(), metadata.st_uid, metadata.st_gid)
            os.fchmod(stream.fileno(), stat.S_IMODE(metadata.st_mode))
        os.utime(target, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))


def freeze_retained_tasks(root, init_pid):
    """Freeze only retained CLI groups before CRIU releases its ptrace stops."""
    required = set(json.loads((root / "restore-mounts.json").read_text())["stopped_tasks"])
    if not required:
        return
    if any(not isinstance(pid, int) or pid <= 1 for pid in required):
        raise ValueError("Invalid retained task identity")
    namespace = os.readlink(f"/proc/{init_pid}/ns/pid")
    found = set()
    groups = set()
    for process in Path("/proc").glob("[0-9]*"):
        try:
            if os.readlink(process / "ns/pid") != namespace:
                continue
            for task in (process / "task").iterdir():
                status = dict(
                    line.split(":", 1)
                    for line in (task / "status").read_text().splitlines()
                    if ":" in line
                )
                namespace_pid = int(status["NSpid"].split()[-1])
                if namespace_pid in required:
                    found.add(namespace_pid)
                    groups.add(int(status["Tgid"]))
        except (FileNotFoundError, ProcessLookupError):
            continue
    if found != required:
        raise RuntimeError(
            f"Retained task inventory missing before resume: {sorted(required - found)}"
        )
    control = Path("/sys/fs/cgroup")
    init_group = Path(f"/proc/{init_pid}/cgroup").read_text().strip().split("0::", 1)[1]
    parent = control / init_group.lstrip("/")
    checkpoint = json.loads((root / "restore-mounts.json").read_text())
    container = checkpoint["container_id"]
    runtime = checkpoint.get("runtime", "podman")
    expected_scope = (
        f"libpod-{container}.scope" if runtime == "podman" else f"docker-{container}.scope"
    )
    if runtime not in {"docker", "podman"} or parent.name != expected_scope:
        raise RuntimeError("Expected the restored container's private cgroup v2 scope")
    freezer = parent / "agency-criu-handoff"
    freezer.mkdir()
    (freezer / "cgroup.freeze").write_text("1")
    originals = {}
    for pid in groups:
        group = Path(f"/proc/{pid}/cgroup").read_text().strip().split("0::", 1)[1]
        original = control / group.lstrip("/")
        if not original.is_relative_to(parent):
            raise RuntimeError("Retained task escaped the restored container cgroup")
        originals[str(pid)] = str(original)
        (freezer / "cgroup.procs").write_text(str(pid))
    (root / "restore-freezer.json").write_text(
        json.dumps({"parent": str(parent), "tasks": originals})
    )
    deadline = time.monotonic() + 10
    while "frozen 1" not in (freezer / "cgroup.events").read_text():
        if time.monotonic() > deadline:
            raise RuntimeError("Restored CLI did not reach the cgroup freezer")
        time.sleep(0.002)


def thaw_retained_tasks(root):
    """Called by the host only after Agency SEIZE + INTERRUPT succeeds."""
    record = root / "restore-freezer.json"
    if not record.exists():
        return
    manifest = json.loads(record.read_text())
    checkpoint = json.loads((root / "restore-mounts.json").read_text())
    container = checkpoint["container_id"]
    runtime = checkpoint.get("runtime", "podman")
    parent = Path(manifest["parent"])
    expected_scope = (
        f"libpod-{container}.scope" if runtime == "podman" else f"docker-{container}.scope"
    )
    if (
        runtime not in {"docker", "podman"}
        or not parent.is_relative_to("/sys/fs/cgroup")
        or parent.name != expected_scope
    ):
        raise RuntimeError("Invalid restore freezer scope")
    freezer = parent / "agency-criu-handoff"
    for pid, original in manifest["tasks"].items():
        if not pid.isdecimal() or not Path(original).is_relative_to(parent):
            raise RuntimeError("Invalid restore freezer task")
    for pid, original in manifest["tasks"].items():
        (Path(original) / "cgroup.procs").write_text(pid)
    (freezer / "cgroup.freeze").write_text("0")
    freezer.rmdir()
    record.unlink()


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    action = os.environ.get("CRTOOLS_SCRIPT_ACTION")
    if action == "post-restore":
        restore_runtime_files(root)
    elif action == "pre-resume":
        freeze_retained_tasks(root, int(os.environ["CRTOOLS_INIT_PID"]))
