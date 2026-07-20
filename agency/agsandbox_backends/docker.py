"""Docker-specific container backend.

Nearly everything about running a container is identical between Docker and
Podman and lives in `._ContainerBackendBase` (`.container`), including the
session-keyring-quota machinery -- see that module's docstring for why both
runtimes are subject to the same kernel quota. `_DockerBackend` itself is
mostly just the `_runtime` tag plus the Docker-specific fast-squash
storage hooks; see `.podman._PodmanBackend` for Podman's equivalents
(`_resolve_image`, plus its own `_locate_layer_diff_dir` /
`_host_to_container_id` against `containers/storage`).

The substantial override here is `_locate_layer_diff_dir()`, feeding
`_ContainerBackendBase._fold_commit_into_accumulator()`'s fast incremental
squashing path (see docs/agsandbox_backends/container.md's "Fast
incremental squashing" section). It supports two Docker storage backends,
both confirmed empirically:

- Classic moby **overlay2** graphdriver: `<DockerRootDir>/image/overlay2/layerdb`
  → `cache-id` → `<DockerRootDir>/overlay2/<cache-id>/diff/`.
- Containerd **overlayfs** snapshotter (`Driver: overlayfs` /
  `io.containerd.snapshotter.v1`): ChainID from the image's RootFS.Layers
  → `ctr -n moby snapshots view/mounts` →
  `/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs/snapshots/<id>/fs/`.

The snapshotter `fs/` directory (and usually the containerd root) must be
readable by this process for the fold to succeed -- same constraint classic
overlay2 has on rootful `/var/lib/docker`. When lookup or read fails, the
base class falls back to `_squash_commit()` export/import.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import uuid
from pathlib import Path

from .container import _ContainerBackendBase


def _translate_id(host_id: int, id_map: "list[tuple[int, int, int]]") -> int:
    """Reverse-lookup a host-side uid/gid through a parsed
    /proc/<pid>/uid_map (or gid_map) -style mapping: each entry is
    `(namespace_start, host_start, length)`. Returns the id unchanged if
    it doesn't fall in any mapped range -- an id we don't understand is
    safer left alone than guessed at."""
    for ns_start, host_start, length in id_map:
        if host_start <= host_id < host_start + length:
            return ns_start + (host_id - host_start)
    return host_id


def _chain_id_for_diff_ids(diff_ids: "list[str]") -> "str | None":
    """OCI/containerd ChainID for *diff_ids* (most-base-first).

    ChainID(L0) = DiffID(L0); ChainID(Ln) = sha256(ChainID(Ln-1) + " " + DiffID(Ln)),
    where both sides of the concatenation keep their `sha256:` prefix
    (containerd's definition). Returns None for an empty list.
    """
    if not diff_ids:
        return None
    chain = diff_ids[0]
    for diff_id in diff_ids[1:]:
        chain = "sha256:" + hashlib.sha256(f"{chain} {diff_id}".encode()).hexdigest()
    return chain


def _parse_ctr_mounts_top_fs(mounts_stdout: str) -> "Path | None":
    """Extract the top layer's on-disk fs directory from `ctr snapshots mounts`
    output. For an overlay mount the first `lowerdir=` component is the
    uppermost (the snapshot we viewed); for a single-layer bind mount the
    bind source is that directory."""
    m = re.search(r"lowerdir=([^,\s]+)", mounts_stdout)
    if m:
        first = m.group(1).split(":")[0]
        return Path(first) if first else None
    m = re.search(r"(?:--bind|-o\s+bind)\s+(\S+)", mounts_stdout)
    if m:
        return Path(m.group(1))
    # `mount --bind SRC DST` without -o bind
    m = re.search(r"^\s*mount\s+(\S+)\s+(\S+)\s*$", mounts_stdout, re.MULTILINE)
    if m and m.group(1) not in ("-t", "--bind"):
        # unlikely; prefer explicit patterns above
        pass
    m = re.search(r"mount\s+--bind\s+(\S+)\s+\S+", mounts_stdout)
    if m:
        return Path(m.group(1))
    return None


class _DockerBackend(_ContainerBackendBase):
    """Manages a single container for one agent via Docker."""

    _runtime = "docker"

    def _docker_info(self) -> "dict | None":
        """Full `docker info` JSON, cached for this backend's lifetime --
        the fields this module reads from it (DockerRootDir, Driver,
        SecurityOptions) are all fixed for as long as the daemon runs.
        None on any failure (unreachable info command, malformed output)
        so callers fall back gracefully.
        """
        cached = getattr(self, "_docker_info_cache", "unset")
        if cached != "unset":
            return cached
        try:
            info = json.loads(
                self._run(
                    [self._runtime, "info", "--format", "{{json .}}"],
                    check=True,
                    timeout=self.inspect_timeout_s,
                ).stdout.decode("utf-8", errors="replace")
            )
        except Exception:
            info = None
        self._docker_info_cache = info
        return info

    def _docker_data_root_and_driver(self) -> "tuple[Path, str] | None":
        """`(DockerRootDir, Driver)`. Returns None on any failure; this
        and everything built on it reaches into undocumented internals,
        so every step degrades to "unknown, use the slow path" rather
        than raising.
        """
        info = self._docker_info()
        if info is None:
            return None
        try:
            return (Path(info["DockerRootDir"]), info.get("Driver", ""))
        except Exception:
            return None

    def _is_rootless(self) -> bool:
        """True iff `docker info` reports the `rootless` security option
        -- the officially documented, supported way to detect this
        (unlike everything else in this module, which reaches into
        undocumented internals)."""
        info = self._docker_info()
        if info is None:
            return False
        return "name=rootless" in (info.get("SecurityOptions") or [])

    def _find_dockerd_pid(self) -> "int | None":
        """The dockerd process's PID, identified as a process literally
        named "dockerd" owned by the SAME user running this Python
        process -- a reliable heuristic specifically for rootless
        Docker, where the daemon and the client necessarily run as the
        same host user by construction (there is no other supported way
        to run rootless Docker). Returns None if no such process is
        found (e.g. non-rootless Docker, where the daemon commonly runs
        as a different user, or /proc isn't readable this way)."""
        my_uid = os.getuid()
        try:
            for pid_str in os.listdir("/proc"):
                if not pid_str.isdigit():
                    continue
                try:
                    with open(f"/proc/{pid_str}/comm") as f:
                        if f.read().strip() != "dockerd":
                            continue
                    if os.stat(f"/proc/{pid_str}").st_uid != my_uid:
                        continue
                    return int(pid_str)
                except (OSError, PermissionError):
                    continue
        except OSError:
            pass
        return None

    @staticmethod
    def _parse_id_map(path: Path) -> "list[tuple[int, int, int]] | None":
        try:
            entries = []
            for line in path.read_text().splitlines():
                parts = line.split()
                if len(parts) != 3:
                    continue
                entries.append((int(parts[0]), int(parts[1]), int(parts[2])))
            return entries or None
        except (OSError, ValueError):
            return None

    def _rootless_id_maps(self) -> "tuple[list, list] | None":
        """`(uid_map, gid_map)` entries for the rootless daemon's own user
        namespace (see `_host_to_container_id()`), cached for this
        backend's lifetime -- a kernel-level property of the running
        daemon process, fixed for its whole lifetime. None if
        unavailable (dockerd's PID not found, /proc unreadable, etc.)."""
        cached = getattr(self, "_rootless_id_maps_cache", "unset")
        if cached != "unset":
            return cached
        result = None
        pid = self._find_dockerd_pid()
        if pid is not None:
            uid_map = self._parse_id_map(Path(f"/proc/{pid}/uid_map"))
            gid_map = self._parse_id_map(Path(f"/proc/{pid}/gid_map"))
            if uid_map is not None and gid_map is not None:
                result = (uid_map, gid_map)
        self._rootless_id_maps_cache = result
        return result

    def _host_to_container_id(self, uid: int, gid: int) -> "tuple[int, int]":
        """See `_ContainerBackendBase._host_to_container_id()`'s
        docstring for why this translation is needed at all. Only
        applies it when this daemon is confirmed rootless AND its user
        namespace maps were readable; otherwise identity, matching
        non-rootless Docker (which needs no translation -- the overlay2
        diff directory's on-disk ownership already IS what the container
        sees)."""
        if not self._is_rootless():
            return (uid, gid)
        maps = self._rootless_id_maps()
        if maps is None:
            return (uid, gid)
        uid_map, gid_map = maps
        return (_translate_id(uid, uid_map), _translate_id(gid, gid_map))

    def _ctr_argv(self) -> "list[str] | None":
        """Argv prefix for talking to containerd (`ctr` or `sudo -n ctr`),
        cached for this backend's lifetime. Docker's containerd-snapshotter
        mode stores image layers in the `moby` namespace; rootful installs
        typically restrict the containerd socket to root, so passwordless
        `sudo -n ctr` is tried when bare `ctr` can't connect. None if
        neither works -- caller falls back to the slow squash path."""
        cached = getattr(self, "_ctr_argv_cache", "unset")
        if cached != "unset":
            return cached
        result = None
        for prefix in (["ctr"], ["sudo", "-n", "ctr"]):
            try:
                completed = subprocess.run(
                    prefix + ["version"],
                    capture_output=True,
                    timeout=self.inspect_timeout_s,
                )
            except Exception as _e:
                # Expected when this prefix can't even be launched (missing
                # binary, sudo denied, …); try the next candidate.
                print(
                    f"[agsandbox_backend] WARNING: ctr probe {' '.join(prefix)} failed: {_e}",
                    file=__import__("sys").stderr,
                    flush=True,
                )
                completed = None
            if completed is not None and completed.returncode == 0:
                result = prefix
                break
        self._ctr_argv_cache = result
        return result

    def _locate_containerd_overlayfs_diff_dir(self, diff_ids: "list[str]") -> "Path | None":
        """Resolve a layer chain to its containerd overlayfs snapshot `fs/`
        directory via `ctr -n moby snapshots view` + `mounts`. *diff_ids*
        must be the image's full RootFS.Layers list (most-base-first)
        ending at the layer whose diff we want -- ChainID is a function of
        the whole prefix, not the tip DiffID alone.
        """
        chain_id = _chain_id_for_diff_ids(diff_ids)
        if chain_id is None:
            return None
        ctr = self._ctr_argv()
        if ctr is None:
            return None
        view = f"agency-locate-{uuid.uuid4().hex}"
        ctr_n = ctr + ["-n", "moby"]
        try:
            created = subprocess.run(
                ctr_n + ["snapshots", "view", view, chain_id],
                capture_output=True,
                timeout=self.inspect_timeout_s,
            )
            if created.returncode != 0:
                return None
            mounts = subprocess.run(
                ctr_n + ["snapshots", "mounts", "/tmp/agency-ctr-unused", view],
                capture_output=True,
                text=True,
                timeout=self.inspect_timeout_s,
            )
            if mounts.returncode != 0:
                return None
            diff_dir = _parse_ctr_mounts_top_fs(mounts.stdout or "")
            if diff_dir is None:
                return None
            # Rootful containerd creates snapshot dirs as 0700. When we
            # reached ctr via sudo -n, also open this one snapshot for
            # the current user so overlay_diff_to_tar can read it --
            # scoped to diff_dir's parent (snapshots/<id>/), not the
            # whole containerd tree.
            if ctr[:2] == ["sudo", "-n"]:
                try:
                    subprocess.run(
                        ["sudo", "-n", "chmod", "-R", "a+rX", str(diff_dir.parent)],
                        capture_output=True,
                        timeout=self.inspect_timeout_s,
                    )
                except Exception as _e:
                    # Best-effort: fold will see an unreadable dir and
                    # degrade to None below rather than hard-failing.
                    print(
                        f"[agsandbox_backend] WARNING: could not chmod "
                        f"containerd snapshot {diff_dir.parent} for fast squash: {_e}",
                        file=__import__("sys").stderr,
                        flush=True,
                    )
            try:
                if not diff_dir.is_dir():
                    return None
                next(diff_dir.iterdir(), None)
            except OSError:
                return None
            return diff_dir
        except Exception as _e:
            # Any unexpected ctr/mounts failure: fast path unavailable,
            # caller falls back to export/import.
            print(
                f"[agsandbox_backend] WARNING: containerd overlayfs layer lookup failed: {_e}",
                file=__import__("sys").stderr,
                flush=True,
            )
            return None
        finally:
            try:
                subprocess.run(
                    ctr_n + ["snapshots", "rm", view],
                    capture_output=True,
                    timeout=self.inspect_timeout_s,
                )
            except Exception as _e:
                # Best-effort cleanup of the temporary view snapshot.
                print(
                    f"[agsandbox_backend] WARNING: could not remove temporary "
                    f"ctr snapshot view {view}: {_e}",
                    file=__import__("sys").stderr,
                    flush=True,
                )

    def _locate_overlay2_layer_diff_dir(self, data_root: Path, diff_id: str) -> "Path | None":
        """Classic moby overlay2 graphdriver: layerdb `diff` → `cache-id`
        → `<data_root>/overlay2/<cache-id>/diff/`."""
        layerdb_root = data_root / "image" / "overlay2" / "layerdb" / "sha256"
        try:
            for entry in layerdb_root.iterdir():
                try:
                    if (entry / "diff").read_text().strip() != diff_id:
                        continue
                    cache_id = (entry / "cache-id").read_text().strip()
                except (OSError, UnicodeDecodeError):
                    continue
                diff_dir = data_root / "overlay2" / cache_id / "diff"
                return diff_dir if diff_dir.is_dir() else None
        except OSError:
            return None
        return None

    def _locate_layer_diff_dir(
        self, diff_id: str, *, diff_ids: "list[str] | None" = None
    ) -> "Path | None":
        """Find the raw on-disk layer diff directory backing *diff_id*.

        Dispatches on `docker info`'s `Driver`:
        - `overlay2`: classic graphdriver layout under DockerRootDir.
        - `overlayfs`: containerd snapshotter; needs *diff_ids* (full
          RootFS.Layers chain ending at *diff_id*) to compute ChainID.

        Returns None when the driver is unsupported, the chain isn't
        provided for overlayfs, or lookup/read fails -- the base class
        treats None as "accumulator unavailable," never as an error.
        """
        root_and_driver = self._docker_data_root_and_driver()
        if root_and_driver is None:
            return None
        data_root, driver = root_and_driver
        if driver == "overlay2":
            return self._locate_overlay2_layer_diff_dir(data_root, diff_id)
        if driver == "overlayfs":
            if not diff_ids or diff_ids[-1] != diff_id:
                return None
            return self._locate_containerd_overlayfs_diff_dir(diff_ids)
        return None
