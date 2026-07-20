"""Docker-specific container backend.

Nearly everything about running a container is identical between Docker and
Podman and lives in `._ContainerBackendBase` (`.container`), including the
session-keyring-quota machinery -- see that module's docstring for why both
runtimes are subject to the same kernel quota. `_DockerBackend` itself is
mostly just the `_runtime` tag; see `.podman._PodmanBackend` for the other
thing that differs between the two (`_resolve_image`).

The one substantial override here is `_locate_layer_diff_dir()`, feeding
`_ContainerBackendBase._fold_commit_into_accumulator()`'s fast incremental
squashing path (see docs/agsandbox_backends/container.md's "Fast
incremental squashing" section). It reaches into Docker's own undocumented
overlay2 graphdriver on-disk layout -- confirmed empirically during
development, not from published docs -- which has no verified Podman
equivalent (Podman uses a different storage backend, `containers/storage`).
The base class's default implementation returns None unconditionally,
which `_fold_commit_into_accumulator()` treats as "no fast path available,
fall back to `_squash_commit()`" -- so `_PodmanBackend` not overriding this
(yet) is completely safe, just slower at squash time until someone
verifies Podman's own storage layout and adds the equivalent override here.
"""

from __future__ import annotations

import json
import os
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

    def _locate_layer_diff_dir(self, diff_id: str) -> "Path | None":
        """Find the raw overlay2 diff directory backing *diff_id* directly
        on disk (`<data_root>/overlay2/<cache-id>/diff/`), bypassing
        `docker diff`/`docker save` entirely -- see
        docs/agsandbox_backends/container.md's "Fast incremental
        squashing" section for the full rationale and empirical
        verification this is based on. This directory is exactly the
        same data a `docker commit` producing this layer already read to
        build it -- reading it again is nearly free, unlike `docker
        diff` (a generic scan costing ~9s on a real ~24GB/many-file image
        regardless of how much actually changed) or `docker save` (cost
        proportional to the whole image).

        Returns None for a missing/unexpected layerdb entry or a
        non-overlay2 storage driver -- the base class's caller treats
        None as "the accumulator can't be trusted," never as an error.
        """
        root_and_driver = self._docker_data_root_and_driver()
        if root_and_driver is None:
            return None
        data_root, driver = root_and_driver
        if driver != "overlay2":
            return None
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
