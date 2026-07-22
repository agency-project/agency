"""Podman-specific container backend.

Podman shares the session-keyring-derived concurrency slot and quota
handling (`_acquire_runtime_slot`/`_release_runtime_slot`/
`_is_quota_exhaustion_error`/`_wait_for_quota_slot`/`_quota_diagnostics`)
with Docker -- see `.container`'s module docstring for why rootless Podman
(via runc) is subject to the exact same kernel quota as Docker, despite its
per-container user namespaces. Nothing here overrides any of them. Beyond
the fully-qualified image name Podman needs for bare references
(`_resolve_image`), the other substantial overrides are the fast
incremental-squashing hooks:

`_locate_layer_diff_dir()` reaches into Podman's `containers/storage`
overlay layout -- confirmed empirically, not from a single published
"how to find a layer's diff dir" doc -- feeding
`_ContainerBackendBase._build_accumulator_for_squash()` the same way
`.docker._DockerBackend` feeds it from Docker's overlay2 graphdriver
(see docs/agsandbox_backends/container.md's "Fast incremental squashing"
section). The correlation is: `podman info`'s `store.graphRoot` +
`store.graphDriverName` → `<graphRoot>/overlay-layers/layers.json`
maps a layer's `diff-digest` (exactly what `podman inspect` reports in
`RootFS.Layers`) to a storage layer `id` →
`<graphRoot>/overlay/<id>/diff/`. Returns None for a missing/unexpected
entry or a non-`overlay` driver, which the base class treats as "no
fast path, fall back to `_squash_commit()`".

`_host_to_container_id()` translates HOST-side ownership of that raw
diff directory into CONTAINER-visible ownership for rootless Podman.
Unlike rootless Docker (which requires finding dockerd's PID and reading
`/proc/<pid>/uid_map`), Podman exposes the maps directly on
`podman info`'s `host.idMappings` -- and `host.security.rootless` is the
documented boolean for detecting rootless mode.
"""

from __future__ import annotations

import json
from pathlib import Path

from .container import _ContainerBackendBase


def _translate_id(host_id: int, id_map: "list[tuple[int, int, int]]") -> int:
    """Reverse-lookup a host-side uid/gid through a parsed id-mapping:
    each entry is `(namespace_start, host_start, length)` -- the same
    shape `/proc/<pid>/uid_map` uses and the shape Podman's
    `host.idMappings` entries convert to. Returns the id unchanged if
    it doesn't fall in any mapped range -- an id we don't understand is
    safer left alone than guessed at."""
    for ns_start, host_start, length in id_map:
        if host_start <= host_id < host_start + length:
            return ns_start + (host_id - host_start)
    return host_id


class _PodmanBackend(_ContainerBackendBase):
    """Manages a single container for one agent via Podman."""

    _runtime = "podman"

    def _resolve_image(self, name: str) -> str:
        """Prefix bare image names with ``localhost/``.

        Podman requires fully-qualified names when no unqualified-search
        registries are configured in /etc/containers/registries.conf.
        Docker accepts bare names fine, so this override is Podman-only.
        """
        if "/" not in name:
            return f"localhost/{name}"
        return name

    def _podman_info(self) -> "dict | None":
        """Full `podman info` JSON, cached for this backend's lifetime --
        the fields this module reads from it (store.graphRoot,
        store.graphDriverName, host.security.rootless, host.idMappings)
        are all fixed for as long as the storage/setup stays put. None
        on any failure so callers fall back gracefully.
        """
        cached = getattr(self, "_podman_info_cache", "unset")
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
        except Exception as _e:
            print(
                f"[agsandbox_backend] WARNING: `podman info` failed, fast squash path "
                f"unavailable for this backend's lifetime: {_e}",
                file=__import__("sys").stderr,
                flush=True,
            )
            info = None
        self._podman_info_cache = info
        return info

    def _podman_graph_root_and_driver(self) -> "tuple[Path, str] | None":
        """`(graphRoot, graphDriverName)`. Returns None on any failure;
        this and everything built on it reaches into storage internals,
        so every step degrades to "unknown, use the slow path" rather
        than raising.
        """
        info = self._podman_info()
        if info is None:
            return None  # already warned in _podman_info()
        try:
            store = info["store"]
            return (Path(store["graphRoot"]), store.get("graphDriverName", ""))
        except Exception as _e:
            print(
                f"[agsandbox_backend] WARNING: `podman info` output missing "
                f"store.graphRoot, fast squash path unavailable: {_e}",
                file=__import__("sys").stderr,
                flush=True,
            )
            return None

    def _is_rootless(self) -> bool:
        """True iff `podman info` reports `host.security.rootless` --
        the officially documented, supported way to detect this."""
        info = self._podman_info()
        if info is None:
            return False
        try:
            return bool(info["host"]["security"]["rootless"])
        except Exception:
            return False

    def _rootless_id_maps(self) -> "tuple[list, list] | None":
        """`(uid_map, gid_map)` entries from `podman info`'s
        `host.idMappings`, converted to the
        `(namespace_start, host_start, length)` tuples `_translate_id()`
        expects, cached for this backend's lifetime. None if unavailable.
        """
        cached = getattr(self, "_rootless_id_maps_cache", "unset")
        if cached != "unset":
            return cached
        result = None
        info = self._podman_info()
        if info is not None:
            try:
                mappings = info["host"]["idMappings"]

                def _convert(entries):
                    out = []
                    for e in entries or []:
                        out.append((int(e["container_id"]), int(e["host_id"]), int(e["size"])))
                    return out or None

                uid_map = _convert(mappings.get("uidmap"))
                gid_map = _convert(mappings.get("gidmap"))
                if uid_map is not None and gid_map is not None:
                    result = (uid_map, gid_map)
                else:
                    print(
                        "[agsandbox_backend] WARNING: rootless Podman confirmed but "
                        "host.idMappings has no usable uidmap/gidmap -- uid/gid "
                        "translation unavailable, fast squash path will fail lchown "
                        "checks for this backend's lifetime",
                        file=__import__("sys").stderr,
                        flush=True,
                    )
            except Exception as _e:
                print(
                    f"[agsandbox_backend] WARNING: could not parse `podman info`'s "
                    f"host.idMappings: {_e}",
                    file=__import__("sys").stderr,
                    flush=True,
                )
                result = None
        self._rootless_id_maps_cache = result
        return result

    def _host_to_container_id(self, uid: int, gid: int) -> "tuple[int, int]":
        """See `_ContainerBackendBase._host_to_container_id()`'s
        docstring for why this translation is needed at all. Only
        applies it when this Podman is confirmed rootless AND its
        idMappings were readable; otherwise identity, matching
        non-rootless Podman (where the overlay diff directory's
        on-disk ownership already IS what the container sees)."""
        if not self._is_rootless():
            return (uid, gid)
        maps = self._rootless_id_maps()
        if maps is None:
            return (uid, gid)
        uid_map, gid_map = maps
        return (_translate_id(uid, uid_map), _translate_id(gid, gid_map))

    def _locate_layer_diff_dir(
        self, diff_id: str, *, diff_ids: "list[str] | None" = None
    ) -> "Path | None":
        """Find the raw overlay diff directory backing *diff_id* directly
        on disk (`<graphRoot>/overlay/<layer-id>/diff/`), bypassing
        `podman diff`/`podman save` entirely -- see
        docs/agsandbox_backends/container.md's "Fast incremental
        squashing" section for the full rationale. Correlates via
        `<graphRoot>/overlay-layers/layers.json`: each entry's
        `diff-digest` matches what `podman inspect` reports in
        `RootFS.Layers`, and its `id` is the directory name under
        `overlay/`. (The storage layer `id` is NOT always equal to the
        digest hex -- confirmed empirically for `podman commit`-produced
        layers -- so the layers.json indirection is required, unlike a
        naive `overlay/<digest-hex>/diff` probe.)

        *diff_ids* is accepted for API parity with `_DockerBackend` (whose
        containerd-snapshotter path needs the full chain) and ignored
        here -- Podman keys layers by DiffID alone via layers.json.

        Returns None for a missing/unexpected layers.json entry or a
        non-overlay storage driver -- the base class's caller treats
        None as "the accumulator can't be trusted," never as an error.
        When multiple layers share a digest (e.g. empty layers), the
        last matching entry wins -- layers.json is append-ordered, so
        that prefers the newest.
        """
        root_and_driver = self._podman_graph_root_and_driver()
        if root_and_driver is None:
            return None  # already warned in _podman_info()/_podman_graph_root_and_driver()
        data_root, driver = root_and_driver
        if driver != "overlay":
            print(
                f"[agsandbox_backend] WARNING: unsupported podman storage driver "
                f"{driver!r} -- fast squash path unavailable for this backend's lifetime",
                file=__import__("sys").stderr,
                flush=True,
            )
            return None
        layers_json = data_root / "overlay-layers" / "layers.json"
        try:
            layers = json.loads(layers_json.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as _e:
            print(
                f"[agsandbox_backend] WARNING: could not read/parse {layers_json}: {_e}",
                file=__import__("sys").stderr,
                flush=True,
            )
            return None
        layer_id = None
        try:
            for entry in layers:
                if entry.get("diff-digest") == diff_id:
                    layer_id = entry.get("id")
        except (AttributeError, TypeError) as _e:
            print(
                f"[agsandbox_backend] WARNING: {layers_json} has an unexpected shape: {_e}",
                file=__import__("sys").stderr,
                flush=True,
            )
            return None
        if not layer_id:
            print(
                f"[agsandbox_backend] WARNING: no layers.json entry found for "
                f"diff_id {diff_id} in {layers_json} -- fast squash unavailable "
                f"for this cycle",
                file=__import__("sys").stderr,
                flush=True,
            )
            return None
        diff_dir = data_root / "overlay" / layer_id / "diff"
        if not diff_dir.is_dir():
            print(
                f"[agsandbox_backend] WARNING: layers.json entry for diff_id "
                f"{diff_id} points at {diff_dir}, which doesn't exist -- fast "
                f"squash unavailable for this cycle",
                file=__import__("sys").stderr,
                flush=True,
            )
            return None
        return diff_dir
