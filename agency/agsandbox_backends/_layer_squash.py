"""Diff-only layer squashing: merge a run of Docker/OCI image layer diffs
into a single layer, without touching whatever sits below them.

`_ContainerBackendBase._squash_commit()` (container.py) used to flatten a
checkpoint's ENTIRE filesystem via `docker export`/`docker import` -- fast to
implement, but its cost is proportional to the whole merged filesystem
(base image + every accumulated diff), not just the diff. For a large base
image (the real `agency-sandbox:latest` is ~24GB), that made every squash
take tens of seconds regardless of how small the actual workspace change
was, confirmed live: forcing a squash at every skill exit pushed two
concurrent agents' teardown past a 120s test timeout.

This module operates one level lower, at the OCI layer-tar level instead of
the merged-filesystem level: each layer in a Docker/OCI image is already
stored as its own small diff tarball (that's what a "layer" *is*). Merging
just the layers ABOVE a known base image into one, while leaving the base
image's own layers completely untouched (same digests, no re-serialization,
genuinely shared on disk), is exactly what a normal `docker commit` already
does for a single increment -- this module generalizes that to N
accumulated increments at once.

Layer diff format (verified empirically against a real `docker save`
output during development, not assumed from the spec alone):
  - A deleted path `<dir>/<name>` is represented by a zero-byte regular
    file entry named `<dir>/.wh.<name>` in the layer that deleted it
    (top-level: `.wh.<name>`, no leading dir).
  - A directory whose lower content should be entirely hidden (rare in
    practice for plain container diffs, but part of the spec) is marked
    by a zero-byte regular file `<dir>/.wh..wh..opq` inside it.
  - Everything else is a normal tar entry (regular file, directory,
    symlink, hardlink, device, fifo) representing an added/modified path.

Known simplification: a hardlink entry's `linkname` references another
path *within the same tar*, by position -- if that target path itself gets
whited-out or overwritten by a later layer while the hardlink survives
unchanged, the merged output could carry a hardlink pointing at a path
that no longer resolves the same way. Not handled explicitly (would need
tracking hardlink target identity independent of path); acceptable given
hardlinks are rare in typical sandbox workspace usage (regular file
writes, not deliberate hardlinking), and this is a diff-flattening
optimization, not a general-purpose image tool.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tarfile
from contextlib import ExitStack
from pathlib import Path
from typing import Callable

_WHITEOUT_PREFIX = ".wh."
_OPAQUE_MARKER = ".wh..wh..opq"


def _normalize(name: str) -> str:
    """Strip a leading "./" and any leading/trailing slashes so the same
    logical path is always keyed identically regardless of which layer
    (or tar tool) produced the entry."""
    name = name.strip("/")
    if name.startswith("./"):
        name = name[2:]
    return name


def _split(path: str) -> "tuple[str, str]":
    """(dirname, basename), with dirname == "" for a top-level path."""
    if "/" in path:
        dirname, basename = path.rsplit("/", 1)
        return dirname, basename
    return "", path


def _is_descendant(path: str, ancestor: str) -> bool:
    if not ancestor:
        return True  # every path is a "descendant" of the root
    return path == ancestor or path.startswith(ancestor + "/")


def merge_layer_tars(layer_paths: "list[Path]", output_path: Path) -> None:
    """Merge *layer_paths* (bottom-to-top diff layers, each a docker/OCI
    layer tar) into one tar at *output_path* implementing the same
    cumulative filesystem diff.

    Each layer is processed as one atomic unit relative to the state
    accumulated from EARLIER layers: first its own whiteouts/opaque
    markers are applied (purging whatever they target from the
    accumulated state), THEN its own regular add entries are merged in.
    Splitting into these two passes per layer -- rather than one pass in
    tar-member order -- means a layer that both makes a directory opaque
    AND adds new files under that same directory behaves correctly
    regardless of which order those entries happen to appear in that
    layer's own tar (a single layer's entries have no ordering semantics
    relative to each other; they're all "current as of this layer").
    """
    final: "dict[str, tuple]" = {}  # path -> ("add", layer_idx, TarInfo) | ("whiteout",)
    opaque_dirs: "set[str]" = set()

    with ExitStack() as stack:
        tars = [stack.enter_context(tarfile.open(p, "r")) for p in layer_paths]

        for idx, tf in enumerate(tars):
            members = tf.getmembers()

            # Pass 1: collect this layer's own whiteouts/opaque markers and
            # apply their purges against the state accumulated so far.
            whiteout_targets: "list[str]" = []
            opaque_targets: "list[str]" = []
            for member in members:
                name = _normalize(member.name)
                if not name:
                    continue
                dirname, basename = _split(name)
                if basename == _OPAQUE_MARKER:
                    opaque_targets.append(dirname)
                elif basename.startswith(_WHITEOUT_PREFIX):
                    real_name = basename[len(_WHITEOUT_PREFIX) :]
                    target = f"{dirname}/{real_name}" if dirname else real_name
                    whiteout_targets.append(target)

            for target in whiteout_targets:
                for existing in [p for p in final if _is_descendant(p, target)]:
                    del final[existing]
                opaque_dirs.difference_update({d for d in opaque_dirs if _is_descendant(d, target)})
                final[target] = ("whiteout",)

            for target_dir in opaque_targets:
                for existing in [
                    p for p in final if p != target_dir and _is_descendant(p, target_dir)
                ]:
                    del final[existing]
                opaque_dirs.add(target_dir)

            # Pass 2: this layer's own regular add entries -- applied AFTER
            # its own purges above, so they always survive regardless of
            # intra-layer ordering.
            for member in members:
                name = _normalize(member.name)
                if not name:
                    continue
                _, basename = _split(name)
                if basename == _OPAQUE_MARKER or basename.startswith(_WHITEOUT_PREFIX):
                    continue
                final[name] = ("add", idx, member)
                opaque_dirs.discard(name)  # replaced by a fresh add; stale opaque marker drops

        _synthesize_missing_parent_dirs(final)

        ordered_paths = sorted(final.keys(), key=lambda p: tuple(p.split("/")))

        with tarfile.open(output_path, "w") as out:
            for path in ordered_paths:
                entry = final[path]
                if entry[0] == "add":
                    _, idx, member = entry
                    if member.isfile():
                        stream = tars[idx].extractfile(member)
                        out.addfile(member, stream)
                    else:
                        out.addfile(member)
                elif entry[0] == "whiteout":
                    dirname, basename = _split(path)
                    wh_name = (
                        f"{dirname}/{_WHITEOUT_PREFIX}{basename}"
                        if dirname
                        else f"{_WHITEOUT_PREFIX}{basename}"
                    )
                    out.addfile(_zero_byte_regular(wh_name))
                if path in opaque_dirs:
                    out.addfile(
                        _zero_byte_regular(f"{path}/{_OPAQUE_MARKER}" if path else _OPAQUE_MARKER)
                    )


def _zero_byte_regular(name: str) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name=name)
    ti.size = 0
    ti.mode = 0o644
    ti.type = tarfile.REGTYPE
    ti.uid = 0
    ti.gid = 0
    return ti


def _synthesize_missing_parent_dirs(final: "dict[str, tuple]") -> None:
    """Every add/opaque entry's parent directory chain must have SOME
    entry in *final* -- most real diffs already include explicit
    directory entries for anything they touch (overlayfs materializes
    them physically), but this defends against a layer that adds a
    nested path without an explicit intermediate directory entry, rather
    than depending on the eventual `docker load`'s own extraction being
    lenient about it."""
    needed: "set[str]" = set()
    for path, entry in list(final.items()):
        if entry[0] != "add":
            continue
        dirname, _ = _split(path)
        while dirname and dirname not in final and dirname not in needed:
            needed.add(dirname)
            dirname, _ = _split(dirname)
    for dirname in needed:
        ti = tarfile.TarInfo(name=dirname)
        ti.type = tarfile.DIRTYPE
        ti.mode = 0o755
        ti.uid = 0
        ti.gid = 0
        final[dirname] = ("add", None, ti)


def overlay_diff_to_tar(
    diff_dir: Path,
    output_path: Path,
    *,
    uid_gid_translate: "Callable[[int, int], tuple[int, int]] | None" = None,
) -> None:
    """Convert a raw overlay2 diff directory (as `docker commit` produces
    internally, before it's ever wrapped into a `docker save`-portable
    layer) into a standard interchange-format layer tar: files/dirs/
    symlinks preserved as-is, and the KERNEL overlayfs whiteout convention
    (a character-special device with major/minor 0,0, replacing whatever
    used to be at that path) converted to the `.wh.<name>` convention
    `merge_layer_tars()` already understands. These are two different
    conventions for the same concept -- confirmed to differ empirically
    during development (a real overlay2 diff directory for a deleted path
    contains a `mknod 0 0` character device there, not a `.wh.`-prefixed
    file -- that naming convention is specific to the portable tar
    interchange format `docker save`/`load` use, not the kernel's own
    on-disk layout).

    *uid_gid_translate*, if given, is called as `(host_uid, host_gid) ->
    (container_uid, container_gid)` for every entry. Needed under
    user-namespace-remapped runtimes (e.g. rootless Docker): the files in
    this directory are only visible to the host process through its own
    (remapped) user namespace, so `os.lstat()` reports HOST-side
    ownership, not the ownership the container itself sees -- confirmed
    empirically during development (a root-owned file inside the
    container showed up as owned by the invoking host user here, not
    uid 0). Loading a layer tar with un-translated host-side ownership
    can fail outright ("failed to Lchown ... invalid argument") since
    those host uids/gids are typically outside any range the runtime is
    willing to write during load. None (the default) applies no
    translation, correct for any runtime that doesn't remap ownership at
    all (e.g. non-rootless Docker/Podman).
    """
    diff_dir = Path(diff_dir)
    with tarfile.open(output_path, "w") as out:
        for root, dirs, files in os.walk(diff_dir):
            rel_root = os.path.relpath(root, diff_dir)
            for name in list(dirs) + list(files):
                full_path = os.path.join(root, name)
                rel_path = name if rel_root == "." else f"{rel_root}/{name}"
                st = os.lstat(full_path)
                if (
                    stat.S_ISCHR(st.st_mode)
                    and os.major(st.st_rdev) == 0
                    and os.minor(st.st_rdev) == 0
                ):
                    dirname, base = _split(rel_path)
                    wh_name = (
                        f"{dirname}/{_WHITEOUT_PREFIX}{base}"
                        if dirname
                        else f"{_WHITEOUT_PREFIX}{base}"
                    )
                    out.addfile(_zero_byte_regular(wh_name))
                    continue
                ti = out.gettarinfo(full_path, arcname=rel_path)
                if uid_gid_translate is not None:
                    ti.uid, ti.gid = uid_gid_translate(ti.uid, ti.gid)
                if ti.isfile():
                    with open(full_path, "rb") as f:
                        out.addfile(ti, f)
                else:
                    out.addfile(ti)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _add_bytes(out: tarfile.TarFile, arcname: str, data: bytes) -> None:
    ti = tarfile.TarInfo(name=arcname)
    ti.size = len(data)
    ti.uid = 0
    ti.gid = 0
    ti.uname = ""
    ti.gname = ""
    out.addfile(ti, io.BytesIO(data))


def _add_file(out: tarfile.TarFile, arcname: str, path: Path) -> None:
    # gettarinfo() reads the file's real (host) uid/gid/mtime -- must be
    # overridden to 0/root, same reason as _add_bytes(): `docker load`
    # under rootless Docker rejects an archive whose wrapper-level entries
    # (as opposed to the CONTENT *inside* a layer blob, which is a nested,
    # separate tar this code never edits) aren't owned by root, failing
    # with an opaque "failed to Lchown ... invalid argument" -- confirmed
    # empirically during development; this is the fix.
    ti = out.gettarinfo(str(path), arcname=arcname)
    ti.uid = 0
    ti.gid = 0
    ti.uname = ""
    ti.gname = ""
    with open(path, "rb") as f:
        out.addfile(ti, f)


def build_save_archive(
    output_path: Path,
    *,
    base_layer_digests: "list[str]",
    merged_blob_path: Path,
    merged_blob_digest: str,
    config_bytes: bytes,
    config_digest: str,
    tag: str,
) -> None:
    """Construct a `docker save`/`load`-compatible archive referencing the
    base image's own layers BY DIGEST ONLY -- their real blob content is
    deliberately never read or included. Both `docker load` and
    `podman load` resolve those digests from their own local store
    (confirmed empirically, including against a real ~24GB, 80-layer
    base image: archive-build and load both completed in well under a
    second, with zero bytes of base-layer content touched). This is what
    makes squashing cheap regardless of base image size -- earlier
    revisions of this function copied the base blobs' bytes into the
    archive, which defeated the entire point.

    Podman (unlike Docker) still requires every `manifest.json` Layers
    path to exist as a member of the archive even when it will resolve
    the content locally -- missing members fail with "Some layer
    tarfiles are missing in the tarball". So each unique base-layer
    digest gets a zero-byte placeholder blob written under
    `blobs/sha256/<hex>` (empty, never the real layer bytes). Docker
    tolerates the same placeholders. Duplicate digests (common for
    empty image layers) share one placeholder member.

    *base_layer_digests* are `sha256:<hex>`-prefixed strings (e.g. from
    `docker inspect --format='{{json .RootFS.Layers}}'`), most-base-first,
    matching the target image's OWN layer order exactly -- NOT file paths.

    *tag* is qualified with an explicit `:latest` if it carries no tag
    component of its own -- unlike `docker tag`/`docker commit`, `docker
    load` does not default a bare repository name to `:latest`; it
    rejects `manifest.json` RepoTags entries without one ("invalid tag"),
    confirmed empirically during development.
    """
    if ":" not in tag.rsplit("/", 1)[-1]:
        tag = f"{tag}:latest"
    base_blob_names = [f"blobs/sha256/{d.split(':', 1)[1]}" for d in base_layer_digests]
    manifest = [
        {
            "Config": f"blobs/sha256/{config_digest}",
            "RepoTags": [tag],
            "Layers": base_blob_names + [f"blobs/sha256/{merged_blob_digest}"],
        }
    ]
    manifest_bytes = json.dumps(manifest).encode()

    with tarfile.open(output_path, "w") as out:
        _add_bytes(out, "manifest.json", manifest_bytes)
        _add_bytes(out, f"blobs/sha256/{config_digest}", config_bytes)
        # Zero-byte placeholders so Podman load accepts the archive; real
        # base-layer content is never read -- see docstring.
        for blob_name in dict.fromkeys(base_blob_names):
            _add_bytes(out, blob_name, b"")
        _add_file(out, f"blobs/sha256/{merged_blob_digest}", merged_blob_path)


__all__ = ["merge_layer_tars", "overlay_diff_to_tar", "sha256_file", "build_save_archive"]
