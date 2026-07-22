"""Tests for _layer_squash.merge_layer_tars() -- the pure, no-docker-needed
tar-manipulation core of diff-only checkpoint squashing. Fixtures are
synthetic layer tars built directly via tarfile, not real docker/podman
output -- see test_docker.py's TestLayerSquashIntegration for a live-docker
correctness/layer-sharing/speed check against this same function.
"""

from __future__ import annotations

import tarfile
import io


import json
import os
import stat

from agency.agsandbox_backends._layer_squash import (
    merge_layer_tars,
    overlay_diff_to_tar,
    sha256_file,
    build_save_archive,
)


def _write_layer(tmp_path, name, entries):
    """entries: list of dicts, each either
    {"path": ..., "kind": "file", "content": bytes, "mode": int (opt)}
    {"path": ..., "kind": "dir", "mode": int (opt)}
    {"path": ..., "kind": "symlink", "target": str}
    {"path": ..., "kind": "whiteout"}                 -- shorthand for a .wh.<name> marker
    {"path": ..., "kind": "opaque"}                    -- shorthand for a <path>/.wh..wh..opq marker
    """
    p = tmp_path / name
    with tarfile.open(p, "w") as tf:
        for e in entries:
            kind = e["kind"]
            if kind == "whiteout":
                dirname, base = (
                    (e["path"].rsplit("/", 1) + [None])[:2] if "/" in e["path"] else ("", e["path"])
                )
                wh_name = f"{dirname}/.wh.{base}" if dirname else f".wh.{base}"
                ti = tarfile.TarInfo(name=wh_name)
                ti.size = 0
                tf.addfile(ti)
            elif kind == "opaque":
                opq_name = f"{e['path']}/.wh..wh..opq" if e["path"] else ".wh..wh..opq"
                ti = tarfile.TarInfo(name=opq_name)
                ti.size = 0
                tf.addfile(ti)
            elif kind == "dir":
                ti = tarfile.TarInfo(name=e["path"])
                ti.type = tarfile.DIRTYPE
                ti.mode = e.get("mode", 0o755)
                tf.addfile(ti)
            elif kind == "symlink":
                ti = tarfile.TarInfo(name=e["path"])
                ti.type = tarfile.SYMTYPE
                ti.linkname = e["target"]
                tf.addfile(ti)
            elif kind == "file":
                content = e["content"]
                ti = tarfile.TarInfo(name=e["path"])
                ti.size = len(content)
                ti.mode = e.get("mode", 0o644)
                tf.addfile(ti, io.BytesIO(content))
            else:
                raise ValueError(f"unknown kind {kind!r}")
    return p


def _read_output(output_path):
    """Return {path: ("file", content) | ("dir",) | ("symlink", target) |
    ("whiteout",) | ("opaque",)} for every entry in the merged tar."""
    result = {}
    with tarfile.open(output_path, "r") as tf:
        for member in tf.getmembers():
            name = member.name.strip("/")
            base = name.rsplit("/", 1)[-1]
            if base == ".wh..wh..opq":
                dirname = name.rsplit("/", 1)[0] if "/" in name else ""
                result.setdefault(dirname, ("dir",))
                result[f"__opaque__:{dirname}"] = ("opaque",)
            elif base.startswith(".wh."):
                dirname = name.rsplit("/", 1)[0] if "/" in name else ""
                real = base[len(".wh.") :]
                target = f"{dirname}/{real}" if dirname else real
                result[target] = ("whiteout",)
            elif member.isdir():
                result[name] = ("dir",)
            elif member.issym():
                result[name] = ("symlink", member.linkname)
            elif member.isfile():
                result[name] = ("file", tf.extractfile(member).read())
    return result


def test_basic_add_across_two_layers(tmp_path):
    l1 = _write_layer(tmp_path, "l1.tar", [{"path": "a", "kind": "file", "content": b"a-content"}])
    l2 = _write_layer(tmp_path, "l2.tar", [{"path": "b", "kind": "file", "content": b"b-content"}])
    out = tmp_path / "out.tar"
    merge_layer_tars([l1, l2], out)
    result = _read_output(out)
    assert result["a"] == ("file", b"a-content")
    assert result["b"] == ("file", b"b-content")


def test_later_layer_overwrites_earlier_file(tmp_path):
    l1 = _write_layer(tmp_path, "l1.tar", [{"path": "a", "kind": "file", "content": b"old"}])
    l2 = _write_layer(tmp_path, "l2.tar", [{"path": "a", "kind": "file", "content": b"new"}])
    out = tmp_path / "out.tar"
    merge_layer_tars([l1, l2], out)
    result = _read_output(out)
    assert result["a"] == ("file", b"new")


def test_whiteout_deletes_top_level_file(tmp_path):
    l1 = _write_layer(tmp_path, "l1.tar", [{"path": "a", "kind": "file", "content": b"x"}])
    l2 = _write_layer(tmp_path, "l2.tar", [{"path": "a", "kind": "whiteout"}])
    out = tmp_path / "out.tar"
    merge_layer_tars([l1, l2], out)
    result = _read_output(out)
    assert result["a"] == ("whiteout",)


def test_whiteout_deletes_nested_file(tmp_path):
    l1 = _write_layer(
        tmp_path,
        "l1.tar",
        [{"path": "dir", "kind": "dir"}, {"path": "dir/f", "kind": "file", "content": b"x"}],
    )
    l2 = _write_layer(tmp_path, "l2.tar", [{"path": "dir/f", "kind": "whiteout"}])
    out = tmp_path / "out.tar"
    merge_layer_tars([l1, l2], out)
    result = _read_output(out)
    assert result["dir/f"] == ("whiteout",)
    assert result["dir"] == ("dir",)  # the directory itself survives


def test_whiteout_of_entire_directory_removes_all_nested_content(tmp_path):
    """rm -rf on a directory: only ONE whiteout for the directory itself,
    not per-file whiteouts for its former contents (matches real docker
    behavior, confirmed empirically during development)."""
    l1 = _write_layer(
        tmp_path,
        "l1.tar",
        [
            {"path": "dir", "kind": "dir"},
            {"path": "dir/f1", "kind": "file", "content": b"x"},
            {"path": "dir/f2", "kind": "file", "content": b"y"},
        ],
    )
    l2 = _write_layer(tmp_path, "l2.tar", [{"path": "dir", "kind": "whiteout"}])
    out = tmp_path / "out.tar"
    merge_layer_tars([l1, l2], out)
    result = _read_output(out)
    assert result["dir"] == ("whiteout",)
    assert "dir/f1" not in result
    assert "dir/f2" not in result


def test_whiteout_then_readd_in_a_later_layer(tmp_path):
    """A path deleted then re-created in a LATER layer must end up as a
    live add, not a whiteout -- last-write-per-path wins."""
    l1 = _write_layer(tmp_path, "l1.tar", [{"path": "a", "kind": "file", "content": b"old"}])
    l2 = _write_layer(tmp_path, "l2.tar", [{"path": "a", "kind": "whiteout"}])
    l3 = _write_layer(tmp_path, "l3.tar", [{"path": "a", "kind": "file", "content": b"reborn"}])
    out = tmp_path / "out.tar"
    merge_layer_tars([l1, l2, l3], out)
    result = _read_output(out)
    assert result["a"] == ("file", b"reborn")


def test_opaque_directory_hides_earlier_content_but_keeps_new_siblings(tmp_path):
    l1 = _write_layer(
        tmp_path,
        "l1.tar",
        [{"path": "dir", "kind": "dir"}, {"path": "dir/old", "kind": "file", "content": b"old"}],
    )
    l2 = _write_layer(
        tmp_path,
        "l2.tar",
        [{"path": "dir", "kind": "opaque"}, {"path": "dir/new", "kind": "file", "content": b"new"}],
    )
    out = tmp_path / "out.tar"
    merge_layer_tars([l1, l2], out)
    result = _read_output(out)
    assert "dir/old" not in result
    assert result["dir/new"] == ("file", b"new")
    assert result["__opaque__:dir"] == ("opaque",)
    assert result["dir"] == ("dir",)


def test_opaque_and_add_in_same_layer_is_order_independent(tmp_path):
    """The opaque marker and the new sibling file live in the SAME layer;
    a single tar has no ordering semantics between its own entries, so the
    result must be identical regardless of which one tarfile happens to
    iterate first -- this is exactly why merge_layer_tars() processes each
    layer's whiteouts/opaque markers in a separate pass BEFORE its own
    adds, rather than in raw tar-member order."""
    l1 = _write_layer(
        tmp_path,
        "l1.tar",
        [{"path": "dir", "kind": "dir"}, {"path": "dir/old", "kind": "file", "content": b"old"}],
    )
    # Opaque marker written AFTER the new file in tar member order this time.
    l2 = _write_layer(
        tmp_path,
        "l2.tar",
        [{"path": "dir/new", "kind": "file", "content": b"new"}, {"path": "dir", "kind": "opaque"}],
    )
    out = tmp_path / "out.tar"
    merge_layer_tars([l1, l2], out)
    result = _read_output(out)
    assert "dir/old" not in result
    assert result["dir/new"] == ("file", b"new")


def test_symlink_preserved(tmp_path):
    l1 = _write_layer(
        tmp_path, "l1.tar", [{"path": "link", "kind": "symlink", "target": "/etc/hostname"}]
    )
    out = tmp_path / "out.tar"
    merge_layer_tars([l1], out)
    result = _read_output(out)
    assert result["link"] == ("symlink", "/etc/hostname")


def test_missing_parent_directory_is_synthesized(tmp_path):
    """A layer that adds a deeply nested path with no explicit
    intermediate directory entries must still produce a loadable tar with
    real directory entries for every ancestor."""
    l1 = _write_layer(
        tmp_path, "l1.tar", [{"path": "a/b/c/file", "kind": "file", "content": b"deep"}]
    )
    out = tmp_path / "out.tar"
    merge_layer_tars([l1], out)
    result = _read_output(out)
    assert result["a"] == ("dir",)
    assert result["a/b"] == ("dir",)
    assert result["a/b/c"] == ("dir",)
    assert result["a/b/c/file"] == ("file", b"deep")


def test_output_lists_parents_before_children(tmp_path):
    """A tar consumer that extracts sequentially should never see a child
    path before its parent directory has been created."""
    l1 = _write_layer(
        tmp_path,
        "l1.tar",
        [
            {"path": "a", "kind": "dir"},
            {"path": "a/b", "kind": "dir"},
            {"path": "a/b/file", "kind": "file", "content": b"x"},
        ],
    )
    out = tmp_path / "out.tar"
    merge_layer_tars([l1], out)
    with tarfile.open(out, "r") as tf:
        names = [m.name.strip("/") for m in tf.getmembers()]
    assert names.index("a") < names.index("a/b") < names.index("a/b/file")


def test_permissions_and_metadata_preserved(tmp_path):
    l1 = _write_layer(
        tmp_path,
        "l1.tar",
        [{"path": "exe", "kind": "file", "content": b"#!/bin/sh\n", "mode": 0o755}],
    )
    out = tmp_path / "out.tar"
    merge_layer_tars([l1], out)
    with tarfile.open(out, "r") as tf:
        member = tf.getmember("exe")
        assert member.mode == 0o755


class TestOverlayDiffToTar:
    """Tests for overlay_diff_to_tar() -- converts a raw overlay2 diff
    directory (kernel char-device whiteouts) into the portable `.wh.`
    tar convention merge_layer_tars() understands. Uses REAL char-device
    creation (major/minor 0,0), not mocks -- confirmed unprivileged
    mknod works for this specific device number in this environment."""

    def _mkwhiteout(self, path):
        os.mknod(str(path), mode=0o644 | stat.S_IFCHR, device=os.makedev(0, 0))

    def test_regular_files_and_dirs_preserved(self, tmp_path):
        diff_dir = tmp_path / "diff"
        (diff_dir / "a" / "b").mkdir(parents=True)
        (diff_dir / "a" / "b" / "f1").write_text("hello")
        out = tmp_path / "out.tar"
        overlay_diff_to_tar(diff_dir, out)
        result = _read_output(out)
        assert result["a"] == ("dir",)
        assert result["a/b"] == ("dir",)
        assert result["a/b/f1"] == ("file", b"hello")

    def test_char_device_whiteout_converted_to_wh_marker(self, tmp_path):
        diff_dir = tmp_path / "diff"
        (diff_dir / "existing").mkdir(parents=True)
        self._mkwhiteout(diff_dir / "existing" / "deleted_file")
        out = tmp_path / "out.tar"
        overlay_diff_to_tar(diff_dir, out)
        result = _read_output(out)
        assert result["existing/deleted_file"] == ("whiteout",)
        assert result["existing"] == ("dir",)

    def test_top_level_whiteout(self, tmp_path):
        diff_dir = tmp_path / "diff"
        diff_dir.mkdir()
        self._mkwhiteout(diff_dir / "gone")
        out = tmp_path / "out.tar"
        overlay_diff_to_tar(diff_dir, out)
        result = _read_output(out)
        assert result["gone"] == ("whiteout",)

    def test_symlink_preserved(self, tmp_path):
        diff_dir = tmp_path / "diff"
        diff_dir.mkdir()
        (diff_dir / "link").symlink_to("/etc/hostname")
        out = tmp_path / "out.tar"
        overlay_diff_to_tar(diff_dir, out)
        result = _read_output(out)
        assert result["link"] == ("symlink", "/etc/hostname")

    def test_output_feeds_directly_into_merge_layer_tars(self, tmp_path):
        """The whole point: this function's output must be a valid input
        layer for merge_layer_tars(), matching the same layer-tar
        contract every other layer (from a real docker save) follows."""
        diff_dir = tmp_path / "diff"
        (diff_dir / "workspace").mkdir(parents=True)
        (diff_dir / "workspace" / "new_file").write_text("cycle content")
        layer_tar = tmp_path / "layer.tar"
        overlay_diff_to_tar(diff_dir, layer_tar)

        merged = tmp_path / "merged.tar"
        merge_layer_tars([layer_tar], merged)
        result = _read_output(merged)
        assert result["workspace/new_file"] == ("file", b"cycle content")


class TestBuildSaveArchive:
    def test_manifest_references_base_by_digest_only_and_merged_layer_in_order(self, tmp_path):
        """Base layers must be referenced by digest alone -- their real
        content is deliberately never read into the archive (that's the
        entire point: squashing must stay cheap regardless of how large
        the base image is). Only the merged layer's actual bytes get
        written; base digests get zero-byte placeholders so Podman load
        accepts the archive."""
        base_digest1 = "sha256:" + "a" * 64
        base_digest2 = "sha256:" + "b" * 64
        merged = tmp_path / "merged"
        merged.write_bytes(b"merged-layer-content")
        merged_digest = sha256_file(merged)
        config_bytes = json.dumps({"fake": "config"}).encode()
        config_digest = sha256_file_bytes(config_bytes)

        out = tmp_path / "archive.tar"
        build_save_archive(
            out,
            base_layer_digests=[base_digest1, base_digest2],
            merged_blob_path=merged,
            merged_blob_digest=merged_digest,
            config_bytes=config_bytes,
            config_digest=config_digest,
            tag="my-tag:latest",
        )

        with tarfile.open(out, "r") as tf:
            names = [m.name for m in tf.getmembers()]
            manifest = json.loads(tf.extractfile("manifest.json").read())
            assert manifest[0]["RepoTags"] == ["my-tag:latest"]
            assert manifest[0]["Config"] == f"blobs/sha256/{config_digest}"
            assert manifest[0]["Layers"] == [
                f"blobs/sha256/{'a' * 64}",
                f"blobs/sha256/{'b' * 64}",
                f"blobs/sha256/{merged_digest}",
            ]
            # Base-layer members exist as zero-byte placeholders (Podman
            # requires the paths to be present); only the merged layer
            # carries real content.
            assert f"blobs/sha256/{'a' * 64}" in names
            assert f"blobs/sha256/{'b' * 64}" in names
            assert tf.getmember(f"blobs/sha256/{'a' * 64}").size == 0
            assert tf.getmember(f"blobs/sha256/{'b' * 64}").size == 0
            assert f"blobs/sha256/{merged_digest}" in names
            assert tf.extractfile(f"blobs/sha256/{merged_digest}").read() == b"merged-layer-content"
            for member in tf.getmembers():
                assert member.uid == 0
                assert member.gid == 0

    def test_duplicate_base_digests_share_one_placeholder(self, tmp_path):
        """Empty image layers commonly reuse one digest many times -- the
        archive must list them all in Layers order but only write one
        placeholder member."""
        empty = "sha256:" + "e" * 64
        other = "sha256:" + "f" * 64
        merged = tmp_path / "merged"
        merged.write_bytes(b"m")
        merged_digest = sha256_file(merged)
        config_bytes = json.dumps({}).encode()
        config_digest = sha256_file_bytes(config_bytes)

        out = tmp_path / "archive.tar"
        build_save_archive(
            out,
            base_layer_digests=[empty, other, empty, empty],
            merged_blob_path=merged,
            merged_blob_digest=merged_digest,
            config_bytes=config_bytes,
            config_digest=config_digest,
            tag="t:latest",
        )
        with tarfile.open(out, "r") as tf:
            names = [m.name for m in tf.getmembers()]
            assert names.count(f"blobs/sha256/{'e' * 64}") == 1
            assert names.count(f"blobs/sha256/{'f' * 64}") == 1
            manifest = json.loads(tf.extractfile("manifest.json").read())
            assert manifest[0]["Layers"] == [
                f"blobs/sha256/{'e' * 64}",
                f"blobs/sha256/{'f' * 64}",
                f"blobs/sha256/{'e' * 64}",
                f"blobs/sha256/{'e' * 64}",
                f"blobs/sha256/{merged_digest}",
            ]

    def test_bare_tag_gets_explicit_latest_qualifier(self, tmp_path):
        """`docker load` rejects a RepoTags entry with no tag component
        ("invalid tag") -- unlike `docker tag`, it does not default a bare
        repository name to `:latest` itself. Confirmed empirically."""
        merged = tmp_path / "merged"
        merged.write_bytes(b"merged")
        merged_digest = sha256_file(merged)
        config_bytes = json.dumps({}).encode()
        config_digest = sha256_file_bytes(config_bytes)

        out = tmp_path / "archive.tar"
        build_save_archive(
            out,
            base_layer_digests=["sha256:" + "c" * 64],
            merged_blob_path=merged,
            merged_blob_digest=merged_digest,
            config_bytes=config_bytes,
            config_digest=config_digest,
            tag="agency/lifecycle-my-sandbox",
        )
        with tarfile.open(out, "r") as tf:
            manifest = json.loads(tf.extractfile("manifest.json").read())
            assert manifest[0]["RepoTags"] == ["agency/lifecycle-my-sandbox:latest"]


def sha256_file_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()
