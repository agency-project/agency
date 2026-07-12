"""Unit and integration tests for the chroot sandbox backend and backend
selection logic in agsandbox_backend.py.

Tests that actually chroot are marked with @chroot and skipped automatically
when unprivileged user namespaces aren't usable on this host (see
agsandbox_backend.chroot_available())."""

from __future__ import annotations

import shutil
import uuid

import pytest
from unittest.mock import patch

from agency.agsandbox_backend import (
    agsandbox_backend,
    chroot_available,
    _ChrootBackend,
    _CHROOT_STATE_ROOT,
    _sanitize_tag,
)

chroot = pytest.mark.skipif(
    not chroot_available(), reason="unprivileged user namespaces not usable"
)


def _make_backend(**kwargs):
    name = f"chroot-test-{uuid.uuid4().hex[:8]}"
    defaults = dict(
        agname=name,
        name=name,
        checkpoint_image=None,
        mounts={},
        agconfig=None,
    )
    defaults.update(kwargs)
    return _ChrootBackend(defaults.pop("agname"), **defaults)


@pytest.fixture(autouse=True)
def _clean_state_root():
    shutil.rmtree(_CHROOT_STATE_ROOT, ignore_errors=True)
    yield
    shutil.rmtree(_CHROOT_STATE_ROOT, ignore_errors=True)


# ---------------------------------------------------------------------------
# _sanitize_tag
# ---------------------------------------------------------------------------


class TestSanitizeTag:
    def test_replaces_slash(self):
        assert "/" not in _sanitize_tag("agency/lifecycle-foo")

    def test_replaces_colon(self):
        assert ":" not in _sanitize_tag("agency/lifecycle-foo:v1")

    def test_distinct_tags_stay_distinct(self):
        assert _sanitize_tag("a/b") != _sanitize_tag("a-b")


# ---------------------------------------------------------------------------
# Backend selection (agsandbox_backend.for_config) -- no chroot/docker
# execution required, just routing logic.
# ---------------------------------------------------------------------------


class TestBackendSelection:
    def test_unknown_backend_raises_value_error(self):
        from agency.agconfig import agConfig
        from agency.agsandbox_backend import agSandboxBackendConfig

        cfg = agConfig(agSandboxBackendConfig(backend="not-a-real-backend"))
        with pytest.raises(ValueError, match="Unknown agsandbox_backend.backend"):
            agsandbox_backend.for_config(
                cfg,
                agname="a",
                name="a",
                checkpoint_image=None,
                base_image="x",
                mounts={},
            )

    def test_explicit_docker_raises_when_unusable(self):
        from agency.agconfig import agConfig
        from agency.agsandbox_backend import agSandboxBackendConfig

        cfg = agConfig(agSandboxBackendConfig(backend="docker"))
        with patch("agency.agsandbox_backend.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="docker"):
                agsandbox_backend.for_config(
                    cfg,
                    agname="a",
                    name="a",
                    checkpoint_image=None,
                    base_image="x",
                    mounts={},
                )

    def test_explicit_chroot_raises_when_unavailable(self):
        from agency.agconfig import agConfig
        from agency.agsandbox_backend import agSandboxBackendConfig

        cfg = agConfig(agSandboxBackendConfig(backend="chroot"))
        with patch("agency.agsandbox_backend.chroot_available", return_value=False):
            with pytest.raises(RuntimeError, match="chroot"):
                agsandbox_backend.for_config(
                    cfg,
                    agname="a",
                    name="a",
                    checkpoint_image=None,
                    base_image="x",
                    mounts={},
                )

    def test_explicit_chroot_builds_chroot_backend(self):
        from agency.agconfig import agConfig
        from agency.agsandbox_backend import agSandboxBackendConfig

        cfg = agConfig(agSandboxBackendConfig(backend="chroot"))
        with patch("agency.agsandbox_backend.chroot_available", return_value=True):
            backend = agsandbox_backend.for_config(
                cfg,
                agname="a",
                name="chroot-select-test",
                checkpoint_image=None,
                base_image="x",
                mounts={},
            )
        assert isinstance(backend, _ChrootBackend)
        backend.destroy()

    def test_auto_prefers_podman_then_docker_then_chroot(self):
        from agency.agsandbox_backend import _auto_detect_runtime

        with patch("agency.agsandbox_backend.get_container_runtime", return_value="podman"):
            assert _auto_detect_runtime() == "podman"

    def test_auto_falls_back_to_chroot_when_no_container_runtime(self):
        from agency.agsandbox_backend import _auto_detect_runtime

        with patch(
            "agency.agsandbox_backend.get_container_runtime",
            side_effect=RuntimeError("no docker/podman"),
        ):
            with patch("agency.agsandbox_backend.chroot_available", return_value=True):
                assert _auto_detect_runtime() == "chroot"

    def test_auto_raises_when_nothing_usable(self):
        from agency.agsandbox_backend import _auto_detect_runtime

        with patch(
            "agency.agsandbox_backend.get_container_runtime",
            side_effect=RuntimeError("no docker/podman"),
        ):
            with patch("agency.agsandbox_backend.chroot_available", return_value=False):
                with pytest.raises(RuntimeError, match="No usable sandbox backend"):
                    _auto_detect_runtime()


# ---------------------------------------------------------------------------
# _ChrootBackend -- functional (requires unprivileged userns + chroot)
# ---------------------------------------------------------------------------


@chroot
class TestChrootBackendExec:
    def test_exec_runs_and_returns_output(self):
        sb = _make_backend()
        try:
            out, rc = sb.exec("echo hello-chroot")
            assert rc == 0
            assert "hello-chroot" in out
        finally:
            sb.destroy()

    def test_exec_default_workdir_is_workspace(self):
        sb = _make_backend()
        try:
            out, rc = sb.exec("pwd")
            assert rc == 0
            assert out.strip() == "/workspace"
        finally:
            sb.destroy()

    def test_cannot_see_host_home_directory(self):
        """The jail must not expose the host's real filesystem outside the
        bind-mounted base dirs and workspace."""
        sb = _make_backend()
        try:
            out, rc = sb.exec("ls /home 2>&1; echo RC=$?")
            assert "No such file or directory" in out or "RC=2" in out
        finally:
            sb.destroy()

    def test_readonly_base_dir_rejects_writes(self):
        sb = _make_backend()
        try:
            out, rc = sb.exec("touch /bin/should-not-exist 2>&1; echo RC=$?")
            assert "RC=0" not in out or rc != 0
        finally:
            sb.destroy()

    def test_python_is_usable_inside_jail(self):
        sb = _make_backend()
        try:
            out, rc = sb.exec("python3 -c 'print(1+1)' 2>&1 || echo NO_PYTHON")
            assert "NO_PYTHON" not in out
        finally:
            sb.destroy()


@chroot
class TestChrootBackendFileIO:
    def test_write_then_read_file_round_trips(self):
        sb = _make_backend()
        try:
            sb.write_file("/workspace/hello.txt", "hello world\n")
            assert sb.read_file("/workspace/hello.txt") == "hello world\n"
        finally:
            sb.destroy()

    def test_write_file_bytes_round_trips(self):
        sb = _make_backend()
        try:
            data = bytes([0x00, 0x01, 0xFF, 0x10])
            sb.write_file_bytes("/workspace/bin.dat", data)
            assert sb.read_file_bytes("/workspace/bin.dat") == data
        finally:
            sb.destroy()

    def test_read_missing_file_raises_file_not_found(self):
        sb = _make_backend()
        try:
            with pytest.raises(FileNotFoundError):
                sb.read_file("/workspace/does-not-exist.txt")
        finally:
            sb.destroy()


@chroot
class TestChrootBackendLifecycle:
    def test_commit_then_new_backend_restores_content(self):
        tag = f"agency/lifecycle-test-{uuid.uuid4().hex[:8]}"
        sb1 = _make_backend()
        try:
            sb1.write_file("/workspace/marker.txt", "checkpoint\n")
            assert sb1.commit(tag) is True
        finally:
            sb1.destroy()

        sb2 = _make_backend(checkpoint_image=tag)
        try:
            assert sb2.read_file("/workspace/marker.txt") == "checkpoint\n"
        finally:
            sb2.destroy()
            _ChrootBackend.delete_image(tag, force=True)

    def test_commit_returns_false_when_never_started(self):
        sb = _make_backend()
        assert sb.commit("agency/never-started") is False
        sb.destroy()

    def test_stop_commit_false_discards_dirty_state(self):
        tag = f"agency/lifecycle-test-{uuid.uuid4().hex[:8]}"
        sb = _make_backend()
        try:
            sb.write_file("/workspace/good.txt", "good\n")
            sb.stop(commit=True)
            sb._ensure_started()
            sb.write_file("/workspace/dirty.txt", "dirty\n")
            sb.stop(commit=False)
            sb._ensure_started()
            content = sb.read_file("/workspace/good.txt")
            assert content == "good\n"
            with pytest.raises(FileNotFoundError):
                sb.read_file("/workspace/dirty.txt")
        finally:
            sb.destroy()
            _ChrootBackend.delete_image(sb._checkpoint_image or tag, force=True)

    def test_restore_materializes_snapshot(self):
        tag = f"agency/lifecycle-test-{uuid.uuid4().hex[:8]}"
        sb = _make_backend()
        try:
            sb.write_file("/workspace/a.txt", "aaa\n")
            sb.commit(tag)
            sb.write_file("/workspace/b.txt", "bbb\n")
            sb.restore(tag)
            assert sb.read_file("/workspace/a.txt") == "aaa\n"
            with pytest.raises(FileNotFoundError):
                sb.read_file("/workspace/b.txt")
        finally:
            sb.destroy()
            _ChrootBackend.delete_image(tag, force=True)

    def test_destroy_removes_jail_directory(self):
        sb = _make_backend()
        sb.exec("true")
        root = sb._root
        assert root.exists()
        sb.destroy()
        assert not root.exists()

    def test_cross_agent_workspace_isolation(self):
        sb1 = _make_backend()
        sb2 = _make_backend()
        try:
            sb1.write_file("/workspace/only-in-1.txt", "secret\n")
            out, rc = sb2.exec("ls /workspace")
            assert "only-in-1.txt" not in out
        finally:
            sb1.destroy()
            sb2.destroy()

    def test_update_limits_is_a_no_op(self):
        sb = _make_backend()
        try:
            sb.update_limits(cpus=2.0, memory="4g")  # must not raise
        finally:
            sb.destroy()


@chroot
class TestChrootImageHelpers:
    def test_tag_image_copies_snapshot(self):
        src_tag = f"agency/src-{uuid.uuid4().hex[:8]}"
        dest_tag = f"agency/dest-{uuid.uuid4().hex[:8]}"
        sb = _make_backend()
        try:
            sb.write_file("/workspace/x.txt", "x\n")
            sb.commit(src_tag)
            _ChrootBackend.tag_image(src_tag, dest_tag)
            sb2 = _make_backend(checkpoint_image=dest_tag)
            try:
                assert sb2.read_file("/workspace/x.txt") == "x\n"
            finally:
                sb2.destroy()
        finally:
            sb.destroy()
            _ChrootBackend.delete_image(src_tag, force=True)
            _ChrootBackend.delete_image(dest_tag, force=True)

    def test_export_then_import_round_trips(self):
        tag = f"agency/export-{uuid.uuid4().hex[:8]}"
        sb = _make_backend()
        try:
            sb.write_file("/workspace/e.txt", "exported\n")
            sb.commit(tag)
            blob = _ChrootBackend.export_image(tag, 30)
            assert isinstance(blob, bytes) and len(blob) > 0
            _ChrootBackend.delete_image(tag, force=True)
            _ChrootBackend.import_image(blob, 30)
            sb2 = _make_backend(checkpoint_image=tag)
            try:
                assert sb2.read_file("/workspace/e.txt") == "exported\n"
            finally:
                sb2.destroy()
        finally:
            sb.destroy()
            _ChrootBackend.delete_image(tag, force=True)

    def test_export_missing_tag_raises(self):
        with pytest.raises(FileNotFoundError):
            _ChrootBackend.export_image(f"agency/no-such-{uuid.uuid4().hex[:8]}", 30)


# ---------------------------------------------------------------------------
# Facade integration -- agSandbox(backend="chroot")
# ---------------------------------------------------------------------------


@chroot
class TestFacadeWithChrootBackend:
    def _make_sandbox(self, **kwargs):
        from agency.agconfig import agConfig
        from agency.agsandbox import agSandbox
        from agency.agsandbox_backend import agSandboxBackendConfig

        cfg = agConfig(agSandboxBackendConfig(backend="chroot"))
        uid = str(uuid.uuid4())
        return agSandbox(uid, agconfig=cfg, **kwargs)

    def test_facade_selects_chroot_backend(self):
        sb = self._make_sandbox()
        try:
            assert isinstance(sb._backend, _ChrootBackend)
        finally:
            sb.destroy()

    def test_facade_exec_and_file_io(self):
        sb = self._make_sandbox()
        try:
            out, rc = sb.exec("echo via-facade")
            assert rc == 0 and "via-facade" in out
            sb.write_file("/workspace/f.txt", "f-content\n")
            assert sb.read_file("/workspace/f.txt") == "f-content\n"
        finally:
            sb.destroy()

    def test_facade_stop_commit_true_sets_checkpoint_image(self):
        sb = self._make_sandbox()
        try:
            sb.exec("true")
            sb.stop(commit=True)
            assert sb._checkpoint_image is not None
        finally:
            sb.destroy()

    def test_facade_fork_preserves_checkpoint_content(self):
        sb = self._make_sandbox()
        fork_sb = None
        try:
            sb.write_file("/workspace/parent.txt", "parent-data\n")
            sb.stop(commit=True)
            fork_sb = sb.fork(str(uuid.uuid4()))
            assert isinstance(fork_sb._backend, _ChrootBackend)
            assert fork_sb.read_file("/workspace/parent.txt") == "parent-data\n"
        finally:
            sb.destroy()
            if fork_sb is not None:
                fork_sb.destroy()


# ---------------------------------------------------------------------------
# agent.py save()/load() wiring -- a checkpoint must round-trip through the
# same backend kind that produced it, not be silently assumed to be a
# docker/podman image tag.
# ---------------------------------------------------------------------------


@chroot
class TestAgentSaveLoadWithChrootBackend:
    def _make_agconfig(self):
        from agency.agconfig import agConfig
        from agency.agsandbox_backend import agSandboxBackendConfig

        return agConfig(
            agSandboxBackendConfig(backend="chroot"),
            {"agllm_backend": {"api_key": "k", "model": "m"}},
        )

    def test_save_records_chroot_image_kind(self, tmp_path):
        import json
        import tarfile
        from agency.agsandbox import agSandbox
        from agency.agent import agent
        from agency.agname import agname

        cfg = self._make_agconfig()
        ag = agent(agconfig=cfg)
        try:
            ag.sandbox = agSandbox(ag.agname, agconfig=cfg)
            ag.sandbox.write_file("/workspace/marker.txt", "data\n")
            ag.sandbox.stop(commit=True)

            ckpt = tmp_path / "agent.ckpt"
            ag.save(ckpt)

            with tarfile.open(ckpt, "r:gz") as tar:
                state = json.loads(tar.extractfile("state.json").read())
            assert state["sandbox_image_kind"] == "chroot"
        finally:
            if ag.sandbox is not None:
                ag.sandbox.destroy()
            agname._allocated.discard(str(ag.agname))

    def test_save_then_load_restores_chroot_workspace(self, tmp_path):
        from agency.agsandbox import agSandbox
        from agency.agent import agent
        from agency.agname import agname

        cfg = self._make_agconfig()
        ag = agent(agconfig=cfg)
        ag2 = None
        try:
            ag.sandbox = agSandbox(ag.agname, agconfig=cfg)
            ag.sandbox.write_file("/workspace/marker.txt", "checkpointed-via-agent\n")
            ag.sandbox.stop(commit=True)

            ckpt = tmp_path / "agent.ckpt"
            ag.save(ckpt)
            saved_agname = str(ag.agname)
            ag.sandbox.destroy()
            agname._allocated.discard(saved_agname)

            ag2 = agent.load(ckpt, agconfig=cfg)
            assert isinstance(ag2.sandbox._backend, _ChrootBackend)
            content = ag2.sandbox.read_file("/workspace/marker.txt")
            assert content == "checkpointed-via-agent\n"
        finally:
            if ag2 is not None and ag2.sandbox is not None:
                ag2.sandbox.destroy()
                agname._allocated.discard(str(ag2.agname))
