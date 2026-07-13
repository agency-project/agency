"""Unit tests for sandbox backend selection logic (agsandbox_backends/base.py):
agsandbox_backend.for_config() and _auto_detect_runtime(). No chroot/docker
execution required, just routing logic -- see test_docker.py/test_podman.py/
test_chroot.py for the concrete backends these route to.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from agency.agsandbox_backends import agsandbox_backend


class TestBackendSelection:
    def test_unknown_backend_raises_value_error(self):
        from agency.agconfig import agConfig
        from agency.agsandbox_backends import agSandboxBackendConfig

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
        from agency.agsandbox_backends import agSandboxBackendConfig

        cfg = agConfig(agSandboxBackendConfig(backend="docker"))
        with patch("agency.agsandbox_backends.base.shutil.which", return_value=None):
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
        from agency.agsandbox_backends import agSandboxBackendConfig

        cfg = agConfig(agSandboxBackendConfig(backend="chroot"))
        with patch("agency.agsandbox_backends.chroot.chroot_available", return_value=False):
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
        from agency.agsandbox_backends import agSandboxBackendConfig
        from agency.agsandbox_backends.chroot import _ChrootBackend

        cfg = agConfig(agSandboxBackendConfig(backend="chroot"))
        with patch("agency.agsandbox_backends.chroot.chroot_available", return_value=True):
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

    def test_explicit_docker_builds_docker_backend(self):
        from agency.agconfig import agConfig
        from agency.agsandbox_backends import agSandboxBackendConfig
        from agency.agsandbox_backends.docker import _DockerBackend

        cfg = agConfig(agSandboxBackendConfig(backend="docker"))
        with patch("agency.agsandbox_backends.container._runtime_works", return_value=True):
            with patch("agency.agsandbox_backends.base.shutil.which", return_value="/usr/bin/docker"):
                backend = agsandbox_backend.for_config(
                    cfg,
                    agname="a",
                    name="docker-select-test",
                    checkpoint_image=None,
                    base_image="x",
                    mounts={},
                )
        assert isinstance(backend, _DockerBackend)

    def test_explicit_podman_builds_podman_backend(self):
        from agency.agconfig import agConfig
        from agency.agsandbox_backends import agSandboxBackendConfig
        from agency.agsandbox_backends.podman import _PodmanBackend

        cfg = agConfig(agSandboxBackendConfig(backend="podman"))
        with patch("agency.agsandbox_backends.container._runtime_works", return_value=True):
            with patch("agency.agsandbox_backends.base.shutil.which", return_value="/usr/bin/podman"):
                backend = agsandbox_backend.for_config(
                    cfg,
                    agname="a",
                    name="podman-select-test",
                    checkpoint_image=None,
                    base_image="x",
                    mounts={},
                )
        assert isinstance(backend, _PodmanBackend)

    def test_auto_prefers_podman_then_docker_then_chroot(self):
        from agency.agsandbox_backends.base import _auto_detect_runtime

        with patch("agency.agsandbox_backends.container.get_container_runtime", return_value="podman"):
            assert _auto_detect_runtime() == "podman"

    def test_auto_falls_back_to_chroot_when_no_container_runtime(self):
        from agency.agsandbox_backends.base import _auto_detect_runtime

        with patch(
            "agency.agsandbox_backends.container.get_container_runtime",
            side_effect=RuntimeError("no docker/podman"),
        ):
            with patch("agency.agsandbox_backends.chroot.chroot_available", return_value=True):
                assert _auto_detect_runtime() == "chroot"

    def test_auto_raises_when_nothing_usable(self):
        from agency.agsandbox_backends.base import _auto_detect_runtime

        with patch(
            "agency.agsandbox_backends.container.get_container_runtime",
            side_effect=RuntimeError("no docker/podman"),
        ):
            with patch("agency.agsandbox_backends.chroot.chroot_available", return_value=False):
                with pytest.raises(RuntimeError, match="No usable sandbox backend"):
                    _auto_detect_runtime()


class TestBackendForImageKind:
    def test_container_kind_returns_container_backend_base(self):
        from agency.agsandbox_backends import backend_for_image_kind
        from agency.agsandbox_backends.container import _ContainerBackendBase

        assert backend_for_image_kind("container") is _ContainerBackendBase

    def test_chroot_kind_returns_chroot_backend(self):
        from agency.agsandbox_backends import backend_for_image_kind
        from agency.agsandbox_backends.chroot import _ChrootBackend

        assert backend_for_image_kind("chroot") is _ChrootBackend

    def test_unknown_kind_raises_value_error(self):
        from agency.agsandbox_backends import backend_for_image_kind

        with pytest.raises(ValueError, match="Unknown sandbox checkpoint image kind"):
            backend_for_image_kind("not-a-real-kind")
