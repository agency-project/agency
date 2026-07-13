"""Unit and integration tests for the Docker-specific sandbox backend
(agency.agsandbox_backends.docker._DockerBackend): dangling-image cleanup on
commit, and the low-level docker-CLI command helpers (_rm_container, _rmi,
_ensure_started's pre-cleanup guard, destroy()'s semaphore release).

Tests that need a real Docker daemon are marked with @pytest.mark.docker and
skipped automatically when Docker is unreachable.
"""

from __future__ import annotations

import io
import subprocess
import sys
import threading
import uuid

import pytest
from unittest.mock import patch


def _docker_available() -> bool:
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


docker = pytest.mark.skipif(not _docker_available(), reason="Docker daemon not reachable")


def _make_sandbox(**kwargs):
    """Build an agSandbox forcing the docker backend -- these tests shell out
    to the ``docker`` CLI directly (or mock ``_DockerBackend._run``), so they
    need every sandbox to actually be a docker container regardless of the
    process-wide auto-detected default (which prefers podman when both are
    usable -- see agsandbox_backends.base.agsandbox_backend.for_config())."""
    from agency.agconfig import agConfig
    from agency.agsandbox import agSandbox
    from agency.agsandbox_backends import agSandboxBackendConfig

    uid = str(uuid.uuid4())
    agconfig = kwargs.pop("agconfig", None)
    cfg = agConfig(agSandboxBackendConfig(backend="docker"), agconfig)
    return agSandbox(uid, agconfig=cfg, **kwargs)


# ---------------------------------------------------------------------------
# Dangling-image eager cleanup (stop(commit=True))
# ---------------------------------------------------------------------------


class TestDanglingImageEagerCleanup:
    """Tests for the eager old-image deletion in stop(commit=True)."""

    def test_no_prune_thread(self):
        """No background agsandbox-prune thread should exist after the refactor."""
        named = [t for t in threading.enumerate() if t.name == "agsandbox-prune"]
        assert not named, "agsandbox-prune thread should have been removed"

    def test_stop_commit_deletes_old_image(self):
        """stop(commit=True) must delete the image that previously held the tag."""
        from agency.agsandbox_backends.docker import _DockerBackend

        sb = _make_sandbox()

        run_calls = []
        fake_old_id = "sha256:deadbeef0000"

        class FakeCompleted:
            def __init__(self, stdout=b"", returncode=0):
                self.stdout = stdout
                self.returncode = returncode

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            run_calls.append(args)
            if "inspect" in args:
                return FakeCompleted(stdout=fake_old_id.encode())
            if "commit" in args:
                return FakeCompleted()
            if "rm" in args:
                return FakeCompleted()
            if "rmi" in args:
                return FakeCompleted()
            return FakeCompleted()

        with patch.object(_DockerBackend, "_run", fake_run):
            with patch.object(sb._backend, "_started", True):
                with patch.object(sb._backend, "_container_running", return_value=True):
                    with patch.object(sb._backend, "_gpu_virtual", False):
                        sb.stop(commit=True)

        rmi_calls = [a for a in run_calls if "rmi" in a]
        assert rmi_calls, "expected docker rmi call for old image"
        assert any(fake_old_id in " ".join(a) for a in rmi_calls), (
            f"rmi call did not reference old image ID; calls: {rmi_calls}"
        )

    def test_stop_commit_skips_rmi_when_no_old_image(self):
        """If the tag does not exist yet (first commit), no rmi call is made."""
        import agency.agsandbox_backends.docker as _mod

        sb = _make_sandbox()

        class FakeCompleted:
            def __init__(self, stdout=b"", returncode=0):
                self.stdout = stdout
                self.returncode = returncode

        run_calls = []

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            run_calls.append(args)
            if "inspect" in args:
                return FakeCompleted(stdout=b"", returncode=1)  # tag not found
            return FakeCompleted()

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with patch.object(sb._backend, "_started", True):
                with patch.object(sb._backend, "_container_running", return_value=True):
                    with patch.object(sb._backend, "_gpu_virtual", False):
                        sb.stop(commit=True)

        rmi_calls = [a for a in run_calls if "rmi" in a]
        assert not rmi_calls, "must not call rmi when there was no previous image"

    def test_stop_commit_rmi_failure_is_best_effort(self):
        """A failing rmi during old-image cleanup must NOT propagate.

        Per Design_sandbox_lifecycle.md's "Dangling image accumulation and
        eager cleanup" section, this rmi is best-effort: a race with another
        agent's inspect/rmi (or a fork still using the image) is expected and
        should leave a dangling image rather than crash stop() — which runs
        after every tool call, so a hard failure here would be far worse than
        the disk-space cost of an occasional dangling image."""
        import agency.agsandbox_backends.docker as _mod

        sb = _make_sandbox()
        fake_old_id = "sha256:cafebabe1234"

        class FakeCompleted:
            def __init__(self, stdout=b"", returncode=0):
                self.stdout = stdout
                self.returncode = returncode

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "inspect" in args:
                return FakeCompleted(stdout=fake_old_id.encode())
            if "rmi" in args:
                raise RuntimeError("image in use")
            return FakeCompleted()

        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        try:
            with patch.object(_mod._DockerBackend, "_run", fake_run):
                with patch.object(sb._backend, "_started", True):
                    with patch.object(sb._backend, "_container_running", return_value=True):
                        with patch.object(sb._backend, "_gpu_virtual", False):
                            sb.stop(commit=True)  # must not raise
        finally:
            sys.stderr = old_stderr

        assert "WARNING" in captured.getvalue()
        assert fake_old_id in captured.getvalue()

    @docker
    def test_repeated_commits_leave_no_dangling_images(self):
        """stop(commit=True) called 3 times to the same tag must leave 0 new dangling images."""
        name = f"test-eager-{uuid.uuid4().hex[:8]}"
        tag = f"agency/lifecycle-{name}"

        def _dangling_ids():
            r = subprocess.run(
                ["docker", "images", "-f", "dangling=true", "-q"],
                capture_output=True,
                text=True,
            )
            return set(ln.strip() for ln in r.stdout.splitlines() if ln.strip())

        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "agency-sandbox:latest",
                "tail",
                "-f",
                "/dev/null",
            ],
            capture_output=True,
            check=True,
        )
        try:
            before = _dangling_ids()
            for _ in range(3):
                old_id_r = subprocess.run(
                    ["docker", "inspect", "--format={{.Id}}", tag],
                    capture_output=True,
                    text=True,
                )
                old_id = old_id_r.stdout.strip() if old_id_r.returncode == 0 else None
                subprocess.run(["docker", "commit", name, tag], capture_output=True, check=True)
                if old_id:
                    subprocess.run(["docker", "rmi", old_id], capture_output=True)
            after = _dangling_ids()
            new_dangling = after - before
            assert len(new_dangling) == 0, (
                f"expected 0 new dangling images with eager cleanup, got {len(new_dangling)}"
            )
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


# ---------------------------------------------------------------------------
# _rm_container / _rmi helpers
# ---------------------------------------------------------------------------


class TestDockerCommandHelpers:
    """Unit tests for _rm_container and _rmi — no real Docker required."""

    def _sb(self):
        return _make_sandbox()

    # --- _rm_container ---

    def test_rm_container_sends_rm_force_args(self):
        sb = self._sb()
        calls = []

        class OK:
            returncode = 0
            stdout = b""

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            calls.append((args, check))
            return OK()

        from agency.agsandbox_backends.docker import _DockerBackend

        with patch.object(_DockerBackend, "_run", fake_run):
            sb._backend._rm_container("my-container")

        assert len(calls) == 1
        args, check = calls[0]
        assert "rm" in args and "-f" in args and "my-container" in args
        assert check is True

    def test_rm_container_raises_on_failure(self):
        sb = self._sb()

        from agency.agsandbox_backends.docker import _DockerBackend

        with patch.object(_DockerBackend, "_run", side_effect=RuntimeError("rm failed")):
            with pytest.raises(RuntimeError, match="rm failed"):
                sb._backend._rm_container("bad-container")

    # --- _rmi ---

    def test_rmi_sends_rmi_args_without_force(self):
        sb = self._sb()
        calls = []

        class OK:
            returncode = 0
            stdout = b""

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            calls.append((args, check))
            return OK()

        import agency.agsandbox_backends.docker as _mod

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            sb._backend._rmi("sha256:abc123")

        assert len(calls) == 1
        args, check = calls[0]
        assert "rmi" in args and "sha256:abc123" in args
        assert "-f" not in args
        assert check is True

    def test_rmi_force_adds_dash_f(self):
        sb = self._sb()
        calls = []

        class OK:
            returncode = 0
            stdout = b""

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            calls.append(args)
            return OK()

        import agency.agsandbox_backends.docker as _mod

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            sb._backend._rmi("myimage:tag", force=True)

        assert "-f" in calls[0]

    def test_rmi_raises_on_failure(self):
        sb = self._sb()

        import agency.agsandbox_backends.docker as _mod

        with patch.object(_mod._DockerBackend, "_run", side_effect=RuntimeError("rmi failed")):
            with pytest.raises(RuntimeError, match="rmi failed"):
                sb._backend._rmi("sha256:deadbeef")

    # --- _ensure_started pre-cleanup guard ---

    def test_ensure_started_skips_rm_when_no_leftover_container(self):
        """No rm when the container doesn't exist before create."""
        sb = self._sb()
        calls = []

        class OK:
            returncode = 0
            stdout = b""

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            calls.append(args)
            return OK()

        import agency.agsandbox_backends.docker as _mod

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            # status returns "" → no leftover container
            with patch.object(sb._backend, "_container_running", return_value=False):
                with patch.object(sb._backend, "_container_status", return_value=""):
                    with patch.object(sb._backend, "_run_with_conflict_retry"):
                        sb._backend._ensure_started()

        rm_calls = [a for a in calls if "rm" in a]
        assert not rm_calls, f"expected no rm call; got {rm_calls}"

    def test_ensure_started_rms_leftover_container(self):
        """rm is issued when a non-running leftover container exists."""
        sb = self._sb()
        calls = []

        class OK:
            returncode = 0
            stdout = b""

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            calls.append((args, check))
            return OK()

        import agency.agsandbox_backends.docker as _mod

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with patch.object(sb._backend, "_container_running", return_value=False):
                with patch.object(sb._backend, "_container_status", return_value="exited"):
                    with patch.object(sb._backend, "_run_with_conflict_retry"):
                        sb._backend._ensure_started()

        rm_calls = [(a, c) for (a, c) in calls if "rm" in a]
        assert rm_calls, "expected rm call for leftover container"
        assert all(c is True for _, c in rm_calls), "rm must use check=True"

    # --- destroy semaphore release ---

    def test_destroy_releases_semaphore_even_when_rm_raises(self):
        """The docker-only concurrency semaphore must be released in finally
        even if rm fails."""
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()

        class OK:
            returncode = 0
            stdout = b""

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "rm" in args:
                raise RuntimeError("rm exploded")
            if "images" in args:
                result = OK()
                result.stdout = b""
                return result
            return OK()

        released = []

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with patch.object(sb._backend, "_started", True):
                with patch.object(sb._backend, "_container_running", return_value=True):
                    with patch.object(sb._backend, "_container_status", return_value="running"):
                        with patch.object(
                            _mod._container_semaphore,
                            "release",
                            side_effect=lambda: released.append(1),
                        ):
                            with pytest.raises(RuntimeError, match="rm exploded"):
                                sb.destroy()

        assert released, "semaphore must be released even when rm raises"

    def test_destroy_skips_rm_when_container_absent(self):
        """destroy() must not call rm when the container does not exist."""
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        calls = []

        class OK:
            returncode = 0
            stdout = b""

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            calls.append(args)
            return OK()

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with patch.object(sb._backend, "_started", False):
                with patch.object(sb._backend, "_container_running", return_value=False):
                    with patch.object(sb._backend, "_container_status", return_value=""):
                        sb.destroy()

        rm_calls = [a for a in calls if "rm" in a and "rmi" not in a]
        assert not rm_calls, f"must not rm when container absent; got {rm_calls}"
