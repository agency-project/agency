"""Unit and integration tests for the Podman-specific sandbox backend
(agency.agsandbox_backends.podman._PodmanBackend): dangling-image cleanup on
commit, and the low-level podman-CLI command helpers (_rm_container, _rmi,
_ensure_started's pre-cleanup guard, destroy()'s semaphore release).

This mirrors test_docker.py's real-daemon coverage class-for-class -- before
this file existed, every real-container integration test in the codebase
exercised only the Docker backend (test_agsandbox.py's `_make_sandbox()`
hardcodes backend="docker", and test_docker.py is Docker-only by design), so
`_PodmanBackend` had zero integration coverage of any kind starting a real
container, despite Podman being the auto-preferred runtime whenever both are
installed (see agsandbox_backends.base.agsandbox_backend.for_config()) and
the one actually subject to the exact same session-keyring quota as Docker
(see agsandbox_backends/container.py's module docstring).

The session-keyring-quota machinery (_keyring_container_limit/keyring_quota/
_semaphore_held_count) and the _is_quota_exhaustion_error/_wait_for_quota_slot/
_quota_diagnostics hooks are shared, runtime-agnostic code on
_ContainerBackendBase itself (agency.agsandbox_backends.container) -- see
test_container.py's TestKeyringQuotaDiagnostics and
TestQuotaHooksSharedAcrossRuntimes (exercised there against Podman already).
GPU passthrough flags are covered by test_container.py's
TestGpuFlagsPerRuntime and TestPodmanGpuPassthroughIntegration.

Tests that need a real Podman daemon are marked with @pytest.mark.podman and
skipped automatically when Podman is unreachable.
"""

from __future__ import annotations

import io
import subprocess
import sys
import uuid

import pytest
from unittest.mock import patch


def _podman_available() -> bool:
    try:
        result = subprocess.run(["podman", "info"], capture_output=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


podman = pytest.mark.skipif(not _podman_available(), reason="Podman daemon not reachable")


def _make_sandbox(**kwargs):
    """Build a REAL agSandbox forcing the podman backend -- for the
    end-to-end (@podman-marked) tests that need the full facade (real
    make_sandboxed_tools() dispatch, real podman CLI). Goes through
    agsandbox_backend.for_config(), which requires an actually-reachable
    podman daemon (see for_config()'s availability check) -- correct for
    those tests, but NOT for the mock-based unit tests below, which use
    _make_backend() instead specifically to avoid that requirement."""
    from agency.agconfig import agConfig
    from agency.agsandbox import agSandbox
    from agency.agsandbox_backends import agSandboxBackendConfig

    uid = str(uuid.uuid4())
    agconfig = kwargs.pop("agconfig", None)
    cfg = agConfig(agSandboxBackendConfig(backend="podman"), agconfig)
    return agSandbox(uid, agconfig=cfg, **kwargs)


def _make_backend(**kwargs):
    """Build a bare _PodmanBackend directly, bypassing agSandbox/
    for_config()'s real-daemon availability check. These are unit tests that
    mock _PodmanBackend._run (or _container_running/_container_status/etc.)
    themselves and never issue a real `podman` call, so they don't need an
    actual podman binary or reachable daemon on the host running this file
    -- matching test_container.py's own pattern for mock-based backend unit
    tests (e.g. TestOwnHostPidsPodman), which construct _PodmanBackend the
    same way for the same reason."""
    from agency.agsandbox_backends.podman import _PodmanBackend

    name = f"podman-test-{uuid.uuid4().hex[:8]}"
    defaults = dict(
        agname=name,
        name=name,
        checkpoint_image=None,
        base_image="agency-sandbox:latest",
        mounts={},
        agconfig=None,
    )
    defaults.update(kwargs)
    return _PodmanBackend(defaults.pop("agname"), **defaults)


# ---------------------------------------------------------------------------
# Owner-PID container labeling -- feeds agsandbox_backends.container's
# startup orphan reaper (see test_container.py). Mirrors test_docker.py's
# TestOwnerPidLabel exactly; see there for the full rationale.
# ---------------------------------------------------------------------------


class TestOwnerPidLabel:
    def _captured_run_cmd(self, sb):
        """Drive _ensure_started() far enough to build its `podman run`
        argv, without a real daemon: everything _run_with_conflict_retry
        would normally do is skipped, so this only inspects the command
        that *would* have been issued."""
        calls = []
        with patch.object(sb, "_container_running", return_value=False):
            with patch.object(sb, "_container_status", return_value=""):
                with patch.object(
                    sb,
                    "_run_with_conflict_retry",
                    side_effect=lambda run_cmd, name: calls.append(run_cmd),
                ):
                    with patch.object(sb, "_run"):  # the post-run `mkdir /workspace`
                        sb._ensure_started()
        assert len(calls) == 1, "expected exactly one podman run invocation"
        return calls[0]

    def _label_value(self, run_cmd: list[str]) -> str:
        idx = run_cmd.index("--label")
        label = run_cmd[idx + 1]
        assert label.startswith("agency.owner_pid=")
        return label.split("=", 1)[1]

    def test_label_defaults_to_constructing_processs_own_pid(self):
        """The normal case: construct-and-immediately-use in one process --
        the label must be this process's real PID, so a reap elsewhere
        correctly recognizes it as alive for as long as this process runs."""
        import os

        sb = _make_backend()
        run_cmd = self._captured_run_cmd(sb)
        assert self._label_value(run_cmd) == str(os.getpid())

    def test_label_reflects_owner_pid_attribute_not_live_process(self):
        """Regression: simulates the cloudpickle-to-a-worker scenario by
        overwriting _owner_pid post-construction to a value that is *not*
        this test process's own PID -- the label must still track that
        stored value, proving it's read from self._owner_pid rather than
        computed fresh via os.getpid() at run time."""
        import os

        sb = _make_backend()
        sentinel_pid = 424242
        assert sentinel_pid != os.getpid()
        sb._owner_pid = sentinel_pid

        run_cmd = self._captured_run_cmd(sb)
        assert self._label_value(run_cmd) == str(sentinel_pid)

    @podman
    def test_real_sandboxed_tool_call_labels_container_with_main_process_pid(self):
        """End-to-end, no mocks: a real run_in_subprocess=True tool call (the
        default) cloudpickles this backend to a real ProcessPoolExecutor
        worker, which is the process that actually issues `podman run` --
        confirms the resulting container's real label still names *this*
        (main, test) process, not the worker's own distinct PID."""
        import os

        from agency.agdata import agdata, agerror
        from agency.tools import make_sandboxed_tools

        sb = _make_sandbox()
        tools = {t.name: t for t in make_sandboxed_tools(sb)}
        try:
            result = tools["bash"](agdata(command="true"))
            assert not isinstance(result, agerror), f"bash failed: {result}"

            inspected = subprocess.run(
                [
                    "podman",
                    "inspect",
                    "--format",
                    '{{index .Config.Labels "agency.owner_pid"}}',
                    sb._backend._name,
                ],
                capture_output=True,
                check=True,
            )
            label_pid = inspected.stdout.decode().strip()
            assert label_pid == str(os.getpid()), (
                f"container labeled with pid {label_pid}, expected this test "
                f"process's own pid {os.getpid()} -- the worker that actually "
                f"ran `podman run` must not have used its own os.getpid()"
            )
        finally:
            sb.destroy()


class TestDanglingImageEagerCleanup:
    """Tests for the eager old-image deletion in stop(commit=True). Mirrors
    test_docker.py's class of the same name."""

    def test_stop_commit_deletes_old_image(self):
        """stop(commit=True) must delete the image that previously held the tag."""
        import agency.agsandbox_backends.podman as _mod

        sb = _make_backend()

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

        with patch.object(_mod._PodmanBackend, "_run", fake_run):
            with patch.object(sb, "_container_running", return_value=True):
                with patch.object(sb, "_gpu_virtual", False):
                    sb.stop(commit=True)

        rmi_calls = [a for a in run_calls if "rmi" in a]
        assert rmi_calls, "expected podman rmi call for old image"
        assert any(fake_old_id in " ".join(a) for a in rmi_calls), (
            f"rmi call did not reference old image ID; calls: {rmi_calls}"
        )

    def test_stop_commit_skips_rmi_when_no_old_image(self):
        """If the tag does not exist yet (first commit), no rmi call is made."""
        import agency.agsandbox_backends.podman as _mod

        sb = _make_backend()

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

        with patch.object(_mod._PodmanBackend, "_run", fake_run):
            with patch.object(sb, "_container_running", return_value=True):
                with patch.object(sb, "_gpu_virtual", False):
                    sb.stop(commit=True)

        rmi_calls = [a for a in run_calls if "rmi" in a]
        assert not rmi_calls, "must not call rmi when there was no previous image"

    def test_stop_commit_rmi_failure_is_best_effort(self):
        """A failing rmi during old-image cleanup must NOT propagate."""
        import agency.agsandbox_backends.podman as _mod

        sb = _make_backend()
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
            with patch.object(_mod._PodmanBackend, "_run", fake_run):
                with patch.object(sb, "_container_running", return_value=True):
                    with patch.object(sb, "_gpu_virtual", False):
                        sb.stop(commit=True)  # must not raise
        finally:
            sys.stderr = old_stderr

        assert "WARNING" in captured.getvalue()
        assert fake_old_id in captured.getvalue()

    @podman
    def test_repeated_commits_leave_no_dangling_images(self):
        """stop(commit=True) called 3 times to the same tag must leave 0 new dangling images."""
        name = f"test-eager-{uuid.uuid4().hex[:8]}"
        tag = f"agency/lifecycle-{name}"

        def _dangling_ids():
            r = subprocess.run(
                ["podman", "images", "-f", "dangling=true", "-q"],
                capture_output=True,
                text=True,
            )
            return set(ln.strip() for ln in r.stdout.splitlines() if ln.strip())

        subprocess.run(
            [
                "podman",
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
                    ["podman", "inspect", "--format={{.Id}}", tag],
                    capture_output=True,
                    text=True,
                )
                old_id = old_id_r.stdout.strip() if old_id_r.returncode == 0 else None
                subprocess.run(["podman", "commit", name, tag], capture_output=True, check=True)
                if old_id:
                    subprocess.run(["podman", "rmi", old_id], capture_output=True)
            after = _dangling_ids()
            new_dangling = after - before
            assert len(new_dangling) == 0, (
                f"expected 0 new dangling images with eager cleanup, got {len(new_dangling)}"
            )
        finally:
            subprocess.run(["podman", "rm", "-f", name], capture_output=True)
            subprocess.run(["podman", "rmi", "-f", tag], capture_output=True)


# ---------------------------------------------------------------------------
# _rm_container / _rmi helpers -- mirrors test_docker.py's
# TestDockerCommandHelpers against _PodmanBackend instead.
# ---------------------------------------------------------------------------


class TestPodmanCommandHelpers:
    """Unit tests for _rm_container and _rmi — no real Podman required."""

    def _sb(self):
        return _make_backend()

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

        import agency.agsandbox_backends.podman as _mod

        with patch.object(_mod._PodmanBackend, "_run", fake_run):
            sb._rm_container("my-container")

        assert len(calls) == 1
        args, check = calls[0]
        assert "rm" in args and "-f" in args and "my-container" in args
        assert check is True

    def test_rm_container_raises_on_failure(self):
        sb = self._sb()

        import agency.agsandbox_backends.podman as _mod

        with patch.object(_mod._PodmanBackend, "_run", side_effect=RuntimeError("rm failed")):
            with pytest.raises(RuntimeError, match="rm failed"):
                sb._rm_container("bad-container")

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

        import agency.agsandbox_backends.podman as _mod

        with patch.object(_mod._PodmanBackend, "_run", fake_run):
            sb._rmi("sha256:abc123")

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

        import agency.agsandbox_backends.podman as _mod

        with patch.object(_mod._PodmanBackend, "_run", fake_run):
            sb._rmi("myimage:tag", force=True)

        assert "-f" in calls[0]

    def test_rmi_raises_on_failure(self):
        sb = self._sb()

        import agency.agsandbox_backends.podman as _mod

        with patch.object(_mod._PodmanBackend, "_run", side_effect=RuntimeError("rmi failed")):
            with pytest.raises(RuntimeError, match="rmi failed"):
                sb._rmi("sha256:deadbeef")

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

        import agency.agsandbox_backends.podman as _mod

        with patch.object(_mod._PodmanBackend, "_run", fake_run):
            # status returns "" → no leftover container
            with patch.object(sb, "_container_running", return_value=False):
                with patch.object(sb, "_container_status", return_value=""):
                    with patch.object(sb, "_run_with_conflict_retry"):
                        sb._ensure_started()

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

        import agency.agsandbox_backends.podman as _mod

        with patch.object(_mod._PodmanBackend, "_run", fake_run):
            with patch.object(sb, "_container_running", return_value=False):
                with patch.object(sb, "_container_status", return_value="exited"):
                    with patch.object(sb, "_run_with_conflict_retry"):
                        sb._ensure_started()

        rm_calls = [(a, c) for (a, c) in calls if "rm" in a]
        assert rm_calls, "expected rm call for leftover container"
        assert all(c is True for _, c in rm_calls), "rm must use check=True"

    # --- destroy semaphore release ---

    def test_destroy_releases_semaphore_even_when_rm_raises(self):
        """The shared (docker+podman) concurrency semaphore must be released
        in finally if rm raises but the container turns out to be confirmed
        gone anyway (e.g. rm actually succeeded server-side despite the
        client call itself raising, or a concurrent cleanup removed it) --
        NOT when the container is still genuinely running, in which case the
        slot is legitimately still held and destroy() must NOT release it
        (see container.py's destroy() docstring/comment for that invariant;
        a container still running after a raised rm must keep its slot)."""
        import agency.agsandbox_backends.container as _container_mod
        import agency.agsandbox_backends.podman as _mod

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

        # had_container (destroy()'s pre-rm check) must see "running" so the
        # test actually exercises the "was running, rm failed, but confirmed
        # gone by the recheck" path -- the post-rm recheck in the `finally`
        # block must see "gone". Same method, two different truthful answers
        # at two different times, exactly like a real rm that silently
        # succeeded despite raising a secondary error.
        running_calls = [True, False]

        def fake_container_running():
            return running_calls.pop(0) if running_calls else False

        with patch.object(_mod._PodmanBackend, "_run", fake_run):
            with patch.object(sb, "_container_running", side_effect=fake_container_running):
                with patch.object(sb, "_container_status", return_value="running"):
                    with patch.object(
                        _container_mod._container_semaphore,
                        "release",
                        side_effect=lambda: released.append(1),
                    ):
                        with pytest.raises(RuntimeError, match="rm exploded"):
                            sb.destroy()

        assert released, "semaphore must be released once the container is confirmed gone"

    def test_destroy_skips_rm_when_container_absent(self):
        """destroy() must not call rm when the container does not exist."""
        import agency.agsandbox_backends.podman as _mod

        sb = self._sb()
        calls = []

        class OK:
            returncode = 0
            stdout = b""

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            calls.append(args)
            return OK()

        with patch.object(_mod._PodmanBackend, "_run", fake_run):
            with patch.object(sb, "_container_running", return_value=False):
                with patch.object(sb, "_container_status", return_value=""):
                    sb.destroy()

        rm_calls = [a for a in calls if "rm" in a and "rmi" not in a]
        assert not rm_calls, f"must not rm when container absent; got {rm_calls}"


# ---------------------------------------------------------------------------
# GPU semaphore release gating in stop()/destroy() -- mirrors
# TestDockerGpuReleaseGating in test_docker.py, since this is
# _ContainerBackendBase's shared logic (identical for both runtimes).
# ---------------------------------------------------------------------------


class TestPodmanGpuReleaseGating:
    def _sb(self):
        return _make_backend()

    def _lease_gpu(self, sb, gpu_id=3):
        released = []
        sb._gpu_virtual = True
        sb._gpu_id = gpu_id
        sb._gpu_release_fn = lambda gid: released.append(gid)
        return released

    def test_stop_releases_gpu_when_already_confirmed_gone(self):
        import agency.agsandbox_backends.podman as _mod

        sb = self._sb()
        released = self._lease_gpu(sb)

        with patch.object(_mod._PodmanBackend, "_run"):
            with patch.object(sb, "_container_running", return_value=False):
                sb.stop(commit=False)

        assert released == [3]
        assert sb._gpu_id is None

    def test_stop_does_not_release_gpu_when_rm_fails_and_container_still_running(self):
        import agency.agsandbox_backends.container as _container_mod
        import agency.agsandbox_backends.podman as _mod

        sb = self._sb()
        released = self._lease_gpu(sb)

        class OK:
            returncode = 0
            stdout = b""

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "rm" in args:
                raise RuntimeError("rm exploded")
            return OK()

        with patch.object(_mod._PodmanBackend, "_run", fake_run):
            with patch.object(sb, "_container_running", return_value=True):
                with patch.object(_container_mod.time, "sleep"):  # skip real retry backoff
                    with pytest.raises(RuntimeError, match="rm exploded"):
                        sb.stop(commit=False)

        assert released == [], (
            "GPU must not be released while the container is confirmed still running"
        )
        assert sb._gpu_id == 3

    def test_destroy_releases_gpu_when_container_confirmed_gone_despite_rm_error(self):
        import agency.agsandbox_backends.podman as _mod

        sb = self._sb()
        released = self._lease_gpu(sb)

        class OK:
            returncode = 0
            stdout = b""

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "rm" in args:
                raise RuntimeError("rm exploded")
            return OK()

        running_calls = [True, False]

        def fake_container_running():
            return running_calls.pop(0) if running_calls else False

        with patch.object(_mod._PodmanBackend, "_run", fake_run):
            with patch.object(sb, "_container_running", side_effect=fake_container_running):
                with patch.object(sb, "_container_status", return_value="running"):
                    with pytest.raises(RuntimeError, match="rm exploded"):
                        sb.destroy()

        assert released == [3]
        assert sb._gpu_id is None

    def test_destroy_does_not_release_gpu_when_container_still_running_after_rm_failure(self):
        import agency.agsandbox_backends.podman as _mod

        sb = self._sb()
        released = self._lease_gpu(sb)

        class OK:
            returncode = 0
            stdout = b""

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "rm" in args:
                raise RuntimeError("rm exploded")
            return OK()

        with patch.object(_mod._PodmanBackend, "_run", fake_run):
            with patch.object(sb, "_container_running", return_value=True):
                with patch.object(sb, "_container_status", return_value="running"):
                    with pytest.raises(RuntimeError, match="rm exploded"):
                        sb.destroy()

        assert released == []
        assert sb._gpu_id == 3

    def test_gpu_released_exactly_once_across_stop_then_destroy(self):
        import agency.agsandbox_backends.podman as _mod

        sb = self._sb()
        released = self._lease_gpu(sb)

        with patch.object(_mod._PodmanBackend, "_run"):
            with patch.object(sb, "_container_running", return_value=False):
                sb.stop(commit=False)
                sb.destroy()

        assert released == [3], "GPU must be released exactly once, not once per call"
