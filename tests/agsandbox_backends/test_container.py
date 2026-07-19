"""Unit tests for agency.agsandbox_backends.container's shared, runtime-
agnostic logic: the startup orphan reaper and its PID-liveness check; the
session-keyring-quota machinery (_keyring_container_limit/keyring_quota/
_semaphore_held_count); _ContainerBackendBase's concrete
_is_quota_exhaustion_error()/_wait_for_quota_slot()/_quota_diagnostics()/
_acquire_runtime_slot()/_release_runtime_slot() hooks plus
_run_with_conflict_retry()'s dispatch through them -- exercised here against
_PodmanBackend specifically to prove docker and podman share identical
behavior (both are subject to the same kernel session-keyring quota; see
container.py's module docstring for why); and _gpu_flags()'s per-runtime
NVIDIA/ROCm flag selection (regression coverage for the bug where Docker's
``--gpus all`` was used unconditionally for Podman too, which Podman accepts
without error but never actually mounts the driver for -- see
TestGpuFlagsPerRuntime, and TestPodmanGpuPassthroughIntegration at the bottom
for the real-container check that would have caught it).

No real docker/podman required for anything above the integration test class
at the bottom -- everything else here mocks subprocess.run/_run, the /proc
reads, and detect_gpus()/shutil.which()."""

from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

import agency.agsandbox_backends.container as _container


class TestPidAlive:
    def test_own_pid_is_alive(self):
        assert _container._pid_alive(os.getpid()) is True

    def test_nonexistent_pid_is_not_alive(self):
        # PID 1 is always init/systemd and always alive; a made-up huge PID
        # almost certainly isn't assigned on any real system.
        assert _container._pid_alive(2**31 - 1) is False

    def test_permission_error_counts_as_alive(self):
        with patch("os.kill", side_effect=PermissionError):
            assert _container._pid_alive(1) is True


class TestReapOrphanedContainers:
    def setup_method(self):
        # Reset the once-per-process guard so each test gets a clean run --
        # otherwise whichever test runs first "wins" for the whole session.
        _container._reap_done = False

    def _fake_ps_result(self, lines: list[str]):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ("\n".join(lines) + "\n" if lines else "").encode()
        return result

    def test_removes_container_owned_by_dead_pid(self):
        dead_pid = 2**31 - 1
        ps_result = self._fake_ps_result([f"abc123\t{dead_pid}\tsandbox-rXXXXXXX-someagent"])
        with patch(
            "agency.agsandbox_backends.container.get_container_runtime", return_value="docker"
        ):
            with patch("subprocess.run", return_value=ps_result) as mock_run:
                _container.reap_orphaned_containers()

        rm_calls = [c for c in mock_run.call_args_list if "rm" in c.args[0]]
        rmi_calls = [c for c in mock_run.call_args_list if "rmi" in c.args[0]]
        assert any("abc123" in c.args[0] for c in rm_calls), "expected rm -f of the dead container"
        assert any("agency/lifecycle-sandbox-rxxxxxxx-someagent" in c.args[0] for c in rmi_calls), (
            "expected an attempt to remove the matching lifecycle image"
        )

    def test_skips_container_owned_by_live_pid(self):
        live_pid = os.getpid()
        ps_result = self._fake_ps_result([f"abc123\t{live_pid}\tsandbox-rXXXXXXX-someagent"])
        with patch(
            "agency.agsandbox_backends.container.get_container_runtime", return_value="docker"
        ):
            with patch("subprocess.run", return_value=ps_result) as mock_run:
                _container.reap_orphaned_containers()

        # Only the initial `docker ps` call should have happened -- no rm/rmi.
        assert mock_run.call_count == 1

    def test_skips_container_owned_by_self(self):
        ps_result = self._fake_ps_result([f"abc123\t{os.getpid()}\tsandbox-rXXXXXXX-someagent"])
        with patch(
            "agency.agsandbox_backends.container.get_container_runtime", return_value="docker"
        ):
            with patch("subprocess.run", return_value=ps_result) as mock_run:
                _container.reap_orphaned_containers()
        assert mock_run.call_count == 1

    def test_no_containers_found_is_a_noop(self):
        ps_result = self._fake_ps_result([])
        with patch(
            "agency.agsandbox_backends.container.get_container_runtime", return_value="docker"
        ):
            with patch("subprocess.run", return_value=ps_result) as mock_run:
                _container.reap_orphaned_containers()
        assert mock_run.call_count == 1

    def test_runs_at_most_once_per_process(self):
        ps_result = self._fake_ps_result([])
        with patch(
            "agency.agsandbox_backends.container.get_container_runtime", return_value="docker"
        ):
            with patch("subprocess.run", return_value=ps_result) as mock_run:
                _container.reap_orphaned_containers()
                _container.reap_orphaned_containers()
                _container.reap_orphaned_containers()
        assert mock_run.call_count == 1, "second/third calls must be no-ops"

    def test_no_usable_runtime_is_a_noop(self):
        with patch(
            "agency.agsandbox_backends.container.get_container_runtime",
            side_effect=RuntimeError("no docker/podman"),
        ):
            with patch("subprocess.run") as mock_run:
                _container.reap_orphaned_containers()
        mock_run.assert_not_called()

    def test_exception_during_ps_call_is_swallowed(self):
        """A failure here must never propagate and block real sandbox construction."""
        with patch(
            "agency.agsandbox_backends.container.get_container_runtime", return_value="docker"
        ):
            with patch("subprocess.run", side_effect=OSError("docker binary vanished")):
                _container.reap_orphaned_containers()  # must not raise
        # Still marked done -- a broken daemon shouldn't retry every construction.
        assert _container._reap_done is True


class TestQuotaHooksSharedAcrossRuntimes:
    """_ContainerBackendBase's quota hooks are concrete (not no-ops) and
    identical for both runtimes -- exercised here via _PodmanBackend
    specifically to prove Podman is NOT exempt from the kernel session-
    keyring quota, despite its per-container user namespaces (see
    container.py's module docstring for why: runc joins/creates the session
    keyring against the real host UID before the container process finishes
    transitioning into its remapped identity)."""

    def _sb(self):
        from agency.agsandbox_backends.podman import _PodmanBackend

        return _PodmanBackend(
            "agent",
            name="podman-quota-hook-test",
            checkpoint_image=None,
            base_image="img",
            mounts={},
            agconfig=None,
        )

    def test_matches_session_key_message(self):
        sb = self._sb()
        assert sb._is_quota_exhaustion_error("unable to create session key: disk quota exceeded")

    def test_matches_disk_quota_exceeded_with_keyring(self):
        sb = self._sb()
        assert sb._is_quota_exhaustion_error("disk quota exceeded for keyring")

    def test_disk_quota_exceeded_without_keyring_does_not_match(self):
        """Docker/podman also emit a plain filesystem "disk quota exceeded"
        for unrelated reasons (e.g. a full overlay volume) -- only the
        keyring variant should trigger the quota-wait path."""
        sb = self._sb()
        assert not sb._is_quota_exhaustion_error("disk quota exceeded")

    def test_unrelated_stderr_does_not_match(self):
        sb = self._sb()
        assert not sb._is_quota_exhaustion_error("no such image: agency-sandbox:latest")

    def test_wait_for_quota_slot_polls_until_a_slot_frees_up(self):
        sb = self._sb()
        quotas = iter([{"free": 0}, {"free": 0}, {"free": 1}])
        with patch.object(_container, "keyring_quota", side_effect=lambda: next(quotas)):
            with patch.object(_container.time, "sleep") as sleep_mock:
                sb._wait_for_quota_slot()
        assert sleep_mock.call_count == 2

    def test_wait_for_quota_slot_gives_up_once_deadline_passes(self):
        sb = self._sb()
        # First monotonic() call establishes the deadline; the second (the
        # loop's own check) reports a time far past it -- the loop body
        # must never run, so it must never sleep either.
        moments = iter([0.0, 10_000.0])
        with patch.object(_container, "keyring_quota", return_value={"free": 0}):
            with patch.object(_container.time, "monotonic", side_effect=lambda: next(moments)):
                with patch.object(_container.time, "sleep") as sleep_mock:
                    sb._wait_for_quota_slot()
        sleep_mock.assert_not_called()

    def test_quota_diagnostics_format(self):
        sb = self._sb()
        with patch.object(_container, "keyring_quota", return_value={"used": 5, "max": 200}):
            with patch.object(_container, "_semaphore_held_count", return_value="3/195"):
                assert (
                    sb._quota_diagnostics()
                    == "[keyring: 5/200 used, framework semaphore: 3/195 held]"
                )

    def test_acquire_and_release_runtime_slot_use_the_shared_semaphore(self):
        sb = self._sb()
        with patch.object(_container._container_semaphore, "acquire") as acquire_mock:
            sb._acquire_runtime_slot()
        acquire_mock.assert_called_once()
        with patch.object(_container._container_semaphore, "release") as release_mock:
            sb._release_runtime_slot()
        release_mock.assert_called_once()


class TestKeyringQuotaDiagnostics:
    def test_keyring_container_limit_uses_kernel_maxkeys_minus_buffer(self):
        with patch("pathlib.Path.read_text", return_value="500\n"):
            limit = _container._keyring_container_limit()
        fields = _container.AgSandboxBackendFields()
        assert limit == 500 - fields.container_limit_buffer

    def test_keyring_container_limit_never_below_floor(self):
        with patch("pathlib.Path.read_text", return_value="1\n"):
            limit = _container._keyring_container_limit()
        fields = _container.AgSandboxBackendFields()
        assert limit == fields.container_limit_floor

    def test_keyring_container_limit_falls_back_when_proc_unreadable(self):
        with patch("pathlib.Path.read_text", side_effect=OSError("no such file")):
            limit = _container._keyring_container_limit()
        fields = _container.AgSandboxBackendFields()
        assert limit == fields.container_limit_fallback - fields.container_limit_buffer

    def test_keyring_quota_reports_used_max_and_free(self):
        def fake_read_text(self):
            return "200\n" if "maxkeys" in str(self) else "a\nb\nc\n"

        with patch("pathlib.Path.read_text", fake_read_text):
            quota = _container.keyring_quota()
        assert quota == {"used": 3, "max": 200, "free": 197}

    def test_keyring_quota_reports_minus_one_when_proc_unreadable(self):
        with patch("pathlib.Path.read_text", side_effect=OSError("no such file")):
            quota = _container.keyring_quota()
        assert quota == {"used": -1, "max": -1, "free": -1}

    def test_semaphore_held_count_reflects_acquired_slots(self):
        """held/limit should go up by exactly one slot per acquire() -- checked
        as a delta against the semaphore's already-live real value rather than
        an assumed absolute count, since _container_semaphore is a real
        process-wide multiprocessing.Semaphore shared with every other test
        (and, now, with both docker and podman backends)."""
        before_held, limit = _container._semaphore_held_count().split("/")
        _container._container_semaphore.acquire()
        try:
            after_held, limit_after = _container._semaphore_held_count().split("/")
        finally:
            _container._container_semaphore.release()
        assert limit_after == limit
        assert int(after_held) == int(before_held) + 1

    def test_semaphore_held_count_falls_back_to_unknown_on_error(self):
        fake_semlock = MagicMock()
        fake_semlock._get_value.side_effect = Exception("boom")
        with patch.object(_container, "_keyring_container_limit", return_value=10):
            with patch.object(_container._container_semaphore, "_semlock", fake_semlock):
                held = _container._semaphore_held_count()
        assert held == "?/10"


class TestRunWithConflictRetryHooks:
    """_run_with_conflict_retry()'s dispatch to the quota hooks, exercised
    against _PodmanBackend so it's clear the dispatch mechanism itself is
    shared/runtime-agnostic rather than accidentally Docker-specific."""

    class _FakeResult:
        def __init__(self, returncode: int, stderr: bytes = b""):
            self.returncode = returncode
            self.stderr = stderr

    def _sb(self):
        from agency.agsandbox_backends.podman import _PodmanBackend

        return _PodmanBackend(
            "agent",
            name="podman-retry-test",
            checkpoint_image=None,
            base_image="img",
            mounts={},
            agconfig=None,
        )

    def test_quota_branch_waits_then_cleans_up_stale_container_and_retries(self):
        sb = self._sb()
        results = iter([self._FakeResult(1, b"quota exhausted"), self._FakeResult(0)])
        with patch.object(sb, "_run", side_effect=lambda *a, **k: next(results)):
            with patch.object(sb, "_is_quota_exhaustion_error", return_value=True):
                with patch.object(sb, "_wait_for_quota_slot") as wait_mock:
                    with patch.object(sb, "_container_status", return_value="created"):
                        with patch.object(sb, "_rm_container") as rm_mock:
                            sb._run_with_conflict_retry(["podman", "run", "--name", "x"], "x")

        wait_mock.assert_called_once()
        rm_mock.assert_called_once_with("x")

    def test_conflict_branch_raises_already_running_when_container_is_up(self):
        sb = self._sb()
        result = self._FakeResult(1, b"Conflict. The container name is already in use")
        with patch.object(sb, "_run", return_value=result):
            with patch.object(sb, "_container_running", return_value=True):
                with pytest.raises(_container._ContainerAlreadyRunning):
                    sb._run_with_conflict_retry(["podman", "run", "--name", "x"], "x")

    def test_conflict_branch_removes_stale_container_and_waits_for_quota_each_attempt(self):
        sb = self._sb()
        result = self._FakeResult(1, b"Conflict. The container name is already in use")
        with patch.object(sb, "_run", return_value=result):
            with patch.object(sb, "_container_running", return_value=False):
                with patch.object(sb, "_container_status", return_value=""):
                    with patch.object(sb, "_rm_container") as rm_mock:
                        with patch.object(sb, "_wait_for_quota_slot") as wait_mock:
                            with patch("time.sleep"):
                                with pytest.raises(RuntimeError, match="failed after retries"):
                                    sb._run_with_conflict_retry(
                                        ["podman", "run", "--name", "x"], "x"
                                    )
        # Every attempt hits the conflict branch, which removes the stale
        # container and then unconditionally waits for a quota slot (a no-op
        # here, but the same code path _DockerBackend relies on to also wait
        # out a keyring exhaustion discovered alongside the conflict).
        assert rm_mock.call_count == sb.conflict_retry_max_attempts
        assert wait_mock.call_count == sb.conflict_retry_max_attempts

    def test_final_error_names_the_live_runtime_and_includes_diagnostics(self):
        sb = self._sb()
        result = self._FakeResult(1, b"Conflict. The container name is already in use")
        with patch.object(sb, "_run", return_value=result):
            with patch.object(sb, "_container_running", return_value=False):
                with patch.object(sb, "_container_status", return_value=""):
                    with patch.object(sb, "_rm_container"):
                        with patch.object(sb, "_wait_for_quota_slot"):
                            with patch.object(sb, "_quota_diagnostics", return_value="[diag: 1/2]"):
                                with patch("time.sleep"):
                                    with pytest.raises(RuntimeError) as exc_info:
                                        sb._run_with_conflict_retry(
                                            ["podman", "run", "--name", "x"], "x"
                                        )
        msg = str(exc_info.value)
        assert "podman run --name x failed after retries" in msg
        assert "[diag: 1/2]" in msg
        assert "already in use" in msg

    def test_non_conflict_non_quota_failure_raises_immediately_without_retry(self):
        sb = self._sb()
        result = self._FakeResult(1, b"no such image: agency-sandbox:latest")
        run_mock = MagicMock(return_value=result)
        with patch.object(sb, "_run", run_mock):
            with pytest.raises(RuntimeError, match="no such image"):
                sb._run_with_conflict_retry(["podman", "run", "--name", "x"], "x")


# ---------------------------------------------------------------------------
# _gpu_flags(runtime) -- regression coverage for the bug where Docker's
# ``--gpus all`` was used unconditionally for Podman too. Podman accepts that
# flag without erroring but never mounts the NVIDIA driver/devices for it, so
# a container started that way silently has zero GPU access. No real
# docker/podman required here -- detect_gpus()/shutil.which() are mocked.
# ---------------------------------------------------------------------------


class TestGpuFlagsPerRuntime:
    def setup_method(self):
        # _gpu_flags_cache is a module-level dict keyed by runtime -- clear it
        # so each test observes only its own mocked detect_gpus()/which().
        _container._gpu_flags_cache.clear()

    def test_docker_gets_gpus_all_for_nvidia(self):
        with patch.object(_container, "detect_gpus", return_value=[0, 1]):
            with patch.object(_container.shutil, "which", return_value="/usr/bin/nvidia-smi"):
                assert _container._gpu_flags("docker") == ["--gpus", "all"]

    def test_podman_gets_cdi_device_flag_for_nvidia_not_docker_syntax(self):
        """The actual regression: Podman must NOT get Docker's --gpus flag --
        it silently accepts it without mounting the driver (confirmed against
        a real host: `podman run --gpus all ... nvidia-smi` prints "WARNING:
        The NVIDIA Driver was not detected" and nvidia-smi isn't even on
        PATH), so it needs the CDI device syntax instead."""
        with patch.object(_container, "detect_gpus", return_value=[0, 1]):
            with patch.object(_container.shutil, "which", return_value="/usr/bin/nvidia-smi"):
                flags = _container._gpu_flags("podman")
        assert flags == ["--device", "nvidia.com/gpu=all"]
        assert "--gpus" not in flags

    def test_docker_and_podman_flags_differ_and_are_cached_independently(self):
        with patch.object(_container, "detect_gpus", return_value=[0, 1]):
            with patch.object(_container.shutil, "which", return_value="/usr/bin/nvidia-smi"):
                docker_flags = _container._gpu_flags("docker")
                podman_flags = _container._gpu_flags("podman")
        assert docker_flags != podman_flags
        # Cached per runtime -- fetching docker's again must not have been
        # clobbered by the podman call (or vice versa).
        assert _container._gpu_flags_cache["docker"] == ["--gpus", "all"]
        assert _container._gpu_flags_cache["podman"] == ["--device", "nvidia.com/gpu=all"]

    def test_rocm_flags_identical_for_both_runtimes(self):
        with patch.object(_container, "detect_gpus", return_value=[0]):
            with patch.object(_container.shutil, "which", return_value=None):
                docker_flags = _container._gpu_flags("docker")
                podman_flags = _container._gpu_flags("podman")
        expected = ["--device", "/dev/kfd", "--device", "/dev/dri"]
        assert docker_flags == expected
        assert podman_flags == expected

    def test_no_gpu_detected_returns_empty_for_both_runtimes(self):
        with patch.object(_container, "detect_gpus", return_value=[]):
            assert _container._gpu_flags("docker") == []
            assert _container._gpu_flags("podman") == []

    def test_backend_construction_passes_its_own_runtime_to_gpu_flags(self):
        """_ContainerBackendBase.__init__ must thread self._runtime through
        to _gpu_flags() rather than hardcoding/omitting it -- this is exactly
        what the bug was: the call site used to be plain `_gpu_flags()`."""
        from agency.agsandbox_backends.podman import _PodmanBackend

        with patch.object(_container, "_gpu_flags", return_value=["sentinel"]) as gpu_flags_mock:
            _PodmanBackend(
                "agent",
                name="podman-gpu-ctor-test",
                checkpoint_image=None,
                base_image="img",
                mounts={},
                agconfig=None,
            )
        gpu_flags_mock.assert_called_once_with("podman")


def _podman_available() -> bool:
    try:
        return subprocess.run(["podman", "info"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def _host_gpu_available() -> bool:
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


podman_gpu = pytest.mark.skipif(
    not (_podman_available() and _host_gpu_available()),
    reason="Podman daemon or NVIDIA GPU not available",
)


class TestPodmanGpuPassthroughIntegration:
    """The check that would have actually caught the regression above: starts
    a REAL podman container (no mocks anywhere) using the exact flags
    _gpu_flags('podman') returns, and asserts nvidia-smi run *inside* that
    container really sees the host's GPUs -- not just that the framework's
    own CUDA_VISIBLE_DEVICES env var plumbing works (see
    test_agsandbox.py's TestSandboxedTools/TestResourceTools, which check
    only that -- a container with zero real GPU access still passes those).

    Requires a real podman binary/daemon and a real NVIDIA GPU -- skipped
    automatically otherwise. Uses the same `agency-sandbox:latest` image the
    rest of the suite's @docker-marked real-daemon tests use, built via
    `images/build.sh`.
    """

    IMAGE = "agency-sandbox:latest"

    @podman_gpu
    def test_nvidia_smi_inside_a_real_podman_container_sees_the_gpus(self):
        flags = _container._gpu_flags("podman")
        assert flags, "expected non-empty GPU flags on a host with a real GPU"
        result = subprocess.run(
            ["podman", "run", "--rm", *flags, self.IMAGE, "nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"nvidia-smi failed inside the container (exit {result.returncode}); "
            f"this is exactly the failure mode of the --gpus-all-on-podman bug:\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )
        assert "GPU" in result.stdout, f"expected a GPU listing, got: {result.stdout!r}"


def _docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def _docker_has_image(image: str) -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "image", "inspect", image], capture_output=True, timeout=10
            ).returncode
            == 0
        )
    except Exception:
        return False


docker_gpu = pytest.mark.skipif(
    not (
        _docker_available() and _host_gpu_available() and _docker_has_image("agency-sandbox:latest")
    ),
    reason="Docker daemon, NVIDIA GPU, or local agency-sandbox:latest docker image not available",
)


class TestDockerGpuPassthroughIntegration:
    """Symmetric to TestPodmanGpuPassthroughIntegration above -- starts a REAL
    docker container (no mocks) using the exact flags _gpu_flags('docker')
    returns and asserts nvidia-smi run *inside* it sees the host's GPUs.

    Docker was never actually broken by the regression this file guards
    against (the bug was Podman incorrectly getting Docker's ``--gpus`` flag,
    not the other way around), but this exists so a future change to the
    Docker branch of _gpu_flags() gets the same real-container safety net
    Podman has, rather than relying on the unit tests in
    TestGpuFlagsPerRuntime alone.

    Requires a real docker binary/daemon, a real NVIDIA GPU, AND a locally
    built `agency-sandbox:latest` docker image -- skipped automatically
    otherwise. Unlike the rest of this suite's plain @docker-marked tests
    (which only check daemon reachability, and can therefore fail outright
    with "pull access denied" on a host where the image was only ever built
    for podman -- see test_docker.py), this also checks the image is
    actually present locally before running, so it degrades to a skip
    instead of a false-positive failure on such a host.
    """

    IMAGE = "agency-sandbox:latest"

    @docker_gpu
    def test_nvidia_smi_inside_a_real_docker_container_sees_the_gpus(self):
        flags = _container._gpu_flags("docker")
        assert flags, "expected non-empty GPU flags on a host with a real GPU"
        result = subprocess.run(
            ["docker", "run", "--rm", *flags, self.IMAGE, "nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"nvidia-smi failed inside the container (exit {result.returncode}):\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )
        assert "GPU" in result.stdout, f"expected a GPU listing, got: {result.stdout!r}"
