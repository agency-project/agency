"""Unit tests for agency.agsandbox_backends.container's shared, runtime-
agnostic logic: the startup orphan reaper and its PID-liveness check; the
session-keyring-quota machinery (_keyring_container_limit/keyring_quota/
_semaphore_held_count); and _ContainerBackendBase's concrete
_is_quota_exhaustion_error()/_wait_for_quota_slot()/_quota_diagnostics()/
_acquire_runtime_slot()/_release_runtime_slot() hooks plus
_run_with_conflict_retry()'s dispatch through them -- exercised here against
_PodmanBackend specifically to prove docker and podman share identical
behavior (both are subject to the same kernel session-keyring quota; see
container.py's module docstring for why). No real docker/podman required --
everything here mocks subprocess.run/_run and the /proc reads."""

from __future__ import annotations

import os
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
        assert run_mock.call_count == 1, "an unrelated failure must not be retried"
