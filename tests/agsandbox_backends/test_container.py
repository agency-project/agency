"""Unit tests for agency.agsandbox_backends.container's shared, runtime-
agnostic logic: the startup orphan reaper and its PID-liveness check. No
real docker/podman required -- everything here mocks subprocess.run."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

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
