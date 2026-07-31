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
reads, and detect_gpus()/shutil.which().

The bottom of the file also covers a second, unrelated regression: agents
routinely put a literal ``CUDA_VISIBLE_DEVICES=<n> ...`` prefix on their own
bash commands (common ML boilerplate, and an easy misreading of
reserve_gpu's "always use cuda:0" instruction). Under plain POSIX var=val-
prefix semantics that silently overrides base.py's exec()-time export for
just that child process, routing real compute onto whatever GPU the agent
hardcoded while the pool's semaphore bookkeeping still believes the sandbox
holds the GPU it actually leased. TestCvdOverrideProtectionIntegration below
proves the ``readonly`` fix in base.py's exec() holds against a real
container on both runtimes."""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
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

        # The container ps call, then the lifecycle-image scan (see
        # TestReapOrphanedLifecycleImages) -- no rm/rmi of anything.
        assert mock_run.call_count == 2
        assert not any("rm" in c.args[0] or "rmi" in c.args[0] for c in mock_run.call_args_list)

    def test_skips_container_owned_by_self(self):
        ps_result = self._fake_ps_result([f"abc123\t{os.getpid()}\tsandbox-rXXXXXXX-someagent"])
        with patch(
            "agency.agsandbox_backends.container.get_container_runtime", return_value="docker"
        ):
            with patch("subprocess.run", return_value=ps_result) as mock_run:
                _container.reap_orphaned_containers()
        assert mock_run.call_count == 2

    def test_no_containers_found_is_a_noop(self):
        ps_result = self._fake_ps_result([])
        with patch(
            "agency.agsandbox_backends.container.get_container_runtime", return_value="docker"
        ):
            with patch("subprocess.run", return_value=ps_result) as mock_run:
                _container.reap_orphaned_containers()
        assert mock_run.call_count == 2

    def test_runs_at_most_once_per_process(self):
        ps_result = self._fake_ps_result([])
        with patch(
            "agency.agsandbox_backends.container.get_container_runtime", return_value="docker"
        ):
            with patch("subprocess.run", return_value=ps_result) as mock_run:
                _container.reap_orphaned_containers()
                _container.reap_orphaned_containers()
                _container.reap_orphaned_containers()
        assert mock_run.call_count == 2, "second/third calls must be no-ops"

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


class TestReapOrphanedLifecycleImages:
    """_reap_orphaned_lifecycle_images() -- the image-side half of the
    reaper, catching lifecycle images whose owning CONTAINER is already
    gone (so TestReapOrphanedContainers's container-label scan above never
    even sees them). Calls the function directly rather than going through
    reap_orphaned_containers() (whose own once-per-process guard and empty
    container-ps step are already covered above).

    Three real subprocess.run shapes are involved, each mocked separately
    by dispatching on which subcommand appears in argv -- unlike
    `docker ps`, `docker images`'s --format context has no .Label/.Labels
    accessor at all (confirmed empirically against a real daemon: an
    earlier version of this scan used `docker images --format
    '...{{.Label "..."}}'` in one combined call, which fails with a
    template-parsing error every time -- returncode != 0, so the whole
    scan silently no-op'd, real bug that shipped and was only caught by
    checking a real host, not by these mocks, since they modeled the
    (wrong) single-call shape too):

    1. `docker images --filter label=... --format {{.Repository}}:{{.Tag}}`
       -- list of candidate tags, no label value.
    2. `docker inspect --format {{index .Config.Labels "..."}} <tag>`
       -- the actual label value, one call per candidate tag.
    3. `docker ps -a --filter ancestor=<tag> --format {{.ID}}` -- the
       safety check, only reached if step 2's PID looks dead.
    """

    def _fake_result(self, returncode=0, stdout=""):
        result = MagicMock()
        result.returncode = returncode
        result.stdout = stdout.encode() if isinstance(stdout, str) else stdout
        return result

    def _dispatcher(self, *, images_lines=None, labels=None, ancestor_lines=None, images_rc=0):
        """*labels* maps tag -> pid string (or None to simulate a missing/
        unparseable label) for the per-tag inspect call. *ancestor_lines*
        is shared across every tag's ancestor check (fine: these tests
        only ever exercise one candidate tag at a time)."""
        labels = labels or {}

        def fake_run(args, **kwargs):
            if "images" in args:
                if images_rc != 0:
                    return self._fake_result(returncode=images_rc)
                lines = images_lines or []
                return self._fake_result(stdout="\n".join(lines) + ("\n" if lines else ""))
            if "inspect" in args:
                tag = args[-1]
                pid_str = labels.get(tag)
                return self._fake_result(stdout="" if pid_str is None else str(pid_str))
            if "ps" in args:
                lines = ancestor_lines if ancestor_lines is not None else []
                return self._fake_result(stdout="\n".join(lines) + ("\n" if lines else ""))
            if "rmi" in args:
                return self._fake_result()
            raise AssertionError(f"unexpected subprocess.run call: {args}")

        return fake_run

    def test_removes_lifecycle_image_owned_by_dead_pid(self):
        dead_pid = 2**31 - 1
        tag = "agency/lifecycle-someagent:latest"
        fake_run = self._dispatcher(images_lines=[tag], labels={tag: dead_pid}, ancestor_lines=[])

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            _container._reap_orphaned_lifecycle_images("docker", os.getpid())

        rmi_calls = [c for c in mock_run.call_args_list if "rmi" in c.args[0]]
        assert any(tag in c.args[0] for c in rmi_calls)

    def test_skips_image_owned_by_live_pid(self):
        tag = "agency/lifecycle-someagent:latest"
        fake_run = self._dispatcher(images_lines=[tag], labels={tag: os.getpid()})
        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            _container._reap_orphaned_lifecycle_images("docker", os.getpid() + 1)
        # images list + one inspect -- never reaches the ancestor check or rmi.
        assert mock_run.call_count == 2
        assert not any("rmi" in c.args[0] for c in mock_run.call_args_list)

    def test_skips_image_owned_by_self(self):
        own_pid = os.getpid()
        tag = "agency/lifecycle-someagent:latest"
        fake_run = self._dispatcher(images_lines=[tag], labels={tag: own_pid})
        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            _container._reap_orphaned_lifecycle_images("docker", own_pid)
        assert mock_run.call_count == 2

    def test_skips_image_with_missing_or_unparseable_label(self):
        tag = "agency/lifecycle-someagent:latest"
        fake_run = self._dispatcher(images_lines=[tag], labels={tag: None})
        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            _container._reap_orphaned_lifecycle_images("docker", os.getpid())
        assert mock_run.call_count == 2, "must not attempt to delete on unparseable evidence"
        assert not any("rmi" in c.args[0] for c in mock_run.call_args_list)

    def test_skips_non_lifecycle_prefixed_image_even_with_dead_label(self):
        dead_pid = 2**31 - 1
        tag = "agency/ckpt-restore-abc123:latest"
        fake_run = self._dispatcher(images_lines=[tag], labels={tag: dead_pid})
        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            _container._reap_orphaned_lifecycle_images("docker", os.getpid())
        # Skipped before ever inspecting it -- only the images list call.
        assert mock_run.call_count == 1, "only agency/lifecycle-* images are in scope"

    def test_skips_image_still_in_use_by_a_container_despite_dead_label(self):
        """The core safety guard: a stale label (e.g. from a checkpoint
        restored via agent.load(), which DOES preserve the original,
        possibly-foreign owner's label) must not cause an image a live
        container is currently running from to be deleted."""
        dead_pid = 2**31 - 1
        tag = "agency/lifecycle-someagent:latest"
        fake_run = self._dispatcher(
            images_lines=[tag], labels={tag: dead_pid}, ancestor_lines=["somecontainerid"]
        )
        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            _container._reap_orphaned_lifecycle_images("docker", os.getpid())
        assert not any("rmi" in c.args[0] for c in mock_run.call_args_list)

    def test_ancestor_check_failure_skips_rather_than_deletes(self):
        """If the safety check itself can't be confirmed (non-zero
        returncode or an exception), the image must be left alone -- never
        deleted on missing evidence."""
        dead_pid = 2**31 - 1
        tag = "agency/lifecycle-someagent:latest"

        def fake_run(args, **kwargs):
            if "images" in args:
                return self._fake_result(stdout=f"{tag}\n")
            if "inspect" in args:
                return self._fake_result(stdout=str(dead_pid))
            if "ps" in args:
                return self._fake_result(returncode=1)
            raise AssertionError(f"unexpected call: {args}")

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            _container._reap_orphaned_lifecycle_images("docker", os.getpid())
        assert not any("rmi" in c.args[0] for c in mock_run.call_args_list)

    def test_images_scan_failure_is_a_noop(self):
        fake_run = self._dispatcher(images_rc=1)
        with patch("subprocess.run", side_effect=fake_run):
            _container._reap_orphaned_lifecycle_images("docker", os.getpid())  # must not raise

    def test_no_images_found_is_a_noop(self):
        fake_run = self._dispatcher(images_lines=[])
        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            _container._reap_orphaned_lifecycle_images("docker", os.getpid())
        assert mock_run.call_count == 1

    def test_real_scan_against_live_daemon_finds_no_broken_template(self):
        """Regression test for the exact bug this class's docstring
        describes: run the real `docker images --filter ... --format
        {{.Repository}}:{{.Tag}}` command (no mocking) and confirm it
        doesn't fail -- this is the specific call whose broken
        `--format` string previously made returncode != 0 look like "no
        orphans found" instead of "the command itself is malformed"."""
        if subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode != 0:
            pytest.skip("Docker daemon not reachable")
        result = subprocess.run(
            [
                "docker",
                "images",
                "--filter",
                "label=agency.owner_pid",
                "--format",
                "{{.Repository}}:{{.Tag}}",
            ],
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")


class TestRelabelOwnerPid:
    """_ContainerBackendBase.relabel_owner_pid() -- used by agent.py's
    save()/load() to scrub/restamp the owner_pid label on a checkpoint's
    embedded image (see that method's docstring). Builds its own throwaway
    container via `docker create` (never started) purely to give
    `docker commit --change` something to commit from."""

    def _fake_completed(self, stdout=b""):
        result = MagicMock()
        result.returncode = 0
        result.stdout = stdout
        return result

    def test_stamps_given_pid_via_create_then_commit(self):
        calls = []

        def fake_run(args, **kwargs):
            calls.append(list(args))
            if "create" in args:
                return self._fake_completed(stdout=b"fakecontainerid123\n")
            return self._fake_completed()

        with patch(
            "agency.agsandbox_backends.container.get_container_runtime", return_value="docker"
        ):
            with patch("subprocess.run", side_effect=fake_run):
                _container._ContainerBackendBase.relabel_owner_pid("agency/ckpt-foo", 4242, 30)

        create_calls = [c for c in calls if "create" in c]
        commit_calls = [c for c in calls if "commit" in c]
        rm_calls = [c for c in calls if c[1:3] == ["rm", "-f"]]
        assert create_calls and create_calls[0][-1] == "agency/ckpt-foo"
        assert commit_calls, calls
        assert "--change" in commit_calls[0]
        change_idx = commit_calls[0].index("--change")
        assert commit_calls[0][change_idx + 1] == "LABEL agency.owner_pid=4242"
        assert commit_calls[0][-2] == "fakecontainerid123"  # the created container's id
        assert commit_calls[0][-1] == "agency/ckpt-foo"
        assert rm_calls and rm_calls[0][-1] == "fakecontainerid123", (
            "the throwaway container must always be cleaned up"
        )

    def test_none_pid_clears_the_label(self):
        calls = []

        def fake_run(args, **kwargs):
            calls.append(list(args))
            if "create" in args:
                return self._fake_completed(stdout=b"cid\n")
            return self._fake_completed()

        with patch(
            "agency.agsandbox_backends.container.get_container_runtime", return_value="docker"
        ):
            with patch("subprocess.run", side_effect=fake_run):
                _container._ContainerBackendBase.relabel_owner_pid("agency/ckpt-foo", None, 30)

        commit_calls = [c for c in calls if "commit" in c]
        change_idx = commit_calls[0].index("--change")
        assert commit_calls[0][change_idx + 1] == "LABEL agency.owner_pid="

    def test_temp_container_removed_even_when_commit_fails(self):
        import subprocess as _sp

        calls = []

        def fake_run(args, **kwargs):
            calls.append(list(args))
            if "create" in args:
                return self._fake_completed(stdout=b"cid\n")
            if "commit" in args:
                raise _sp.CalledProcessError(1, args)
            return self._fake_completed()

        with patch(
            "agency.agsandbox_backends.container.get_container_runtime", return_value="docker"
        ):
            with patch("subprocess.run", side_effect=fake_run):
                with pytest.raises(_sp.CalledProcessError):
                    _container._ContainerBackendBase.relabel_owner_pid("agency/ckpt-foo", 1, 30)

        rm_calls = [c for c in calls if c[1:3] == ["rm", "-f"]]
        assert rm_calls and rm_calls[0][-1] == "cid"


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


class TestRun:
    """_run()'s own defense against a docker/podman CLI child stuck in
    uninterruptible kernel sleep (D-state) -- a real incident in production
    showed a wedged daemon leave several agents' conversations frozen for
    HOURS, because subprocess.run(timeout=...)'s own kill()+wait() sequence
    is not actually a reliable ceiling: SIGKILL cannot preempt a D-state
    process, so process.wait() after the kill can itself block forever (see
    run_with_unkillable_child_grace()'s docstring in agsandbox_backends/base.py).

    All of these mock subprocess.run directly (never _run itself, unlike
    TestRunWithConflictRetryHooks above) so the real threading/timeout logic
    inside _run() actually executes. timeout=/unkillable_child_grace_s are
    both overridden to ~0.05-0.1s so the "give up" tests complete in well
    under a second while still exercising the real code path.
    """

    def _sb(self):
        from agency.agsandbox_backends.podman import _PodmanBackend

        return _PodmanBackend(
            "agent",
            name="podman-run-test",
            checkpoint_image=None,
            base_image="img",
            mounts={},
            agconfig=None,
        )

    def test_fast_path_returns_the_real_completed_process(self):
        sb = self._sb()
        fake_result = subprocess.CompletedProcess(["podman", "x"], 0, b"out", b"")
        with patch("subprocess.run", return_value=fake_result) as mock_run:
            result = sb._run(["podman", "x"], timeout=5)
        mock_run.assert_called_once()
        assert result is fake_result

    def test_check_true_failure_raises_runtime_error_with_existing_message_format(self):
        sb = self._sb()
        err = subprocess.CalledProcessError(1, ["podman", "x"], output=b"", stderr=b"boom")
        with patch("subprocess.run", side_effect=err):
            with pytest.raises(RuntimeError, match=r"podman x failed \(exit 1\): boom"):
                sb._run(["podman", "x"], check=True, timeout=5)

    def test_ordinary_fast_timeout_still_raises_timeout_expired_unchanged(self):
        sb = self._sb()
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(["podman", "x"], 5),
        ):
            with pytest.raises(subprocess.TimeoutExpired):
                sb._run(["podman", "x"], timeout=5)

    def test_gives_up_instead_of_hanging_when_child_outlives_timeout_and_grace(self):
        """Simulates the actual incident: subprocess.run itself never returns
        (standing in for a real child stuck in D-state past SIGKILL). _run()
        must still return control to the caller -- not hang -- once
        timeout + unkillable_child_grace_s has elapsed."""
        import time as _time

        sb = self._sb()
        with patch.object(_container.AgSandboxBackendFields, "unkillable_child_grace_s", 0.05):
            with patch("subprocess.run", side_effect=lambda *a, **k: _time.sleep(10)):
                start = _time.monotonic()
                with pytest.raises(subprocess.TimeoutExpired):
                    sb._run(["podman", "x"], timeout=0.05)
                elapsed = _time.monotonic() - start
        # Must give up close to timeout + grace_s (0.1s total), not hang for
        # anywhere near the simulated child's real 10s "hang".
        assert elapsed < 5

    def test_semaphore_is_released_immediately_on_give_up_not_leaked(self):
        """Regression coverage for the amplification risk: a wedged call must
        not also starve every OTHER sandbox's ability to make a docker/podman
        call by holding the shared concurrency semaphore forever."""
        import time as _time

        sb = self._sb()
        sem = _container._get_docker_semaphore()
        with patch.object(_container.AgSandboxBackendFields, "unkillable_child_grace_s", 0.05):
            with patch("subprocess.run", side_effect=lambda *a, **k: _time.sleep(10)):
                with pytest.raises(subprocess.TimeoutExpired):
                    sb._run(["podman", "x"], timeout=0.05)
        # The slot must be free again immediately -- a non-blocking acquire
        # succeeds right away if _run() released it on give-up.
        acquired = sem.acquire(blocking=False)
        assert acquired, "docker semaphore slot was not released on give-up"
        if acquired:
            sem.release()


# ---------------------------------------------------------------------------
# _own_host_pids() -- host PIDs currently inside this container via
# docker/podman top (exec sessions are siblings of init, not /proc children).
# GPU release runs after stop()/destroy()'s own container-removal step has
# already completed synchronously -- no separate polling/is_clear check.
#
# Syntax is NOT interchangeable between runtimes:
#   docker: `docker top <name> -eo pid` -- real ps(1) flags, host PIDs.
#   podman: `podman top <name> hpid` -- podman's own positional descriptor.
#     Using ps(1)-style `-eo`/`-o` flags on podman silently takes a DIFFERENT
#     code path (runs a real `ps` inside the container's own namespace) and
#     would report container-LOCAL pids -- exactly as wrong as the old bug,
#     just via a different mechanism. Podman's plain `pid` descriptor is
#     also container-local; only `hpid` maps to the real host PID.
# ---------------------------------------------------------------------------


class TestOwnHostPidsDocker:
    def _sb(self):
        from agency.agsandbox_backends.docker import _DockerBackend

        return _DockerBackend(
            "agent",
            name="own-host-pids-test",
            checkpoint_image=None,
            base_image="img",
            mounts={},
            agconfig=None,
        )

    def test_uses_docker_top_dash_eo_pid(self):
        """Must use real ps(1)-style -eo pid syntax for docker, not podman's
        positional-descriptor syntax."""
        sb = self._sb()
        ok = MagicMock()
        ok.returncode = 0
        ok.stdout = b"PID\n100\n"
        with patch("subprocess.run", return_value=ok) as run:
            sb._own_host_pids()
        args = run.call_args.args[0]
        assert args[:2] == ["docker", "top"]
        assert args[-2:] == ["-eo", "pid"]

    def test_parses_pids_skipping_header(self):
        sb = self._sb()
        ok = MagicMock()
        ok.returncode = 0
        ok.stdout = b"PID\n100\n101\n102\n"
        with patch("subprocess.run", return_value=ok):
            assert sb._own_host_pids() == {100, 101, 102}

    def test_includes_exec_spawned_sibling_pids(self):
        """The exact regression case: docker top lists the init process AND
        a separately-exec'd sibling process together, in one call -- no
        ancestry relationship between them required."""
        sb = self._sb()
        ok = MagicMock()
        ok.returncode = 0
        # init(100) and an unrelated exec'd sibling(200) -- 200 is NOT a
        # child of 100 in this scenario, matching the real bug.
        ok.stdout = b"PID\n100\n200\n"
        with patch("subprocess.run", return_value=ok):
            assert sb._own_host_pids() == {100, 200}

    def test_returns_empty_when_top_fails(self):
        sb = self._sb()
        fail = MagicMock()
        fail.returncode = 1
        with patch("subprocess.run", return_value=fail):
            assert sb._own_host_pids() == set()

    def test_returns_empty_when_only_header_present(self):
        """No processes (container gone/just created) -- header line alone
        must not be misparsed as a pid."""
        sb = self._sb()
        ok = MagicMock()
        ok.returncode = 0
        ok.stdout = b"PID\n"
        with patch("subprocess.run", return_value=ok):
            assert sb._own_host_pids() == set()


class TestOwnHostPidsPodman:
    def _sb(self):
        from agency.agsandbox_backends.podman import _PodmanBackend

        return _PodmanBackend(
            "agent",
            name="own-host-pids-test",
            checkpoint_image=None,
            base_image="img",
            mounts={},
            agconfig=None,
        )

    def test_uses_podman_hpid_not_ps_style_flags(self):
        """Must use podman's own positional `hpid` descriptor -- NOT `-eo`/
        `-o`, which silently switches podman to running a real `ps` inside
        the container's own namespace and reports container-local pids."""
        sb = self._sb()
        ok = MagicMock()
        ok.returncode = 0
        ok.stdout = b"HPID\n100\n"
        with patch("subprocess.run", return_value=ok) as run:
            sb._own_host_pids()
        args = run.call_args.args[0]
        assert args[:2] == ["podman", "top"]
        assert "hpid" in args
        assert "-eo" not in args
        assert "-o" not in args

    def test_parses_hpids_skipping_header(self):
        sb = self._sb()
        ok = MagicMock()
        ok.returncode = 0
        ok.stdout = b"HPID\n100\n101\n"
        with patch("subprocess.run", return_value=ok):
            assert sb._own_host_pids() == {100, 101}

    def test_includes_exec_spawned_sibling_pids(self):
        sb = self._sb()
        ok = MagicMock()
        ok.returncode = 0
        ok.stdout = b"HPID\n100\n200\n"
        with patch("subprocess.run", return_value=ok):
            assert sb._own_host_pids() == {100, 200}

    def test_returns_empty_when_top_fails(self):
        sb = self._sb()
        fail = MagicMock()
        fail.returncode = 1
        with patch("subprocess.run", return_value=fail):
            assert sb._own_host_pids() == set()

    def test_returns_empty_when_only_header_present(self):
        """No processes (container gone/just created) -- header line alone
        must not be misparsed as a pid."""
        sb = self._sb()
        ok = MagicMock()
        ok.returncode = 0
        ok.stdout = b"HPID\n"
        with patch("subprocess.run", return_value=ok):
            assert sb._own_host_pids() == set()

    def test_ignores_a_plain_pid_style_header_if_ever_returned(self):
        """Defensive: even if some podman version's header text differs from
        the exact 'HPID' string, the header-skip must be positional (skip
        line 0 unconditionally), not a string match against 'HPID' -- a
        string-match approach would silently misparse a differently-worded
        header as a real pid if it happened to be a non-digit string, but
        could just as easily mis-skip a real numeric first pid if the header
        were ever blank. This pins the current positional behavior."""
        sb = self._sb()
        ok = MagicMock()
        ok.returncode = 0
        ok.stdout = b"Some Other Header Text\n100\n101\n"
        with patch("subprocess.run", return_value=ok):
            assert sb._own_host_pids() == {100, 101}


# ---------------------------------------------------------------------------
# _gpu_flags(runtime) -- regression coverage for:
# (1) Docker's ``--gpus all`` was used unconditionally for Podman too. Podman
#     accepts that flag without erroring but never mounts the NVIDIA
#     driver/devices for it, so a container started that way silently has
#     zero GPU access.
# (2) A container created (e.g. via a bash/read_file/write_file call) before
#     reserve_gpu() was ever called on its sandbox got zero GPU devices at
#     `docker/podman run` time, and neither runtime supports hot-attaching a
#     device afterward -- so that container was permanently stuck without
#     GPU access even after reserve_gpu() ran. EVERY GPU on the host is now
#     attached to EVERY container unconditionally, regardless of
#     reserve_gpu() state, removing that ordering dependency entirely (and,
#     as a side effect, still letting stop()/start() hand a resumed
#     container a *different* physical GPU without recreating it). The
#     accepted trade-off: CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES becomes
#     the ONLY restriction keeping a GPU-reserving sandbox off other GPUs --
#     see _gpu_flags()'s docstring in container.py for the full reasoning.
# No real docker/podman required here -- detect_gpus()/shutil.which() are
# mocked, and /dev/dri is faked via a real tmp directory for the AMD branch.
# ---------------------------------------------------------------------------


class TestGpuFlagsPerRuntime:
    def setup_method(self):
        # _gpu_kind_cache/_amd_render_node_paths_cache are module-level,
        # process-lifetime caches keyed by runtime -- clear them so each test
        # observes only its own mocked detect_gpus()/which()/dev-listing.
        _container._gpu_kind_cache.clear()
        _container._amd_render_node_paths_cache = None

    def test_no_gpu_detected_returns_empty_for_both_runtimes(self):
        with patch.object(_container, "detect_gpus", return_value=[]):
            assert _container._gpu_flags("docker") == []
            assert _container._gpu_flags("podman") == []

    def test_docker_attaches_all_gpus_for_nvidia(self):
        with patch.object(_container, "detect_gpus", return_value=[0, 1]):
            with patch.object(_container.shutil, "which", return_value="/usr/bin/nvidia-smi"):
                assert _container._gpu_flags("docker") == ["--gpus", "all"]

    def test_podman_attaches_all_gpus_via_cdi_not_docker_syntax(self):
        """The original regression: Podman must NOT get Docker's --gpus flag
        -- it silently accepts it without mounting the driver (confirmed
        against a real host: `podman run --gpus all ... nvidia-smi` prints
        "WARNING: The NVIDIA Driver was not detected" and nvidia-smi isn't
        even on PATH), so it needs the CDI device syntax instead."""
        with patch.object(_container, "detect_gpus", return_value=[0, 1]):
            with patch.object(_container.shutil, "which", return_value="/usr/bin/nvidia-smi"):
                flags = _container._gpu_flags("podman")
        assert flags == ["--device", "nvidia.com/gpu=all"]
        assert "--gpus" not in flags

    def test_docker_and_podman_flags_differ_for_nvidia(self):
        with patch.object(_container, "detect_gpus", return_value=[0, 1]):
            with patch.object(_container.shutil, "which", return_value="/usr/bin/nvidia-smi"):
                docker_flags = _container._gpu_flags("docker")
                podman_flags = _container._gpu_flags("podman")
        assert docker_flags != podman_flags
        assert docker_flags == ["--gpus", "all"]
        assert podman_flags == ["--device", "nvidia.com/gpu=all"]

    def test_rocm_flags_attach_every_render_node(self, tmp_path):
        """AMD: /dev/kfd (shared control device) plus EVERY renderD* node
        found -- there's no single "all" flag for ROCm the way `--gpus
        all`/CDI `=all` covers NVIDIA, so each node is attached explicitly.
        Identical for both runtimes (only the NVIDIA branch differs by
        runtime), and identical regardless of how many GPUs detect_gpus()
        reports (every render node found is attached either way)."""
        dri = tmp_path / "dri"
        dri.mkdir()
        (dri / "renderD128").touch()
        (dri / "renderD129").touch()
        (dri / "card0").touch()  # not a render node -- must never be selected
        with patch.object(_container, "detect_gpus", return_value=[0, 1]):
            with patch.object(_container.shutil, "which", return_value=None):
                with patch.object(
                    _container, "amd_render_node_paths_by_pci_bus", return_value=None
                ):
                    with patch.object(
                        _container, "Path", lambda p: dri if p == "/dev/dri" else Path(p)
                    ):
                        docker_flags = _container._gpu_flags("docker")
                        podman_flags = _container._gpu_flags("podman")
        expected = [
            "--device",
            "/dev/kfd",
            "--device",
            str(dri / "renderD128"),
            "--device",
            str(dri / "renderD129"),
        ]
        assert docker_flags == expected
        assert podman_flags == expected

    def test_rocm_single_render_node_present(self, tmp_path):
        """Only one render node on disk -- attach just that one plus the
        shared control device, not an empty/missing second entry."""
        dri = tmp_path / "dri"
        dri.mkdir()
        (dri / "renderD128").touch()
        with patch.object(_container, "detect_gpus", return_value=[0]):
            with patch.object(_container.shutil, "which", return_value=None):
                with patch.object(
                    _container, "amd_render_node_paths_by_pci_bus", return_value=None
                ):
                    with patch.object(
                        _container, "Path", lambda p: dri if p == "/dev/dri" else Path(p)
                    ):
                        flags = _container._gpu_flags("docker")
        assert flags == ["--device", "/dev/kfd", "--device", str(dri / "renderD128")]

    def test_rocm_flags_use_pci_bus_ordering_when_available(self, tmp_path):
        """When amd_render_node_paths_by_pci_bus() successfully builds a
        mapping, _gpu_flags() must attach the nodes in ITS ordering, not
        naive sorted /dev/dri order -- this is the actual fix: on real 8x
        MI350X hardware the two disagree for every GPU (see agresources.py's
        amd_render_node_paths_by_pci_bus docstring). Every GPU is attached
        either way now, but the ORDER the flags list them in still follows
        the PCI-bus mapping when available."""
        dri = tmp_path / "dri"
        dri.mkdir()
        (dri / "renderD128").touch()
        (dri / "renderD129").touch()
        pci_ordered = [str(dri / "renderD129"), str(dri / "renderD128")]  # reversed
        with patch.object(_container, "detect_gpus", return_value=[0, 1]):
            with patch.object(_container.shutil, "which", return_value=None):
                with patch.object(
                    _container, "amd_render_node_paths_by_pci_bus", return_value=pci_ordered
                ):
                    with patch.object(
                        _container, "Path", lambda p: dri if p == "/dev/dri" else Path(p)
                    ):
                        flags = _container._gpu_flags("docker")
        assert flags == [
            "--device",
            "/dev/kfd",
            "--device",
            str(dri / "renderD129"),
            "--device",
            str(dri / "renderD128"),
        ]

    def test_backend_construction_no_longer_eagerly_caches_gpu_flags(self):
        """_gpu_flags() now depends on the per-sandbox leased gpu_id, which
        isn't known at construction time (it's acquired lazily, right before
        the container is actually created in _ensure_started()) -- so, unlike
        the old design, construction must NOT eagerly call _gpu_flags() or
        cache a fixed flag list on the instance."""
        from agency.agsandbox_backends.podman import _PodmanBackend

        with patch.object(_container, "_gpu_flags", return_value=["sentinel"]) as gpu_flags_mock:
            sb = _PodmanBackend(
                "agent",
                name="podman-gpu-ctor-test",
                checkpoint_image=None,
                base_image="img",
                mounts={},
                agconfig=None,
            )
        gpu_flags_mock.assert_not_called()
        assert "_gpu_flags" not in vars(sb)


class TestEnsureStartedAttachesGpuRegardlessOfReserveOrder:
    """Regression coverage for the bug this fixes: _gpu_flags() used to be
    gated on `self._gpu_virtual` (whether reserve_gpu() had been called
    *before* the container was created), so a container created via a first
    bash/read_file/write_file call before reserve_gpu() ran was permanently
    stuck without GPU device access -- reserve_gpu() could flip the
    sandbox's own flag afterward, but neither docker nor podman can
    hot-attach a device to an already-running container. `_gpu_flags()` is
    now called unconditionally, so the `docker/podman run` invocation must
    include GPU flags even when `_gpu_virtual` is still False at the moment
    the container is actually created."""

    def _sb(self):
        from agency.agsandbox_backends.podman import _PodmanBackend

        return _PodmanBackend(
            "agent",
            name="podman-gpu-order-test",
            checkpoint_image=None,
            base_image="img",
            mounts={},
            agconfig=None,
        )

    def test_run_cmd_includes_gpu_flags_when_container_created_before_reserve_gpu(self):
        sb = self._sb()
        assert sb._gpu_virtual is False  # reserve_gpu() has NOT been called yet

        with patch.object(_container, "_gpu_flags", return_value=["--device", "sentinel"]) as gf:
            with patch.object(sb, "_inspect_container_state", return_value=(False, "", None)):
                with patch.object(sb, "_acquire_runtime_slot"):
                    with patch.object(sb, "_resolve_image", return_value="img"):
                        with patch.object(sb, "_cfs_supported", return_value=False):
                            with patch.object(sb, "_run_with_conflict_retry") as run_retry:
                                with patch.object(sb, "_run"):
                                    with patch.object(
                                        sb, "_snapshot_pids_started", return_value=set()
                                    ):
                                        sb._ensure_started()

        gf.assert_called_once_with(sb._runtime)
        run_cmd = run_retry.call_args_list[0].args[0]
        assert "--device" in run_cmd and "sentinel" in run_cmd, (
            f"expected GPU flags in the run command even though reserve_gpu() was never "
            f"called before container creation: {run_cmd}"
        )

    def test_docker_run_uses_profile_workload_cgroup_parent(self):
        from agency.agsandbox_backends.docker import _DockerBackend

        sb = _DockerBackend(
            "agent",
            name="docker-profile-cgroup-test",
            checkpoint_image=None,
            base_image="img",
            mounts={},
            agconfig=None,
        )
        with patch.object(_container, "_gpu_flags", return_value=[]):
            with patch.object(
                _container.agprof,
                "container_cgroup_parent",
                return_value="agprof-12ab.slice",
            ):
                with patch.object(sb, "_inspect_container_state", return_value=(False, "", None)):
                    with patch.object(sb, "_acquire_runtime_slot"):
                        with patch.object(sb, "_resolve_image", return_value="img"):
                            with patch.object(sb, "_cfs_supported", return_value=False):
                                with patch.object(sb, "_run_with_conflict_retry") as run_retry:
                                    with patch.object(sb, "_run"):
                                        with patch.object(
                                            sb, "_snapshot_pids_started", return_value=set()
                                        ):
                                            sb._ensure_started()

        run_cmd = run_retry.call_args_list[0].args[0]
        assert "--cgroup-parent=agprof-12ab.slice" in run_cmd


def _host_rocm_available() -> bool:
    try:
        return (
            subprocess.run(["rocm-smi", "--showid"], capture_output=True, timeout=10).returncode
            == 0
        )
    except Exception:
        return False


rocm_hardware = pytest.mark.skipif(
    not _host_rocm_available(), reason="No real ROCm/AMD GPU hardware available"
)


class TestAmdRenderNodeLiveHardware:
    """Runs the real (unmocked) AMD GPU-to-render-node scoping against
    actual ROCm hardware -- no mocks anywhere, unlike TestGpuFlagsPerRuntime
    above. Skipped automatically without a real rocm-smi + AMD GPU(s).

    This is the check that actually caught the bug being regression-tested
    here: on an 8x MI350X host, naive sorted /dev/dri/renderD* order did
    NOT correspond to rocm-smi's GPU index for ANY of the 8 GPUs (each GPU
    there exposes itself plus 7 XCD/compute-partition sibling render nodes
    -- 64 nodes total -- and even the primary node's number doesn't sort in
    GPU-index order; e.g. GPU 3's real node was the numerically LOWEST of
    the 64 present, not the 4th). These tests independently recompute the
    "ground truth" GPU-to-render-node mapping from rocm-smi --showbus and
    /sys/class/drm/*/device -- without going through
    amd_render_node_paths_by_pci_bus itself -- and assert the real
    _gpu_flags()/_amd_render_node_paths() output agrees with it for every
    real GPU on this host.
    """

    _PCI_BUS_RE = re.compile(r"([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f])$")

    def _ground_truth_bus_by_gpu_id(self) -> "dict[int, str]":
        result = subprocess.run(
            ["rocm-smi", "--showbus", "--csv"], capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0, f"rocm-smi --showbus failed: {result.stderr!r}"
        mapping = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("device"):
                continue
            card, bus = (c.strip() for c in line.split(",")[:2])
            mapping[int(card[4:])] = bus.lower()
        return mapping

    def _real_render_node_bus(self, path: str) -> "str | None":
        name = os.path.basename(path)
        target = os.path.realpath(f"/sys/class/drm/{name}/device")
        match = self._PCI_BUS_RE.search(target)
        return match.group(1) if match else None

    @rocm_hardware
    def test_amd_render_node_paths_match_gpu_pci_bus_on_real_hardware(self):
        _container._gpu_kind_cache.clear()
        _container._amd_render_node_paths_cache = None
        bus_by_gpu_id = self._ground_truth_bus_by_gpu_id()
        assert bus_by_gpu_id, "expected at least one real AMD GPU"

        render_nodes = _container._amd_render_node_paths()
        assert len(render_nodes) == len(bus_by_gpu_id), (
            f"expected one resolved render node per real GPU ({len(bus_by_gpu_id)}), "
            f"got {len(render_nodes)}: {render_nodes}"
        )

        for gpu_id, expected_bus in bus_by_gpu_id.items():
            actual_bus = self._real_render_node_bus(render_nodes[gpu_id])
            assert actual_bus == expected_bus, (
                f"gpu_id={gpu_id}: _amd_render_node_paths() picked "
                f"{render_nodes[gpu_id]} (bus={actual_bus}), but rocm-smi says "
                f"GPU {gpu_id} is actually on bus {expected_bus}"
            )

        # No two GPUs should ever be scoped to the same render node.
        assert len(set(render_nodes)) == len(render_nodes)

    @rocm_hardware
    def test_gpu_flags_attach_every_real_render_node(self):
        """_gpu_flags() attaches every render node _amd_render_node_paths()
        resolves unconditionally -- both so the hibernate model can hand a
        resumed container a different physical GPU without recreating it,
        and so a container created before reserve_gpu() is ever called
        still gets device access (see _gpu_flags()'s docstring in
        container.py)."""
        _container._gpu_kind_cache.clear()
        _container._amd_render_node_paths_cache = None
        bus_by_gpu_id = self._ground_truth_bus_by_gpu_id()
        assert bus_by_gpu_id, "expected at least one real AMD GPU"

        flags = _container._gpu_flags("docker")
        assert flags[:2] == ["--device", "/dev/kfd"]
        device_flags = flags[2:]
        assert device_flags[::2] == ["--device"] * len(bus_by_gpu_id), (
            f"expected one --device pair per real GPU ({len(bus_by_gpu_id)}): {flags}"
        )
        device_paths = device_flags[1::2]
        assert len(set(device_paths)) == len(device_paths), (
            f"no two GPUs should ever resolve to the same render node: {device_paths}"
        )
        for gpu_id, expected_bus in bus_by_gpu_id.items():
            actual_bus = self._real_render_node_bus(device_paths[gpu_id])
            assert actual_bus == expected_bus, (
                f"gpu_id={gpu_id}: _gpu_flags() attached {device_paths[gpu_id]} "
                f"(bus={actual_bus}), but rocm-smi says GPU {gpu_id} is on bus {expected_bus}"
            )


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


podman_only = pytest.mark.skipif(not _podman_available(), reason="Podman daemon not reachable")


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


docker_only = pytest.mark.skipif(not _docker_available(), reason="Docker daemon not reachable")


class TestOwnHostPidsRealContainerIntegration:
    """_own_host_pids() against a REAL container -- no GPU/special image
    required, just a plain generic image, unlike the GPU integration classes
    below.

    The container is started with just `tail -f /dev/null` as init (matching
    _ensure_started()'s real invocation), and the "work" is spawned via a
    SEPARATE `docker exec -d` call -- matching exactly how _container_exec()
    invokes every real command in production, including the entire harness
    workload -- rather than a background job within the initial `run`
    command. This is deliberate: an earlier version of this test used
    `sh -c "sleep 30 & sleep 30"` (background jobs, children of the
    container's own init/shell process), which happened to be reachable by
    the old /proc-child-walk implementation and therefore never caught the
    real regression -- an exec'd process is a SIBLING under the runtime's
    supervisor, not a descendant of init, and the old implementation only
    ever found {init_pid} in real production use as a result. This test's
    shape is what would have caught that regression.
    """

    @docker_only
    def test_includes_a_separately_execd_sibling_process(self):
        import uuid
        from agency.agsandbox_backends.docker import _DockerBackend

        name = f"own-host-pids-real-{uuid.uuid4().hex[:8]}"
        try:
            subprocess.run(
                ["docker", "run", "-d", "--name", name, "alpine", "tail", "-f", "/dev/null"],
                check=True,
                capture_output=True,
                timeout=30,
            )
            init_pid = int(
                subprocess.run(
                    ["docker", "inspect", "--format", "{{.State.Pid}}", name],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                ).stdout.strip()
            )
            # A SEPARATE exec call -- not a background job within `run` above.
            subprocess.run(
                ["docker", "exec", "-d", name, "sh", "-c", "sleep 60"],
                check=True,
                capture_output=True,
                timeout=30,
            )
            time.sleep(0.5)  # let the exec'd process actually start

            # Ground truth: the exec'd sleep is NOT a /proc child of init --
            # this is the exact condition that broke the old implementation.
            children = Path(f"/proc/{init_pid}/task/{init_pid}/children").read_text().split()
            assert children == [], (
                "test assumption broken: the exec'd process IS a /proc child of "
                "init on this host, so this test can't distinguish the fix from "
                "the old broken behavior"
            )

            sb = _DockerBackend(
                "agent",
                name=name,
                checkpoint_image=None,
                base_image="alpine",
                mounts={},
                agconfig=None,
            )
            result = sb._own_host_pids()

            top = subprocess.run(
                ["docker", "top", name, "-eo", "pid"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            expected = {
                int(line.strip()) for line in top.stdout.splitlines()[1:] if line.strip().isdigit()
            }
            assert len(expected) >= 2, "expected both init and the exec'd sleep in docker top"
            assert result == expected
            assert init_pid in result
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)


class TestOwnHostPidsRealPodmanContainerIntegration:
    """Podman counterpart of TestOwnHostPidsRealContainerIntegration --
    skipped on this dev box (no podman installed) but intended to run for
    real in CI, where podman IS available. Same shape as the Docker version:
    a separate `podman exec -d` call, not a background job within the
    initial `run`, since that's the exact distinction that mattered for the
    real regression (an exec'd process is a PID-namespace sibling, not a
    /proc descendant of init)."""

    @podman_only
    def test_includes_a_separately_execd_sibling_process(self):
        import uuid
        from agency.agsandbox_backends.podman import _PodmanBackend

        name = f"own-host-pids-real-podman-{uuid.uuid4().hex[:8]}"
        try:
            subprocess.run(
                ["podman", "run", "-d", "--name", name, "alpine", "tail", "-f", "/dev/null"],
                check=True,
                capture_output=True,
                timeout=30,
            )
            init_pid = int(
                subprocess.run(
                    ["podman", "inspect", "--format", "{{.State.Pid}}", name],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                ).stdout.strip()
            )
            # A SEPARATE exec call -- not a background job within `run` above.
            subprocess.run(
                ["podman", "exec", "-d", name, "sh", "-c", "sleep 60"],
                check=True,
                capture_output=True,
                timeout=30,
            )
            time.sleep(0.5)  # let the exec'd process actually start

            # Ground truth: the exec'd sleep is NOT a /proc child of init --
            # this is the exact condition that broke the old implementation.
            children = Path(f"/proc/{init_pid}/task/{init_pid}/children").read_text().split()
            assert children == [], (
                "test assumption broken: the exec'd process IS a /proc child of "
                "init on this host, so this test can't distinguish the fix from "
                "the old broken behavior"
            )

            sb = _PodmanBackend(
                "agent",
                name=name,
                checkpoint_image=None,
                base_image="alpine",
                mounts={},
                agconfig=None,
            )
            result = sb._own_host_pids()

            top = subprocess.run(
                ["podman", "top", name, "hpid"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            expected = {
                int(line.strip()) for line in top.stdout.splitlines()[1:] if line.strip().isdigit()
            }
            assert len(expected) >= 2, "expected both init and the exec'd sleep in podman top hpid"
            assert result == expected
            assert init_pid in result
        finally:
            subprocess.run(["podman", "rm", "-f", name], capture_output=True, timeout=30)


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


class TestCvdOverrideProtectionIntegration:
    """Regression test for the "agent hijacks its own GPU isolation" bug:
    base.py's exec() exports CUDA_VISIBLE_DEVICES=<leased physical id> fresh
    before every command, but a command that itself starts with
    "CUDA_VISIBLE_DEVICES=<other> ..." -- extremely common ML boilerplate
    (``CUDA_VISIBLE_DEVICES=0 python train.py``), and exactly what
    reserve_gpu's "always use cuda:0 inside your scripts" instruction is
    prone to being misread as -- shadows that export for its own child
    process under plain POSIX var=val-prefix semantics. The pool's semaphore
    bookkeeping still believes the sandbox holds the GPU it actually leased;
    the child process runs on whatever GPU the agent hardcoded instead.

    Uses a REAL container (no mocks) on both runtimes, going through the
    actual agSandbox/agResourcePool/reserve_gpu path an agent uses -- not raw
    ``docker run``/``podman run`` -- because the fix under test
    (``readonly CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES`` in base.py's
    exec()) lives in exactly that command-wrapping logic. Requires a real
    docker/podman daemon, a real NVIDIA GPU, and the local
    agency-sandbox:latest image -- skipped otherwise (see the docker_gpu/
    podman_gpu markers above).
    """

    def _check(self, backend: str) -> None:
        import uuid

        from agency.agconfig import agConfig
        from agency.agdata import agdata
        from agency.agresources import agResourcePool
        from agency.agsandbox import agSandbox
        from agency.agsandbox_backends import agSandboxBackendConfig
        from agency.tools import make_sandboxed_tools

        cfg = agConfig(agSandboxBackendConfig(backend=backend))
        sb = agSandbox(str(uuid.uuid4()), agconfig=cfg)
        pool = agResourcePool(mark_gpus=False)
        assert pool.gpus, "expected at least one real GPU to be detected on this host"
        tools = {t.name: t for t in make_sandboxed_tools(sb, pool)}
        try:
            tools["reserve_gpu"].fn(agdata())

            leased_out, rc = sb.exec("echo $CUDA_VISIBLE_DEVICES")
            assert rc == 0
            leased_gpu_id = leased_out.strip()
            assert leased_gpu_id.isdigit(), f"expected a leased GPU id, got {leased_out!r}"

            # An id that can never collide with a real leased id -- proves
            # this is the agent's hardcoded value winning, not a coincidence.
            hijack_attempt = "999"
            hijack_out, rc = sb.exec(
                f"CUDA_VISIBLE_DEVICES={hijack_attempt} python3 -c "
                "\"import os; print(os.environ['CUDA_VISIBLE_DEVICES'])\""
            )
            assert rc == 0, f"python3 should still run (using the real value): {hijack_out!r}"
            seen_gpu_id = hijack_out.strip().splitlines()[-1]
            assert seen_gpu_id == leased_gpu_id, (
                f"agent's inline CUDA_VISIBLE_DEVICES={hijack_attempt} override took "
                f"effect inside the container -- python saw {seen_gpu_id!r} instead of "
                f"the harness-leased {leased_gpu_id!r}. GPU isolation is not enforced."
            )
        finally:
            sb.destroy()

    @docker_gpu
    def test_docker_agent_cannot_hijack_cuda_visible_devices(self):
        self._check("docker")

    @podman_gpu
    def test_podman_agent_cannot_hijack_cuda_visible_devices(self):
        self._check("podman")
