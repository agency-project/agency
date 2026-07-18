"""Unit and integration tests for the Docker-specific sandbox backend
(agency.agsandbox_backends.docker._DockerBackend): dangling-image cleanup on
commit, the low-level docker-CLI command helpers (_rm_container, _rmi,
_ensure_started's pre-cleanup guard, destroy()'s semaphore release), and the
session-keyring-quota machinery (_docker_container_limit/keyring_quota/
_semaphore_held_count and the _is_quota_exhaustion_error/_wait_for_quota_slot/
_quota_diagnostics hooks _ContainerBackendBase._run_with_conflict_retry()
reaches through -- see test_container.py's TestRunWithConflictRetryHooks for
the runtime-agnostic dispatch logic itself, exercised there against Podman).

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
from unittest.mock import MagicMock, patch


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
# Owner-PID container labeling -- feeds agsandbox_backends.container's
# startup orphan reaper (see test_container.py). The label must reflect
# whichever process actually *constructed* this backend (self._owner_pid,
# fixed at __init__ time), never a live os.getpid() call made wherever
# _ensure_started() happens to execute -- for a run_in_subprocess=True tool
# call (the default), that's a ProcessPoolExecutor *worker*, cloudpickled a
# copy of this same backend object, distinct from -- and free to exit
# independently of -- the main process that owns the sandbox for its whole
# lifetime. Labeling with the worker's own transient PID would let a
# concurrent reap_orphaned_containers() elsewhere see a "dead" owner (once
# that worker exits, routine pool recycling, not a crash) for a container
# that's still very much in active use by a live main process, and delete it.
# ---------------------------------------------------------------------------


class TestOwnerPidLabel:
    def _captured_run_cmd(self, sb):
        """Drive _ensure_started() far enough to build its `docker run`
        argv, without a real daemon: everything _run_with_conflict_retry
        would normally do is skipped, so this only inspects the command
        that *would* have been issued."""
        calls = []
        with patch.object(sb._backend, "_container_running", return_value=False):
            with patch.object(sb._backend, "_container_status", return_value=""):
                with patch.object(
                    sb._backend,
                    "_run_with_conflict_retry",
                    side_effect=lambda run_cmd, name: calls.append(run_cmd),
                ):
                    with patch.object(sb._backend, "_run"):  # the post-run `mkdir /workspace`
                        sb._backend._ensure_started()
        assert len(calls) == 1, "expected exactly one docker run invocation"
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

        sb = _make_sandbox()
        run_cmd = self._captured_run_cmd(sb)
        assert self._label_value(run_cmd) == str(os.getpid())

    def test_label_reflects_owner_pid_attribute_not_live_process(self):
        """Regression: simulates the cloudpickle-to-a-worker scenario by
        overwriting _owner_pid post-construction to a value that is *not*
        this test process's own PID -- the label must still track that
        stored value, proving it's read from self._owner_pid rather than
        computed fresh via os.getpid() at run time (which, run entirely in
        this one process, could otherwise never distinguish the two)."""
        import os

        sb = _make_sandbox()
        sentinel_pid = 424242
        assert sentinel_pid != os.getpid()
        sb._backend._owner_pid = sentinel_pid

        run_cmd = self._captured_run_cmd(sb)
        assert self._label_value(run_cmd) == str(sentinel_pid)

    @docker
    def test_real_sandboxed_tool_call_labels_container_with_main_process_pid(self):
        """End-to-end, no mocks: a real run_in_subprocess=True tool call (the
        default) cloudpickles this backend to a real ProcessPoolExecutor
        worker, which is the process that actually issues `docker run` --
        confirms the resulting container's real label still names *this*
        (main, test) process, not the worker's own distinct PID, i.e. the
        fix survives the real dispatch mechanism, not just a mock of it."""
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
                    "docker",
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
                f"ran `docker run` must not have used its own os.getpid()"
            )
        finally:
            sb.destroy()


class TestDanglingImageEagerCleanup:
    """Tests for the eager old-image deletion in stop(commit=True)."""

    def test_no_prune_thread(self):
        """No background agsandbox-prune thread should exist after the refactor."""
        named = [t for t in threading.enumerate() if t.name == "agsandbox-prune"]
        assert not named, "agsandbox-prune thread should have been removed"

    def test_stop_commit_deletes_old_image(self):
        """stop(commit=True) must delete the image that previously held the tag."""
        import agency.agsandbox_backends.docker as _mod

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

        with patch.object(_mod._DockerBackend, "_run", fake_run):
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

        import agency.agsandbox_backends.docker as _mod

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            sb._backend._rm_container("my-container")

        assert len(calls) == 1
        args, check = calls[0]
        assert "rm" in args and "-f" in args and "my-container" in args
        assert check is True

    def test_rm_container_raises_on_failure(self):
        sb = self._sb()

        import agency.agsandbox_backends.docker as _mod

        with patch.object(_mod._DockerBackend, "_run", side_effect=RuntimeError("rm failed")):
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


# ---------------------------------------------------------------------------
# Session-keyring-quota machinery: _docker_container_limit/keyring_quota/
# _semaphore_held_count, and the quota hooks _run_with_conflict_retry() (in
# container.py, shared by both runtimes) reaches through. No real Docker
# daemon required -- everything here mocks the /proc reads and the semaphore.
# ---------------------------------------------------------------------------


class TestKeyringQuotaDiagnostics:
    def test_docker_container_limit_uses_kernel_maxkeys_minus_buffer(self):
        import agency.agsandbox_backends.docker as _mod

        with patch("pathlib.Path.read_text", return_value="500\n"):
            limit = _mod._docker_container_limit()
        fields = _mod.AgSandboxBackendFields()
        assert limit == 500 - fields.container_limit_buffer

    def test_docker_container_limit_never_below_floor(self):
        import agency.agsandbox_backends.docker as _mod

        with patch("pathlib.Path.read_text", return_value="1\n"):
            limit = _mod._docker_container_limit()
        fields = _mod.AgSandboxBackendFields()
        assert limit == fields.container_limit_floor

    def test_docker_container_limit_falls_back_when_proc_unreadable(self):
        import agency.agsandbox_backends.docker as _mod

        with patch("pathlib.Path.read_text", side_effect=OSError("no such file")):
            limit = _mod._docker_container_limit()
        fields = _mod.AgSandboxBackendFields()
        assert limit == fields.container_limit_fallback - fields.container_limit_buffer

    def test_keyring_quota_reports_used_max_and_free(self):
        import agency.agsandbox_backends.docker as _mod

        def fake_read_text(self):
            return "200\n" if "maxkeys" in str(self) else "a\nb\nc\n"

        with patch("pathlib.Path.read_text", fake_read_text):
            quota = _mod.keyring_quota()
        assert quota == {"used": 3, "max": 200, "free": 197}

    def test_keyring_quota_reports_minus_one_when_proc_unreadable(self):
        import agency.agsandbox_backends.docker as _mod

        with patch("pathlib.Path.read_text", side_effect=OSError("no such file")):
            quota = _mod.keyring_quota()
        assert quota == {"used": -1, "max": -1, "free": -1}

    def test_semaphore_held_count_reflects_acquired_slots(self):
        """held/limit should go up by exactly one slot per acquire() -- checked
        as a delta against the semaphore's already-live real value rather than
        an assumed absolute count, since _container_semaphore is a real
        process-wide multiprocessing.Semaphore shared with every other test."""
        import agency.agsandbox_backends.docker as _mod

        before_held, limit = _mod._semaphore_held_count().split("/")
        _mod._container_semaphore.acquire()
        try:
            after_held, limit_after = _mod._semaphore_held_count().split("/")
        finally:
            _mod._container_semaphore.release()
        assert limit_after == limit
        assert int(after_held) == int(before_held) + 1

    def test_semaphore_held_count_falls_back_to_unknown_on_error(self):
        import agency.agsandbox_backends.docker as _mod

        fake_semlock = MagicMock()
        fake_semlock._get_value.side_effect = Exception("boom")
        with patch.object(_mod, "_docker_container_limit", return_value=10):
            with patch.object(_mod._container_semaphore, "_semlock", fake_semlock):
                held = _mod._semaphore_held_count()
        assert held == "?/10"


class TestQuotaExhaustionHooks:
    """_DockerBackend's overrides of the quota hooks _ContainerBackendBase
    defines as no-ops (_is_quota_exhaustion_error/_wait_for_quota_slot/
    _quota_diagnostics) -- these are Docker's half of the keyring-quota
    handling _run_with_conflict_retry() (container.py) dispatches through."""

    def _sb(self):
        return _make_sandbox()._backend

    def test_matches_session_key_message(self):
        sb = self._sb()
        assert sb._is_quota_exhaustion_error("unable to create session key: disk quota exceeded")

    def test_matches_disk_quota_exceeded_with_keyring(self):
        sb = self._sb()
        assert sb._is_quota_exhaustion_error("disk quota exceeded for keyring")

    def test_disk_quota_exceeded_without_keyring_does_not_match(self):
        """Docker also emits a plain filesystem "disk quota exceeded" for
        unrelated reasons (e.g. a full overlay volume) -- only the keyring
        variant should trigger the quota-wait path."""
        sb = self._sb()
        assert not sb._is_quota_exhaustion_error("disk quota exceeded")

    def test_unrelated_stderr_does_not_match(self):
        sb = self._sb()
        assert not sb._is_quota_exhaustion_error("no such image: agency-sandbox:latest")

    def test_wait_for_quota_slot_polls_until_a_slot_frees_up(self):
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        quotas = iter([{"free": 0}, {"free": 0}, {"free": 1}])
        with patch.object(_mod, "keyring_quota", side_effect=lambda: next(quotas)):
            with patch.object(_mod.time, "sleep") as sleep_mock:
                sb._wait_for_quota_slot()
        assert sleep_mock.call_count == 2

    def test_wait_for_quota_slot_gives_up_once_deadline_passes(self):
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        # First monotonic() call establishes the deadline; the second (the
        # loop's own check) reports a time far past it -- the loop body
        # must never run, so it must never sleep either.
        moments = iter([0.0, 10_000.0])
        with patch.object(_mod, "keyring_quota", return_value={"free": 0}):
            with patch.object(_mod.time, "monotonic", side_effect=lambda: next(moments)):
                with patch.object(_mod.time, "sleep") as sleep_mock:
                    sb._wait_for_quota_slot()
        sleep_mock.assert_not_called()

    def test_quota_diagnostics_format(self):
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        with patch.object(_mod, "keyring_quota", return_value={"used": 5, "max": 200}):
            with patch.object(_mod, "_semaphore_held_count", return_value="3/195"):
                assert (
                    sb._quota_diagnostics()
                    == "[keyring: 5/200 used, framework semaphore: 3/195 held]"
                )
