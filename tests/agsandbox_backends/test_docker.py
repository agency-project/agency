"""Unit and integration tests for the Docker-specific sandbox backend
(agency.agsandbox_backends.docker._DockerBackend): dangling-image cleanup on
commit, and the low-level docker-CLI command helpers (_rm_container, _rmi,
_ensure_started's pre-cleanup guard, destroy()'s semaphore release).

The session-keyring-quota machinery (_keyring_container_limit/keyring_quota/
_semaphore_held_count) and the _is_quota_exhaustion_error/_wait_for_quota_slot/
_quota_diagnostics hooks are shared, runtime-agnostic code that now lives on
_ContainerBackendBase itself (agency.agsandbox_backends.container) -- see
test_container.py's TestKeyringQuotaDiagnostics, TestQuotaHooksSharedAcross
Runtimes, and TestRunWithConflictRetryHooks (exercised there against Podman,
specifically to prove docker and podman are subject to the identical kernel
quota).

Tests that need a real Docker daemon are marked with @pytest.mark.docker and
skipped automatically when Docker is unreachable.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from pathlib import Path

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
        """stop(commit=True) must delete the image that previously held the
        tag -- only possible on a squash cycle (a plain commit's result is
        always a child of the old image, so the runtime refuses to delete
        it; old_image_id is only looked up at all when this cycle squashes,
        see container.py's stop())."""
        import agency.agsandbox_backends.docker as _mod

        sb = _make_sandbox()
        sb._backend._commits_since_squash = sb._backend.checkpoint_squash_interval - 1

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
            with patch.object(sb._backend, "_container_running", return_value=True):
                with patch.object(sb._backend, "_gpu_virtual", False):
                    sb.stop(commit=True)

        rmi_calls = [a for a in run_calls if "rmi" in a]
        assert rmi_calls, "expected docker rmi call for old image"
        assert any(fake_old_id in " ".join(a) for a in rmi_calls), (
            f"rmi call did not reference old image ID; calls: {rmi_calls}"
        )

    def test_stop_commit_skips_rmi_when_no_old_image(self):
        """If the tag does not exist yet (first squash), no rmi call is made."""
        import agency.agsandbox_backends.docker as _mod

        sb = _make_sandbox()
        sb._backend._commits_since_squash = sb._backend.checkpoint_squash_interval - 1

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
            with patch.object(sb._backend, "_container_running", return_value=True):
                with patch.object(sb._backend, "_gpu_virtual", False):
                    sb.stop(commit=True)

        rmi_calls = [a for a in run_calls if "rmi" in a]
        assert not rmi_calls, "must not call rmi when there was no previous image"

    def test_plain_commit_cycle_skips_old_image_lookup_entirely(self):
        """A plain commit's result is ALWAYS a child of whatever it
        replaces -- the runtime will refuse to delete that old image no
        matter what, so there's no point even looking it up. A non-squash
        stop(commit=True) must issue zero old-image-lookup/ps/rmi calls
        (the doomed-to-fail cleanup this test originally guarded against),
        even though it now also does a `docker inspect ...RootFS.Layers`
        call for the diff accumulator (see TestCheckpointAccumulator) --
        that inspect is for a different purpose and is expected here."""
        import agency.agsandbox_backends.docker as _mod

        sb = _make_sandbox()
        sb._backend._commits_since_squash = 0  # far from the squash threshold
        calls = []

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            calls.append(list(args))
            return _FakeCompleted()

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with patch.object(sb._backend, "_container_running", return_value=True):
                with patch.object(sb._backend, "_gpu_virtual", False):
                    sb.stop(commit=True)

        assert not any("--format={{.Id}}" in c for c in calls), (
            f"must not do the old-image-lookup inspect on a plain commit: {calls}"
        )
        assert not any("ps" in c for c in calls), (
            f"must not check ancestor on a plain commit: {calls}"
        )
        assert not any("rmi" in c for c in calls), (
            f"must not attempt rmi on a plain commit: {calls}"
        )

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
        sb._backend._commits_since_squash = sb._backend.checkpoint_squash_interval - 1
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
                with patch.object(sb._backend, "_container_running", return_value=True):
                    with patch.object(sb._backend, "_gpu_virtual", False):
                        sb.stop(commit=True)  # must not raise
        finally:
            sys.stderr = old_stderr

        assert "WARNING" in captured.getvalue()
        assert fake_old_id in captured.getvalue()

    def test_old_image_ancestor_check_runs_after_container_removal(self):
        """Regression test: the ancestor-based "is the old image still in
        use" check must run AFTER `_rm_container`, not before. This
        container was itself `run` FROM the old image (restart-from-
        checkpoint), so checking before removal always finds THIS SAME
        container as a false-positive "still in use" match and never
        actually deletes anything -- a real, pre-existing bug confirmed
        live against a real docker daemon (see docs/agsandbox_backends/
        container.md's "Layer-depth squashing" section for how this
        surfaced). Only reachable on a squash cycle -- old_image_id is
        only looked up then (a plain commit's result can never actually
        free its parent, so there's no point checking)."""
        import agency.agsandbox_backends.docker as _mod

        sb = _make_sandbox()
        sb._backend._commits_since_squash = sb._backend.checkpoint_squash_interval - 1
        fake_old_id = "sha256:deadbeef0000"
        call_order = []

        class FakeCompleted:
            def __init__(self, stdout=b"", returncode=0):
                self.stdout = stdout
                self.returncode = returncode

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "inspect" in args:
                return FakeCompleted(stdout=fake_old_id.encode())
            if "rm" in args:
                call_order.append("rm")
                return FakeCompleted()
            if "ps" in args:
                call_order.append("ps")
                return FakeCompleted()  # empty -- "not in use"
            if "rmi" in args:
                call_order.append("rmi")
                return FakeCompleted()
            return FakeCompleted()

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with patch.object(sb._backend, "_container_running", return_value=True):
                with patch.object(sb._backend, "_gpu_virtual", False):
                    sb.stop(commit=True)

        assert call_order == ["rm", "ps", "rmi"], f"expected rm before ps/rmi, got {call_order}"

    def test_old_image_cleanup_skipped_when_rm_fails(self):
        """If _rm_container never succeeds, the container may genuinely
        still be running from the old image -- skip the ancestor check
        entirely rather than spending a docker call to reconfirm what
        rm's own failure already implies."""
        import agency.agsandbox_backends.docker as _mod

        sb = _make_sandbox()
        fake_old_id = "sha256:deadbeef0001"
        ps_or_rmi_called = []

        class FakeCompleted:
            def __init__(self, stdout=b"", returncode=0):
                self.stdout = stdout
                self.returncode = returncode

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "inspect" in args:
                return FakeCompleted(stdout=fake_old_id.encode())
            if "rm" in args:
                raise RuntimeError("rm failed")
            if "ps" in args or "rmi" in args:
                ps_or_rmi_called.append(list(args))
            return FakeCompleted()

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with patch.object(sb._backend, "_container_running", return_value=True):
                with patch.object(sb._backend, "_gpu_virtual", False):
                    with pytest.raises(RuntimeError):
                        sb.stop(commit=True)

        assert ps_or_rmi_called == []

    @docker
    def test_real_plain_commit_cycle_cannot_delete_its_parent(self):
        """A plain commit's result is a DIFF layer on top of whatever it
        replaces -- docker refuses to delete an image while a dependent
        child image still exists ("has dependent child images"), so two
        consecutive plain-commit cycles can NEVER free the first one's
        image. This is a real, unavoidable docker constraint, not a bug:
        confirmed live during development (see docs/agsandbox_backends/
        container.md's "Layer-depth squashing" section) -- asserting it
        explicitly here so it reads as an intentional, understood
        limitation rather than something a future change accidentally
        "fixes" into an infinite retry loop. stop() now skips even
        attempting the old-image lookup/delete on a plain-commit cycle
        (it's guaranteed to fail, so there's no point trying), so this
        test only checks the end state -- the old image genuinely
        surviving -- not that a delete was attempted and failed."""
        sb = _make_sandbox()
        sb._backend._base_image = "alpine:latest"
        try:
            sb.exec("echo one")
            sb.stop(commit=True)  # checkpoint 1 (plain commit)
            first_image_id = subprocess.run(
                ["docker", "inspect", "--format={{.Id}}", sb._backend._lifecycle_tag()],
                capture_output=True,
                text=True,
            ).stdout.strip()
            assert first_image_id

            sb.exec("echo two")
            sb.stop(commit=True)  # checkpoint 2 -- also a plain commit (chain, not squash)

            still_present = (
                subprocess.run(
                    ["docker", "image", "inspect", first_image_id], capture_output=True
                ).returncode
                == 0
            )
            assert still_present, (
                "checkpoint 1's image is a parent of checkpoint 2's -- must survive"
            )
        finally:
            sb.destroy()

    @docker
    def test_real_squash_cycle_reclaims_the_entire_prior_chain(self):
        """The payoff of squashing isn't just bounded layer depth -- once a
        squash lands (a parentless image), NOTHING depends on the prior
        chain anymore, so the eager old-image cleanup (fixed above to run
        after container removal) can finally succeed and `docker rmi`
        cascades to free the whole accumulated chain in one shot, not just
        the single most-recent link. End-to-end through the REAL
        agSandbox/backend stop() flow, not a hand-rolled docker command
        sequence (which is what test_repeated_commits_leave_no_dangling_
        images below does, and would not have caught either the ordering
        bug this fixes or confirmed this cascade)."""
        sb = _make_sandbox()
        sb._backend._base_image = "alpine:latest"
        try:
            sb.exec("echo one")
            sb.stop(commit=True)  # checkpoint 1 (plain commit)
            first_image_id = subprocess.run(
                ["docker", "inspect", "--format={{.Id}}", sb._backend._lifecycle_tag()],
                capture_output=True,
                text=True,
            ).stdout.strip()
            assert first_image_id

            sb.exec("echo two")
            # Force this cycle to squash instead of plain-committing.
            sb._backend._commits_since_squash = sb._backend.checkpoint_squash_interval - 1
            sb.stop(commit=True)  # checkpoint 2 -- squash: parentless, frees checkpoint 1

            still_present = (
                subprocess.run(
                    ["docker", "image", "inspect", first_image_id], capture_output=True
                ).returncode
                == 0
            )
            assert not still_present, "squashing should have reclaimed the entire prior chain"
        finally:
            sb.destroy()

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
# Layer-depth squashing -- stop(commit=True) periodically flattens the
# checkpoint chain (export/import) instead of always stacking a diff on top
# of it, to stay under the container runtime's hard layer-depth cap. See
# container.py's _squash_commit() and docs/agsandbox_backends/container.md's
# "Layer-depth squashing" section.
# ---------------------------------------------------------------------------


class _FakeCompleted:
    def __init__(self, stdout=b"", returncode=0):
        self.stdout = stdout
        self.returncode = returncode


class TestCheckpointSquash:
    """checkpoint_squash_interval is a tier-1 GlobalConfigParam (write-once,
    process-wide -- same as every other timeout/retry knob in this file), so
    it can't be overridden per-test via agConfig once anything in the
    process has already read its default. Tests instead pre-seed
    `_commits_since_squash` directly (a plain instance attribute) to land
    exactly at/below the real configured interval, exercising the same
    threshold-crossing logic without fighting that immutability."""

    def _sb(self):
        return _make_sandbox()

    def test_squashes_when_threshold_is_reached(self):
        """One commit away from the (real, whatever-it-is) configured
        interval: this stop(commit=True) call must do the normal plain
        commit FIRST (unchanged, always happens -- see
        TestCheckpointAccumulator for why this cycle's own diff-
        accumulator fold isn't mockable this simply and falls back), and
        ADDITIONALLY squash. The mocked `docker info` here returns no
        parseable JSON, so the fast accumulator path can't be trusted and
        falls back to `_squash_commit()`'s export/import -- the point of
        this test is the plain-commit-always-happens/counter-resets
        behavior, not which squash implementation ends up running."""
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        interval = sb._backend.checkpoint_squash_interval
        sb._backend._commits_since_squash = interval - 1
        calls = []

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            calls.append(list(args))
            return _FakeCompleted()

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with patch.object(sb._backend, "_container_running", return_value=True):
                with patch.object(sb._backend, "_gpu_virtual", False):
                    sb.stop(commit=True)

        assert any(c[:2] == [sb._backend._runtime, "commit"] for c in calls), (
            f"expected the normal plain commit to still happen: {calls}"
        )
        assert any("export" in c for c in calls), (
            f"expected the squash fallback's export call: {calls}"
        )
        assert any("import" in c for c in calls), (
            f"expected the squash fallback's import call: {calls}"
        )
        assert sb._backend._commits_since_squash == 0  # reset after squashing

    def test_does_not_squash_below_threshold(self):
        """Far from the threshold: stop(commit=True) must commit and must
        NOT additionally squash."""
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        sb._backend._commits_since_squash = 0
        calls = []

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            calls.append(list(args))
            return _FakeCompleted()

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with patch.object(sb._backend, "_container_running", return_value=True):
                with patch.object(sb._backend, "_gpu_virtual", False):
                    sb.stop(commit=True)

        assert any(c[:2] == [sb._backend._runtime, "commit"] for c in calls), (
            f"expected a plain commit: {calls}"
        )
        assert not any("export" in c for c in calls), f"must not squash yet: {calls}"
        assert sb._backend._commits_since_squash == 1

    def test_force_squash_flattens_regardless_of_counter(self):
        """stop(commit=True, force_squash=True) must squash even when
        nowhere near checkpoint_squash_interval -- used at skill exit
        (agskill.py's teardown) so a sandbox never hands back control at
        an arbitrary mid-chain depth. The commit still happens too."""
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        sb._backend._commits_since_squash = 0  # far from the periodic threshold
        calls = []

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            calls.append(list(args))
            return _FakeCompleted()

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with patch.object(sb._backend, "_container_running", return_value=True):
                with patch.object(sb._backend, "_gpu_virtual", False):
                    sb.stop(commit=True, force_squash=True)

        assert any(c[:2] == [sb._backend._runtime, "commit"] for c in calls), (
            f"expected the normal plain commit to still happen: {calls}"
        )
        assert any("export" in c for c in calls), f"expected an export call: {calls}"
        assert any("import" in c for c in calls), f"expected an import call: {calls}"

    def test_force_squash_resets_the_counter(self):
        """A forced squash must reset _commits_since_squash the same as a
        periodic one, so the NEXT stop(commit=True) (without force_squash)
        starts counting fresh rather than immediately squashing again."""
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        sb._backend._commits_since_squash = 5  # mid-count, nowhere near threshold

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            return _FakeCompleted()

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with patch.object(sb._backend, "_container_running", return_value=True):
                with patch.object(sb._backend, "_gpu_virtual", False):
                    sb.stop(commit=True, force_squash=True)

        assert sb._backend._commits_since_squash == 0

    def test_squash_pipes_export_stdout_into_import_stdin(self):
        """The flatten must round-trip the container's actual export bytes
        into import's stdin -- not shell out with a literal pipe (this
        codebase never invokes a shell for docker/podman calls) and not
        silently drop the payload."""
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        sb._backend._commits_since_squash = sb._backend.checkpoint_squash_interval - 1
        captured = {}
        fake_tar_bytes = b"FAKE_EXPORTED_FILESYSTEM_BYTES"

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "export" in args:
                return _FakeCompleted(stdout=fake_tar_bytes)
            if "import" in args:
                captured["input"] = input
                captured["args"] = list(args)
            return _FakeCompleted()

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with patch.object(sb._backend, "_container_running", return_value=True):
                with patch.object(sb._backend, "_gpu_virtual", False):
                    sb.stop(commit=True)

        assert captured["input"] == fake_tar_bytes
        assert captured["args"][-1] == sb._backend._lifecycle_tag()
        assert captured["args"][-2] == "-"  # import reads from stdin, not a file path

    def test_squash_failure_is_best_effort_and_does_not_reset_counter(self):
        """If every squash path fails (both the accumulator fast path and
        the _squash_commit() fallback), stop() must NOT raise -- the
        normal plain commit above it already succeeded, so a squash
        failure only means the layer chain keeps growing until the next
        attempt, not that this cycle's checkpoint was lost.
        _commits_since_squash must stay at or above the threshold (not
        silently reset) so the next stop() tries again."""
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        threshold = sb._backend.checkpoint_squash_interval
        sb._backend._commits_since_squash = threshold - 1

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "export" in args or "import" in args:
                raise RuntimeError("simulated export/import failure")
            return _FakeCompleted()

        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        try:
            with patch.object(_mod._DockerBackend, "_run", fake_run):
                with patch.object(sb._backend, "_container_running", return_value=True):
                    with patch.object(sb._backend, "_gpu_virtual", False):
                        sb.stop(commit=True)  # must not raise
        finally:
            sys.stderr = old_stderr

        assert sb._backend._commits_since_squash >= threshold
        assert "WARNING" in captured.getvalue()
        assert "squash failed" in captured.getvalue()

    @docker
    @pytest.mark.timeout(180)
    def test_squash_flattens_real_layer_depth(self):
        """A real docker export|import must reset RootFS.Layers to 1,
        confirming the flatten genuinely resets depth rather than just
        re-tagging the same growing chain. Uses the tiny local `alpine`
        image rather than the real (multi-GB) agency-sandbox image --
        this only needs to exercise the export/import mechanism itself,
        and a large image makes the round-trip genuinely slow."""

        def _layer_count(ref):
            r = subprocess.run(
                ["docker", "inspect", "--format={{len .RootFS.Layers}}", ref],
                capture_output=True,
                text=True,
            )
            return int(r.stdout.strip())

        name = f"test-squash-{uuid.uuid4().hex[:8]}"
        tag = f"agency/lifecycle-{name}"
        subprocess.run(
            ["docker", "run", "-d", "--name", name, "alpine:latest", "tail", "-f", "/dev/null"],
            capture_output=True,
            check=True,
        )
        try:
            # Build up a few real layers first via plain commits, same as an
            # un-squashed checkpoint chain would.
            for i in range(3):
                subprocess.run(
                    ["docker", "exec", name, "sh", "-c", f"echo {i} > /marker-{i}"],
                    capture_output=True,
                    check=True,
                )
                subprocess.run(["docker", "commit", name, tag], capture_output=True, check=True)
            depth_before = _layer_count(tag)
            assert depth_before > 1, "expected multiple stacked layers before squashing"

            export_proc = subprocess.run(
                ["docker", "export", name], capture_output=True, check=True
            )
            subprocess.run(
                ["docker", "import", "-", tag],
                input=export_proc.stdout,
                capture_output=True,
                check=True,
            )
            depth_after = _layer_count(tag)
            assert depth_after == 1, f"expected depth 1 after squash, got {depth_after}"
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


# ---------------------------------------------------------------------------
# Fast incremental squashing -- the diff accumulator, fed cheaply after each
# plain commit by reading that commit's own on-disk overlay2 diff directory
# directly (see container.py's _fold_commit_into_accumulator() and
# docker.py's _locate_layer_diff_dir()), letting squash time skip `docker
# save`/`docker diff` on the whole chain entirely. See
# docs/agsandbox_backends/container.md's "Fast incremental squashing"
# section for the full design and the real-world numbers that motivated it.
# ---------------------------------------------------------------------------


class TestLocateLayerDiffDir:
    """Tests for _DockerBackend._locate_layer_diff_dir() -- reaches into
    docker's own overlay2 on-disk layout. Uses REAL temp directories
    structured to match that layout (layerdb entries + cache-id files +
    overlay2 diff dirs), not a real docker daemon -- this is pure
    filesystem-correlation logic once `docker info`'s own result is
    known, so mocking that one call is enough to exercise it fully."""

    def _sb(self):
        return _make_sandbox()

    def _fake_docker_root(self, tmp_path, *, driver="overlay2"):
        root = tmp_path / "docker-root"
        (root / "image" / "overlay2" / "layerdb" / "sha256").mkdir(parents=True)
        (root / "overlay2").mkdir(parents=True)
        return root

    def _add_layerdb_entry(self, root, diff_id, cache_id, *, with_content=True):
        entry = root / "image" / "overlay2" / "layerdb" / "sha256" / f"entry-{cache_id}"
        entry.mkdir(parents=True)
        (entry / "diff").write_text(diff_id)
        (entry / "cache-id").write_text(cache_id)
        if with_content:
            diff_dir = root / "overlay2" / cache_id / "diff"
            diff_dir.mkdir(parents=True)
            (diff_dir / "marker").write_text("x")
        return entry

    def test_finds_diff_dir_matching_digest(self, tmp_path):
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        root = self._fake_docker_root(tmp_path)
        self._add_layerdb_entry(root, "sha256:target123", "cache-target")
        self._add_layerdb_entry(root, "sha256:other456", "cache-other")

        with patch.object(
            _mod._DockerBackend, "_docker_data_root_and_driver", return_value=(root, "overlay2")
        ):
            result = sb._backend._locate_layer_diff_dir("sha256:target123")

        assert result == root / "overlay2" / "cache-target" / "diff"
        assert (result / "marker").read_text() == "x"

    def test_returns_none_when_digest_not_found(self, tmp_path):
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        root = self._fake_docker_root(tmp_path)
        self._add_layerdb_entry(root, "sha256:other456", "cache-other")

        with patch.object(
            _mod._DockerBackend, "_docker_data_root_and_driver", return_value=(root, "overlay2")
        ):
            result = sb._backend._locate_layer_diff_dir("sha256:nonexistent")

        assert result is None

    def test_returns_none_for_unsupported_driver(self, tmp_path):
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        root = self._fake_docker_root(tmp_path)
        self._add_layerdb_entry(root, "sha256:target123", "cache-target")

        with patch.object(
            _mod._DockerBackend, "_docker_data_root_and_driver", return_value=(root, "devicemapper")
        ):
            result = sb._backend._locate_layer_diff_dir("sha256:target123")

        assert result is None

    def test_returns_none_when_info_lookup_fails(self):
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        with patch.object(_mod._DockerBackend, "_docker_data_root_and_driver", return_value=None):
            result = sb._backend._locate_layer_diff_dir("sha256:anything")
        assert result is None

    def test_returns_none_when_layerdb_dir_present_but_content_missing(self, tmp_path):
        """A matching layerdb entry whose overlay2 diff dir doesn't
        actually exist (e.g. already cleaned up) must degrade to None,
        not raise."""
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        root = self._fake_docker_root(tmp_path)
        self._add_layerdb_entry(root, "sha256:target123", "cache-target", with_content=False)

        with patch.object(
            _mod._DockerBackend, "_docker_data_root_and_driver", return_value=(root, "overlay2")
        ):
            result = sb._backend._locate_layer_diff_dir("sha256:target123")

        assert result is None

    def test_overlayfs_requires_diff_ids_chain_ending_at_diff_id(self, tmp_path):
        """Containerd snapshotter path keys layers by ChainID -- calling
        without a matching RootFS.Layers prefix must return None rather
        than guessing from the tip DiffID alone."""
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        root = self._fake_docker_root(tmp_path)
        with patch.object(
            _mod._DockerBackend, "_docker_data_root_and_driver", return_value=(root, "overlayfs")
        ):
            assert sb._backend._locate_layer_diff_dir("sha256:tip") is None
            assert (
                sb._backend._locate_layer_diff_dir(
                    "sha256:tip", diff_ids=["sha256:base", "sha256:not-tip"]
                )
                is None
            )

    def test_overlayfs_uses_ctr_mounts_top_fs(self, tmp_path):
        """Mocked ctr view+mounts: the first lowerdir component is the
        layer's own fs directory (confirmed against real containerd)."""
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        root = self._fake_docker_root(tmp_path)
        snap_fs = tmp_path / "snapshots" / "9" / "fs"
        snap_fs.mkdir(parents=True)
        (snap_fs / "marker").write_text("ok")
        diff_ids = ["sha256:" + "a" * 64, "sha256:" + "b" * 64]
        mounts_out = (
            f"mount -t overlay overlay /tmp/x -o "
            f"lowerdir={snap_fs}:{tmp_path}/snapshots/8/fs,index=off\n"
        )

        class _Done:
            def __init__(self, returncode=0, stdout="", stderr=b""):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        calls = []

        def fake_run(args, **kwargs):
            calls.append(list(args))
            if "mounts" in args:
                return _Done(stdout=mounts_out)
            return _Done()

        with patch.object(
            _mod._DockerBackend, "_docker_data_root_and_driver", return_value=(root, "overlayfs")
        ):
            with patch.object(_mod._DockerBackend, "_ctr_argv", return_value=["ctr"]):
                with patch.object(_mod.subprocess, "run", side_effect=fake_run):
                    result = sb._backend._locate_layer_diff_dir(diff_ids[-1], diff_ids=diff_ids)

        assert result == snap_fs
        assert any("view" in c for c in calls)
        assert any("mounts" in c for c in calls)
        assert any(c[-2:] == ["snapshots", "rm"] or "rm" in c for c in calls)

    def test_chain_id_matches_containerd_definition(self):
        from agency.agsandbox_backends.docker import _chain_id_for_diff_ids
        import hashlib

        d0 = "sha256:" + "a" * 64
        d1 = "sha256:" + "b" * 64
        assert _chain_id_for_diff_ids([d0]) == d0
        expected = "sha256:" + hashlib.sha256(f"{d0} {d1}".encode()).hexdigest()
        assert _chain_id_for_diff_ids([d0, d1]) == expected

    def test_parse_ctr_mounts_top_fs(self):
        from agency.agsandbox_backends.docker import _parse_ctr_mounts_top_fs
        from pathlib import Path

        out = (
            "mount -t overlay overlay /tmp/x -o "
            "lowerdir=/var/lib/containerd/.../snapshots/1049/fs:"
            "/var/lib/containerd/.../snapshots/1046/fs,index=off\n"
        )
        assert _parse_ctr_mounts_top_fs(out) == Path("/var/lib/containerd/.../snapshots/1049/fs")
        assert _parse_ctr_mounts_top_fs(
            "mount --bind /var/lib/containerd/.../snapshots/1/fs /tmp/x\n"
        ) == Path("/var/lib/containerd/.../snapshots/1/fs")

    @docker
    def test_real_containerd_overlayfs_commit_layer_locates(self):
        """End-to-end against live Docker+containerd: commit a marker file
        and resolve its RootFS tip to a readable snapshot fs/ directory
        containing that marker. Skips when this host isn't on the
        overlayfs snapshotter or the snapshot tree isn't readable."""
        info = json.loads(
            subprocess.run(
                ["docker", "info", "--format", "{{json .}}"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        if info.get("Driver") != "overlayfs":
            pytest.skip(f"Docker Driver is {info.get('Driver')!r}, not overlayfs")

        # Rootful containerd stores are often 0700; make the snapshotter
        # tree traversable/readable for this process when passwordless
        # sudo is available so overlay_diff_to_tar can use the path.
        snap_root = Path("/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs")
        if not os.access(snap_root, os.R_OK | os.X_OK):
            chmod = subprocess.run(
                ["sudo", "-n", "chmod", "-R", "a+rX", str(snap_root.parent), str(snap_root)],
                capture_output=True,
            )
            if chmod.returncode != 0 or not os.access(snap_root, os.R_OK | os.X_OK):
                pytest.skip("containerd snapshotter storage not readable")

        name = f"test-ctrd-locate-{uuid.uuid4().hex[:8]}"
        tag = f"agency-test-ctrd-locate-{uuid.uuid4().hex[:8]}:latest"
        sb = self._sb()
        try:
            subprocess.run(
                ["docker", "run", "-d", "--name", name, "alpine:latest", "sleep", "3600"],
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["docker", "exec", name, "sh", "-c", "echo ctrd-locate > /tmp/ctrd-locate.txt"],
                capture_output=True,
                check=True,
            )
            subprocess.run(["docker", "commit", name, tag], capture_output=True, check=True)
            layers = json.loads(
                subprocess.run(
                    ["docker", "inspect", "--format={{json .RootFS.Layers}}", tag],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
            )
            diff_dir = sb._backend._locate_layer_diff_dir(layers[-1], diff_ids=layers)
            assert diff_dir is not None, f"failed to locate diff dir for {layers[-1]}"
            assert (diff_dir / "tmp" / "ctrd-locate.txt").read_text() == "ctrd-locate\n"
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


class TestHostToContainerId:
    """Tests for _DockerBackend._host_to_container_id() -- the rootless
    Docker uid/gid translation feeding overlay_diff_to_tar() via
    _fold_commit_into_accumulator(). Mocks _docker_info()/PID discovery
    rather than a real rootless daemon; the reverse-mapping arithmetic
    itself is exercised directly against real /proc-style uid_map/gid_map
    content captured from an actual rootless daemon."""

    def _sb(self):
        return _make_sandbox()

    def test_identity_when_not_rootless(self):
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        with patch.object(
            _mod._DockerBackend, "_docker_info", return_value={"SecurityOptions": []}
        ):
            assert sb._backend._host_to_container_id(1000, 1000) == (1000, 1000)

    def test_identity_when_docker_info_unavailable(self):
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        with patch.object(_mod._DockerBackend, "_docker_info", return_value=None):
            assert sb._backend._host_to_container_id(1000, 1000) == (1000, 1000)

    def test_identity_when_rootless_but_maps_unreadable(self):
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        with patch.object(
            _mod._DockerBackend, "_docker_info", return_value={"SecurityOptions": ["name=rootless"]}
        ):
            with patch.object(_mod._DockerBackend, "_find_dockerd_pid", return_value=None):
                assert sb._backend._host_to_container_id(200682, 3662) == (200682, 3662)

    def test_translates_using_real_captured_uid_gid_maps(self):
        """uid_map/gid_map content captured from a real rootless dockerd
        during development: container uid 0 maps to exactly host uid
        200682 (the invoking user), and container uids/gids 1-65536 map
        to a large host-side range starting at 1355350016 (uid_map) /
        3663 (gid_map here, standing in for whatever /etc/subgid assigns)."""
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        uid_map = [(0, 200682, 1), (1, 1355350016, 65536)]
        gid_map = [(0, 3662, 1), (1, 3663, 65536)]
        with patch.object(
            _mod._DockerBackend, "_docker_info", return_value={"SecurityOptions": ["name=rootless"]}
        ):
            with patch.object(
                _mod._DockerBackend, "_rootless_id_maps", return_value=(uid_map, gid_map)
            ):
                assert sb._backend._host_to_container_id(200682, 3662) == (0, 0)
                assert sb._backend._host_to_container_id(1355350017, 3664) == (2, 2)

    def test_translate_id_leaves_unmapped_host_id_unchanged(self):
        from agency.agsandbox_backends.docker import _translate_id

        assert _translate_id(999999, [(0, 200682, 1)]) == 999999

    def test_find_dockerd_pid_matches_process_by_name_and_owner(self, tmp_path, monkeypatch):
        """Fakes /proc as a tmp_path tree: PID 111 is "dockerd" owned by
        our own uid (the match), PID 222 is some other process owned by
        a different uid (must be skipped)."""
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        (tmp_path / "111").mkdir()
        (tmp_path / "111" / "comm").write_text("dockerd\n")
        (tmp_path / "222").mkdir()
        (tmp_path / "222" / "comm").write_text("bash\n")

        my_uid = os.getuid()
        real_stat = os.stat

        def fake_listdir(path):
            assert path == "/proc"
            return ["111", "222"]

        def fake_stat(path):
            if path == "/proc/111":
                return real_stat(tmp_path / "111")
            return real_stat(tmp_path)  # some other uid (our own test process's cwd)

        real_open = open

        def fake_open(path, *a, **kw):
            if str(path).startswith("/proc/"):
                pid = str(path).split("/")[2]
                return real_open(tmp_path / pid / "comm", *a, **kw)
            return real_open(path, *a, **kw)

        monkeypatch.setattr(_mod.os, "listdir", fake_listdir)
        monkeypatch.setattr(_mod.os, "stat", fake_stat)
        monkeypatch.setattr(_mod.os, "getuid", lambda: my_uid)
        monkeypatch.setattr(_mod, "open", fake_open, raising=False)

        assert real_stat(tmp_path / "111").st_uid == my_uid

        result = sb._backend._find_dockerd_pid()
        assert result == 111

    def test_find_dockerd_pid_returns_none_when_absent(self, monkeypatch):
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        monkeypatch.setattr(_mod.os, "listdir", lambda p: [])
        assert sb._backend._find_dockerd_pid() is None

    def test_parse_id_map_reads_real_proc_style_content(self, tmp_path):
        import agency.agsandbox_backends.docker as _mod

        p = tmp_path / "uid_map"
        p.write_text("         0     200682          1\n         1 1355350016      65536\n")
        assert _mod._DockerBackend._parse_id_map(p) == [(0, 200682, 1), (1, 1355350016, 65536)]

    def test_parse_id_map_returns_none_for_missing_file(self, tmp_path):
        import agency.agsandbox_backends.docker as _mod

        assert _mod._DockerBackend._parse_id_map(tmp_path / "nonexistent") is None


class TestCheckpointAccumulator:
    """Tests for _fold_commit_into_accumulator() and
    _accumulator_squash_commit() -- the fast squash path fed by
    TestLocateLayerDiffDir's lookup. Mocks _locate_layer_diff_dir directly
    (rather than the whole docker-root filesystem dance) since that
    lookup mechanism is already covered on its own above."""

    def _sb(self):
        return _make_sandbox()

    def _make_real_diff_dir(self, tmp_path, name, files):
        d = tmp_path / name
        d.mkdir()
        for rel, content in files.items():
            p = d / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        return d

    def test_fold_builds_accumulator_from_overlay_diff_dir(self, tmp_path):
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        diff_dir = self._make_real_diff_dir(tmp_path, "diff1", {"workspace/f1": "one"})

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "--format={{json .RootFS.Layers}}" in args:
                return _FakeCompleted(stdout=b'["sha256:layer1"]')
            return _FakeCompleted()

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with patch.object(_mod._DockerBackend, "_locate_layer_diff_dir", return_value=diff_dir):
                sb._backend._fold_commit_into_accumulator("some-tag")

        assert sb._backend._accumulated_layer_count == 1
        assert sb._backend._accumulated_diff_path is not None
        with tarfile.open(sb._backend._accumulated_diff_path, "r") as tf:
            content = tf.extractfile("workspace/f1").read()
        assert content == b"one"

    def test_fold_invalidates_accumulator_when_diff_dir_not_found(self):
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "--format={{json .RootFS.Layers}}" in args:
                return _FakeCompleted(stdout=b'["sha256:layer1"]')
            return _FakeCompleted()

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with patch.object(_mod._DockerBackend, "_locate_layer_diff_dir", return_value=None):
                sb._backend._fold_commit_into_accumulator("some-tag")

        assert sb._backend._accumulated_diff_path is None
        assert sb._backend._accumulated_layer_count == -1

    def test_fold_accumulates_across_multiple_cycles(self, tmp_path):
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        diff1 = self._make_real_diff_dir(tmp_path, "diff1", {"a": "1"})
        diff2 = self._make_real_diff_dir(tmp_path, "diff2", {"b": "2"})

        layers_seen = ["sha256:layer1"]

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "--format={{json .RootFS.Layers}}" in args:
                return _FakeCompleted(stdout=json.dumps(layers_seen).encode())
            return _FakeCompleted()

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with patch.object(_mod._DockerBackend, "_locate_layer_diff_dir", return_value=diff1):
                sb._backend._commits_since_squash = 1
                sb._backend._fold_commit_into_accumulator("tag")
            layers_seen.append("sha256:layer2")
            with patch.object(_mod._DockerBackend, "_locate_layer_diff_dir", return_value=diff2):
                sb._backend._commits_since_squash = 2
                sb._backend._fold_commit_into_accumulator("tag")

        assert sb._backend._accumulated_layer_count == 2
        with tarfile.open(sb._backend._accumulated_diff_path, "r") as tf:
            names = {m.name for m in tf.getmembers()}
        assert names == {"a", "b"}

    def test_accumulator_squash_raises_when_layer_count_mismatched(self, tmp_path):
        """Simulates the post-fork scenario: the accumulator (fresh, 0
        layers tracked) doesn't match the real gap between the current
        checkpoint and the base (>0, since the chain has real history) --
        must raise so the caller falls back, never silently squash an
        incomplete diff."""
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        sb._backend._accumulated_diff_path = tmp_path / "fake.tar"
        sb._backend._accumulated_diff_path.write_bytes(b"")
        sb._backend._accumulated_layer_count = 0  # fresh, but...

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "--format={{json .RootFS.Layers}}" in args:
                # ...the real gap is 2 layers, not 0.
                if args[-1] == "agency-sandbox:latest":
                    return _FakeCompleted(stdout=b'["sha256:base1"]')
                return _FakeCompleted(stdout=b'["sha256:base1", "sha256:c1", "sha256:c2"]')
            return _FakeCompleted()

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with pytest.raises(RuntimeError, match="accumulator tracks"):
                sb._backend._accumulator_squash_commit("some-tag")

    def test_accumulator_squash_raises_when_base_not_a_prefix(self, tmp_path):
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        sb._backend._accumulated_diff_path = tmp_path / "fake.tar"
        sb._backend._accumulated_diff_path.write_bytes(b"")
        sb._backend._accumulated_layer_count = 1

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "--format={{json .RootFS.Layers}}" in args:
                if args[-1] == "agency-sandbox:latest":
                    return _FakeCompleted(stdout=b'["sha256:base1"]')
                # Current chain does NOT start with the base's own layer.
                return _FakeCompleted(stdout=b'["sha256:different", "sha256:c1"]')
            return _FakeCompleted()

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with pytest.raises(RuntimeError):
                sb._backend._accumulator_squash_commit("some-tag")

    def test_accumulator_squash_raises_when_no_accumulator(self):
        sb = self._sb()
        assert sb._backend._accumulated_diff_path is None
        with pytest.raises(RuntimeError, match="no checkpoint diff accumulator"):
            sb._backend._accumulator_squash_commit("some-tag")

    @docker
    @pytest.mark.timeout(180)
    def test_real_end_to_end_fast_squash_against_large_base_image(self):
        """The full, real thing: several plain-commit cycles against the
        actual multi-GB agency-sandbox:latest image, each folding into
        the accumulator via the real overlay2 lookup (no mocking at all),
        then a real squash -- must complete in well under the ~77-180s
        the slower paths took (measured during development), produce
        correct content, and leave the base image's own layers
        genuinely untouched (proving no re-serialization happened)."""
        sb = _make_sandbox()

        base_layers_before = subprocess.run(
            ["docker", "inspect", "--format={{json .RootFS.Layers}}", "agency-sandbox:latest"],
            capture_output=True,
            text=True,
        ).stdout

        sb.exec("mkdir -p /workspace/proj && echo one > /workspace/proj/f1")
        sb.stop(commit=True)
        sb.exec("echo two > /workspace/proj/f2 && rm /workspace/proj/f1")
        sb.stop(commit=True)
        sb.exec("mkdir -p /workspace/proj/sub && echo three > /workspace/proj/sub/f3")

        sb._backend._commits_since_squash = sb._backend.checkpoint_squash_interval - 1
        t0 = time.time()
        sb.stop(commit=True)  # this cycle commits AND squashes
        elapsed = time.time() - t0

        try:
            # Classic overlay2 often lands well under 10s; containerd
            # overlayfs pays sudo-ctr + per-snapshot chmod on each fold,
            # so allow more headroom while still rejecting the ~77-180s
            # export/import fallback.
            assert elapsed < 45, f"fast squash took {elapsed:.1f}s -- expected well under 45s"

            base_layers_after = subprocess.run(
                ["docker", "inspect", "--format={{json .RootFS.Layers}}", "agency-sandbox:latest"],
                capture_output=True,
                text=True,
            ).stdout
            assert base_layers_after == base_layers_before

            tag = sb._backend._lifecycle_tag()
            layers = int(
                subprocess.run(
                    ["docker", "inspect", "--format={{len .RootFS.Layers}}", tag],
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            )
            base_layer_count = len(json.loads(base_layers_before))
            assert layers == base_layer_count + 1, (
                "expected exactly one new layer on top of the base"
            )

            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    tag,
                    "sh",
                    "-c",
                    "ls /workspace/proj/f1 2>&1; cat /workspace/proj/f2; cat /workspace/proj/sub/f3",
                ],
                capture_output=True,
                text=True,
            )
            assert "No such file" in result.stdout or "No such file" in result.stderr
            assert "two" in result.stdout
            assert "three" in result.stdout
        finally:
            sb.destroy()


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

    def test_destroy_does_not_release_semaphore_when_container_still_running_after_rm_failure(
        self,
    ):
        """If rm genuinely fails and the container is confirmed STILL
        running afterward, the runtime slot must NOT be released -- it's
        still physically held. Releasing here would over-credit the
        semaphore (letting one more container start than the host's kernel
        keyring quota actually allows) for a container that never actually
        went away. Regression test for the double-release/over-credit bug
        this exact scenario used to cause: destroy() releasing unconditionally
        in `finally` regardless of whether removal actually succeeded."""
        import agency.agsandbox_backends.container as _container_mod
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
            # Still running both before AND after the failed rm attempt --
            # removal truly never happened.
            with patch.object(sb._backend, "_container_running", return_value=True):
                with patch.object(sb._backend, "_container_status", return_value="running"):
                    with patch.object(
                        _container_mod._container_semaphore,
                        "release",
                        side_effect=lambda: released.append(1),
                    ):
                        with pytest.raises(RuntimeError, match="rm exploded"):
                            sb.destroy()

        assert not released, (
            "semaphore must NOT be released while the container is confirmed still running"
        )

    def test_destroy_releases_semaphore_when_container_confirmed_gone_despite_rm_error(self):
        """If rm raises (e.g. a transient secondary error) but the container
        is actually confirmed gone by the time destroy() checks again, the
        runtime slot must still be released -- it really is free now, and an
        error from rm alone shouldn't strand the slot as unreleasable
        forever."""
        import agency.agsandbox_backends.container as _container_mod
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

        # had_container (destroy()'s pre-rm check) must see "running" so the
        # test actually exercises the "was running, rm failed, but confirmed
        # gone by the recheck" path -- the post-rm recheck in the `finally`
        # block must see "gone". Same method, two different truthful answers
        # at two different times, exactly like a real rm that silently
        # succeeded despite raising a secondary error.
        running_calls = [True, False]

        def fake_container_running():
            return running_calls.pop(0) if running_calls else False

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with patch.object(
                sb._backend, "_container_running", side_effect=fake_container_running
            ):
                with patch.object(sb._backend, "_container_status", return_value="running"):
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
            with patch.object(sb._backend, "_container_running", return_value=False):
                with patch.object(sb._backend, "_container_status", return_value=""):
                    sb.destroy()

        rm_calls = [a for a in calls if "rm" in a and "rmi" not in a]
        assert not rm_calls, f"must not rm when container absent; got {rm_calls}"


# ---------------------------------------------------------------------------
# GPU semaphore release gating in stop()/destroy() -- the real GPU semaphore
# (agResourcePool.release_gpu, wired in via reserve_gpu) must only be
# released once the container is CONFIRMED torn down, exactly like the
# runtime-slot semaphore above -- releasing it while the container might
# still be running and actually using the GPU would let something else
# acquire the same physical GPU concurrently.
# ---------------------------------------------------------------------------


class TestDockerGpuReleaseGating:
    def _sb(self):
        return _make_sandbox()

    def _lease_gpu(self, sb, gpu_id=3):
        released = []
        sb._gpu_virtual = True
        sb._gpu_id = gpu_id
        sb._gpu_release_fn = lambda gid: released.append(gid)
        return released

    def test_stop_releases_gpu_when_already_confirmed_gone(self):
        """stop()'s early-return branch (container already not running when
        stop() is entered) must still release a leased GPU."""
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        released = self._lease_gpu(sb)

        with patch.object(_mod._DockerBackend, "_run"):
            with patch.object(sb._backend, "_container_running", return_value=False):
                sb.stop(commit=False)

        assert released == [3]
        assert sb._gpu_id is None

    def test_stop_releases_gpu_via_main_teardown_path_after_successful_rm(self):
        """The other release site in stop() -- reached via the main
        teardown path (container was running, rm succeeded) rather than the
        early-return branch above."""
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        released = self._lease_gpu(sb)

        class OK:
            returncode = 0
            stdout = b""

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            return OK()

        # top-of-stop() check (must be True to take the main path), then the
        # post-rm runtime-slot check, then the GPU-release check -- 3 calls.
        running_calls = [True, False, False]

        def fake_container_running():
            return running_calls.pop(0) if running_calls else False

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with patch.object(
                sb._backend, "_container_running", side_effect=fake_container_running
            ):
                sb.stop(commit=False)

        assert released == [3]
        assert sb._gpu_id is None

    def test_stop_does_not_release_gpu_when_rm_fails_and_container_still_running(self):
        import agency.agsandbox_backends.container as _container_mod
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        released = self._lease_gpu(sb)

        class OK:
            returncode = 0
            stdout = b""

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "rm" in args:
                raise RuntimeError("rm exploded")
            return OK()

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with patch.object(sb._backend, "_container_running", return_value=True):
                with patch.object(_container_mod.time, "sleep"):  # skip real retry backoff
                    with pytest.raises(RuntimeError, match="rm exploded"):
                        sb.stop(commit=False)

        assert released == [], (
            "GPU must not be released while the container is confirmed still running"
        )
        assert sb._gpu_id == 3, "gpu_id must be left untouched when release didn't happen"

    def test_destroy_releases_gpu_when_container_confirmed_gone_despite_rm_error(self):
        import agency.agsandbox_backends.docker as _mod

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

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with patch.object(
                sb._backend, "_container_running", side_effect=fake_container_running
            ):
                with patch.object(sb._backend, "_container_status", return_value="running"):
                    with pytest.raises(RuntimeError, match="rm exploded"):
                        sb.destroy()

        assert released == [3]
        assert sb._gpu_id is None

    def test_destroy_does_not_release_gpu_when_container_still_running_after_rm_failure(self):
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        released = self._lease_gpu(sb)

        class OK:
            returncode = 0
            stdout = b""

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "rm" in args:
                raise RuntimeError("rm exploded")
            return OK()

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with patch.object(sb._backend, "_container_running", return_value=True):
                with patch.object(sb._backend, "_container_status", return_value="running"):
                    with pytest.raises(RuntimeError, match="rm exploded"):
                        sb.destroy()

        assert released == []
        assert sb._gpu_id == 3

    def test_gpu_released_exactly_once_across_stop_then_destroy(self):
        """stop() tears the container down and releases the GPU; a later
        destroy() call on the same sandbox must see gpu_id already cleared
        and must not release a second time."""
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()
        released = self._lease_gpu(sb)

        with patch.object(_mod._DockerBackend, "_run"):
            with patch.object(sb._backend, "_container_running", return_value=False):
                sb.stop(commit=False)
                sb.destroy()

        assert released == [3], "GPU must be released exactly once, not once per call"

    def test_stop_rm_exc_takes_precedence_over_commit_exc(self):
        """When both the commit retries AND the rm retries exhaust, stop()
        must still run its GPU/runtime-slot release checks and raise --
        specifically the rm failure, since rm is stop()'s last word on
        whether the container is actually gone (a failed commit alone
        doesn't mean teardown didn't happen)."""
        import agency.agsandbox_backends.container as _container_mod
        import agency.agsandbox_backends.docker as _mod

        sb = self._sb()

        class OK:
            returncode = 0
            stdout = b""

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "commit" in args:
                raise RuntimeError("commit exploded")
            if "rm" in args:
                raise RuntimeError("rm exploded")
            return OK()

        with patch.object(_mod._DockerBackend, "_run", fake_run):
            with patch.object(sb._backend, "_container_running", return_value=True):
                with patch.object(_container_mod.time, "sleep"):  # skip real retry backoff
                    with pytest.raises(RuntimeError, match="rm exploded"):
                        sb.stop(commit=True)


# Session-keyring-quota machinery (_keyring_container_limit/keyring_quota/
# _semaphore_held_count) and the quota hooks (_is_quota_exhaustion_error/
# _wait_for_quota_slot/_quota_diagnostics) now live entirely on
# _ContainerBackendBase (agency.agsandbox_backends.container) since docker and
# podman share them identically -- see test_container.py's
# TestKeyringQuotaDiagnostics and TestQuotaHooksSharedAcrossRuntimes.
