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
import json
import subprocess
import sys
import time
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
    """Tests for the eager old-image deletion in commit(). Mirrors
    test_docker.py's class of the same name."""

    def test_stop_commit_deletes_old_image(self):
        """commit() must delete the image that previously held the tag --
        only possible on a squash cycle (a plain commit's result is always a
        child of the old image, so the runtime refuses to delete it;
        old_image_id is only looked up at all when this cycle squashes, see
        container.py's commit()). checkpoint_squash_max_depth is patched
        down to 1 to force this cycle to squash without needing a real deep
        chain."""
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
            if "--format={{json .RootFS.Layers}}" in args:
                return FakeCompleted(stdout=b'["sha256:layer0"]')
            if "--format={{.Id}}" in args:
                return FakeCompleted(stdout=fake_old_id.encode())
            if "commit" in args:
                return FakeCompleted()
            if "rm" in args:
                return FakeCompleted()
            if "rmi" in args:
                return FakeCompleted()
            return FakeCompleted()

        with patch.object(_mod._PodmanBackend, "checkpoint_squash_max_depth", 1):
            with patch.object(_mod._PodmanBackend, "_run", fake_run):
                with patch.object(sb, "_container_status", return_value="running"):
                    with patch.object(sb, "_gpu_virtual", False):
                        sb.commit()

        rmi_calls = [a for a in run_calls if "rmi" in a]
        assert rmi_calls, "expected podman rmi call for old image"
        assert any(fake_old_id in " ".join(a) for a in rmi_calls), (
            f"rmi call did not reference old image ID; calls: {rmi_calls}"
        )

    def test_stop_commit_skips_rmi_when_no_old_image(self):
        """If the tag does not exist yet (first squash), no rmi call is made."""
        import agency.agsandbox_backends.podman as _mod

        sb = _make_backend()

        class FakeCompleted:
            def __init__(self, stdout=b"", returncode=0):
                self.stdout = stdout
                self.returncode = returncode

        run_calls = []

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            run_calls.append(args)
            if "--format={{json .RootFS.Layers}}" in args:
                return FakeCompleted(stdout=b'["sha256:layer0"]')
            if "--format={{.Id}}" in args:
                return FakeCompleted(stdout=b"", returncode=1)  # tag not found
            return FakeCompleted()

        with patch.object(_mod._PodmanBackend, "checkpoint_squash_max_depth", 1):
            with patch.object(_mod._PodmanBackend, "_run", fake_run):
                with patch.object(sb, "_container_status", return_value="running"):
                    with patch.object(sb, "_gpu_virtual", False):
                        sb.commit()

        rmi_calls = [a for a in run_calls if "rmi" in a]
        assert not rmi_calls, "must not call rmi when there was no previous image"

    def test_plain_commit_cycle_with_no_previous_image_skips_ps_and_rmi(self):
        """commit() now looks up whatever image the tag currently points to
        on EVERY cycle (step 0), not just squash cycles -- under the
        hibernate model, successive commits on the same never-recreated
        container are siblings, not parent/child, so the previous tag
        image is always safe to reclaim once the tag moves off it. But
        when there's no previous image at all (the tag doesn't exist yet),
        there's nothing to check or delete, so the `ps`/`rmi` follow-up
        calls must still be skipped. Mirrors test_docker.py's version; see
        that test's docstring for the full reasoning. A shallow chain
        keeps the depth check itself from triggering a squash."""
        import agency.agsandbox_backends.podman as _mod

        sb = _make_backend()
        calls = []

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            calls.append(list(args))
            if "--format={{json .RootFS.Layers}}" in args:
                return _FakeCompleted(stdout=json.dumps(["sha256:layer0"]).encode())
            if "--format={{.Id}}" in args:
                return _FakeCompleted(stdout=b"", returncode=1)  # tag not found yet
            return _FakeCompleted()

        with patch.object(_mod._PodmanBackend, "_run", fake_run):
            with patch.object(sb, "_container_status", return_value="running"):
                with patch.object(sb, "_gpu_virtual", False):
                    sb.commit()

        id_lookups = [c for c in calls if "--format={{.Id}}" in c]
        assert id_lookups, "step 0 must still look up the tag's current image every cycle"
        assert not any("ps" in c for c in calls), (
            f"must not check ancestor when there was no previous image: {calls}"
        )
        assert not any("rmi" in c for c in calls), (
            f"must not attempt rmi when there was no previous image: {calls}"
        )

    def test_stop_commit_rmi_failure_is_best_effort(self):
        """A failing rmi during old-image cleanup must NOT propagate.
        Only reachable on a squash cycle -- see test_stop_commit_deletes_
        old_image's docstring."""
        import agency.agsandbox_backends.podman as _mod

        sb = _make_backend()
        fake_old_id = "sha256:cafebabe1234"

        class FakeCompleted:
            def __init__(self, stdout=b"", returncode=0):
                self.stdout = stdout
                self.returncode = returncode

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "--format={{json .RootFS.Layers}}" in args:
                return FakeCompleted(stdout=b'["sha256:layer0"]')
            if "--format={{.Id}}" in args:
                return FakeCompleted(stdout=fake_old_id.encode())
            if "rmi" in args:
                raise RuntimeError("image in use")
            return FakeCompleted()

        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        try:
            with patch.object(_mod._PodmanBackend, "checkpoint_squash_max_depth", 1):
                with patch.object(_mod._PodmanBackend, "_run", fake_run):
                    with patch.object(sb, "_container_status", return_value="running"):
                        with patch.object(sb, "_gpu_virtual", False):
                            sb.commit()  # must not raise
        finally:
            sys.stderr = old_stderr

        assert "WARNING" in captured.getvalue()
        assert fake_old_id in captured.getvalue()

    def test_old_image_ancestor_checks_run_via_ps_then_rmi(self):
        """The ancestor-based "is the old image still in use" check must
        run via `ps` then `rmi`, in that order -- twice per cycle when a
        squash is due, once for step 0's previous-tag-image cleanup (which
        now runs on every cycle, before the plain commit) and once for the
        squash's own superseded-plain-commit cleanup (step 3, after the
        squash). Under the OLD design, stop(commit=True) bundled a
        container remove+recreate into the same call, so this check had to
        run AFTER that internal removal. Under the NEW design, commit()
        never removes the container at all (see its docstring: "it keeps
        running (or stays hibernating)"), so there is no `rm` call here to
        order against any more -- mirrors test_docker.py's version of this
        test; see there for the full reasoning."""
        import agency.agsandbox_backends.podman as _mod

        sb = _make_backend()
        fake_old_id = "sha256:deadbeef0000"
        call_order = []

        class FakeCompleted:
            def __init__(self, stdout=b"", returncode=0):
                self.stdout = stdout
                self.returncode = returncode

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "--format={{json .RootFS.Layers}}" in args:
                return FakeCompleted(stdout=b'["sha256:layer0"]')
            if "--format={{.Id}}" in args:
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

        with patch.object(_mod._PodmanBackend, "checkpoint_squash_max_depth", 1):
            with patch.object(_mod._PodmanBackend, "_run", fake_run):
                with patch.object(sb, "_container_status", return_value="running"):
                    with patch.object(sb, "_gpu_virtual", False):
                        sb.commit()

        assert "rm" not in call_order, (
            f"commit() must never call rm -- it never removes the container: {call_order}"
        )
        assert call_order == ["ps", "rmi", "ps", "rmi"], (
            f"expected ps-then-rmi for step 0's previous-image cleanup, then again for "
            f"the squash's own superseded-commit cleanup, got {call_order}"
        )

    @podman
    def test_real_plain_commit_cycle_cleans_up_previous_sibling(self):
        """commit() never removes or recreates the container -- so two
        consecutive plain commit() calls on the SAME never-recreated
        container each just re-snapshot that container's own writable
        layer as a fresh base+1 image (confirmed empirically against a
        real docker daemon in test_docker.py's mirror of this test: repeated
        `commit` on a container that was never removed/recreated always
        yields images of the SAME depth, never chained onto each other).
        checkpoint 1's image is therefore not a parent of checkpoint 2's
        under the new design -- which means it's immediately safe to
        delete once the tag moves off it: commit()'s step 0/1a does exactly
        that on every cycle, not just squash cycles. Unverified against a
        real podman daemon in the environment this was written in (podman
        wasn't reachable there), but the underlying mechanism is shared,
        runtime-agnostic code on `_ContainerBackendBase`.

        Uses agency-sandbox:latest, not a bare `alpine:latest` -- Podman's
        `_resolve_image()` prefixes bare names with `localhost/` (see its
        docstring), and a `localhost/alpine:latest` that was never actually
        pulled/tagged locally makes Podman attempt a real network pull
        against a registry literally named `localhost`, which fails outright
        on any host without that exact local tag already present. The
        already-locally-built `agency-sandbox:latest` used by the rest of
        this file's real-daemon tests doesn't have that problem."""
        sb = _make_sandbox()
        sb._backend._base_image = "agency-sandbox:latest"
        try:
            sb.exec("echo one")
            sb.commit()  # checkpoint 1 (plain commit)
            first_image_id = subprocess.run(
                ["podman", "inspect", "--format={{.Id}}", sb._backend._lifecycle_tag()],
                capture_output=True,
                text=True,
            ).stdout.strip()
            assert first_image_id

            sb.exec("echo two")
            sb.commit()  # checkpoint 2 -- also a plain commit, on the SAME container

            still_present = (
                subprocess.run(
                    ["podman", "image", "inspect", first_image_id], capture_output=True
                ).returncode
                == 0
            )
            assert not still_present, (
                "checkpoint 1's image is a sibling, not a parent, of checkpoint 2's -- "
                "commit() must clean it up once the tag moves off it, not leave it dangling"
            )
        finally:
            sb.destroy()

    @podman
    def test_real_squash_cycle_reclaims_its_own_superseded_commit(self):
        """The payoff of squashing: the transient plain-commit image
        commit()'s own step 1 produces gets reclaimed the moment the squash
        (step 3) supersedes it, since nothing was ever `run` from that
        transient image. Mirrors test_docker.py's confirmed-live version of
        this test (see its docstring for the full reasoning on why this is
        a narrower guarantee than the OLD design's "reclaims the WHOLE
        accumulated chain"). Unverified against a real podman daemon in the
        environment this was written in (podman wasn't reachable there).

        Uses agency-sandbox:latest, not a bare `alpine:latest` -- see
        test_real_plain_commit_cycle_cleans_up_previous_sibling's docstring
        for why a bare name that resolves to `localhost/alpine:latest` via
        Podman's `_resolve_image()` fails outright unless that exact tag was
        already pulled locally."""
        import agency.agsandbox_backends.podman as _mod

        sb = _make_sandbox()
        sb._backend._base_image = "agency-sandbox:latest"

        def _dangling_ids():
            r = subprocess.run(
                ["podman", "images", "-f", "dangling=true", "-q"],
                capture_output=True,
                text=True,
            )
            return set(ln.strip() for ln in r.stdout.splitlines() if ln.strip())

        base_layers_before = subprocess.run(
            [
                "podman",
                "inspect",
                "--format={{json .RootFS.Layers}}",
                "localhost/agency-sandbox:latest",
            ],
            capture_output=True,
            text=True,
        ).stdout

        try:
            sb.exec("echo one")
            before = _dangling_ids()
            with patch.object(_mod._PodmanBackend, "checkpoint_squash_max_depth", 1):
                sb.commit()
            after = _dangling_ids()

            assert after - before == set(), (
                "a squashing commit() must not leave its own superseded "
                f"plain-commit image behind as a dangling leftover: {after - before}"
            )

            base_layer_count = len(json.loads(base_layers_before))
            tag = sb._backend._lifecycle_tag()
            layers = int(
                subprocess.run(
                    ["podman", "inspect", "--format={{len .RootFS.Layers}}", tag],
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            )
            assert layers in (1, base_layer_count + 1), (
                f"expected either a fully flattened (1) or fast-path-merged "
                f"({base_layer_count + 1}) result, got {layers}"
            )
        finally:
            sb.destroy()

    @podman
    def test_repeated_commits_leave_no_dangling_images(self):
        """commit() called 3 times to the same tag must leave 0 new dangling images."""
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
# Layer-depth squashing -- mirrors test_docker.py's TestCheckpointSquash
# against _PodmanBackend instead. See container.py's _squash_commit() and
# docs/agsandbox_backends/container.md's "Layer-depth squashing" section.
# ---------------------------------------------------------------------------


class _FakeCompleted:
    def __init__(self, stdout=b"", returncode=0):
        self.stdout = stdout
        self.returncode = returncode


class TestCheckpointSquash:
    """Squashing is triggered purely by the chain's actual current depth
    (`checkpoint_squash_max_depth`) -- the `force_squash=True` escape hatch
    is gone entirely; see test_docker.py's mirror of this class for the
    full rationale. Where a test needs squashing to trigger without a large
    real chain, checkpoint_squash_max_depth is patched down to a tiny value
    instead."""

    def test_squashes_when_depth_at_or_above_max_depth(self):
        """Mirrors test_docker.py's version -- the plain commit always
        happens first (unchanged), and squashing is a separate,
        additional step; since the mocked `podman info` returns no
        parseable JSON here, the fast accumulator path can't be trusted
        and falls back to `_squash_commit()`'s export/import."""
        import agency.agsandbox_backends.podman as _mod

        sb = _make_backend()
        max_depth = sb.checkpoint_squash_max_depth
        deep_chain = [f"sha256:layer{i}" for i in range(max_depth)]
        calls = []

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            calls.append(list(args))
            if "--format={{json .RootFS.Layers}}" in args:
                return _FakeCompleted(stdout=json.dumps(deep_chain).encode())
            return _FakeCompleted()

        with patch.object(_mod._PodmanBackend, "_run", fake_run):
            with patch.object(sb, "_container_status", return_value="running"):
                with patch.object(sb, "_gpu_virtual", False):
                    sb.commit()

        assert any(c[:2] == [sb._runtime, "commit"] for c in calls), (
            f"expected the normal plain commit to still happen: {calls}"
        )
        assert any("export" in c for c in calls), (
            f"expected the squash fallback's export call once depth reaches the cap: {calls}"
        )
        assert any("import" in c for c in calls), (
            f"expected the squash fallback's import call: {calls}"
        )

    def test_does_not_squash_when_depth_is_below_max_depth(self):
        """Mirrors test_docker.py's version."""
        import agency.agsandbox_backends.podman as _mod

        sb = _make_backend()
        shallow_chain = ["sha256:layer0", "sha256:layer1"]
        calls = []

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            calls.append(list(args))
            if "--format={{json .RootFS.Layers}}" in args:
                return _FakeCompleted(stdout=json.dumps(shallow_chain).encode())
            return _FakeCompleted()

        with patch.object(_mod._PodmanBackend, "_run", fake_run):
            with patch.object(sb, "_container_status", return_value="running"):
                with patch.object(sb, "_gpu_virtual", False):
                    sb.commit()

        assert any(c[:2] == [sb._runtime, "commit"] for c in calls), (
            f"expected a plain commit: {calls}"
        )
        assert not any("export" in c for c in calls), f"must not squash yet: {calls}"

    def test_squash_check_failure_does_not_raise_and_skips_squash(self):
        """Mirrors test_docker.py's version."""
        import agency.agsandbox_backends.podman as _mod

        sb = _make_backend()
        calls = []

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            calls.append(list(args))
            if "--format={{json .RootFS.Layers}}" in args:
                return _FakeCompleted(stdout=b"not json")
            return _FakeCompleted()

        with patch.object(_mod._PodmanBackend, "_run", fake_run):
            with patch.object(sb, "_container_status", return_value="running"):
                with patch.object(sb, "_gpu_virtual", False):
                    sb.commit()  # must not raise

        assert not any("export" in c for c in calls), f"must not squash: {calls}"

    def test_squash_pipes_export_stdout_into_import_stdin(self):
        """checkpoint_squash_max_depth is patched down to 1 to force this
        cycle to squash instead of the removed force_squash= parameter."""
        import agency.agsandbox_backends.podman as _mod

        sb = _make_backend()
        captured = {}
        fake_tar_bytes = b"FAKE_EXPORTED_FILESYSTEM_BYTES"

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "export" in args:
                return _FakeCompleted(stdout=fake_tar_bytes)
            if "import" in args:
                captured["input"] = input
                captured["args"] = list(args)
                return _FakeCompleted()
            if "--format={{json .RootFS.Layers}}" in args:
                return _FakeCompleted(stdout=b'["sha256:layer0"]')
            return _FakeCompleted()

        with patch.object(_mod._PodmanBackend, "checkpoint_squash_max_depth", 1):
            with patch.object(_mod._PodmanBackend, "_run", fake_run):
                with patch.object(sb, "_container_status", return_value="running"):
                    with patch.object(sb, "_gpu_virtual", False):
                        sb.commit()

        assert captured["input"] == fake_tar_bytes
        assert captured["args"][-1] == sb._lifecycle_tag()
        assert captured["args"][-2] == "-"  # import reads from stdin, not a file path

    def test_squash_failure_is_best_effort(self):
        """Mirrors test_docker.py's version: commit() must NOT raise when
        every squash path fails, since the normal plain commit above it
        already succeeded. checkpoint_squash_max_depth is patched down to 1
        to force this cycle to squash."""
        import agency.agsandbox_backends.podman as _mod

        sb = _make_backend()

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "export" in args or "import" in args:
                raise RuntimeError("simulated export/import failure")
            if "--format={{json .RootFS.Layers}}" in args:
                return _FakeCompleted(stdout=b'["sha256:layer0"]')
            return _FakeCompleted()

        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        try:
            with patch.object(_mod._PodmanBackend, "checkpoint_squash_max_depth", 1):
                with patch.object(_mod._PodmanBackend, "_run", fake_run):
                    with patch.object(sb, "_container_status", return_value="running"):
                        with patch.object(sb, "_gpu_virtual", False):
                            sb.commit()  # must not raise
        finally:
            sys.stderr = old_stderr

        assert "WARNING" in captured.getvalue()
        assert "squash failed" in captured.getvalue()

    @podman
    @pytest.mark.timeout(180)
    def test_squash_flattens_real_layer_depth(self):
        """Uses the tiny local `alpine` image rather than the real
        (multi-GB) agency-sandbox image -- this only needs to exercise the
        export/import mechanism itself. Mirrors test_docker.py's version;
        unverified in the environment this was written in (podman wasn't
        reachable there)."""

        def _layer_count(ref):
            r = subprocess.run(
                ["podman", "inspect", "--format={{len .RootFS.Layers}}", ref],
                capture_output=True,
                text=True,
            )
            return int(r.stdout.strip())

        name = f"test-squash-{uuid.uuid4().hex[:8]}"
        tag = f"agency/lifecycle-{name}"
        subprocess.run(
            ["podman", "run", "-d", "--name", name, "alpine:latest", "tail", "-f", "/dev/null"],
            capture_output=True,
            check=True,
        )
        try:
            for i in range(3):
                subprocess.run(
                    ["podman", "exec", name, "sh", "-c", f"echo {i} > /marker-{i}"],
                    capture_output=True,
                    check=True,
                )
                subprocess.run(["podman", "commit", name, tag], capture_output=True, check=True)
            depth_before = _layer_count(tag)
            assert depth_before > 1, "expected multiple stacked layers before squashing"

            export_proc = subprocess.run(
                ["podman", "export", name], capture_output=True, check=True
            )
            subprocess.run(
                ["podman", "import", "-", tag],
                input=export_proc.stdout,
                capture_output=True,
                check=True,
            )
            depth_after = _layer_count(tag)
            assert depth_after == 1, f"expected depth 1 after squash, got {depth_after}"
        finally:
            subprocess.run(["podman", "rm", "-f", name], capture_output=True)
            subprocess.run(["podman", "rmi", "-f", tag], capture_output=True)


# ---------------------------------------------------------------------------
# Fast incremental squashing -- mirrors test_docker.py's
# TestLocateLayerDiffDir / TestHostToContainerId / end-to-end accumulator
# coverage against _PodmanBackend's containers/storage overlay layout.
# See docs/agsandbox_backends/container.md's "Fast incremental squashing"
# section and podman.py's module docstring.
# ---------------------------------------------------------------------------


class TestLocateLayerDiffDir:
    """Tests for _PodmanBackend._locate_layer_diff_dir() -- reaches into
    Podman's containers/storage overlay on-disk layout. Uses REAL temp
    directories structured to match that layout (layers.json +
    overlay/<id>/diff dirs), not a real podman daemon -- this is pure
    filesystem-correlation logic once `podman info`'s own result is
    known, so mocking that one call is enough to exercise it fully."""

    def _sb(self):
        return _make_backend()

    def _fake_podman_root(self, tmp_path, *, driver="overlay"):
        root = tmp_path / "podman-root"
        (root / "overlay-layers").mkdir(parents=True)
        (root / "overlay").mkdir(parents=True)
        (root / "overlay-layers" / "layers.json").write_text("[]")
        return root

    def _add_layer_entry(self, root, diff_id, layer_id, *, with_content=True):
        layers_path = root / "overlay-layers" / "layers.json"
        layers = json.loads(layers_path.read_text())
        layers.append({"id": layer_id, "diff-digest": diff_id})
        layers_path.write_text(json.dumps(layers))
        if with_content:
            diff_dir = root / "overlay" / layer_id / "diff"
            diff_dir.mkdir(parents=True)
            (diff_dir / "marker").write_text("x")
        return layer_id

    def test_finds_diff_dir_matching_digest(self, tmp_path):
        import agency.agsandbox_backends.podman as _mod

        sb = self._sb()
        root = self._fake_podman_root(tmp_path)
        self._add_layer_entry(root, "sha256:target123", "layer-target")
        self._add_layer_entry(root, "sha256:other456", "layer-other")

        with patch.object(
            _mod._PodmanBackend, "_podman_graph_root_and_driver", return_value=(root, "overlay")
        ):
            result = sb._locate_layer_diff_dir("sha256:target123")

        assert result == root / "overlay" / "layer-target" / "diff"
        assert (result / "marker").read_text() == "x"

    def test_prefers_last_matching_entry_when_digest_duplicated(self, tmp_path):
        """Empty layers commonly share a digest; layers.json is
        append-ordered, so the last match is the newest."""
        import agency.agsandbox_backends.podman as _mod

        sb = self._sb()
        root = self._fake_podman_root(tmp_path)
        self._add_layer_entry(root, "sha256:shared", "layer-older")
        self._add_layer_entry(root, "sha256:shared", "layer-newer")

        with patch.object(
            _mod._PodmanBackend, "_podman_graph_root_and_driver", return_value=(root, "overlay")
        ):
            result = sb._locate_layer_diff_dir("sha256:shared")

        assert result == root / "overlay" / "layer-newer" / "diff"

    def test_returns_none_when_digest_not_found(self, tmp_path):
        import agency.agsandbox_backends.podman as _mod

        sb = self._sb()
        root = self._fake_podman_root(tmp_path)
        self._add_layer_entry(root, "sha256:other456", "layer-other")

        with patch.object(
            _mod._PodmanBackend, "_podman_graph_root_and_driver", return_value=(root, "overlay")
        ):
            result = sb._locate_layer_diff_dir("sha256:nonexistent")

        assert result is None

    def test_returns_none_for_non_overlay_driver(self, tmp_path):
        import agency.agsandbox_backends.podman as _mod

        sb = self._sb()
        root = self._fake_podman_root(tmp_path)
        self._add_layer_entry(root, "sha256:target123", "layer-target")

        with patch.object(
            _mod._PodmanBackend, "_podman_graph_root_and_driver", return_value=(root, "vfs")
        ):
            result = sb._locate_layer_diff_dir("sha256:target123")

        assert result is None

    def test_returns_none_when_info_lookup_fails(self):
        import agency.agsandbox_backends.podman as _mod

        sb = self._sb()
        with patch.object(_mod._PodmanBackend, "_podman_graph_root_and_driver", return_value=None):
            result = sb._locate_layer_diff_dir("sha256:anything")
        assert result is None

    def test_returns_none_when_layers_json_entry_present_but_content_missing(self, tmp_path):
        """A matching layers.json entry whose overlay diff dir doesn't
        actually exist (e.g. already cleaned up) must degrade to None,
        not raise."""
        import agency.agsandbox_backends.podman as _mod

        sb = self._sb()
        root = self._fake_podman_root(tmp_path)
        self._add_layer_entry(root, "sha256:target123", "layer-target", with_content=False)

        with patch.object(
            _mod._PodmanBackend, "_podman_graph_root_and_driver", return_value=(root, "overlay")
        ):
            result = sb._locate_layer_diff_dir("sha256:target123")

        assert result is None

    def test_returns_none_when_layers_json_missing(self, tmp_path):
        import agency.agsandbox_backends.podman as _mod

        sb = self._sb()
        root = self._fake_podman_root(tmp_path)
        (root / "overlay-layers" / "layers.json").unlink()

        with patch.object(
            _mod._PodmanBackend, "_podman_graph_root_and_driver", return_value=(root, "overlay")
        ):
            result = sb._locate_layer_diff_dir("sha256:anything")

        assert result is None

    @podman
    def test_real_commit_layer_locates_via_live_storage(self):
        """End-to-end against a real podman commit: the top RootFS.Layers
        digest must resolve to a real overlay diff dir containing the
        file we wrote (no mocking of info/storage)."""
        name = f"test-locate-{uuid.uuid4().hex[:8]}"
        tag = f"localhost/agency-test-locate-{uuid.uuid4().hex[:8]}:latest"
        subprocess.run(
            ["podman", "run", "-d", "--name", name, "alpine:latest", "sleep", "3600"],
            capture_output=True,
            check=True,
        )
        sb = self._sb()
        try:
            subprocess.run(
                ["podman", "exec", name, "sh", "-c", "echo locate-marker > /tmp/locate.txt"],
                capture_output=True,
                check=True,
            )
            subprocess.run(["podman", "commit", name, tag], capture_output=True, check=True)
            layers = json.loads(
                subprocess.run(
                    ["podman", "inspect", "--format={{json .RootFS.Layers}}", tag],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
            )
            diff_dir = sb._locate_layer_diff_dir(layers[-1])
            assert diff_dir is not None, f"failed to locate diff dir for {layers[-1]}"
            assert (diff_dir / "tmp" / "locate.txt").is_file()
            assert (diff_dir / "tmp" / "locate.txt").read_text() == "locate-marker\n"
        finally:
            subprocess.run(["podman", "rm", "-f", name], capture_output=True)
            subprocess.run(["podman", "rmi", "-f", tag], capture_output=True)


class TestHostToContainerId:
    """Tests for _PodmanBackend._host_to_container_id() -- the rootless
    Podman uid/gid translation feeding overlay_diff_to_tar() via
    `_build_accumulator_for_squash()`. Mocks `_podman_info()` rather than
    a real rootless setup; the reverse-mapping arithmetic itself is
    exercised against idMappings shaped like a real `podman info`."""

    def _sb(self):
        return _make_backend()

    def test_identity_when_not_rootless(self):
        import agency.agsandbox_backends.podman as _mod

        sb = self._sb()
        with patch.object(
            _mod._PodmanBackend,
            "_podman_info",
            return_value={"host": {"security": {"rootless": False}}},
        ):
            assert sb._host_to_container_id(1000, 1000) == (1000, 1000)

    def test_identity_when_podman_info_unavailable(self):
        import agency.agsandbox_backends.podman as _mod

        sb = self._sb()
        with patch.object(_mod._PodmanBackend, "_podman_info", return_value=None):
            assert sb._host_to_container_id(1000, 1000) == (1000, 1000)

    def test_identity_when_rootless_but_maps_unavailable(self):
        import agency.agsandbox_backends.podman as _mod

        sb = self._sb()
        with patch.object(
            _mod._PodmanBackend,
            "_podman_info",
            return_value={
                "host": {
                    "security": {"rootless": True},
                    "idMappings": {},
                }
            },
        ):
            assert sb._host_to_container_id(1000, 1000) == (1000, 1000)

    def test_translates_using_podman_info_id_mappings(self):
        """idMappings shape captured from a real rootless `podman info`:
        container uid 0 maps to host uid 1000, and container uids 1-65536
        map to host 100000+."""
        import agency.agsandbox_backends.podman as _mod

        sb = self._sb()
        info = {
            "host": {
                "security": {"rootless": True},
                "idMappings": {
                    "uidmap": [
                        {"container_id": 0, "host_id": 1000, "size": 1},
                        {"container_id": 1, "host_id": 100000, "size": 65536},
                    ],
                    "gidmap": [
                        {"container_id": 0, "host_id": 1000, "size": 1},
                        {"container_id": 1, "host_id": 100000, "size": 65536},
                    ],
                },
            }
        }
        with patch.object(_mod._PodmanBackend, "_podman_info", return_value=info):
            assert sb._host_to_container_id(1000, 1000) == (0, 0)
            assert sb._host_to_container_id(100001, 100001) == (2, 2)

    def test_translate_id_leaves_unmapped_host_id_unchanged(self):
        from agency.agsandbox_backends.podman import _translate_id

        assert _translate_id(999999, [(0, 1000, 1)]) == 999999


class TestCheckpointAccumulator:
    """Tests for the fast squash path fed by TestLocateLayerDiffDir's
    lookup, against _PodmanBackend. Mirrors the fold/accumulator checks
    in test_docker.py; the shared merge/load logic lives on the base
    class, so these confirm the Podman hooks wire into it correctly."""

    def _sb(self):
        return _make_backend()

    def _make_real_diff_dir(self, tmp_path, name, files):
        d = tmp_path / name
        d.mkdir()
        for rel, content in files.items():
            p = d / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        return d

    def test_build_accumulator_from_overlay_diff_dir(self, tmp_path):
        import tarfile

        import agency.agsandbox_backends.podman as _mod

        sb = self._sb()
        sb._squash_base_diff_ids = []
        diff_dir = self._make_real_diff_dir(tmp_path, "diff1", {"workspace/f1": "one"})

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "--format={{json .RootFS.Layers}}" in args:
                return _FakeCompleted(stdout=b'["sha256:layer1"]')
            return _FakeCompleted()

        with patch.object(_mod._PodmanBackend, "_run", fake_run):
            with patch.object(_mod._PodmanBackend, "_locate_layer_diff_dir", return_value=diff_dir):
                sb._build_accumulator_for_squash("some-tag")

        assert sb._accumulated_layer_count == 1
        assert sb._accumulated_diff_path is not None
        with tarfile.open(sb._accumulated_diff_path, "r") as tf:
            content = tf.extractfile("workspace/f1").read()
        assert content == b"one"
        sb._reset_accumulator()

    def test_build_accumulator_raises_when_diff_dir_not_found(self):
        import agency.agsandbox_backends.podman as _mod

        sb = self._sb()
        sb._squash_base_diff_ids = []

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "--format={{json .RootFS.Layers}}" in args:
                return _FakeCompleted(stdout=b'["sha256:layer1"]')
            return _FakeCompleted()

        with patch.object(_mod._PodmanBackend, "_run", fake_run):
            with patch.object(_mod._PodmanBackend, "_locate_layer_diff_dir", return_value=None):
                with pytest.raises(RuntimeError, match="could not locate on-disk diff"):
                    sb._build_accumulator_for_squash("some-tag")

        assert sb._accumulated_diff_path is None
        assert sb._accumulated_layer_count == 0

    @podman
    @pytest.mark.timeout(180)
    def test_real_end_to_end_fast_squash_against_base_image(self):
        """Several plain-commit cycles against localhost/agency-sandbox:latest,
        each folding via the real containers/storage lookup (no mocking),
        then a real squash -- must complete quickly, produce correct
        content, and leave the base image's own layers untouched.

        checkpoint_squash_max_depth is patched down to force the third
        commit() to squash instead of the removed force_squash= parameter.
        rm_container() is called between cycles so each commit() is a
        genuine incremental layer on top of the PREVIOUS checkpoint (a
        fresh container recreated FROM that checkpoint) rather than a
        repeated commit of the SAME never-recreated container -- confirmed
        empirically (see test_docker.py's mirror of this test) that the
        latter always yields a same-depth sibling image every time, never a
        growing chain, since `podman commit` always snapshots the
        container's cumulative writable-layer diff relative to its own
        fixed run-time base. Without the real incremental layers this
        produces, the accumulator's own fold-count bookkeeping mismatches
        the real chain depth and the fast path can't be trusted -- exactly
        the scenario this test exists to exercise."""
        import agency.agsandbox_backends.podman as _mod

        sb = _make_sandbox()

        base_ref = "localhost/agency-sandbox:latest"
        base_layers_before = subprocess.run(
            ["podman", "inspect", "--format={{json .RootFS.Layers}}", base_ref],
            capture_output=True,
            text=True,
        ).stdout

        sb.exec("mkdir -p /workspace/proj && echo one > /workspace/proj/f1")
        sb.commit()
        sb.rm_container()
        sb.exec("echo two > /workspace/proj/f2 && rm /workspace/proj/f1")
        sb.commit()
        sb.rm_container()
        sb.exec("mkdir -p /workspace/proj/sub && echo three > /workspace/proj/sub/f3")

        t0 = time.time()
        with patch.object(_mod._PodmanBackend, "checkpoint_squash_max_depth", 1):
            sb.commit()  # this cycle commits AND squashes
        elapsed = time.time() - t0

        try:
            assert elapsed < 20, f"fast squash took {elapsed:.1f}s -- expected well under 20s"

            base_layers_after = subprocess.run(
                ["podman", "inspect", "--format={{json .RootFS.Layers}}", base_ref],
                capture_output=True,
                text=True,
            ).stdout
            assert base_layers_after == base_layers_before

            tag = sb._backend._lifecycle_tag()
            layers = int(
                subprocess.run(
                    ["podman", "inspect", "--format={{len .RootFS.Layers}}", tag],
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
                    "podman",
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

    def test_ensure_started_resumes_hibernating_container(self):
        """When _inspect_container_state() reports (running=False, status=
        truthy) -- i.e. hibernating via stop(), not removed -- _ensure_
        started() must resume it in place via `podman start`, NOT
        force-remove and recreate it. Mirrors test_docker.py's version of
        this test; see there for the full reasoning, including why
        _container_running()/_container_status() are patched too (as a
        tripwire against a regression back to calling them directly instead
        of the merged _inspect_container_state())."""
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
            with patch.object(sb, "_container_running", return_value=True):
                with patch.object(sb, "_container_status", return_value=""):
                    with patch.object(
                        sb, "_inspect_container_state", return_value=(False, "exited", None)
                    ):
                        sb._ensure_started()

        rm_calls = [(a, c) for (a, c) in calls if "rm" in a]
        assert not rm_calls, f"must not remove a hibernating container; got {rm_calls}"
        start_calls = [(a, c) for (a, c) in calls if "start" in a]
        assert start_calls, "expected `podman start` to resume the hibernating container"
        assert start_calls[0][0] == [sb._runtime, "start", sb._name], (
            f"unexpected start invocation: {start_calls[0][0]}"
        )

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

    def test_rm_container_releases_gpu_when_already_confirmed_gone(self):
        """rm_container()'s early-return branch (no container at all when
        rm_container() is entered) must still release a leased GPU. stop()
        (hibernate) never releases the GPU any more -- see test_docker.py's
        mirror of this test for the full reasoning."""
        import agency.agsandbox_backends.podman as _mod

        sb = self._sb()
        released = self._lease_gpu(sb)

        with patch.object(_mod._PodmanBackend, "_run"):
            with patch.object(sb, "_container_status", return_value=""):
                sb.rm_container()

        assert released == [3]
        assert sb._gpu_id is None

    def test_rm_container_does_not_release_gpu_when_rm_fails_and_container_still_running(self):
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
            with patch.object(sb, "_container_status", return_value="running"):
                with patch.object(sb, "_container_running", return_value=True):
                    with patch.object(_container_mod.time, "sleep"):  # skip real retry backoff
                        with pytest.raises(RuntimeError, match="rm exploded"):
                            sb.rm_container()

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

    def test_gpu_released_exactly_once_across_rm_container_then_destroy(self):
        """rm_container() tears the container down and releases the GPU; a
        later destroy() call on the same sandbox must see gpu_id already
        cleared and must not release a second time."""
        import agency.agsandbox_backends.podman as _mod

        sb = self._sb()
        released = self._lease_gpu(sb)

        with patch.object(_mod._PodmanBackend, "_run"):
            with patch.object(sb, "_container_status", return_value=""):
                with patch.object(sb, "_container_running", return_value=False):
                    sb.rm_container()
                    sb.destroy()

        assert released == [3], "GPU must be released exactly once, not once per call"
