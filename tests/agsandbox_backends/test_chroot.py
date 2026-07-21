"""Unit and integration tests for the chroot sandbox backend.

Tests that actually chroot are marked with @chroot and skipped automatically
when unprivileged user namespaces aren't usable on this host (see
agency.agsandbox_backends.chroot.chroot_available())."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import uuid

import pytest
from unittest.mock import patch

from agency.agsandbox_backends.chroot import (
    chroot_available,
    _ChrootBackend,
    _CHROOT_STATE_ROOT,
    _chroot_gpu_dev_paths,
    _sanitize_tag,
)
import agency.agsandbox_backends.chroot as _chroot_mod

chroot = pytest.mark.skipif(
    not chroot_available(), reason="unprivileged user namespaces not usable"
)


def _make_backend(**kwargs):
    name = f"chroot-test-{uuid.uuid4().hex[:8]}"
    defaults = dict(
        agname=name,
        name=name,
        checkpoint_image=None,
        mounts={},
        agconfig=None,
    )
    defaults.update(kwargs)
    return _ChrootBackend(defaults.pop("agname"), **defaults)


@pytest.fixture(autouse=True)
def _clean_state_root():
    shutil.rmtree(_CHROOT_STATE_ROOT, ignore_errors=True)
    yield
    shutil.rmtree(_CHROOT_STATE_ROOT, ignore_errors=True)


# ---------------------------------------------------------------------------
# _sanitize_tag
# ---------------------------------------------------------------------------


class TestSanitizeTag:
    def test_replaces_slash(self):
        assert "/" not in _sanitize_tag("agency/lifecycle-foo")

    def test_replaces_colon(self):
        assert ":" not in _sanitize_tag("agency/lifecycle-foo:v1")

    def test_distinct_tags_stay_distinct(self):
        assert _sanitize_tag("a/b") != _sanitize_tag("a-b")


# ---------------------------------------------------------------------------
# _own_host_pids -- chroot processes run directly on the host (no PID
# namespace to translate through, unlike _ContainerBackendBase's version),
# and now delegates straight to get_live_pids() (PGID-matched -- see
# chroot.py's module docstring), so a fake /proc table via monkeypatched
# _read_proc_table() is enough to test this, no real chroot jail needed.
# GPU release itself no longer waits on anything backend-specific at all:
# release_gpu() just releases immediately, called after stop()/destroy()'s
# own kill+teardown has already run synchronously (see
# agresources.release_gpu()'s docstring).
# ---------------------------------------------------------------------------


class TestOwnHostPids:
    def test_matches_pgid_matched_pids(self, monkeypatch):
        sb = _make_backend()
        sb._invocation_pgids = {4242}
        monkeypatch.setattr(
            sb,
            "_read_proc_table",
            lambda script, timeout: ("111 1 4242 S\n222 1 4242 S\n", 0),
        )
        assert sb._own_host_pids() == {111, 222}

    def test_empty_when_nothing_tracked(self):
        sb = _make_backend()
        assert sb._own_host_pids() == set()


class TestChrootDoesNotAdoptUnwatchedLivePids:
    def test_flag_disabled(self):
        assert _make_backend()._adopt_unwatched_live_pids is False

    def test_get_live_pids_excludes_unrelated_process_groups(self, monkeypatch):
        """A live process outside every tracked _invocation_pgids must never
        appear in get_live_pids() -- PGID matching, not host-wide presence,
        decides membership here."""
        sb = _make_backend()
        sb._invocation_pgids = {4242}
        # Fake /proc table: tracked-group member 111 alive, unrelated stranger
        # 999999 (a different, untracked pgid) also alive.
        monkeypatch.setattr(
            sb,
            "_read_proc_table",
            lambda script, timeout: ("111 1 4242 S\n999999 1 555 S\n", 0),
        )
        live = sb.get_live_pids()
        assert live == {111}


# ---------------------------------------------------------------------------
# _has_pending_background_work / _live_pgid_matched_pids /
# _kill_all_sandbox_processes -- this backend tracks background work purely
# via process groups (see chroot.py's module docstring). Each exec() call's
# underlying unshare invocation is its own new process-group leader (see
# _run_unshared()), and that pgid is recorded into _invocation_pgids. A
# process-group match is a real, kernel-tracked relationship (not a timing
# guess the way an earlier "non-baseline since jail start" scan was) --
# immune to host churn, and confirmed empirically to survive a child
# outliving its exited parent's reparenting, as well as to catch a child
# spawned well after the exec() call that backgrounded its parent already
# returned (something an earlier before/after-diff mechanism, since
# removed, could never see). It's also verified enough to act on
# destructively via os.killpg(), unlike a baseline scan, which could never
# confirm a candidate pid's ownership well enough to justify SIGKILLing it.
#
# (GPU release itself doesn't need any of this: release_gpu() no longer
# waits/polls on anything at all -- see agresources.release_gpu()'s
# docstring for why the caller's own kill+teardown sequence, which already
# ran before release_gpu() is even called, is sufficient confirmation.)
#
# The accepted trade-off: a process that calls setsid()/setpgid() to detach
# into its own new group (nohup/setsid/disown and similar daemonizing
# idioms) escapes this tracking entirely -- confirmed empirically (see
# TestChrootDelayedChildSafety.test_setsid_detached_daemon_is_not_tracked
# below) and accepted deliberately: the alternative (a baseline scan) traded
# that same blind spot for a *worse* one, making
# _has_pending_background_work() never resolve to "done" quickly on any host
# with background churn -- confirmed empirically too.
# ---------------------------------------------------------------------------


class TestChrootInvocationPgidTracking:
    def _sb(self, **kwargs):
        sb = _make_backend()
        sb._daemon_pids = set()
        sb._watched_pids = {}
        sb._invocation_pgids = {4242}
        for k, v in kwargs.items():
            setattr(sb, k, v)
        return sb

    def test_no_tracked_groups_no_pending_work_without_scanning(self, monkeypatch):
        """A fast path: with nothing in _invocation_pgids, don't even bother
        running a /proc scan."""
        sb = self._sb(_invocation_pgids=set())

        def _fail(*a, **kw):
            raise AssertionError("must not scan /proc when there are no tracked groups")

        monkeypatch.setattr(sb, "_read_proc_table", _fail)
        assert sb._has_pending_background_work() is False

    def test_matching_pgid_flags_pending_work(self, monkeypatch):
        """The exact regression: a pid that was NEVER in _watched_pids (a
        delayed child of an already-exited, already-untracked parent) must
        still be found via its inherited pgid and flag pending work."""
        sb = self._sb()
        monkeypatch.setattr(sb, "_read_proc_table", lambda script, timeout: ("999 1 4242 S\n", 0))
        assert sb._has_pending_background_work() is True
        assert sb._invocation_pgids == {4242}, "still-alive group must not be pruned"

    def test_no_matching_pgid_prunes_group(self, monkeypatch):
        """Once a scan finds nothing alive in a tracked group, that group is
        confirmed permanently gone (a pgid ceases to exist once its last
        member exits) and safe to forget."""
        sb = self._sb()
        monkeypatch.setattr(
            sb,
            "_read_proc_table",
            lambda script, timeout: ("999 1 555 S\n", 0),  # unrelated pgid
        )
        assert sb._has_pending_background_work() is False
        assert sb._invocation_pgids == set()

    def test_zombie_in_matching_group_excluded(self, monkeypatch):
        sb = self._sb()
        monkeypatch.setattr(sb, "_read_proc_table", lambda script, timeout: ("999 1 4242 Z\n", 0))
        assert sb._has_pending_background_work() is False

    def test_daemon_descendant_in_matching_group_excluded(self, monkeypatch):
        """A child of an explicitly-released daemon must not count as
        pending work, matching get_live_pids()'s own daemon-propagation --
        even though it shares a tracked invocation's pgid."""
        sb = self._sb(_daemon_pids={500})
        monkeypatch.setattr(
            sb, "_read_proc_table", lambda script, timeout: ("500 1 4242 S\n999 500 4242 S\n", 0)
        )
        assert sb._has_pending_background_work() is False

    def test_kill_all_sandbox_processes_kills_process_groups(self):
        sb = self._sb()
        killed_pgids = []
        with patch("os.killpg", side_effect=lambda pgid, sig: killed_pgids.append(pgid)):
            sb._kill_all_sandbox_processes()
        assert killed_pgids == [4242]

    def test_kill_all_sandbox_processes_ignores_already_dead_groups(self):
        sb = self._sb()

        def _raise_killpg(pgid, sig):
            raise ProcessLookupError()

        with patch("os.killpg", side_effect=_raise_killpg):
            sb._kill_all_sandbox_processes()  # must not raise


# ---------------------------------------------------------------------------
# Live background-process tracking -- the chroot analogue of
# test_agsandbox.py's TestAgSandboxPIDTracking (@docker). Regression coverage
# for two real, previously-silent bugs, both found and fixed against a real
# jail rather than assumed from reading the code:
#
# 1. Before the /proc-visibility fix in agsandbox_backends/base.py + chroot.py,
#    mounting (fresh OR bind) a procfs inside the jail silently failed (kernel
#    EINVAL -- see chroot.py's module docstring), so `/proc` inside every jail
#    was always empty and _watched_pids/get_live_pids() never tracked a
#    single background process, no matter what a command spawned.
#
# 2. _exec_with_pid_tracking()'s chroot sub-invocation used to capture output
#    via `$(...)` command substitution -- a real OS pipe, whose EOF (and so
#    $(...)'s own return) blocks until EVERY process holding the write end
#    closes it, including a backgrounded descendant of the chrooted command.
#    That silently made exec() block for the ENTIRE runtime of any `cmd &`
#    before ever reaching its own after-diff -- exactly the case this
#    mechanism exists to detect without waiting on. Confirmed empirically
#    (a `sleep 3 &` made exec() itself take ~3.8s) and fixed by capturing
#    output via a temp file instead, whose reads don't block on other
#    processes' open write handles the way a pipe's do.
#
# Unlike docker/podman, chroot has no isolated PID namespace of its own, so
# get_live_pids() here is PGID-matched rather than baseline-diffed (see
# chroot.py's module docstring): every PID sharing a tracked invocation's
# process group counts, including one spawned well after the exec() call
# that started that invocation already returned. get_live_pids() and
# _has_pending_background_work() are backed by the exact same scan, so they
# can never disagree with each other the way an earlier, since-removed
# before/after-diff `get_live_pids()` and PGID-based
# `_has_pending_background_work()` sometimes did.
# ---------------------------------------------------------------------------


@chroot
class TestChrootBackendPIDTracking:
    def setup_method(self, _):
        self.sb = _make_backend()

    def teardown_method(self, _):
        self.sb.destroy()

    def test_background_pid_tracked(self):
        self.sb.exec("sleep 5 &")
        assert len(self.sb.get_live_pids()) > 0

    def test_foreground_spawned_child_tracked(self):
        # A foreground command that internally forks a child and exits.
        # The child inherits the invocation's pgid, so PGID matching finds
        # it regardless of the shell's own job table.
        self.sb.write_file(
            "/workspace/spawner.py",
            ("import subprocess\nsubprocess.Popen(['sleep', '5'])\n"),  # detached, not waited on
        )
        self.sb.exec("python3 /workspace/spawner.py")
        assert len(self.sb.get_live_pids()) > 0

    def test_parent_exits_child_survives_still_tracked(self):
        # Parent spawns a child then exits. Child is reparented (to the real
        # host init, since there's no PID namespace of our own) but keeps
        # the same pgid regardless -- PGID matching finds it either way.
        self.sb.write_file(
            "/workspace/spawner.py",
            (
                "import subprocess, os\n"
                "subprocess.Popen(['sleep', '5'])\n"  # child detaches
                "os._exit(0)\n"  # parent exits immediately
            ),
        )
        self.sb.exec("python3 /workspace/spawner.py")
        assert len(self.sb.get_live_pids()) > 0

    def test_multiple_children_tracked_within_same_call(self):
        # A single exec() call whose command spawns multiple detached
        # children before it returns -- all of them inherit the same
        # invocation pgid, so a single tracked group covers the whole set.
        self.sb.write_file(
            "/workspace/parent.py",
            (
                "import subprocess\nsubprocess.Popen(['sleep', '5'])\nsubprocess.Popen(['sleep', '5'])\n"
            ),
        )
        self.sb.exec("python3 /workspace/parent.py")
        assert len(self.sb.get_live_pids()) >= 2

    def test_double_forked_daemon_tracked(self):
        # Classic Unix double-fork: grandchild is reparented to the real host
        # init and completely detached from the shell's job table, but keeps
        # the same pgid regardless of reparenting.
        self.sb.write_file(
            "/workspace/daemon.py",
            (
                "import os, time\n"
                "if os.fork() == 0:\n"  # first fork
                "    if os.fork() == 0:\n"  # second fork — grandchild
                "        time.sleep(5)\n"  # grandchild runs in background
                "    os._exit(0)\n"  # intermediate child exits
                "os.wait()\n"  # parent waits for intermediate child
            ),
        )
        self.sb.exec("python3 /workspace/daemon.py")
        assert len(self.sb.get_live_pids()) > 0

    def test_get_live_pids_returns_running(self):
        self.sb.exec("sleep 5 &")
        live = self.sb.get_live_pids()
        assert len(live) > 0

    def test_get_live_pids_removes_exited(self):
        marker = f"agencytest{uuid.uuid4().hex[:8]}"
        # Set argv[0] to a random, practically-unique marker via `exec -a` so
        # our own backgrounded process can be conclusively identified,
        # rather than assuming the whole set is ours -- this host's own
        # setup/mount overhead per exec() call (unshare + chroot + ~40 bind
        # mounts, no persistent daemon to cache it across calls) can itself
        # take upward of half a second. `sleep 3` gives comfortable margin
        # over that per-call overhead so our own process is genuinely still
        # running when tracked, without relying on an unrealistically short
        # duration a busy host could race past entirely.
        self.sb.exec(f"exec -a {marker} sleep 3 &")

        def _find_marked(pids):
            found = set()
            for pid in pids:
                try:
                    cmdline = open(f"/proc/{pid}/cmdline", "rb").read()
                except OSError:
                    continue
                if marker.encode() in cmdline:
                    found.add(pid)
            return found

        marked = _find_marked(self.sb.get_live_pids())
        assert marked, f"expected the marked sleep to be tracked, got {self.sb.get_live_pids()}"
        time.sleep(4.0)  # comfortable margin past sleep 3's own exit
        live = self.sb.get_live_pids()
        surviving = _find_marked(marked & live)
        assert not surviving, f"expected the marked sleep to have exited, still see {surviving}"

    def test_get_live_pids_empty_when_no_background(self):
        self.sb.exec("echo hi")
        assert self.sb.get_live_pids() == set()

    def test_daemon_release_removes_pid_from_monitoring(self):
        self.sb.exec("sleep 5 &")
        live = self.sb.get_live_pids()
        assert len(live) > 0
        for pid in list(live):
            self.sb.release_daemon(pid)
        assert self.sb.get_live_pids() == set()

    def test_daemon_children_also_excluded(self):
        # Release a parent as daemon (before its child even exists); a child
        # it spawns later must also be excluded via daemon-status propagation.
        self.sb.write_file(
            "/workspace/daemon_parent.py",
            (
                "import subprocess, time\n"
                "time.sleep(0.3)\n"  # ensure release happens before the child exists
                "subprocess.Popen(['sleep', '5'])\n"
                "time.sleep(5)\n"
            ),
        )
        self.sb.exec("python3 /workspace/daemon_parent.py &")
        parent_pids = self.sb.get_live_pids()
        assert parent_pids, "expected the backgrounded parent to be tracked"
        for pid in parent_pids:
            self.sb.release_daemon(pid)
        time.sleep(0.6)  # let the child spawn, now that its parent is already a daemon
        assert self.sb.get_live_pids() == set()

    def test_pid_status_summary_no_processes(self):
        summary = self.sb.pid_status_summary()
        assert "no background" in summary

    def test_pid_status_summary_with_running_process(self):
        # No elapsed-time figure here, unlike docker/podman -- there is no
        # per-PID capture timestamp once tracking is PGID-only (see
        # chroot.py's pid_status_summary() docstring).
        self.sb.exec("sleep 5 &")
        summary = self.sb.pid_status_summary()
        assert "PID" in summary


@chroot
class TestChrootDelayedChildSafety:
    """End-to-end reproduction, against a real jail, of the bug PGID-based
    tracking was introduced to fix: a child spawned by an already-tracked
    process AFTER that process's own exec() call already returned used to
    be invisible to the old before/after-diff `_watched_pids` mechanism,
    but not to `_has_pending_background_work()`'s pgid scan -- a real gap
    since closed by making get_live_pids() itself PGID-based too (see
    chroot.py's module docstring), so both now agree the child is alive."""

    def _spawn_delayed_child(self, sb):
        sb.write_file(
            "/workspace/parent.py",
            (
                "import subprocess, os, time\n"
                "time.sleep(0.5)\n"  # spawn after exec()'s own diff has already run
                "subprocess.Popen(['sleep', '20'])\n"
                "os._exit(0)\n"  # parent exits immediately, child detaches
            ),
        )
        sb.exec("python3 /workspace/parent.py &")
        # 1.5s, not 1.0s: confirmed empirically that under real host load
        # variance, python3 interpreter startup + this 0.5s internal delay
        # can occasionally take long enough that a tighter margin flakes.
        time.sleep(1.5)

    def test_has_pending_background_work_true_while_delayed_child_alive(self):
        sb = _make_backend()
        try:
            self._spawn_delayed_child(sb)
            # The top-level parent already exited via os._exit(0); only the
            # delayed child (spawned after exec() already returned) is still
            # running. PGID matching finds it via the invocation's own
            # tracked group, and get_live_pids()/_has_pending_background_work()
            # necessarily agree since both are backed by the same scan now.
            assert sb.get_live_pids(), (
                "expected the delayed child to still be found via PGID matching"
            )
            assert sb._has_pending_background_work() is True, (
                "a live process from this exact jail is still running -- "
                "wait_for_processes() must not skip waiting for it"
            )
        finally:
            sb.destroy()

    def test_wait_for_processes_does_not_report_clean_while_untracked_child_alive(self):
        from agency.agsandbox import agSandbox
        from agency.agconfig import agConfig
        from agency.agsandbox_backends import agSandboxBackendConfig

        cfg = agConfig(agSandboxBackendConfig(backend="chroot"))
        sb = agSandbox(str(uuid.uuid4()), agconfig=cfg)
        try:
            self._spawn_delayed_child(sb)
            msg = sb.wait_for_processes("test-skill", None, ping_interval_s=1, poll_interval_s=0.2)
            assert msg is not None, (
                "wait_for_processes() reported the sandbox as clean while a real "
                "background process from this exact sandbox was still running"
            )
        finally:
            sb.destroy()

    def test_setsid_detached_daemon_is_not_tracked(self):
        """The accepted, deliberate blind spot (see chroot.py's module
        docstring and TestChrootInvocationPgidTracking's class comment): a
        process that detaches into its own new process group via setsid
        (the same mechanism `nohup`/`disown`/many "daemonize" library
        patterns use) is invisible to _invocation_pgids, since it no longer
        shares its invocation's pgid. Pinned here explicitly so this is a
        visible, intentional decision rather than an implicit gap -- confirm
        it stays this way rather than silently regressing to "worse" (a
        crash) or silently improving to "better" (which would need a design
        change to rely on, not just an incidental fix)."""
        sb = _make_backend()
        try:
            sb.exec("setsid sleep 20 < /dev/null > /dev/null 2>&1 &")
            time.sleep(0.5)
            assert sb._has_pending_background_work() is False, (
                "documenting the accepted gap: a setsid-detached process is "
                "not caught by pgid-based tracking, unlike a plain background job"
            )
        finally:
            sb.destroy()


# ---------------------------------------------------------------------------
# _chroot_gpu_dev_paths(gpu_id) -- scopes the jail's GPU device bind-mounts to
# the single leased gpu_id (mirrors container.py's _gpu_flags(runtime,
# gpu_id) after this session's fix there) so a sandboxed process cannot open
# another leased jail's GPU device node directly by guessing its path. No
# real chroot/GPU required -- _all_chroot_gpu_dev_paths() (the raw host
# detection step) is mocked, isolating the scoping logic tested here from
# nvidia-smi/rocm-smi/real /dev contents.
# ---------------------------------------------------------------------------


class TestChrootGpuDevPaths:
    def test_gpu_id_none_returns_empty_even_with_gpus_on_host(self):
        with patch.object(
            _chroot_mod,
            "_all_chroot_gpu_dev_paths",
            return_value=["/dev/nvidiactl", "/dev/nvidia0"],
        ):
            assert _chroot_gpu_dev_paths(None) == []

    def test_no_gpu_devices_on_host_returns_empty(self):
        with patch.object(_chroot_mod, "_all_chroot_gpu_dev_paths", return_value=[]):
            assert _chroot_gpu_dev_paths(0) == []

    def test_nvidia_control_devices_plus_single_indexed_device(self):
        all_paths = [
            "/dev/nvidiactl",
            "/dev/nvidia-uvm",
            "/dev/nvidia-uvm-tools",
            "/dev/nvidia0",
            "/dev/nvidia1",
            "/dev/nvidia2",
        ]
        with patch.object(_chroot_mod, "_all_chroot_gpu_dev_paths", return_value=all_paths):
            with patch.object(_chroot_mod.shutil, "which", return_value="/usr/bin/nvidia-smi"):
                paths = _chroot_gpu_dev_paths(1)
        assert "/dev/nvidia1" in paths
        assert "/dev/nvidia0" not in paths
        assert "/dev/nvidia2" not in paths
        for control in ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"):
            assert control in paths

    def test_nvidia_gpu_id_out_of_range_returns_only_control_devices(self):
        all_paths = ["/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia0"]
        with patch.object(_chroot_mod, "_all_chroot_gpu_dev_paths", return_value=all_paths):
            with patch.object(_chroot_mod.shutil, "which", return_value="/usr/bin/nvidia-smi"):
                paths = _chroot_gpu_dev_paths(5)
        assert paths == ["/dev/nvidiactl", "/dev/nvidia-uvm"]

    def test_amd_control_device_plus_single_render_node(self):
        """amd_render_node_paths_by_pci_bus() is explicitly forced to None
        here (no rocm-smi/real sysfs in this test), which exercises the
        naive-sorted-order FALLBACK path specifically -- see
        test_amd_uses_pci_bus_ordering_when_available below for the mapping
        actually being applied, and TestChrootAmdRenderNodeLiveHardware for
        the real-hardware check that found sorted order alone is wrong on a
        multi-GPU AMD host."""
        all_paths = ["/dev/kfd", "/dev/dri/card0", "/dev/dri/renderD128", "/dev/dri/renderD129"]
        with patch.object(_chroot_mod, "_all_chroot_gpu_dev_paths", return_value=all_paths):
            with patch.object(_chroot_mod.shutil, "which", return_value=None):
                with patch.object(
                    _chroot_mod, "amd_render_node_paths_by_pci_bus", return_value=None
                ):
                    paths = _chroot_gpu_dev_paths(1)
        assert paths == ["/dev/kfd", "/dev/dri/renderD129"]

    def test_amd_gpu_id_out_of_range_returns_only_control_device(self):
        all_paths = ["/dev/kfd", "/dev/dri/renderD128"]
        with patch.object(_chroot_mod, "_all_chroot_gpu_dev_paths", return_value=all_paths):
            with patch.object(_chroot_mod.shutil, "which", return_value=None):
                with patch.object(
                    _chroot_mod, "amd_render_node_paths_by_pci_bus", return_value=None
                ):
                    paths = _chroot_gpu_dev_paths(5)
        assert paths == ["/dev/kfd"]

    def test_amd_uses_pci_bus_ordering_when_available(self):
        """When amd_render_node_paths_by_pci_bus() successfully builds a
        mapping, _chroot_gpu_dev_paths() must use ITS ordering, not naive
        sorted /dev/dri order -- this is the actual fix: on real 8x MI350X
        hardware the two disagree for every GPU (see agresources.py's
        amd_render_node_paths_by_pci_bus docstring)."""
        all_paths = ["/dev/kfd", "/dev/dri/renderD128", "/dev/dri/renderD129"]
        pci_ordered = ["/dev/dri/renderD129", "/dev/dri/renderD128"]  # reversed
        with patch.object(_chroot_mod, "_all_chroot_gpu_dev_paths", return_value=all_paths):
            with patch.object(_chroot_mod.shutil, "which", return_value=None):
                with patch.object(
                    _chroot_mod, "amd_render_node_paths_by_pci_bus", return_value=pci_ordered
                ):
                    paths_0 = _chroot_gpu_dev_paths(0)
                    paths_1 = _chroot_gpu_dev_paths(1)
        assert paths_0 == ["/dev/kfd", "/dev/dri/renderD129"]
        assert paths_1 == ["/dev/kfd", "/dev/dri/renderD128"]


class TestChrootSetupLinesGpuScoping:
    """Integration check that _setup_lines() actually wires
    _chroot_gpu_dev_paths(self._gpu_id) into the jail's /dev bind mounts,
    rather than every host GPU device -- no real chroot required, this only
    inspects the generated shell lines."""

    def test_only_the_scoped_gpu_device_is_bind_mounted(self, tmp_path, monkeypatch):
        scoped_dev = tmp_path / "nvidia1"
        scoped_dev.touch()
        sb = _make_backend()
        sb._gpu_id = 1
        monkeypatch.setattr(
            _chroot_mod,
            "_chroot_gpu_dev_paths",
            lambda gpu_id: [str(scoped_dev)] if gpu_id == 1 else [],
        )
        joined = "\n".join(sb._setup_lines())
        assert str(scoped_dev) in joined
        assert "nvidia0" not in joined

    def test_no_gpu_leased_bind_mounts_no_gpu_devices(self, monkeypatch):
        sb = _make_backend()
        sb._gpu_id = None
        monkeypatch.setattr(_chroot_mod, "_chroot_gpu_dev_paths", lambda gpu_id: [])
        joined = "\n".join(sb._setup_lines())
        assert "nvidia" not in joined
        assert "renderD" not in joined
        assert "kfd" not in joined


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


class TestChrootAmdRenderNodeLiveHardware:
    """Runs the real (unmocked) AMD GPU-to-render-node scoping against
    actual ROCm hardware -- no mocks anywhere. Skipped automatically without
    a real rocm-smi + AMD GPU(s). Mirrors
    test_container.py's TestAmdRenderNodeLiveHardware.

    This is the check that actually caught the bug being regression-tested
    here: on an 8x MI350X host, naive sorted /dev/dri/renderD* order did
    NOT correspond to rocm-smi's GPU index for ANY of the 8 GPUs (each GPU
    there exposes itself plus 7 XCD/compute-partition sibling render nodes
    -- 64 nodes total -- and even the primary node's number doesn't sort in
    GPU-index order). This test independently recomputes the "ground truth"
    GPU-to-render-node mapping from rocm-smi --showbus and
    /sys/class/drm/*/device -- without going through
    amd_render_node_paths_by_pci_bus itself -- and asserts the real
    _chroot_gpu_dev_paths() output agrees with it for every real GPU on
    this host.
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
    def test_chroot_gpu_dev_paths_match_gpu_pci_bus_on_real_hardware(self):
        _chroot_mod._gpu_dev_paths_cache = None
        bus_by_gpu_id = self._ground_truth_bus_by_gpu_id()
        assert bus_by_gpu_id, "expected at least one real AMD GPU"

        seen_devices = set()
        for gpu_id, expected_bus in bus_by_gpu_id.items():
            paths = _chroot_gpu_dev_paths(gpu_id)
            assert paths and os.path.basename(paths[0]) == "kfd"
            assert len(paths) == 2, f"expected a scoped render node for gpu_id={gpu_id}: {paths}"
            render_path = paths[1]
            actual_bus = self._real_render_node_bus(render_path)
            assert actual_bus == expected_bus, (
                f"gpu_id={gpu_id}: _chroot_gpu_dev_paths() picked {render_path} "
                f"(bus={actual_bus}), but rocm-smi says GPU {gpu_id} is actually "
                f"on bus {expected_bus}"
            )
            assert render_path not in seen_devices, (
                f"gpu_id={gpu_id} was scoped to {render_path}, already used by another gpu_id"
            )
            seen_devices.add(render_path)


# ---------------------------------------------------------------------------
# /dev -- individual per-file bind mounts (not a whole-directory bind, which
# silently fails identically to /proc -- see module docstring), verified
# against a real jail rather than just checking the setup script's text.
# ---------------------------------------------------------------------------


@chroot
class TestChrootDevFiles:
    def test_dev_null_is_writable(self):
        sb = _make_backend()
        try:
            out, rc = sb.exec("echo hi > /dev/null && echo OK")
            assert rc == 0
            assert "OK" in out
        finally:
            sb.destroy()

    def test_dev_urandom_is_readable(self):
        sb = _make_backend()
        try:
            out, rc = sb.exec("head -c 8 /dev/urandom | wc -c")
            assert rc == 0
            assert out.strip() == "8"
        finally:
            sb.destroy()

    def test_backgrounding_survives_without_dev_null(self):
        """A shell backgrounding a job commonly redirects its stdin to
        /dev/null -- if /dev/null didn't exist (the state before this fix),
        some shells silently fail to start the background job at all."""
        sb = _make_backend()
        try:
            sb.exec("sleep 5 &")
            assert len(sb.get_live_pids()) > 0
        finally:
            sb.destroy()


# ---------------------------------------------------------------------------
# _ChrootBackend -- functional (requires unprivileged userns + chroot)
# ---------------------------------------------------------------------------


@chroot
class TestChrootBackendExec:
    def test_exec_runs_and_returns_output(self):
        sb = _make_backend()
        try:
            out, rc = sb.exec("echo hello-chroot")
            assert rc == 0
            assert "hello-chroot" in out
        finally:
            sb.destroy()

    def test_exec_default_workdir_is_workspace(self):
        sb = _make_backend()
        try:
            out, rc = sb.exec("pwd")
            assert rc == 0
            assert out.strip() == "/workspace"
        finally:
            sb.destroy()

    def test_cannot_see_host_home_directory(self):
        """The jail must not expose the host's real filesystem outside the
        bind-mounted base dirs and workspace."""
        sb = _make_backend()
        try:
            out, rc = sb.exec("ls /home 2>&1; echo RC=$?")
            assert "No such file or directory" in out or "RC=2" in out
        finally:
            sb.destroy()

    def test_readonly_base_dir_rejects_writes(self):
        sb = _make_backend()
        try:
            out, rc = sb.exec("touch /bin/should-not-exist 2>&1; echo RC=$?")
            assert "RC=0" not in out or rc != 0
        finally:
            sb.destroy()

    def test_python_is_usable_inside_jail(self):
        sb = _make_backend()
        try:
            out, rc = sb.exec("python3 -c 'print(1+1)' 2>&1 || echo NO_PYTHON")
            assert "NO_PYTHON" not in out
        finally:
            sb.destroy()


@chroot
class TestChrootBackendFileIO:
    def test_write_then_read_file_round_trips(self):
        sb = _make_backend()
        try:
            sb.write_file("/workspace/hello.txt", "hello world\n")
            assert sb.read_file("/workspace/hello.txt") == "hello world\n"
        finally:
            sb.destroy()

    def test_write_file_bytes_round_trips(self):
        sb = _make_backend()
        try:
            data = bytes([0x00, 0x01, 0xFF, 0x10])
            sb.write_file_bytes("/workspace/bin.dat", data)
            assert sb.read_file_bytes("/workspace/bin.dat") == data
        finally:
            sb.destroy()

    def test_read_missing_file_raises_file_not_found(self):
        sb = _make_backend()
        try:
            with pytest.raises(FileNotFoundError):
                sb.read_file("/workspace/does-not-exist.txt")
        finally:
            sb.destroy()


@chroot
class TestChrootBackendLifecycle:
    def test_ensure_started_does_not_wipe_workspace_across_worker_processes(self):
        """Regression test: tool calls with run_in_subprocess=True (the
        default) each get a fresh cloudpickled copy of the backend, so a
        per-process flag would be False in every worker's own copy
        regardless of what an earlier worker already did. _ensure_started()
        must use the workspace directory's existence on disk as ground
        truth -- otherwise every worker's first touch re-runs
        _materialize_workspace() and wipes out whatever a *different*
        worker already wrote (this exact bug shipped and was caught against
        a real chroot-backed run: a file written by one tool dispatch was
        gone by the time the very next dispatch tried to read it back)."""
        import pickle

        orig = _make_backend()
        try:
            worker1 = pickle.loads(pickle.dumps(orig))
            worker1.write_file("/workspace/inputs/full_text_123.txt", "important content\n")

            worker2 = pickle.loads(pickle.dumps(orig))
            content = worker2.read_file("/workspace/inputs/full_text_123.txt")
            assert content == "important content\n"
        finally:
            orig.destroy()

    def test_commit_commits_even_when_this_process_never_started(self):
        """Same root cause as above, for commit(): the orchestrating process
        calls commit() on a sandbox whose actual workspace content was
        written entirely by worker-process tool calls, which this process's
        own copy never directly observed. commit() must still find and
        checkpoint that real work rather than silently skip committing."""
        import pickle

        orig = _make_backend()
        restored = None
        try:
            worker = pickle.loads(pickle.dumps(orig))
            worker.write_file("/workspace/data.txt", "from-worker\n")

            assert orig.commit() is True, "commit() must have committed"
            assert orig._checkpoint_image is not None

            restored = _make_backend(checkpoint_image=orig._checkpoint_image)
            assert restored.read_file("/workspace/data.txt") == "from-worker\n"
        finally:
            if restored is not None:
                restored.destroy()
            if orig._checkpoint_image:
                _ChrootBackend.delete_image(orig._checkpoint_image, force=True)
            orig.destroy()

    def test_commit_then_new_backend_restores_content(self):
        """sb2 must materialize (and finish reading) its own copy of the
        snapshot BEFORE sb1.destroy() runs: commit() now leaves the live
        workspace alone but still sets self._checkpoint_image = tag (see
        commit()'s docstring), and destroy() unconditionally deletes
        whatever snapshot self._checkpoint_image points at -- exactly the
        tag this test just committed. Reading sb2 first (which copies the
        snapshot into sb2's own, independent workspace directory) makes
        sb1's later cleanup of that same tag irrelevant to sb2's already-
        materialized copy."""
        tag = f"agency/lifecycle-test-{uuid.uuid4().hex[:8]}"
        sb1 = _make_backend()
        sb2 = None
        try:
            sb1.write_file("/workspace/marker.txt", "checkpoint\n")
            assert sb1.commit(tag) is True

            sb2 = _make_backend(checkpoint_image=tag)
            assert sb2.read_file("/workspace/marker.txt") == "checkpoint\n"
        finally:
            sb1.destroy()
            if sb2 is not None:
                sb2.destroy()
            _ChrootBackend.delete_image(tag, force=True)

    def test_commit_returns_false_when_never_started(self):
        sb = _make_backend()
        assert sb.commit("agency/never-started") is False
        sb.destroy()

    def test_rm_container_discards_dirty_state(self):
        """commit() then rm_container() reproduces the old
        stop(commit=True)-then-stop(commit=False) revert sequence: commit()
        checkpoints "good.txt" without touching the live workspace,
        rm_container() discards the workspace outright (including the later
        "dirty.txt"), and the next _ensure_started() re-materializes from
        the last checkpoint -- which never saw "dirty.txt" in the first
        place."""
        sb = _make_backend()
        try:
            sb.write_file("/workspace/good.txt", "good\n")
            sb.commit()
            sb.rm_container()
            sb._ensure_started()
            sb.write_file("/workspace/dirty.txt", "dirty\n")
            sb.rm_container()
            sb._ensure_started()
            content = sb.read_file("/workspace/good.txt")
            assert content == "good\n"
            with pytest.raises(FileNotFoundError):
                sb.read_file("/workspace/dirty.txt")
        finally:
            sb.destroy()

    def test_restore_materializes_snapshot(self):
        tag = f"agency/lifecycle-test-{uuid.uuid4().hex[:8]}"
        sb = _make_backend()
        try:
            sb.write_file("/workspace/a.txt", "aaa\n")
            sb.commit(tag)
            sb.write_file("/workspace/b.txt", "bbb\n")
            sb.restore(tag)
            assert sb.read_file("/workspace/a.txt") == "aaa\n"
            with pytest.raises(FileNotFoundError):
                sb.read_file("/workspace/b.txt")
        finally:
            sb.destroy()
            _ChrootBackend.delete_image(tag, force=True)

    def test_destroy_removes_jail_directory(self):
        sb = _make_backend()
        sb.exec("true")
        root = sb._root
        assert root.exists()
        sb.destroy()
        assert not root.exists()

    def test_cross_agent_workspace_isolation(self):
        sb1 = _make_backend()
        sb2 = _make_backend()
        try:
            sb1.write_file("/workspace/only-in-1.txt", "secret\n")
            out, rc = sb2.exec("ls /workspace")
            assert "only-in-1.txt" not in out
        finally:
            sb1.destroy()
            sb2.destroy()

    def test_update_limits_is_a_no_op(self):
        sb = _make_backend()
        try:
            sb.update_limits(cpus=2.0, memory="4g")  # must not raise
        finally:
            sb.destroy()

    @staticmethod
    def _find_marked_pid(pids, marker):
        """Identify the specific pid matching *marker* among *pids* --
        blindly grabbing an arbitrary tracked pid (e.g. next(iter(...)))
        is unsafe: a single tracked process group can contain more than one
        pid (the wrapping shell alongside the actual backgrounded command),
        so the marker is what pins down the specific one being tested."""
        for pid in pids:
            try:
                cmdline = open(f"/proc/{pid}/cmdline", "rb").read()
            except OSError:
                continue
            if marker.encode() in cmdline:
                return pid
        return None

    def test_destroy_kills_background_process_before_removing_root(self):
        """Regression: destroy() used to just clear _watched_pids and
        rmtree the jail root, never actually sending a background process
        any signal -- it would keep running independently, chrooted into a
        directory tree that had just been deleted out from under it. Unlike
        the container backend, nothing here does this implicitly (there's
        no container-removal-equivalent hard kill for a chroot jail), so
        destroy() must explicitly terminate anything still running first."""
        sb = _make_backend()
        marker = f"agencytest{uuid.uuid4().hex[:8]}"
        try:
            sb.exec(f"exec -a {marker} sleep 30 &")
            tracked_pid = self._find_marked_pid(sb.get_live_pids(), marker)
            assert tracked_pid is not None, (
                f"expected the marked sleep to be tracked, got {sb.get_live_pids()}"
            )
        finally:
            sb.destroy()

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and os.path.exists(f"/proc/{tracked_pid}"):
            time.sleep(0.1)
        assert not os.path.exists(f"/proc/{tracked_pid}"), (
            "backgrounded process must not survive destroy()"
        )

    def test_stop_kills_background_process_but_preserves_workspace(self):
        """stop() hibernates: it must still kill any background process (the
        same kill step rm_container()/destroy() use) but must NOT touch the
        workspace -- unlike rm_container(), a hibernated sandbox's state,
        files included, has to survive stop()."""
        sb = _make_backend()
        marker = f"agencytest{uuid.uuid4().hex[:8]}"
        try:
            sb.write_file("/workspace/kept.txt", "kept\n")
            sb.exec(f"exec -a {marker} sleep 30 &")
            tracked_pid = self._find_marked_pid(sb.get_live_pids(), marker)
            assert tracked_pid is not None, (
                f"expected the marked sleep to be tracked, got {sb.get_live_pids()}"
            )
            sb.stop()

            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and os.path.exists(f"/proc/{tracked_pid}"):
                time.sleep(0.1)
            assert not os.path.exists(f"/proc/{tracked_pid}"), (
                "backgrounded process must not survive stop()"
            )
            sb._ensure_started()
            assert sb.read_file("/workspace/kept.txt") == "kept\n", (
                "stop() must not touch the workspace -- only rm_container() discards it"
            )
        finally:
            sb.destroy()

    def test_rm_container_kills_background_process_before_reverting_workspace(self):
        """Same regression as test_destroy_kills_background_process_before_removing_root,
        for rm_container(): a background job from the dirty state being
        discarded must not be left running independently once the
        workspace is reverted out from under it."""
        sb = _make_backend()
        marker = f"agencytest{uuid.uuid4().hex[:8]}"
        try:
            sb.exec(f"exec -a {marker} sleep 30 &")
            tracked_pid = self._find_marked_pid(sb.get_live_pids(), marker)
            assert tracked_pid is not None, (
                f"expected the marked sleep to be tracked, got {sb.get_live_pids()}"
            )
            sb.rm_container()

            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and os.path.exists(f"/proc/{tracked_pid}"):
                time.sleep(0.1)
            assert not os.path.exists(f"/proc/{tracked_pid}"), (
                "backgrounded process must not survive rm_container()"
            )
        finally:
            sb.destroy()


# ---------------------------------------------------------------------------
# GPU semaphore release in stop()/destroy() -- unlike the container backend,
# chroot has no daemon process to inspect for "confirmed torn down", so there
# is no such gate here: the kill attempt in _kill_all_sandbox_processes() IS
# the confirmation (see its docstring), and release always follows it
# unconditionally. No real chroot required --
# _kill_all_sandbox_processes()/_materialize_workspace() aren't invoked with
# a real jail here, only stop()/destroy()'s own release-ordering logic is
# exercised.
# ---------------------------------------------------------------------------


class TestChrootGpuReleaseGating:
    def _lease_gpu(self, sb, gpu_id=3):
        released = []
        sb._gpu_virtual = True
        sb._gpu_id = gpu_id
        sb._gpu_release_fn = lambda gid: released.append(gid)
        return released

    def test_stop_releases_gpu_after_kill_attempt(self, monkeypatch):
        sb = _make_backend()
        released = self._lease_gpu(sb)
        monkeypatch.setattr(sb, "_kill_all_sandbox_processes", lambda: None)

        sb.stop()

        assert released == [3]
        assert sb._gpu_id is None

    def test_kill_runs_before_release(self, monkeypatch):
        order = []
        sb = _make_backend()
        sb._gpu_virtual = True
        sb._gpu_id = 3
        sb._gpu_release_fn = lambda gid: order.append(("release", gid))
        monkeypatch.setattr(sb, "_kill_all_sandbox_processes", lambda: order.append(("kill",)))

        sb.stop()

        assert order == [("kill",), ("release", 3)]

    def test_gpu_released_exactly_once_across_stop_then_destroy(self, monkeypatch):
        sb = _make_backend()
        released = self._lease_gpu(sb)
        monkeypatch.setattr(sb, "_kill_all_sandbox_processes", lambda: None)

        sb.stop()
        sb.destroy()

        assert released == [3], "GPU must be released exactly once, not once per call"

    def test_destroy_releases_gpu_exactly_once(self, monkeypatch):
        """destroy() calls rm_container() internally -- confirm that single
        call chain still only releases once, not via some hidden double
        invocation."""
        sb = _make_backend()
        released = self._lease_gpu(sb)
        monkeypatch.setattr(sb, "_kill_all_sandbox_processes", lambda: None)

        sb.destroy()

        assert released == [3]


@chroot
class TestChrootImageHelpers:
    def test_tag_image_copies_snapshot(self):
        src_tag = f"agency/src-{uuid.uuid4().hex[:8]}"
        dest_tag = f"agency/dest-{uuid.uuid4().hex[:8]}"
        sb = _make_backend()
        try:
            sb.write_file("/workspace/x.txt", "x\n")
            sb.commit(src_tag)
            _ChrootBackend.tag_image(src_tag, dest_tag)
            sb2 = _make_backend(checkpoint_image=dest_tag)
            try:
                assert sb2.read_file("/workspace/x.txt") == "x\n"
            finally:
                sb2.destroy()
        finally:
            sb.destroy()
            _ChrootBackend.delete_image(src_tag, force=True)
            _ChrootBackend.delete_image(dest_tag, force=True)

    def test_export_then_import_round_trips(self):
        tag = f"agency/export-{uuid.uuid4().hex[:8]}"
        sb = _make_backend()
        try:
            sb.write_file("/workspace/e.txt", "exported\n")
            sb.commit(tag)
            blob = _ChrootBackend.export_image(tag, 30)
            assert isinstance(blob, bytes) and len(blob) > 0
            _ChrootBackend.delete_image(tag, force=True)
            _ChrootBackend.import_image(blob, 30)
            sb2 = _make_backend(checkpoint_image=tag)
            try:
                assert sb2.read_file("/workspace/e.txt") == "exported\n"
            finally:
                sb2.destroy()
        finally:
            sb.destroy()
            _ChrootBackend.delete_image(tag, force=True)

    def test_export_missing_tag_raises(self):
        with pytest.raises(FileNotFoundError):
            _ChrootBackend.export_image(f"agency/no-such-{uuid.uuid4().hex[:8]}", 30)

    def test_relabel_owner_pid_is_a_noop(self):
        """A chroot snapshot has no label/metadata concept at all (see
        _ContainerBackendBase's real version, which this mirrors for API
        parity so agent.py's save()/load() can call it generically
        regardless of backend) -- must not raise even for a tag that
        doesn't exist."""
        _ChrootBackend.relabel_owner_pid(f"agency/no-such-{uuid.uuid4().hex[:8]}", 12345, 30)
        _ChrootBackend.relabel_owner_pid(f"agency/no-such-{uuid.uuid4().hex[:8]}", None, 30)


# ---------------------------------------------------------------------------
# Facade integration -- agSandbox(backend="chroot")
# ---------------------------------------------------------------------------


@chroot
class TestFacadeWithChrootBackend:
    def _make_sandbox(self, **kwargs):
        from agency.agconfig import agConfig
        from agency.agsandbox import agSandbox
        from agency.agsandbox_backends import agSandboxBackendConfig

        cfg = agConfig(agSandboxBackendConfig(backend="chroot"))
        uid = str(uuid.uuid4())
        return agSandbox(uid, agconfig=cfg, **kwargs)

    def test_facade_selects_chroot_backend(self):
        sb = self._make_sandbox()
        try:
            assert isinstance(sb._backend, _ChrootBackend)
        finally:
            sb.destroy()

    def test_facade_exec_and_file_io(self):
        sb = self._make_sandbox()
        try:
            out, rc = sb.exec("echo via-facade")
            assert rc == 0 and "via-facade" in out
            sb.write_file("/workspace/f.txt", "f-content\n")
            assert sb.read_file("/workspace/f.txt") == "f-content\n"
        finally:
            sb.destroy()

    def test_facade_commit_sets_checkpoint_image(self):
        sb = self._make_sandbox()
        try:
            sb.exec("true")
            sb.commit()
            assert sb._checkpoint_image is not None
        finally:
            sb.destroy()

    def test_facade_fork_preserves_checkpoint_content(self):
        sb = self._make_sandbox()
        fork_sb = None
        try:
            sb.write_file("/workspace/parent.txt", "parent-data\n")
            sb.commit()
            fork_sb = sb.fork(str(uuid.uuid4()))
            assert isinstance(fork_sb._backend, _ChrootBackend)
            assert fork_sb.read_file("/workspace/parent.txt") == "parent-data\n"
        finally:
            sb.destroy()
            if fork_sb is not None:
                fork_sb.destroy()


# ---------------------------------------------------------------------------
# agent.py save()/load() wiring -- a checkpoint must round-trip through the
# same backend kind that produced it, not be silently assumed to be a
# docker/podman image tag.
# ---------------------------------------------------------------------------


@chroot
class TestAgentSaveLoadWithChrootBackend:
    def _make_agconfig(self):
        from agency.agconfig import agConfig
        from agency.agsandbox_backends import agSandboxBackendConfig

        return agConfig(
            agSandboxBackendConfig(backend="chroot"),
            {"agllm_backend": {"api_key": "k", "model": "m"}},
        )

    def test_save_records_chroot_image_kind(self, tmp_path):
        import json
        import tarfile
        from agency.agsandbox import agSandbox
        from agency.agent import agent
        from agency.agname import agname

        cfg = self._make_agconfig()
        ag = agent(agconfig=cfg)
        try:
            ag.sandbox = agSandbox(ag.agname, agconfig=cfg)
            ag.sandbox.write_file("/workspace/marker.txt", "data\n")
            ag.sandbox.commit()

            ckpt = tmp_path / "agent.ckpt"
            ag.save(ckpt)

            with tarfile.open(ckpt, "r:gz") as tar:
                state = json.loads(tar.extractfile("state.json").read())
            assert state["sandbox_image_kind"] == "chroot"
        finally:
            if ag.sandbox is not None:
                ag.sandbox.destroy()
            agname._allocated.discard(str(ag.agname))

    def test_save_then_load_restores_chroot_workspace(self, tmp_path):
        from agency.agsandbox import agSandbox
        from agency.agent import agent
        from agency.agname import agname

        cfg = self._make_agconfig()
        ag = agent(agconfig=cfg)
        ag2 = None
        try:
            ag.sandbox = agSandbox(ag.agname, agconfig=cfg)
            ag.sandbox.write_file("/workspace/marker.txt", "checkpointed-via-agent\n")
            ag.sandbox.commit()

            ckpt = tmp_path / "agent.ckpt"
            ag.save(ckpt)
            saved_agname = str(ag.agname)
            ag.sandbox.destroy()
            agname._allocated.discard(saved_agname)

            ag2 = agent.load(ckpt, agconfig=cfg)
            assert isinstance(ag2.sandbox._backend, _ChrootBackend)
            content = ag2.sandbox.read_file("/workspace/marker.txt")
            assert content == "checkpointed-via-agent\n"
        finally:
            if ag2 is not None and ag2.sandbox is not None:
                ag2.sandbox.destroy()
                agname._allocated.discard(str(ag2.agname))


# ---------------------------------------------------------------------------
# Real ProcessPoolExecutor dispatch -- the actual code path a live agent run
# uses (run_in_subprocess=True, the default), as opposed to calling a
# backend's methods directly in-process. This is what surfaced the
# _ensure_started()/stop() worker-vs-main-process bugs fixed above: calling
# a tool's .fn() directly, or driving the backend object in one process,
# never exercises cloudpickle sending a fresh copy of the sandbox to a
# ProcessPoolExecutor worker for every single call.
# ---------------------------------------------------------------------------


@chroot
class TestChrootSandboxedToolsDispatch:
    def _make_sandbox(self):
        from agency.agconfig import agConfig
        from agency.agsandbox import agSandbox
        from agency.agsandbox_backends import agSandboxBackendConfig

        cfg = agConfig(agSandboxBackendConfig(backend="chroot"))
        return agSandbox(str(uuid.uuid4()), agconfig=cfg)

    def test_files_persist_across_process_pool_tool_calls(self):
        """Files written by the write tool in one worker process must be
        readable by the read tool in a subsequent, separately-dispatched
        worker process call -- the chroot-backend analogue of
        test_docker.py's identically-named container-backend test."""
        from agency.agdata import agdata, agerror
        from agency.tools import make_sandboxed_tools

        sb = self._make_sandbox()
        tools = {t.name: t for t in make_sandboxed_tools(sb)}
        try:
            w = tools["write"](agdata(file_path="/workspace/cross.txt", content="cross-worker\n"))
            assert not isinstance(w, agerror), f"write failed: {w}"
            r = tools["read"](agdata(file_path="/workspace/cross.txt"))
            assert not isinstance(r, agerror), f"read failed after cross-worker write: {r}"
            assert "cross-worker" in r.content
        finally:
            sb.destroy()

    def test_bash_then_read_across_process_pool_tool_calls(self):
        """A file created by the bash tool in one worker process must be
        readable by the read tool in the next, separately-dispatched call --
        matches the exact real-world shape of the bug (agfile.prepare()'s
        sandbox.write_file() in one dispatch, the read tool in the next)."""
        from agency.agdata import agdata, agerror
        from agency.tools import make_sandboxed_tools

        sb = self._make_sandbox()
        tools = {t.name: t for t in make_sandboxed_tools(sb)}
        try:
            b = tools["bash"](
                agdata(
                    command="mkdir -p /workspace/inputs && echo hi > /workspace/inputs/full_text_1.txt"
                )
            )
            assert not isinstance(b, agerror), f"bash failed: {b}"
            r = tools["read"](agdata(file_path="/workspace/inputs/full_text_1.txt"))
            assert not isinstance(r, agerror), f"read failed after cross-worker bash write: {r}"
            assert "hi" in r.content
        finally:
            sb.destroy()

    def test_agent_run_offloads_and_reads_back_large_input(self):
        """End-to-end: a real agskill run whose input schema triggers
        agschema's size-based offload (sandbox.write_file in the prepare
        step, executed in the calling thread) followed by the LLM calling
        the read tool (a separate ProcessPoolExecutor dispatch) to read it
        back -- the exact real-world flow that surfaced this bug."""
        from agency.agconfig import agConfig
        from agency.agdata import agdata
        from agency.agskill import agskill
        from agency.agschema import agSchemaConfig
        from agency.agsandbox_backends import agSandboxBackendConfig
        from agency.agent import agent

        cfg = agConfig(
            agSandboxBackendConfig(backend="chroot"),
            agSchemaConfig(input_offload_chars=10),  # force offload for a short string
            {"agllm_backend": {"api_key": "k", "model": "m"}},
        )

        skill = agskill(name="repro", system_prompt="", input_schema=agdata(text=str))

        def fake_execute_react(ag, prev_ctx, skill_input, max_steps=None, **_):
            # Replicate execute_react()'s real step 2 (input prep) explicitly,
            # since replacing execute_react wholesale also removes that step
            # -- it isn't called automatically just because ag.sandbox exists.
            skill.input_schema.prepare_inputs_in_sandbox(
                skill_input,
                ag.sandbox,
                skill.name,
                context_limit=ag.llm.context_limit,
                agconfig=ag.agconfig,
            )
            # skill_input.text has now been offloaded to a path reference --
            # read it back via the same tool-dispatch path a real ReAct loop
            # (running against an LLM) would use.
            from agency.tools import make_sandboxed_tools

            tools = {t.name: t for t in make_sandboxed_tools(ag.sandbox)}
            path = skill_input.text.split("saved to ")[1].split(" —")[0]
            r = tools["read"](agdata(file_path=path))
            return agdata(answer=r.content), prev_ctx, []

        skill.execute_react = fake_execute_react

        ag = agent(agconfig=cfg)
        try:
            long_text = "x" * 100
            result = ag.run(skill, agdata(text=long_text)).answer
            assert long_text in result
        finally:
            if ag.sandbox is not None:
                ag.sandbox.destroy()
