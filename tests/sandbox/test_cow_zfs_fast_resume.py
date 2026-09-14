import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agency.configs.agconfig import agconfig
from agency.sandbox import checkpoint as cp
from agency.sandbox.base import agsandbox_backend
from agency.sandbox.container import _ContainerBackendBase


def _backend(runtime="podman"):
    cfg = agconfig()
    cfg.sandbox.backend = runtime
    cfg.sandbox.checkpoint_backend = "cow_zfs"
    cfg.sandbox.checkpoint_zfs_parent = "tank/agency"
    cfg.sandbox.checkpoint_fast_resume = True
    state = {"running": True}
    backend = _ContainerBackendBase.__new__(_ContainerBackendBase)
    backend._agconfig = cfg
    backend._runtime = runtime
    backend._name = "sandbox"
    backend._gpu_count_requested = 0
    backend._checkpointer = cp.CowZfsCheckpoint()
    backend._normalized_checkpoint_manifest = Mock(return_value={"schema_version": 1})
    backend._container_running = lambda: state["running"]
    backend._stop_image = Mock(side_effect=lambda: state.update(running=False))
    backend._ensure_started_profiled = Mock(side_effect=lambda: state.update(running=True))
    backend._register_prof_container = Mock()
    backend._checkpoint_storage = SimpleNamespace(
        prepared=True,
        dataset="tank/agency/private",
        prepare=Mock(),
        capture_dataset=Mock(return_value="tank/agency/private"),
        snapshot=Mock(),
        snapshot_stats=Mock(return_value={"zfs_snapshot_used_bytes": 0}),
        rollback=Mock(),
        delete=Mock(),
        criu_ready=True,
        criu_setup_error=None,
        fast_root=Path("/zfs/agency-fast-resume"),
        prepare_restore_mounts=Mock(),
        runtime_files_restored=Mock(),
    )
    return backend, state


def test_fast_resume_is_optional_and_requires_cow_zfs():
    cfg = agconfig()
    cfg.sandbox.checkpoint_fast_resume = True
    with pytest.raises(ValueError, match="requires checkpoint_backend='cow_zfs'"):
        agsandbox_backend()._validate_config(cfg)


def test_dump_and_restore_criu_stats_use_distinct_keys():
    raw = '{"podman_checkpoint_duration": 123}'

    assert cp._criu_stats(raw) == {"podman_criu_stats": {"podman_checkpoint_duration": 123}}
    assert cp._criu_stats(raw, "podman_criu_restore_stats") == {
        "podman_criu_restore_stats": {"podman_checkpoint_duration": 123}
    }


def test_successful_process_checkpoint_remains_an_acceleration_layer(monkeypatch):
    backend, state = _backend()
    artifact = {"name": "agency-test", "daemon_count": 1, "live_pty": True}

    def dump(_sandbox):
        state["running"] = False
        return artifact

    restore = Mock(side_effect=lambda _sandbox, _artifact: state.update(running=True))
    monkeypatch.setattr(backend._checkpointer, "_runtime_checkpoint", dump)
    monkeypatch.setattr(backend._checkpointer, "_runtime_restore", restore)

    handle = backend.checkpoint()
    assert handle.stats["fast_resume_available"] is True
    assert handle.stats["fast_resume_live_pty"] is True
    assert backend._checkpoint_storage.snapshot.call_count == 1

    backend._checkpointer.restore(backend, handle)
    restore.assert_called_once_with(backend, artifact)
    assert handle.stats["fast_resume_used"] is True
    assert state["running"] is True


def test_docker_restore_retries_containerd_checkpoint_publish_race(monkeypatch):
    backend, state = _backend("docker")
    backend._agency_harness_daemon_handles = {}
    backend._acquire_runtime_slot = Mock()
    backend._release_runtime_slot = Mock()
    backend._infrastructure_pids = {}
    starts = 0

    def run(args, **_kwargs):
        nonlocal starts
        starts += 1
        if starts < 3:
            raise RuntimeError(
                "failed to upload checkpoint to containerd: commit failed: "
                "content sha256:abc: already exists"
            )
        state["running"] = True
        return subprocess.CompletedProcess(args, 0, b"sandbox\n", b"")

    backend._run = Mock(side_effect=run)
    monkeypatch.setattr(cp.time, "sleep", Mock())
    monkeypatch.setattr(cp.CowZfsCheckpoint, "_resume_daemons", Mock())
    from agency.sandbox import _criu_restore_mounts

    monkeypatch.setattr(_criu_restore_mounts, "thaw_retained_tasks", Mock())
    artifact = {
        "name": "agency-checkpoint",
        "runtime": "docker",
        "daemon_count": 0,
        "pid_tracking": {},
    }

    backend._checkpointer._runtime_restore(backend, artifact)

    assert starts == 3
    assert state["running"] is True


def test_process_dump_failure_still_publishes_filesystem_checkpoint(monkeypatch):
    backend, state = _backend()
    monkeypatch.setattr(
        backend._checkpointer,
        "_runtime_checkpoint",
        Mock(side_effect=cp.CheckpointCapabilityError("CRIU unavailable")),
    )

    handle = backend.checkpoint()

    assert handle.stats["fast_resume_available"] is False
    assert handle.stats["fast_resume_checkpoint_error"] == "CheckpointCapabilityError"
    backend._stop_image.assert_called_once_with()
    assert state["running"] is False


def test_process_restore_failure_rolls_back_again_for_fresh_start(monkeypatch):
    backend, state = _backend()
    artifact = {"name": "agency-test", "daemon_count": 1, "live_pty": True}

    def dump(_sandbox):
        state["running"] = False
        return artifact

    monkeypatch.setattr(backend._checkpointer, "_runtime_checkpoint", dump)
    handle = backend.checkpoint()
    monkeypatch.setattr(
        backend._checkpointer,
        "_runtime_restore",
        Mock(side_effect=RuntimeError("restore rejected")),
    )

    backend._checkpointer.restore(backend, handle)

    assert backend._checkpoint_storage.rollback.call_count == 2
    assert handle.stats["fast_resume_used"] is False
    assert handle.stats["fast_resume_restore_error"] == "RuntimeError"
    assert backend._checkpointer.fast_artifact is None
    assert state["running"] is False


def test_docker_checkpoint_and_restore_use_private_checkpoint_directory(monkeypatch):
    backend, state = _backend("docker")
    backend._agency_harness_daemon_handles = {}
    backend._mark_runtime_checkpoint_stopped = Mock(side_effect=lambda: state.update(running=False))
    backend._acquire_runtime_slot = Mock()
    backend._release_runtime_slot = Mock()
    backend._infrastructure_pids = {}
    commands = []

    def run(args, **_kwargs):
        commands.append(args)
        if args[1] == "inspect":
            payload = [
                {
                    "Id": "a" * 64,
                    "Config": {"Tty": False},
                    "HostConfig": {"NetworkMode": "host"},
                    "ExecIDs": None,
                }
            ]
            return subprocess.CompletedProcess(args, 0, json.dumps(payload).encode(), b"")
        if args[1] == "checkpoint":
            return subprocess.CompletedProcess(args, 0, b"", b"")
        if args[1] == "start":
            state["running"] = True
            return subprocess.CompletedProcess(args, 0, b"sandbox\n", b"")
        raise AssertionError(args)

    backend._run = Mock(side_effect=run)
    monkeypatch.setattr(cp, "check_process_resources", lambda _sandbox: 123)
    monkeypatch.setattr(cp.time, "monotonic", Mock(side_effect=range(10)))
    monkeypatch.setattr(cp.CowZfsCheckpoint, "_resume_daemons", Mock())
    from agency.sandbox import _criu_restore_mounts

    monkeypatch.setattr(_criu_restore_mounts, "thaw_retained_tasks", Mock())

    artifact = backend._checkpointer._runtime_checkpoint(backend)
    assert artifact["runtime"] == "docker"
    assert commands[-1][:3] == [
        "docker",
        "checkpoint",
        "create",
    ]
    assert commands[-1][-2:] == ["sandbox", artifact["name"]]
    backend._checkpoint_storage.prepare_restore_mounts.assert_called_once()
    assert state["running"] is False

    backend._checkpointer._runtime_restore(backend, artifact)
    assert commands[-1] == [
        "docker",
        "start",
        "--checkpoint",
        artifact["name"],
        "sandbox",
    ]
    assert artifact["restore_stats"] == {"docker_criu_restore_seconds": 1}
    _criu_restore_mounts.thaw_retained_tasks.assert_called_once_with(
        Path("/zfs/agency-fast-resume")
    )
    assert state["running"] is True


def test_docker_fast_checkpoint_rejects_managed_network_namespace(monkeypatch):
    backend, _state = _backend("docker")
    backend._agency_harness_daemon_handles = {}
    inspection = [
        {
            "Id": "a" * 64,
            "Config": {"Tty": False},
            "HostConfig": {"NetworkMode": "bridge"},
            "ExecIDs": None,
        }
    ]
    backend._run = Mock(
        return_value=subprocess.CompletedProcess(
            ["docker", "inspect", "sandbox"], 0, json.dumps(inspection).encode(), b""
        )
    )
    monkeypatch.setattr(cp, "check_process_resources", lambda _sandbox: 123)

    with pytest.raises(cp.CheckpointCapabilityError, match="--network=host"):
        backend._checkpointer._runtime_checkpoint(backend)


def test_docker_fast_artifact_cleanup_uses_private_directory():
    backend, _state = _backend("docker")
    backend._run = Mock()
    artifact = {
        "name": "agency-checkpoint",
    }

    backend._checkpointer._delete_fast_artifact(backend, artifact)

    backend._run.assert_called_once_with(
        [
            "docker",
            "checkpoint",
            "rm",
            "sandbox",
            "agency-checkpoint",
        ],
        check=False,
        timeout=30,
    )


def test_docker_runtime_bind_files_are_repaired_from_private_backup(tmp_path):
    from agency.sandbox._criu_restore_mounts import restore_runtime_files

    runtime_root = tmp_path / "docker"
    fast_root = runtime_root / "agency-fast-resume" / "sandbox"
    container_root = runtime_root / "containers" / ("a" * 64)
    container_root.mkdir(parents=True)
    fast_root.mkdir(parents=True)
    paths = {}
    for name in ("hosts", "hostname", "resolv.conf"):
        path = container_root / name
        path.write_text("checkpoint-" + name)
        paths[name] = path
    storage = cp.ZfsRuntimeStorage.__new__(cp.ZfsRuntimeStorage)
    storage.runtime = "docker"
    storage.path = runtime_root
    storage.fast_root = fast_root
    storage.prepare_restore_mounts(
        {
            "Id": "a" * 64,
            "HostsPath": str(paths["hosts"]),
            "HostnamePath": str(paths["hostname"]),
            "ResolvConfPath": str(paths["resolv.conf"]),
        },
        "tank/docker/container@agency-checkpoint",
        [],
    )
    for path in paths.values():
        path.write_text("regenerated")

    restore_runtime_files(fast_root)

    for name, path in paths.items():
        assert path.read_text() == "checkpoint-" + name
