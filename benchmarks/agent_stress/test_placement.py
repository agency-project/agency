"""Preparation-only tests: no engine invocation, child workload, or container launch."""

from unittest.mock import Mock

import pytest

from benchmarks.agent_stress.placement import make_layout, parse_topology, validate_layout


def topology():
    return [{"cpu": i, "core": i % 16, "socket": 0, "node": 0} for i in range(32)]


def test_layout_reserves_complete_smt_groups_and_keeps_agents_on_separate_cores():
    rows = topology()
    layout = make_layout(rows, [1, 2, 4, 8, 12])
    assert layout["os_runtime_headroom_cpus"] == [0, 1, 16, 17]
    assert layout["driver_engine_monitor_cpus"] == [2, 3, 18, 19]
    assert [s["cpus"] for s in layout["levels"]["8"]] == [[i] for i in range(4, 12)]
    assert layout["levels"]["2"] == layout["levels"]["8"][:2]
    validate_layout(layout, rows, [1, 2, 8])


def test_layout_rejects_smt_overcommit_and_changed_topology():
    with pytest.raises(ValueError, match="exceeds 12"):
        make_layout(topology(), [16])
    layout = make_layout(topology(), [1, 2])
    with pytest.raises(ValueError, match="differs"):
        validate_layout(layout, topology()[:-1], [1])


def test_parser_ignores_offline_cpus_and_comments():
    assert parse_topology("# header\n2,1,0,0,Y\n1,0,0,0,N\n") == [
        {"cpu": 2, "core": 1, "socket": 0, "node": 0}
    ]


@pytest.mark.parametrize("runtime", ["docker", "podman"])
@pytest.mark.parametrize("checkpoint", [None, "committed-image"])
def test_container_creation_preserves_cpuset_and_resource_limits_after_rollback(
    monkeypatch, runtime, checkpoint
):
    from agency.configs.agconfig import agconfig, resourcesconfig, sandboxconfig
    from agency.sandbox import container
    from agency.sandbox.docker import _DockerBackend
    from agency.sandbox.podman import _PodmanBackend

    backend_cls = _DockerBackend if runtime == "docker" else _PodmanBackend
    cfg = agconfig(
        sandboxconfig(cpuset_cpus="4", cpuset_mems="0"),
        resourcesconfig(idle_cpus=1, idle_memory="512m"),
    )
    sb = backend_cls(
        "agent",
        name="placement-test",
        checkpoint_image=checkpoint,
        base_image="base",
        mounts={},
        agconfig=cfg,
    )
    monkeypatch.setattr(container, "_gpu_flags", lambda _: [])
    monkeypatch.setattr(container.agprof, "container_cgroup_parent", lambda: None)
    monkeypatch.setattr(sb, "_inspect_container_state", lambda: (False, ""))
    monkeypatch.setattr(sb, "_acquire_runtime_slot", lambda: None)
    monkeypatch.setattr(sb, "_resolve_image", lambda _: "base")
    monkeypatch.setattr(sb, "_cfs_supported", lambda: True)
    monkeypatch.setattr(sb, "_snapshot_pids_started", lambda: set())
    launch = Mock()
    monkeypatch.setattr(sb, "_run_with_conflict_retry", launch)
    monkeypatch.setattr(sb, "_run", Mock())
    sb._ensure_started_profiled()
    argv = launch.call_args.args[0]
    assert "--cpuset-cpus=4" in argv
    assert "--cpuset-mems=0" in argv
    assert "--memory=512m" in argv
    assert "--cpus=1" in argv
    assert (checkpoint or "base") in argv
