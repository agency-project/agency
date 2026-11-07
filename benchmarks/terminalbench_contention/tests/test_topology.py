"""CPU pool selection from sysfs topology, without assuming CPU numbering."""

import pytest

from benchmarks.terminalbench_contention import topology
from benchmarks.terminalbench_contention.topology import (
    InsufficientCpuError,
    build_layout,
    format_cpu_list,
    parse_cpu_list,
    read_topology,
)


def make_sysfs(root, cores, threads=2, scheme="split", offline=()):
    """Write a fake /sys/devices/system/cpu tree.

    scheme="split":    siblings are cpu N and cpu N+cores (0,8 / 1,9 ...)
    scheme="adjacent": siblings are cpu 2N and 2N+1       (0,1 / 2,3 ...)
    """
    mapping = {}
    for core in range(cores):
        cpus = (
            [core + t * cores for t in range(threads)]
            if scheme == "split"
            else [core * threads + t for t in range(threads)]
        )
        for cpu in cpus:
            mapping[cpu] = (core, cpus)
    root.mkdir(parents=True, exist_ok=True)
    online = sorted(c for c in mapping if c not in offline)
    (root / "online").write_text(format_cpu_list(online) + "\n")
    for cpu, (core, siblings) in mapping.items():
        directory = root / f"cpu{cpu}" / "topology"
        directory.mkdir(parents=True)
        (directory / "core_id").write_text(f"{core}\n")
        (directory / "physical_package_id").write_text("0\n")
        (directory / "thread_siblings_list").write_text(format_cpu_list(siblings) + "\n")
        (root / f"cpu{cpu}" / "node0").mkdir()
    return root


def test_cpu_list_round_trip():
    assert parse_cpu_list("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]
    assert format_cpu_list([11, 10, 8, 3, 2, 1, 0]) == "0-3,8,10-11"
    assert format_cpu_list([5]) == "5" and format_cpu_list([]) == ""
    assert parse_cpu_list("") == []


@pytest.mark.parametrize(
    "scheme,pool,reserved",
    [
        ("split", [2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15], [0, 1, 8, 9]),
        ("adjacent", list(range(4, 16)), [0, 1, 2, 3]),
    ],
)
def test_pool_follows_physical_cores_under_either_numbering_scheme(
    tmp_path, scheme, pool, reserved
):
    layout = build_layout(read_topology(make_sysfs(tmp_path, 8, scheme=scheme)))
    assert layout.pool_cpus == pool and layout.reserved_cpus == reserved
    assert len(layout.experimental_cores) == 6 and len(layout.reserved_cores) == 2
    assert all(len(core) == 2 for core in layout.experimental_cores)
    assert layout.smt_active and layout.physical_cores == 8 and layout.logical_cpus == 16
    assert layout.unused_cpus == [] and layout.smt_policy == "siblings"
    # No reserved thread shares a physical core with an experimental thread.
    assert not set(layout.pool_cpus) & set(layout.reserved_cpus)
    assert layout.to_dict()["pool_arg"] == format_cpu_list(pool)


def test_primary_policy_uses_one_thread_per_core_and_leaves_siblings_idle(tmp_path):
    layout = build_layout(read_topology(make_sysfs(tmp_path, 8)), smt_policy="primary")
    assert layout.pool_cpus == [2, 3, 4, 5, 6, 7]
    assert layout.unused_cpus == [10, 11, 12, 13, 14, 15]
    assert layout.reserved_cpus == [0, 1, 8, 9]


def test_smt_disabled_host(tmp_path):
    layout = build_layout(read_topology(make_sysfs(tmp_path, 8, threads=1)))
    assert not layout.smt_active and layout.pool_cpus == [2, 3, 4, 5, 6, 7]
    assert layout.reserved_cpus == [0, 1]


def test_extra_cores_join_the_reserved_side(tmp_path):
    layout = build_layout(read_topology(make_sysfs(tmp_path, 12, threads=1)))
    assert layout.pool_cpus == [2, 3, 4, 5, 6, 7]
    assert layout.reserved_cpus == [0, 1, 8, 9, 10, 11]


def test_insufficient_physical_cores_is_reported(tmp_path):
    rows = read_topology(make_sysfs(tmp_path, 6))
    with pytest.raises(InsufficientCpuError, match="6 online physical cores"):
        build_layout(rows)
    # The same host is acceptable for a smaller, explicitly configured pool.
    assert len(build_layout(rows, experimental_physical_cores=4).pool_cpus) == 8


def test_partially_offline_sibling_is_rejected(tmp_path):
    rows = read_topology(make_sysfs(tmp_path, 8, offline=(15,)))
    with pytest.raises(InsufficientCpuError, match="differing online thread counts"):
        build_layout(rows)


def test_inconsistent_sibling_information_is_rejected(tmp_path):
    root = make_sysfs(tmp_path, 8)
    (root / "cpu3" / "topology" / "thread_siblings_list").write_text("3,12\n")
    with pytest.raises(InsufficientCpuError, match="disagrees"):
        build_layout(read_topology(root))


def test_invalid_policy_and_empty_sysfs(tmp_path):
    rows = read_topology(make_sysfs(tmp_path / "ok", 8))
    with pytest.raises(ValueError):
        build_layout(rows, smt_policy="half")
    with pytest.raises(ValueError):
        build_layout(rows, reserved_physical_cores=0)
    (tmp_path / "empty").mkdir()
    with pytest.raises(InsufficientCpuError):
        read_topology(tmp_path / "empty")


def test_detect_layout_on_this_host_matches_sysfs():
    config = dict(experimental_physical_cores=6, reserved_physical_cores=2, smt_policy="siblings")
    try:
        layout = topology.detect_layout(config)
    except InsufficientCpuError as exc:
        pytest.skip(f"host too small for the experiment: {exc}")
    assert len(layout.experimental_cores) == 6
    for core in layout.experimental_cores:
        siblings = (
            topology.SYSFS_CPU / f"cpu{core[0]}" / "topology" / "thread_siblings_list"
        ).read_text()
        assert parse_cpu_list(siblings) == core
