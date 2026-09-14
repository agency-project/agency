"""Generate CPU placement from a read-only inventory, without running Agency."""

import argparse
import json
import os
from pathlib import Path
import subprocess


def parse_topology(text):
    rows = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        cpu, core, socket, node, online = line.split(",")
        if online == "Y":
            rows.append(
                {"cpu": int(cpu), "core": int(core), "socket": int(socket), "node": int(node)}
            )
    return sorted(rows, key=lambda row: row["cpu"])


def make_layout(topology, ns, node=0):
    cores = {}
    for row in topology:
        if row["node"] == node:
            cores.setdefault((row["socket"], row["core"]), []).append(row["cpu"])
    groups = [sorted(cpus) for _, cpus in sorted(cores.items())]
    if len(groups) < 5:
        raise ValueError("Need at least five exposed cores in the selected NUMA node")
    # Reserve complete SMT groups; never place two agents on sibling threads.
    os_cpus = sorted(groups[0] + groups[1])
    driver_cpus = sorted(groups[2] + groups[3])
    workers = [group[0] for group in groups[4:]]
    levels = {}
    for n in ns:
        if n < 1 or n > len(workers):
            raise ValueError(
                f"N={n} exceeds {len(workers)} separate worker cores after reservation"
            )
        levels[str(n)] = [{"slot": i, "cpus": [workers[i]], "mems": [node]} for i in range(n)]
    return {
        "schema_version": 1,
        "topology": topology,
        "numa_node": node,
        "os_runtime_headroom_cpus": os_cpus,
        "driver_engine_monitor_cpus": driver_cpus,
        "worker_cpus": workers,
        "worker_smt_siblings_unused_by_benchmark": [g[1:] for g in groups[4:]],
        "max_separate_core_agents": len(workers),
        "levels": levels,
        "exclusive_host_isolation": False,
        "note": "Host services and other users are not repinned. Headroom is an allocation, not an exclusive reservation.",
    }


def validate_layout(layout, topology, ns):
    expected = make_layout(topology, [int(n) for n in layout["levels"]], layout["numa_node"])
    if layout != expected:
        raise ValueError(
            "Saved topology/placement differs from current host or deterministic policy"
        )
    for n in ns:
        if str(n) not in layout["levels"]:
            raise ValueError(f"N={n} is absent from the reviewed CPU layout")


def apply_driver_affinity(layout, ns):
    result = subprocess.run(
        ["lscpu", "-p=CPU,CORE,SOCKET,NODE,ONLINE"], capture_output=True, text=True, check=True
    )
    validate_layout(layout, parse_topology(result.stdout), ns)
    if len(list(Path("/proc/self/task").iterdir())) != 1:
        raise RuntimeError("Set driver affinity before importing Agency or starting any threads")
    wanted = set(layout["driver_engine_monitor_cpus"])
    if not wanted <= os.sched_getaffinity(0):
        raise RuntimeError("Driver CPU allocation is outside the launching process allowed set")
    os.sched_setaffinity(0, wanted)
    if os.sched_getaffinity(0) != wanted:
        raise RuntimeError("Driver affinity was not applied")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--ns", nargs="+", type=int, default=[1, 2, 4, 8, 12])
    parser.add_argument("--node", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    topology = parse_topology(inventory["commands"]["topology"]["stdout"])
    layout = make_layout(topology, args.ns, args.node)
    with args.out.open("x") as output:
        json.dump(layout, output, indent=2)
