"""CPU topology detection and the experimental CPU pool.

Nothing here assumes a logical CPU numbering scheme: physical cores are found
from sysfs ``topology/`` files, and SMT siblings are whatever the kernel says
share a core.

Policy (identical for every condition):
* the two physical cores with the lowest (package, core) identifiers, which
  include logical CPU 0 on this class of machine, are reserved for the OS, the
  runner, the samplers and grading;
* the next six physical cores form one shared pool for every agent;
* SMT: by default the pool contains both hardware threads of each of its six
  cores (``smt_policy="siblings"``). Reserving and allocating whole cores keeps
  the reserved/experimental boundary on physical-core lines, so no reserved
  thread shares a core with an experimental thread. ``"primary"`` exposes one
  thread per core instead and leaves the sibling threads unused by anyone in
  the workload.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

SMT_POLICIES = ("siblings", "primary")
SYSFS_CPU = Path("/sys/devices/system/cpu")


class InsufficientCpuError(RuntimeError):
    """The host cannot provide the reserved cores plus the experimental pool."""


def parse_cpu_list(text: str) -> list:
    cpus = set()
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        bounds = [int(x) for x in part.split("-")]
        cpus.update(range(bounds[0], bounds[-1] + 1))
    return sorted(cpus)


def format_cpu_list(cpus) -> str:
    """Compact kernel-style list: [2,3,4,10] -> '2-4,10'."""
    cpus = sorted(set(int(c) for c in cpus))
    parts, start = [], None
    previous = None
    for cpu in cpus:
        if start is None:
            start = previous = cpu
        elif cpu == previous + 1:
            previous = cpu
        else:
            parts.append(f"{start}-{previous}" if previous != start else str(start))
            start = previous = cpu
    if start is not None:
        parts.append(f"{start}-{previous}" if previous != start else str(start))
    return ",".join(parts)


def read_topology(sysfs: Path = SYSFS_CPU) -> list:
    """One row per online logical CPU, straight from sysfs."""
    online_file = sysfs / "online"
    online = parse_cpu_list(online_file.read_text()) if online_file.exists() else None
    rows = []
    for directory in sorted(sysfs.glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        cpu = int(directory.name[3:])
        if online is not None and cpu not in online:
            continue
        topology = directory / "topology"
        if not (topology / "core_id").exists():
            continue

        def read(name, default=None):
            try:
                return (topology / name).read_text().strip()
            except OSError:
                return default

        def read_int(path):
            try:
                return int(path.read_text().strip())
            except (OSError, ValueError):
                return None

        nodes = [int(p.name[4:]) for p in directory.glob("node[0-9]*")]
        rows.append(
            dict(
                cpu=cpu,
                core=int(read("core_id")),
                package=int(read("physical_package_id", "0")),
                node=nodes[0] if nodes else 0,
                siblings=parse_cpu_list(read("thread_siblings_list", str(cpu))),
                max_khz=read_int(directory / "cpufreq" / "cpuinfo_max_freq"),
            )
        )
    if not rows:
        raise InsufficientCpuError(f"No CPU topology found under {sysfs}")
    return rows


@dataclass
class CpuLayout:
    smt_policy: str
    smt_active: bool
    physical_cores: int
    logical_cpus: int
    numa_nodes: list
    reserved_cores: list  # [[cpu, ...], ...] one entry per reserved physical core
    experimental_cores: list  # one entry per experimental physical core
    reserved_cpus: list  # logical CPUs for OS, runner, samplers, grading
    pool_cpus: list  # logical CPUs every agent shares
    unused_cpus: list  # sibling threads left idle under smt_policy="primary"
    heterogeneous_max_frequency: bool
    topology: list

    @property
    def pool_arg(self) -> str:
        return format_cpu_list(self.pool_cpus)

    @property
    def reserved_arg(self) -> str:
        return format_cpu_list(self.reserved_cpus)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.update(pool_arg=self.pool_arg, reserved_arg=self.reserved_arg)
        return data


def build_layout(
    rows: list,
    *,
    experimental_physical_cores: int = 6,
    reserved_physical_cores: int = 2,
    smt_policy: str = "siblings",
) -> CpuLayout:
    if smt_policy not in SMT_POLICIES:
        raise ValueError(f"smt_policy must be one of {SMT_POLICIES}")
    if experimental_physical_cores < 1 or reserved_physical_cores < 1:
        raise ValueError("At least one experimental and one reserved physical core is required")
    cores: dict = {}
    for row in rows:
        cores.setdefault((row["package"], row["core"]), set()).add(row["cpu"])
    # A core's threads must agree with the kernel's sibling lists.
    for row in rows:
        group = cores[(row["package"], row["core"])]
        online_siblings = set(row["siblings"]) & {r["cpu"] for r in rows}
        if online_siblings != group:
            raise InsufficientCpuError(
                f"CPU {row['cpu']}: sibling list {sorted(online_siblings)} disagrees with "
                f"core grouping {sorted(group)}"
            )
    groups = [sorted(cpus) for _, cpus in sorted(cores.items())]
    needed = experimental_physical_cores + reserved_physical_cores
    if len(groups) < needed:
        raise InsufficientCpuError(
            f"Host has {len(groups)} online physical cores; the experiment needs "
            f"{reserved_physical_cores} reserved + {experimental_physical_cores} experimental"
        )
    widths = {len(g) for g in groups}
    if len(widths) != 1:
        raise InsufficientCpuError(
            f"Physical cores have differing online thread counts {sorted(widths)}; "
            "bring all SMT siblings online or offline consistently"
        )
    nodes = sorted({row["node"] for row in rows})
    reserved = groups[:reserved_physical_cores]
    experimental = groups[reserved_physical_cores:needed]
    if smt_policy == "siblings":
        pool = sorted(cpu for group in experimental for cpu in group)
        unused = []
    else:
        pool = sorted(group[0] for group in experimental)
        unused = sorted(cpu for group in experimental for cpu in group[1:])
    # Cores beyond the reserved and experimental sets join the reserved side.
    spare = sorted(cpu for group in groups[needed:] for cpu in group)
    reserved_cpus = sorted([cpu for group in reserved for cpu in group] + spare)
    frequencies = {row["max_khz"] for row in rows if row["cpu"] in pool and row["max_khz"]}
    return CpuLayout(
        smt_policy=smt_policy,
        smt_active=widths.pop() > 1,
        physical_cores=len(groups),
        logical_cpus=len(rows),
        numa_nodes=nodes,
        reserved_cores=reserved,
        experimental_cores=experimental,
        reserved_cpus=reserved_cpus,
        pool_cpus=pool,
        unused_cpus=unused,
        heterogeneous_max_frequency=len(frequencies) > 1,
        topology=rows,
    )


def detect_layout(cpu_config: dict, sysfs: Path = SYSFS_CPU) -> CpuLayout:
    return build_layout(
        read_topology(sysfs),
        experimental_physical_cores=cpu_config["experimental_physical_cores"],
        reserved_physical_cores=cpu_config["reserved_physical_cores"],
        smt_policy=cpu_config["smt_policy"],
    )
