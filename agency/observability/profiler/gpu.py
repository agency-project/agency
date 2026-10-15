"""Low-overhead NVML gauges, with explicit identity and collection state.

Each gauge has a numeric ``:state`` companion so the existing live event and
saved counter transports preserve failures without inventing a numeric value.
States: 0 valid, 1 no new samples, 2 unsupported, 3 collection failure,
4 stale (e.g. an exited process), 5 unavailable (e.g. an NVML memory sentinel).
Process keys contain device index/UUID, PID/start ticks, and escaped sandbox ID.
Legacy label-only counters remain readable but are no longer emitted.
Process SM activity is never summed. ``processes:mem_mb`` is reported compute
process VRAM, independent of device memory; sandbox totals use the same rows.
"""

import math
import os
import time
from pathlib import Path
from urllib.parse import quote

VALID, NO_SAMPLES, UNSUPPORTED, ERROR, STALE, UNAVAILABLE = range(6)


class GPUCollector:
    def __init__(self, nvml, samples, scopes, *, proc_root=Path("/proc")):
        self.nvml, self.samples, self.scopes = nvml, samples, scopes
        self.proc_root = proc_root
        self.started_us = time.time_ns() // 1000
        self.boot_us = self.started_us - time.clock_gettime_ns(time.CLOCK_BOOTTIME) // 1000
        self.clock_ticks = os.sysconf("SC_CLK_TCK")
        self.last = {}
        self.known = {}
        self.owners = {}
        self.devices = {}
        self.totals = {}

    def state(self, t, name, state):
        self.samples.append((t, name + ":state", float(state)))

    def value(self, t, name, value):
        self.samples.append((t, name, float(value)))
        self.state(t, name, VALID)

    def query(self, call):
        try:
            return call(), VALID
        except Exception as exc:
            # NVML exposes numeric error codes; mocks and older bindings need
            # not define every exception subclass.
            code = getattr(exc, "value", None)
            if code == getattr(self.nvml, "NVML_ERROR_NOT_SUPPORTED", 3):
                return None, UNSUPPORTED
            if code == getattr(self.nvml, "NVML_ERROR_NOT_FOUND", 6):
                return None, NO_SAMPLES
            return None, ERROR

    def identity(self, pid):
        """Refresh cgroup ownership on every tick; never cache by PID alone."""
        root = self.proc_root / str(pid)
        start = None
        sandbox = None
        try:
            stat = (root / "stat").read_text()
            start = int(stat[stat.rfind(") ") + 2 :].split()[19])
            cg = (root / "cgroup").read_text()
            relative = next(line[3:] for line in cg.splitlines() if line.startswith("0::"))
            full = "/sys/fs/cgroup/" + relative.lstrip("/")
            # Most specific scope wins if nested registrations exist.
            for label, path in sorted(self.scopes().items(), key=lambda item: -len(item[1])):
                if full == path or full.startswith(path + "/"):
                    sandbox = label
                    break
            again = (root / "stat").read_text()
            if int(again[again.rfind(") ") + 2 :].split()[19]) != start:
                return f"{pid}-unknown", None, None
        except (OSError, ValueError, IndexError, StopIteration):
            # Keep the PID measurement but do not infer an owner from comm.
            sandbox = None
        born_us = (
            self.boot_us + start * 1_000_000 // self.clock_ticks if start is not None else None
        )
        return f"{pid}-{start if start is not None else 'unknown'}", sandbox, born_us

    def tick(self, t, i, handle, sampled_at):
        device = f"gpu{i}"
        if i not in self.devices:
            uuid, _ = self.query(lambda: self.nvml.nvmlDeviceGetUUID(handle))
            if isinstance(uuid, bytes):
                uuid = uuid.decode(errors="replace")
            self.devices[i] = quote(str(uuid or device), safe="")
        device_values = {}
        for metric, call in (
            ("util_pct", lambda: self.nvml.nvmlDeviceGetUtilizationRates(handle).gpu),
            ("mem_mb", lambda: self.nvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20),
            ("power_w", lambda: self.nvml.nvmlDeviceGetPowerUsage(handle) / 1000),
        ):
            value, state = self.query(call)
            name = f"{device}:{metric}"
            if state == VALID and isinstance(value, (int, float)) and math.isfinite(value):
                self.value(t, name, value)
                device_values[metric] = value
            else:
                self.state(t, name, UNAVAILABLE if state == VALID else state)
        self.processes(t, i, handle, sampled_at, device_values)

    def processes(self, t, i, handle, sampled_at, device_values):
        device = f"gpu{i}"
        wall_us = int(sampled_at * 1_000_000)
        old = self.known.get(i, set())
        active = set()
        identities = {}

        def identify(pid):
            if pid not in identities:
                identity, sandbox, born = self.identity(pid)
                owner = quote(sandbox, safe="") if sandbox is not None else "~"
                prefix = f"{device}:proc:{identity}:sandbox:{owner}:device:{self.devices[i]}"
                prior = self.owners.get((i, identity))
                # A migrated process's buffered samples cannot be assigned to
                # its new sandbox before we observed that ownership change.
                since = prior[1] if prior and prior[0] == owner else wall_us if prior else 0
                self.owners[i, identity] = (owner, since)
                identities[pid] = (prefix, sandbox, max(born or 0, since))
            return identities[pid]

        rows, memory_state = self.query(
            lambda: self.nvml.nvmlDeviceGetComputeRunningProcesses(handle)
        )
        total_name = f"{device}:processes:mem_mb"
        totals = {total_name: 0.0}
        invalid = set()
        seen = {}
        memory_prefixes = set()
        if memory_state == VALID:
            # Repeated identical NVML rows are one process. Conflicting rows
            # (e.g. instance-scoped reporting) are unavailable, not guessed.
            for row in rows:
                if row.pid in seen and seen[row.pid] != row.usedGpuMemory:
                    seen[row.pid] = None
                else:
                    seen.setdefault(row.pid, row.usedGpuMemory)
            for pid, value in seen.items():
                prefix, sandbox, _ = identify(pid)
                name = prefix + ":mem_mb"
                active.add(prefix)
                memory_prefixes.add(prefix)
                group = (
                    f"{device}:sandbox:{quote(sandbox, safe='')}:process_mem_mb"
                    if sandbox is not None
                    else None
                )
                if group:
                    totals.setdefault(group, 0.0)
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0 <= value < 2**64 - 1
                ):
                    invalid.update(filter(None, [name, total_name, group]))
                    continue
                self.value(t, name, value / 2**20)
                totals[total_name] += value / 2**20
                if group:
                    totals[group] += value / 2**20
            for name, value in totals.items():
                if name not in invalid:
                    self.value(t, name, value)
            for name in invalid:
                self.state(t, name, UNAVAILABLE)
        else:
            self.state(t, total_name, memory_state)
        # Previously emitted sandbox totals must not retain a fresh-looking
        # value after process exit, migration, or a failed enumeration.
        previous_totals = self.totals.get(i, set())
        for name in previous_totals - totals.keys():
            if memory_state == VALID:
                self.value(t, name, 0)
            else:
                self.state(t, name, memory_state)
        self.totals[i] = previous_totals | totals.keys()

        last = self.last.get(i, self.started_us)
        rows, util_state = self.query(
            lambda: self.nvml.nvmlDeviceGetProcessUtilization(handle, last)
        )
        newest = last
        sampled = set()
        latest_util = {}
        seen_samples = set()
        if util_state == VALID:
            for row in sorted(rows, key=lambda row: row.timeStamp):
                timestamp = row.timeStamp
                if timestamp <= last or timestamp > wall_us:
                    continue
                newest = max(newest, timestamp)
                prefix, _, born = identify(row.pid)
                if timestamp < born or (row.pid, timestamp) in seen_samples:
                    continue
                seen_samples.add((row.pid, timestamp))
                if not 0 <= row.smUtil <= 100:
                    continue
                active.add(prefix)
                sampled.add(prefix)
                latest_util[prefix] = row.smUtil
                at = t - (wall_us - timestamp) * 1000
                # Measurement time and collection state are separate clocks:
                # a buffered sample can precede the previous empty poll.
                self.samples.append((at, prefix + ":util_pct", float(row.smUtil)))
            self.last[i] = newest
            if not sampled:
                util_state = NO_SAMPLES
        self.state(t, f"{device}:processes:util_pct", util_state)
        for prefix in old | active:
            if prefix in sampled:
                self.state(t, prefix + ":util_pct", VALID)
            # Preserve the legacy, explicitly estimated power series under
            # distinct process keys. It is never added to measured power.
            if prefix in latest_util and "power_w" in device_values and "util_pct" in device_values:
                share = min(1.0, latest_util[prefix] / max(device_values["util_pct"], 1.0))
                self.value(t, prefix + ":power_w_est", device_values["power_w"] * share)
            else:
                self.state(t, prefix + ":power_w_est", UNAVAILABLE)
            if prefix not in sampled:
                self.state(
                    t,
                    prefix + ":util_pct",
                    STALE
                    if memory_state == VALID and prefix not in active
                    else NO_SAMPLES
                    if util_state == VALID
                    else util_state,
                )
            if prefix not in memory_prefixes or memory_state != VALID:
                self.state(t, prefix + ":mem_mb", STALE if memory_state == VALID else memory_state)
        self.known[i] = active if memory_state == VALID else old | active
