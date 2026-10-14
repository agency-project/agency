"""System-state footer for tool results: host load and pressure as seen from inside the sandbox.

/proc/loadavg and /proc/pressure are host-wide (not namespaced); /sys/fs/cgroup/*.pressure is this
sandbox's own cgroup. Stdlib only: runs inside the sandbox with the native harness.
"""
from __future__ import annotations

import os


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def _psi_some(text: str) -> "float | None":
    for line in text.splitlines():
        if line.startswith("some "):
            try:
                return float(line.split()[1].split("=")[1])
            except (IndexError, ValueError):
                return None
    return None


def snapshot() -> dict:
    out: dict = {"cpus": os.cpu_count()}
    load = _read("/proc/loadavg").split()
    if load:
        out["load1"] = float(load[0])
    for res in ("cpu", "memory", "io"):
        out[f"host_{res}_psi"] = _psi_some(_read(f"/proc/pressure/{res}"))
        out[f"sandbox_{res}_psi"] = _psi_some(_read(f"/sys/fs/cgroup/{res}.pressure"))
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith("MemAvailable:"):
            out["mem_available_gb"] = int(line.split()[1]) / 2**20
            break
    return out


def footer(elapsed_s: float, snap: "dict | None" = None) -> str:
    """One line, e.g. '[system: 41 s; host load 34.1 on 16 CPUs; CPU pressure 62% (sandbox 70%); ...]'."""
    s = snap if snap is not None else snapshot()
    parts = [f"{elapsed_s:.0f} s" if elapsed_s >= 1 else f"{elapsed_s:.1f} s"]
    if "load1" in s:
        parts.append(f"host load {s['load1']:.1f} on {s['cpus']} CPUs")
    for res, name in (("cpu", "CPU"), ("memory", "memory"), ("io", "IO")):
        host, sb = s.get(f"host_{res}_psi"), s.get(f"sandbox_{res}_psi")
        if host is None and sb is None:
            continue
        text = f"{name} pressure {host:.0f}%" if host is not None else f"{name} pressure ?"
        if sb is not None:
            text += f" (sandbox {sb:.0f}%)"
        parts.append(text)
    if "mem_available_gb" in s:
        parts.append(f"{s['mem_available_gb']:.1f} GB memory free")
    return "[system: " + "; ".join(parts) + "]"


def assess(s: dict) -> str:
    """Load level from runnable tasks per CPU and CPU pressure: 'idle', 'busy' or 'heavily loaded'."""
    per_cpu = s.get("load1", 0.0) / max(s.get("cpus") or 1, 1)
    # A sandbox throttled by its own CPU quota stalls without raising host load, so take the worse of the two.
    psi = max(s.get("host_cpu_psi") or 0.0, s.get("sandbox_cpu_psi") or 0.0)
    if per_cpu >= 1.5 or psi >= 40:
        return "heavily loaded"
    if per_cpu >= 0.7 or psi >= 10:
        return "busy"
    return "idle"


ADVICE = {
    "idle": "The machine is not under load, so this command's timing reflects the code itself.",
    "busy": ("Other workloads are using a large share of the CPUs, so CPU-bound commands may run noticeably slower "
             "than on an idle machine."),
    "heavily loaded": (
        "Other workloads are competing for the CPUs, so CPU-bound commands can run several times slower than on an "
        "idle machine, and timing-sensitive tests can time out or fail because of the load rather than the code. Do "
        "not treat slowness, a timeout or a timing-dependent failure as evidence about your change until you have "
        "ruled out the load: re-run with a longer timeout, repeat the check, compare against the unmodified code "
        "under the same conditions, or measure CPU time instead of wall time."),
}


def explained_footer(elapsed_s: float, snap: "dict | None" = None) -> str:
    """Multi-line footer that says what each number means and how the load should affect conclusions."""
    s = snap if snap is not None else snapshot()
    level = assess(s)
    lines = [f"[system load while this command ran ({elapsed_s:.0f} s wall time): machine is {level.upper()}"]
    if "load1" in s:
        lines.append(f" - load average {s['load1']:.1f} on {s['cpus']} CPUs = "
                     f"{s['load1'] / max(s['cpus'] or 1, 1):.1f} runnable tasks per CPU (above 1 means tasks queue for a CPU)")
    for res, name in (("cpu", "CPU"), ("memory", "memory"), ("io", "IO")):
        host, sb = s.get(f"host_{res}_psi"), s.get(f"sandbox_{res}_psi")
        if host is None and sb is None:
            continue
        text = f" - {name} pressure: tasks were stalled waiting for {name} {host:.0f}% of the last 10 s" if host is not None \
            else f" - {name} pressure: unknown"
        if sb is not None:
            text += f" ({sb:.0f}% inside this sandbox)"
        lines.append(text)
    if "mem_available_gb" in s:
        lines.append(f" - {s['mem_available_gb']:.1f} GB memory available")
    lines.append(f" {ADVICE[level]}]")
    return "\n".join(lines)
