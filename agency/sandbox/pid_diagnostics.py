"""Opt-in observations. Never updates the sandbox's process-tracking sets."""

import json
import time
import copy

MARKER = "__AGENCY_PID_IDENTITY__:"

# Shell builtins only: no observer child gets added to the before/after PID diff.
IDENTITY_SHELL = (
    '__stat=""; IFS= read -r __stat < "$__d/stat" 2>/dev/null || :\n'
    '__harness=unknown; __cmd=""\n'
    'IFS= read -r __cmd < "$__d/cmdline" 2>/dev/null || :\n'
    'case "$__cmd" in *agency.harness.daemon*) __harness=harness_daemon ;; esac\n'
    f'printf "\\n{MARKER}%s|%s|%s\\n" "$__p" "$__harness" "$__stat"\n'
)


def parse_stat(raw):
    """Linux stat field 22 is a boot-relative identity, not a wall-clock time."""
    left = raw.index("(")
    right = raw.rindex(")")
    fields = raw[right + 2 :].split()
    return {
        "comm": raw[left + 1 : right],
        "state": fields[0],
        "ppid": int(fields[1]),
        "start_ticks": int(fields[19]),
    }


def extract_identities(output):
    identities = {}
    clean = []
    for line in output.splitlines(keepends=True):
        if line.startswith(MARKER):
            try:
                pid, harness, raw = line[len(MARKER) :].strip().split("|", 2)
                identities[int(pid)] = {**parse_stat(raw), "harness_identity": harness}
            except (ValueError, IndexError):
                continue  # Missing/racing proc entries remain explicitly unavailable.
        else:
            clean.append(line)
    return "".join(clean), identities


def register(backend, pid, source, identity=None):
    if not getattr(backend._agconfig.sandbox, "hibernation_diagnostics", False):
        return
    if not hasattr(backend, "_pid_registration_history"):
        backend._pid_registration_history = {}
    history = backend._pid_registration_history.setdefault(pid, [])
    history.append(
        {
            "registered_epoch": time.time(),
            "registered_monotonic": time.monotonic(),
            "source": source,
            "identity": identity,
            "identity_unavailable_reason": None if identity else "not observed at registration",
        }
    )


SNAPSHOT_SCRIPT = r"""
import os,json,time
from pathlib import Path
rows={}
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid(): continue
 try:
  raw=(p/'stat').read_text(); left=raw.index('('); right=raw.rindex(')'); f=raw[right+2:].split()
  cmd=(p/'cmdline').read_bytes()
  try: exe=os.path.basename(os.readlink(p/'exe'))
  except OSError: exe=None
  rows[p.name]={'comm':raw[left+1:right],'state':f[0],'ppid':int(f[1]),'start_ticks':int(f[19]),'exe_name':exe,'harness_identity':'harness_daemon' if b'agency.harness.daemon' in cmd else 'unknown'}
 except (OSError,ValueError,IndexError): pass
print(json.dumps({'processes':rows,'observed_epoch':time.time(),'clock_ticks_per_second':os.sysconf('SC_CLK_TCK'),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))
"""


def decision_snapshot(backend, pending):
    # Freeze the original decision and sets BEFORE the read-only probe. Never
    # call get_live_pids(), which adopts/prunes PIDs and changes the experiment.
    result = {
        "decision_epoch": time.time(),
        "pending_background_work": pending,
        "watched": dict(backend._watched_pids),
        "baseline_pids": sorted(backend._baseline_pids or ()),
        "daemon_pids": sorted(backend._daemon_pids),
        "infrastructure_pids": sorted(getattr(backend, "_infrastructure_pids", ())),
        "infrastructure_identities": dict(getattr(backend, "_infrastructure_pids", {})),
        "ptrace_managed_pids": sorted(backend._ptrace_managed_pids),
        "registration_history": copy.deepcopy(getattr(backend, "_pid_registration_history", {})),
        "probe_status": "unsupported_backend",
        "processes": {},
    }
    if getattr(backend, "_runtime", None) not in {"docker", "podman"}:
        return result
    start = time.perf_counter()
    try:
        response = backend._run(
            [backend._runtime, "exec", backend._name, "python3", "-c", SNAPSHOT_SCRIPT],
            timeout=10,
            check=True,
        )
        result.update(json.loads(response.stdout))
        result["probe_status"] = "observed"
    except Exception as exc:
        result["probe_status"] = "error"
        result["probe_error"] = type(exc).__name__
    result["probe_wall_s"] = time.perf_counter() - start
    result["watched_evidence"] = classify_watched(result)
    return result


def classify_watched(snapshot):
    rows = []
    for pid, watched_since in snapshot["watched"].items():
        pid = int(pid)
        current = snapshot["processes"].get(str(pid))
        history = snapshot["registration_history"].get(
            pid, snapshot["registration_history"].get(str(pid), [])
        )
        registered = history[-1].get("identity") if history else None
        infrastructure = snapshot.get("infrastructure_identities", {})
        infrastructure_start = infrastructure.get(pid, infrastructure.get(str(pid)))
        known = (
            pid in snapshot["baseline_pids"]
            or pid in snapshot["daemon_pids"]
            or (
                current is not None
                and infrastructure_start is not None
                and current["start_ticks"] == infrastructure_start
            )
        )
        harness = bool(current and current.get("harness_identity") == "harness_daemon")
        if snapshot["probe_status"] != "observed" or pid in snapshot["ptrace_managed_pids"]:
            classification = "unresolved"
        elif current is None:
            classification = "absent_at_decision_probe"
        elif registered and current["start_ticks"] != registered["start_ticks"]:
            classification = "pid_reused"
        elif current["state"] == "Z":
            classification = "zombie"
        elif known or harness:
            classification = "infrastructure"
        else:
            classification = (
                "live_process_identity_verified" if registered else "live_process_identity_unknown"
            )
        rows.append(
            {
                "pid": pid,
                "watched_since_monotonic": watched_since,
                "registered": registered,
                "current": current,
                "recognized_by_tracking_as_infrastructure": known,
                "recognized_by_probe_as_harness": harness,
                "classification": classification,
            }
        )
    return rows
