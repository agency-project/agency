"""Stopped-task handoff between Agency's tracer and CRIU.

Only the owning tracer thread executes these functions. PTRACE_INTERRUPT
requires SEIZE attachment, so the COW path uses SEIZE from initial launch.
The detached tree is job-control stopped. The CRIU pre-resume hook also
freezes restored tasks until Agency seizes them: the tested CRIU/kernel does
not reliably retain the job-control stop. Reattachment never authorizes a
new attempt to run.
"""

import os
import errno
import signal
import time
from pathlib import Path


def task_status(pid):
    raw = Path(f"/proc/{pid}/status").read_text()
    return dict(line.split(":", 1) for line in raw.splitlines() if ":" in line)


def _wait_stops(loop, pids, deadline):
    from . import _ctypes_defs as pt

    while True:
        with loop._lock:
            live = set(loop._known_pids)
            parked = set(loop._parked_pids)
            loop._held_pids |= live
        # Fork/clone events discovered during the handoff expand its scope.
        for pid in live - pids - parked:
            try:
                pt.ptrace(pt.PTRACE_INTERRUPT, pid, 0, 0)
            except pt.PtraceError as exc:
                if exc.errno != errno.ESRCH:
                    raise
                # Exec from a non-leader can remove a former TID without
                # another wait status. Forget only an affirmatively absent task.
                if not Path(f"/proc/{pid}").exists():
                    loop._forget(pid, -1)
                # Otherwise drain the exiting descendant's wait status below.
            pids.add(pid)
        if live and live <= parked:
            return live
        if not live:
            raise RuntimeError("CLI exited during checkpoint handoff")
        if time.monotonic() >= deadline:
            states = {}
            for missing in live - parked:
                try:
                    states[missing] = task_status(missing)["State"].strip()
                except FileNotFoundError:
                    states[missing] = "gone"
            raise TimeoutError(f"Process tree did not stop for checkpoint handoff: {states}")
        pid, status = os.waitpid(-1, os.WNOHANG | pt.WAIT_ALL | pt.WAIT_NOTHREAD)
        if pid:
            if pid not in loop.live_pids():
                loop._remember_spawn(pid, is_process=None)
                with loop._lock:
                    loop._held_pids.add(pid)
            loop._dispatch(pid, status)
            # A seccomp/fork/exec stop can consume the pending interrupt and
            # resume the task. Interrupt it again until it is actually parked.
            pids.discard(pid)
        else:
            time.sleep(0.002)


def detach(loop, timeout):
    from . import _ctypes_defs as pt

    if loop._checkpoint_detached:
        return
    if not loop._seize_mode:
        raise RuntimeError("Checkpoint handoff requires a SEIZE-attached Agency tracer")
    deadline = time.monotonic() + timeout
    live = _wait_stops(loop, set(), deadline)
    # Queue an unmaskable job-control stop while every task is ptrace-stopped.
    # No tracee executes userspace between DETACH and the kernel's group stop.
    groups = {int(task_status(pid)["Tgid"]) for pid in live}
    for pid in groups:
        os.kill(pid, signal.SIGSTOP)
    loop._checkpoint_detached = True
    for pid in sorted(live, reverse=True):
        pt.ptrace(pt.PTRACE_DETACH, pid, 0, 0)
    while True:
        statuses = [task_status(pid) for pid in live]
        if all(s["State"].strip().startswith("T") and int(s["TracerPid"]) == 0 for s in statuses):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Detached CLI did not reach untraced group-stop")
        time.sleep(0.002)
    loop._checkpoint_group_stopped = True


def reattach(loop, timeout):
    from . import _ctypes_defs as pt

    if not loop._checkpoint_detached:
        return
    with loop._lock:
        pids = set(loop._known_pids)
        loop._parked_pids.clear()
        loop._held_pids = set(pids)
    if not loop._checkpoint_seized:
        for pid in pids:
            status = task_status(pid)
            if not status["State"].strip().startswith("T") or int(status["TracerPid"]):
                raise RuntimeError(f"Restored task {pid} is not untraced and stopped")
        for pid in pids:
            pt.ptrace(pt.PTRACE_SEIZE, pid, 0, pt.ALL_TRACE_OPTIONS)
    _wait_stops(loop, set(), time.monotonic() + timeout)
    loop._checkpoint_detached = False
    loop._checkpoint_seized = False
    # Keep every task parked until the daemon binds the next attempt/policy.


def seize_frozen(loop, timeout):
    from . import _ctypes_defs as pt

    if not loop._checkpoint_detached:
        return
    with loop._lock:
        pids = set(loop._known_pids)
        loop._held_pids = set(pids)
        loop._parked_pids.clear()
    for pid in pids:
        status = task_status(pid)
        group = Path(f"/proc/{pid}/cgroup").read_text().strip()
        if int(status["TracerPid"]) or not group.endswith("/agency-criu-handoff"):
            raise RuntimeError(f"Restored task {pid} is not in the restore freezer")
    for pid in pids:
        pt.ptrace(pt.PTRACE_SEIZE, pid, 0, pt.ALL_TRACE_OPTIONS)
        pt.ptrace(pt.PTRACE_INTERRUPT, pid, 0, 0)
    loop._checkpoint_seized = True


def resume_groups(loop):
    if not loop._checkpoint_group_stopped:
        return
    groups = {int(task_status(pid)["Tgid"]) for pid in loop.live_pids()}
    for pid in groups:
        os.kill(pid, signal.SIGCONT)
    loop._checkpoint_group_stopped = False
