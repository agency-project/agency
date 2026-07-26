"""Host-side counterpart to `_in_container_entrypoint.py` -- deploys that
self-contained script into a docker/podman-backed sandbox container and
drives it over `docker/podman exec -i`'s stdio, translating its JSON event
protocol into real `agpolicy.check()` calls.

Exposes `InContainerRelay`, which duck-types the same surface
`agProxyPtraceHandle` (agproxy_ptrace.py) already expects from a `TracerLoop`
(`join`, `read_output`, `live_pids`, `on_spawn`, `on_exit`, `kill`) -- so
`agProxyPtraceHandle(relay)` wraps it completely unchanged, and every
existing caller (`wire_to_sandbox`, `agharness_backends/claude_code.py`)
keeps using the exact same handle interface regardless of which launch path
produced it. See docs/Design_harness_integration.md's "Prerequisites"
(Component 3) for why this exists: a docker/podman-backed sandbox's
container has its own PID namespace, and a fork() on the host cannot land a
traced child inside it -- `docker exec` attaches into the container's own
namespaces, so forking *inside* the exec'd process (which is exactly what
`_in_container_entrypoint.py` does) is what actually solves this.

Empirically verified against a real running container (agency-sandbox
image, docker exec -i): correct container-local PIDs, correct cwd
resolution against the container's own filesystem, a write landing in the
container's real `/workspace` (confirmed independently), and a `deny`
decision correctly blocking one specific execve while allowing others.
"""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ...agent import agent
    from ...agpolicy import agpolicy

_ENTRYPOINT_SOURCE = (Path(__file__).parent / "_in_container_entrypoint.py").read_bytes()
_ENTRYPOINT_CONTAINER_PATH = "/tmp/.agproxy_ptrace_entrypoint.py"
_RELAY_SOURCE = (Path(__file__).parent / "_tcp_to_uds_relay.py").read_bytes()
_RELAY_CONTAINER_PATH = "/tmp/.agproxy_tcp_to_uds_relay.py"
_MOUNTED_GATEWAY_DIR = "/var/run/agency_llm_gateway"


def deploy_entrypoint(sandbox) -> str:
    """Write the self-contained entrypoint script into *sandbox*'s
    container filesystem via the same `write_file_bytes` primitive native
    tool dispatch already uses -- no docker-cp, no new sandbox-side
    machinery. Returns the in-container path to invoke. Idempotent: safe to
    call before every launch (re-writing identical bytes is cheap and
    avoids needing to track "did we already deploy this" state across
    launches, which would otherwise need to survive container
    restart/hibernate)."""
    sandbox.write_file_bytes(_ENTRYPOINT_CONTAINER_PATH, _ENTRYPOINT_SOURCE)
    return _ENTRYPOINT_CONTAINER_PATH


class InContainerRelay:
    """One launch's worth of state: the `docker/podman exec -i` subprocess,
    its reader thread, and the pid/callback bookkeeping `agProxyPtraceHandle`
    needs. Construct fresh per launch, matching `TracerLoop`'s own
    not-reusable convention."""

    def __init__(self, sandbox, policy: "agpolicy", ag: "agent | None") -> None:
        self._sandbox = sandbox
        self._policy = policy
        self._ag = ag
        self._proc: "subprocess.Popen | None" = None

        self._known_pids: "set[int]" = set()
        self._options_lock = threading.Lock()
        self._returncode: "int | None" = None
        self._stdout = ""
        self._stderr = ""
        self._error: "str | None" = None
        self._finished = threading.Event()

        self._spawn_callbacks: "list[Callable[[int], None]]" = []
        self._exit_callbacks: "list[Callable[[int, int], None]]" = []
        self._spawn_log: "list[int]" = []
        self._exit_log: "list[tuple[int, int]]" = []

        self._stdin_lock = threading.Lock()
        self._reader_thread: "threading.Thread | None" = None

    # -- registration: same replay-safe contract as TracerLoop -------------

    def on_spawn(self, callback: "Callable[[int], None]") -> None:
        with self._options_lock:
            backlog = list(self._spawn_log)
            self._spawn_callbacks.append(callback)
        for pid in backlog:
            callback(pid)

    def on_exit(self, callback: "Callable[[int, int], None]") -> None:
        with self._options_lock:
            backlog = list(self._exit_log)
            self._exit_callbacks.append(callback)
        for pid, code in backlog:
            callback(pid, code)

    def live_pids(self) -> "set[int]":
        with self._options_lock:
            return set(self._known_pids)

    # -- lifecycle -----------------------------------------------------------

    def start(self, argv: "list[str]", envp: "dict[str, str]", cwd: str, syscalls: "list[str]") -> None:
        entrypoint_path = deploy_entrypoint(self._sandbox)
        runtime, container_name = self._runtime_and_container_name()

        self._proc = subprocess.Popen(
            [runtime, "exec", "-i", container_name, "python3", entrypoint_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        spec = {"argv": argv, "envp": envp, "cwd": cwd, "syscalls": list(syscalls)}
        with self._stdin_lock:
            self._proc.stdin.write(json.dumps(spec) + "\n")
            self._proc.stdin.flush()

        # Block until the root process is confirmed spawned (or launch
        # failed) -- same contract as TracerLoop.start(): the caller gets
        # a live handle back, not one that might still silently fail to
        # ever start.
        started = threading.Event()

        def _unblock_on_first_event(*_args) -> None:
            started.set()

        self._reader_thread = threading.Thread(
            target=self._read_loop, args=(started,), name="agproxy_ptrace-in-container", daemon=True,
        )
        self._reader_thread.start()
        started.wait(timeout=30)
        if self._error is not None:
            raise RuntimeError(f"in-container ptrace entrypoint failed to launch: {self._error}")

    def _runtime_and_container_name(self) -> "tuple[str, str]":
        # Reaches into the sandbox backend's private runtime/name accessors
        # -- the same established convention `wire_to_sandbox` already uses
        # (`sandbox._backend.ingest_ptrace_pids`) rather than adding new
        # public surface to agsandbox for a caller this internal.
        return _runtime_and_container_name(self._sandbox)

    def _read_loop(self, started: threading.Event) -> None:
        proc = self._proc
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = msg.get("type")
            if kind == "spawn":
                self._remember_spawn(msg["pid"])
                started.set()
            elif kind == "exit":
                self._forget(msg["pid"], msg["code"])
            elif kind == "event":
                self._handle_event(msg)
            elif kind == "result":
                self._stdout = msg.get("stdout", "")
                self._stderr = msg.get("stderr", "")
                if self._returncode is None:
                    self._returncode = msg.get("returncode", -1)
                break
            elif kind == "error":
                self._error = msg.get("message", "unknown error")
                started.set()
                break
        try:
            proc.stderr.read()
        except Exception:
            pass
        self._finished.set()

    def _handle_event(self, msg: dict) -> None:
        from ...agpolicy import agdecision  # local import: avoid import cycle at module load
        from ..agproxy_ptrace import agsyscallevent

        event = agsyscallevent(
            syscall=msg["syscall"], pid=msg["pid"], tid=msg["pid"],
            argv=msg.get("argv"), envp=msg.get("envp"), path=msg.get("path"),
            timestamp=msg.get("timestamp", 0.0),
        )
        decision = self._policy.check(self._ag, event)
        reply = {"type": "decision", "kind": decision.kind, "new_args": decision.new_args}
        with self._stdin_lock:
            self._proc.stdin.write(json.dumps(reply) + "\n")
            self._proc.stdin.flush()

    def _remember_spawn(self, pid: int) -> None:
        with self._options_lock:
            self._known_pids.add(pid)
            self._spawn_log.append(pid)
            callbacks = list(self._spawn_callbacks)
        for cb in callbacks:
            cb(pid)

    def _forget(self, pid: int, exit_code: int) -> None:
        # Deliberately does NOT touch self._returncode here -- that must
        # only ever come from the entrypoint's own final "result" message,
        # which tracks its root pid specifically (_in_container_entrypoint's
        # _Tracer._forget mirrors this). A shell whose denied child exits
        # with e.g. 126 before the shell itself finishes would otherwise
        # have that child's exit code mistaken for the whole launch's
        # returncode -- caught by the real-API probe during development
        # (a `/bin/true` denial: sh's forked-but-never-exec'd child exits
        # 126 first, well before the root sh process finishes and exits 0).
        with self._options_lock:
            self._known_pids.discard(pid)
            self._exit_log.append((pid, exit_code))
            callbacks = list(self._exit_callbacks)
        for cb in callbacks:
            cb(pid, exit_code)

    def read_output(self) -> "tuple[str, str]":
        return self._stdout, self._stderr

    def join(self, timeout: "float | None" = None) -> "int | None":
        if not self._finished.wait(timeout):
            return None
        if self._reader_thread is not None:
            self._reader_thread.join()
        if self._error is not None:
            return -1
        return self._returncode if self._returncode is not None else -1

    def kill(self) -> None:
        # Reach the traced tree via the container's own namespace -- a
        # `docker/podman exec` invocation of `kill` runs inside the same
        # PID namespace as the pids we're tracking, so it can signal them
        # directly by the pids we already know, without needing a
        # dedicated out-of-band control message on the entrypoint's stdio
        # protocol (which only reads synchronously between events today).
        runtime, container_name = self._runtime_and_container_name()
        for pid in self.live_pids():
            try:
                subprocess.run(
                    [runtime, "exec", container_name, "kill", "-9", str(pid)],
                    capture_output=True, timeout=5,
                )
            except Exception:
                pass
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass


def start_llm_relay(sandbox, uds_path: str) -> "tuple[subprocess.Popen, int]":
    """Deploy and launch `_tcp_to_uds_relay.py` inside *sandbox*'s container,
    forwarding a container-local TCP port to the host's *uds_path* (already
    started via `agProxyLLM.ensure_uds_started()`, and already visible
    inside the container at `_MOUNTED_GATEWAY_DIR` because
    `agsandbox.agSandbox.__init__` bind-mounts that directory into every
    container-backed sandbox unconditionally). Returns `(process, port)` --
    the caller owns the process's lifetime (`stop_llm_relay`) and points
    the harness's `ANTHROPIC_BASE_URL` at `http://127.0.0.1:<port>`, which
    is the CONTAINER's own loopback (the relay listens there), not the
    host's -- reachable regardless of the container runtime's networking
    mode, since neither side of this hop ever leaves the container's
    network namespace."""
    sandbox.write_file_bytes(_RELAY_CONTAINER_PATH, _RELAY_SOURCE)
    runtime, container_name = _runtime_and_container_name(sandbox)
    container_sock_path = f"{_MOUNTED_GATEWAY_DIR}/{Path(uds_path).name}"

    port_out, port_rc = sandbox.exec(
        "python3 -c \"import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); "
        "print(s.getsockname()[1]); s.close()\"",
        workdir="/",
    )
    if port_rc != 0:
        raise RuntimeError(f"failed to pick a free port inside the container: {port_out}")
    port = int(port_out.strip())

    proc = subprocess.Popen(
        [runtime, "exec", "-i", container_name, "python3", _RELAY_CONTAINER_PATH,
         container_sock_path, str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    ready_line = proc.stdout.readline()
    if ready_line.strip() != "READY":
        proc.terminate()
        raise RuntimeError(f"in-container TCP-to-UDS relay failed to start: {ready_line!r}")
    return proc, port


def stop_llm_relay(proc: "subprocess.Popen | None") -> None:
    if proc is not None:
        proc.terminate()


def _runtime_and_container_name(sandbox) -> "tuple[str, str]":
    backend = sandbox._backend
    backend._ensure_started()
    return backend._runtime, backend._container_name()


__all__ = ["InContainerRelay", "deploy_entrypoint", "start_llm_relay", "stop_llm_relay"]
