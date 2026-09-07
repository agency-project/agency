from __future__ import annotations
import ctypes
import os
import queue
import re
import signal
import subprocess
import threading
import time
import traceback as _traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Iterable, TypeVar

from ..configs.agconfig import (
    GPU_DETECT_TIMEOUT_S,
    IDLE_CHECK_INTERVAL_S,
    MARKER_MB,
    MEMORY_DETECT_FALLBACK_MB,
    SYSCTL_DETECT_TIMEOUT_S,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_T = TypeVar("_T")
# Kept as a plain module attribute -- tests/conftest.py monkeypatches this by
# name (`monkeypatch.setattr(_agutil_module, "_BATCH_INTERVAL_S", 0.0)`) to
# speed up streaming tests.
_BATCH_INTERVAL_S: float = 0.1  # main thread drains stream every 100 ms


_THINKING_RE = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_PATH_RE = re.compile(r"^(/[\w.\-]+)+$")
_CAMEL_CASE_RE = re.compile(r"(?<!^)(?=[A-Z])")

# Lowercase alphanumeric alphabet for agent ID suffixes.
# 4 digits → 36⁴ = 1 679 616 unique values per noun.
_B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


def format_exception(e: BaseException) -> str:
    """Return the full traceback + exception message as a single string.

    Must be called from inside an except block so traceback.format_exc()
    captures the live stack.
    """
    tb = _traceback.format_exc()
    if tb and not tb.startswith("NoneType"):
        return tb.rstrip()
    return f"{type(e).__name__}: {e}"


@contextmanager
def sigterm_as_exit(label: str = "agency") -> "Generator[threading.Event, None, None]":
    """Install a SIGTERM handler for the duration of this ``with`` block that
    converts a plain ``kill <pid>`` into a normal Python exit (``SystemExit``)
    instead of the OS's default immediate termination.

    Without this, SIGTERM bypasses every ``atexit`` cleanup hook the
    framework relies on (live sandbox teardown in agsandbox.py, the tool
    worker pool in agtool.py, a webui/graphui server subprocess, ...) exactly
    like SIGKILL does -- the interpreter never regains control, so none of
    that ever runs. Converting SIGTERM into ``SystemExit`` here lets whatever
    code is running inside the ``with`` block unwind through its own
    ``finally`` blocks and reach normal interpreter shutdown instead, where
    those hooks fire exactly as they would on any other clean exit.

    SIGKILL itself can never be caught by any process, so there's no
    equivalent possible for it -- resuming cleanly after a SIGKILL relies on
    the framework's own self-healing (e.g. ``sandbox.container``'s
    startup orphan reaper reclaiming a dead run's containers), not on
    anything a context manager can do.

    Yields a ``threading.Event`` that's set if SIGTERM was actually received
    during the block, so callers can distinguish a signal-triggered exit from
    a normal one (e.g. to skip an otherwise-unconditional "wait for user
    input" step -- the caller asked this process to exit, not to linger for a
    second signal).

    Only installs the handler when called from the main thread --
    ``signal.signal()`` raises otherwise. From any other thread this is a
    no-op: it yields an ``Event`` that's simply never set, since a background
    thread already can't rely on Ctrl+C/KeyboardInterrupt working here either.
    *label* is used only in the message printed when SIGTERM is caught (e.g.
    ``"[agwebui] Received SIGTERM, shutting down..."``).
    """
    received = threading.Event()
    if threading.current_thread() is not threading.main_thread():
        yield received
        return

    def _handle_sigterm(signum, frame) -> None:
        received.set()
        print(f"\n[{label}] Received SIGTERM, shutting down...", flush=True)
        raise SystemExit(0)

    prev_handler = signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        yield received
    finally:
        signal.signal(signal.SIGTERM, prev_handler)


class _LLMIdleTimeout(Exception):
    """Raised by _iter_batched when no chunk arrives within the applicable timeout."""


# ---------------------------------------------------------------------------
# Iteration helpers
# ---------------------------------------------------------------------------


def _iter_batched(
    iterable: Iterable[_T],
    idle_timeout: float | None = None,
    stream_timeout: float | None = None,
) -> Generator[list[_T], None, None]:
    """Drain *iterable* in a background thread; yield batches to the caller.

    The background thread does minimal Python per item (one queue.put).
    The calling thread sleeps for _BATCH_INTERVAL_S between drains, releasing
    the GIL for that entire interval so other threads run unimpeded.
    GIL acquisitions drop from O(items) to O(items / avg_batch_size).

    *idle_timeout*   — seconds to wait for the **first** chunk before giving up
                       and treating the connection as dead (triggers a retry).
    *stream_timeout* — seconds to wait between chunks **after** streaming has
                       started.  A gap here means the model stalled mid-generation;
                       the partial response is discarded and the call retried.
                       Defaults to None (no mid-stream timeout — wait indefinitely
                       once tokens are flowing).
    """
    _SENTINEL = object()
    q: queue.SimpleQueue = queue.SimpleQueue()

    exc_box: list[BaseException] = []

    def _drain() -> None:
        try:
            for item in iterable:
                q.put(item)
        except BaseException as e:
            exc_box.append(e)
        finally:
            q.put(_SENTINEL)

    threading.Thread(target=_drain, daemon=True).start()

    _last_item = time.monotonic()
    _streaming = False  # True once the first chunk has been received

    while True:
        # Pick the applicable timeout: pre-first-chunk uses idle_timeout (tight,
        # detects dead servers); post-first-chunk uses stream_timeout (loose or
        # None, tolerates model thinking gaps without discarding partial output).
        _current_timeout = stream_timeout if _streaming else idle_timeout
        try:
            if _current_timeout is not None:
                item = q.get(timeout=IDLE_CHECK_INTERVAL_S)
            else:
                item = q.get()
        except queue.Empty:
            if _current_timeout is not None and time.monotonic() - _last_item >= _current_timeout:
                label = "mid-stream" if _streaming else "pre-first-chunk"
                raise _LLMIdleTimeout(f"no chunk received for {_current_timeout:.0f}s ({label})")
            continue

        _last_item = time.monotonic()
        _streaming = True

        if item is _SENTINEL:
            if exc_box:
                raise exc_box[0]
            return

        # Sleep for one interval — background thread accumulates more items
        # while this thread holds no Python state (GIL fully released).
        time.sleep(_BATCH_INTERVAL_S)

        # Drain everything buffered during the sleep in one burst.
        batch: list[_T] = [item]
        while True:
            try:
                item = q.get_nowait()
                if item is _SENTINEL:
                    yield batch
                    return
                batch.append(item)
            except queue.Empty:
                break

        yield batch


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _strip_thinking(content: str) -> str:
    """Remove <think>…</think> / <thinking>…</thinking> blocks from model output."""
    return _THINKING_RE.sub("", content).strip()


def _extract_thinking(content: str) -> str:
    """Return the concatenated text of all thinking blocks, or empty string if none."""
    return "\n\n".join(m.group(1).strip() for m in _THINKING_RE.finditer(content))


def _looks_like_path(s: str) -> bool:
    """Return True if s looks like a sandbox path (file or directory)."""
    return bool(_PATH_RE.match(s.strip())) if isinstance(s, str) else False


def _camel_to_snake(key: str) -> str:
    """Normalize one dict key from camelCase/PascalCase to snake_case.

    Idempotent on keys that are already snake_case or single-word (no
    uppercase letters to act on). Used to tolerate LLMs that emit tool-call
    arguments in camelCase even though our tool schemas declare snake_case.
    """
    return _CAMEL_CASE_RE.sub("_", key).lower()


# ---------------------------------------------------------------------------
# Agent name helpers
# ---------------------------------------------------------------------------


def _b36_suffix(n: int, width: int = 4) -> str:
    """Encode *n* as a fixed-width base-36 string (0000…0009, 000a…)."""
    base = len(_B36)
    digits = []
    for _ in range(width):
        digits.append(_B36[n % base])
        n //= base
    return "".join(reversed(digits))


# sizeof(struct sockaddr_un.sun_path) on Linux -- a kernel ABI constant,
# NOT a filesystem limit (PATH_MAX is 4096), so a socket path can be a
# perfectly legal *file* path and still be unbindable as a *socket*. The
# usable length is one less: the path is NUL-terminated inside the array.
UDS_SUN_PATH_MAX = 108

# Records which process owns a per-run directory, so a later run can prove
# the owner is dead before removing it -- the directory-level counterpart to
# container.py's `agency.owner_pid` label. One per run directory (covering
# gateways/, scratch/, sandboxes/, config_homes/ together), not one per
# subdirectory.
_RUN_DIR_OWNER_FILE = "owner.pid"

_AGENCY_RUN_ID: "str | None" = None
_run_dir: "Path | None" = None
_run_dir_lock = threading.Lock()
_run_dir_reap_done = False
_gateway_dir = None


def pid_alive(pid: int) -> bool:
    """Return True if *pid* refers to a currently-running process on this host.

    The canonical copy: `sandbox/container.py` delegates here so
    the "is this owner still alive?" test behind every reaper in the
    framework has exactly one implementation to reason about.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists, just not signalable by us -- "exists" is the correct
        # answer either way, and refusing to reap is the safe direction.
        return True
    return True


def agency_run_id() -> str:
    """Stable per-process id used to namespace every run-scoped path this
    process owns -- this run's directory under `agency_tmp_dir()`
    (gateways/scratch/sandboxes/config_homes) and under `agency_runs_dir()`
    (logs/profiler), and (via `sandbox/container.py`'s own
    `_RUN_ID = agency_run_id()`) every container/image name. A uuid rather
    than a pid: pids are recycled, so a successive run could otherwise
    inherit a dead run's name and adopt its leftovers."""
    global _AGENCY_RUN_ID
    if _AGENCY_RUN_ID is None:
        import uuid

        _AGENCY_RUN_ID = f"r{uuid.uuid4().hex[:8]}"
    return _AGENCY_RUN_ID


_RUN_TS = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def agency_run_dir_name() -> str:
    """`{timestamp}_{run_id}` -- shared by agency_tmp_dir() and
    agency_runs_dir() so their per-run directories are identically named."""
    return f"{_RUN_TS}_{agency_run_id()}"


def agency_tmp_dir():
    """Root for local-filesystem-only run state (UDS sockets, squash
    scratch, chroot state, config-homes). Hardcoded to `/tmp/agency-{uid}`
    rather than `tempfile.gettempdir()`, which honours `$TMPDIR` -- fatal
    for a live socket if something sweeps it. The `-{uid}` suffix is
    per-user isolation (same as tmux's `/tmp/tmux-$UID`). Overridable via
    `AGENCY_TMP_ROOT`. Short by design: see `UDS_SUN_PATH_MAX`."""
    root = Path(os.environ.get("AGENCY_TMP_ROOT", f"/tmp/agency-{os.getuid()}"))
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    _ensure_owned_by_us(root)
    return root


def _ensure_owned_by_us(path: Path) -> None:
    """Raise a clear error if *path* isn't owned by this uid -- typically a
    container runtime's root daemon auto-creating a missing bind-mount
    source first. Diagnoses only; fixing ownership needs root."""
    owner_uid = path.stat().st_uid
    if owner_uid != os.getuid():
        raise RuntimeError(
            f"{path} exists but is owned by uid {owner_uid}, not this "
            f"process's uid {os.getuid()}. This commonly happens when a "
            "container runtime's root daemon auto-creates a missing "
            "bind-mount source directory before Agency's own code gets to "
            f"-- remove {path} as its owner (likely root) and retry; "
            "nothing here can fix ownership without root."
        )


def agency_runs_dir():
    """Root for run state meant to be found and inspected by a human
    (logs/DBs, profiler output, cross-run agent saves) -- plain files only,
    safe on any filesystem. Defaults to `./agency_runs`, resolved relative
    to CWD, so it's visible next to wherever the caller is working.
    Overridable via `AGENCY_RUNS_ROOT`."""
    root = Path(os.environ.get("AGENCY_RUNS_ROOT", "agency_runs")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _ensure_owned_by_us(root)
    return root


def agency_cache_root():
    """Root for cross-run state found without knowing which run created it
    (today: the harness binary cache). Under `~/.cache/agency` by default,
    not run-scoped like the other two roots. Overridable via
    `AGENCY_CACHE_ROOT`."""
    return Path(os.environ.get("AGENCY_CACHE_ROOT", str(Path.home() / ".cache" / "agency")))


# Default log directory -- shared by agent.py and orchestrator.py (both need
# it, and neither may import the other at module level), and moved out of
# agent.py for that reason.
_DEFAULT_LOG_DIR = agency_runs_dir() / agency_run_dir_name() / "logs"


def _agency_run_dir() -> Path:
    """This process's per-run directory under `agency_tmp_dir()` -- shared
    parent of `gw/`, `scratch/`, `sandboxes/`, `config_homes/`. Created
    (owner.pid stamped, orphans reaped) once, lazily, on first need."""
    global _run_dir
    if _run_dir is not None:
        return _run_dir
    with _run_dir_lock:
        if _run_dir is not None:
            return _run_dir
        d = agency_tmp_dir() / agency_run_dir_name()
        d.mkdir(parents=True, exist_ok=True)
        # Ownership as a pid (not baked into the dir name) so a later run
        # can prove this one is gone -- pids get recycled, uuids don't.
        (d / _RUN_DIR_OWNER_FILE).write_text(f"{os.getpid()}\n", encoding="utf-8")
        _run_dir = d
        _register_run_dir_cleanup(d)
        _reap_orphaned_run_dirs()
        return d


def agharness_llm_gateway_dir():
    """Fixed, well-known host directory a docker/podman-backed harness
    launch's Unix-domain-socket LLM gateway lives in -- `gw/` under
    this run's own directory (see `_agency_run_dir`). Named `gw`, not
    `gateways`, since it's the one subdirectory here spent from the
    108-byte `sun_path` socket budget (see `new_uds_path`). Shared between
    `agsandbox.py` (which bind-mounts this directory into every
    container-backed sandbox unconditionally -- cheap and harmless for a
    sandbox that never runs a harness, the same "attach unconditionally,
    gate on use" pattern already used for GPU passthrough flags) and
    `harness/agproxy_llm.py` (which places its UDS socket file
    inside it once a container-backed harness actually launches). Kept
    here, not in either of those two modules, specifically to avoid a
    layering dependency in either direction -- `agsandbox` sits below
    `agharness`/`agproxy_llm` in this codebase's intended import graph, so
    neither should import from the other just for this constant. A bind
    mount is a live view of the host directory, not a snapshot, so it's
    safe for the socket file to not exist yet at container-creation time
    and appear later once a harness actually launches.

    Scoped to one subdirectory per run, for the same three reasons container
    names are (`sandbox/container.py`'s `_RUN_ID`):

    * **Lifecycle.** This root is deliberately outside `$TMPDIR` and so is
      never externally cleaned; a flat directory shared by every run would
      accumulate sockets with no owner and no disposal point. A per-run
      directory makes cleanup a single atomic removal -- see
      `_register_run_dir_cleanup`.
    * **Isolation.** `agsandbox.py` bind-mounts this directory read-write
      into *every* container, so a flat directory would let any container
      read, connect to, and delete every concurrent run's sockets on the
      host. Mounting only the current run's directory removes that entirely
      while keeping the container-side path unchanged.
    * **Attribution.** A socket's owning run is readable from its path
      instead of having to be inferred from timestamps.

    Safe to mount per-run because a container never outlives the run that
    created it: container names embed their own per-process run id, and
    reuse/resume only ever applies to containers this same process created
    (see `_ContainerBackendBase._ensure_started`).
    """
    global _gateway_dir
    if _gateway_dir is not None:
        return _gateway_dir
    d = _agency_run_dir() / "gw"
    d.mkdir(parents=True, exist_ok=True)
    _gateway_dir = d
    return d


def agency_run_scratch_dir():
    """`scratch/` under this run's directory -- docker/podman squash
    working files. Cycle files are already uniquely named, so concurrent
    squashes safely share this directory."""
    d = _agency_run_dir() / "scratch"
    d.mkdir(parents=True, exist_ok=True)
    return d


def agency_config_homes_dir():
    """`config_homes/` under this run's directory -- parent for each
    harness launch's isolated config-home dir."""
    d = _agency_run_dir() / "config_homes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _register_run_dir_cleanup(own_dir) -> None:
    """Remove this run's entire per-run directory (gateways/, scratch/,
    sandboxes/, config_homes/ together) on normal exit -- the counterpart
    to `agsandbox.py`'s `_cleanup_all_sandboxes`, and best-effort in exactly
    the same way (a failure here must never take down an exiting process)."""
    import atexit
    import shutil

    def _cleanup() -> None:
        try:
            shutil.rmtree(own_dir, ignore_errors=True)
        except Exception as _e:  # pragma: no cover -- interpreter teardown
            print(f"[agutil] WARNING: run dir cleanup failed for {own_dir}: {_e}")

    atexit.register(_cleanup)


def _dir_has_live_listener(d) -> bool:
    """True if any socket under *d*'s `gw/` subdirectory still has
    something accepting on it.

    The safety check before a destructive sweep, mirroring the "no container
    still uses this image as its ancestor" confirmation
    `_reap_orphaned_lifecycle_images()` performs even after its label check
    says the owner is dead: an owner file can be stale or hand-copied, and
    the cost of being wrong is deleting a live run's bridge.
    """
    import socket as _socket

    gateways_dir = d / "gw"
    if not gateways_dir.is_dir():
        return False
    for sock in gateways_dir.glob("*.sock"):
        s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        try:
            s.settimeout(0.5)
            s.connect(str(sock))
            return True  # someone is accepting -- this run is alive
        except OSError:
            continue  # refused/stale/unreachable -- no listener behind it
        finally:
            s.close()
    return False


def _reap_orphaned_run_dirs() -> None:
    """Remove per-run directories left by SIGKILL'd runs, whose `atexit`
    cleanup never got to run.

    Deliberately keyed on proof the owner is gone (`pid_alive`) and never on
    age: a Unix socket's mtime never updates, so every long-lived socket
    looks arbitrarily stale to an age-based reaper -- which is exactly how an
    external `$TMPDIR` cleaner silently deleted live sockets out from under
    running servers, the failure this whole layout exists to prevent.

    Runs once per process, best-effort throughout, mirroring
    `container.py`'s `reap_orphaned_containers()`.
    """
    global _run_dir_reap_done
    if _run_dir_reap_done:
        return
    _run_dir_reap_done = True
    import shutil

    try:
        root = agency_tmp_dir()
        own_pid = os.getpid()
        for d in root.iterdir():
            if not d.is_dir() or d == _run_dir:
                continue
            owner = d / _RUN_DIR_OWNER_FILE
            try:
                owner_pid = int(owner.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                continue  # no readable owner record -- never guess, leave it
            if owner_pid == own_pid or pid_alive(owner_pid):
                continue
            if _dir_has_live_listener(d):
                continue  # stale owner record over a live run -- leave it
            print(
                f"[agutil] Reaping run dir {d.name!r}, orphaned by dead "
                f"process {owner_pid} (likely SIGKILL'd)",
                flush=True,
            )
            shutil.rmtree(d, ignore_errors=True)
    except Exception as _e:
        print(f"[agutil] WARNING: startup run dir reap failed: {_e}")


def new_uds_path(prefix: str) -> str:
    """A fresh socket path `<gateway dir>/<prefix>-<8 hex>.sock`, checked
    against the `sun_path` budget before anything tries to bind it.

    The id is 8 hex characters, not a full 32-character uuid4 hex: these
    names only need to be unique within one directory on one host, and the
    24 characters saved are the difference between fitting and not fitting
    for any caller whose paths are deeper than the default (a `log_dir`
    override, for instance). Validating here rather than at bind time turns
    the kernel's bare `OSError: AF_UNIX path too long`, raised several
    frames inside uvicorn with no mention of which path or what the limit
    is, into an error naming both.
    """
    import uuid

    path = str(agharness_llm_gateway_dir() / f"{prefix}-{uuid.uuid4().hex[:8]}.sock")
    if len(path) >= UDS_SUN_PATH_MAX:
        raise RuntimeError(
            f"Unix-domain socket path is {len(path)} bytes, which exceeds the "
            f"{UDS_SUN_PATH_MAX}-byte sockaddr_un.sun_path limit: {path!r}. "
            "This is a kernel ABI limit on socket paths, unrelated to PATH_MAX -- "
            "shorten the gateway directory (see agutil.agency_tmp_dir)."
        )
    return path


def uds_listener_is_live(path: "str | None", thread) -> bool:
    """True only if *path* still exists on disk AND *thread* is still
    serving it -- the two ways a UDS listener silently stops working while
    its owner still believes it is up.

    Either half failing leaves the same unrecoverable state, since the
    cached path keeps being handed out: an external cleanup can delete the
    socket file from under a live server (a Unix socket's mtime never
    updates, so every age-based reaper sees a long-lived one as stale), and
    a server thread that exits for any reason takes the socket file with it,
    because uvicorn unlinks it on shutdown."""
    import os

    if not path or not os.path.exists(path):
        return False
    return thread is not None and thread.is_alive()


def agharness_binary_cache_dir():
    """Fixed, well-known host directory holding a cached copy of each
    external harness binary (e.g. `claude`), bind-mounted read-only into
    every docker/podman-backed sandbox unconditionally -- same
    "attach unconditionally, gate on use" pattern as
    `agharness_llm_gateway_dir`. Exists because the sandbox's own base
    image (built for arbitrary agent tasks) has no reason to carry a
    ~250MB+ harness binary, and re-copying one into every fresh container
    on every launch would be slow and, for a network-isolated sandbox,
    impossible. Populated lazily, on the host, the first time a
    container-backed launch needs a binary this cache doesn't have yet
    (see `agharness_backends/claude_code.py`'s in-container binary
    resolution) -- never fetched from the network by Agency itself, only
    copied from whatever the host's own `shutil.which()` already resolves,
    so this never depends on knowing an install URL. Under `agency_cache_root()`
    rather than a run-scoped tempdir (unlike the gateway socket dir above):
    this should survive process restarts and successive runs so the ~250MB
    copy happens once per host, not once per Agency process lifetime."""
    d = agency_cache_root() / "harness_bin"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Fixed container-side mount point for agency_package_dir() below -- shared
# between agsandbox.py (which bind-mounts it) and any in-container
# entrypoint (agharness_backends/native.py's react-loop process, or a
# container-relocated agproxy_llm) that needs to know where to point
# PYTHONPATH to import agency.
AGENCY_PACKAGE_CONTAINER_MOUNT = "/opt/agency_pkg"

# Fixed container-side mount points for agharness_llm_gateway_dir() and this
# run's logs/ directory -- shared between agsandbox.py (which bind-mounts
# them) and harness_daemon_launcher.py (which points the in-container
# daemon at them), so the literal is defined exactly once instead of
# hardcoded independently in both files.
AGENCY_LLM_GATEWAY_CONTAINER_MOUNT = "/var/run/agency_llm_gateway"
AGENCY_LOGS_CONTAINER_MOUNT = "/var/run/agency_logs"


def agency_package_dir():
    """Host directory containing the `agency` package currently running in
    *this* process -- the parent of `agency/__init__.py`'s own directory,
    i.e. what needs to be on `PYTHONPATH` for `import agency` to resolve.
    Bind-mounted read-only into every container-backed sandbox at
    `AGENCY_PACKAGE_CONTAINER_MOUNT`, same "attach unconditionally, gate on
    use" pattern as `agharness_llm_gateway_dir`/`agharness_binary_cache_dir`
    above -- so an in-container entrypoint always runs the EXACT same code
    the host process is running, not a second, potentially-stale copy
    baked into the sandbox's base image.

    Unlike those two, this is not a fixed scratch location -- it's resolved
    dynamically from `agency.__file__`, since it has to be wherever *this*
    process's own code actually lives (a dev checkout, an editable install,
    a site-packages install are all valid; none should be hardcoded)."""
    import agency as _agency_pkg
    from pathlib import Path

    return Path(_agency_pkg.__file__).resolve().parent.parent


def ensure_python_packages_in_container(sandbox, packages, *, timeout_s: int = 180) -> None:
    """Ensure each of `packages` (import names, e.g. `"fastapi"`) is
    importable inside `sandbox`'s container, installing any that are
    missing via `pip3 install`. Confirmed real gap: `agency-sandbox:latest`
    carries `httpx`/`pydantic` but not `fastapi`/`uvicorn`/`openai` --
    needed by both a container-relocated `agproxy_llm` and a future
    full react-loop entrypoint that imports `agency` itself.

    Checks each package's actual importability first, not just its
    presence in `pip list` (a package can be listed but broken, or absent
    but shadowed by something else on the path) -- and only invokes pip for
    the ones genuinely missing, so a container whose checkpoint image
    already has everything installed (reused across skill calls, see
    agskill.py's commit() boundary) pays this cost exactly once per fresh
    container, not on every launch. Requires the container to have
    outbound network access -- true today (see docs/Design_harness_
    integration.md's network lockdown discussion, deferred).

    Raises RuntimeError if pip itself fails (e.g. no network, a genuinely
    broken package name) -- this is a real prerequisite-provisioning
    failure, not something to silently swallow."""
    import shlex

    missing = [
        pkg for pkg in packages if sandbox.exec(f'python3 -c "import {pkg}"', timeout=30)[1] != 0
    ]
    if not missing:
        return

    install_cmd = "pip3 install --quiet " + " ".join(shlex.quote(p) for p in missing)
    out, rc = sandbox.exec(install_cmd, timeout=timeout_s)
    if rc != 0:
        raise RuntimeError(f"failed to install {missing} inside container: {out}")


# ---------------------------------------------------------------------------
# Host GPU/CPU/memory detection -- shared by agResourcePool (orchestrator/
# agresources.py) and the sandbox backends (container.py/chroot.py, for
# device-node mounting), which is exactly why this lives here rather than
# alongside agResourcePool itself: sandbox is a low-level module that
# agResourcePool's owner (the orchestrator) sits above, so these can't live
# in the orchestrator package without creating an import cycle.
# ---------------------------------------------------------------------------


def _visible_device_remap(env_names: "tuple[str, ...]") -> dict[int, int]:
    """Build a physical-id -> remapped-index map from whichever of env_names
    is set (e.g. CUDA_VISIBLE_DEVICES="0,3,5,7" -> {0:0, 3:1, 5:2, 7:3}),
    mirroring how the driver itself remaps physical GPUs to indices 0..N-1
    inside a process that only sees a restricted device list."""
    for name in env_names:
        val = os.environ.get(name, "")
        if not val or val.lower() in ("nodevfiles", "none"):
            continue
        try:
            ids = [int(x.strip()) for x in val.split(",") if x.strip().lstrip("-").isdigit()]
            if ids:
                return {phys: idx for idx, phys in enumerate(ids)}
        except Exception as _e:
            # DATACOLLECTOR: append -- process-level (pool is shared, not per-agent).
            print(f"[agutil] WARNING: could not parse {name}={val!r}: {_e}")
    return {}


def _allocate_gpu_markers_cuda(gpu_ids: list[int], marker_bytes: int) -> bool:
    """Try the CUDA driver API. Returns True if libcuda was found at all
    (regardless of whether individual per-GPU allocations went on to
    succeed) so the caller knows not to also try the ROCm/HIP path -- a
    cuInit failure means a broken/inaccessible NVIDIA driver, not "try AMD
    instead," since there's no AMD hardware to fall back to on an NVIDIA
    host anyway. Returns False only when libcuda.so.1 isn't present at all."""
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return False
    if cuda.cuInit(0) != 0:
        return True

    # When CUDA_VISIBLE_DEVICES is set (e.g. "0,3,5,7"), the CUDA driver
    # remaps physical GPUs to indices 0..N-1.  gpu_ids are physical IDs, so
    # we must convert to the remapped index before calling CUDA APIs.
    cuda_index = _visible_device_remap(("CUDA_VISIBLE_DEVICES",))

    for gpu_id in gpu_ids:
        try:
            dev = cuda_index.get(gpu_id, gpu_id)
            context = ctypes.c_void_p()
            ptr = ctypes.c_void_p()
            if cuda.cuCtxCreate_v2(ctypes.byref(context), 0, dev) != 0:
                continue
            cuda.cuMemAlloc_v2(ctypes.byref(ptr), marker_bytes)
            # Leave context current; allocation persists for the process lifetime.
        except Exception as _e:
            # DATACOLLECTOR: append -- process-level (pool is shared, not per-agent).
            print(f"[agutil] WARNING: CUDA marker allocation failed for GPU {gpu_id}: {_e}")
    return True


def _allocate_gpu_markers_rocm(gpu_ids: list[int], marker_bytes: int) -> None:
    """ROCm/HIP equivalent of _allocate_gpu_markers_cuda. HIP's runtime API
    manages a context implicitly per device (hipSetDevice + hipMalloc)
    rather than CUDA driver API's explicit per-device context object, so
    there's no analogue of cuCtxCreate to call here."""
    try:
        hip = ctypes.CDLL("libamdhip64.so")
    except OSError:
        return
    if hip.hipInit(0) != 0:
        return

    hip_index = _visible_device_remap(("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))

    for gpu_id in gpu_ids:
        try:
            dev = hip_index.get(gpu_id, gpu_id)
            if hip.hipSetDevice(dev) != 0:
                continue
            ptr = ctypes.c_void_p()
            hip.hipMalloc(ctypes.byref(ptr), marker_bytes)
            # Leave allocated; persists for the process lifetime, same as the
            # CUDA path above.
        except Exception as _e:
            # DATACOLLECTOR: append -- process-level (pool is shared, not per-agent).
            print(f"[agutil] WARNING: ROCm marker allocation failed for GPU {gpu_id}: {_e}")


def _allocate_gpu_markers(gpu_ids: list[int]) -> None:
    """Allocate marker_mb of VRAM on each GPU directly in the calling process.

    Tries the CUDA driver API first, falling back to ROCm/HIP — no torch
    dependency required either way. Allocations live for the process
    lifetime, which is fine: the memory is tiny (128 MB per GPU by default)
    and there is no need to release it mid-run. Runs silently if neither
    CUDA nor ROCm is available.
    """
    marker_bytes = MARKER_MB * 1024 * 1024
    if _allocate_gpu_markers_cuda(gpu_ids, marker_bytes):
        return
    _allocate_gpu_markers_rocm(gpu_ids, marker_bytes)


def _cvd_filter(gpu_ids: list[int]) -> list[int]:
    """Filter gpu_ids to the subset allowed by CUDA_VISIBLE_DEVICES,
    HIP_VISIBLE_DEVICES, or ROCR_VISIBLE_DEVICES (whichever is set; checked
    in that order so a CUDA restriction always wins if somehow more than one
    is set at once)."""
    for _env in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        cvd = os.environ.get(_env, "")
        if not cvd or cvd.lower() in ("nodevfiles", "none"):
            continue
        try:
            allowed = {int(x.strip()) for x in cvd.split(",") if x.strip().lstrip("-").isdigit()}
            if allowed:
                return [g for g in gpu_ids if g in allowed]
        except Exception as _e:
            # DATACOLLECTOR: append -- process-level (pool is shared, not per-agent).
            print(f"[agutil] WARNING: could not parse {_env}={cvd!r}: {_e}")
    return gpu_ids


def detect_gpus() -> list[int]:
    """Return GPU IDs visible to nvidia-smi or rocm-smi, filtered by CUDA_VISIBLE_DEVICES."""
    _timeout = GPU_DETECT_TIMEOUT_S
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=_timeout,
        )
        if result.returncode == 0 and result.stdout.strip():
            ids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
            return _cvd_filter(ids)
    except Exception as _e:
        # Expected on any host without an NVIDIA driver/nvidia-smi installed --
        # falls through to the ROCm probe below.
        # DATACOLLECTOR: append -- process-level, routine expected fallback, low priority.
        print(f"[agutil] nvidia-smi probe failed, trying rocm-smi: {_e}")
    try:
        result = subprocess.run(
            ["rocm-smi", "--showid", "--csv"],
            capture_output=True,
            text=True,
            timeout=_timeout,
        )
        if result.returncode == 0 and result.stdout.strip():
            ids = []
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line or line.lower().startswith("device"):
                    continue
                # First column is "cardN" — extract the numeric suffix.
                col = line.split(",")[0].strip().lower()
                if col.startswith("card"):
                    try:
                        ids.append(int(col[4:]))
                    except ValueError:
                        pass
                else:
                    try:
                        ids.append(int(col))
                    except ValueError:
                        pass
            if ids:
                return _cvd_filter(ids)
    except Exception as _e:
        # Expected on any host without an AMD driver/rocm-smi installed.
        # DATACOLLECTOR: append -- process-level, routine (end of fallback chain), low priority.
        print(f"[agutil] rocm-smi probe failed, no GPUs detected: {_e}")
    return []


# Matches the trailing PCI bus address segment of a resolved sysfs device
# path (e.g. ".../0000:75:00.0" -> "0000:75:00.0").
_PCI_BUS_RE = re.compile(r"([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f])$")


def _amd_render_node_pci_bus(name: str) -> "str | None":
    """Resolve the real PCI bus address (e.g. "0000:75:00.0") backing a
    /dev/dri render node by name (e.g. "renderD128"), by following
    /sys/class/drm/<name>/device. Returns None for XCD/compute-partition
    sibling nodes: MI300/MI350-class GPUs expose one render node per
    accelerator-complex-die under a "amdgpu_xcp_N" platform device even
    while the GPU itself is in unpartitioned (SPX) mode, and only the one
    node with a real PCI parent maps 1:1 to a physical GPU."""
    target = os.path.realpath(f"/sys/class/drm/{name}/device")
    match = _PCI_BUS_RE.search(target)
    return match.group(1).lower() if match else None


def amd_render_node_paths_by_pci_bus(candidates: "list[str]") -> "list[str] | None":
    """Reorder *candidates* (absolute /dev/dri/renderD* paths) so index N is
    the render node for rocm-smi's GPU N, by cross-referencing `rocm-smi
    --showbus` (GPU index -> PCI bus) against each candidate's own resolved
    PCI bus -- NOT by assuming sorted order already matches GPU index.

    Confirmed necessary on real 8x MI350X hardware: each GPU there exposes
    itself plus 7 XCD/compute-partition sibling render nodes (64 nodes total
    for 8 GPUs), and even the primary node's number doesn't sort in the same
    order as rocm-smi's GPU index -- e.g. GPU 3's real node was the
    numerically LOWEST of the 64 present, not the 4th. Naively picking
    sorted-index N (the old behavior) scoped several GPU IDs to nodes
    belonging to a different physical GPU entirely.

    Returns None (caller should fall back to naive sorted order) if
    `rocm-smi --showbus` fails/is unavailable, or any GPU's bus can't be
    matched to exactly one candidate -- a partial mapping is never applied,
    since trusting it for some GPUs and not others would be worse than the
    naive fallback it's meant to replace.
    """
    try:
        result = subprocess.run(
            ["rocm-smi", "--showbus", "--csv"],
            capture_output=True,
            text=True,
            timeout=GPU_DETECT_TIMEOUT_S,
        )
    except Exception:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None

    bus_by_gpu_id: "dict[int, str]" = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("device"):
            continue
        parts = [c.strip() for c in line.split(",")]
        if len(parts) < 2 or not parts[0].lower().startswith("card"):
            continue
        try:
            gpu_id = int(parts[0][4:])
        except ValueError:
            continue
        bus_by_gpu_id[gpu_id] = parts[1].lower()
    if not bus_by_gpu_id:
        return None

    render_by_bus: "dict[str, str]" = {}
    for path in candidates:
        bus = _amd_render_node_pci_bus(os.path.basename(path))
        if bus is not None:
            render_by_bus.setdefault(bus, path)

    ordered = []
    for gpu_id in range(max(bus_by_gpu_id) + 1):
        bus = bus_by_gpu_id.get(gpu_id)
        path = render_by_bus.get(bus) if bus is not None else None
        if path is None:
            return None
        ordered.append(path)
    return ordered


def detect_cpus() -> int:
    """Return the number of logical CPU cores on this host."""
    return os.cpu_count() or 1


def detect_memory_mb() -> int:
    """Return total host RAM in MB, read from /proc/meminfo (Linux) or via sysctl (macOS)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024  # kB → MB
    except Exception as _e:
        # Expected on non-Linux hosts (e.g. macOS has no /proc) -- falls
        # through to the sysctl probe below.
        # DATACOLLECTOR: append -- process-level, routine expected fallback (non-Linux host), low priority.
        print(f"[agutil] /proc/meminfo read failed, trying sysctl: {_e}")
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=SYSCTL_DETECT_TIMEOUT_S,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) // (1024 * 1024)
    except Exception as _e:
        # DATACOLLECTOR: append -- process-level; both probes failed, worth real visibility.
        print(f"[agutil] sysctl memory probe failed, using configured fallback: {_e}")
    return MEMORY_DETECT_FALLBACK_MB
