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
    """Install a SIGTERM handler for this ``with`` block that converts a
    plain kill into ``SystemExit``, so atexit/cleanup hooks run instead of
    the OS's immediate termination. Main-thread only (a no-op elsewhere).
    Yields an Event set if SIGTERM was actually received; *label* names the
    process in the printed shutdown message."""
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
    """Normalize camelCase/PascalCase to snake_case; idempotent on inputs
    already snake_case."""
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
    """Stable per-process id namespacing every run-scoped path (tmp dirs,
    run dirs, container/image names). A uuid, not a pid, since pids are
    recycled and could otherwise adopt a dead run's leftovers."""
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
    """Root for local-filesystem-only run state (sockets, squash scratch,
    chroot state, config-homes). Hardcoded to `/tmp/agency-{uid}`, not
    `tempfile.gettempdir()`, since a `$TMPDIR` sweep would kill a live
    socket. Overridable via `AGENCY_TMP_ROOT`; kept short for
    `UDS_SUN_PATH_MAX`."""
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
    """Per-run `gw/` directory (under `_agency_run_dir`) holding this run's
    harness LLM-gateway UDS socket. Bind-mounted into every container-backed
    sandbox by agsandbox.py; the harness backend places its socket file
    here once launched. Scoped per-run so cleanup, isolation between
    concurrent runs, and attribution all fall out of the directory
    boundary."""
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
    """Fresh socket path `<gateway dir>/<prefix>-<8 hex>.sock`,
    length-checked against the `sun_path` budget before anything tries to
    bind it, so a too-long path fails here with a clear message instead of
    deep inside uvicorn as a bare, unlabeled `OSError`."""
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
    alive -- either can fail independently while the cached path keeps
    being handed out (external cleanup can delete the socket; the server
    thread unlinks it on exit)."""
    import os

    if not path or not os.path.exists(path):
        return False
    return thread is not None and thread.is_alive()


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
    """Host directory containing the `agency` package this process is
    running -- what must be on `PYTHONPATH` for `import agency` to
    resolve. Bind-mounted read-only into every container so it runs the
    exact same code, never a stale copy."""
    import agency as _agency_pkg
    from pathlib import Path

    return Path(_agency_pkg.__file__).resolve().parent.parent


def _container_can_reach_pypi(sandbox, timeout_s: int = 30) -> bool:
    """Whether the container has outbound network to a package index."""
    probe = (
        'python3 -c "import socket; socket.setdefaulttimeout(5); '
        "socket.create_connection(('pypi.org', 443))\""
    )
    try:
        return sandbox.exec(probe, timeout=timeout_s)[1] == 0
    except Exception:
        return False


def ensure_python_packages_in_container(sandbox, packages, *, timeout_s: int = 180) -> None:
    """Ensure each of `packages` (import names) is importable inside
    `sandbox`'s container, `pip3 install`-ing any missing after checking
    real importability (not just `pip list` presence). Raises RuntimeError
    if pip itself fails."""
    import shlex

    missing = [
        pkg for pkg in packages if sandbox.exec(f'python3 -c "import {pkg}"', timeout=30)[1] != 0
    ]
    if not missing:
        return

    if not _container_can_reach_pypi(sandbox):
        raise RuntimeError(
            f"cannot install {missing} inside the container: no outbound network "
            "(probed pypi.org:443). Either give the sandbox network access, or bake "
            "these packages into the base image. A sandbox started with "
            "sandboxconfig.flags=['--network', 'none'] is the usual cause; a private "
            "package index on an isolated network would also probe as unreachable."
        )

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
    """Allocate marker_mb of VRAM per GPU in this process (CUDA first,
    falling back to ROCm/HIP). Lives for the process lifetime; silent
    no-op if neither driver is available."""
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
    """Resolve the real PCI bus address backing a /dev/dri render node by
    name, via /sys/class/drm/<name>/device. Returns None for XCD/compute-
    partition sibling nodes (only the node with a real PCI parent maps 1:1
    to a physical GPU)."""
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
    sorted-index N scopes several GPU IDs to nodes belonging to a
    different physical GPU entirely.

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
