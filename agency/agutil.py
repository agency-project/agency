from __future__ import annotations
import os
import queue
import re
import signal
import threading
import time
import traceback as _traceback
from contextlib import contextmanager
from typing import Generator, Iterable, TypeVar

from .agconfig import GlobalConfigParam, _AgConfigViewBase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_T = TypeVar("_T")
# Kept as a plain module attribute (not a ConfigParam) -- tests/conftest.py
# monkeypatches this by name (`monkeypatch.setattr(_agutil_module,
# "_BATCH_INTERVAL_S", 0.0)`) to speed up streaming tests; a descriptor would
# silently break that.
_BATCH_INTERVAL_S: float = 0.1  # main thread drains stream every 100 ms


# Exists only to register agutil's config fields (via __set_name__ at import
# time). Tier 1 (global): _iter_batched is a free function with no agconfig
# threaded through it, so — like _AgResourcePoolFields -- reads use a
# throwaway instance and GlobalConfigParam ignores it anyway, always routing
# to agConfig.GLOBAL.
class _AgUtilFields:
    idle_check_interval_s = GlobalConfigParam(
        "agutil", default=1.0
    )  # how often to check idle timeout

    def __init__(self, agconfig=None) -> None:
        self._agconfig = agconfig


class agUtilConfig(_AgConfigViewBase):
    """View over an agConfig for pre-setting agutil tunables in one call::

        cfg = agConfig(agUtilConfig(idle_check_interval_s=0.5))

    See `_AgConfigViewBase` in agconfig.py for the shared mechanics.
    """

    _OWNER = "agutil"


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
    the framework's own self-healing (e.g. ``agsandbox_backends.container``'s
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
                item = q.get(timeout=_AgUtilFields().idle_check_interval_s)
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

# Records which process owns a gateway directory, so a later run can prove
# the owner is dead before removing it -- the directory-level counterpart to
# container.py's `agency.owner_pid` label.
_GATEWAY_OWNER_FILE = "owner.pid"

_AGENCY_RUN_ID: "str | None" = None
_gateway_dir = None
_gateway_lock = threading.Lock()
_gateway_reap_done = False


def pid_alive(pid: int) -> bool:
    """Return True if *pid* refers to a currently-running process on this host.

    The canonical copy: `agsandbox_backends/container.py` delegates here so
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
    """Stable per-process id used to namespace this run's host-side runtime
    state. A uuid rather than a pid, matching `container.py`'s `_RUN_ID`:
    pids are recycled, so a successive run could otherwise inherit a dead
    run's name and adopt its leftovers."""
    global _AGENCY_RUN_ID
    if _AGENCY_RUN_ID is None:
        import uuid

        _AGENCY_RUN_ID = f"r{uuid.uuid4().hex[:8]}"
    return _AGENCY_RUN_ID


def agency_tmp_root():
    """Root of every host-side runtime path agency owns: run logs
    (`agent._DEFAULT_LOG_DIR`) and the UDS gateway (`agharness_llm_gateway_dir`)
    both live under here.

    Deliberately hardcoded to `/tmp/agency-{uid}` rather than derived from
    `tempfile.gettempdir()`: `gettempdir()` honours `$TMPDIR`, which on a
    shared host is routinely redirected to scratch space under an aggressive
    cleanup policy. Files there are *supposed* to be deletable, which is
    survivable for a scratch file and fatal for a live socket -- a reaped
    socket leaves the server advertising a path that no longer exists, and
    every request across the bridge then fails with a bare ENOENT (observed
    in practice: a `$TMPDIR` on a full shared volume being swept every few
    minutes, taking live sockets with it). Logs were already immune only
    because `_DEFAULT_LOG_DIR` happened to hardcode `/tmp` while this
    function honoured `$TMPDIR`; both now share one root, so one policy
    covers both.

    Namespaced by uid (not username -- always short, and it's already the
    actual permission boundary) so two OS users on the same shared host don't
    collide: a bare `/tmp/agency` is created and owned by whichever user
    happens to run agency first, `PermissionError`-ing every other user on
    every later run. Overridable via `AGENCY_TMP_ROOT` for anyone who wants a
    different location.

    A short root matters for a second reason -- see `UDS_SUN_PATH_MAX` and
    `new_uds_path`: every character here is spent from a 108-byte budget.
    """
    from pathlib import Path

    override = os.environ.get("AGENCY_TMP_ROOT")
    if override:
        return Path(override)
    return Path(f"/tmp/agency-{os.getuid()}")


def agharness_llm_gateway_dir():
    """Fixed, well-known host directory a docker/podman-backed harness
    launch's Unix-domain-socket LLM gateway lives in. Shared between
    `agsandbox.py` (which bind-mounts this directory into every
    container-backed sandbox unconditionally -- cheap and harmless for a
    sandbox that never runs a harness, the same "attach unconditionally,
    gate on use" pattern already used for GPU passthrough flags) and
    `agharness_internal/agproxy_llm.py` (which places its UDS socket file
    inside it once a container-backed harness actually launches). Kept
    here, not in either of those two modules, specifically to avoid a
    layering dependency in either direction -- `agsandbox` sits below
    `agharness`/`agproxy_llm` in this codebase's intended import graph, so
    neither should import from the other just for this constant. A bind
    mount is a live view of the host directory, not a snapshot, so it's
    safe for the socket file to not exist yet at container-creation time
    and appear later once a harness actually launches.

    Named `gw` rather than `agency_llm_gateway` purely for path length: the
    directory name is spent from every socket's 108-byte `sun_path` budget
    (see `new_uds_path`), and the container side of the bind mount keeps the
    long, self-describing name (`/var/run/agency_llm_gateway`) where no such
    budget applies.

    Scoped to one subdirectory per run, for the same three reasons container
    names are (`agsandbox_backends/container.py`'s `_RUN_ID`):

    * **Lifecycle.** This root is deliberately outside `$TMPDIR` and so is
      never externally cleaned; a flat directory shared by every run would
      accumulate sockets with no owner and no disposal point. A per-run
      directory makes cleanup a single atomic removal -- see
      `_register_gateway_cleanup`.
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
    with _gateway_lock:
        if _gateway_dir is not None:
            return _gateway_dir
        d = agency_tmp_root() / "gw" / agency_run_id()
        d.mkdir(parents=True, exist_ok=True)
        # The directory-level equivalent of a container's `agency.owner_pid`
        # label: ownership is recorded as a pid so a later run can *prove*
        # this run is gone before deleting anything, while the directory NAME
        # stays a uuid (a pid could be recycled by an unrelated process --
        # the same split container.py's _RUN_ID comment describes).
        (d / _GATEWAY_OWNER_FILE).write_text(f"{os.getpid()}\n", encoding="utf-8")
        _gateway_dir = d
        _register_gateway_cleanup(d)
        _reap_orphaned_gateway_dirs()
        return d


def _register_gateway_cleanup(own_dir) -> None:
    """Remove this run's socket directory on normal exit -- the counterpart
    to `agsandbox.py`'s `_cleanup_all_sandboxes`, and best-effort in exactly
    the same way (a failure here must never take down an exiting process)."""
    import atexit
    import shutil

    def _cleanup() -> None:
        try:
            shutil.rmtree(own_dir, ignore_errors=True)
        except Exception as _e:  # pragma: no cover -- interpreter teardown
            print(f"[agutil] WARNING: gateway cleanup failed for {own_dir}: {_e}")

    atexit.register(_cleanup)


def _dir_has_live_listener(d) -> bool:
    """True if any socket in *d* still has something accepting on it.

    The safety check before a destructive sweep, mirroring the "no container
    still uses this image as its ancestor" confirmation
    `_reap_orphaned_lifecycle_images()` performs even after its label check
    says the owner is dead: an owner file can be stale or hand-copied, and
    the cost of being wrong is deleting a live run's bridge.
    """
    import socket as _socket

    for sock in d.glob("*.sock"):
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


def _reap_orphaned_gateway_dirs() -> None:
    """Remove gateway directories left by SIGKILL'd runs, whose `atexit`
    cleanup never got to run.

    Deliberately keyed on proof the owner is gone (`pid_alive`) and never on
    age: a Unix socket's mtime never updates, so every long-lived socket
    looks arbitrarily stale to an age-based reaper -- which is exactly how an
    external `$TMPDIR` cleaner silently deleted live sockets out from under
    running servers, the failure this whole layout exists to prevent.

    Runs once per process, best-effort throughout, mirroring
    `container.py`'s `reap_orphaned_containers()`.
    """
    global _gateway_reap_done
    if _gateway_reap_done:
        return
    _gateway_reap_done = True
    import shutil

    try:
        root = agency_tmp_root() / "gw"
        own_pid = os.getpid()
        for d in root.iterdir():
            if not d.is_dir() or d == _gateway_dir:
                continue
            owner = d / _GATEWAY_OWNER_FILE
            try:
                owner_pid = int(owner.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                continue  # no readable owner record -- never guess, leave it
            if owner_pid == own_pid or pid_alive(owner_pid):
                continue
            if _dir_has_live_listener(d):
                continue  # stale owner record over a live run -- leave it
            print(
                f"[agutil] Reaping gateway dir {d.name!r}, orphaned by dead "
                f"process {owner_pid} (likely SIGKILL'd)",
                flush=True,
            )
            shutil.rmtree(d, ignore_errors=True)
    except Exception as _e:
        print(f"[agutil] WARNING: startup gateway reap failed: {_e}")


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
            "shorten the gateway directory (see agutil.agency_tmp_root)."
        )
    return path


def reserve_uds_path(current: "str | None", prefix: str) -> str:
    """The path a service should bind: its previously-reserved one if it has
    one, otherwise a fresh path from `new_uds_path`.

    Reusing the path across restarts is what makes recovery possible at all.
    A container is told its bridge's socket name once, at launch, and neither
    docker nor podman can change a running container's mounts -- so a
    restarted server that minted a *new* random name would be invisible to
    every container already pointed at the old one. Any stale file left at
    the reserved path is removed first, since binding onto an existing socket
    file fails outright.
    """
    if current:
        from pathlib import Path

        Path(current).unlink(missing_ok=True)
        return current
    return new_uds_path(prefix)


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
    so this never depends on knowing an install URL. Under the user's home
    directory rather than a tempdir (unlike the gateway socket dir above):
    this should survive process restarts so the ~250MB copy happens once
    per host, not once per Agency process lifetime."""
    from pathlib import Path

    d = Path.home() / ".cache" / "agency_harness_bin"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Fixed container-side mount point for agency_package_dir() below -- shared
# between agsandbox.py (which bind-mounts it) and any in-container
# entrypoint (agharness_backends/native.py's react-loop process, or a
# container-relocated agproxy_llm) that needs to know where to point
# PYTHONPATH to import agency.
AGENCY_PACKAGE_CONTAINER_MOUNT = "/opt/agency_pkg"


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
