from __future__ import annotations
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
