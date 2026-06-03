"""agterm — color-coded real-time terminal logging for agent events.

Each agent is automatically assigned a unique ANSI color.  All output goes to
stderr so it doesn't interfere with structured stdout output.

Usage:
    agterm.enabled = False   # silence all terminal output
    agterm.enabled = True    # (default) re-enable

Output format:
    HH:MM:SS.mmm  [uuid8]  [EVENT   ]  message          (file.py:lineno)
"""
from __future__ import annotations
import sys
import inspect
import threading
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Colour palette — 8 visually distinct ANSI foreground colours for agents
# ---------------------------------------------------------------------------
_AGENT_COLORS = [
    "\033[97m",   # bright white
    "\033[96m",   # bright cyan
    "\033[93m",   # bright yellow
    "\033[92m",   # bright green
    "\033[95m",   # bright magenta
    "\033[94m",   # bright blue
    "\033[91m",   # bright red
    "\033[33m",   # orange/dark yellow
]

# Fixed greyscale styles for event tags — colours are reserved for agent IDs
_EVENT_STYLES: dict[str, str] = {
    "CREATED  ": "\033[1m",    # bold
    "FORKED   ": "\033[1m",    # bold
    "DESTROYED": "\033[2m",    # dim
    "SKILL ▶  ": "\033[1m",    # bold
    "SKILL ✓  ": "\033[0m",    # normal
    "SKILL ✗  ": "\033[7m",    # reverse video  (visible without colour)
    "LLM      ": "\033[0m",    # normal
    "TOOL     ": "\033[0m",    # normal
}

_RESET = "\033[0m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"


class agterm:
    """Per-agent colour-coded terminal event logger.

    Class-level settings:
        agterm.enabled = False   silence all output
    """
    enabled: bool = True
    _lock: threading.Lock = threading.Lock()
    _color_counter: int = 0
    _agname_colors: dict[str, str] = {}  # agname → ANSI color

    def __init__(self, agname: str) -> None:
        with agterm._lock:
            idx = agterm._color_counter % len(_AGENT_COLORS)
            agterm._color_counter += 1
        self._color = _AGENT_COLORS[idx]
        self._id = agname
        agterm._agname_colors[self._id] = self._color  # register for cross-agent colorization

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _colorize_agnames(msg: str) -> str:
        """Wrap every registered agent UUID appearing in msg with its color."""
        for uid, color in agterm._agname_colors.items():
            if uid in msg:
                msg = msg.replace(uid, f"{color}{_BOLD}{uid}{_RESET}")
        return msg

    # ------------------------------------------------------------------
    # Public log method
    # ------------------------------------------------------------------

    def log(self, event: str, msg: str, depth: int = 1) -> None:
        """Emit one log line.

        depth=1  → source location is the direct caller of log()
        depth=2  → source location is the caller's caller
        """
        if not agterm.enabled:
            return
        frame = inspect.currentframe()
        for _ in range(depth):
            if frame is not None:
                frame = frame.f_back
        filename = (frame.f_code.co_filename.rsplit("/", 1)[-1]
                    if frame is not None else "?")
        lineno   = frame.f_lineno if frame is not None else 0

        ts        = f"{_DIM}{datetime.now(timezone.utc).strftime('%H:%M:%S.%f')[:-3]}{_RESET}"
        agent_tag = f"{self._color}{_BOLD}[{self._id}]{_RESET}"
        ev_key    = event.ljust(9)[:9]
        ev_style  = _EVENT_STYLES.get(ev_key, "")
        ev_tag    = f"{ev_style}[{ev_key}]{_RESET}"
        src       = f"{_DIM}({filename}:{lineno}){_RESET}"

        line = f"{ts}  {agent_tag}  {ev_tag}  {agterm._colorize_agnames(msg)}  {src}"
        with agterm._lock:
            try:
                from . import agui as _agui
                if _agui._active is not None:
                    _agui._active.add_log(line)
                    return
            except Exception:
                pass
            print(line, file=sys.stderr, flush=True)
