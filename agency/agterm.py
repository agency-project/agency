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
import random as _random
import sys
import inspect
import threading
from datetime import datetime

# ---------------------------------------------------------------------------
# Colour palette — ~54 visually distinct xterm-256 colours, randomised once
# per process so consecutive agents get varied assignments.
#
# Strategy: sample the 6×6×6 cube at component levels {0, 2, 4, 5}
# (values 0 / 135 / 215 / 255).  That yields 4³ = 64 candidates; after
# removing the 4 cube-greys and the 6 near-black entries (max level ≤ 2)
# we get 54 clearly visible, well-spread colours.
# ---------------------------------------------------------------------------
def _make_color_palette() -> list[str]:
    _LEVELS = (0, 2, 4, 5)
    indices: list[int] = []
    for r in _LEVELS:
        for g in _LEVELS:
            for b in _LEVELS:
                if r == g == b:       # cube grey diagonal
                    continue
                if max(r, g, b) <= 2: # near-black (brightest component ≤ 135)
                    continue
                indices.append(16 + 36 * r + 6 * g + b)
    colors = [f"\033[38;5;{idx}m" for idx in indices]
    _random.shuffle(colors)
    return colors


_AGENT_COLORS: list[str] = _make_color_palette()

# Fixed greyscale styles for event tags — colours are reserved for agent IDs
_EVENT_STYLES: dict[str, str] = {
    "CREATED  ": "\033[1m",    # bold
    "FORKED   ": "\033[1m",    # bold
    "DESTROYED": "\033[2m",    # dim
    "SKILL ▶  ": "\033[1m",    # bold
    "SKILL ✓  ": "\033[0m",    # normal
    "SKILL ✗  ": "\033[7m",    # reverse video  (visible without colour)
    "LLM ▶    ": "\033[0m",    # normal
    "LLM ✓    ": "\033[0m",    # normal
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
        self._tokens: int | None = None
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

        ts        = f"{_DIM}{datetime.now().strftime('%H:%M:%S.%f')[:-3]}{_RESET}"
        agent_tag = f"{self._color}{_BOLD}[{self._id}]{_RESET}"
        ev_key    = event.ljust(9)[:9]
        ev_style  = _EVENT_STYLES.get(ev_key, "")
        ev_tag    = f"{ev_style}[{ev_key}]{_RESET}"
        tok_tag   = f"  {_DIM}({self._tokens} toks){_RESET}" if self._tokens is not None else ""
        src       = f"{_DIM}({filename}:{lineno}){_RESET}"

        line = f"{ts}  {agent_tag}  {ev_tag}  {agterm._colorize_agnames(msg)}{tok_tag}  {src}"
        with agterm._lock:
            try:
                from . import agui as _agui
                if _agui._active is not None:
                    _agui._active.add_log(line)
                    return
            except Exception:
                pass
            print(line, file=sys.stderr, flush=True)
