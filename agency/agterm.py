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
                if r == g == b:  # cube grey diagonal
                    continue
                if max(r, g, b) <= 2:  # near-black (brightest component ≤ 135)
                    continue
                indices.append(16 + 36 * r + 6 * g + b)
    colors = [f"\033[38;5;{idx}m" for idx in indices]
    _random.shuffle(colors)
    return colors


_AGENT_COLORS: list[str] = _make_color_palette()

# Fixed greyscale styles for event tags — colours are reserved for agent IDs
_EVENT_STYLES: dict[str, str] = {
    "CREATED  ": "\033[1m",  # bold
    "FORKED   ": "\033[1m",  # bold
    "DESTROYED": "\033[2m",  # dim
    "SKILL ▶  ": "\033[1m",  # bold
    "SKILL ✓  ": "\033[0m",  # normal
    "SKILL ✗  ": "\033[7m",  # reverse video  (visible without colour)
    "ERROR ✗  ": "\033[7m",  # reverse video  (agdata-level errors)
    "LLM ▶    ": "\033[0m",  # normal
    "LLM ✓    ": "\033[0m",  # normal
    "TOOL     ": "\033[0m",  # normal
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVENT_TAG_WIDTH = (
    9  # Fixed column width for event tag display in log lines (e.g. 'DESTROYED' is 9 chars)
)

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


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
        try:
            from . import agwebui as _agwebui

            if _agwebui._active is not None:
                from .agwebui.emitter import ansi_to_hex as _ansi_to_hex
                from ._context import _active_team as _at

                _team = _at.get(None)
                _team_name = _team.team_name if _team is not None else None
                _agwebui._active.emitter.agent_registered(
                    self._id, _ansi_to_hex(self._color), team=_team_name
                )
        except Exception as _e:
            print(f"[agterm] WARNING: agent_registered push failed for {self._id}: {_e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _colorize_agnames(msg: str) -> str:
        """Wrap every registered agent UUID appearing in msg with its color."""
        for uid, color in list(agterm._agname_colors.items()):
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
        filename = frame.f_code.co_filename.rsplit("/", 1)[-1] if frame is not None else "?"
        lineno = frame.f_lineno if frame is not None else 0

        ts = f"{_DIM}{datetime.now().strftime('%H:%M:%S.%f')[:-3]}{_RESET}"
        agent_tag = f"{self._color}{_BOLD}[{self._id}]{_RESET}"
        ev_key = event.ljust(EVENT_TAG_WIDTH)[:EVENT_TAG_WIDTH]
        ev_style = _EVENT_STYLES.get(ev_key, "")
        ev_tag = f"{ev_style}[{ev_key}]{_RESET}"
        tok_tag = f"  {_DIM}({self._tokens} toks){_RESET}" if self._tokens is not None else ""
        src = f"{_DIM}({filename}:{lineno}){_RESET}"

        line = f"{ts}  {agent_tag}  {ev_tag}  {agterm._colorize_agnames(msg)}{tok_tag}  {src}"
        _is_error = "✗" in event
        with agterm._lock:
            try:
                from . import agwebui as _agwebui

                if _agwebui._active is not None:
                    _agwebui._active.emitter.log(line)
                    if not _is_error:
                        return  # non-error events go to webui only when it's active
            except Exception as _e:
                # Falls through to the stderr print below regardless.
                print(
                    f"[agterm] WARNING: failed to forward log line to webui: {_e}", file=sys.stderr
                )
            print(line, file=sys.stderr, flush=True)
