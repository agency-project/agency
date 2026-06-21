"""agwebui_emitter — writes structured UI events to a JSONL file.

The execution process calls these methods; the standalone web server tails
the file and pushes events to connected browsers.  No agency imports here so
this module can be imported from both sides if needed.
"""
from __future__ import annotations

import json
import re as _re
import threading
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# ANSI → hex colour conversion (mirrors agterm's palette)
# ---------------------------------------------------------------------------

def _xterm256_hex(n: int) -> str:
    if n < 16:
        _ANSI16 = [
            "#000000", "#aa0000", "#00aa00", "#aa8800",
            "#0000aa", "#aa00aa", "#00aaaa", "#aaaaaa",
            "#555555", "#ff5555", "#55ff55", "#ffff55",
            "#5555ff", "#ff55ff", "#55ffff", "#ffffff",
        ]
        return _ANSI16[n]
    if n < 232:
        idx = n - 16
        def _c(lvl: int) -> int: return 0 if lvl == 0 else 55 + 40 * lvl
        return f"#{_c(idx // 36):02x}{_c((idx // 6) % 6):02x}{_c(idx % 6):02x}"
    v = 8 + (n - 232) * 10
    return f"#{v:02x}{v:02x}{v:02x}"


def ansi_to_hex(ansi: str) -> str:
    """Convert an agterm ANSI escape code to a CSS hex colour string."""
    m = _re.match(r"\033\[38;5;(\d+)m", ansi)
    if m:
        return _xterm256_hex(int(m.group(1)))
    return "#d4d4d4"


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------

class agwebui_emitter:
    """Thread-safe JSONL event writer for the web UI."""

    def __init__(self, run_dir: Path) -> None:
        self._event_file = run_dir / "ui_events.jsonl"
        self._reply_dir  = run_dir / "ui_replies"
        self._reply_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Latest cumulative token counts per agent; flushed again on done().
        self._token_state: dict[str, tuple[int, int, int, int]] = {}
        # Registration state re-emitted in done() for late-joining clients.
        self._agent_registry: dict[str, dict] = {}   # agname -> event dict
        self._team_registry:  dict[str, dict] = {}   # team_name -> event dict

    # ------------------------------------------------------------------
    # Core emit
    # ------------------------------------------------------------------

    def emit(self, event: dict) -> None:
        line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            with open(self._event_file, "a", encoding="utf-8") as f:
                f.write(line)

    # ------------------------------------------------------------------
    # Typed emitters
    # ------------------------------------------------------------------

    def log(self, line: str) -> None:
        self.emit({"type": "log", "line": line, "ts": time.time()})

    def agent_registered(self, agname: str, hex_color: str, team: str | None = None) -> None:
        ev = {
            "type": "agent_registered",
            "agname": agname,
            "color": hex_color,
            "team": team,
            "ts": time.time(),
        }
        with self._lock:
            self._agent_registry[agname] = ev
        self.emit(ev)

    def agent_state(
        self, agname: str, state: str, skill: str | None, tool: str | None,
        color: str | None = None, team: str | None = None,
    ) -> None:
        self.emit({
            "type": "agent_state",
            "agname": agname,
            "state": state,
            "skill": skill,
            "tool": tool,
            "color": color,
            "team": team,
            "ts": time.time(),
        })

    def team_registered(self, team_name: str, agent_names: list[str]) -> None:
        ev = {
            "type": "team_registered",
            "team_name": team_name,
            "agents": agent_names,
            "ts": time.time(),
        }
        with self._lock:
            self._team_registry[team_name] = ev
        self.emit(ev)

    def push_messages(self, agname: str, messages: list[dict]) -> None:
        try:
            json.dumps(messages)
        except Exception:
            return
        self.emit({
            "type": "messages_snapshot",
            "agname": agname,
            "messages": messages,
            "ts": time.time(),
        })

    _ASK_TIMEOUT_REPLY = "[no human available — timed out]"

    def ask_human(self, agname: str, ask_id: str, question: str,
                  timeout_s: float | None = 300) -> str:
        """Emit ask event then block-poll until the web UI delivers a reply or timeout.

        Pass ``timeout_s=None`` to wait indefinitely (for interactive use cases).
        """
        self.emit({
            "type": "ask_human",
            "agname": agname,
            "ask_id": ask_id,
            "question": question,
            "ts": time.time(),
        })
        reply_file = self._reply_dir / f"{ask_id}.txt"
        deadline = (time.time() + timeout_s) if timeout_s is not None else None
        while not reply_file.exists():
            if deadline is not None and time.time() >= deadline:
                self.emit({"type": "human_reply", "ask_id": ask_id,
                           "agname": agname, "reply": self._ASK_TIMEOUT_REPLY})
                return self._ASK_TIMEOUT_REPLY
            time.sleep(0.2)
        text = reply_file.read_text(encoding="utf-8").strip()
        try:
            reply_file.unlink()
        except Exception:
            pass
        return text

    def token_update(
        self,
        agname: str,
        agent_input: int,
        agent_output: int,
        global_input: int,
        global_output: int,
    ) -> None:
        """Emit cumulative token counts for one agent and the framework total."""
        with self._lock:
            self._token_state[agname] = (agent_input, agent_output, global_input, global_output)
        self.emit({
            "type":         "token_update",
            "agname":       agname,
            "agent_input":  agent_input,
            "agent_output": agent_output,
            "global_input": global_input,
            "global_output": global_output,
            "ts": time.time(),
        })

    def resource_update(
        self,
        gpus_acquired: int,
        gpus_total: int,
        cpus_acquired: float,
        cpus_total: int,
        memory_acquired_mb: int,
        memory_total_mb: int,
    ) -> None:
        """Emit current resource acquisition counts for the dashboard badge."""
        self.emit({
            "type":                "resource_update",
            "gpus_acquired":       gpus_acquired,
            "gpus_total":          gpus_total,
            "cpus_acquired":       round(cpus_acquired, 1),
            "cpus_total":          cpus_total,
            "memory_acquired_mb":  memory_acquired_mb,
            "memory_total_mb":     memory_total_mb,
            "ts": time.time(),
        })

    def done(self) -> None:
        # Re-emit registration and token state at the end of the file so that
        # late-joining clients (which only see the last TAIL_BYTES) always
        # receive a complete roster and current token counts.
        with self._lock:
            agents  = list(self._agent_registry.values())
            teams   = list(self._team_registry.values())
            tokens  = dict(self._token_state)
        now = time.time()
        for ev in agents:
            self.emit({**ev, "ts": now})
        for ev in teams:
            self.emit({**ev, "ts": now})
        for agname, (ai, ao, gi, go) in tokens.items():
            self.emit({
                "type":         "token_update",
                "agname":       agname,
                "agent_input":  ai,
                "agent_output": ao,
                "global_input": gi,
                "global_output": go,
                "ts": now,
            })
        self.emit({"type": "done", "ts": now})
