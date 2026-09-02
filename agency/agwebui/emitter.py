"""Execution-side publisher for lightweight Web UI and global events.

Detailed agent data stays in each agent's SQLite database. The standalone Web
UI reads that data only when an agent is selected.
"""

from __future__ import annotations

import re as _re
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# ANSI → hex colour conversion (mirrors agterm's palette)
# ---------------------------------------------------------------------------


def _xterm256_hex(n: int) -> str:
    if n < 16:
        _ANSI16 = [
            "#000000",
            "#aa0000",
            "#00aa00",
            "#aa8800",
            "#0000aa",
            "#aa00aa",
            "#00aaaa",
            "#aaaaaa",
            "#555555",
            "#ff5555",
            "#55ff55",
            "#ffff55",
            "#5555ff",
            "#ff55ff",
            "#55ffff",
            "#ffffff",
        ]
        return _ANSI16[n]
    if n < 232:
        idx = n - 16

        def _c(lvl: int) -> int:
            return 0 if lvl == 0 else 55 + 40 * lvl

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
    """Publish UI events through the process-wide asynchronous collector."""

    _ASK_TIMEOUT_REPLY = "[no human available — timed out]"

    def __init__(self, run_dir: Path) -> None:
        from ..agcollector import get_global_data_collector

        intended_db_path = run_dir / "agency.sqlite3"
        self._reply_dir = run_dir / "ui_replies"
        self._reply_dir.mkdir(parents=True, exist_ok=True)
        self._collector = get_global_data_collector(default_db_path=intended_db_path)
        self._db_path = Path(self._collector.db_path)

        pointer = run_dir / "global_data_path.txt"
        pointer_tmp = pointer.with_suffix(".tmp")
        pointer_tmp.write_text(str(self._db_path.resolve()), encoding="utf-8")
        pointer_tmp.replace(pointer)

    def emit(self, event: dict) -> None:
        """Enqueue one lightweight UI/global event without doing disk I/O."""
        try:
            self._collector.record_ui_event(event)
        except Exception as exc:
            print(f"[agwebui] WARNING: global event publish failed: {exc}")

    def _flush_prune(self) -> None:
        """Drain the global writer; retained under the legacy test-helper name."""
        self._collector.flush(timeout_s=10)

    def log(self, line: str) -> None:
        self.emit({"type": "log", "line": line, "ts": time.time()})

    def agent_registered(self, agname: str, hex_color: str, team: str | None = None) -> None:
        """Emit presentation metadata; agent DB discovery uses the global catalog."""
        self.emit(
            {
                "type": "agent_registered",
                "agname": agname,
                "color": hex_color,
                "team": team,
                "ts": time.time(),
            }
        )

    def agent_state(
        self,
        agname: str,
        state: str,
        skill: str | None,
        tool: str | None,
        color: str | None = None,
        team: str | None = None,
    ) -> None:
        """Agent state is persisted only in the agent's own database."""
        return None

    def agent_config(self, agname: str, config: dict) -> None:
        """Agent configuration is loaded on demand from its own database."""
        return None

    def team_registered(self, team_name: str, agent_names: list[str]) -> None:
        self.emit(
            {
                "type": "team_registered",
                "team_name": team_name,
                "agents": agent_names,
                "ts": time.time(),
            }
        )

    def push_messages(self, agname: str, messages: list[dict]) -> None:
        """Message history is loaded on demand from the selected agent DB."""
        return None

    def ask_human(
        self, agname: str, ask_id: str, question: str, timeout_s: float | None = 300
    ) -> str:
        """Emit a question and wait for the Web UI's command/reply file."""
        self.emit(
            {
                "type": "ask_human",
                "agname": agname,
                "ask_id": ask_id,
                "question": question,
                "ts": time.time(),
            }
        )
        reply_file = self._reply_dir / f"{ask_id}.txt"
        deadline = (time.time() + timeout_s) if timeout_s is not None else None
        while not reply_file.exists():
            if deadline is not None and time.time() >= deadline:
                self.emit(
                    {
                        "type": "human_reply",
                        "ask_id": ask_id,
                        "agname": agname,
                        "reply": self._ASK_TIMEOUT_REPLY,
                        "ts": time.time(),
                    }
                )
                return self._ASK_TIMEOUT_REPLY
            time.sleep(0.2)
        text = reply_file.read_text(encoding="utf-8").strip()
        try:
            reply_file.unlink()
        except Exception as exc:
            print(f"[agwebui] WARNING: failed to clean up reply file {reply_file}: {exc}")
        return text

    def token_update(
        self,
        agname: str,
        agent_input: int,
        agent_output: int,
        global_input: int,
        global_output: int,
    ) -> None:
        """Detailed token usage remains in the per-agent database."""
        return None

    def resource_update(
        self,
        gpus_acquired: int,
        gpus_total: int,
        cpus_acquired: float,
        cpus_total: int,
        memory_acquired_mb: int,
        memory_total_mb: int,
    ) -> None:
        self.emit(
            {
                "type": "resource_update",
                "gpus_acquired": gpus_acquired,
                "gpus_total": gpus_total,
                "cpus_acquired": round(cpus_acquired, 1),
                "cpus_total": cpus_total,
                "memory_acquired_mb": memory_acquired_mb,
                "memory_total_mb": memory_total_mb,
                "ts": time.time(),
            }
        )

    def done(self) -> None:
        self.emit({"type": "done", "ts": time.time()})
        self._collector.flush(timeout_s=10)
