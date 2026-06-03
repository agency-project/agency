"""agUI — split-screen TUI for monitoring and interacting with running agents.

Layout::

    ┌──────────────────────────────┬──────────────────┐
    │  Shared log (all agents)     │ Agents           │
    │  scrolling stdout            │                  │
    │                              │ ● rock_042       │
    │                              │   running        │
    ├──────────────────────────────│ ○ sand_001       │
    │  rock_042  [1/3]  Tab/S-Tab  │   idle           │
    │                              │                  │
    │  ? What license to use?      │                  │
    │  > _                         │                  │
    └──────────────────────────────┴──────────────────┘

    Tab / Shift-Tab cycles the focused agent in the bottom pane.
    When an agent calls ask_human(), the interaction pane auto-switches
    to that agent and waits for the user's reply.

Usage::

    from agency import agent, agUI

    with agUI():
        ag = agent(llm_config=..., agskills=[...])
        result = ag.run("task", agdata(...))

If agUI() is not used, all output falls back to stderr and ask_human()
falls back to a plain input() call — the framework works identically
without the TUI.
"""
from __future__ import annotations

import io
import queue
import sys
import threading
from typing import TYPE_CHECKING, Any

# Map agterm ANSI codes → Rich color names for the agent list panel
_ANSI_TO_RICH: dict[str, str] = {
    "\033[97m": "bright_white",
    "\033[96m": "bright_cyan",
    "\033[93m": "bright_yellow",
    "\033[92m": "bright_green",
    "\033[95m": "bright_magenta",
    "\033[94m": "bright_blue",
    "\033[91m": "bright_red",
    "\033[33m": "yellow",
}

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import RichLog, Input, Static

# Module-level singleton — set while an agUI() context is active.
_active: "agUI | None" = None


class _UIWriter(io.TextIOBase):
    """Stdout proxy that routes print() output to the shared log pane."""

    def __init__(self, ui: "agUI") -> None:
        self._ui = ui
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._ui.add_log(line)
        return len(text)

    def flush(self) -> None:
        if self._buf:
            self._ui.add_log(self._buf)
            self._buf = ""

    def isatty(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Internal Textual application
# ---------------------------------------------------------------------------

class _AgencyApp(App):
    CSS = """
    Screen {
        layout: horizontal;
        overflow: hidden;
    }
    #left {
        width: 1fr;
        layout: vertical;
    }
    #shared-log {
        height: 5fr;
        border: solid $primary;
    }
    #interaction {
        height: 7fr;
        layout: vertical;
        border: solid $accent;
    }
    #agent-history {
        height: 1fr;
        padding: 0 1;
    }
    #agent-input {
        height: 1;
        border: none;
        margin: 0 1 1 1;
        padding: 0 1;
    }
    #right {
        width: 33;
        layout: vertical;
        border-left: solid $primary-darken-1;
        padding: 1 1;
    }
    #right-title {
        text-style: bold underline;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("tab",       "cycle_fwd", "→ agent", priority=True, show=True),
        Binding("shift+tab", "cycle_bck", "← agent", priority=True, show=True),
        Binding("q",         "quit",      "Quit",    show=True),
        Binding("ctrl+c",    "quit",      "Quit",    priority=True, show=False),
    ]

    def __init__(self, ready_event: threading.Event,
                 done_event: threading.Event) -> None:
        super().__init__()
        self._ready_event = ready_event
        self._done_event  = done_event
        self._agents: list[str] = []
        self._agent_idx: int = 0
        self._histories: dict[str, list[tuple[str, str]]] = {}
        self._pending: dict[str, tuple[str, "queue.Queue[str]"]] = {}
        self._dot_tick: int = 0

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="left"):
                yield RichLog(id="shared-log", highlight=False, markup=True)
                with Vertical(id="interaction"):
                    yield ScrollableContainer(id="agent-history")
                    yield Input(placeholder="type reply, then Enter…", id="agent-input")
            with Vertical(id="right"):
                yield Static("Agent List", id="right-title")
                yield Static("", id="agent-list")

    def on_mount(self) -> None:
        self.query_one("#shared-log", RichLog).border_title = "Shared Log"
        self._refresh_interaction_title()
        self.set_interval(1.0, self._refresh_agent_list)
        self.query_one("#agent-input", Input).focus()
        self._ready_event.set()

    def on_unmount(self) -> None:
        self._done_event.set()

    def _mark_done(self) -> None:
        self._add_log(
            "\n\033[1;32m✓ All done\033[0m  —  "
            "\033[1mq\033[0m to exit"
        )

    # ------------------------------------------------------------------
    # Thread-safe update methods — call via call_from_thread()
    # ------------------------------------------------------------------

    def _add_log(self, line: str) -> None:
        from rich.text import Text
        self.query_one("#shared-log", RichLog).write(Text.from_ansi(line))

    def _add_agent_msg(self, agname: str, text: str, role: str) -> None:
        self._histories.setdefault(agname, []).append((role, text))
        if agname not in self._agents:
            self._agents.append(agname)
        if self._current_agent() == agname:
            self._render_history()

    def _post_question(self, agname: str, question: str,
                       reply_q: "queue.Queue[str]") -> None:
        self._pending[agname] = (question, reply_q)
        self._histories.setdefault(agname, []).append(("question", question))
        if agname not in self._agents:
            self._agents.append(agname)
        self._agent_idx = self._agents.index(agname)
        self._refresh_interaction_title()
        self._render_history()
        self.query_one("#agent-input", Input).focus()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_agent(self) -> str | None:
        if not self._agents:
            return None
        return self._agents[self._agent_idx % len(self._agents)]

    def _refresh_interaction_title(self) -> None:
        pane = self.query_one("#interaction", Vertical)
        ag = self._current_agent()
        n = len(self._agents)
        if ag:
            waiting = "  ?" if ag in self._pending else ""
            pane.border_title = f" {ag}  [{self._agent_idx + 1}/{n}]  Tab/S-Tab{waiting}"
        else:
            pane.border_title = " No agents"

    def _render_history(self) -> None:
        import json as _json
        container = self.query_one("#agent-history", ScrollableContainer)
        container.remove_children()
        ag = self._current_agent()

        # Show the agent's chat context (snapshot from last resolved skill run)
        if ag is not None:
            try:
                from .agent import agent as Agent
                live_map = {a.agname: a for a in Agent.all()}
                a = live_map.get(ag)
                msgs = list(a._snapshot_messages) if a is not None else []
            except Exception:
                msgs = []

            for msg in msgs:
                role = msg.get("role", "?")
                content = msg.get("content") or ""

                if role == "system":
                    # Show first line of system prompt as a dim separator
                    first_line = content.split("\n")[0][:120]
                    container.mount(Static(f"[dim]─── sys: {first_line}[/]"))

                elif role == "user":
                    # Skill input — show as compact JSON header
                    preview = content[:300].replace("\n", " ")
                    container.mount(Static(f"[bold cyan]▶ user[/]  {preview}"))

                elif role == "assistant":
                    tool_calls = msg.get("tool_calls") or []
                    if tool_calls:
                        for tc in tool_calls:
                            fn = tc.get("function", {})
                            name = fn.get("name", "?")
                            try:
                                raw = _json.loads(fn.get("arguments", "{}"))
                                args_lines = [f"  {k}: {str(v)[:120]}" for k, v in raw.items()]
                                args_text = "\n".join(args_lines)
                            except Exception:
                                args_text = f"  {fn.get('arguments','')[:200]}"
                            container.mount(Static(
                                f"[bold yellow]⚙ {name}[/]\n[dim]{args_text}[/]"
                            ))
                    if content:
                        # Full assistant text, preserve newlines, dim label
                        container.mount(Static(
                            f"[bold green]◆ asst[/]\n{content}"
                        ))

                elif role == "tool":
                    # Tool result — show up to 600 chars, preserve structure
                    preview = content[:600]
                    container.mount(Static(f"[dim]← {preview}[/]"))

            # Running indicator — label depends on ui_state, dots cycle each tick
            if ag is not None:
                try:
                    from .agent import agent as Agent
                    live_map = {a.agname: a for a in Agent.all()}
                    a = live_map.get(ag)
                    if a is not None and a._history.is_pending():
                        dots = "." * ((self._dot_tick % 3) + 1)
                        st = getattr(a, "_ui_state", {})
                        state = st.get("state", "skill")
                        tool  = st.get("tool")
                        if state == "llm":
                            label = f"LLM Thinking{dots}"
                        elif state == "tool" and tool:
                            label = f"Tool Running: {tool}{dots}"
                        elif state == "tool":
                            label = f"Tool Running{dots}"
                        elif state == "proc_wait":
                            label = f"Waiting for processes{dots}"
                        else:
                            label = f"Running{dots}"
                        container.mount(Static(f"[bold yellow]▶ {label}[/]"))
                except Exception:
                    pass

        # Pending ask_human question for this agent
        if ag in self._pending:
            question, _ = self._pending[ag]
            container.mount(Static(f"[bold yellow]? {question}[/]"))

        self.call_after_refresh(
            lambda: self.query_one("#agent-history", ScrollableContainer)
                        .scroll_end(animate=False)
        )

    def _refresh_agent_list(self) -> None:
        try:
            from .agent import agent as Agent
            from .agterm import agterm
            live = Agent.all()
        except Exception:
            return

        live_map = {a.agname: a for a in live}

        # Sync _agents to only live agents, preserving order
        current = self._current_agent()
        self._agents = [n for n in self._agents if n in live_map]
        for a in live:
            if a.agname not in self._agents:
                self._agents.append(a.agname)
        # Re-anchor the index to the same agent, or clamp it
        if current and current in self._agents:
            self._agent_idx = self._agents.index(current)
        elif self._agents:
            self._agent_idx = min(self._agent_idx, len(self._agents) - 1)
        else:
            self._agent_idx = 0
        self._refresh_interaction_title()

        lines: list[str] = []

        for name in self._agents:
            a = live_map.get(name)
            # Resolve agent's color from agterm registry
            ansi = agterm._agname_colors.get(name, "")
            color = _ANSI_TO_RICH.get(ansi, "white")

            if a is None:
                continue

            st = getattr(a, "_ui_state", {"state": "inactive"})
            state  = st.get("state", "inactive")
            skill  = st.get("skill")
            tool   = st.get("tool")

            if state == "inactive":
                lines.append(f"[dim]○ {name}\n  idle[/]")

            elif state == "llm":
                lines.append(
                    f"[bold {color}]● {name}[/]\n"
                    f"  [dim]{skill}[/]: [cyan]Waiting LLM[/]"
                )

            elif state == "tool":
                lines.append(
                    f"[bold {color}]● {name}[/]\n"
                    f"  [dim]{skill}[/]: [yellow]{tool}[/]"
                )

            elif state == "proc_wait":
                lines.append(
                    f"[bold {color}]● {name}[/]\n"
                    f"  [dim]{skill}[/]: [dim]Waiting shell[/]"
                )

            elif state == "human":
                lines.append(
                    f"[bold {color}]● {name}[/] [bold yellow]?[/]\n"
                    f"  [dim]{skill}[/]: [yellow]Waiting human[/]"
                )

            else:  # "skill" — generic running
                lines.append(
                    f"[bold {color}]● {name}[/]\n"
                    f"  [dim]{skill}[/] - running"
                )

        self.query_one("#agent-list", Static).update("\n".join(lines))

        # Advance dot animation counter and refresh history pane when agent is active
        self._dot_tick += 1
        ag = self._current_agent()
        if ag is not None:
            a = live_map.get(ag)
            if a is not None and getattr(a, "_ui_state", {}).get("state") != "inactive":
                self._render_history()

    # ------------------------------------------------------------------
    # Key actions
    # ------------------------------------------------------------------

    def action_cycle_fwd(self) -> None:
        if self._agents:
            self._agent_idx = (self._agent_idx + 1) % len(self._agents)
            self._refresh_interaction_title()
            self._render_history()

    def action_cycle_bck(self) -> None:
        if self._agents:
            self._agent_idx = (self._agent_idx - 1) % len(self._agents)
            self._refresh_interaction_title()
            self._render_history()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        ag = self._current_agent()
        if ag is None:
            return
        if ag in self._pending:
            _, reply_q = self._pending.pop(ag)
            self._histories.setdefault(ag, []).append(("user", text))
            self._render_history()
            self._refresh_interaction_title()
            reply_q.put(text)
        else:
            # Unsolicited message — inject into agent's ReAct inbox and show in UI.
            self._histories.setdefault(ag, []).append(("user", text))
            self._render_history()
            try:
                from .agent import agent as Agent
                for a in Agent.all():
                    if a.agname == ag:
                        a._inbox.put(text)
                        break
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Public context manager
# ---------------------------------------------------------------------------

class agUI:
    """Split-screen TUI for monitoring and interacting with running agents.

    Primary usage — ``agUI.run(fn)`` runs the TUI in the calling (main) thread
    and executes *fn* in a background worker thread.  This is required because
    Textual installs terminal signal handlers that only work from the main
    thread::

        def main():
            ag = agent(llm_config=..., agskills=[...])
            result = ag.run("task", agdata(...))

        if __name__ == "__main__":
            agUI.run(main)

    Parameters
    ----------
    linger : bool
        When True (default), keep the UI open after the script finishes and
        show a "Done — press q to exit" banner.  When False, exit immediately.
    """

    def __init__(self, linger: bool = True) -> None:
        self._linger = linger

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    @classmethod
    def run(cls, fn: "Any", *args, linger: bool = True, **kwargs) -> None:
        """Run *fn(\\*args, \\*\\*kwargs)* with the TUI active.

        Textual occupies the calling (main) thread; *fn* runs in a worker
        thread.  Blocks until *fn* completes and the user dismisses the UI
        (or immediately if *linger=False*).
        """
        global _active
        ready = threading.Event()
        done  = threading.Event()
        app   = _AgencyApp(ready, done)

        ui: agUI     = cls.__new__(cls)
        ui._linger   = linger
        ui._app      = app
        ui._done     = done

        def _worker() -> None:
            global _active
            ready.wait(timeout=10)
            _active = ui
            old_stdout = sys.stdout
            sys.stdout = _UIWriter(ui)
            try:
                fn(*args, **kwargs)
            except Exception:
                import traceback
                # Write traceback to shared log before restoring stdout
                for line in traceback.format_exc().splitlines():
                    ui.add_log(line)
            finally:
                sys.stdout = old_stdout
                _active = None
                if linger:
                    try:
                        app.call_from_thread(app._mark_done)
                    except Exception:
                        pass
                else:
                    try:
                        app.exit()
                    except Exception:
                        pass

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        try:
            app.run()          # blocks in main thread — Textual owns the terminal
        except KeyboardInterrupt:
            pass
        finally:
            t.join(timeout=10)

    # ------------------------------------------------------------------
    # Called from any thread (agterm, ask_human tool, agent loop)
    # ------------------------------------------------------------------

    def add_log(self, line: str) -> None:
        """Route a log line to the shared log pane."""
        try:
            self._app.call_from_thread(self._app._add_log, line)
        except Exception:
            pass

    def add_agent_message(self, agname: str, text: str,
                          role: str = "agent") -> None:
        """Append a message to an agent's interaction history."""
        try:
            self._app.call_from_thread(self._app._add_agent_msg, agname, text, role)
        except Exception:
            pass

    def ask_human(self, agname: str, question: str) -> str:
        """Block the calling thread until the user replies."""
        reply_q: queue.Queue[str] = queue.Queue()
        try:
            self._app.call_from_thread(self._app._post_question, agname, question, reply_q)
        except Exception:
            return input(f"\n[{agname}] {question}\n> ")
        return reply_q.get()
