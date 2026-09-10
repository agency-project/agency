"""Lesson 10: swap harnesses and wrap a workload with the monitoring UI."""

from __future__ import annotations

import os
from pathlib import Path

from agency import Agent, agdata, agskill
from agency.observability.agwebui import agwebui

from _common import close_sandboxes, run_example, tutorial_config


ECHO = agskill(
    name="harness_echo",
    system_prompt="Return the supplied harness label exactly.",
    input_schema=agdata(label=str),
    output_schema=agdata(answer=str),
)


def main() -> None:
    cfg, run_dir = tutorial_config("10_harnesses_and_webui")
    agents: list[Agent] = []

    def workload() -> None:
        codex = Agent("codex-harness", agconfig=cfg, harness="codex")
        native = Agent("native-harness", agconfig=cfg, harness="native")
        agents.extend([codex, native])
        codex_result = codex.run(ECHO, agdata(label="CODEX"))
        native_result = native.run(ECHO, agdata(label="NATIVE"))
        print(f"harness results: {codex_result.answer}, {native_result.answer}")

    port = int(os.environ.get("AGENCY_WEBUI_PORT", "7861"))
    linger = os.environ.get("AGENCY_WEBUI_LINGER") == "1"
    if linger:
        print(
            "interactive Web UI mode: keep this process running and tunnel with "
            f"`ssh -N -L {port}:127.0.0.1:{port} <ec2-host>`"
        )
    else:
        print("temporary Web UI mode: the server stops when the workload finishes")
    agwebui.run(
        workload,
        run_dir=Path(cfg.agent.log_dir),
        port=port,
        linger=linger,
    )
    if not linger:
        print("web UI stopped; its server log remains available")
    print(f"web UI log: {Path(cfg.agent.log_dir) / 'server.log'}")
    print(f"artifacts: {run_dir}")
    close_sandboxes(agents)


if __name__ == "__main__":
    run_example(main)
