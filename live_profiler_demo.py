import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "examples"))

from agency import Agent, agdata, agskill
from agency.observability.agwebui import agwebui
from _common import close_sandboxes, run_example, tutorial_config


def main():
    cfg, run_dir = tutorial_config("live_codex_demo", harness="codex")
    runtime = next((name for name in ("podman", "docker") if shutil.which(name)), None)
    if runtime is None:
        raise RuntimeError(
            "The Codex demo requires Docker or Podman. The chroot backend "
            "does not provide /proc/self/exe, which Codex requires. "
            "Install a container runtime, then rerun this script."
        )
    cfg.sandbox.backend = runtime
    cfg.sandbox.gpu_passthrough = False
    cfg.orchestrator.db_path = str(Path(cfg.agent.log_dir) / "global_data.sqlite3")
    cfg.harness_adapter.binary_path = str(
        Path(sys.prefix) / "codex-0.147.0/package/vendor/x86_64-unknown-linux-musl/bin/codex"
    )
    if not Path(cfg.harness_adapter.binary_path).is_file():
        raise RuntimeError("The local Codex 0.147.0 binary is missing from this .venv.")
    learner = Agent("live-codex", agconfig=cfg, harness="codex")
    task = agskill(
        name="live_tool_demo",
        prompt=(
            "Use separate shell tool calls to perform these steps: "
            "1. Create /workspace/live-demo/squares.py that prints squares "
            "of integers 1 through 10. "
            "2. Read the file back. "
            "3. Execute it and verify all ten values. "
            "4. Compute the file's SHA-256 hash. "
            "Report the verified output and hash."
        ),
        input_schema=agdata(),
        output_schema=agdata(summary=str),
    )

    def workload():
        input("Open the live profiler, then press Enter to start: ")
        result = learner.run(task, agdata())
        result.wait(timeout=300)
        fields = result.to_dict()
        if "error" in fields:
            raise RuntimeError(f"Agent task failed: {fields['error']}")
        print(fields["summary"])

    try:
        agwebui.run(
            workload,
            run_dir=Path(cfg.agent.log_dir),
            port=7861,
            linger=True,
        )
    finally:
        close_sandboxes([learner])


if __name__ == "__main__":
    run_example(main)
