"""Run a native coding agent; add --webui to view the run and completed trace."""

import argparse
import os
from pathlib import Path

from agency import Agent, agdata, agskill, sigterm_as_exit
from agency.configs.agconfig import agconfig, llmconfig, resourcesconfig, sandboxconfig


def main() -> None:
    cfg = agconfig(
        llmconfig(
            provider="openai",
            base_url="https://api.openai.com/v1",
            model=os.environ.get("OPENAI_MODEL", "gpt-6-luna"),
            api_key=os.environ["OPENAI_API_KEY"],
            reasoning_effort="none",
            max_completion_tokens=4096,
        ),
        sandboxconfig(gpu_passthrough=False),
        resourcesconfig(idle_cpus=1),
    )
    task = agskill(
        name="code_and_test",
        prompt="Complete the request in /workspace. Run the code, then summarize the result.",
        input_schema=agdata(request=str),
        output_schema=agdata(summary=str),
    )
    worker = Agent("coder", agconfig=cfg, harness="native")
    result = worker.run(
        task,
        agdata(
            request="Write sum_even(numbers) in Python and run assertions for empty and mixed lists."
        ),
    )
    result.wait()
    print(result.summary)
    print(f"Run directory: {Path(worker.data_logger.db_path).parent.parent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--webui", action="store_true", help="Serve the dashboard on port 7860")
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("export OPENAI_API_KEY before running this example")
    if args.webui:
        from agency.observability.agwebui import agwebui

        agwebui.run(main)
    else:
        with sigterm_as_exit("agency-quickstart"):
            main()
