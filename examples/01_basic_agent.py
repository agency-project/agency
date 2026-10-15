"""Lesson 1: one agent, one typed skill, and one pending result."""

from __future__ import annotations

from agency import Agent, agdata, agskill

from _common import close_sandboxes, example_wait_timeout, run_example, tutorial_config


def main() -> None:
    cfg, run_dir = tutorial_config("01_basic_agent")
    summarize = agskill(
        name="summarize",
        prompt=(
            "Summarize the topic for the requested audience in one sentence."
            "\nAfter submitting all output fields, finish with a short confirmation."
        ),
        input_schema=agdata(topic=str, audience=str),
        output_schema=agdata(summary=str),
    )
    learner = Agent("learner", agconfig=cfg)

    result = learner.run(summarize, agdata(topic="sandboxed agents", audience="Python developers"))
    print(f"submitted: pending={result.is_pending()}")
    result.wait(timeout=example_wait_timeout())
    print(f"summary: {result.summary}")
    print(f"artifacts: {run_dir}")

    close_sandboxes([learner])


if __name__ == "__main__":
    run_example(main)
