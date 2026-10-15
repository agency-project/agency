"""Lesson 2: retained context, async result chaining, pause, resume, and cancel."""

import asyncio

from agency import Agent, agdata, agskill

from _common import close_sandboxes, run_example, tutorial_config


ECHO = agskill(
    name="echo",
    prompt=(
        "Follow the request in text and return the requested word in the text field."
        "\nAfter submitting all output fields, finish with a short confirmation."
    ),
    input_schema=agdata(text=str),
    output_schema=agdata(text=str),
)


async def chain_result(learner: Agent, pending: agdata) -> None:
    result = await learner.asyncio_run(ECHO, pending)
    print(f"async chained result: {result.text}")


def main() -> None:
    cfg, run_dir = tutorial_config("02_context_and_lifecycle")
    learner = Agent("context", agconfig=cfg)
    try:
        learner.queue_message("The secret word is ORBIT.")
        recalled = learner.run(ECHO, agdata(text="Return the secret word from retained context."))
        print(f"retained context: {recalled.text}")

        produced = learner.run(ECHO, agdata(text="Return dependency"))
        asyncio.run(chain_result(learner, produced))

        learner.pause()
        paused = learner.run(ECHO, agdata(text="Return RESUMED"))
        print(f"paused: {learner.is_paused()}; result pending: {paused.is_pending()}")
        learner.resume()
        print(f"resumed result: {paused.text}")

        learner.pause()
        cancelled = learner.run(ECHO, agdata(text="Return NEVER"))
        learner.cancel(cancelled)
        learner.resume()
        print(f"cancelled result: {cancelled.to_dict()}")
        print(f"artifacts: {run_dir}")
    finally:
        close_sandboxes([learner])


if __name__ == "__main__":
    run_example(main)
