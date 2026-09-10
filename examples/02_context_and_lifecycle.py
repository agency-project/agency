"""Lesson 2: context ordering, async calls, and agent lifecycle controls."""

from __future__ import annotations

import asyncio
import time

from agency import Agent, agcontext, agdata, agrawstring, agskill

from _common import close_sandboxes, run_example, tutorial_config


RAW_ECHO = agskill(
    name="raw_echo",
    system_prompt="Return only the requested final word, with no punctuation or explanation.",
    input_schema=agdata(request=agrawstring),
    output_schema=agdata(answer=agrawstring),
)

MAKE_TEXT = agskill(
    name="make_text",
    system_prompt="Return the requested token exactly.",
    input_schema=agdata(token=str),
    output_schema=agdata(text=str),
)

WRAP_TEXT = agskill(
    name="wrap_text",
    system_prompt="Wrap the supplied text in square brackets.",
    input_schema=agdata(text=str),
    output_schema=agdata(wrapped=str),
)


async def async_calls(learner: Agent) -> None:
    first = await learner.asyncio_run(RAW_ECHO, agdata(request="Return ASYNC_AGENT"))
    second = await RAW_ECHO.asyncio_run(learner, agdata(request="Return ASYNC_SKILL"))
    print(f"async results: {first.answer}, {second.answer}")


def main() -> None:
    cfg, run_dir = tutorial_config("02_context_and_lifecycle")
    learner = Agent("context", agconfig=cfg)

    queued_message = "For the next request, the secret word is ORBIT."
    learner.queue_message(queued_message)
    print(f"queued message: {queued_message}")
    recalled = learner.run(
        RAW_ECHO,
        agdata(request="Return the secret word from retained context."),
    )
    print(f"queued context: {recalled.answer}")

    produced = learner.run(MAKE_TEXT, agdata(token="dependency"))
    consumed = learner.run(WRAP_TEXT, produced)
    print(f"dependency fan-through: {consumed.wrapped}")

    asyncio.run(async_calls(learner))

    completed = learner.run(RAW_ECHO, agdata(request="Return RED"))
    assert completed.answer == "RED"
    learner.redirect(
        completed,
        "For the next request, the redirected secret word is BLUE.",
    )
    redirected = learner.run(
        RAW_ECHO,
        agdata(request="Return the redirected secret word from retained context."),
    )
    assert redirected.answer == "BLUE"
    print("completed-result redirect became future context: RED -> BLUE")

    learner.pause()
    paused = learner.run(RAW_ECHO, agdata(request="Return RESUMED"))
    time.sleep(1)
    print(f"paused: requested={learner.is_paused()}, pending={paused.is_pending()}")
    learner.resume()
    print(f"resumed result: {paused.answer}")

    learner.pause()
    cancelled = learner.run(RAW_ECHO, agdata(request="Return NEVER"))
    learner.cancel(cancelled)
    learner.resume()
    print(f"cancelled result: {cancelled.to_dict()}")

    copied: agcontext = learner.context.copy()
    assert copied.get_resolved_transcript() == learner.ctx.get_resolved_transcript()
    print(f"context messages: {len(learner.history.messages)}")
    print(f"artifacts: {run_dir}")

    close_sandboxes([learner])


if __name__ == "__main__":
    run_example(main)
