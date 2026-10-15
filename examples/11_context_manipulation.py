"""Lesson 11: rewrite what the LLM backend is sent with ``patch_history``.

A patch replaces one message block (as the harness sends it) with other blocks.
It is kept in ``message_patch_lookup`` on the agent's ``agcontext``, so it applies
to every later LLM call, and ``pause()``/``resume()`` choose which call is first.
This lesson patches a whole block -- a fact the conversation relies on -- not
just a word inside it. See docs/Message_Patches.md.
"""

from __future__ import annotations

import time

from agency import Agent, MessageBlock, agdata, agrawstring, agskill, get_orchestrator

from _common import close_sandboxes, example_wait_timeout, run_example, tutorial_config


GIVE_INFO = agskill(
    name="give_info",
    prompt="Return the provided information.",
    input_schema=agdata(request=agrawstring),
    output_schema=agdata(answer=agrawstring),
)

STATE_FACTS = agdata(request="Alice has a blue car, Bob has a red car.")
ASK_COLOR = agdata(request="What is the color of Alice's car?")
ASK_COLOR_AGAIN = agdata(
    request=("It seems that your previous answer is incorrect. What is the color of Alice's car?")
)


def find_block(history: agdata, needle: str) -> MessageBlock:
    """The user's text block that contains *needle* in ``agent.history``."""
    for message in history.messages:
        if message["role"] != "user":
            continue
        for block in message["blocks"]:
            if block.get("type") == "text" and needle in block["text"]:
                return MessageBlock(message["role"], block)
    raise LookupError(f"no user text block contains {needle!r}")


def wait_until_running(cfg) -> None:
    """Wait until the orchestrator has started the submitted skill run."""
    orchestrator = get_orchestrator(cfg)
    deadline = time.monotonic() + example_wait_timeout()
    while orchestrator.snapshot().running_count < 1:
        if time.monotonic() > deadline:
            raise TimeoutError("the skill run did not start")
        time.sleep(0.1)


def main() -> None:
    cfg, run_dir = tutorial_config("11_context_manipulation")
    learner = Agent("patched", agconfig=cfg)

    # 1. State the facts the rest of the conversation relies on. ``history`` shows
    #    the resulting block exactly as the harness sent it, so that is where the
    #    block to patch comes from.
    stated = learner.run(GIVE_INFO, STATE_FACTS)
    stated.wait(timeout=example_wait_timeout())
    if "error" in stated.to_dict():
        raise SystemExit(f"stating the facts failed: {stated.to_dict()['error']}")
    print(f"Initial information: {stated.answer}")
    facts = find_block(learner.history, "Alice has a blue car")
    print(f"Block to patch: {facts!r} (hash {facts.hash[:12]})")

    # 2. Ask about them before any patch exists.
    first_answer = learner.run(GIVE_INFO, ASK_COLOR)
    print(f"Agent recall before the patch: {first_answer.answer}")

    # 3. A patch made while a skill run is active lands at that skill run's next
    #    LLM call. pause() returns once the harness is stopped, so the patch is
    #    in place before the call that resume() lets through. The replacement is
    #    a whole new block, not the old one with a word swapped inside it, so the
    #    harness receives an entirely different fact.
    swapped_facts = MessageBlock(
        facts.role,
        {"type": "text", "index": 0, "text": "Alice has a red car, Bob has a blue car."},
    )
    learner.pause()
    corrected = learner.run(GIVE_INFO, ASK_COLOR_AGAIN)
    wait_until_running(cfg)
    key = learner.patch_history(facts, [swapped_facts])
    learner.resume()
    print(f"Agent recall after the patch: {corrected.answer}")

    # 4. Withdrawing the patch restores the harness's own block -- again timed
    #    with pause()/resume() around the skill run it should land before.
    learner.pause()
    reverted = learner.run(GIVE_INFO, ASK_COLOR_AGAIN)
    wait_until_running(cfg)
    learner.remove_history_patch(key)
    learner.resume()
    print(f"Agent recall after removing the patch: {reverted.answer}")

    print(f"artifacts: {run_dir}")
    close_sandboxes([learner])


if __name__ == "__main__":
    run_example(main)
