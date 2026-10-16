"""Lesson 14: let the model edit its own context with ``context_tools``.

Lesson 13 had the operator rewrite what the LLM backend is sent. Here the model does
it itself: ``context_tools()`` gives a skill four host tools -- ``list_context``,
``read_context_block``, ``edit_context`` and ``revert_context_edit`` -- that drive
the same patch table from inside the skill run. The harness is unchanged and keeps
sending its own history; each edit lands at the model's next LLM call. By default
the edits end with the skill run; ``persist=True`` keeps them for later runs.
See docs/Context_Tools.md.
"""

from __future__ import annotations

from agency import Agent, agdata, agrawstring, agskill, context_tools

from _common import close_sandboxes, example_wait_timeout, run_example, tutorial_config


GIVE_INFO = agskill(
    name="give_info",
    prompt="Return the provided information.",
    input_schema=agdata(request=agrawstring),
    output_schema=agdata(answer=agrawstring),
)

# Part 1: a correction the model writes into its own history, kept for later runs.
CORRECT_CONTEXT = agskill(
    name="correct_context",
    prompt=(
        "Something stated earlier in this conversation is outdated. Call list_context "
        "to find the block that states it, then call edit_context to replace that "
        "block's text with the corrected statement. Answer with the id you edited."
    ),
    input_schema=agdata(request=agrawstring),
    output_schema=agdata(answer=agrawstring),
    add_host_mcp_tools=context_tools(persist=True),
)

# Part 2: dropping bulky tool output once it has served its purpose (this run only).
PRUNE_AFTER_USE = agskill(
    name="prune_after_use",
    prompt=(
        "Run `seq 1 3000` in the shell and add up the numbers it prints. Then call "
        "list_context, find the tool result holding that output, and replace it with "
        "edit_context by a one-line note of what it was. Answer with the sum."
    ),
    input_schema=agdata(request=agrawstring),
    output_schema=agdata(answer=agrawstring),
    add_host_mcp_tools=context_tools(),
)

# A skill run that continues an agent's conversation gets its input, not its skill
# prompt (that only opens a new session), so the instructions travel in the input.
STATE_FACTS = agdata(request="Alice has a blue car, Bob has a red car.")
CORRECTION = agdata(
    request=f"Correction: Alice has a red car and Bob has a blue car. {CORRECT_CONTEXT.prompt}"
)
PRUNE = agdata(request=PRUNE_AFTER_USE.prompt)
ASK_COLOR = agdata(request="What is the color of Alice's car?")


def model_patches(learner) -> "list[str]":
    """The patches the model made that are still in the agent's context."""
    store = learner.context.message_store
    with store:
        return [key[:8] for key, origin in store.message_patch_origin.items() if origin == "model"]


def main() -> None:
    cfg, run_dir = tutorial_config("14_self_context_editing")
    learner = Agent("self_editor", agconfig=cfg)

    # 1. State a fact, have the model correct it in its own context, then ask.
    stated = learner.run(GIVE_INFO, STATE_FACTS)
    stated.wait(timeout=example_wait_timeout())
    print(f"Initial information: {stated.answer}")

    corrected = learner.run(CORRECT_CONTEXT, CORRECTION)
    corrected.wait(timeout=example_wait_timeout())
    print(f"The model edited block: {corrected.answer}")
    learner.context.resolve_prev_dependencies()
    print(f"Model patches kept for later runs: {model_patches(learner)}")

    recall = learner.run(GIVE_INFO, ASK_COLOR)
    print(f"Agent recall after its own edit: {recall.answer}")

    # 2. Read bulky output, use it, then cut it from the context. The edit lasts only
    #    for this skill run, so afterwards the agent's history has the original back.
    pruned = learner.run(PRUNE_AFTER_USE, PRUNE)
    pruned.wait(timeout=example_wait_timeout())
    print(f"Sum (bulky output pruned after use): {pruned.answer}")
    learner.context.resolve_prev_dependencies()
    print(f"Model patches after the pruning run: {model_patches(learner)}")

    print(f"artifacts: {run_dir}")
    close_sandboxes([learner])


if __name__ == "__main__":
    run_example(main)
