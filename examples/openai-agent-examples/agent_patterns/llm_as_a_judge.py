"""
Port of agent_patterns/llm_as_a_judge.py from openai-agents-python.

Original: Story outline generator iterated with LLM evaluator until the judge
          is satisfied. Uses shared history across turns.
Port: Two agskills on one agent; loop feeds evaluator feedback back as input.
"""
import os
from agency import agent, agskill, agdata

LLM_CONFIG = {
    "base_url": os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:18000/v1"),
    "api_key":  os.environ.get("VLLM_API_KEY", ""),
    "model":    os.environ.get("VLLM_MODEL",   ""),
    "temperature":       0.6,
    "max_tokens":        16000,
    "top_p":             0.95,
    "top_k":             50,
    "repetition_penalty": 1.1,
}

generator_skill = agskill(
    name="generate",
    system_prompt=(
        "You generate a very short story outline based on the user's input. "
        "If feedback is provided, use it to improve the outline."
    ),
    input_schema=agdata(prompt=str),
    output_schema=agdata(outline=str),
    replace_tools=[],
)

evaluator_skill = agskill(
    name="evaluate",
    system_prompt=(
        "You evaluate a story outline and decide if it's good enough. "
        "If not good enough, provide specific feedback on what to improve. "
        "Never give a pass on the first try. After 5 attempts give a pass if it's good enough."
    ),
    input_schema=agdata(outline=str),
    output_schema=agdata(score=str, feedback=str),
    replace_tools=[],
)

_VALID_SCORES = {"pass", "needs_improvement", "fail"}

def validate_score(ev: agdata) -> None:
    if str(ev.score).lower() not in _VALID_SCORES:
        raise ValueError(f"score must be one of {_VALID_SCORES}, got {ev.score!r}")

ag = agent(llm_config=LLM_CONFIG)

if __name__ == "__main__":
    prompt = input("What kind of story would you like? ") or "A detective story in space."
    current_prompt = prompt
    latest_outline = None

    for attempt in range(10):
        gen = ag.run(generator_skill, agdata(prompt=current_prompt))
        latest_outline = gen.outline
        print(f"[attempt {attempt+1}] Outline: {latest_outline[:80]}...")

        ev = ag.run(evaluator_skill, agdata(outline=latest_outline))
        validate_score(ev)
        print(f"  Score: {ev.score}  Feedback: {ev.feedback[:60]}...")

        if str(ev.score).lower() == "pass":
            print("Judge is satisfied!")
            break

        # Feed back the feedback for the next iteration
        current_prompt = f"{prompt}\n\nPrevious outline:\n{latest_outline}\n\nFeedback: {ev.feedback}"

    print(f"\nFinal outline:\n{latest_outline}")
