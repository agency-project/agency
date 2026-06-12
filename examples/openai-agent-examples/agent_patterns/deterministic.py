"""
Port of agent_patterns/deterministic.py from openai-agents-python.

Original: Sequential pipeline — outline → quality check (gate) → story.
Port: Three agskills chained on a single agent's history.
      The gate uses output_validator on the checker skill.
"""
import os
from agency import agent, agskill, agdata, AgError

LLM_CONFIG = {
    "base_url": os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:18000/v1"),
    "api_key":  os.environ.get("VLLM_API_KEY", ""),
    "model":    os.environ.get("VLLM_MODEL",   "google/gemma-4-E2B-it"),
    "temperature":       0.6,
    "max_tokens":        16000,
    "top_p":             0.95,
    "top_k":             50,
    "repetition_penalty": 1.1,
}

outline_skill = agskill(
    name="outline",
    system_prompt="Generate a very short story outline based on the user's input.",
    input_schema=agdata(prompt=str),
    output_schema=agdata(outline=str),
    replace_tools=[],
)

checker_skill = agskill(
    name="check",
    system_prompt=(
        "Read the given story outline and judge its quality. "
        "Determine if it is a sci-fi story. "
        "Set good_quality to true/false and is_scifi to true/false."
    ),
    input_schema=agdata(outline=str),
    output_schema=agdata(good_quality=bool, is_scifi=bool),
    replace_tools=[],
)

story_skill = agskill(
    name="story",
    system_prompt="Write a short story based on the given outline.",
    input_schema=agdata(outline=str),
    output_schema=agdata(story=str),
    replace_tools=[],
)

ag = agent(llm_config=LLM_CONFIG)

if __name__ == "__main__":
    prompt = input("What kind of story do you want? ") or "Write a short sci-fi story."

    # Step 1: generate outline
    r1 = ag.run(outline_skill, agdata(prompt=prompt))
    print(f"Outline: {r1.outline}\n")

    # Step 2: check quality and genre
    r2 = ag.run(checker_skill, agdata(outline=r1.outline))

    if not r2.good_quality:
        print("Outline is not good quality — stopping.")
        raise SystemExit(0)

    if not r2.is_scifi:
        print("Outline is not a sci-fi story — stopping.")
        raise SystemExit(0)

    print("Outline passed quality check — writing story...")

    # Step 3: write the story
    r3 = ag.run(story_skill, agdata(outline=r1.outline))
    print(f"\nStory:\n{r3.story}")
