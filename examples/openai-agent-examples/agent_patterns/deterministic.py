"""
Port of agent_patterns/deterministic.py from openai-agents-python.

Original: Sequential pipeline — outline → quality check (gate) → story.
Port: Three agskills chained on a single agent's history.
      Caller checks good_quality and is_scifi on the checker result and stops if they fail.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_config import make_llm_config
from agency import agent, agskill, agdata, AgError

LLM_CONFIG = make_llm_config(max_tokens=16000)

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
