"""
Port of basic/hello_world.py from openai-agents-python.

Original: Agent that responds only in haikus.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_config import make_llm_config
from agency import agent, agskill, agdata

LLM_CONFIG = make_llm_config(max_tokens=16000)

haiku_skill = agskill(
    name="haiku",
    system_prompt="You only respond in haikus.",
    input_schema=agdata(message=str),
    output_schema=agdata(response=str),
    replace_tools=[],
)

ag = agent(llm_config=LLM_CONFIG)

if __name__ == "__main__":
    result = ag.run(haiku_skill, agdata(message="Tell me about recursion in programming."))
    print(result.response)
    # Function calls itself,
    # Looping in smaller pieces,
    # Endless by design.
