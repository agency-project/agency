"""
Port of basic/hello_world.py from openai-agents-python.

Original: Agent that responds only in haikus.
"""
import os
from agency import agent, agskill, agdata

LLM_CONFIG = {
    "base_url": os.environ.get("VLLM_BASE_URL", ""),
    "api_key":  os.environ.get("VLLM_API_KEY", ""),
    "model":    os.environ.get("VLLM_MODEL",   ""),
}

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
