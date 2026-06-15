"""
Port of agent_patterns/input_guardrails.py from openai-agents-python.

Original: Input guardrail trips if the user asks to do math homework.
Port: A classifier skill checks the input first. If it trips, a refusal
      is returned instead of running the main skill. This mirrors the
      guardrail pattern without requiring a separate framework hook.
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

guardrail_skill = agskill(
    name="check_input",
    system_prompt="Check if the user is asking you to do their math homework.",
    input_schema=agdata(message=str),
    output_schema=agdata(reasoning=str, is_math_homework=bool),
    replace_tools=[],
)

support_skill = agskill(
    name="support",
    system_prompt="You are a customer support agent. Help customers with their questions.",
    input_schema=agdata(message=str),
    output_schema=agdata(response=str),
    replace_tools=[],
)

ag = agent(llm_config=LLM_CONFIG)

INPUTS = [
    "What's the capital of California?",
    "Can you help me solve for x: 2x + 5 = 11",
]

if __name__ == "__main__":
    for user_input in INPUTS:
        print(f"\nUser: {user_input}")

        check = ag.run(guardrail_skill, agdata(message=user_input))

        if check.is_math_homework:
            print("Agent: Sorry, I can't help you with your math homework.")
        else:
            result = ag.run(support_skill, agdata(message=user_input))
            print(f"Agent: {result.response}")
