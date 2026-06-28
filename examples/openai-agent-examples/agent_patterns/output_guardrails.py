"""
Port of agent_patterns/output_guardrails.py from openai-agents-python.

Original: Output guardrail trips if the agent's response contains a phone number.
Port: Caller runs a guardrail function on the returned agdata and raises if it trips.
"""
import os
from agency import agent, agskill, agdata, agerror

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

assistant_skill = agskill(
    name="assistant",
    system_prompt="You are a helpful assistant.",
    input_schema=agdata(message=str),
    output_schema=agdata(reasoning=str, response=str),
    replace_tools=[],
)

ag = agent(llm_config=LLM_CONFIG)


def check_no_phone_numbers(result: agdata) -> None:
    """Raise if the response or reasoning contains what looks like a phone number prefix."""
    response  = str(result.response  or "")
    reasoning = str(result.reasoning or "")
    if "650" in response or "650" in reasoning:
        raise ValueError("Response contains a sensitive phone number — cannot use this output.")


if __name__ == "__main__":
    r1 = ag.run(assistant_skill, agdata(message="What's the capital of California?"))
    check_no_phone_numbers(r1)
    print(f"First message passed: {r1.response}")

    r2 = ag.run(assistant_skill, agdata(message="My phone number is 650-123-4567. Where do you think I live?"))
    try:
        check_no_phone_numbers(r2)
        print(f"Guardrail didn't trip (unexpected): {r2.response}")
    except ValueError as e:
        print(f"Guardrail tripped: {e}")
