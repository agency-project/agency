"""
Port of agent_patterns/output_guardrails.py from openai-agents-python.

Original: Output guardrail trips if the agent's response contains a phone number.
Port: output_validator on the agskill checks the output before it resolves.
      If it fails all retries, AgError is raised.
"""
import os
from agency import agent, agskill, agdata, AgError

LLM_CONFIG = {
    "base_url": os.environ.get("VLLM_BASE_URL", "https://kimi.js-park.info:18000/v1"),
    "api_key":  os.environ.get("VLLM_API_KEY", ""),
    "model":    os.environ.get("VLLM_MODEL", "moonshotai/Kimi-K2.6"),
}


def _no_phone_numbers(result: agdata) -> list[str]:
    """Trip if the response or reasoning contains what looks like a phone number prefix."""
    response  = str(getattr(result, "response",  "") or "")
    reasoning = str(getattr(result, "reasoning", "") or "")
    if "650" in response or "650" in reasoning:
        return ["Response contains a sensitive phone number — cannot return this output."]
    return []


assistant_skill = agskill(
    name="assistant",
    system_prompt="You are a helpful assistant.",
    input_schema=agdata(message="str"),
    output_schema=agdata(reasoning="str", response="str"),
    output_validator=_no_phone_numbers,
    max_retries=1,
    tools=[],
)

ag = agent(llm_config=LLM_CONFIG)

if __name__ == "__main__":
    # Should pass
    r1 = ag.run(assistant_skill, agdata(message="What's the capital of California?"))
    print(f"First message passed: {r1.response}")

    # Should trip the guardrail
    try:
        r2 = ag.run(assistant_skill, agdata(message="My phone number is 650-123-4567. Where do you think I live?"))
        print(f"Guardrail didn't trip (unexpected): {r2.response}")
    except AgError as e:
        print(f"Guardrail tripped: {e}")
