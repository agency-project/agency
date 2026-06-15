"""
Port of agent_patterns/routing.py from openai-agents-python.

Original: Triage agent detects language and routes to the appropriate agent.
Port: A router skill returns the detected language and response;
      subsequent turns re-use the same agent.
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

french_skill = agskill(
    name="french",
    system_prompt="You only speak French.",
    input_schema=agdata(message=str),
    output_schema=agdata(response=str),
    replace_tools=[],
)

spanish_skill = agskill(
    name="spanish",
    system_prompt="You only speak Spanish.",
    input_schema=agdata(message=str),
    output_schema=agdata(response=str),
    replace_tools=[],
)

english_skill = agskill(
    name="english",
    system_prompt="You only speak English.",
    input_schema=agdata(message=str),
    output_schema=agdata(response=str),
    replace_tools=[],
)

triage_skill = agskill(
    name="triage",
    system_prompt=(
        "Detect the language of the user's message. "
        "Reply with the detected language (french, spanish, or english) in the 'language' field "
        "and pass the message through unchanged in the 'message' field."
    ),
    input_schema=agdata(message=str),
    output_schema=agdata(language=str, message=str),
    replace_tools=[],
)

ag = agent(llm_config=LLM_CONFIG)

_skill_map = {
    "french": french_skill,
    "spanish": spanish_skill,
    "english": english_skill,
}

if __name__ == "__main__":
    msg = input("Hi! We speak French, Spanish and English. How can I help? ") or \
          "Hello, how do I say good evening in French?"

    while True:
        route = ag.run(triage_skill, agdata(message=msg))
        skill_name = route.language.lower().strip()
        if skill_name not in _skill_map:
            skill_name = "english"

        result = ag.run(_skill_map[skill_name], agdata(message=route.message))
        print(result.response)

        try:
            msg = input("\nEnter a message (or empty to quit): ")
        except EOFError:
            break
        if not msg:
            break
