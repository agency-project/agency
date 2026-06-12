"""
Port of agent_patterns/agents_as_tools.py from openai-agents-python.

Original: Orchestrator calls translator sub-agents as tools.
Each translator agent is invoked synchronously inside a tool function.
"""
import os
from agency import agent, agskill, agdata
from agency.agtool import agtool

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


def _make_translator_tool(language: str, instructions: str) -> agtool:
    """Create a tool that spins up a sub-agent to translate text."""
    translate_skill = agskill(
        name="translate",
        system_prompt=instructions,
        input_schema=agdata(text=str),
        output_schema=agdata(translation=str),
        replace_tools=[],
    )

    def _fn(arg: agdata) -> agdata:
        sub_ag = agent(llm_config=LLM_CONFIG)
        result = sub_ag.run(translate_skill, agdata(text=str(arg.text)))
        return agdata(translation=result.translation)

    return agtool(
        name=f"translate_to_{language.lower()}",
        description=f"Translate the user's message to {language}.",
        fn=_fn,
        params={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to translate"}},
            "required": ["text"],
        },
        need_sandbox=False,
    )


translate_to_spanish = _make_translator_tool("Spanish", "You translate the user's message to Spanish.")
translate_to_french  = _make_translator_tool("French",  "You translate the user's message to French.")
translate_to_italian = _make_translator_tool("Italian", "You translate the user's message to Italian.")

orchestrator_skill = agskill(
    name="orchestrate",
    system_prompt=(
        "You are a translation agent. Use the tools given to you to translate. "
        "If asked for multiple translations, call the relevant tools in order. "
        "Never translate on your own — always use the provided tools."
    ),
    input_schema=agdata(message=str),
    output_schema=agdata(result=str),
    add_tools=[translate_to_spanish, translate_to_french, translate_to_italian],
)

synthesizer_skill = agskill(
    name="synthesize",
    system_prompt="Inspect the translations, correct them if needed, and produce a final concatenated response.",
    input_schema=agdata(translations=str),
    output_schema=agdata(final_response=str),
    replace_tools=[],
)

ag = agent(llm_config=LLM_CONFIG)

if __name__ == "__main__":
    msg = input("What would you like translated, and to which languages? ") or \
          "Translate 'Hello, world!' to French and Spanish."

    orchestrated = ag.run(orchestrator_skill, agdata(message=msg))
    print(f"Translations: {orchestrated.result}")

    final = ag.run(synthesizer_skill, agdata(translations=orchestrated.result))
    print(f"\nFinal response:\n{final.final_response}")
