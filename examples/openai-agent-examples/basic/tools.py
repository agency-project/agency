"""
Port of basic/tools.py from openai-agents-python.

Original: Agent with a get_weather function tool.
"""
import os
from agency import agent, agskill, agdata
from agency.agtool import agtool

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


def _get_weather_fn(arg: agdata) -> agdata:
    city = str(arg.city)
    print(f"[debug] get_weather called for {city}")
    return agdata(city=city, temperature_range="14-20C", conditions="Sunny with wind.")


get_weather = agtool(
    name="get_weather",
    description="Get the current weather information for a specified city.",
    fn=_get_weather_fn,
    params={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The city to get the weather for"},
        },
        "required": ["city"],
    },
    need_sandbox=False,
)

weather_skill = agskill(
    name="weather",
    system_prompt="You are a helpful agent.",
    input_schema=agdata(question=str),
    output_schema=agdata(response=str),
    add_tools=[get_weather],
)

ag = agent(llm_config=LLM_CONFIG)

if __name__ == "__main__":
    result = ag.run(weather_skill, agdata(question="What's the weather in Tokyo?"))
    print(result.response)
    # The weather in Tokyo is sunny with temperatures between 14-20°C.
