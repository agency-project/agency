"""
Port of agent_patterns/parallelization.py from openai-agents-python.

Original: Run same translation agent 3x in parallel with asyncio.gather,
          then pick the best result.

Port: Uses agent.asyncio_run() — an async wrapper over run() — so the three
      forks can be awaited concurrently with asyncio.gather without blocking
      the event loop.  The sync fork fan-out pattern is shown as an alternative.
"""
import asyncio
import os
from agency import agent, agskill, agdata

LLM_CONFIG = {
    "base_url": os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:18000/v1"),
    "api_key":  os.environ.get("VLLM_API_KEY", ""),
    "model":    os.environ.get("VLLM_MODEL",   "google/gemma-4-E2B-it"),
}

translator_skill = agskill(
    name="translate",
    system_prompt="You translate the user's message to Spanish.",
    input_schema=agdata(text=str),
    output_schema=agdata(translation=str),
    replace_tools=[],
)

picker_skill = agskill(
    name="pick_best",
    system_prompt="You pick the best Spanish translation from the given options.",
    input_schema=agdata(original=str, translations=str),
    output_schema=agdata(best=str),
    replace_tools=[],
)

parent = agent(llm_config=LLM_CONFIG)


# ---------------------------------------------------------------------------
# Async version — equivalent to asyncio.gather in the original
# ---------------------------------------------------------------------------

async def run_async(msg: str) -> str:
    # All three asyncio_run() calls are submitted immediately and awaited
    # concurrently — none blocks the event loop.
    r1, r2, r3 = await asyncio.gather(
        agent(parent).asyncio_run(translator_skill, agdata(text=msg)),
        agent(parent).asyncio_run(translator_skill, agdata(text=msg)),
        agent(parent).asyncio_run(translator_skill, agdata(text=msg)),
    )
    translations = "\n".join(f"{i+1}. {r.translation}" for i, r in enumerate([r1, r2, r3]))
    print(f"\nTranslations:\n{translations}")

    best = await parent.asyncio_run(picker_skill, agdata(original=msg, translations=translations))
    return best.best


# ---------------------------------------------------------------------------
# Sync version — fork fan-out (no asyncio required)
# ---------------------------------------------------------------------------

def run_sync(msg: str) -> str:
    pending = [agent(parent).run(translator_skill, agdata(text=msg)) for _ in range(3)]
    translations = "\n".join(f"{i+1}. {r.translation}" for i, r in enumerate(pending))
    print(f"\nTranslations:\n{translations}")

    best = parent.run(picker_skill, agdata(original=msg, translations=translations))
    return best.best


if __name__ == "__main__":
    msg = input("Enter a message to translate to Spanish: ") or "Good morning!"

    print("\n--- async version (asyncio.gather) ---")
    best = asyncio.run(run_async(msg))
    print(f"\nBest translation: {best}")
