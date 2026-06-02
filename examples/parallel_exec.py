"""
Parallel execution example.

Demonstrates the two natural parallelism patterns enabled by a single run() method:

  1. Same-agent chain  — sequential calls on one agent are automatically ordered
                         through the history chain; results are pending agdata.

  2. Fork fan-out      — agent(parent) creates a local copy; each copy's run()
                         fires immediately and returns a pending agdata.  Multiple
                         forks run concurrently without any explicit thread management.

Run:
    uv run python examples/parallel_exec.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _run_dir import make_run_dir

from src import agent, agskill, agdata

LLM_CONFIG = {
    "base_url": os.environ.get("VLLM_BASE_URL", "https://kimi.js-park.info:18000/v1"),
    "api_key":  os.environ.get("VLLM_API_KEY", ""),
    "model":    os.environ.get("VLLM_MODEL",     "moonshotai/Kimi-K2.6"),
}


def demo_sequential_chain(run_dir: Path):
    """One agent runs two tasks sequentially via the history chain."""
    print("=" * 60)
    print("Pattern 1: sequential chain on one agent")
    print("=" * 60)

    writer = agskill(
        name="writer",
        system_prompt=(
            "Write the given content to the given file path using the write tool. "
            "The path is inside the sandbox container."
        ),
        input_schema=agdata(file_path="str", content="str"),
        output_schema=agdata(path="str", status="str"),
    )

    # No tools= — uses default sandboxed tool list
    ag = agent(llm_config=LLM_CONFIG, agskills=[writer])

    t0 = time.perf_counter()
    r1 = ag.run("writer", agdata(file_path="/workspace/out.txt", content="first write"))
    r2 = ag.run("writer", agdata(file_path="/workspace/out.txt", content="second write"))

    elapsed = time.perf_counter() - t0
    print(f"  r1 status={r1.status!r}  r2 status={r2.status!r}")
    print(f"  total history: {len(ag.history.messages)} messages  elapsed {elapsed:.2f}s")
    print()


def demo_fork_fanout():
    """Three local copies summarise texts concurrently via agent(parent).run()."""
    print("=" * 60)
    print("Pattern 2: fork fan-out — agent(parent).run() per input")
    print("=" * 60)

    summariser = agskill(
        name="summarise",
        system_prompt="Summarise the given text in one sentence.",
        input_schema=agdata(text="str"),
        output_schema=agdata(summary="str"),
        tools=[],
    )

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require large amounts of labelled training data.",
        "Python is widely used in scientific computing and data analysis.",
    ]

    parent = agent(llm_config=LLM_CONFIG, agskills=[summariser])

    t0 = time.perf_counter()
    pending = [agent(parent).run("summarise", agdata(text=t)) for t in texts]
    elapsed_submit = time.perf_counter() - t0

    for i, r in enumerate(pending):
        print(f"  text {i}: {r.summary!r}")

    elapsed_total = time.perf_counter() - t0
    print(f"  submitted in {elapsed_submit:.3f}s   total {elapsed_total:.2f}s")
    print(f"  parent history unchanged: {len(parent.history.messages)} messages")
    print()


if __name__ == "__main__":
    from src import AgError
    print(f"Endpoint : {LLM_CONFIG['base_url']}")
    print(f"Model    : {LLM_CONFIG['model']}\n")
    run_dir = make_run_dir("parallel_exec")
    print(f"Run dir  : {run_dir}\n")
    agent.log_dir = run_dir / "logs"
    try:
        demo_sequential_chain(run_dir)
        demo_fork_fanout()
    except AgError as e:
        print(f"\nERROR: {e}")
