"""
Parallel execution example.

Demonstrates the two natural parallelism patterns enabled by a single run() method:

  1. SequentialChainTeam  — sequential calls on one agent are automatically ordered
                            through the history chain; results are pending agdata.

  2. ForkFanoutTeam       — agent(parent) creates a local copy; each copy's run()
                            fires immediately and returns a pending agdata.  Multiple
                            forks run concurrently without any explicit thread management.

Run:
    uv run python examples/parallel_exec.py
"""
import os
import time
from pathlib import Path

from agency import agent, agdata, agteam
from agency.common_skills import WriterSkill, SummariserSkill


def _make_run_dir(name: str) -> Path:
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(__file__).parent.parent / "runs" / f"{ts}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


LLM_CONFIG = {
    "base_url": os.environ.get("VLLM_BASE_URL", "https://kimi.js-park.info:18000/v1"),
    "api_key":  os.environ.get("VLLM_API_KEY", ""),
    "model":    os.environ.get("VLLM_MODEL",   "moonshotai/Kimi-K2.6"),
}


# ---------------------------------------------------------------------------
# Team 1: sequential chain
# ---------------------------------------------------------------------------

class SequentialChainTeam(agteam):
    """One agent runs two file-write tasks sequentially via the history chain."""

    llm_config = LLM_CONFIG

    def setup(self) -> None:
        self.writer = WriterSkill()
        self.agent = agent(agskills=[self.writer])

    def run(self) -> None:
        print("=" * 60)
        print("Pattern 1: sequential chain on one agent")
        print("=" * 60)

        t0 = time.perf_counter()
        r1 = self.agent.run("writer", agdata(file_path="/workspace/out.txt", content="first write"))
        r2 = self.agent.run("writer", agdata(file_path="/workspace/out.txt", content="second write"))
        elapsed = time.perf_counter() - t0

        print(f"  r1 status={r1.status!r}  r2 status={r2.status!r}")
        print(f"  total history: {len(self.agent.history.messages)} messages  elapsed {elapsed:.2f}s")
        print()


# ---------------------------------------------------------------------------
# Team 2: fork fan-out
# ---------------------------------------------------------------------------

class ForkFanoutTeam(agteam):
    """Forks one agent per text; all summaries run concurrently."""

    llm_config = LLM_CONFIG

    # Default texts — override at construction time via texts=[ ... ]
    _default_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require large amounts of labelled training data.",
        "Python is widely used in scientific computing and data analysis.",
    ]

    def setup(self) -> None:
        self.summariser = SummariserSkill()
        self.parent = agent(agskills=[self.summariser])

    def run(self) -> None:
        print("=" * 60)
        print("Pattern 2: fork fan-out — agent(parent).run() per input")
        print("=" * 60)

        texts = getattr(self, "texts", self._default_texts)

        t0 = time.perf_counter()
        pending = [
            agent(self.parent).run("summarise", agdata(text=t))
            for t in texts
        ]
        elapsed_submit = time.perf_counter() - t0

        for i, r in enumerate(pending):
            print(f"  text {i}: {r.summary!r}")

        elapsed_total = time.perf_counter() - t0
        print(f"  submitted in {elapsed_submit:.3f}s   total {elapsed_total:.2f}s")
        print(f"  parent history unchanged: {len(self.parent.history.messages)} messages")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from agency import AgError, agUI

    run_dir = _make_run_dir("parallel_exec")
    agent.log_dir    = run_dir / "logs"
    agent.output_dir = run_dir / "agent_output"

    def _script() -> None:
        print(f"Endpoint : {LLM_CONFIG['base_url']}")
        print(f"Model    : {LLM_CONFIG['model']}\n")
        print(f"Run dir  : {run_dir}\n")
        try:
            SequentialChainTeam().run()
            ForkFanoutTeam().run()
        except AgError as e:
            print(f"\nERROR: {e}")

    agUI.run(_script)
