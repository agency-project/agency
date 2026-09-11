"""
Parallel execution example.

Demonstrates the two natural parallelism patterns enabled by a single run() method:

  1. SequentialChainTeam  — sequential calls on one agent are automatically ordered
                            through the context chain; results are Invocations.

  2. ForkFanoutTeam       — agent.fork(parent) creates a local copy; each copy's run()
                            fires immediately and returns an Invocation.  Multiple
                            forks run concurrently without any explicit thread management.

Run:
    uv run python examples/parallel_exec.py
"""

import os
import time

from agency import agent, agdata, agskill, agteam
from agency.configs.agconfig import agconfig, llmconfig
from agency.agtype import agpath

# See ../README.md for Anthropic or Bedrock agconfig examples.
cfg = agconfig(
    llmconfig(
        provider="OpenAI_Compatible",
        base_url=os.environ["LLM_BASE_URL"],
        model=os.environ["LLM_MODEL"],
        api_key=os.environ["LLM_API_KEY"],
    )
)


continuation_skill = agskill(
    name="continuation",
    prompt="Continue the given text in one sentence.",
    input_schema=agdata(text=str),
    output_schema=agdata(continuation=str),
)

writer_skill = agskill(
    name="writer",
    prompt=(
        "Write the given content to the given file path using the write tool. "
        "The path is inside the sandbox container."
    ),
    input_schema=agdata(file_path=agpath, content=str),
    output_schema=agdata(path=agpath),
)

# ---------------------------------------------------------------------------
# Team 1: sequential chain
# ---------------------------------------------------------------------------


class SequentialChainTeam(agteam):
    """One agent runs two file-write tasks sequentially via the history chain."""

    def setup(self) -> None:
        self.writer = writer_skill
        self.agent = agent(agconfig=cfg)

    def run(self) -> None:
        print("=" * 60)
        print("Pattern 1: sequential chain on one agent")
        print("=" * 60)

        t0 = time.perf_counter()
        self.agent.run(self.writer, agdata(file_path="/workspace/out.txt", content="first write"))
        self.agent.run(self.writer, agdata(file_path="/workspace/out.txt", content="second write"))
        elapsed = time.perf_counter() - t0

        print(
            f"  total history: {len(self.agent.history.messages)} messages  elapsed {elapsed:.2f}s"
        )
        print()


# ---------------------------------------------------------------------------
# Team 2: fork fan-out
# ---------------------------------------------------------------------------


class ForkFanoutTeam(agteam):
    """Forks one agent per text; all summaries run concurrently."""

    # Default texts — override at construction time via texts=[ ... ]
    _default_texts = [
        "The quick brown fox jumps over ",
        "Machine learning models require large amounts of ",
        "Python is widely used in ",
    ]

    def setup(self) -> None:
        self.continuation = continuation_skill
        self.parent = agent(agconfig=cfg, name="parent")

    def run(self) -> None:
        print("=" * 60)
        print("Pattern 2: fork fan-out — agent.fork(parent).run() per input")
        print("=" * 60)

        texts = getattr(self, "texts", self._default_texts)

        t0 = time.perf_counter()
        pending = [agent.fork(self.parent).run(self.continuation, agdata(text=t)) for t in texts]
        elapsed_submit = time.perf_counter() - t0

        for i, r in enumerate(pending):
            print(f"  text {i}: {r.continuation!r}")

        elapsed_total = time.perf_counter() - t0
        print(f"  submitted in {elapsed_submit:.3f}s   total {elapsed_total:.2f}s")
        print(f"  parent history unchanged: {len(self.parent.history.messages)} messages")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from agency import AgError, agsync

    def _script() -> None:
        try:
            seq_team = SequentialChainTeam()
            seq_team.run()
            agsync(seq_team)
            fork_team = ForkFanoutTeam()
            fork_team.run()
            agsync(fork_team)
        except AgError as e:
            print(f"\nERROR: {e}")

    from agency.observability.agwebui import agwebui

    agwebui.run(_script, port=8006)
