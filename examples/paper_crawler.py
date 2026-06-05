"""
paper_crawler.py — Research paper crawler with parallel summarisation.

Workflow:
  1. find_papers skill  — agent searches arxiv for the topic; returns a list of
                          {title, url, abstract} dicts as agdata.
  2. agent(main_agent)  — one local copy per paper; each copy's run() fires
                          immediately and returns a pending agdata.  All summaries
                          run concurrently with no explicit thread management.
  3. compile_report     — the list of pending agdata is passed directly to the
                          original agent; each is resolved automatically before
                          the skill starts.

The report is written to /agent_output/<uuid>/report.md inside the container
and appears on the host at <run_dir>/agent_output/<uuid>/report.md.

Run:
    uv run python examples/paper_crawler.py
    uv run python examples/paper_crawler.py "speculative decoding"
    MAX_PAPERS=6 uv run python examples/paper_crawler.py "flash attention"
"""
import os
import sys
from pathlib import Path

from agency import agent, agdata, agteam, agsync
from agency.common_skills import FindPapersSkill, SummarisePaperSkill, CompileReportSkill


def _make_run_dir(name: str) -> Path:
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(__file__).parent.parent / "runs" / f"{ts}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


LLM_CONFIG = {
    "base_url": os.environ.get("VLLM_BASE_URL", "https://kimi.js-park.info:18000/v1"),
    "api_key":  os.environ.get("VLLM_API_KEY", ""),
    "model":    os.environ.get("VLLM_MODEL",   "Qwen/Qwen3.5-397B-A17B-FP8"),
}
MAX_PAPERS = int(os.environ.get("MAX_PAPERS", "6"))


# ---------------------------------------------------------------------------
# Team definition
# ---------------------------------------------------------------------------

class PaperCrawlerTeam(agteam):
    """Self-contained paper-crawling team.

    Parameters
    ----------
    topic : str
        Research topic to search on arxiv.
    max_papers : int
        Maximum number of papers to fetch (default 16).
    llm_config : dict | None
        LLM endpoint config; falls back to the class-level default.

    Usage::

        team = PaperCrawlerTeam(topic="KV cache quantization")
        team.run()
    """

    llm_config = LLM_CONFIG

    def setup(self) -> None:
        max_p = getattr(self, "max_papers", MAX_PAPERS)

        self.find_papers     = FindPapersSkill(max_papers=max_p)
        self.summarise_paper = SummarisePaperSkill()
        self.compile_report  = CompileReportSkill()

        self.main_agent = agent()

    def run(self) -> agdata:
        topic       = getattr(self, "topic", "machine learning")
        output_path = f"{self.main_agent.container_output_path}/report.md"
        host_report = self.main_agent.output_path / "report.md"

        print(f"Agent    : {self.main_agent.agname}")
        print(f"Topic    : {topic!r}")
        print(f"Output   : {host_report}")
        print()

        print("Step 1 — searching for papers...")
        papers = self.main_agent.run(self.find_papers, agdata(topic=topic)).papers
        if not papers:
            print("  No papers found — try a different topic or re-run.")
            return agdata(error="no papers found")
        print(f"  found {len(papers)} papers:")
        for p in papers:
            print(f"    • {p['title'][:70]}")
        print()

        print("Step 2 — submitting parallel summarisation tasks...")
        summaries = [
            agent(self.main_agent).run(
                self.summarise_paper,
                agdata(title=p["title"], url=p["url"], abstract=p["abstract"]),
            )
            for p in papers
        ]
        for i, p in enumerate(papers):
            print(f"  [{i}] {p['title'][:60]}...")
        print()

        print("Step 3 — compiling markdown report...")
        result = self.main_agent.run(
            self.compile_report,
            agdata(topic=topic, summaries=summaries, output_path=output_path),
        )
        print(f"  report written → {result.report_path}  ({result.paper_count} papers)")

        if host_report.exists():
            print(f"\n--- report preview ---\n{host_report.read_text()[:400]}\n...")

        print(f"\nMain agent history: {len(self.main_agent.history.messages)} messages total")
        return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from agency import AgError, agUI

    topic   = " ".join(sys.argv[1:]) or "KV cache quantization"
    run_dir = _make_run_dir("paper_crawler")

    agent.log_dir    = run_dir / "logs"
    agent.output_dir = run_dir / "agent_output"

    def _script() -> None:
        print(f"Endpoint : {LLM_CONFIG['base_url']}")
        print(f"Model    : {LLM_CONFIG['model']}")
        print(f"Run dir  : {run_dir}\n")
        try:
            topics = [topic] if topic != "KV cache quantization" else [
                "KV cache quantization",
                "flash attention",
                "speculative decoding",
            ]
            teams = [PaperCrawlerTeam(topic=t) for t in topics]
            pending = [t.run() for t in teams]  # all start immediately
            agsync(teams)
            for t, r in zip(topics, pending):
                print(f"\n[{t}] report → {r.report_path}  ({r.paper_count} papers)")
        except AgError as e:
            print(f"\nERROR: {e}")

    agUI.run(_script)
