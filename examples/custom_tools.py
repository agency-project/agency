"""
custom_tools.py — Research paper crawler with parallel summarisation.

Workflow:
  1. find_papers skill  — agent searches arxiv for the topic; returns a list of
                          {title, url, abstract} dicts as agdata.
  2. agent(main_agent)  — one local copy per paper; each copy's run() fires
                          immediately and returns a pending agdata.  All summaries
                          run concurrently with no explicit thread management.
  3. compile_report     — the list of pending agdata is passed directly to the
                          original agent; each is resolved automatically before
                          the skill starts.  The skill returns the report as an
                          agfile — the framework reads it back from the sandbox
                          and the host writes it to disk with no shared mounts.

Run:
    uv run python examples/custom_tools.py
    uv run python examples/custom_tools.py "speculative decoding"
    MAX_PAPERS=6 uv run python examples/custom_tools.py "flash attention"
"""
import io
import os
import re
from pathlib import Path

import fitz
import html2text
import httpx

from agency import agent, agdata, agfile, agskill, agteam, agsync, agtool
from agency.agutil import format_exception as _fmt_exc


LLM_CONFIG = {
    "base_url":             os.environ.get("VLLM_BASE_URL", ""),
    "api_key":              os.environ.get("VLLM_API_KEY",  ""),
    "model":                "",
}
MAX_PAPERS = int(os.environ.get("MAX_PAPERS", "4"))
_MAX_CHARS = 32_000


class FindPapersSkill(agskill):
    def __init__(self, max_papers: int = 10, **kwargs):
        self.max_papers = max_papers

        search_papers = agtool(
            name="search_papers",
            description="Search Hugging Face Papers for AI research papers. Returns title, URL, and abstract for each result.",
            fn=self._search,
            run_in_subprocess=False,
            params={
                "type": "object",
                "properties": {
                    "query":       {"type": "string",  "description": "Search query string"},
                    "max_results": {"type": "integer", "description": f"Max results (default {max_papers})"},
                },
                "required": ["query"],
            },
        )

        super().__init__(
            name="find_papers",
            system_prompt=(
                "You are a research assistant. "
                "You MUST call the search_papers tool to find recent relevant papers on the given topic. "
                "Do NOT skip the tool call or invent papers. "
                "Return the full list of papers exactly as the tool provided them."
            ),
            input_schema=agdata(topic=str),
            output_schema=agdata(papers=[{"title": str, "url": str, "abstract": str}], count=int),

            replace_tools=[search_papers],
            **kwargs,
        )
        self.search_papers = search_papers

    def _search(self, arg: agdata) -> agdata:
        query = str(arg.query)
        n = int(getattr(arg, "max_results", self.max_papers))
        try:
            resp = httpx.get(
                "https://huggingface.co/api/papers/search",
                params={"q": query, "limit": n},
                timeout=20,
                follow_redirects=True,
            )
            resp.raise_for_status()
        except Exception as e:
            return agdata(error=_fmt_exc(e), papers=[])
        try:
            data = resp.json()
        except Exception as e:
            return agdata(error=_fmt_exc(e), papers=[])
        papers = []
        for item in data:
            paper    = item.get("paper", item) if isinstance(item, dict) else {}
            title    = paper.get("title", "").strip()
            abstract = paper.get("summary", "")[:600]
            arxiv_id = paper.get("id", "")
            url      = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
            if title:
                papers.append({"title": title, "url": url, "abstract": abstract})
        return agdata(papers=papers, count=len(papers))

class SummarisePaperSkill(agskill):
    def __init__(self, **kwargs):

        fetch_paper = agtool(
            name="fetch_paper",
            description=(
                "Fetch the full text of an arxiv paper given its URL. "
                "Returns up to 32000 characters at a time. "
                "If truncated=true in the result, call again with offset incremented by 32000 to read the next chunk."
                "The arxiv URLs follow these formats:"
                "Abstract HTML page: https://arxiv.org/abs/2601.12345"
                "PDF page: https://arxiv.org/pdf/2601.12345"
                "HTML page: https://arxiv.org/html/2601.12345"
                "HTML page is not always available for older papers."
            ),
            fn=self._fetch_paper,
            params={
                "type": "object",
                "properties": {
                    "url":    {"type": "string",  "description": "The arxiv paper URL (abs, pdf, or html form)"},
                    "offset": {"type": "integer", "description": "Character offset to start reading from (default 0)"},
                },
                "required": ["url"],
            },
        )
    
        super().__init__(
            name="summarise_paper",
            system_prompt=(
                "You are a research paper summariser. "
                "You MUST call the fetch_paper tool to retrieve the full paper text before summarising. "
                "Do NOT summarise from the abstract alone. "
                "If the result has truncated=true, keep calling fetch_paper with increasing offset values "
                "(0, 32000, 64000, …) until truncated=false, then write your summary. "
                "After reading the full paper, write a concise technical summary that captures "
                "the core contribution, method, results, limitations and conclusions."
            ),
            input_schema=agdata(title=str, url=str, abstract=str),
            output_schema=agdata(summary=str),
            replace_tools=[fetch_paper],
            **kwargs,
        )
        self.fetch_paper = fetch_paper

    def _arxiv_html_url(self, url: str) -> str:
        m = re.search(r"arxiv\.org/(?:abs|pdf|html)/([^\s/?#]+)", url)
        if not m:
            return url
        return f"https://arxiv.org/html/{m.group(1)}"

    def _fetch_paper(self, arg: agdata) -> agdata:
        url = str(arg.url)
        html_url = self._arxiv_html_url(url)
        offset = int(getattr(arg, "offset", 0) or 0)
        try:
            resp = httpx.get(html_url, timeout=30, follow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            return agdata(error=_fmt_exc(e), text="", url=html_url)
        converter = html2text.HTML2Text()
        converter.ignore_links = True
        converter.ignore_images = True
        converter.body_width = 0
        text = converter.handle(resp.text)
        lines = text.splitlines()
        start = next((i for i, l in enumerate(lines) if l.startswith("# ")), 0)
        full = "\n".join(lines[start:])
        chunk = full[offset : offset + _MAX_CHARS]
        truncated = (offset + _MAX_CHARS) < len(full)
        return agdata(text=chunk, url=html_url, offset=offset, truncated=truncated)


class CompileReportSkill(agskill):
    def __init__(self, **kwargs):
        super().__init__(
            name="compile_report",
            system_prompt=(
                "You are a research report writer. "
                "Given a topic and a list of paper summaries, use the write tool to save "
                "a well-structured markdown report to /workspace/report.md inside the sandbox. "
                "The report should have: a title, a brief introduction, "
                "one section per paper with its title, URL, and summary, "
                "and a concluding paragraph. "
                "Return /workspace/report.md as the report field."
            ),
            input_schema=agdata(topic=str, summaries=list),
            output_schema=agdata(report=agfile, paper_count=int),
            **kwargs,
        )


def _make_run_dir(name: str) -> Path:
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(__file__).parent.parent / "runs" / f"{ts}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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

    def run(self, output_dir: "Path | None" = None) -> agdata:
        topic = getattr(self, "topic", "machine learning")

        print(f"Agent    : {self.main_agent.agname}")
        print(f"Topic    : {topic!r}")
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
            agent.fork(self.main_agent).run(
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
            agdata(topic=topic, summaries=summaries),
        )
        print(f"  compiled  ({result.paper_count} papers, {len(result.report)} chars)")

        if output_dir is not None:
            slug = topic.lower().replace(" ", "_")[:40]
            host_path = Path(output_dir) / f"{slug}_report.md"
            host_path.write_text(result.report)
            print(f"  saved     → {host_path}")
            print(f"\n--- report preview ---\n{result.report[:400]}\n...")

        print(f"\nMain agent history: {len(self.main_agent.history.messages)} messages total")
        return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from agency import AgError
    import sys

    topic   = " ".join(sys.argv[1:]) or "KV cache quantization"
    run_dir = _make_run_dir("custom_tools")

    agent.log_dir = run_dir / "logs"
    reports_dir   = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    def _script() -> None:
        print(f"Endpoint : {LLM_CONFIG['base_url']}")
        print(f"Run dir  : {run_dir}\n")
        try:
            topics = [topic] if topic != "KV cache quantization" else [
                "KV cache quantization",
                "speculative decoding",
            ]
            teams = [PaperCrawlerTeam(topic=t) for t in topics]
            pending = [t.run(output_dir=reports_dir) for t in teams]  # all start immediately
            agsync(teams)
            for t, r in zip(topics, pending):
                print(f"\n[{t}] {r.paper_count} papers  ({len(r.report)} chars)")
        except AgError as e:
            print(f"\nERROR: {e}")

    from agency.agwebui import agwebui
    agwebui.run(_script)
