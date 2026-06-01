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

Run:
    uv run python examples/paper_crawler.py
    uv run python examples/paper_crawler.py "speculative decoding"
    MAX_PAPERS=6 uv run python examples/paper_crawler.py "flash attention"
"""
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _run_dir import make_run_dir

import httpx

from src import agent, agskill, agdata
from src.agtool import agtool
from src.tools import write

LLM_CONFIG = {
    "base_url": os.environ.get("VLLM_BASE_URL", "https://kimi.js-park.info:18000/v1"),
    "api_key":  os.environ.get("VLLM_API_KEY", ""),
    "model":    os.environ.get("VLLM_MODEL",     "moonshotai/Kimi-K2.6"),
}
MAX_PAPERS = int(os.environ.get("MAX_PAPERS", "10"))

# ---------------------------------------------------------------------------
# Custom tool: search arxiv
# ---------------------------------------------------------------------------

def _search_arxiv_fn(arg: agdata) -> agdata:
    query = str(arg.query).replace(" ", "+")  # type: ignore[arg-type]
    max_results = int(getattr(arg, "max_results", MAX_PAPERS))
    url = (
        f"https://export.arxiv.org/api/query"
        f"?search_query=all:{query}&start=0&max_results={max_results}&sortBy=relevance"
    )
    try:
        resp = httpx.get(url, timeout=20, follow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        return agdata(error=str(e), papers=[])

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    papers = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.find("atom:title", ns).text or "").strip().replace("\n", " ")  # type: ignore[union-attr]
        abstract = (entry.find("atom:summary", ns).text or "").strip()[:600]  # type: ignore[union-attr]
        link = (entry.find("atom:id", ns).text or "").strip()  # type: ignore[union-attr]
        papers.append({"title": title, "url": link, "abstract": abstract})
    return agdata(papers=papers, count=len(papers))


search_arxiv = agtool(
    name="search_arxiv",
    description="Search arxiv for papers. Returns title, URL, and abstract for each result.",
    fn=_search_arxiv_fn,
    params={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query string"},
            "max_results": {"type": "integer", "description": f"Max results (default {MAX_PAPERS})"},
        },
        "required": ["query"],
    },
)

# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

find_papers_skill = agskill(
    name="find_papers",
    system_prompt=(
        "You are a research assistant. "
        "You MUST call the search_arxiv tool to find recent relevant papers on the given topic. "
        "Do NOT skip the tool call or invent papers. "
        "Return the full list of papers exactly as the tool provided them."
    ),
    input_schema=agdata(topic="str"),
    output_schema=agdata(papers="list", count="int"),
    output_validator=lambda r: (
        ["papers list is empty — you MUST call the search_arxiv tool before responding"]
        if not r._data.get("papers") else []
    ),
)

summarise_paper_skill = agskill(
    name="summarise_paper",
    system_prompt=(
        "You are a research paper summariser. "
        "Given the title, URL, and abstract of a paper, write a concise"
        "technical summary that captures the core contribution, method, results, limitations and conclusions."
    ),
    input_schema=agdata(title="str", url="str", abstract="str"),
    output_schema=agdata(summary="str"),
    tools=[],
)

compile_report_skill = agskill(
    name="compile_report",
    system_prompt=(
        "You are a research report writer. "
        "Given a topic and a list of paper summaries, use the write tool to save "
        "a well-structured markdown report to the given output_path. "
        "The report should have: a title, a brief introduction, "
        "one section per paper with its title, URL, and summary, "
        "and a concluding paragraph."
    ),
    input_schema=agdata(topic="str", summaries="list", output_path="str"),
    output_schema=agdata(report_path="str", paper_count="int"),
)

# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

def run(topic: str = "KV cache quantization", output_dir: str = "."):
    output_path = str(Path(output_dir) / "report.md")
    agent.log_dir = Path(output_dir) / "logs"

    main_agent = agent(
        llm_config=LLM_CONFIG,
        agskills=[find_papers_skill, summarise_paper_skill, compile_report_skill],
        tools=[search_arxiv, write],
    )

    print(f"Endpoint : {LLM_CONFIG['base_url']}")
    print(f"Model    : {LLM_CONFIG['model']}")
    print(f"Topic    : {topic!r}")
    print(f"Output   : {output_path}")
    print()

    # Step 1: find papers — AgError raised automatically if skill fails
    print("Step 1 — searching for papers...")
    papers = main_agent.run("find_papers", agdata(topic=topic)).papers
    if not papers:
        print("  No papers found — try a different topic or re-run.")
        return
    print(f"  found {len(papers)} papers:")
    for p in papers:
        print(f"    • {p['title'][:70]}")
    print()

    # Step 2: one local agent copy per paper — all run() calls fire in parallel
    print("Step 2 — submitting parallel summarisation tasks...")
    summaries = [
        agent(main_agent).run(
            "summarise_paper",
            agdata(title=p["title"], url=p["url"], abstract=p["abstract"]),
        )
        for p in papers   # each is a pending agdata; all running concurrently
    ]
    for i, p in enumerate(papers):
        print(f"  [{i}] {p['title'][:60]}...")
    print()

    # Step 3: compile report — pending agdata list resolved automatically
    print("Step 3 — compiling markdown report...")
    r3 = main_agent.run(
        "compile_report",
        agdata(topic=topic, summaries=summaries, output_path=output_path),
    )
    print(f"  report written → {r3.report_path}  ({r3.paper_count} papers)")
    if Path(output_path).exists():
        print(f"\n--- report preview ---\n{Path(output_path).read_text()[:400]}\n...")

    print(f"\nMain agent history: {len(main_agent.history.messages)} messages total")


if __name__ == "__main__":
    from src import AgError
    topic = " ".join(sys.argv[1:]) or "KV cache quantization"
    run_dir = make_run_dir("paper_crawler")
    print(f"Run dir  : {run_dir}\n")
    try:
        run(topic=topic, output_dir=str(run_dir))
    except AgError as e:
        print(f"\nERROR: {e}")
