import xml.etree.ElementTree as ET

import httpx

from ..agskill import agskill
from ..agdata import agdata
from ..agtool import agtool


class FindPapersSkill(agskill):
    """Skill that searches arxiv for papers on a given topic.

    Parameters
    ----------
    max_papers : int
        Maximum number of results to fetch per query (default 16).
    """

    def __init__(self, max_papers: int = 16, **kwargs):
        max_p = max_papers

        def _search_arxiv_fn(arg: agdata) -> agdata:
            query = str(arg.query).replace(" ", "+")  # type: ignore[arg-type]
            n = int(getattr(arg, "max_results", max_p))
            url = (
                f"https://export.arxiv.org/api/query"
                f"?search_query=all:{query}&start=0&max_results={n}&sortBy=relevance"
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
                title    = (entry.find("atom:title",   ns).text or "").strip().replace("\n", " ")  # type: ignore[union-attr]
                abstract = (entry.find("atom:summary", ns).text or "").strip()[:600]               # type: ignore[union-attr]
                link     = (entry.find("atom:id",      ns).text or "").strip()                     # type: ignore[union-attr]
                papers.append({"title": title, "url": link, "abstract": abstract})
            return agdata(papers=papers, count=len(papers))

        search_arxiv = agtool(
            name="search_arxiv",
            description="Search arxiv for papers. Returns title, URL, and abstract for each result.",
            fn=_search_arxiv_fn,
            params={
                "type": "object",
                "properties": {
                    "query":       {"type": "string",  "description": "Search query string"},
                    "max_results": {"type": "integer", "description": f"Max results (default {max_p})"},
                },
                "required": ["query"],
            },
        )

        super().__init__(
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
            tools=[search_arxiv],
            **kwargs,
        )
        self.search_arxiv = search_arxiv


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import pytest
from unittest.mock import patch


def _make_atom(entries):
    """Build an Atom feed string from a list of (title, summary, id) tuples."""
    items = "".join(
        f"  <entry>"
        f"<title>{t}</title>"
        f"<summary>{s}</summary>"
        f"<id>{i}</id>"
        f"</entry>\n"
        for t, s, i in entries
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">\n'
        + items + "</feed>"
    )


def _mock_http(monkeypatch, text):
    mock_resp = type("R", (), {
        "text": text,
        "raise_for_status": lambda self: None,
    })()
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
    return mock_resp


_LLM = {"api_key": "k", "model": "m"}


def test_find_papers_skill_fixed_attributes():
    s = FindPapersSkill()
    assert s.name == "find_papers"
    assert len(s.tools) == 1
    assert s.tools[0].name == "search_arxiv"
    assert s.search_arxiv is s.tools[0]
    assert "topic" in s.input_schema._data
    assert "papers" in s.output_schema._data
    assert "count" in s.output_schema._data
    assert s.output_validator is not None
    assert "search_arxiv" in s.system_prompt
    assert "MUST" in s.system_prompt


@pytest.mark.parametrize("max_papers", [1, 4, 8, 16, 32, 100])
def test_find_papers_max_papers_reflected_in_tool_description(max_papers):
    s = FindPapersSkill(max_papers=max_papers)
    desc = s.tools[0].params["properties"]["max_results"]["description"]
    assert str(max_papers) in desc


@pytest.mark.parametrize("max_retries", [0, 1, 3, 5, 10])
def test_find_papers_max_retries_kwarg(max_retries):
    assert FindPapersSkill(max_retries=max_retries).max_retries == max_retries


@pytest.mark.parametrize("papers,expect_errors", [
    ([], True),
    (None, True),
    ([{"title": "x"}], False),
    ([{"title": "a"}, {"title": "b"}], False),
    ([{"title": "x"} for _ in range(10)], False),
])
def test_find_papers_output_validator(papers, expect_errors):
    s = FindPapersSkill()
    result = agdata(papers=papers or [], count=len(papers) if papers else 0)
    # Directly set _data to simulate empty-list case even when papers=[]
    result._data["papers"] = papers if papers else []
    errors = s.output_validator(result)
    if expect_errors:
        assert len(errors) >= 1
        assert "search_arxiv" in errors[0]
    else:
        assert errors == []


def test_find_papers_search_tool_parses_multiple_entries(monkeypatch):
    entries = [
        ("Attention Is All You Need",   "Transformer paper.",     "https://arxiv.org/abs/1706.03762"),
        ("BERT: Pre-training of DNNs",  "BERT paper.",            "https://arxiv.org/abs/1810.04805"),
        ("GPT-3",                       "Language model paper.",  "https://arxiv.org/abs/2005.14165"),
        ("LoRA: Low-Rank Adaptation",   "Fine-tuning paper.",     "https://arxiv.org/abs/2106.09685"),
    ]
    _mock_http(monkeypatch, _make_atom(entries))
    s = FindPapersSkill()
    result = s.search_arxiv(agdata(query="transformers"))
    assert result.count == 4
    assert len(result.papers) == 4
    assert result.papers[0]["title"] == "Attention Is All You Need"
    assert result.papers[1]["url"] == "https://arxiv.org/abs/1810.04805"
    assert result.papers[2]["abstract"] == "Language model paper."
    assert result.papers[3]["title"] == "LoRA: Low-Rank Adaptation"


def test_find_papers_search_tool_encodes_spaces_in_query(monkeypatch):
    captured = {}
    def _fake_get(url, **kw):
        captured["url"] = url
        return type("R", (), {"text": _make_atom([]), "raise_for_status": lambda self: None})()
    monkeypatch.setattr(httpx, "get", _fake_get)

    s = FindPapersSkill()
    s.search_arxiv(agdata(query="flash attention mechanism"))
    assert "flash+attention+mechanism" in captured["url"]
    assert " " not in captured["url"]


def test_find_papers_search_tool_respects_max_results_override(monkeypatch):
    captured = {}
    def _fake_get(url, **kw):
        captured["url"] = url
        return type("R", (), {"text": _make_atom([]), "raise_for_status": lambda self: None})()
    monkeypatch.setattr(httpx, "get", _fake_get)

    s = FindPapersSkill(max_papers=16)
    s.search_arxiv(agdata(query="attention", max_results=3))
    assert "max_results=3" in captured["url"]


def test_find_papers_search_tool_abstract_truncated_to_600(monkeypatch):
    long_abstract = "x" * 1000
    _mock_http(monkeypatch, _make_atom([("T", long_abstract, "https://arxiv.org/abs/1234.5678")]))
    s = FindPapersSkill()
    result = s.search_arxiv(agdata(query="x"))
    assert len(result.papers[0]["abstract"]) == 600


@pytest.mark.parametrize("exc_type,exc_msg", [
    (httpx.ConnectError,   "connection refused"),
    (httpx.TimeoutException, "timed out"),
    (httpx.HTTPStatusError,  "404"),
    (ValueError,             "unexpected value"),
])
def test_find_papers_search_tool_handles_errors(monkeypatch, exc_type, exc_msg):
    if exc_type is httpx.HTTPStatusError:
        def _raise(*a, **kw):
            raise exc_type(exc_msg, request=None, response=None)
    else:
        def _raise(*a, **kw):
            raise exc_type(exc_msg)
    monkeypatch.setattr(httpx, "get", _raise)

    s = FindPapersSkill()
    result = s.search_arxiv(agdata(query="attention"))
    assert result._data.get("error") is not None
    assert result._data.get("papers") == []


def test_find_papers_empty_feed_returns_zero_count(monkeypatch):
    _mock_http(monkeypatch, _make_atom([]))
    s = FindPapersSkill()
    result = s.search_arxiv(agdata(query="obscure topic with no results"))
    assert result.count == 0
    assert result.papers == []
