import httpx

from ..agskill import agskill
from ..agdata import agdata
from ..agtool import agtool


class FindPapersSkill(agskill):
    """Skill that searches Hugging Face Papers for papers on a given topic.

    Parameters
    ----------
    max_papers : int
        Maximum number of results to fetch per query (default 16).
    """

    def __init__(self, max_papers: int = 16, **kwargs):
        self.max_papers = max_papers
        search_papers = agtool(
            name="search_papers",
            description="Search Hugging Face Papers for AI research papers. Returns title, URL, and abstract for each result.",
            fn=self._search,
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
            input_schema=agdata(topic="str"),
            output_schema=agdata(papers="list", count="int"),
            output_validator=self._validate_output,
            tools=[search_papers],
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
            return agdata(error=str(e), papers=[])

        try:
            data = resp.json()
        except Exception as e:
            return agdata(error=f"JSON parse error: {e}", papers=[])

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

    def _validate_output(self, r: agdata) -> list[str]:
        if not r._data.get("papers"):
            return ["papers list is empty — you MUST call the search_papers tool before responding"]
        return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import pytest


def _mock_hf_response(monkeypatch, items):
    """Patch httpx.get to return a JSON response with *items*."""
    mock_resp = type("R", (), {
        "raise_for_status": lambda self: None,
        "json":             lambda self: items,
    })()
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
    return mock_resp


def _paper(title, summary="abstract text.", arxiv_id="2301.00001"):
    return {"paper": {"title": title, "summary": summary, "id": arxiv_id}}


_LLM = {"api_key": "k", "model": "m"}


def test_find_papers_skill_fixed_attributes():
    s = FindPapersSkill()
    assert s.name == "find_papers"
    assert len(s.tools) == 1
    assert s.tools[0].name == "search_papers"
    assert s.search_papers is s.tools[0]
    assert "topic" in s.input_schema._data
    assert "papers" in s.output_schema._data
    assert "count" in s.output_schema._data
    assert s.output_validator is not None
    assert "search_papers" in s.system_prompt
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
    result._data["papers"] = papers if papers else []
    errors = s._validate_output(result)
    if expect_errors:
        assert len(errors) >= 1
        assert "search_papers" in errors[0]
    else:
        assert errors == []


def test_find_papers_search_tool_parses_multiple_entries(monkeypatch):
    items = [
        _paper("Attention Is All You Need",  "Transformer paper.",    "1706.03762"),
        _paper("BERT: Pre-training of DNNs", "BERT paper.",           "1810.04805"),
        _paper("GPT-3",                      "Language model paper.", "2005.14165"),
        _paper("LoRA: Low-Rank Adaptation",  "Fine-tuning paper.",    "2106.09685"),
    ]
    _mock_hf_response(monkeypatch, items)
    s = FindPapersSkill()
    result = s._search(agdata(query="transformers"))
    assert result.count == 4
    assert len(result.papers) == 4
    assert result.papers[0]["title"] == "Attention Is All You Need"
    assert result.papers[1]["url"] == "https://arxiv.org/abs/1810.04805"
    assert result.papers[2]["abstract"] == "Language model paper."
    assert result.papers[3]["title"] == "LoRA: Low-Rank Adaptation"


def test_find_papers_search_tool_builds_arxiv_url(monkeypatch):
    _mock_hf_response(monkeypatch, [_paper("My Paper", "abstract", "2301.12345")])
    s = FindPapersSkill()
    result = s._search(agdata(query="test"))
    assert result.papers[0]["url"] == "https://arxiv.org/abs/2301.12345"


def test_find_papers_search_tool_passes_query_and_limit(monkeypatch):
    captured = {}
    def _fake_get(_url, params=None, **_kw):
        captured["params"] = params
        return type("R", (), {"raise_for_status": lambda self: None, "json": lambda self: []})()
    monkeypatch.setattr(httpx, "get", _fake_get)

    s = FindPapersSkill(max_papers=7)
    s._search(agdata(query="flash attention mechanism"))
    assert captured["params"]["q"] == "flash attention mechanism"
    assert captured["params"]["limit"] == 7


def test_find_papers_search_tool_respects_max_results_override(monkeypatch):
    captured = {}
    def _fake_get(_url, params=None, **_kw):
        captured["params"] = params
        return type("R", (), {"raise_for_status": lambda self: None, "json": lambda self: []})()
    monkeypatch.setattr(httpx, "get", _fake_get)

    s = FindPapersSkill(max_papers=16)
    s._search(agdata(query="attention", max_results=3))
    assert captured["params"]["limit"] == 3


def test_find_papers_search_tool_abstract_truncated_to_600(monkeypatch):
    long_abstract = "x" * 1000
    _mock_hf_response(monkeypatch, [_paper("T", long_abstract, "2301.00001")])
    s = FindPapersSkill()
    result = s._search(agdata(query="x"))
    assert len(result.papers[0]["abstract"]) == 600


def test_find_papers_skips_entries_without_title(monkeypatch):
    items = [
        {"paper": {"title": "",    "summary": "abstract", "id": "2301.00001"}},
        {"paper": {"title": "Real Paper", "summary": "abstract", "id": "2301.00002"}},
    ]
    _mock_hf_response(monkeypatch, items)
    s = FindPapersSkill()
    result = s._search(agdata(query="test"))
    assert result.count == 1
    assert result.papers[0]["title"] == "Real Paper"


def test_find_papers_empty_response_returns_zero_count(monkeypatch):
    _mock_hf_response(monkeypatch, [])
    s = FindPapersSkill()
    result = s._search(agdata(query="obscure topic with no results"))
    assert result.count == 0
    assert result.papers == []


@pytest.mark.parametrize("exc_type,exc_msg", [
    (httpx.ConnectError,     "connection refused"),
    (httpx.TimeoutException, "timed out"),
    (httpx.HTTPStatusError,  "429"),
    (ValueError,             "unexpected value"),
])
def test_find_papers_search_tool_handles_errors(monkeypatch, exc_type, exc_msg):
    if exc_type is httpx.HTTPStatusError:
        def _raise(*_a, **_kw):
            raise exc_type(exc_msg, request=None, response=None)
    else:
        def _raise(*_a, **_kw):
            raise exc_type(exc_msg)
    monkeypatch.setattr(httpx, "get", _raise)

    s = FindPapersSkill()
    result = s._search(agdata(query="attention"))
    assert result._data.get("error") is not None
    assert result._data.get("papers") == []
