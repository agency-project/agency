import re

import html2text
import httpx

from ..agskill import agskill
from ..agdata import agdata
from ..agtool import agtool

_MAX_CHARS = 32_000


def _arxiv_html_url(url: str) -> str:
    """Convert any arxiv URL form to its HTML version URL."""
    m = re.search(r"arxiv\.org/(?:abs|pdf|html)/([^\s/?#]+)", url)
    if not m:
        return url
    paper_id = m.group(1).removesuffix(".pdf")
    return f"https://arxiv.org/html/{paper_id}"


class SummarisePaperSkill(agskill):
    """Skill that fetches and summarises a full arxiv paper.

    Uses the ``fetch_paper`` tool to retrieve the HTML version of the paper
    before writing its summary.
    """

    def __init__(self, **kwargs):
        fetch_paper = agtool(
            name="fetch_paper",
            description=(
                "Fetch the full text of an arxiv paper given its URL. "
                "Returns the paper content as plain text (up to 32 000 characters)."
            ),
            fn=self._fetch_paper,
            params={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The arxiv paper URL (abs, pdf, or html form)"},
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
                "After reading the full paper, write a concise technical summary that captures "
                "the core contribution, method, results, limitations and conclusions."
            ),
            input_schema=agdata(title="str", url="str", abstract="str"),
            output_schema=agdata(summary="str"),
            tools=[fetch_paper],
            **kwargs,
        )
        self.fetch_paper = fetch_paper

    def _fetch_paper(self, arg: agdata) -> agdata:
        url = str(arg.url)
        html_url = _arxiv_html_url(url)
        try:
            resp = httpx.get(html_url, timeout=30, follow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            return agdata(error=str(e), text="", url=html_url)

        converter = html2text.HTML2Text()
        converter.ignore_links = True
        converter.ignore_images = True
        converter.body_width = 0
        text = converter.handle(resp.text)

        lines = text.splitlines()
        start = next((i for i, l in enumerate(lines) if l.startswith("#")), 0)
        trimmed = "\n".join(lines[start:])[:_MAX_CHARS]

        return agdata(text=trimmed, url=html_url, truncated=len(trimmed) == _MAX_CHARS)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import pytest
from unittest.mock import patch


def _make_stream(content: str):
    class _Delta:
        def __init__(self, c):
            self.content = c
            self.tool_calls = None
            self.model_extra = {}
            self.reasoning_content = None
    class _Choice:
        def __init__(self, c): self.delta = _Delta(c)
    class _Usage:
        prompt_tokens = 5
    class _Chunk:
        def __init__(self, c=None, final=False):
            self.choices = [_Choice(c)] if not final else []
            self.usage = _Usage() if final else None
    return iter([_Chunk(content), _Chunk(final=True)])


_LLM = {"api_key": "k", "model": "m"}


@pytest.mark.parametrize("input_url,expected_html_url", [
    ("https://arxiv.org/abs/2301.12345",        "https://arxiv.org/html/2301.12345"),
    ("https://arxiv.org/pdf/2301.12345.pdf",    "https://arxiv.org/html/2301.12345"),
    ("https://arxiv.org/html/2301.12345",       "https://arxiv.org/html/2301.12345"),
    ("https://arxiv.org/abs/2301.12345v2",      "https://arxiv.org/html/2301.12345v2"),
    ("https://arxiv.org/abs/1706.03762",        "https://arxiv.org/html/1706.03762"),
    ("https://arxiv.org/pdf/1810.04805v3.pdf",  "https://arxiv.org/html/1810.04805v3"),
])
def test_arxiv_html_url_conversion(input_url, expected_html_url):
    assert _arxiv_html_url(input_url) == expected_html_url


@pytest.mark.parametrize("non_arxiv_url", [
    "https://example.com/paper.pdf",
    "https://openreview.net/forum?id=abc123",
    "https://proceedings.mlr.press/v97/paper.html",
    "https://github.com/user/repo",
])
def test_arxiv_html_url_non_arxiv_passthrough(non_arxiv_url):
    assert _arxiv_html_url(non_arxiv_url) == non_arxiv_url


def _mock_http(monkeypatch, text):
    mock_resp = type("R", (), {
        "text": text,
        "raise_for_status": lambda self: None,
    })()
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)


def test_fetch_paper_returns_text_and_correct_url(monkeypatch):
    html = "<html><body><h1>Introduction</h1><p>This paper proposes X.</p></body></html>"
    _mock_http(monkeypatch, html)
    s = SummarisePaperSkill()
    result = s._fetch_paper(agdata(url="https://arxiv.org/abs/2301.12345"))
    assert len(result.text) > 0
    assert result.url == "https://arxiv.org/html/2301.12345"
    assert result.truncated is False


def test_fetch_paper_skips_lines_before_first_heading(monkeypatch):
    html = (
        "<html><body>"
        "<p>Navigation bar text to skip.</p>"
        "<p>More boilerplate.</p>"
        "<h1># Introduction</h1>"
        "<p>Actual content starts here.</p>"
        "</body></html>"
    )
    _mock_http(monkeypatch, html)
    s = SummarisePaperSkill()
    result = s._fetch_paper(agdata(url="https://arxiv.org/abs/2301.12345"))
    assert "Navigation bar" not in result.text
    assert "Introduction" in result.text


def test_fetch_paper_truncates_and_flags_long_content(monkeypatch):
    long_html = "<html><body><h1># Title</h1><p>" + ("word " * 10_000) + "</p></body></html>"
    _mock_http(monkeypatch, long_html)
    s = SummarisePaperSkill()
    result = s._fetch_paper(agdata(url="https://arxiv.org/abs/2301.12345"))
    assert len(result.text) <= _MAX_CHARS
    assert result.truncated is True


def test_fetch_paper_short_content_not_flagged_truncated(monkeypatch):
    html = "<html><body><h1># Title</h1><p>Short paper.</p></body></html>"
    _mock_http(monkeypatch, html)
    s = SummarisePaperSkill()
    result = s._fetch_paper(agdata(url="https://arxiv.org/abs/2301.12345"))
    assert len(result.text) < _MAX_CHARS
    assert result.truncated is False


@pytest.mark.parametrize("exc_type,exc_args", [
    (httpx.ConnectError,    ("connection refused",)),
    (httpx.TimeoutException, ("timed out",)),
    (httpx.HTTPStatusError,  ("404", None, None)),
    (ValueError,             ("unexpected value",)),
])
def test_fetch_paper_handles_errors(monkeypatch, exc_type, exc_args):
    def _raise(*a, **kw):
        raise exc_type(*exc_args)
    monkeypatch.setattr(httpx, "get", _raise)
    s = SummarisePaperSkill()
    result = s._fetch_paper(agdata(url="https://arxiv.org/abs/2301.12345"))
    assert result._data.get("error") is not None
    assert result._data.get("text") == ""


def test_summarise_paper_skill_fixed_attributes():
    s = SummarisePaperSkill()
    assert s.name == "summarise_paper"
    assert len(s.tools) == 1
    assert s.tools[0].name == "fetch_paper"
    assert s.fetch_paper is s.tools[0]
    assert "fetch_paper" in s.system_prompt
    assert "MUST" in s.system_prompt
    assert "abstract alone" in s.system_prompt
    assert "title" in s.input_schema._data
    assert "url" in s.input_schema._data
    assert "abstract" in s.input_schema._data
    assert "summary" in s.output_schema._data


@pytest.mark.parametrize("max_retries", [0, 1, 2, 5, 10])
def test_summarise_paper_skill_max_retries_kwarg(max_retries):
    assert SummarisePaperSkill(max_retries=max_retries).max_retries == max_retries


def test_summarise_paper_skill_run_missing_required_fields():
    s = SummarisePaperSkill()
    result, _, _ = s.run(_LLM, agdata(), agdata(messages=[]), [])
    assert result._data.get("error") is not None

    result2, _, _ = s.run(_LLM, agdata(title="T", abstract="A"), agdata(messages=[]), [])
    assert result2._data.get("error") is not None

    result3, _, _ = s.run(_LLM, agdata(url="https://arxiv.org/abs/1234.5678", abstract="A"),
                          agdata(messages=[]), [])
    assert result3._data.get("error") is not None
