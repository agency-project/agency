import io
import re

import fitz  # pymupdf
import html2text
import httpx

from ..agskill import agskill
from ..agdata import agdata, _fmt_exc
from ..agtool import agtool

_MAX_CHARS = 32_000


def _arxiv_html_url(url: str) -> str:
    """Convert any arxiv URL form to its HTML version URL."""
    m = re.search(r"arxiv\.org/(?:abs|pdf|html)/([^\s/?#]+)", url)
    if not m:
        return url
    paper_id = m.group(1).removesuffix("")
    return f"https://arxiv.org/html/{paper_id}"


def _arxiv_pdf_url(url: str) -> str:
    """Convert any arxiv URL form to its PDF URL."""
    m = re.search(r"arxiv\.org/(?:abs|pdf|html)/([^\s/?#]+)", url)
    if not m:
        return url
    return f"https://arxiv.org/pdf/{m.group(1)}"


def fetch_full_paper_text(url: str) -> str:
    """Fetch the complete text of an arxiv paper as a single string.

    Tries the arxiv HTML page first (cleaner text). Falls back to the PDF
    via pymupdf if HTML is unavailable (older papers, 404s, etc.).
    Returns an empty string only if both attempts fail.
    """
    # ── Try HTML first ────────────────────────────────────────────────────────
    html_url = _arxiv_html_url(url)
    try:
        resp = httpx.get(html_url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        converter = html2text.HTML2Text()
        converter.ignore_links = True
        converter.ignore_images = True
        converter.body_width = 0
        text = converter.handle(resp.text)
        lines = text.splitlines()
        start = next((i for i, ln in enumerate(lines) if ln.startswith("# ")), 0)
        result = "\n".join(lines[start:]).strip()
        if result:
            return result
    except Exception:
        pass

    # ── Fall back to PDF ──────────────────────────────────────────────────────
    pdf_url = _arxiv_pdf_url(url)
    try:
        resp = httpx.get(pdf_url, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        doc = fitz.open(stream=io.BytesIO(resp.content), filetype="pdf")
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages).strip()
    except Exception:
        return ""


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

    def _fetch_paper(self, arg: agdata) -> agdata:
        url = str(arg.url)
        html_url = _arxiv_html_url(url)
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


# ---------------------------------------------------------------------------
# Tests (only defined when pytest is installed — it is a dev dependency)
# ---------------------------------------------------------------------------

try:
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
        return [_Chunk(content), _Chunk(final=True)]

    _LLM = {"api_key": "k", "model": "m"}

    @pytest.mark.parametrize("input_url,expected_html_url", [
        ("https://arxiv.org/abs/2301.12345",        "https://arxiv.org/html/2301.12345"),
        ("https://arxiv.org/pdf/2301.12345",    "https://arxiv.org/html/2301.12345"),
        ("https://arxiv.org/html/2301.12345",       "https://arxiv.org/html/2301.12345"),
        ("https://arxiv.org/abs/2301.12345v2",      "https://arxiv.org/html/2301.12345v2"),
        ("https://arxiv.org/abs/1706.03762",        "https://arxiv.org/html/1706.03762"),
        ("https://arxiv.org/pdf/1810.04805v3",  "https://arxiv.org/html/1810.04805v3"),
    ])
    def test_arxiv_html_url_conversion(input_url, expected_html_url):
        assert _arxiv_html_url(input_url) == expected_html_url

    @pytest.mark.parametrize("non_arxiv_url", [
        "https://example.com/paper",
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
        assert result.offset == 0

    def test_fetch_paper_short_content_not_flagged_truncated(monkeypatch):
        html = "<html><body><h1># Title</h1><p>Short paper.</p></body></html>"
        _mock_http(monkeypatch, html)
        s = SummarisePaperSkill()
        result = s._fetch_paper(agdata(url="https://arxiv.org/abs/2301.12345"))
        assert len(result.text) < _MAX_CHARS
        assert result.truncated is False

    def test_fetch_paper_offset_returns_next_chunk(monkeypatch):
        # Build content slightly longer than two chunks
        body = "x" * (_MAX_CHARS + 100)
        long_html = f"<html><body><h1># T</h1><p>{body}</p></body></html>"
        _mock_http(monkeypatch, long_html)
        s = SummarisePaperSkill()
        r0 = s._fetch_paper(agdata(url="https://arxiv.org/abs/2301.12345"))
        assert r0.truncated is True
        assert len(r0.text) == _MAX_CHARS

        r1 = s._fetch_paper(agdata(url="https://arxiv.org/abs/2301.12345", offset=_MAX_CHARS))
        assert r1.truncated is False
        assert len(r1.text) > 0
        assert r1.offset == _MAX_CHARS

    def test_fetch_paper_offset_zero_same_as_default(monkeypatch):
        html = "<html><body><h1># Title</h1><p>Short paper.</p></body></html>"
        _mock_http(monkeypatch, html)
        s = SummarisePaperSkill()
        r_default = s._fetch_paper(agdata(url="https://arxiv.org/abs/2301.12345"))
        r_zero    = s._fetch_paper(agdata(url="https://arxiv.org/abs/2301.12345", offset=0))
        assert r_default.text == r_zero.text

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
        assert len(s.replace_tools) == 1
        assert s.replace_tools[0].name == "fetch_paper"
        assert s.fetch_paper is s.replace_tools[0]
        assert "fetch_paper" in s.system_prompt
        assert "MUST" in s.system_prompt
        assert "abstract alone" in s.system_prompt
        assert "truncated" in s.system_prompt
        assert "offset" in s.replace_tools[0].params["properties"]
        assert "title" in s.input_schema._data
        assert "url" in s.input_schema._data
        assert "abstract" in s.input_schema._data
        assert "summary" in s.output_schema._data

    @pytest.mark.parametrize("max_retries", [0, 1, 2, 5, 10])
    def test_summarise_paper_skill_max_retries_kwarg(max_retries):
        assert SummarisePaperSkill(max_retries=max_retries).max_retries == max_retries

    def test_summarise_paper_skill_run_missing_required_fields():
        s = SummarisePaperSkill()
        result, _, _ = s.run(_LLM, agdata(), agdata(messages=[]), sandbox=None)
        assert result._data.get("error") is not None

        result2, _, _ = s.run(_LLM, agdata(title="T", abstract="A"), agdata(messages=[]), sandbox=None)
        assert result2._data.get("error") is not None

        result3, _, _ = s.run(_LLM, agdata(url="https://arxiv.org/abs/1234.5678", abstract="A"),
                              agdata(messages=[]), sandbox=None)
        assert result3._data.get("error") is not None

except ImportError:
    pass
