from ..agskill import agskill
from ..agdata import agdata


class CompileReportSkill(agskill):
    """Skill that compiles a structured markdown research report from paper summaries."""

    def __init__(self, **kwargs):
        super().__init__(
            name="compile_report",
            system_prompt=(
                "You are a research report writer. "
                "Given a topic and a list of paper summaries, use the write tool to save "
                "a well-structured markdown report to the given output_path inside the sandbox. "
                "The report should have: a title, a brief introduction, "
                "one section per paper with its title, URL, and summary, "
                "and a concluding paragraph."
            ),
            input_schema=agdata(topic="str", summaries="list", output_path="str"),
            output_schema=agdata(report_path="str", paper_count="int"),
            **kwargs,
        )


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


def test_compile_report_skill_fixed_attributes():
    s = CompileReportSkill()
    assert s.name == "compile_report"
    assert "topic" in s.input_schema._data
    assert "summaries" in s.input_schema._data
    assert "output_path" in s.input_schema._data
    assert "report_path" in s.output_schema._data
    assert "paper_count" in s.output_schema._data
    assert "write tool" in s.system_prompt.lower()
    assert "markdown" in s.system_prompt.lower()


@pytest.mark.parametrize("max_retries", [0, 1, 3, 5, 10])
def test_compile_report_skill_max_retries_kwarg(max_retries):
    assert CompileReportSkill(max_retries=max_retries).max_retries == max_retries


@pytest.mark.parametrize("topic,summaries,output_path,exp_count", [
    ("attention",          ["s1"],                        "/out/report.md",   1),
    ("flash attention",    ["s1", "s2", "s3"],            "/out/report.md",   3),
    ("speculative decode", ["s1", "s2", "s3", "s4", "s5"], "/tmp/rep.md",    5),
    ("LoRA",               [],                            "/agent_out/r.md",  0),
    ("KV cache",           ["s"] * 10,                   "/workspace/r.md", 10),
])
def test_compile_report_skill_run_various_inputs(topic, summaries, output_path, exp_count):
    s = CompileReportSkill()
    with patch("openai.OpenAI") as M:
        M.return_value.chat.completions.create.return_value = \
            _make_stream(f'{{"report_path": "{output_path}", "paper_count": {exp_count}}}')
        result, hist, delta = s.run(
            _LLM,
            agdata(topic=topic, summaries=summaries, output_path=output_path),
            agdata(messages=[]), [],
        )
    assert result.report_path == output_path
    assert result.paper_count == exp_count
    assert isinstance(hist, agdata)
    assert isinstance(delta, list)


def test_compile_report_skill_run_missing_required_fields():
    s = CompileReportSkill()
    # Missing all three fields
    result, _, _ = s.run(_LLM, agdata(), agdata(messages=[]), [])
    assert result._data.get("error") is not None

    # Missing output_path
    result2, _, _ = s.run(_LLM, agdata(topic="x", summaries=["s"]), agdata(messages=[]), [])
    assert result2._data.get("error") is not None

    # Missing summaries
    result3, _, _ = s.run(_LLM, agdata(topic="x", output_path="/tmp/r.md"),
                          agdata(messages=[]), [])
    assert result3._data.get("error") is not None


def test_compile_report_skill_run_output_missing_field_triggers_retry():
    s = CompileReportSkill(max_retries=1)
    responses = iter([
        _make_stream('{"report_path": "/tmp/r.md"}'),         # missing paper_count → retry
        _make_stream('{"report_path": "/tmp/r.md", "paper_count": 2}'),
    ])
    with patch("openai.OpenAI") as M:
        M.return_value.chat.completions.create.side_effect = lambda **_: next(responses)
        result, _, _ = s.run(
            _LLM,
            agdata(topic="x", summaries=["a", "b"], output_path="/tmp/r.md"),
            agdata(messages=[]), [],
        )
    assert result.paper_count == 2


def test_compile_report_skill_run_exhausted_retries_returns_error():
    s = CompileReportSkill(max_retries=2)
    with patch("openai.OpenAI") as M:
        M.return_value.chat.completions.create.side_effect = lambda **_: \
            _make_stream('{"wrong_field": "bad"}')
        result, _, _ = s.run(
            _LLM,
            agdata(topic="x", summaries=["s"], output_path="/tmp/r.md"),
            agdata(messages=[]), [],
        )
    assert result._data.get("error") is not None
