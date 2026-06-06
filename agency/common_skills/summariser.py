from ..agskill import agskill
from ..agdata import agdata


class SummariserSkill(agskill):
    """Skill that summarises a given text in one sentence."""

    def __init__(self, **kwargs):
        super().__init__(
            name="summarise",
            system_prompt="Summarise the given text in one sentence.",
            input_schema=agdata(text=str),
            output_schema=agdata(summary=str),
            tools=[],
            **kwargs,
        )


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

    def test_summariser_skill_fixed_attributes():
        s = SummariserSkill()
        assert s.name == "summarise"
        assert s.tools == []
        assert s.input_schema is not None
        assert s.output_schema is not None
        assert "text" in s.input_schema._data
        assert "summary" in s.output_schema._data
        assert "one sentence" in s.system_prompt.lower()

    @pytest.mark.parametrize("max_retries", [0, 1, 2, 5, 10])
    def test_summariser_skill_max_retries_kwarg(max_retries):
        assert SummariserSkill(max_retries=max_retries).max_retries == max_retries

    @pytest.mark.parametrize("text,summary", [
        ("The quick brown fox jumps over the lazy dog.", "A fox jumped over a dog."),
        ("Machine learning requires large labelled datasets.", "ML needs labelled data."),
        ("Python is popular in scientific computing.", "Python is widely used in science."),
        ("Transformers changed natural language processing.", "Transformers revolutionised NLP."),
        ("The paper proposes a new attention mechanism.", "A novel attention mechanism is proposed."),
        ("Results show 5% improvement on ImageNet.", "A 5% ImageNet improvement was achieved."),
    ])
    def test_summariser_skill_run_various_inputs(text, summary):
        s = SummariserSkill()
        with patch("openai.OpenAI") as M:
            M.return_value.chat.completions.create.return_value = \
                _make_stream(f'{{"summary": "{summary}"}}')
            result, hist, delta = s.run(_LLM, agdata(text=text), agdata(messages=[]), [])
        assert result.summary == summary
        assert isinstance(hist, agdata)
        assert isinstance(delta, list)

    def test_summariser_skill_run_missing_text_input():
        s = SummariserSkill()
        result, _, _ = s.run(_LLM, agdata(), agdata(messages=[]), [])
        assert result._data.get("error") is not None

    def test_summariser_skill_run_output_missing_summary_triggers_retry():
        s = SummariserSkill(max_retries=1)
        responses = iter([
            _make_stream('{"result": "no summary key here"}'),  # wrong field → retry
            _make_stream('{"summary": "Retried summary."}'),
        ])
        with patch("openai.OpenAI") as M:
            M.return_value.chat.completions.create.side_effect = lambda **_: next(responses)
            result, _, _ = s.run(_LLM, agdata(text="Some text."), agdata(messages=[]), [])
        assert result.summary == "Retried summary."

    def test_summariser_skill_run_exhausted_retries_returns_error():
        s = SummariserSkill(max_retries=2)
        with patch("openai.OpenAI") as M:
            M.return_value.chat.completions.create.side_effect = lambda **_: \
                _make_stream('{"wrong_key": "value"}')
            result, _, _ = s.run(_LLM, agdata(text="Some text."), agdata(messages=[]), [])
        assert result._data.get("error") is not None

    def test_summariser_skill_history_grows_after_run():
        s = SummariserSkill()
        with patch("openai.OpenAI") as M:
            M.return_value.chat.completions.create.return_value = \
                _make_stream('{"summary": "Short."}')
            _, hist, delta = s.run(_LLM, agdata(text="Some text."), agdata(messages=[]), [])
        assert len(hist.messages) >= 2
        assert len(delta) >= 2

except ImportError:
    pass
