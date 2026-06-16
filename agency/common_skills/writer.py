from ..agskill import agskill
from ..agdata import agdata


class WriterSkill(agskill):
    """Skill that writes content to a file path inside the sandbox container."""

    def __init__(self, **kwargs):
        super().__init__(
            name="writer",
            system_prompt=(
                "Write the given content to the given file path using the write tool. "
                "The path is inside the sandbox container."
            ),
            input_schema=agdata(file_path=str, content=str),
            output_schema=agdata(path=str, status=str),
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
                self.content = c; self.tool_calls = None
                self.model_extra = {}; self.reasoning_content = None
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

    def test_writer_skill_fixed_attributes():
        s = WriterSkill()
        assert s.name == "writer"
        assert s.input_schema is not None
        assert s.output_schema is not None
        assert "file_path" in s.input_schema._data
        assert "content" in s.input_schema._data
        assert "path" in s.output_schema._data
        assert "status" in s.output_schema._data
        assert "write tool" in s.system_prompt.lower()
        assert "sandbox" in s.system_prompt.lower()

    @pytest.mark.parametrize("max_retries,expected", [
        (0, 0), (1, 1), (3, 3), (5, 5), (10, 10),
    ])
    def test_writer_skill_max_retries_kwarg(max_retries, expected):
        assert WriterSkill(max_retries=max_retries).max_retries == expected

    @pytest.mark.parametrize("file_path,content,exp_path,exp_status", [
        ("/workspace/out.txt",    "hello world",      "/workspace/out.txt",    "ok"),
        ("/tmp/data.json",        '{"key": "val"}',   "/tmp/data.json",        "written"),
        ("/agent_output/rep.md",  "# Report\n\n...",  "/agent_output/rep.md",  "success"),
        ("/workspace/empty.txt",  "",                 "/workspace/empty.txt",  "ok"),
        ("/deep/nested/dir/f.py", "print('hi')",      "/deep/nested/dir/f.py", "ok"),
    ])
    def test_writer_skill_run_various_paths_and_content(file_path, content, exp_path, exp_status):
        s = WriterSkill()
        with patch("openai.OpenAI") as M:
            M.return_value.chat.completions.create.return_value = \
                _make_stream(f'{{"path": "{exp_path}", "status": "{exp_status}"}}')
            result, hist, delta, _ = s.run(_LLM, agdata(file_path=file_path, content=content),
                                        agdata(messages=[]), sandbox=None)
        assert result.path == exp_path
        assert result.status == exp_status
        assert isinstance(hist, agdata)
        assert isinstance(delta, list)

    def test_writer_skill_run_missing_required_input_fields():
        s = WriterSkill()
        result, *_ = s.run(_LLM, agdata(), agdata(messages=[]), sandbox=None)
        assert result._data.get("error") is not None
        result2, *_ = s.run(_LLM, agdata(file_path="/tmp/f.txt"), agdata(messages=[]), sandbox=None)
        assert result2._data.get("error") is not None
        result3, *_ = s.run(_LLM, agdata(content="text"), agdata(messages=[]), sandbox=None)
        assert result3._data.get("error") is not None

    def test_writer_skill_run_output_missing_field_triggers_retry():
        s = WriterSkill(max_retries=1)
        responses = iter([
            _make_stream('{"path": "/tmp/f.txt"}'),
            _make_stream('{"path": "/tmp/f.txt", "status": "ok"}'),
        ])
        with patch("openai.OpenAI") as M:
            M.return_value.chat.completions.create.side_effect = lambda **_: next(responses)
            result, *_ = s.run(_LLM, agdata(file_path="/tmp/f.txt", content="x"),
                                 agdata(messages=[]), sandbox=None)
        assert result.status == "ok"

    def test_writer_skill_history_grows_with_messages():
        s = WriterSkill()
        with patch("openai.OpenAI") as M:
            M.return_value.chat.completions.create.return_value = \
                _make_stream('{"path": "/tmp/f.txt", "status": "ok"}')
            _, hist, delta, _tok = s.run(_LLM, agdata(file_path="/tmp/f.txt", content="hi"),
                                   agdata(messages=[]), sandbox=None)
        assert len(hist.messages) >= 2
        assert len(delta) >= 2

except ImportError:
    pass
