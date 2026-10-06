from agency.tandem_harness import review
from agency.tandem_harness.review import review_report

SOURCE = "\n".join(f"value_{i} = compute_something({i}) + offset_{i}" for i in range(40))


class _Llm:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def dispatch(self, model, messages, tools=None, **kwargs):
        self.calls.append((model, messages))
        return {"message": {"role": "assistant", "content": self.reply}, "usage": {"prompt_tokens": 50, "completion_tokens": 7}}


def test_code_heavy_report_gets_the_models_note(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text(SOURCE)
    llm = _Llm("mod.py: `value_3 = compute_something(3) + offset_3` — offset should be subtracted")
    note, stats = review_report("Here is mod.py:\n" + SOURCE, [str(path)], llm, "sup")
    assert note.startswith("mod.py:") and stats["code_lines"] == 40 and stats["output_tokens"] == 7
    assert llm.calls[0][0] == "sup" and SOURCE in llm.calls[0][1][0]["content"]


def test_none_reply_and_small_reports_add_nothing(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text(SOURCE)
    assert review_report(SOURCE, [str(path)], _Llm("`none`"), "sup")[0] is None
    small = "\n".join(SOURCE.split("\n")[: review.MIN_CODE_LINES - 1])
    llm = _Llm("anything")
    assert review_report(small, [str(path)], llm, "sup")[0] is None and not llm.calls
