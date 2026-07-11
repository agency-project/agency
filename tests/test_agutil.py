"""Tests for agutil utility helpers."""
from agency.agutil import _strip_thinking, _extract_thinking


def test_strip_thinking_removes_think_tag():
    assert _strip_thinking("<think>reasoning</think>answer") == "answer"


def test_strip_thinking_removes_thinking_tag():
    assert _strip_thinking("<thinking>deep thought</thinking>result") == "result"


def test_strip_thinking_no_tag_unchanged():
    assert _strip_thinking("plain answer") == "plain answer"


def test_extract_thinking_returns_content():
    assert _extract_thinking("<think>my reasoning</think>answer") == "my reasoning"


def test_extract_thinking_no_tag_returns_empty():
    assert _extract_thinking("no thinking here") == ""


def test_extract_thinking_multiple_blocks():
    text = "<think>first</think>middle<think>second</think>end"
    result = _extract_thinking(text)
    assert "first" in result and "second" in result
