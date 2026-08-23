"""Tests for agimage — multimodal image input support."""

import base64
import json
from unittest.mock import MagicMock

import pytest

from agency.agdata import agdata
from agency.agschema import agschema
from agency.agtype import agtype, agimage
from agency.agskill import agskill


# ---------------------------------------------------------------------------
# agimage class
# ---------------------------------------------------------------------------


def test_agimage_is_agtype_subclass():
    assert issubclass(agimage, agtype)


def test_agimage_schema_type():
    assert agimage.schema_type() == "image"


def test_agimage_needs_no_sandbox():
    assert agimage.needs_sandbox() is False


def test_agimage_extra_input_prompt_mentions_field_name():
    prompt = agimage.extra_input_prompt("photo")
    assert "photo" in prompt
    assert "image" in prompt.lower()


def test_agimage_extra_output_prompt_empty():
    # Images are input-only; no output prompt expected
    assert agimage.extra_output_prompt("photo", "skill") == ""


# ---------------------------------------------------------------------------
# agimage.prepare — URL and data URL passthrough
# ---------------------------------------------------------------------------


def test_agimage_prepare_http_url_passthrough():
    val, paths = agimage.prepare("http://example.com/img.jpg", None, "sk", "photo")
    assert val == "http://example.com/img.jpg"
    assert paths == []


def test_agimage_prepare_https_url_passthrough():
    val, paths = agimage.prepare("https://example.com/img.png", None, "sk", "photo")
    assert val == "https://example.com/img.png"
    assert paths == []


def test_agimage_prepare_data_url_passthrough():
    data_url = "data:image/png;base64,abc123"
    val, paths = agimage.prepare(data_url, None, "sk", "photo")
    assert val == data_url
    assert paths == []


def test_agimage_prepare_non_string_passthrough():
    val, paths = agimage.prepare(None, None, "sk", "photo")
    assert val is None
    assert paths == []


# ---------------------------------------------------------------------------
# agimage.prepare — local file encoding
# ---------------------------------------------------------------------------


def test_agimage_prepare_local_file_base64_encodes(tmp_path):
    img_file = tmp_path / "test.jpg"
    raw = b"\xff\xd8\xff\xe0fake jpeg bytes"
    img_file.write_bytes(raw)

    val, paths = agimage.prepare(str(img_file), None, "sk", "photo")

    assert paths == []  # no sandbox paths written
    assert val.startswith("data:image/jpeg;base64,")
    encoded = val[len("data:image/jpeg;base64,") :]
    assert base64.b64decode(encoded) == raw


def test_agimage_prepare_local_png_uses_correct_mime(tmp_path):
    img_file = tmp_path / "test.png"
    img_file.write_bytes(b"\x89PNG fake")

    val, paths = agimage.prepare(str(img_file), None, "sk", "photo")

    assert val.startswith("data:image/png;base64,")


def test_agimage_prepare_missing_file_raises(tmp_path):
    missing = str(tmp_path / "no_such_file.jpg")
    with pytest.raises(FileNotFoundError):
        agimage.prepare(missing, None, "sk", "photo")


# ---------------------------------------------------------------------------
# agdata serialization
# ---------------------------------------------------------------------------


def test_agdata_serializes_agimage_as_image():
    d = agdata(photo=agimage)
    assert json.loads(d.to_json()) == {"photo": "image"}


def test_agdata_serializes_list_agimage():
    d = agdata(frames=list[agimage])
    parsed = json.loads(d.to_json())
    assert parsed == {"frames": "list[image]"}


# ---------------------------------------------------------------------------
# agskill._build_system_prompt — agimage fields included
# ---------------------------------------------------------------------------


def test_system_prompt_includes_agimage_input_instructions():
    sk = agskill(
        "describe",
        "Describe the image.",
        input_schema=agdata(question=str, photo=agimage),
    )
    prompt = sk._build_system_prompt()
    assert "photo" in prompt
    assert "image" in prompt.lower()


def test_system_prompt_includes_list_agimage_instructions():
    sk = agskill(
        "compare",
        "Compare images.",
        input_schema=agdata(question=str, frames=list[agimage]),
    )
    prompt = sk._build_system_prompt()
    assert "frames" in prompt


def test_system_prompt_agimage_no_sandbox_warning():
    # agimage is not file-backed so the "File-backed fields" banner should
    # still appear (the banner is shared), but the prompt for agimage itself
    # should reference "image", not "read tool"
    sk = agskill(
        "describe",
        "Describe.",
        input_schema=agdata(photo=agimage),
    )
    prompt = sk._build_system_prompt()
    assert "photo" in prompt
    assert "read tool" not in prompt


# ---------------------------------------------------------------------------
# agskill._build_user_content — plain vs multimodal
# ---------------------------------------------------------------------------


def test_build_user_content_no_images_returns_plain_string():
    sk = agskill("t", "", input_schema=agdata(question=str))
    inp = agdata(question="hello")
    content = sk._build_user_content(inp)
    assert isinstance(content, str)
    assert "hello" in content


def test_build_user_content_no_schema_returns_plain_string():
    sk = agskill("t", "")
    inp = agdata(question="hello")
    content = sk._build_user_content(inp)
    assert isinstance(content, str)


def test_build_user_content_single_image_returns_array():
    sk = agskill("t", "", input_schema=agdata(question=str, photo=agimage))
    inp = agdata(question="what is this?", photo="https://example.com/img.jpg")
    content = sk._build_user_content(inp)
    assert isinstance(content, list)
    types = [part["type"] for part in content]
    assert "text" in types
    assert "image_url" in types


def test_build_user_content_image_url_correct():
    sk = agskill("t", "", input_schema=agdata(photo=agimage))
    inp = agdata(photo="https://example.com/img.jpg")
    content = sk._build_user_content(inp)
    image_parts = [p for p in content if p["type"] == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"] == "https://example.com/img.jpg"


def test_build_user_content_image_field_replaced_with_placeholder():
    sk = agskill("t", "", input_schema=agdata(question=str, photo=agimage))
    inp = agdata(question="what?", photo="data:image/jpeg;base64,AAAA")
    content = sk._build_user_content(inp)
    text_part = next(p for p in content if p["type"] == "text")
    parsed = json.loads(text_part["text"].split("\n", 1)[1])
    assert parsed["photo"] == "[image attached]"
    assert parsed["question"] == "what?"


def test_build_user_content_list_images_all_injected():
    sk = agskill("t", "", input_schema=agdata(frames=list[agimage]))
    urls = ["https://example.com/a.jpg", "https://example.com/b.jpg"]
    inp = agdata(frames=urls)
    content = sk._build_user_content(inp)
    image_parts = [p for p in content if p["type"] == "image_url"]
    assert len(image_parts) == 2
    assert image_parts[0]["image_url"]["url"] == urls[0]
    assert image_parts[1]["image_url"]["url"] == urls[1]


def test_build_user_content_list_images_placeholder_shows_count():
    sk = agskill("t", "", input_schema=agdata(frames=list[agimage]))
    inp = agdata(frames=["https://a.com/1.jpg", "https://a.com/2.jpg"])
    content = sk._build_user_content(inp)
    text_part = next(p for p in content if p["type"] == "text")
    parsed = json.loads(text_part["text"].split("\n", 1)[1])
    assert "2" in parsed["frames"]


def test_build_user_content_mixed_fields_non_image_preserved():
    sk = agskill("t", "", input_schema=agdata(label=str, photo=agimage))
    inp = agdata(label="cat", photo="https://example.com/cat.jpg")
    content = sk._build_user_content(inp)
    text_part = next(p for p in content if p["type"] == "text")
    parsed = json.loads(text_part["text"].split("\n", 1)[1])
    assert parsed["label"] == "cat"


# ---------------------------------------------------------------------------
# _prepare_agtype_inputs — list[agimage]
# ---------------------------------------------------------------------------


def test_prepare_agtype_inputs_list_agimage_encodes_each(tmp_path):
    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    img_a.write_bytes(b"fake_a")
    img_b.write_bytes(b"fake_b")

    inp = agdata(frames=[str(img_a), str(img_b)])
    schema = agdata(frames=list[agimage])
    paths, _ = agschema(schema).prepare_inputs_in_sandbox(inp, MagicMock(), "sk")

    assert paths == []  # agimage doesn't write sandbox paths
    vals = inp._data["frames"]
    assert isinstance(vals, list)
    assert len(vals) == 2
    assert vals[0].startswith("data:image/jpeg;base64,")
    assert vals[1].startswith("data:image/jpeg;base64,")
    assert base64.b64decode(vals[0].split(",", 1)[1]) == b"fake_a"
    assert base64.b64decode(vals[1].split(",", 1)[1]) == b"fake_b"


def test_prepare_agtype_inputs_single_agimage_encodes(tmp_path):
    img = tmp_path / "photo.png"
    img.write_bytes(b"fake_png")

    inp = agdata(photo=str(img))
    schema = agdata(photo=agimage)
    agschema(schema).prepare_inputs_in_sandbox(inp, MagicMock(), "sk")

    assert inp._data["photo"].startswith("data:image/png;base64,")


def test_prepare_agtype_inputs_url_agimage_not_modified():
    inp = agdata(photo="https://example.com/img.jpg")
    schema = agdata(photo=agimage)
    agschema(schema).prepare_inputs_in_sandbox(inp, MagicMock(), "sk")
    assert inp._data["photo"] == "https://example.com/img.jpg"
