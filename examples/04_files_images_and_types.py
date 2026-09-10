"""Lesson 4: specialized field types for text, bytes, paths, and images."""

from __future__ import annotations

import base64

from agency import Agent, agbinary, agdata, agfile, agimage, agpath, agrawstring, agskill, agtype

from _common import close_sandboxes, run_example, tutorial_config


class slug(agtype):
    """A tiny custom agtype: lower-case words joined with hyphens."""

    @classmethod
    def schema_type(cls) -> str:
        return "slug"

    @classmethod
    def validate_input_value(cls, value: object) -> str | None:
        if not isinstance(value, str) or not value or value.strip("abcdefghijklmnopqrstuvwxyz-"):
            return "must contain only lower-case letters and hyphens"
        return None


def main() -> None:
    cfg, run_dir = tutorial_config("04_files_images_and_types")
    # The native harness exposes its sandbox-native read/write/bash tools
    # directly, which makes file-transfer semantics especially visible.
    learner = Agent("typed-files", agconfig=cfg, harness="native")

    transform = agskill(
        name="transform_artifacts",
        system_prompt=(
            "Use bash to run `mkdir -p /workspace/outputs`, then run `tr '[:lower:]' "
            "'[:upper:]' < DOCUMENT_PATH > /workspace/outputs/uppercase.txt`, replacing "
            "DOCUMENT_PATH with the document path in the input. Copy the binary input "
            "byte-for-byte to /workspace/outputs/copy.bin. Return both output paths and "
            "return destination unchanged."
        ),
        input_schema=agdata(document=agfile, payload=agbinary, destination=agpath),
        output_schema=agdata(uppercase=agfile, copied=agbinary, saved_at=agpath),
    )
    result = learner.run(
        transform,
        agdata(
            document="Agency moves typed data safely.",
            payload=b"\x00agency\xff",
            destination="/workspace/outputs",
        ),
    )
    assert result.uppercase.strip() == "AGENCY MOVES TYPED DATA SAFELY."
    assert result.copied == b"\x00agency\xff"
    assert result.saved_at == "/workspace/outputs"
    print(f"file result: {result.uppercase.strip()}")
    print(f"binary bytes: {len(result.copied)}")
    print(f"path result: {result.saved_at}")

    # Start a fresh conversation because a harness session is specialized by
    # the output protocol of the skill that created it.
    raw_learner = Agent("raw-text", agconfig=cfg)
    raw = agskill(
        name="raw_text",
        system_prompt="Return exactly the requested word and nothing else.",
        input_schema=agdata(prompt=agrawstring),
        output_schema=agdata(answer=agrawstring),
    )
    raw_result = raw_learner.run(raw, agdata(prompt="Return TYPE"))
    assert raw_result.answer == "TYPE"
    print(f"raw string: {raw_result.answer}")

    assert slug.validate_input_value("agency-tutorial") is None
    assert slug.validate_input_value("Not A Slug") is not None
    print(f"custom agtype schema label: {slug.schema_type()}")

    # A 1x1 PNG keeps the multimodal example self-contained. The native
    # harness receives the image as a real content block.
    pixel = run_dir / "pixel.png"
    pixel.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
    )
    vision = Agent("vision", agconfig=cfg, harness="native")
    describe = agskill(
        name="describe_image",
        system_prompt="Describe the attached image very briefly.",
        input_schema=agdata(question=str, image=agimage),
        output_schema=agdata(description=str),
    )
    description = vision.run(describe, agdata(question="What is visible?", image=str(pixel)))
    print(f"image description: {description.description}")
    print(f"artifacts: {run_dir}")

    close_sandboxes([learner, raw_learner, vision])


if __name__ == "__main__":
    run_example(main)
