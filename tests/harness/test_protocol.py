from __future__ import annotations

import json
from dataclasses import asdict

from agency.harness.protocol import HarnessAttemptRequest, HarnessAttemptResult, PromptPayload


def test_harness_attempt_request_round_trips_through_json():
    request = HarnessAttemptRequest(
        harness="claude_code",
        max_steps=20,
        resume_session_id="session-122",
        prior_session_blob_b64="cHJpb3I=",
        prompt=PromptPayload(
            system_instruction="system",
            user_content="fix the bug",
        ),
    )

    decoded = json.loads(json.dumps(asdict(request)))

    assert decoded == {
        "prompt": {
            "system_instruction": "system",
            "user_content": "fix the bug",
            "output_instruction": None,
        },
        "harness": "claude_code",
        "max_steps": 20,
        "resume_session_id": "session-122",
        "prior_session_blob_b64": "cHJpb3I=",
    }
    prompt = PromptPayload(**decoded.pop("prompt"))
    assert HarnessAttemptRequest(prompt=prompt, **decoded) == request


def test_multimodal_prompt_payload_round_trips_through_json():
    payload = PromptPayload(
        system_instruction="inspect the image",
        user_content=[
            {"type": "text", "text": "What is shown?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ],
        output_instruction="Return JSON",
    )

    decoded = json.loads(json.dumps(asdict(payload)))

    assert PromptPayload(**decoded) == payload


def test_harness_attempt_result_round_trips_through_json():
    result = HarnessAttemptResult(
        ok=True,
        final_text="done",
        input_tokens=100,
        output_tokens=20,
        session_id="session-123",
        session_blob_b64="c2Vzc2lvbiBzdGF0ZQ==",
    )

    decoded = json.loads(json.dumps(asdict(result)))

    assert decoded == {
        "ok": True,
        "final_text": "done",
        "input_tokens": 100,
        "output_tokens": 20,
        "session_id": "session-123",
        "session_blob_b64": "c2Vzc2lvbiBzdGF0ZQ==",
        "error_message": "",
    }
    assert HarnessAttemptResult(**decoded) == result
