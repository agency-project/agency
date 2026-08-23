from __future__ import annotations

from dataclasses import asdict

from agency.engine.types import (
    CapabilityToken,
    ExecutionId,
    HarnessError,
    HarnessErrorKind,
    SkillExecutionRequest,
    SkillExecutionResponse,
    SkillExecutionStatus,
)


def test_run_request_has_the_minimal_wire_shape():
    request = SkillExecutionRequest(
        execution_id=ExecutionId("exec-1"),
        capability_token=CapabilityToken("cap-1"),
        config={
            "agent": {"engine": "claude_code"},
            "agharness": {
                "binary_path": None,
                "gateway_mode": "passthrough",
                "mediation_mode": "auto",
            },
            "execution": {"max_steps": 20, "timeout_s": 60},
        },
        prompt="compiled skill prompt",
    )

    assert asdict(request) == {
        "execution_id": "exec-1",
        "capability_token": "cap-1",
        "config": {
            "agent": {"engine": "claude_code"},
            "agharness": {
                "binary_path": None,
                "gateway_mode": "passthrough",
                "mediation_mode": "auto",
            },
            "execution": {"max_steps": 20, "timeout_s": 60},
        },
        "prompt": "compiled skill prompt",
    }


def test_run_request_preserves_multimodal_prompt_blocks():
    prompt = [
        {"type": "text", "text": "inspect this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]
    request = SkillExecutionRequest(
        execution_id=ExecutionId("exec-1"),
        capability_token=CapabilityToken("cap-1"),
        config={"agent": {"engine": "codex"}},
        prompt=prompt,
    )

    assert request.prompt == prompt


def test_run_result_success_is_derived_from_status():
    result = SkillExecutionResponse(
        execution_id=ExecutionId("exec-1"),
        status=SkillExecutionStatus.SUCCEEDED,
        final_text="done",
    )

    assert result.ok is True


def test_run_result_carries_structured_failure():
    result = SkillExecutionResponse(
        execution_id=ExecutionId("exec-1"),
        status=SkillExecutionStatus.FAILED,
        exit_code=1,
        error=HarnessError(
            kind=HarnessErrorKind.HARNESS_EXITED,
            message="harness exited with status 1",
            retryable=False,
            details={"stderr": "bad configuration"},
        ),
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is HarnessErrorKind.HARNESS_EXITED
