from __future__ import annotations

from dataclasses import asdict

from agency.engine.types import (
    CapabilityToken,
    ContextId,
    ContextRef,
    ExecutionId,
    HarnessError,
    HarnessErrorKind,
    HarnessRunRequest,
    HarnessRunResult,
    HarnessRunStatus,
)


def test_run_request_has_the_minimal_wire_shape():
    request = HarnessRunRequest(
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
        context=ContextRef(id=ContextId("ctx-1"), version=3),
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
        "context": {"id": "ctx-1", "version": 3},
        "prompt": "compiled skill prompt",
    }


def test_run_request_preserves_multimodal_prompt_blocks():
    prompt = [
        {"type": "text", "text": "inspect this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]
    request = HarnessRunRequest(
        execution_id=ExecutionId("exec-1"),
        capability_token=CapabilityToken("cap-1"),
        config={"agent": {"engine": "codex"}},
        context=ContextRef(),
        prompt=prompt,
    )

    assert request.prompt == prompt


def test_run_result_success_is_derived_from_status():
    result = HarnessRunResult(
        execution_id=ExecutionId("exec-1"),
        status=HarnessRunStatus.SUCCEEDED,
        context=ContextRef(id=ContextId("ctx-1"), version=1),
        final_text="done",
    )

    assert result.ok is True


def test_run_result_carries_structured_failure():
    result = HarnessRunResult(
        execution_id=ExecutionId("exec-1"),
        status=HarnessRunStatus.FAILED,
        context=ContextRef(),
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
