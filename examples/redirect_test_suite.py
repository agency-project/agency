"""Executable, deterministic examples for ``Invocation.redirect()`` semantics.

The model is scripted so the synchronization is reliable, but each scenario
uses Agency's real invocation control and native ReAct execution paths.

Run directly::

    uv run python examples/redirect_test_suite.py

Or run as a pytest suite::

    uv run pytest examples/redirect_test_suite.py -q
"""

from __future__ import annotations

import copy
import json
import tempfile
import threading
from contextlib import contextmanager

from agency._agent_control import AgentControl
from agency.native_harness import tools
from agency.native_harness.react_loop import run_react_loop


WAIT_SECONDS = 2.0
REDIRECT_PREFIX = "[AGENCY INVOCATION MESSAGE]"


class ScriptedLlm:
    """Return predetermined responses while recording every model request."""

    def __init__(self, responses):
        self._responses = iter(responses)
        self.requests = []

    def dispatch(self, model, messages, tool_schemas=None, **_kwargs):
        self.requests.append((model, copy.deepcopy(messages), copy.deepcopy(tool_schemas)))
        response = next(self._responses)
        return response() if callable(response) else copy.deepcopy(response)


class ControlBridge:
    """Use the same checkpoint surface that the native harness bridge calls."""

    def __init__(self, invocation):
        self.invocation = invocation

    def checkpoint(self, boundary_id, *, allow_messages, phase):
        decision = self.invocation._checkpoint(
            boundary_id,
            allow_messages=allow_messages,
            phase=phase,
        )
        return {
            "action_admitted": decision.action_admitted,
            "cancelled": decision.cancelled,
            "destroyed": decision.destroyed,
            "invocation_messages": [
                {"sequence": entry.sequence, "content": entry.content}
                for entry in decision.invocation_messages
            ],
        }

    @staticmethod
    def check_tool_policy(_tool_name, _tool_input):
        return {"decision": "allow", "reason": None}


def tool_response(*labels):
    return {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call-{index}",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"label": label}),
                    },
                }
                for index, label in enumerate(labels)
            ],
        },
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def final_response(text="done"):
    return {
        "message": {"role": "assistant", "content": text},
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def redirect_texts(messages):
    texts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and content.startswith(REDIRECT_PREFIX):
            texts.append(content.removeprefix(REDIRECT_PREFIX).strip())
    return texts


@contextmanager
def replace_bash_tool(handler):
    original = tools.TOOL_DISPATCH["bash"]
    tools.TOOL_DISPATCH["bash"] = handler
    try:
        yield
    finally:
        tools.TOOL_DISPATCH["bash"] = original


def start_loop(invocation, llm, offload_dir):
    outcome = {}

    def run():
        try:
            outcome["result"] = run_react_loop(
                [{"role": "user", "content": "start"}],
                "scripted-model",
                llm,
                bridge=ControlBridge(invocation),
                context_limit=None,
                max_steps=4,
                offload_dir=offload_dir,
            )
        except BaseException as error:
            outcome["error"] = error

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    return worker, outcome


def finish_loop(worker, outcome):
    worker.join(timeout=WAIT_SECONDS)
    assert not worker.is_alive(), "execution loop did not finish"
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def test_redirect_while_tool_is_running():
    control = AgentControl()
    invocation = control.begin_invocation("tool-running")
    tool_a_started = threading.Event()
    release_tool_a = threading.Event()
    events = []

    def blocking_tool(arguments):
        label = json.loads(arguments)["label"]
        events.append(f"tool-{label}-started")
        if label == "A":
            tool_a_started.set()
            assert release_tool_a.wait(timeout=WAIT_SECONDS)
        events.append(f"tool-{label}-finished")
        return json.dumps({"ok": label})

    llm = ScriptedLlm([tool_response("A", "B"), final_response("redirected")])
    with tempfile.TemporaryDirectory() as offload_dir, replace_bash_tool(blocking_tool):
        worker, outcome = start_loop(invocation, llm, offload_dir)
        try:
            assert tool_a_started.wait(timeout=WAIT_SECONDS)
            invocation.redirect("Do not run tool B; finish now.")
            assert events == ["tool-A-started"]
            release_tool_a.set()
            result = finish_loop(worker, outcome)
        finally:
            release_tool_a.set()
            worker.join(timeout=WAIT_SECONDS)

    assert result.status == "done"
    assert result.final_text == "redirected"
    assert events == ["tool-A-started", "tool-A-finished"]
    assert redirect_texts(llm.requests[1][1]) == ["Do not run tool B; finish now."]


def test_redirect_while_model_is_generating_discards_stale_tool_and_final():
    stale_responses = [
        tool_response("stale-action"),
        final_response("stale-final-answer"),
    ]

    for stale_response in stale_responses:
        control = AgentControl()
        invocation = control.begin_invocation("model-generating")
        generation_started = threading.Event()
        release_generation = threading.Event()
        tool_calls = []

        def stale_generation():
            generation_started.set()
            assert release_generation.wait(timeout=WAIT_SECONDS)
            return stale_response

        llm = ScriptedLlm([stale_generation, final_response("fresh-answer")])
        with tempfile.TemporaryDirectory() as offload_dir, replace_bash_tool(tool_calls.append):
            worker, outcome = start_loop(invocation, llm, offload_dir)
            try:
                assert generation_started.wait(timeout=WAIT_SECONDS)
                invocation.redirect("Replace the in-flight draft.")
                release_generation.set()
                result = finish_loop(worker, outcome)
            finally:
                release_generation.set()
                worker.join(timeout=WAIT_SECONDS)

        assert result.final_text == "fresh-answer"
        assert tool_calls == []
        assert len(llm.requests) == 2
        assert redirect_texts(llm.requests[0][1]) == []
        assert redirect_texts(llm.requests[1][1]) == ["Replace the in-flight draft."]


def test_redirect_races_with_final_completion():
    accepted_control = AgentControl()
    accepted_invocation = accepted_control.begin_invocation("redirect-wins")
    start_redirect = threading.Event()
    redirect_accepted = threading.Event()

    def redirect_before_final():
        assert start_redirect.wait(timeout=WAIT_SECONDS)
        accepted_invocation.redirect("revise before closing")
        redirect_accepted.set()

    accepted_worker = threading.Thread(target=redirect_before_final, daemon=True)
    accepted_worker.start()
    start_redirect.set()
    assert redirect_accepted.wait(timeout=WAIT_SECONDS)
    accepted = accepted_invocation._checkpoint_final_answer("first-final")
    accepted_worker.join(timeout=WAIT_SECONDS)

    assert [entry.content for entry in accepted.invocation_messages] == ["revise before closing"]
    assert accepted_invocation.phase == "boundary"
    accepted_invocation._acknowledge_redirects(accepted.invocation_messages)
    accepted_invocation._checkpoint_final_answer("revised-final")
    assert accepted_invocation.phase == "closing"

    closing_control = AgentControl()
    closing_invocation = closing_control.begin_invocation("closing-wins")
    closing_committed = threading.Event()
    redirect_failed = threading.Event()

    def redirect_after_final():
        assert closing_committed.wait(timeout=WAIT_SECONDS)
        try:
            closing_invocation.redirect("too late")
        except RuntimeError:
            redirect_failed.set()

    closing_worker = threading.Thread(target=redirect_after_final, daemon=True)
    closing_worker.start()
    closing_invocation._checkpoint_final_answer("committed-final")
    closing_committed.set()
    closing_worker.join(timeout=WAIT_SECONDS)

    assert closing_invocation.phase == "closing"
    assert redirect_failed.is_set()


def test_failed_model_call_retains_redirect_for_retry():
    control = AgentControl()
    invocation = control.begin_invocation("failed-model")
    invocation.redirect("retain across retry")

    with tempfile.TemporaryDirectory() as offload_dir:
        failed_llm = ScriptedLlm([{"error": "provider disconnected"}])
        failed = run_react_loop(
            [{"role": "user", "content": "start"}],
            "scripted-model",
            failed_llm,
            bridge=ControlBridge(invocation),
            offload_dir=offload_dir,
        )
        assert failed.status == "error"
        assert [entry.content for entry in invocation._pending_messages] == ["retain across retry"]

        retry_llm = ScriptedLlm([final_response("retry-succeeded")])
        succeeded = run_react_loop(
            [{"role": "user", "content": "start"}],
            "scripted-model",
            retry_llm,
            bridge=ControlBridge(invocation),
            offload_dir=offload_dir,
        )

    assert redirect_texts(failed_llm.requests[0][1]) == ["retain across retry"]
    assert redirect_texts(retry_llm.requests[0][1]) == ["retain across retry"]
    assert succeeded.final_text == "retry-succeeded"
    assert not invocation._pending_messages


def test_multiple_redirects_remain_fifo_when_second_arrives_in_flight():
    control = AgentControl()
    invocation = control.begin_invocation("redirect-fifo")
    invocation.redirect("redirect A")
    processing_a = threading.Event()
    release_a = threading.Event()

    def response_while_processing_a():
        processing_a.set()
        assert release_a.wait(timeout=WAIT_SECONDS)
        return final_response("stale-after-A")

    llm = ScriptedLlm([response_while_processing_a, final_response("A-then-B")])
    with tempfile.TemporaryDirectory() as offload_dir:
        worker, outcome = start_loop(invocation, llm, offload_dir)
        try:
            assert processing_a.wait(timeout=WAIT_SECONDS)
            assert redirect_texts(llm.requests[0][1]) == ["redirect A"]
            invocation.redirect("redirect B")
            release_a.set()
            result = finish_loop(worker, outcome)
        finally:
            release_a.set()
            worker.join(timeout=WAIT_SECONDS)

    assert result.final_text == "A-then-B"
    assert redirect_texts(llm.requests[1][1]) == ["redirect A", "redirect B"]
    assert not invocation._pending_messages


def test_no_redirect_happy_path_executes_tools_and_completes():
    control = AgentControl()
    invocation = control.begin_invocation("happy-path")
    executed = []

    def normal_tool(arguments):
        label = json.loads(arguments)["label"]
        executed.append(label)
        return json.dumps({"ok": label})

    llm = ScriptedLlm([tool_response("normal"), final_response("complete")])
    with tempfile.TemporaryDirectory() as offload_dir, replace_bash_tool(normal_tool):
        result = run_react_loop(
            [{"role": "user", "content": "start"}],
            "scripted-model",
            llm,
            bridge=ControlBridge(invocation),
            offload_dir=offload_dir,
        )

    assert executed == ["normal"]
    assert result.status == "done"
    assert result.final_text == "complete"
    assert invocation.phase == "closing"


CASES = [
    test_no_redirect_happy_path_executes_tools_and_completes,
    test_redirect_while_tool_is_running,
    test_redirect_while_model_is_generating_discards_stale_tool_and_final,
    test_redirect_races_with_final_completion,
    test_failed_model_call_retains_redirect_for_retry,
    test_multiple_redirects_remain_fifo_when_second_arrives_in_flight,
]


def main():
    for case in CASES:
        case()
        print(f"PASS {case.__name__}")
    print(f"\n{len(CASES)} redirect scenarios passed")


if __name__ == "__main__":
    main()
