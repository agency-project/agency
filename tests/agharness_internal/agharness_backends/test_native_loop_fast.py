"""Fast (no-Docker), in-process tests for the native engine's react-loop
logic, via `_native_loop_harness.py`. Complements test_native.py's real-
Docker tests -- those prove the launch+bridge mechanism itself works
against a genuine container; these prove the loop's OWN behavior (tool
calling, tool-output offload, max_steps, compaction) fast, exercising the
real `_run_react_loop()` code, not a fake."""

from __future__ import annotations

import json
from unittest.mock import patch

from agency.agdata import agdata
from agency.agskill import agskill

from ._native_loop_harness import NativeLoopHarness, content_chunks, tool_call_chunks


class _CapturedSpan:
    def __init__(self, owner, name, span_id, metadata):
        self.owner = owner
        self.name = name
        self.span_id = span_id
        self.metadata = dict(metadata or {})

    def __enter__(self):
        self.owner.spans.append(self)
        return self

    def annotate(self, **metadata):
        self.metadata.update(metadata)

    def __exit__(self, exc_type, _exc_value, _traceback):
        self.metadata.setdefault("outcome", "failure" if exc_type else "success")


class _CapturedProfiler:
    def __init__(self):
        self.spans = []

    def span(self, name, *, span_id=None, metadata=None):
        return _CapturedSpan(self, name, span_id, metadata)


def test_react_loop_emits_exact_turn_and_tool_intervals(monkeypatch):
    h = NativeLoopHarness()
    module = h.module
    profiler = _CapturedProfiler()
    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_7",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command":"printf m5-marker"}',
                            },
                        }
                    ],
                },
                "usage": {},
            },
            {"message": {"role": "assistant", "content": "done"}, "usage": {}},
        ]
    )
    monkeypatch.setattr(module, "_fetch_context_limit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "_dispatch_via_terminus",
        lambda *_args, **_kwargs: next(responses),
    )

    response = module._run_react_loop_inner(
        {
            "token": "tok",
            "terminus_sock": "/unused",
            "model": "m",
            "messages": [{"role": "user", "content": "run a tool"}],
            "max_steps": 3,
        },
        profiler,
    )

    assert response["status"] == "done"
    assert response["turn_count"] == 2
    assert [span.name for span in profiler.spans] == ["turn0", "tool:bash", "turn1"]
    tool_span = profiler.spans[1]
    assert tool_span.span_id == "tool:tc_7"
    assert tool_span.metadata["arguments"] == '{"command":"printf m5-marker"}'
    assert "m5-marker" in tool_span.metadata["result"]


def test_compaction_emits_exact_interval(monkeypatch):
    h = NativeLoopHarness()
    module = h.module
    profiler = _CapturedProfiler()
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]

    monkeypatch.setattr(module._agllm_pure, "should_compact", lambda *_args: True)
    monkeypatch.setattr(
        module._agllm_pure,
        "split_for_compaction",
        lambda *_args: (messages[0], messages[1], [{"role": "assistant", "content": "old"}], []),
    )
    monkeypatch.setattr(module._agllm_pure, "prune_tool_outputs", lambda head: head)
    monkeypatch.setattr(
        module._agllm_pure,
        "build_summary_prompt_messages",
        lambda *_args: [{"role": "user", "content": "summarize"}],
    )
    monkeypatch.setattr(
        module._agllm_pure,
        "assemble_compacted_messages",
        lambda *_args: [{"role": "system", "content": "compacted"}],
    )
    monkeypatch.setattr(
        module,
        "_dispatch_via_terminus",
        lambda *_args, **_kwargs: {
            "message": {"role": "assistant", "content": "old turns summarized"},
            "usage": {},
        },
    )

    compacted, summary = module._maybe_compact(
        messages,
        100,
        "/unused",
        "tok",
        "m",
        None,
        profiler,
    )

    assert summary == "old turns summarized"
    assert compacted == [{"role": "system", "content": "compacted"}]
    assert [span.name for span in profiler.spans] == ["llm:compact"]
    assert profiler.spans[0].metadata["outcome"] == "success"


def test_retry_sleep_emits_exact_backoff_interval(monkeypatch):
    import httpx
    import time

    h = NativeLoopHarness()
    module = h.module
    profiler = _CapturedProfiler()

    class FakeResponse:
        def __init__(self, status_code, lines=()):
            self.status_code = status_code
            self._lines = lines
            self.text = "temporarily unavailable"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return self.text.encode()

        def iter_lines(self):
            return iter(self._lines)

    responses = iter(
        [
            FakeResponse(503),
            FakeResponse(
                200,
                [
                    'data: {"choices":[{"delta":{"content":"done"}}]}',
                    "data: [DONE]",
                ],
            ),
        ]
    )

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return next(responses)

    monkeypatch.setattr(httpx, "HTTPTransport", lambda **_kwargs: object())
    monkeypatch.setattr(httpx, "Client", FakeClient)
    monkeypatch.setattr(module, "_dispatch_retry_backoff_s", lambda _attempt: 0.25)
    monkeypatch.setattr(time, "sleep", lambda _delay: None)

    response = module._dispatch_via_terminus(
        "/unused",
        "tok",
        {"model": "m", "messages": []},
        max_retries=2,
        profiler=profiler,
    )

    assert response["message"]["content"] == "done"
    assert [span.name for span in profiler.spans] == ["llm:retry_backoff"]
    assert profiler.spans[0].metadata == {
        "attempt": 0,
        "delay_ms": 250.0,
        "outcome": "success",
    }


def test_bash_tool_round_trip():
    h = NativeLoopHarness()
    try:
        h.fake_client.chat.completions.create.side_effect = [
            tool_call_chunks("bash", {"command": "echo hello-from-bash"}),
            content_chunks("the command printed hello-from-bash"),
        ]
        req = h.base_request(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "run echo hello-from-bash"},
            ]
        )
        resp = h.run(req)

        assert resp["status"] == "done", resp
        assert "hello-from-bash" in resp["final_text"]
        tool_msgs = [m for m in resp["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        tool_content = json.loads(tool_msgs[0]["content"])
        assert "hello-from-bash" in tool_content["output"]
        assert tool_content["returncode"] == 0
        assert h.fake_client.chat.completions.create.call_count == 2
    finally:
        h.stop()


def test_max_steps_exhausted_returns_error():
    h = NativeLoopHarness()
    try:
        # The mocked model never stops calling tools -- every dispatch gets
        # another tool-call chunk, so the loop must hit max_steps rather
        # than looping forever.
        h.fake_client.chat.completions.create.side_effect = [
            tool_call_chunks("bash", {"command": "true"}, call_id=f"call_{i}") for i in range(10)
        ]
        req = h.base_request(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "keep calling tools forever"},
            ],
            max_steps=3,
        )
        resp = h.run(req)

        assert resp["status"] == "error"
        assert "max_steps" in resp["message"]
        assert h.fake_client.chat.completions.create.call_count == 3
    finally:
        h.stop()


def test_oversized_tool_output_is_offloaded_to_a_file(tmp_path):
    h = NativeLoopHarness()
    try:
        # Point the offload directory at a real, writable tmp dir instead
        # of the hardcoded /workspace (not guaranteed to exist/be writable
        # outside a real container) -- see _offload_if_oversized's own
        # docstring for why this is a plain module-level constant.
        offload_dir = tmp_path / "long_tool_call_outputs"
        h.module._OFFLOAD_DIR = str(offload_dir)

        huge_command = "python3 -c \"print('x' * 50000)\""
        h.fake_client.chat.completions.create.side_effect = [
            tool_call_chunks("bash", {"command": huge_command}),
            content_chunks("done"),
        ]
        req = h.base_request(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "run the huge command"},
            ]
        )
        resp = h.run(req)

        assert resp["status"] == "done", resp
        tool_msgs = [m for m in resp["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        tool_content = json.loads(tool_msgs[0]["content"])
        # The oversized result must have been replaced with a short note,
        # not the raw 50000-char output.
        assert "note" in tool_content
        assert "saved to" in tool_content["note"]
        offloaded_files = list(offload_dir.glob("bash_*.txt"))
        assert len(offloaded_files) == 1
        # The offloaded file holds the FULL output, not a truncated tail --
        # this is the whole point of offload vs. the old truncate-and-lose
        # behavior.
        saved = offloaded_files[0].read_text()
        assert saved.count("x") >= 50000
    finally:
        h.stop()


def test_compaction_triggers_and_replaces_old_turns_with_a_summary():
    h = NativeLoopHarness()
    try:
        history = [{"role": "system", "content": "sys"}]
        for i in range(5):
            history.append({"role": "user", "content": f"turn {i}"})
            history.append({"role": "assistant", "content": f"reply {i}" * 200})

        h.fake_client.chat.completions.create.side_effect = [
            content_chunks("summary of earlier turns"),  # the compaction summarization call
            content_chunks("final answer after compaction"),  # the real turn
        ]
        req = h.base_request(history + [{"role": "user", "content": "final question"}])

        with patch("agency.agllm.agllm.fetch_context_limit", return_value=100):
            resp = h.run(req)

        assert resp["status"] == "done", resp
        assert "final answer after compaction" in resp["final_text"]
        # Compaction must have shrunk the message list -- it started with
        # system + 10 history + 1 new user turn = 12 messages, plus the
        # final assistant reply; a real compaction pass collapses the
        # early turns into one summary message.
        assert len(resp["messages"]) < 13
    finally:
        h.stop()


def test_mcp_tool_discovery_and_call_end_to_end():
    h = NativeLoopHarness(with_mcp=True)
    try:
        h.fake_client.chat.completions.create.side_effect = [
            tool_call_chunks("daemon_release", {"pid": 4242}, call_id="call_mcp"),
            content_chunks("released the daemon"),
        ]
        req = h.base_request(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "release daemon pid 4242"},
            ]
        )
        resp = h.run(req)

        assert resp["status"] == "done", resp
        h.fake_ag.sandbox.release_daemon.assert_called_once_with(4242)
        tool_msgs = [m for m in resp["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        tool_content = json.loads(tool_msgs[0]["content"])
        assert "released as daemon" in tool_content.get("message", "")
    finally:
        h.stop()


def test_messenger_delivers_pending_inbox_message_before_first_turn():
    h = NativeLoopHarness(with_messenger=True)
    try:
        delivered = {"done": False}

        def _drain_inbox(messages):
            if delivered["done"]:
                return False
            messages.append({"role": "user", "content": "please also check the logs"})
            delivered["done"] = True
            return True

        h.fake_ag._drain_inbox.side_effect = _drain_inbox
        h.fake_client.chat.completions.create.side_effect = [
            content_chunks("Got it: please also check the logs.")
        ]
        req = h.base_request(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "original task"}]
        )

        with patch("agency.agllm.agllm.fetch_context_limit", return_value=100_000):
            resp = h.run(req)

        assert resp["status"] == "done", resp
        h.fake_ag._check_pause.assert_called()
        h.fake_ag._drain_inbox.assert_called()
        joined = json.dumps(resp["messages"])
        assert "please also check the logs" in joined
        assert "Got it" in resp["final_text"]
    finally:
        h.stop()


def test_unknown_tool_call_returns_error_result():
    """Ported from tests/test_agskill.py's (retired)
    test_unknown_tool_error_in_history -- the entrypoint's dispatch table
    has an identical unknown-tool fallback for names it doesn't recognize
    (built-in or MCP)."""
    h = NativeLoopHarness()
    try:
        h.fake_client.chat.completions.create.side_effect = [
            tool_call_chunks("ghost", {}),
            content_chunks("done"),
        ]
        req = h.base_request(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "call a bogus tool"}]
        )
        resp = h.run(req)

        assert resp["status"] == "done", resp
        tool_msgs = [m for m in resp["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "unknown tool" in tool_msgs[0]["content"]
    finally:
        h.stop()


def test_submit_output_all_fields_collected():
    """Ported from tests/test_agskill.py's (retired)
    test_return_output_all_fields_correct -- native's structured-output
    mechanism is `submit_output` (one MCP call per field) instead of the
    old per-field `return_<field>` tool, so this proves the MCP-level
    mechanism at the entrypoint tier: field values land in
    `agmcp_server.collected_output(token)` correctly typed. The one level
    above this (native.py's own reprompt-until-complete loop, wrapping the
    result into `agdata`) is Docker-only coverage today -- see
    test_native.py's TestNativeBackendRealEndToEnd."""
    skill = agskill(name="s", system_prompt="", output_schema=agdata(summary=str, word_count=int))
    h = NativeLoopHarness(with_mcp=True, mcp_skill=skill)
    try:
        h.fake_client.chat.completions.create.side_effect = [
            tool_call_chunks(
                "submit_output",
                {"field": "summary", "value": json.dumps("great paper")},
                call_id="call_1",
            ),
            tool_call_chunks(
                "submit_output", {"field": "word_count", "value": json.dumps(2)}, call_id="call_2"
            ),
            content_chunks("done"),
        ]
        req = h.base_request(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "summarize"}]
        )
        resp = h.run(req)

        assert resp["status"] == "done", resp
        assert h.mcp_server.collected_output(h.token) == {
            "summary": "great paper",
            "word_count": 2,
        }
    finally:
        h.stop()


def test_submit_output_type_error_returns_immediate_feedback():
    """Ported from tests/test_agskill.py's (retired)
    test_return_output_type_error_immediate_feedback -- submit_output
    validates inline via output_schema.check_field() on every call, no
    outer retry needed for a single bad value followed by a corrected one."""
    skill = agskill(name="s", system_prompt="", output_schema=agdata(word_count=int))
    h = NativeLoopHarness(with_mcp=True, mcp_skill=skill)
    try:
        h.fake_client.chat.completions.create.side_effect = [
            tool_call_chunks(
                "submit_output",
                {"field": "word_count", "value": json.dumps("not-a-number")},
                call_id="call_1",
            ),
            tool_call_chunks(
                "submit_output", {"field": "word_count", "value": json.dumps(3)}, call_id="call_2"
            ),
            content_chunks("done"),
        ]
        req = h.base_request(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "count words"}]
        )
        resp = h.run(req)

        assert resp["status"] == "done", resp
        tool_msgs = [m for m in resp["messages"] if m.get("role") == "tool"]
        first_result = json.loads(tool_msgs[0]["content"])
        assert "error" in first_result, first_result
        assert h.mcp_server.collected_output(h.token) == {"word_count": 3}
    finally:
        h.stop()


def test_token_usage_is_tracked():
    """New coverage enabled by this session's usage-tracking fix (native
    previously returned no usage data at all, see native.py's module
    docstring) -- proves _run_react_loop()'s response carries real,
    accumulated prompt/completion token counts, not an estimate."""
    from openai.types.completion_usage import CompletionUsage

    h = NativeLoopHarness()
    try:
        chunks = content_chunks("hello")
        chunks[-1].usage = CompletionUsage(prompt_tokens=11, completion_tokens=4, total_tokens=15)
        h.fake_client.chat.completions.create.side_effect = [chunks]
        req = h.base_request(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        )
        resp = h.run(req)

        assert resp["status"] == "done", resp
        assert resp["usage"] == {"input_tokens": 11, "output_tokens": 4}
    finally:
        h.stop()


def test_final_text_preserves_raw_content_verbatim():
    """Ported from tests/test_agrawstring.py's (retired) test_raw_output_*
    family -- the entrypoint's final_text is the model's raw content,
    untouched (no JSON parsing/reformatting), preserving quotes/newlines.
    native.py's `_NativeBackend.execute()` wraps this verbatim into
    `agdata(**{raw_key: final_text})` for an agrawstring output schema --
    that one extra wrapping step is Docker-only coverage today (see
    test_native.py's TestNativeBackendRealEndToEnd)."""
    h = NativeLoopHarness()
    try:
        prose = 'He said "hello"\nShe replied "goodbye"'
        h.fake_client.chat.completions.create.side_effect = [content_chunks(prose)]
        req = h.base_request(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "write"}]
        )
        resp = h.run(req)

        assert resp["status"] == "done", resp
        assert resp["final_text"] == prose
    finally:
        h.stop()


# ---------------------------------------------------------------------------
# Custom tools (skill.add_tools/replace_tools -- Phase 0.F). native.py ships
# a host-authored `fn` into the container as cloudpickled, base64-encoded
# bytes (`req["custom_tools"]`); these tests build that same payload
# directly (real cloudpickle, no mocking) and hand it to the real
# `_run_react_loop()` via this harness, proving the actual mechanism, not a
# stand-in for it.
# ---------------------------------------------------------------------------


def _custom_tool_entry(name: str, description: str, params: dict, fn) -> dict:
    import base64
    import cloudpickle

    return {
        "name": name,
        "description": description,
        "params": params,
        "fn_b64": base64.b64encode(cloudpickle.dumps(fn)).decode(),
    }


def test_custom_tool_round_trip():
    """A host-authored closure, written against the REAL agency.agdata
    classes (exactly as a real add_tools author would write it), cloudpickled
    and shipped in as native.py does -- proves the sys.modules stub in
    _native_in_container_entrypoint.py actually resolves those class
    references to agdata_pure.py's shim at unpickling time, end to end."""
    from agency.agdata import agdata as real_agdata, agerror as real_agerror

    def double_fn(arg):
        if not hasattr(arg, "n"):
            return real_agerror("n is required")
        return real_agdata(doubled=arg.n * 2)

    entry = _custom_tool_entry(
        "double",
        "Doubles a number.",
        {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
        double_fn,
    )

    h = NativeLoopHarness()
    try:
        h.fake_client.chat.completions.create.side_effect = [
            tool_call_chunks("double", {"n": 21}),
            content_chunks("the answer is 42"),
        ]
        req = h.base_request(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "double 21"}]
        )
        req["custom_tools"] = [entry]
        resp = h.run(req)

        assert resp["status"] == "done", resp
        tool_msgs = [m for m in resp["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert json.loads(tool_msgs[0]["content"]) == {"doubled": 42}
        assert "42" in resp["final_text"]
    finally:
        h.stop()


def test_replace_tools_suppresses_builtins():
    """`suppress_builtins=True` (native.py sends this for `skill.replace_tools`,
    including plan_mode's `replace_tools=[]`) must remove bash/read/write/
    edit/glob/grep/webfetch/todowrite from the dispatch table entirely --
    checked here by having the (mocked) model try to call `bash` anyway and
    asserting it comes back as an unknown tool, the same signal
    test_unknown_tool_call_returns_error_result uses for a truly unknown
    name."""
    h = NativeLoopHarness()
    try:
        h.fake_client.chat.completions.create.side_effect = [
            tool_call_chunks("bash", {"command": "echo hi"}),
            content_chunks("done"),
        ]
        req = h.base_request(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "run something"}]
        )
        req["suppress_builtins"] = True
        req["custom_tools"] = []
        resp = h.run(req)

        assert resp["status"] == "done", resp
        tool_msgs = [m for m in resp["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "unknown tool" in tool_msgs[0]["content"]
    finally:
        h.stop()


def test_custom_tool_load_failure_does_not_abort_the_run():
    """`cloudpickle.dumps(tool.fn)` failing is checked host-side, in
    native.py, before this entrypoint ever sees a request (see
    test_native.py's own fail-fast test) -- but `cloudpickle.loads` failing
    HERE, inside the container (a version mismatch, a third-party import
    the closure needs that isn't installed) is a real, distinct failure
    mode this entrypoint must degrade gracefully from: that one tool call
    fails clearly, the rest of the run continues."""
    entry = {
        "name": "broken",
        "description": "a tool whose shipped bytes don't unpickle",
        "params": {"type": "object", "properties": {}},
        "fn_b64": "not-valid-base64-or-pickle-bytes",
    }

    h = NativeLoopHarness()
    try:
        h.fake_client.chat.completions.create.side_effect = [
            tool_call_chunks("broken", {}),
            content_chunks("recovered"),
        ]
        req = h.base_request(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "call broken"}]
        )
        req["custom_tools"] = [entry]
        resp = h.run(req)

        assert resp["status"] == "done", resp
        tool_msgs = [m for m in resp["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "failed to load" in tool_msgs[0]["content"]
        assert "recovered" in resp["final_text"]
    finally:
        h.stop()


def test_describe_exception_spells_out_exception_group_members():
    """A `run` op that raises reports one string and nothing else -- so an
    ExceptionGroup summarised as "(1 sub-exception)" erases the only part
    naming what broke. The MCP client paths run under anyio task groups, so
    that wrapper is the common shape for a real bridge failure, not an edge
    case."""
    from ._native_loop_harness import load_entrypoint_module

    entrypoint = load_entrypoint_module()

    plain = entrypoint._describe_exception(RuntimeError("boom"))
    assert plain == "RuntimeError: boom"

    group = ExceptionGroup("unhandled errors in a TaskGroup", [ConnectionRefusedError("no uds")])
    described = entrypoint._describe_exception(group)
    assert "ConnectionRefusedError: no uds" in described
    assert "unhandled errors in a TaskGroup" in described

    nested = ExceptionGroup("outer", [ExceptionGroup("inner", [TimeoutError("read timeout")])])
    assert "TimeoutError: read timeout" in entrypoint._describe_exception(nested)
