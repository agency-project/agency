"""Tests for the native engine's in-container launch+bridge foundation
(agharness_backends/native.py and _native_in_container_entrypoint.py).

Real, docker-backed only -- this module's entire purpose is proving the
launch+bridge mechanism works against a genuine container (persistent
detached process, agency-package bind-mount, Unix-domain-socket
reachability), so there is no meaningful mocked-only tier the way
test_claude_code.py has one for orchestration logic that doesn't need a
real binary. Skipped automatically when Docker/Podman is unavailable, same
convention as test_agsandbox.py.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _docker_available() -> bool:
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


docker = pytest.mark.skipif(not _docker_available(), reason="Docker daemon not reachable")


def _make_sandbox(**kwargs):
    from agency.agconfig import agConfig
    from agency.agsandbox import agSandbox
    from agency.agsandbox_backends import agSandboxBackendConfig

    uid = str(uuid.uuid4())
    agconfig = kwargs.pop("agconfig", None)
    cfg = agConfig(agSandboxBackendConfig(backend="docker"), agconfig)
    return agSandbox(uid, agconfig=cfg, **kwargs)


class TestNativeInContainerEntrypoint:
    @docker
    def test_launch_and_ping_round_trip(self):
        from agency.agharness_internal.agharness_backends.native import (
            launch_in_container_entrypoint,
            ping,
        )

        sb = _make_sandbox()
        try:
            sock_path = launch_in_container_entrypoint(sb, timeout_s=30)
            resp = ping(sock_path)
            assert resp["status"] == "ok"
            assert "pid" in resp
            assert isinstance(resp["pid"], int)
        finally:
            sb.destroy()

    @docker
    def test_entrypoint_sees_the_bind_mounted_agency_package(self):
        """The in-container process must see the bind-mounted `agency`
        package at the expected container path -- proving the mount is
        genuinely visible from inside the container the entrypoint runs
        in, not just present on the host side of the bind mount."""
        from agency.agutil import AGENCY_PACKAGE_CONTAINER_MOUNT
        from agency.agharness_internal.agharness_backends.native import (
            launch_in_container_entrypoint,
            ping,
        )

        sb = _make_sandbox()
        try:
            sock_path = launch_in_container_entrypoint(sb, timeout_s=30)
            resp = ping(sock_path)
            assert resp["agency_package_visible"] is True
            assert resp["agency_package_marker_path"].startswith(AGENCY_PACKAGE_CONTAINER_MOUNT)
        finally:
            sb.destroy()

    @docker
    def test_entrypoint_process_is_persistent_across_multiple_pings(self):
        """The launched process must be the SAME long-lived process across
        repeated round-trips (matching a persistent container's whole
        premise), not a fresh one spun up per request."""
        from agency.agharness_internal.agharness_backends.native import (
            launch_in_container_entrypoint,
            ping,
        )

        sb = _make_sandbox()
        try:
            sock_path = launch_in_container_entrypoint(sb, timeout_s=30)
            pid1 = ping(sock_path)["pid"]
            pid2 = ping(sock_path)["pid"]
            pid3 = ping(sock_path)["pid"]
            assert pid1 == pid2 == pid3
        finally:
            sb.destroy()

    @docker
    def test_ping_unreachable_socket_raises(self):
        from agency.agharness_internal.agharness_backends.native import ping

        with pytest.raises(Exception):
            ping("/tmp/definitely-does-not-exist-nobody-listens-here.sock", timeout_s=1)


def _content_chunks(text: str, finish_reason: str = "stop"):
    """A one-chunk stream carrying the full content delta -- real, valid
    ChatCompletionChunk objects, since these cross a genuine HTTP boundary
    (terminus -> in-container process) as SSE frames and get parsed back
    out; a hand-rolled fake lacking real pydantic fields wouldn't
    round-trip. Dispatch is ALWAYS streaming now (see
    _native_in_container_entrypoint.py's _dispatch_via_terminus docstring
    for why), so every mocked response here is a list of chunks, never a
    single non-streaming ChatCompletion."""
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta

    delta = ChoiceDelta(content=text)
    choice = ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)
    return [
        ChatCompletionChunk(
            id="chatcmpl-test",
            object="chat.completion.chunk",
            created=0,
            model="m",
            choices=[choice],
        )
    ]


def _tool_call_chunks(command: str, call_id: str = "call_1", finish_reason: str = "tool_calls"):
    """A one-chunk stream carrying a complete tool-call delta (id, name,
    and arguments all in the same chunk -- a real stream would typically
    split these across chunks, but the reassembly logic in
    _dispatch_via_terminus concatenates regardless of how many chunks a
    given delta arrives across, so a single chunk is a valid, simpler
    stream shape to test with)."""
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import (
        Choice as ChunkChoice,
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )

    tc_delta = ChoiceDeltaToolCall(
        index=0,
        id=call_id,
        type="function",
        function=ChoiceDeltaToolCallFunction(
            name="bash", arguments=json.dumps({"command": command})
        ),
    )
    delta = ChoiceDelta(tool_calls=[tc_delta])
    choice = ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)
    return [
        ChatCompletionChunk(
            id="chatcmpl-test",
            object="chat.completion.chunk",
            created=0,
            model="m",
            choices=[choice],
        )
    ]


def _tool_call_named_chunks(
    tool_name: str, arguments: dict, call_id: str = "call_1", finish_reason: str = "tool_calls"
):
    """Like `_tool_call_chunks`, but for an arbitrary tool name/arguments --
    used to drive the react loop toward calling a dynamically-discovered
    MCP tool instead of the hardcoded `bash` one."""
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import (
        Choice as ChunkChoice,
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )

    tc_delta = ChoiceDeltaToolCall(
        index=0,
        id=call_id,
        type="function",
        function=ChoiceDeltaToolCallFunction(name=tool_name, arguments=json.dumps(arguments)),
    )
    delta = ChoiceDelta(tool_calls=[tc_delta])
    choice = ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)
    return [
        ChatCompletionChunk(
            id="chatcmpl-test",
            object="chat.completion.chunk",
            created=0,
            model="m",
            choices=[choice],
        )
    ]


class TestNativeReactLoop:
    """Full loop mechanism against a REAL container and a REAL
    agLLMTerminus over a REAL Unix domain socket -- only the backend
    client itself is mocked (no real credentials needed to verify the
    plumbing: dispatch routing, in-container bash execution, loop
    continuation, final-answer detection)."""

    @docker
    def test_bash_tool_round_trip_with_mocked_llm(self):
        from agency.agharness_internal.agllm_terminus import agLLMTerminus
        from agency.agharness_internal.agharness_backends.native import (
            _BASH_TOOL_SCHEMA,
            launch_in_container_entrypoint,
            run_react_loop,
        )

        responses = [
            _tool_call_chunks("echo hello-from-bash"),
            _content_chunks("the command printed hello-from-bash"),
        ]
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = responses
        fake_ag = MagicMock()
        fake_ag.llm.backend.make_client.return_value = fake_client

        terminus = agLLMTerminus()
        token = "tok"
        terminus.register(token, fake_ag)

        sb = _make_sandbox()
        try:
            sock_path = launch_in_container_entrypoint(sb, timeout_s=30)
            terminus_host_sock = terminus.ensure_uds_started()
            container_terminus_sock = f"/var/run/agency_llm_gateway/{Path(terminus_host_sock).name}"

            request = {
                "token": token,
                "terminus_sock": container_terminus_sock,
                "model": "m",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "run echo hello-from-bash"},
                ],
                "tools": [_BASH_TOOL_SCHEMA],
                "max_steps": 5,
            }
            resp = run_react_loop(sock_path, request, timeout_s=60)

            assert resp["status"] == "done"
            assert "hello-from-bash" in resp["final_text"]

            # The bash tool must have genuinely run INSIDE the container,
            # not been faked -- its real stdout is in the tool message.
            tool_msgs = [m for m in resp["messages"] if m.get("role") == "tool"]
            assert len(tool_msgs) == 1
            tool_content = json.loads(tool_msgs[0]["content"])
            assert "hello-from-bash" in tool_content["output"]
            assert tool_content["returncode"] == 0

            # Both LLM turns were genuinely dispatched (2 calls, not 1) --
            # confirms the loop actually continued after the tool result.
            assert fake_client.chat.completions.create.call_count == 2
        finally:
            sb.destroy()
            terminus.stop_uds()

    @docker
    def test_dispatch_retries_transient_terminus_error_and_recovers(self):
        """Native's own retry policy (NOT agllm_terminus's -- see that
        module's and _dispatch_via_terminus's own comments for why retry
        lives here): a transient provider error on the first attempt must
        not fail the whole run -- the terminus reports it as a 503 (safe
        to retry, nothing was sent yet), native's own loop retries, and
        the second attempt's real content reaches the model turn."""
        import openai
        import httpx as _httpx

        from agency.agharness_internal.agllm_terminus import agLLMTerminus
        from agency.agharness_internal.agharness_backends.native import (
            launch_in_container_entrypoint,
            run_react_loop,
        )

        responses = [
            openai.APIConnectionError(request=_httpx.Request("POST", "http://x")),
            _content_chunks("recovered after retry"),
        ]
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = responses
        fake_ag = MagicMock()
        fake_ag.llm.backend.make_client.return_value = fake_client

        terminus = agLLMTerminus()
        token = "tok-retry"
        terminus.register(token, fake_ag)

        sb = _make_sandbox()
        try:
            sock_path = launch_in_container_entrypoint(sb, timeout_s=30)
            terminus_host_sock = terminus.ensure_uds_started()
            container_terminus_sock = f"/var/run/agency_llm_gateway/{Path(terminus_host_sock).name}"

            request = {
                "token": token,
                "terminus_sock": container_terminus_sock,
                "model": "m",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "say something"},
                ],
                "max_steps": 5,
            }
            resp = run_react_loop(sock_path, request, timeout_s=60)

            assert resp["status"] == "done", resp
            assert "recovered after retry" in resp["final_text"]
            # The mock's side_effect list only has 2 entries -- if the
            # loop had NOT retried, this call_count would be 1 and the
            # error would have propagated as a run failure instead.
            assert fake_client.chat.completions.create.call_count == 2
        finally:
            sb.destroy()
            terminus.stop_uds()

    @docker
    def test_compaction_triggers_and_replaces_old_turns_with_a_summary(self):
        """Native's own compaction (design: lives in this loop, not
        agllm_terminus, since the terminus never owns conversation
        history -- see agllm_pure.py's docstring). A conversation with
        more than tail_turns (3) worth of assistant turns, sized past the
        (deliberately tiny, via context_limit) compact threshold, must get
        its older turns replaced by a summary -- proven by asserting the
        summarization call actually happened (2 real dispatches, not 1)
        and the summary text survives into the final message history."""
        from agency.agharness_internal.agllm_terminus import agLLMTerminus
        from agency.agharness_internal.agharness_backends.native import (
            launch_in_container_entrypoint,
            run_react_loop,
        )

        responses = [
            _content_chunks("SUMMARY: steps 1-2 done."),  # the compaction call
            _content_chunks("Steps 3-5 done, task complete."),  # the real turn
        ]
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = responses
        fake_ag = MagicMock()
        fake_ag.llm.backend.make_client.return_value = fake_client

        terminus = agLLMTerminus()
        token = "tok-compact"
        terminus.register(token, fake_ag)

        sb = _make_sandbox()
        try:
            sock_path = launch_in_container_entrypoint(sb, timeout_s=30)
            terminus_host_sock = terminus.ensure_uds_started()
            container_terminus_sock = f"/var/run/agency_llm_gateway/{Path(terminus_host_sock).name}"

            messages = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Please help me with a coding task."},
                {"role": "assistant", "content": "Working on step 1. " * 5},
                {"role": "user", "content": "continue"},
                {"role": "assistant", "content": "Working on step 2. " * 5},
                {"role": "user", "content": "continue"},
                {"role": "assistant", "content": "Working on step 3. " * 5},
                {"role": "user", "content": "continue"},
                {"role": "assistant", "content": "Working on step 4. " * 5},
                {"role": "user", "content": "continue"},
                {"role": "assistant", "content": "Working on step 5. " * 5},
                {"role": "user", "content": "What's the final status?"},
            ]
            request = {
                "token": token,
                "terminus_sock": container_terminus_sock,
                "model": "m",
                "messages": messages,
                "max_steps": 5,
            }
            # Patch fetch_context_limit directly rather than setting
            # fake_ag.llm.backend.context_limit -- that function's first
            # branch (`isinstance(llm_config, AgLLMBackendFields)`) is
            # False for a bare MagicMock, so it falls through to
            # agllm_backend.for_config() and ignores whatever attribute
            # was set (a real footgun found while writing this test: the
            # int assignment silently does nothing). Deliberately tiny so
            # this short conversation already trips should_compact().
            with patch("agency.agllm.agllm.fetch_context_limit", return_value=100):
                resp = run_react_loop(sock_path, request, timeout_s=60)

            assert resp["status"] == "done", resp
            assert "Steps 3-5 done" in resp["final_text"]
            # Both the compaction summarization call AND the real turn
            # were genuinely dispatched -- if compaction never triggered,
            # this would be 1, not 2, and step-1/2 content would still be
            # present verbatim below.
            assert fake_client.chat.completions.create.call_count == 2

            joined = json.dumps(resp["messages"])
            assert "SUMMARY: steps 1-2 done." in joined
            # The old, pre-compaction turn 1/2 content must be GONE --
            # replaced by the summary, not merely supplemented by it.
            assert "Working on step 1." not in joined
            assert "Working on step 2." not in joined
            # The tail (most recent turns) must survive untouched.
            assert "Working on step 5." in joined
        finally:
            sb.destroy()
            terminus.stop_uds()

    @docker
    def test_messenger_delivers_pending_inbox_message_before_first_turn(self):
        """Native's own pause/inbox check-in (agHarnessMessenger) -- a
        pending inbox message must reach the model as an ordinary user
        turn before the first real dispatch, mirroring what
        execute_react() already does in-process via
        ag._check_pause()/ag._drain_inbox()."""
        from agency.agharness_internal.agllm_terminus import agLLMTerminus
        from agency.agharness_internal.agharness_messenger import agHarnessMessenger
        from agency.agharness_internal.agharness_backends.native import (
            launch_in_container_entrypoint,
            run_react_loop,
        )

        responses = [_content_chunks("Got it: please also check the logs.")]
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = responses
        fake_ag = MagicMock()
        fake_ag.llm.backend.make_client.return_value = fake_client

        delivered = {"done": False}

        def _drain_inbox(messages):
            if delivered["done"]:
                return False
            messages.append({"role": "user", "content": "please also check the logs"})
            delivered["done"] = True
            return True

        fake_ag._drain_inbox.side_effect = _drain_inbox

        terminus = agLLMTerminus()
        messenger = agHarnessMessenger()
        token = "tok-messenger"
        terminus.register(token, fake_ag)
        messenger.register(token, fake_ag)

        sb = _make_sandbox()
        try:
            sock_path = launch_in_container_entrypoint(sb, timeout_s=30)
            terminus_host_sock = terminus.ensure_uds_started()
            messenger_host_sock = messenger.ensure_uds_started()
            container_terminus_sock = f"/var/run/agency_llm_gateway/{Path(terminus_host_sock).name}"
            container_messenger_sock = (
                f"/var/run/agency_llm_gateway/{Path(messenger_host_sock).name}"
            )

            request = {
                "token": token,
                "terminus_sock": container_terminus_sock,
                "messenger_sock": container_messenger_sock,
                "model": "m",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "original task"},
                ],
                "max_steps": 5,
            }
            # Patch fetch_context_limit so compaction can't accidentally
            # trigger and consume the one mocked response meant for the
            # real turn -- see the identical note in the compaction test
            # above for why fake_ag.llm.backend.context_limit alone
            # doesn't work (isinstance check fails for a bare MagicMock).
            with patch("agency.agllm.agllm.fetch_context_limit", return_value=100_000):
                resp = run_react_loop(sock_path, request, timeout_s=60)

            assert resp["status"] == "done", resp
            fake_ag._check_pause.assert_called()
            fake_ag._drain_inbox.assert_called()
            joined = json.dumps(resp["messages"])
            assert "please also check the logs" in joined
            # The mocked model's response references the injected message
            # -- proves it was genuinely part of the dispatched request,
            # not just appended locally and never sent.
            assert "Got it" in resp["final_text"]
        finally:
            sb.destroy()
            terminus.stop_uds()
            messenger.stop_uds()

    @docker
    def test_mcp_tool_discovery_and_call_end_to_end(self):
        """The in-container react loop must be able to discover and call a
        real agMCPServer tool (not just bash) -- proves the whole Phase 4
        bridge: entrypoint -> real `mcp` client -> UDS -> agMCPServer ->
        real sandbox mutation, the same path reserve_cpu/cpu_release/
        daemon_release/submit_output actually take."""
        from agency.agharness_internal.agllm_terminus import agLLMTerminus
        from agency.agharness_internal.agmcp_server import agMCPServer
        from agency.agharness_internal.agharness_backends.native import (
            launch_in_container_entrypoint,
            run_react_loop,
        )

        responses = [
            _tool_call_named_chunks("daemon_release", {"pid": 4242}, call_id="call_mcp"),
            _content_chunks("released the daemon"),
        ]
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = responses
        fake_ag = MagicMock()
        fake_ag.llm.backend.make_client.return_value = fake_client

        terminus = agLLMTerminus()
        mcp_server = agMCPServer()
        token = "tok-mcp"
        terminus.register(token, fake_ag)
        mcp_server.register(token, fake_ag, MagicMock())

        sb = _make_sandbox()
        try:
            sock_path = launch_in_container_entrypoint(sb, timeout_s=30)
            terminus_host_sock = terminus.ensure_uds_started()
            mcp_host_sock = mcp_server.ensure_uds_started()
            container_terminus_sock = f"/var/run/agency_llm_gateway/{Path(terminus_host_sock).name}"
            container_mcp_sock = f"/var/run/agency_llm_gateway/{Path(mcp_host_sock).name}"

            request = {
                "token": token,
                "terminus_sock": container_terminus_sock,
                "mcp_server_sock": container_mcp_sock,
                "model": "m",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "release daemon pid 4242"},
                ],
                "max_steps": 5,
            }
            resp = run_react_loop(sock_path, request, timeout_s=60)

            assert resp["status"] == "done", resp
            fake_ag.sandbox.release_daemon.assert_called_once_with(4242)

            tool_msgs = [m for m in resp["messages"] if m.get("role") == "tool"]
            assert len(tool_msgs) == 1
            tool_content = json.loads(tool_msgs[0]["content"])
            assert "released as daemon" in tool_content.get("message", "")
        finally:
            sb.destroy()
            terminus.stop_uds()
            mcp_server.stop_uds()

    @docker
    def test_write_then_read_builtin_tools_round_trip(self):
        """The full local built-in tool set (Phase 0's retirement work) --
        write then read, exercised with NO `tools` in the request at all,
        proving the entrypoint supplies its own built-in schemas by
        default rather than requiring the caller to enumerate them."""
        from agency.agharness_internal.agllm_terminus import agLLMTerminus
        from agency.agharness_internal.agharness_backends.native import (
            launch_in_container_entrypoint,
            run_react_loop,
        )

        responses = [
            _tool_call_named_chunks(
                "write",
                {"file_path": "/workspace/native_tool_test.txt", "content": "agency-write-marker"},
                call_id="call_write",
            ),
            _tool_call_named_chunks(
                "read",
                {"file_path": "/workspace/native_tool_test.txt"},
                call_id="call_read",
            ),
            _content_chunks("done"),
        ]
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = responses
        fake_ag = MagicMock()
        fake_ag.llm.backend.make_client.return_value = fake_client

        terminus = agLLMTerminus()
        token = "tok-builtin-tools"
        terminus.register(token, fake_ag)

        sb = _make_sandbox()
        try:
            sock_path = launch_in_container_entrypoint(sb, timeout_s=30)
            terminus_host_sock = terminus.ensure_uds_started()
            container_terminus_sock = f"/var/run/agency_llm_gateway/{Path(terminus_host_sock).name}"

            request = {
                "token": token,
                "terminus_sock": container_terminus_sock,
                "model": "m",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "write then read the file"},
                ],
                "max_steps": 5,
            }
            resp = run_react_loop(sock_path, request, timeout_s=60)

            assert resp["status"] == "done", resp
            tool_msgs = [m for m in resp["messages"] if m.get("role") == "tool"]
            assert len(tool_msgs) == 2
            write_result = json.loads(tool_msgs[0]["content"])
            assert write_result["created"] is True
            read_result = json.loads(tool_msgs[1]["content"])
            assert "agency-write-marker" in read_result["content"]
        finally:
            sb.destroy()
            terminus.stop_uds()

    @docker
    def test_max_steps_exhausted_returns_error(self):
        """A model that never stops calling tools must fail cleanly at
        max_steps, not hang or loop forever."""
        from agency.agharness_internal.agllm_terminus import agLLMTerminus
        from agency.agharness_internal.agharness_backends.native import (
            _BASH_TOOL_SCHEMA,
            launch_in_container_entrypoint,
            run_react_loop,
        )

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [
            _tool_call_chunks("true") for _ in range(10)
        ]
        fake_ag = MagicMock()
        fake_ag.llm.backend.make_client.return_value = fake_client

        terminus = agLLMTerminus()
        token = "tok"
        terminus.register(token, fake_ag)

        sb = _make_sandbox()
        try:
            sock_path = launch_in_container_entrypoint(sb, timeout_s=30)
            terminus_host_sock = terminus.ensure_uds_started()
            container_terminus_sock = f"/var/run/agency_llm_gateway/{Path(terminus_host_sock).name}"

            request = {
                "token": token,
                "terminus_sock": container_terminus_sock,
                "model": "m",
                "messages": [{"role": "user", "content": "loop forever"}],
                "tools": [_BASH_TOOL_SCHEMA],
                "max_steps": 3,
            }
            resp = run_react_loop(sock_path, request, timeout_s=60)

            assert resp["status"] == "error"
            assert "max_steps" in resp["message"]
        finally:
            sb.destroy()
            terminus.stop_uds()


def _bedrock_available() -> bool:
    import os

    return bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK"))


real_bedrock = pytest.mark.skipif(
    not _bedrock_available(), reason="AWS_BEARER_TOKEN_BEDROCK not set on this host"
)


@real_bedrock
class TestNativeBackendRealEndToEnd:
    """Tier 2, like test_claude_code.py's `real_claude` tests: the actual
    `_NativeBackend.execute()` contract, a real container, a real
    agLLMTerminus, and a real model (Bedrock, via AWS_BEARER_TOKEN_BEDROCK)
    -- called directly rather than through agent.run(), since engine=
    "native" isn't wired into agharness_backend.for_config() yet (Phase 0,
    deliberately not done until this backend supports enough of the real
    contract)."""

    def test_bash_tool_use_end_to_end(self):
        from agency.agconfig import agConfig
        from agency.agcontext import agcontext
        from agency.agdata import agdata
        from agency.agent import agent
        from agency.agllm_backends import agBedrockBackendConfig
        from agency.agsandbox_backends import agSandboxBackendConfig
        from agency.agskill import agskill
        from agency.agharness_internal.agharness_backends.native import _NativeBackend

        cfg = agConfig(
            agSandboxBackendConfig(backend="docker"),
            agBedrockBackendConfig(model="us.anthropic.claude-sonnet-5"),
        )
        sandbox = _make_sandbox(agconfig=cfg)
        ag = agent(agconfig=cfg, sandbox=sandbox)
        skill = agskill(
            name="native_bash_tool_test",
            system_prompt=(
                "You have a bash tool. Use it to run the exact command the user "
                "gives you, then report its output back in one short sentence."
            ),
        )

        try:
            backend = _NativeBackend(cfg)
            result, ctx, delta = backend.execute(
                ag,
                agcontext(),
                agdata(instruction="Run: echo agency-native-e2e-marker"),
                max_steps=5,
                skill=skill,
            )
            raw = result.to_dict()
            assert "error" not in raw, raw
            assert "agency-native-e2e-marker" in raw.get("result", "")

            # The bash tool must have genuinely run -- not the model
            # hallucinating the output without calling anything.
            tool_msgs = [m for m in ctx.messages if m.get("role") == "tool"]
            assert len(tool_msgs) >= 1, "model never actually called the bash tool"
            assert any("agency-native-e2e-marker" in m["content"] for m in tool_msgs), (
                "bash tool's real output never reached the model"
            )
        finally:
            sandbox.destroy()

    def test_add_tools_custom_tool_end_to_end(self):
        """A genuine `skill.add_tools` round trip through a real container:
        the closure is cloudpickled host-side (native.py), shipped in as
        base64 bytes, unpickled against the real `agdata_pure.py` file
        inside the container, and called for real -- not the fast-seam's
        in-process `_run_react_loop()` call, an actual `docker exec`d
        process. Complements test_native_loop_fast.py's mocked-LLM coverage
        of the same mechanism (tool-list merge, offload, sys.modules stub)
        with proof the real end-to-end pickle round trip works too."""
        from agency.agconfig import agConfig
        from agency.agcontext import agcontext
        from agency.agdata import agdata, agerror
        from agency.agent import agent
        from agency.agllm_backends import agBedrockBackendConfig
        from agency.agtool import agtool
        from agency.agsandbox_backends import agSandboxBackendConfig
        from agency.agskill import agskill
        from agency.agharness_internal.agharness_backends.native import _NativeBackend

        def double_fn(arg):
            if not hasattr(arg, "n"):
                return agerror("n is required")
            return agdata(doubled=arg.n * 2)

        tool = agtool(
            "double",
            "Doubles an integer and returns the result.",
            double_fn,
            params={
                "type": "object",
                "properties": {"n": {"type": "integer", "description": "the number to double"}},
                "required": ["n"],
            },
        )

        cfg = agConfig(
            agSandboxBackendConfig(backend="docker"),
            agBedrockBackendConfig(model="us.anthropic.claude-sonnet-5"),
        )
        sandbox = _make_sandbox(agconfig=cfg)
        ag = agent(agconfig=cfg, sandbox=sandbox)
        skill = agskill(
            name="native_add_tools_test",
            system_prompt=(
                "You have a `double` tool. Use it to double the exact number "
                "the user gives you, then report the result in one short sentence."
            ),
            add_tools=[tool],
        )

        try:
            backend = _NativeBackend(cfg)
            result, ctx, delta = backend.execute(
                ag,
                agcontext(),
                agdata(instruction="Double the number 21."),
                max_steps=5,
                skill=skill,
            )
            raw = result.to_dict()
            assert "error" not in raw, raw
            assert "42" in raw.get("result", "")

            tool_msgs = [m for m in ctx.messages if m.get("role") == "tool"]
            assert len(tool_msgs) >= 1, "model never actually called the double tool"
            assert any("42" in m["content"] for m in tool_msgs), (
                "double tool's real (in-container) output never reached the model"
            )
        finally:
            sandbox.destroy()

    def test_structured_output_end_to_end(self):
        """A genuinely-working submit_output round trip: real container,
        real agLLMTerminus, real agMCPServer, real model -- proving
        _NativeBackend's structured-output support (retired the old
        explicit-reject-everything-structured behavior) actually works,
        not just that it no longer raises."""
        from agency.agconfig import agConfig
        from agency.agcontext import agcontext
        from agency.agdata import agdata
        from agency.agent import agent
        from agency.agllm_backends import agBedrockBackendConfig
        from agency.agsandbox_backends import agSandboxBackendConfig
        from agency.agskill import agskill
        from agency.agharness_internal.agharness_backends.native import _NativeBackend

        cfg = agConfig(
            agSandboxBackendConfig(backend="docker"),
            agBedrockBackendConfig(model="us.anthropic.claude-sonnet-5"),
        )
        sandbox = _make_sandbox(agconfig=cfg)
        ag = agent(agconfig=cfg, sandbox=sandbox)
        skill = agskill(
            name="native_structured_output_test",
            system_prompt="You produce a structured greeting.",
            output_schema=agdata(greeting=str, word_count=int),
        )

        try:
            backend = _NativeBackend(cfg)
            result, ctx, delta = backend.execute(
                ag,
                agcontext(),
                agdata(instruction="Greet the user with exactly 3 words, then report the count."),
                max_steps=8,
                skill=skill,
            )
            raw = result.to_dict()
            assert "error" not in raw, raw
            assert isinstance(raw.get("greeting"), str) and raw["greeting"]
            assert isinstance(raw.get("word_count"), int)
        finally:
            sandbox.destroy()
