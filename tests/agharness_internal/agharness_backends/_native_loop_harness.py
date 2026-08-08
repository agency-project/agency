"""Fast, no-Docker test seam for the native engine's react-loop logic.

Runs `_native_in_container_entrypoint.py`'s `_run_react_loop()` directly,
in-process -- no `docker exec`, no subprocess, no container. This exercises
the REAL loop/tool-dispatch/compaction/retry/offload logic (not a fake),
against REAL (but local) `agLLMTerminus`/`agMCPServer`/`agHarnessMessenger`
instances over real Unix-domain sockets -- only the actual LLM provider
call is mocked (`fake_ag.llm.backend.make_client.return_value = fake_client`,
the same convention every other terminus-facing test in this repo uses).

Complements `test_native.py`'s real-Docker tests: those prove the
launch+bridge mechanism itself works against a genuine container; this
proves the loop's OWN behavior (schema retry, compaction, pause, tool-
output offload, max_steps, tool-calling mechanics, ...) fast, without
needing Docker at all -- the same fine-grained, mocked-LLM coverage
`execute_react()`'s direct-call tests used to give the old host-process
loop, now against the loop that actually replaces it.

The one piece of production code this depends on is
`_native_in_container_entrypoint.py`'s own `_AGENCY_PACKAGE_CONTAINER_MOUNT`
env-var override (see that module's comment) -- pointed at this repo
checkout's own `agency/` directory instead of the container mount path, so
`_load_agtool_pure()`/`_load_agllm_pure()` find real files to load. Inert
in every real (container) launch, which never sets that env var.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENTRYPOINT_PATH = (
    _REPO_ROOT
    / "agency"
    / "agharness_internal"
    / "agharness_backends"
    / "_native_in_container_entrypoint.py"
)


def load_entrypoint_module():
    """A fresh module object per call (never cached in sys.modules) --
    matches how a real launch only ever loads this file once per
    container process, while giving each test its own isolated
    module-level state (e.g. `_todo_store`) rather than leaking it
    between tests the way a single shared import would."""
    os.environ.setdefault("AGENCY_PACKAGE_CONTAINER_MOUNT", str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "_native_in_container_entrypoint_test", str(_ENTRYPOINT_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Mocked streaming-chunk builders -- real ChatCompletionChunk objects, since
# these cross a genuine HTTP boundary (terminus -> entrypoint) as SSE
# frames and get parsed back out; a hand-rolled fake lacking real pydantic
# fields wouldn't round-trip. Mirrors test_native.py's own helpers of the
# same name/shape (kept as a separate small copy here rather than a shared
# import, so this harness has no dependency on that file's own docker-gated
# test classes).
# ---------------------------------------------------------------------------


def content_chunks(text: str, finish_reason: str = "stop"):
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


def tool_call_chunks(
    tool_name: str, arguments: dict, call_id: str = "call_1", finish_reason: str = "tool_calls"
):
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


class NativeLoopHarness:
    """Owns the real terminus/mcp_server/messenger instances (and the
    freshly-loaded entrypoint module) for one test. `stop()` tears down
    every UDS server started -- call it in a `finally:`, mirroring
    test_native.py's own `sb.destroy()`/`terminus.stop_uds()` pattern."""

    def __init__(
        self, *, with_mcp: bool = False, with_messenger: bool = False, mcp_skill=None
    ) -> None:
        from agency.agharness_internal.agllm_terminus import agLLMTerminus

        self.module = load_entrypoint_module()
        self.token = "tok"

        self.fake_client = MagicMock()
        self.fake_ag = MagicMock()
        self.fake_ag.llm.backend.make_client.return_value = self.fake_client

        self.terminus = agLLMTerminus()
        self.terminus.register(self.token, self.fake_ag)

        self.mcp_server = None
        if with_mcp:
            from agency.agharness_internal.agmcp_server import agMCPServer

            self.mcp_server = agMCPServer()
            # A plain MagicMock() skill is fine for tools (reserve_cpu/
            # cpu_release/daemon_release/ask_human) that don't touch
            # skill.output_schema -- pass a real `agskill` via `mcp_skill`
            # for submit_output tests, which validate against a real schema.
            self.mcp_server.register(self.token, self.fake_ag, mcp_skill or MagicMock())

        self.messenger = None
        if with_messenger:
            from agency.agharness_internal.agharness_messenger import agHarnessMessenger

            self.messenger = agHarnessMessenger()
            self.messenger.register(self.token, self.fake_ag)

    def base_request(self, messages: list, max_steps: int = 5) -> dict:
        """A `run_react_loop()`-shaped request dict, pre-filled with this
        harness's own token/sockets -- callers can still override/extend
        any key (e.g. add `"tools"`) before calling `run()`."""
        req = {
            "token": self.token,
            "terminus_sock": self.terminus.ensure_uds_started(),
            "model": "m",
            "messages": messages,
            "max_steps": max_steps,
        }
        if self.mcp_server is not None:
            req["mcp_server_sock"] = self.mcp_server.ensure_uds_started()
        if self.messenger is not None:
            req["messenger_sock"] = self.messenger.ensure_uds_started()
        return req

    def run(self, req: dict) -> dict:
        """Calls `_run_react_loop()` directly, in-process -- no socket
        framing at all (that framing only exists for the real
        socketserver-based `_Handler.handle()`, which this bypasses
        entirely, same as native.py's own client bypasses nothing but the
        container hop)."""
        return self.module._run_react_loop(req)

    def stop(self) -> None:
        self.terminus.stop_uds()
        if self.mcp_server is not None:
            self.mcp_server.stop_uds()
        if self.messenger is not None:
            self.messenger.stop_uds()
