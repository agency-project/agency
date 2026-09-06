"""Real MCP over HTTP with the Harness Manager in a separate process."""

from __future__ import annotations

import asyncio
import base64
import multiprocessing
import os
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from types import SimpleNamespace

import cloudpickle
import httpx
import httpx2
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agency.agdata import agdata
from agency.agskill import agskill
from agency.agtool import agtool
from agency.configs.agconfig import agconfig, dataloggerconfig, hostserverconfig, llmconfig
from agency.engine.clients import SandboxInteractionClient
from agency.engine.engine import AgentEngine
from agency.engine.host_servers.host_server_manager import HostServerManager
from agency.harness.daemon import HarnessManager
from agency.harness.protocol import HarnessAttemptRequest, HarnessAttemptResult, PromptPayload
from agency.llm.usage_tracker import LlmUsageTracker
from agency.native_harness.mcp_client import _decode_tool_result
from agency.observability.agdatalogger import agDataLogger


def _serve_manager(directory, ready, release, stop):
    def attempt(request):
        ready.put(request.attempt_token)
        if not release.wait(20):
            return HarnessAttemptResult(ok=False, error_message="test did not release attempt")
        if request.prompt.user_content == "raise":
            raise RuntimeError("adapter failed")
        return HarnessAttemptResult(ok=request.prompt.user_content != "fail", final_text="done")

    manager = HarnessManager(
        f"{directory}/sandbox.sock",
        f"{directory}/host.sock",
        "sandbox-mcp-test",
        "native",
        attempt_handler=attempt,
        harness_api_port=0,
    )
    try:
        manager.start()
        ready.put(manager._harness_api.base_url)
        stop.wait(60)
    finally:
        manager.stop()


@pytest.fixture
def sandbox_manager():
    ctx = multiprocessing.get_context("spawn")
    ready, release, stop = ctx.Queue(), ctx.Event(), ctx.Event()
    host_calls = []
    host_lock = threading.Lock()

    def host_tool(arg):
        # This deliberately cannot be cloudpickled: it belongs on the host.
        with host_lock:
            host_calls.append(arg.to_dict())
        return agdata(pid=os.getpid(), value="host")

    with tempfile.TemporaryDirectory(prefix="agency-mcp-", dir="/tmp") as directory:
        config = agconfig(
            llmconfig(model="test-model"),
            hostserverconfig(uds_path=f"{directory}/host.sock"),
            dataloggerconfig(db_path=f"{directory}/agent.db"),
        )
        agent = SimpleNamespace(
            agconfig=config,
            harness="native",
            data_logger=agDataLogger(config),
            llm_usage_tracker=LlmUsageTracker(),
        )
        skill = agskill("host", "test", add_host_mcp_tools=[agtool("host_probe", "", host_tool)])
        host = HostServerManager(agent, SimpleNamespace(), skill, SimpleNamespace())
        process = ctx.Process(target=_serve_manager, args=(directory, ready, release, stop))
        client = SandboxInteractionClient(f"{directory}/sandbox.sock", timeout_s=25)
        engine = AgentEngine(agent)
        engine._host_server_manager = host
        engine._sandbox_interaction_client = client
        host.start()
        process.start()
        try:
            base_url = ready.get(timeout=15)

            @contextmanager
            def attempt(tools=None, *, fail=False):
                release.clear()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        engine._run_attempt,
                        PromptPayload("system", "fail" if fail else "user"),
                        sandbox_mcp_tools=tools,
                    )
                    try:
                        token = ready.get(timeout=10)
                        yield token
                    finally:
                        release.set()
                        result = future.result(timeout=10)
                        assert result.ok is not fail, result.error_message

            yield SimpleNamespace(
                attempt=attempt,
                client=client,
                base_url=base_url,
                pid=process.pid,
                host_calls=host_calls,
                engine=engine,
                ready=ready,
                release=release,
            )
        finally:
            release.set()
            stop.set()
            process.join(timeout=15)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            client.close()
            host.stop()
            ready.close()
        assert process.exitcode == 0


@asynccontextmanager
async def _session(url, token):
    async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}) as http:
        async with streamable_http_client(url, http_client=http) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


def test_sandbox_tools_execute_in_daemon_with_persistent_state_and_host_proxy(sandbox_manager):
    runtime = sandbox_manager
    captured = []
    initialized = []
    seed = 40

    def new_state():
        initialized.append(1)
        return {"count": seed}

    def counter(arg, state):
        captured.append(arg.amount)
        state["count"] += arg.amount
        return agdata(
            pid=os.getpid(),
            count=state["count"],
            closure_calls=len(captured),
            initializations=len(initialized),
        )

    tool = agtool(
        "foo",
        "Count in the sandbox",
        counter,
        params={
            "type": "object",
            "properties": {"amount": {"type": "integer"}},
            "required": ["amount"],
        },
        persistent_vars={"state": new_state},
    )

    async def exercise(token):
        async with _session(f"{runtime.base_url}/sandbox/mcp", token) as session:
            listed = await session.list_tools()
            assert [t.name for t in listed.tools] == ["foo"]
            assert listed.tools[0].description == tool.description
            assert listed.tools[0].input_schema["properties"]["amount"]["type"] == "integer"
            for count, calls in [(42, 1), (44, 2)]:
                result = await session.call_tool("foo", {"amount": 2})
                assert not result.is_error
                assert _decode_tool_result(result) == {
                    "pid": runtime.pid,
                    "count": count,
                    "closure_calls": calls,
                    "initializations": 1,
                }
        async with _session(f"{runtime.base_url}/mcp", token) as session:
            assert "foo" not in {t.name for t in (await session.list_tools()).tools}
            result = await session.call_tool("host_probe", {})
            assert not result.is_error
            assert _decode_tool_result(result) == {"pid": os.getpid(), "value": "host"}

    for _ in range(2):
        with runtime.attempt([tool]) as token:
            asyncio.run(exercise(token))
    assert runtime.pid != os.getpid()
    assert captured == []
    assert initialized == []
    assert runtime.host_calls == [{}, {}]
    assert (
        httpx.post(
            f"{runtime.base_url}/sandbox/mcp", headers={"Authorization": f"Bearer {token}"}
        ).status_code
        == 401
    )


@pytest.mark.parametrize("fail", [False, True])
def test_attempt_cleanup_removes_tools_and_rejects_old_tokens(sandbox_manager, fail):
    runtime = sandbox_manager
    foo = agtool("foo", "", lambda arg: agdata(value="foo"))
    bar = agtool("bar", "", lambda arg: agdata(value="bar"))

    async def check_tool(token, name):
        async with _session(f"{runtime.base_url}/sandbox/mcp", token) as session:
            assert [t.name for t in (await session.list_tools()).tools] == [name]
            result = await session.call_tool(name, {})
            assert _decode_tool_result(result) == {"value": name}
            if name == "bar":
                assert (await session.call_tool("foo", {})).is_error

    with runtime.attempt([foo], fail=fail) as old_token:
        asyncio.run(check_tool(old_token, "foo"))
    with runtime.attempt() as no_tools_token:
        assert (
            httpx.post(
                f"{runtime.base_url}/sandbox/mcp",
                headers={"Authorization": f"Bearer {no_tools_token}"},
            ).status_code
            == 401
        )
    with runtime.attempt([bar]) as token:
        for rejected in (old_token, no_tools_token, "unknown", ""):
            assert (
                httpx.post(
                    f"{runtime.base_url}/sandbox/mcp",
                    headers={"Authorization": f"Bearer {rejected}"} if rejected else {},
                ).status_code
                == 401
            )
        asyncio.run(check_tool(token, "bar"))


@pytest.mark.parametrize(
    "failure", ["base64", "pickle", "restore", "shape", "duplicate", "registration"]
)
def test_sandbox_setup_failure_returns_clean_attempt_error_and_allows_next_attempt(
    sandbox_manager,
    failure,
):
    runtime = sandbox_manager
    tool = agtool("foo", "", lambda arg: arg)

    class BrokenRestore(agtool):
        def __setstate__(self, state):
            raise ValueError("private callable contents")

    payloads = {
        "base64": "invalid base64!",
        "pickle": base64.b64encode(b"private broken pickle").decode(),
        "restore": base64.b64encode(
            cloudpickle.dumps([BrokenRestore("restore", "", lambda arg: arg)])
        ).decode(),
        "shape": base64.b64encode(cloudpickle.dumps([123])).decode(),
        "duplicate": base64.b64encode(cloudpickle.dumps([tool, tool])).decode(),
        "registration": base64.b64encode(
            cloudpickle.dumps(
                [
                    tool,
                    agtool(
                        "bad_schema", "", lambda arg: arg, params={"properties": {"bad-name": {}}}
                    ),
                ]
            )
        ).decode(),
    }
    token = uuid.uuid4().hex
    result = runtime.client.run_harness_attempt(
        HarnessAttemptRequest(
            PromptPayload("system", "user"),
            "native",
            attempt_token=token,
            sandbox_mcp_tools_b64=payloads[failure],
        )
    )
    assert not result.ok
    assert "sandbox MCP setup failed" in result.error_message
    assert "private broken pickle" not in result.error_message
    assert "private callable contents" not in result.error_message
    assert payloads[failure] not in result.error_message
    assert (
        httpx.post(
            f"{runtime.base_url}/sandbox/mcp", headers={"Authorization": f"Bearer {token}"}
        ).status_code
        == 401
    )
    with runtime.attempt():
        pass


def test_sandbox_tools_are_removed_when_attempt_handler_raises(sandbox_manager):
    runtime = sandbox_manager
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            runtime.engine._run_attempt,
            PromptPayload("system", "raise"),
            sandbox_mcp_tools=[agtool("foo", "", lambda arg: arg)],
        )
        token = runtime.ready.get(timeout=10)
        runtime.release.set()
        with pytest.raises(httpx.HTTPStatusError):
            future.result(timeout=10)
    assert (
        httpx.post(
            f"{runtime.base_url}/sandbox/mcp", headers={"Authorization": f"Bearer {token}"}
        ).status_code
        == 401
    )
    with runtime.attempt():
        pass


def test_attempt_cleanup_drains_running_sandbox_tool_before_returning(sandbox_manager, tmp_path):
    runtime = sandbox_manager
    started = str(tmp_path / "started")
    release = str(tmp_path / "release")
    finished = str(tmp_path / "finished")

    def blocking_tool(arg):
        Path(started).touch()
        deadline = time.monotonic() + 10
        while not Path(release).exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("test did not release tool")
            time.sleep(0.01)
        Path(finished).touch()
        return agdata(done=True)

    async def call(token):
        async with _session(f"{runtime.base_url}/sandbox/mcp", token) as session:
            await session.list_tools()
            return await session.call_tool("blocking", {})

    with ThreadPoolExecutor(max_workers=2) as executor:
        attempt = executor.submit(
            runtime.engine._run_attempt,
            PromptPayload("system", "user"),
            sandbox_mcp_tools=[agtool("blocking", "", blocking_tool)],
        )
        token = runtime.ready.get(timeout=10)
        call_future = executor.submit(asyncio.run, call(token))
        try:
            deadline = time.monotonic() + 5
            while not Path(started).exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert Path(started).exists()
            runtime.release.set()
            # Wait until cleanup invalidates the endpoint, while the admitted
            # Python call still holds up completion of the attempt RPC.
            deadline = time.monotonic() + 5
            with httpx.Client() as http:
                while time.monotonic() < deadline:
                    response = http.get(
                        f"{runtime.base_url}/sandbox/mcp",
                        headers={"Authorization": f"Bearer {token}"},
                    )
                    if response.status_code == 401:
                        break
                    time.sleep(0.01)
            assert response.status_code == 401
            assert not attempt.done()
            assert not Path(finished).exists()
        finally:
            Path(release).touch()
            runtime.release.set()
        assert attempt.result(timeout=10).ok
        assert Path(finished).exists()
        # The session may close during retirement, cancelling its response.
        # Regardless, the Python implementation must finish before the RPC.
        call_future.exception(timeout=10)
    with runtime.attempt():
        pass
