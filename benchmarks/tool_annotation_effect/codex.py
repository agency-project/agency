"""Codex CLI experiments with annotations at Agency's Responses gateway.

Codex owns its tools, history and compaction. Only schemas sent to the executing
model and returned tool arguments are annotated; executable inputs stay intact.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shlex
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from fastapi import Request

from .common import native, read_events
from .execution import agent_config, trace_metrics
from .adapters import SweBenchAdapter


class CodexGateway:
    def __init__(self, model, config, trial, directory, *, client=None):
        from agency.harness.adapters.codex import CodexAdapter
        from agency.llm.agllm import agllm
        from agency.configs.agconfig import agconfig, llmconfig, agentconfig

        self.config = config
        self.model = model
        self.arm = trial["arm"]
        self.token = secrets.token_hex(32)
        self.trace = native("annotations").TraceWriter(
            str(Path(directory) / "events.jsonl"),
            agent_id="agent",
            run_id=trial["trial_id"],
            secrets=(self.token,),
        )
        cfg = agconfig(
            llmconfig(
                provider=model["provider"],
                model=model["model"],
                base_url=os.environ[model["base_url_env"]],
                api_key=os.environ[model["api_key_env"]],
                context_limit=config["context_limit"],
                **model.get("settings", {}),
            ),
            agentconfig(harness="codex"),
        )
        self.adapter = CodexAdapter(cfg)
        self.backend = agllm.for_config(cfg)
        self.started = time.monotonic()
        self.calls = 0
        self.error = ""
        self.original_calls = {}
        self.results_seen = set()
        self.client = client or native("llm_client").LLMClient(
            os.environ[model["base_url_env"]].rstrip("/").removesuffix("/v1"),
            os.environ[model["api_key_env"]],
            timeout_s=config["budgets"]["timeout_s"],
            model_settings={**model.get("settings", {}), "stream_options": {"include_usage": True}},
            observer=self.trace,
            retry_rate_limits=True,
            max_attempts=13,
            deadline=self.started + config["budgets"]["timeout_s"],
            send_internal_kind=False,
        )

    def dispatch(self, body):
        from openai.types.chat import ChatCompletion

        context = self.adapter._format_context_harness_to_agency(body)
        for message in context["messages"]:
            for block in message["blocks"]:
                call_id = block.get("tool_call_id")
                if (
                    block["type"] == "tool_result"
                    and call_id in self.original_calls
                    and call_id not in self.results_seen
                ):
                    self.results_seen.add(call_id)
                    record = self.original_calls[call_id]
                    text = block.get("text", "")
                    self.trace(
                        "tool_result",
                        {
                            "event_id": record["event_id"],
                            "model_tool_call_id": call_id,
                            "result": text,
                            "category": result_category(text),
                            "duration_ns": None,
                        },
                    )
                if block["type"] == "tool_use" and block["id"] in self.original_calls:
                    block["arguments"] = self.original_calls[block["id"]]["arguments"]
        if self.calls >= self.config["budgets"]["max_steps"]:
            self.error = f"exceeded max_steps={self.config['budgets']['max_steps']}"
            raise RuntimeError(self.error)
        if time.monotonic() - self.started >= self.config["budgets"]["timeout_s"]:
            self.error = "dispatch deadline exhausted"
            raise RuntimeError(self.error)
        annotation = native("annotations")
        context["tools"] = annotation.augment_schemas(context.get("tools") or [], self.arm)
        instruction = annotation.instruction(self.arm)
        if instruction:
            if context["messages"] and context["messages"][0]["role"] in ("system", "developer"):
                context["messages"][0]["blocks"].append(
                    {
                        "type": "text",
                        "index": len(context["messages"][0]["blocks"]),
                        "text": instruction,
                    }
                )
            else:
                context["messages"].insert(
                    0,
                    {
                        "role": "system",
                        "blocks": [{"type": "text", "index": 0, "text": instruction}],
                    },
                )
        if self.calls == 0:
            self.trace(
                "treatment",
                {
                    "arm": self.arm,
                    "harness": "codex",
                    "version": annotation.VERSION,
                    "schemas": context["tools"],
                    "instruction": instruction,
                    "codex_version": self.config["codex_version"],
                },
            )
        self.calls += 1
        request = self.backend._format_context_agency_to_backend(context)
        self.trace("codex_request", {"exchange": self.calls - 1, "context": context})
        response = self.client.dispatch(
            self.model["model"], request["messages"], request.get("tools")
        )
        if response.get("error"):
            self.error = response["error"]
            raise RuntimeError(self.error)
        self.trace("model_exchange", {"exchange": self.calls - 1, "response": response})
        sdk = ChatCompletion.model_validate(
            {
                "id": "agency-codex",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": response["message"],
                        "finish_reason": "tool_calls"
                        if response["message"].get("tool_calls")
                        else "stop",
                    }
                ],
                "usage": response.get("usage"),
            }
        )
        result = self.backend._format_context_backend_to_agency(sdk)
        calls = [block for block in result["message"]["blocks"] if block["type"] == "tool_use"]
        for block in calls:
            original = block["arguments"]
            executable, metadata = annotation.extract(original, self.arm)
            event_id = annotation.event_id()
            self.original_calls[block["id"]] = {"event_id": event_id, "arguments": original}
            try:
                arguments = json.loads(executable)
            except ValueError:
                arguments = {"raw": executable}
            self.trace(
                "tool_annotation",
                {
                    "event_id": event_id,
                    "model_tool_call_id": block["id"],
                    "tool_name": block["name"],
                    "arguments": arguments,
                    "annotation": metadata,
                    "exchange": self.calls - 1,
                    "batch_size": len(calls),
                },
            )
            block["arguments"] = executable
        return result

    @contextmanager
    def serve(self):
        import uvicorn
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse, StreamingResponse
        from agency.harness.adapters.codex import _responses_tool_routes

        app = FastAPI()

        @app.post("/v1/responses")
        async def responses(request: Request):
            if request.headers.get("authorization") != "Bearer " + self.token:
                return JSONResponse({"error": {"message": "invalid token"}}, status_code=401)
            body = await request.json()
            try:
                result = await asyncio.to_thread(self.dispatch, body)
            except Exception as error:
                self.error = native("annotations").redact(str(error))
                return JSONResponse({"error": {"message": self.error}}, status_code=400)
            routes = _responses_tool_routes(body)
            if body.get("stream"):
                frames = self.adapter._format_agency_stream_to_harness(
                    [{"type": "done", **result}], self.model["model"], tool_routes=routes
                )
                return StreamingResponse(frames, media_type="text/event-stream")
            return JSONResponse(
                self.adapter._format_context_agency_to_harness(
                    result, self.model["model"], tool_routes=routes
                )
            )

        server = uvicorn.Server(
            uvicorn.Config(app, host="0.0.0.0", port=0, log_level="warning", access_log=False)
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        try:
            while not server.started:
                if not thread.is_alive() or time.monotonic() >= deadline:
                    raise RuntimeError("Codex gateway failed to start")
                time.sleep(0.01)
            port = server.servers[0].sockets[0].getsockname()[1]
            yield f"http://{self.config['codex_gateway_host']}:{port}"
        finally:
            server.should_exit = True
            self.client.close()
            thread.join(timeout=10)


def result_category(text):
    import re

    if re.search(r'(?:Process exited with code|exit_code["\s:]+)\s*[1-9]', text) or text.startswith(
        "Error:"
    ):
        return "error"
    return "ok"


def verified_binary(config):
    import hashlib

    path = Path(config["codex_binary_path"])
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != config["codex_binary_sha256"]:
        raise ValueError("Codex binary differs from the frozen plan")
    return path


def codex_files(gateway, base_url, prompt, config):
    home = "/opt/agency-codex-state"
    toml = (
        f'model = {json.dumps(gateway.model["model"])}\nmodel_provider = "agency"\n'
        f"model_context_window = {config['context_limit']}\n"
        'model_reasoning_effort = "none"\ncheck_for_update_on_startup = false\n'
        '[model_providers.agency]\nname = "Agency"\nwire_api = "responses"\n'
        f'base_url = {json.dumps(base_url + "/v1")}\nenv_key = "AGENCY_PROXY_API_KEY"\n'
        "request_max_retries = 0\nstream_max_retries = 0\n"
        "[features]\ncode_mode = false\ncode_mode_host = false\nmulti_agent = false\ngoals = false\n"
    )
    command = (
        f"CODEX_HOME={home} AGENCY_PROXY_API_KEY={shlex.quote(gateway.token)} "
        "/opt/agency-codex/codex exec --skip-git-repo-check "
        "--dangerously-bypass-approvals-and-sandbox --json "
        f"--output-last-message {home}/final.txt - < {home}/prompt.txt "
        f"> {home}/stdout.jsonl 2> {home}/stderr.txt"
    )
    return {home + "/config.toml": toml, home + "/prompt.txt": prompt}, command


def execute_swebench(task, trial, model, config, directory):
    from .sandbox import EpisodeSandbox

    directory = Path(directory)
    began = time.monotonic()
    cfg = agent_config({**config, "base_image": task["image"]}, model, trial, directory, "agent")
    cfg.agent.harness = "codex"
    cfg.harness_adapter.binary_path = "/opt/agency-codex/codex"
    # This path runs the Codex CLI directly, so Agency's native loop is never launched.
    sandbox = EpisodeSandbox("codex-" + trial["trial_id"], agconfig=cfg)
    try:
        sandbox.write_file_bytes("/opt/agency-codex/codex", verified_binary(config).read_bytes())
        version, code = sandbox.exec(
            "chmod 755 /opt/agency-codex/codex && /opt/agency-codex/codex --version"
        )
        if code or version.strip() != "codex-cli " + config["codex_version"]:
            raise RuntimeError("Prepared Codex binary failed its version check")
        prompt = SweBenchAdapter().setup(task, sandbox)
        setup_seconds = time.monotonic() - began
        started = time.monotonic()
        gateway = CodexGateway(model, config, trial, directory)
        with gateway.serve() as base_url:
            files, command = codex_files(
                gateway, base_url, config["system_prompt"] + "\n\n" + prompt, config
            )
            for path, content in files.items():
                sandbox.write_file(path, content)
            try:
                _, code = sandbox.exec(command, timeout=config["budgets"]["timeout_s"])
                failure = "infrastructure" if code and not gateway.error else None
            except Exception as error:
                gateway.error = str(error)
                failure = (
                    "budget"
                    if isinstance(error, TimeoutError) or "timed out" in str(error)
                    else "infrastructure"
                )
        agent_seconds = time.monotonic() - started
        for name in ("stdout.jsonl", "stderr.txt", "final.txt"):
            try:
                (directory / name).write_text(sandbox.read_file("/opt/agency-codex-state/" + name))
            except (RuntimeError, OSError):
                pass
        patch = SweBenchAdapter().extract_patch(sandbox)
        (directory / "prediction.patch").write_text(patch)
        SweBenchAdapter().export_prediction(
            task, patch, model["id"], directory / "prediction.jsonl"
        )
        events = read_events(directory / "events.jsonl")
        if "max_steps=" in gateway.error or "deadline exhausted" in gateway.error:
            failure = "budget"
        elif gateway.error:
            failure = failure or "infrastructure"
        final = directory / "final.txt"
        return {
            "events": events,
            "metrics": trace_metrics(events),
            "failure": failure,
            "error": gateway.error,
            "harness": "codex",
            "agent_executions": 1,
            "final_text": final.read_text() if final.exists() else "",
            "setup_seconds": setup_seconds,
            "agent_seconds": agent_seconds,
            "end_to_end_seconds": time.monotonic() - began,
        }
    finally:
        sandbox.destroy()
