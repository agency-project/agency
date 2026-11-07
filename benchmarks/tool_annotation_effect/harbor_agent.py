"""Harbor 0.23.0 external agent running Agency's real native ReAct loop.

Harbor owns the environment and verifier. Built-ins execute there, never in a
second Agency sandbox. Import this module only from a Harbor process.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import time

from harbor.agents.base import BaseAgent
from harbor.agents.installed.base import NonZeroAgentExitCodeError

from .common import ROOT, atomic_json, native, read_events
from .execution import trace_metrics


class AgencyNativeAgent(BaseAgent):
    def __init__(
        self,
        *args,
        arm="baseline",
        model_config=None,
        experiment_config=None,
        run_id=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.arm = arm
        self.model_config = model_config or {}
        self.experiment_config = experiment_config or {}
        self.run_id = run_id or "harbor"

    @staticmethod
    def name():
        return "agency-native-annotations"

    def version(self):
        return "1"

    async def setup(self, environment):
        if getattr(self, "mcp_servers", []):
            raise RuntimeError(
                "Task-provided Harbor MCP servers are not supported by this adapter; native Agency MCP discovery is supported"
            )
        python = self.experiment_config.get("tool_python_path", "python3")
        result = await environment.exec("command -v " + shlex.quote(python) + " && command -v bash")
        if result.return_code != 0:
            raise RuntimeError(
                "Agency native tools need python3 and bash in the Harbor task image; prepare the image explicitly"
            )
        await environment.upload_file(
            ROOT / "agency/native_harness/tools.py", "/tmp/agency_native_tools.py"
        )
        await environment.exec("mkdir -p /tmp/agency-tool-output")

    async def run(self, instruction, environment, context):
        event_loop = asyncio.get_running_loop()
        config = self.experiment_config
        model = self.model_config
        base_url = os.environ[model["base_url_env"]].rstrip("/").removesuffix("/v1")
        api_key = os.environ[model["api_key_env"]]
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        trace = native("annotations").TraceWriter(
            str(self.logs_dir / "events.jsonl"), agent_id="terminal-agent", run_id=self.run_id
        )
        tools = native("tools")
        remote_script = (
            "import importlib.util,base64,sys; "
            "s=importlib.util.spec_from_file_location('agency_tools','/tmp/agency_native_tools.py'); "
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m); "
            "r=m.TOOL_DISPATCH[sys.argv[1]](base64.b64decode(sys.argv[2]).decode()); "
            "print(m.offload_if_oversized(sys.argv[1],m.uuid.uuid4().hex,r,'/tmp/agency-tool-output'))"
        )

        def remote_call(name, arguments):
            encoded = base64.b64encode(arguments.encode()).decode()
            python = config.get("tool_python_path", "python3")
            command = shlex.quote(python) + " -c " + shlex.quote(remote_script)
            command += " " + shlex.quote(name) + " " + shlex.quote(encoded)
            future = asyncio.run_coroutine_threadsafe(
                environment.exec(command, timeout_sec=config["budgets"]["timeout_s"]), event_loop
            )
            response = future.result(timeout=config["budgets"]["timeout_s"] + 10)
            if response.return_code:
                return json.dumps({"error": response.stderr or response.stdout})
            return response.stdout.strip()

        dispatch = {
            name: (lambda arguments, name=name: remote_call(name, arguments))
            for name in tools.TOOL_DISPATCH
        }
        started = time.monotonic()
        llm = native("llm_client").LLMClient(
            base_url,
            api_key,
            timeout_s=config["budgets"]["timeout_s"],
            model_settings={**model.get("settings", {}), "stream_options": {"include_usage": True}},
            observer=trace,
            retry_rate_limits=True,
            max_attempts=13,
            deadline=started + config["budgets"]["timeout_s"],
            send_internal_kind=False,
        )
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    native("react_loop").run_react_loop,
                    [
                        {
                            "role": "system",
                            "content": config.get(
                                "system_prompt", "Complete the task using the available tools."
                            ),
                        },
                        {"role": "user", "content": instruction},
                    ],
                    self.model_name,
                    llm,
                    annotation_arm=self.arm,
                    observer=trace,
                    toolset=(list(tools.BUILTIN_TOOL_SCHEMAS.values()), dispatch),
                    context_limit=config["context_limit"],
                    max_steps=config["budgets"]["max_steps"],
                    offload_dir=str(self.logs_dir / "tool-outputs"),
                ),
                timeout=config["budgets"]["timeout_s"],
            )
            atomic_json(
                self.logs_dir / "native-result.json",
                {
                    "status": result.status,
                    "final_text": result.final_text,
                    "error": result.message,
                    "agent_seconds": time.monotonic() - started,
                },
            )
            metrics = trace_metrics(read_events(self.logs_dir / "events.jsonl"))
            context.n_input_tokens = metrics["input_tokens"]
            context.n_output_tokens = metrics["output_tokens"]
            context.metadata = {"annotation_arm": self.arm, "run_id": self.run_id, **metrics}
            if result.status != "done":
                if (
                    "exceeded max_steps=" in result.message
                    or "dispatch deadline exhausted" in result.message
                ):
                    # Harbor catches this installed-agent failure and still runs
                    # its verifier on the attempted environment.
                    raise NonZeroAgentExitCodeError(result.message)
                raise RuntimeError(result.message)
        finally:
            llm.close()


class AgencyCodexAgent(AgencyNativeAgent):
    @staticmethod
    def name():
        return "agency-codex-annotations"

    async def setup(self, environment):
        if getattr(self, "mcp_servers", []):
            raise RuntimeError("Task-provided MCP servers are not supported by this adapter")
        from .codex import verified_binary

        await environment.exec("mkdir -p /opt/agency-codex /opt/agency-codex-state")
        await environment.upload_file(
            verified_binary(self.experiment_config), "/opt/agency-codex/codex"
        )
        await environment.exec("chmod 755 /opt/agency-codex/codex")
        response = await environment.exec("/opt/agency-codex/codex --version")
        expected = self.experiment_config["codex_version"]
        if response.return_code or expected not in response.stdout:
            raise RuntimeError(f"Expected prepared Codex {expected} executable")
        await environment.exec("mkdir -p /opt/agency-codex-state")

    async def run(self, instruction, environment, context):
        from .codex import CodexGateway, codex_files

        started = time.monotonic()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        gateway = CodexGateway(
            self.model_config,
            self.experiment_config,
            {"arm": self.arm, "trial_id": self.run_id},
            self.logs_dir,
        )
        with gateway.serve() as base_url:
            files, command = codex_files(
                gateway,
                base_url,
                self.experiment_config["system_prompt"] + "\n\n" + instruction,
                self.experiment_config,
            )
            for number, (path, content) in enumerate(files.items()):
                local = self.logs_dir / f"upload-{number}"
                local.write_text(content)
                await environment.upload_file(local, path)
                local.unlink()
            response = await environment.exec(
                command, timeout_sec=self.experiment_config["budgets"]["timeout_s"]
            )
        for name in ("stdout.jsonl", "stderr.txt", "final.txt"):
            response_file = await environment.exec("cat /opt/agency-codex-state/" + name)
            (self.logs_dir / name).write_text(response_file.stdout or "")
        final = (self.logs_dir / "final.txt").read_text()
        error = gateway.error
        if response.return_code and not error:
            error = "Codex exited with code " + str(response.return_code)
        atomic_json(
            self.logs_dir / "codex-result.json",
            {
                "status": "error" if error else "done",
                "error": error,
                "final_text": final,
                "agent_seconds": time.monotonic() - started,
                "harness": "codex",
            },
        )
        metrics = trace_metrics(read_events(self.logs_dir / "events.jsonl"))
        context.n_input_tokens = metrics["input_tokens"]
        context.n_output_tokens = metrics["output_tokens"]
        context.metadata = {"harness": "codex", "annotation_arm": self.arm, **metrics}
        if "max_steps=" in error or "deadline exhausted" in error:
            raise NonZeroAgentExitCodeError(error)
        if error:
            raise RuntimeError(error)
