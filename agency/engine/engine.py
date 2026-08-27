from __future__ import annotations

from typing import TYPE_CHECKING

from ..harness.protocol import HarnessAttemptRequest, HarnessAttemptResult, PromptPayload
from ..profiler import agprof
from ..sandbox.agsandbox import agSandbox
from .harness_daemon_launcher import ensure_harness_daemon
from .host_servers.host_server_manager import HostServerManager
from .types import ExecutionResult

if TYPE_CHECKING:
    from ..agconfig import agConfig
    from ..agcontext import agcontext
    from ..agdata import agdata
    from ..agent import agent
    from ..agresources import agResourcePool
    from ..agskill import agskill
    from .clients import SandboxInteractionClient


class AgentEngine:
    """Host-side execution owner for one agent."""

    def __init__(self, agent: "agent") -> None:
        self._agent = agent
        self._host_server_manager: "HostServerManager | None" = None
        self._sandbox_interaction_client: "SandboxInteractionClient | None" = None
        self._execution_prompt: "PromptPayload | None" = None

    def set_config(self, agconfig: "agConfig") -> None:
        if self._host_server_manager is not None:
            self._host_server_manager.set_config(agconfig)

    @property
    def host_server_manager(self) -> "HostServerManager":
        if self._host_server_manager is None:
            raise RuntimeError("execute() has not built a HostServerManager yet")
        return self._host_server_manager

    @property
    def llm_transcripts(self) -> list[dict]:
        if self._host_server_manager is None:
            return []
        return self._host_server_manager.llm_transcripts

    def execute(
        self,
        context: "agcontext",
        skill: "agskill",
        skill_input: "agdata",
        resource_pool: "agResourcePool",
        sandbox: agSandbox,
        max_steps: "int | None" = None,
    ) -> ExecutionResult:
        """Execute one request and own its complete sandbox transaction."""

        sandbox_lock = sandbox._lock
        sandbox_lock.acquire()
        try:
            try:
                execution = self._execute_harness(
                    context,
                    skill,
                    skill_input,
                    resource_pool,
                    sandbox,
                    max_steps=max_steps,
                )
            except BaseException:
                self._discard_sandbox(sandbox)
                raise

            if self._execution_failed(execution):
                self._discard_sandbox(sandbox)
                return execution

            try:
                self._commit_sandbox(sandbox)
            except BaseException:
                self._discard_sandbox(sandbox)
                raise
            return execution
        finally:
            sandbox_lock.release()

    def _execute_harness(
        self,
        context: "agcontext",
        skill: "agskill",
        skill_input: "agdata",
        resource_pool: "agResourcePool",
        sandbox: agSandbox,
        max_steps: "int | None" = None,
    ) -> ExecutionResult:
        """Run host services and the sandbox-side harness while locked."""

        # Start connections
        self._host_server_manager = HostServerManager(self._agent, sandbox, skill, resource_pool)
        try:
            host_uds_path = self._host_server_manager.start()
            # start harness manager daemon
            engine_name = str(
                getattr(self._agent, "agname", getattr(self._agent, "harness", "agent"))
            )
            handle = ensure_harness_daemon(
                sandbox,
                host_uds_path,
                engine_name,
                agconfig=self._agent.agconfig,
            )

            # Obtain Host -> Sandbox handle
            self._sandbox_interaction_client = handle.client()

            # build the prompt
            prompt = self._build_prompt_payload(skill, skill_input)
            self._execution_prompt = prompt
            retries_left = skill.max_output_schema_retries
            attempt: "HarnessAttemptResult | None" = None
            prior_session = getattr(self._agent, "_harness_sessions", {}).get(self._agent.harness)
            resume_session_id = prior_session.get("session_id") if prior_session else None
            prior_session_blob_b64 = prior_session.get("blob_b64") if prior_session else None
            while True:
                # Send the request through sandbox interaction server
                attempt = self._run_attempt(
                    prompt,
                    skill=skill,
                    max_steps=max_steps,
                    resume_session_id=resume_session_id,
                    prior_session_blob_b64=prior_session_blob_b64,
                )
                if not attempt.ok:
                    break
                if attempt.session_id:
                    resume_session_id = attempt.session_id
                    prior_session_blob_b64 = attempt.session_blob_b64
                    if attempt.session_blob_b64 is not None:
                        sessions = getattr(self._agent, "_harness_sessions", None)
                        if sessions is None:
                            sessions = {}
                            self._agent._harness_sessions = sessions
                        sessions[self._agent.harness] = {
                            "session_id": attempt.session_id,
                            "blob_b64": attempt.session_blob_b64,
                        }
                if self._recover_structured_output(skill, attempt, sandbox) is not None:
                    break
                missing = self._missing_output_fields(skill)
                if not missing or retries_left <= 0:
                    break
                retries_left -= 1
                prompt = self._build_retry_prompt(
                    missing, system_instruction=prompt.system_instruction
                )
            return self._build_execution_result(context, skill, attempt, sandbox)
        finally:
            if self._sandbox_interaction_client is not None:
                self._sandbox_interaction_client.close()
                self._sandbox_interaction_client = None
            self._host_server_manager.stop()

    def _execution_failed(self, execution: ExecutionResult) -> bool:
        return not execution.ok

    def _discard_sandbox(self, sandbox: "agSandbox") -> None:
        with agprof.span("teardown:discard"):
            sandbox.rm_container()
        self._agent.inbox.put(
            "Note: the previous skill call failed. Its sandbox workspace "
            "changes have been discarded and the workspace has been reverted "
            "to the last successful checkpoint."
        )

    def _commit_sandbox(self, sandbox: "agSandbox") -> None:
        with agprof.span("teardown:commit"):
            try:
                sandbox.commit()
            finally:
                if not sandbox._has_pending_background_work():
                    try:
                        sandbox.stop()
                    except Exception as exc:
                        print(
                            f"[engine] WARNING: post-commit hibernate failed "
                            f"for {self._agent.agname}: {exc}"
                        )

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _build_prompt_payload(self, skill: "agskill", skill_input: "agdata") -> PromptPayload:
        from ..harness import agharness

        return PromptPayload(
            system_instruction=skill._build_system_prompt(),
            user_content=agharness.build_user_turn_prompt(skill, skill_input),
            output_instruction=agharness.build_output_format_instruction(skill),
        )

    def _build_retry_prompt(
        self, missing: "list[str]", *, system_instruction: str = ""
    ) -> PromptPayload:
        return PromptPayload(
            system_instruction=system_instruction,
            user_content=(
                "[HARNESS SYSTEM] You have not yet provided all required output "
                f"fields. Still missing: {missing}. Call the submit_output tool "
                "once for each of them."
            ),
            output_instruction=None,
        )

    def _run_attempt(
        self,
        prompt: PromptPayload,
        *,
        skill: "agskill | None" = None,
        max_steps: "int | None" = None,
        resume_session_id: "str | None" = None,
        prior_session_blob_b64: "str | None" = None,
    ) -> HarnessAttemptResult:
        if self._sandbox_interaction_client is None:
            raise RuntimeError("Harness Manager client is not configured")
        replacement_tools = getattr(skill, "replace_tools", None) if skill is not None else None
        custom_tools = (
            replacement_tools
            if replacement_tools is not None
            else (getattr(skill, "add_tools", None) if skill is not None else None)
        )
        if custom_tools:
            return HarnessAttemptResult(
                ok=False,
                error_message=(
                    "custom Python add_tools/replace_tools cannot cross the harness daemon "
                    "boundary; expose them through MCP instead"
                ),
            )
        request = HarnessAttemptRequest(
            prompt=prompt,
            harness=self._agent.harness,
            max_steps=max_steps,
            resume_session_id=resume_session_id,
            prior_session_blob_b64=prior_session_blob_b64,
            suppress_builtin_tools=replacement_tools is not None,
        )
        return self._sandbox_interaction_client.run_harness_attempt(request)

    def _missing_output_fields(self, skill: "agskill") -> "list[str]":
        if (
            skill.output_schema is None
            or getattr(skill.output_schema, "raw_key", lambda: None)() is not None
        ):
            return []
        collected = self._host_server_manager.host_mcp_server.collected_output()
        required = set(skill.output_schema._data.keys())
        return sorted(required - set(collected.keys()))

    def _recover_structured_output(
        self,
        skill: "agskill",
        attempt: "HarnessAttemptResult",
        sandbox: agSandbox,
    ) -> "agdata | None":
        """Accept the JSON response contract used by non-MCP harnesses."""
        from ..agdata import agerror

        output_schema = skill.output_schema
        if (
            output_schema is None
            or getattr(output_schema, "raw_key", lambda: None)() is not None
            or not hasattr(output_schema, "validate_and_recover")
        ):
            return None
        recovered, _paths = output_schema.validate_and_recover(attempt.final_text, sandbox)
        return None if isinstance(recovered, agerror) else recovered

    def _build_execution_result(
        self,
        context: "agcontext",
        skill: "agskill",
        attempt: "HarnessAttemptResult | None",
        sandbox: agSandbox,
    ) -> ExecutionResult:
        from ..agdata import agdata, agerror

        system_message = {"role": "system", "content": skill._build_system_prompt()}
        if attempt is None or not attempt.ok:
            message = attempt.error_message if attempt is not None else "no attempt was made"
            return ExecutionResult(
                output=agerror(message),
                context=context,
                delta=[system_message],
                ok=False,
                error_message=message,
            )

        output_schema = skill.output_schema
        if output_schema is not None and output_schema.raw_key() is None:
            collected = self._host_server_manager.host_mcp_server.collected_output()
            missing = sorted(set(output_schema._data) - set(collected))
            recovered = self._recover_structured_output(skill, attempt, sandbox)
            if not missing:
                output = agdata(**collected)
                ok = True
                error_message = ""
            elif recovered is not None:
                output = recovered
                ok = True
                error_message = ""
            else:
                message = (
                    "structured output incomplete after retries -- response was not valid JSON "
                    "and submit_output was never called for: " + ", ".join(missing)
                )
                output = agerror(message)
                ok = False
                error_message = message
        else:
            output_key = output_schema.raw_key() if output_schema is not None else "result"
            output = agdata(**{output_key: attempt.final_text})
            ok = True
            error_message = ""

        context.total_input_tokens += attempt.input_tokens
        context.total_output_tokens += attempt.output_tokens

        messages: "list[dict]" = []
        if self._execution_prompt is not None:
            messages.append({"role": "user", "content": self._execution_prompt.user_content})
        messages.append({"role": "assistant", "content": attempt.final_text})
        context.messages.extend(messages)

        return ExecutionResult(
            output=output,
            context=context,
            delta=[system_message, *messages],
            ok=ok,
            error_message=error_message,
        )
