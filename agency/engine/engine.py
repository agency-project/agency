from __future__ import annotations

from typing import TYPE_CHECKING

from ..harness.protocol import HarnessAttemptRequest, HarnessAttemptResult, PromptPayload
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

    def execute(
        self,
        context: "agcontext",
        skill: "agskill",
        skill_input: "agdata",
        resource_pool: "agResourcePool",
        max_steps: "int | None" = None,
    ) -> ExecutionResult:
        """Execute one declarative skill request through the host services."""

        # Start connections
        self._host_server_manager = HostServerManager(
            self._agent, self._agent.sandbox, skill, resource_pool
        )
        host_uds_path = self._host_server_manager.start()
        try:
            # start harness manager daemon
            engine_name = str(
                getattr(self._agent, "agname", getattr(self._agent, "harness", "agent"))
            )
            handle = ensure_harness_daemon(
                self._agent.sandbox,
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
                missing = self._missing_output_fields(skill)
                if not missing or retries_left <= 0:
                    break
                retries_left -= 1
                prompt = self._build_retry_prompt(
                    missing, system_instruction=prompt.system_instruction
                )
            return self._build_execution_result(context, skill, attempt)
        finally:
            if self._sandbox_interaction_client is not None:
                self._sandbox_interaction_client.close()
                self._sandbox_interaction_client = None
            self._host_server_manager.stop()

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
        max_steps: "int | None" = None,
        resume_session_id: "str | None" = None,
        prior_session_blob_b64: "str | None" = None,
    ) -> HarnessAttemptResult:
        if self._sandbox_interaction_client is None:
            raise RuntimeError("Harness Manager client is not configured")
        request = HarnessAttemptRequest(
            prompt=prompt,
            harness=self._agent.harness,
            max_steps=max_steps,
            resume_session_id=resume_session_id,
            prior_session_blob_b64=prior_session_blob_b64,
        )
        return self._sandbox_interaction_client.run_harness_attempt(request)

    def _run_attempt(self, prompt: PromptPayload) -> HarnessAttemptResult:
        interaction = self._host_server_manager.interaction_server
        waiter = interaction.expect_attempt_result()
        try:
            self._send_run_attempt(prompt)
            return interaction.wait_for_attempt_result(waiter)
        except BaseException:
            interaction.cancel_expected_attempt(waiter)
            raise

    def _send_run_attempt(self, prompt: PromptPayload) -> None:
        """Send one attempt directly to the sandbox daemon server.

        The host-side protocol boundary is explicit now; the daemon client
        will implement it once the sandbox-side server exists.
        """
        raise NotImplementedError("sandbox daemon client is not configured")

    def _missing_output_fields(self, skill: "agskill") -> "list[str]":
        if skill.output_schema is None:
            return []
        collected = self._host_server_manager.host_mcp_server.collected_output()
        required = set(skill.output_schema._data.keys())
        return sorted(required - set(collected.keys()))

    def _build_execution_result(
        self, context: "agcontext", skill: "agskill", attempt: "HarnessAttemptResult | None"
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
            if missing:
                message = (
                    "structured output incomplete after retries -- submit_output was never "
                    "called for: " + ", ".join(missing)
                )
                output = agerror(message)
                ok = False
                error_message = message
            else:
                output = agdata(**collected)
                ok = True
                error_message = ""
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
