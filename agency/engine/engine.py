from __future__ import annotations

import threading
import uuid
from typing import TYPE_CHECKING

from ..harness.protocol import HarnessAttemptRequest, HarnessAttemptResult, PromptPayload
from ..observability.profiler import agprof
from ..sandbox.agsandbox import agSandbox
from .harness_daemon_launcher import ensure_harness_daemon
from .host_servers.host_server_manager import HostServerManager

if TYPE_CHECKING:
    from ..agconfig import agConfig
    from ..agcontext import agcontext
    from ..agdata import agdata
    from ..agent import agent
    from ..orchestrator.agresources import agResourcePool
    from ..agskill import agskill
    from .._submission import Invocation
    from .clients import SandboxInteractionClient


class _NoopDecision:
    cancelled = False
    destroyed = False
    invocation_messages: tuple = ()


class _NoopInvocation:
    """Lifecycle compatibility for direct ``AgentEngine.execute`` callers."""

    @staticmethod
    def _checkpoint(_boundary_id: str, *, allow_messages: bool, phase: str) -> _NoopDecision:
        del allow_messages, phase
        return _NoopDecision()

    @staticmethod
    def _claim_completion() -> bool:
        return True

    @staticmethod
    def _note_model_result(*, has_tool_calls: bool) -> None:
        del has_tool_calls

    @staticmethod
    def is_cancelled() -> bool:
        return False

    @staticmethod
    def is_destroyed() -> bool:
        return False


_NOOP_INVOCATION = _NoopInvocation()


class AgentEngine:
    """Host-side execution owner for one agent."""

    def __init__(self, agent: "agent") -> None:
        self._agent = agent
        self._host_server_manager: "HostServerManager | None" = None
        self._sandbox_interaction_client: "SandboxInteractionClient | None" = None
        self._services_lock = threading.RLock()
        self._services_closed = True
        self._pending_session_update: "tuple[agcontext, str, str, str, int | None] | None" = None

    def set_config(self, agconfig: "agConfig") -> None:
        if self._host_server_manager is not None:
            self._host_server_manager.set_config(agconfig)

    def close(self) -> None:
        """Idempotently stop host services owned by this fresh engine."""
        with self._services_lock:
            if self._services_closed:
                return
            client = self._sandbox_interaction_client
            manager = self._host_server_manager
            client_error: "BaseException | None" = None
            manager_error: "BaseException | None" = None
            if client is not None:
                try:
                    client.close()
                except BaseException as exc:
                    # The LLM server must still stop so streamed deltas are
                    # finalized even when closing the sandbox RPC client fails.
                    client_error = exc
                else:
                    self._sandbox_interaction_client = None
            if manager is not None:
                try:
                    manager.stop()
                except BaseException as exc:
                    manager_error = exc
                else:
                    self._host_server_manager = None
            self._services_closed = (
                self._sandbox_interaction_client is None and self._host_server_manager is None
            )

        if client_error is not None:
            raise client_error
        if manager_error is not None:
            raise manager_error

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
        sandbox: agSandbox,
        max_steps: "int | None" = None,
        invocation: "Invocation | None" = None,
    ) -> "agdata":
        """Execute one request and own its complete sandbox transaction."""

        from ..agdata import agerror

        active_invocation = invocation if invocation is not None else _NOOP_INVOCATION
        sandbox_lock = sandbox._lock
        sandbox_lock.acquire()
        try:
            failed = True
            self._pending_session_update = None
            try:
                admission = active_invocation._checkpoint(
                    "engine:before-harness",
                    allow_messages=False,
                    phase="infrastructure",
                )
                if admission.destroyed or admission.cancelled:
                    return self._controlled_error(active_invocation, admission.destroyed)

                output = self._execute_harness(
                    context,
                    skill,
                    skill_input,
                    resource_pool,
                    sandbox,
                    max_steps=max_steps,
                    invocation=active_invocation,
                )
                completion = active_invocation._checkpoint(
                    "engine:before-commit",
                    allow_messages=False,
                    phase="boundary",
                )
                if completion.destroyed or completion.cancelled:
                    return self._controlled_error(active_invocation, completion.destroyed)
                if isinstance(output, agerror):
                    return output
                if not active_invocation._claim_completion():
                    return self._controlled_error(
                        active_invocation,
                        active_invocation.is_destroyed(),
                    )
                with agprof.span("teardown:commit"):
                    try:
                        sandbox.commit()
                    finally:
                        if not sandbox._has_pending_background_work():
                            try:
                                sandbox.stop()
                            except Exception as exc:
                                # DATACOLLECTOR: append -- ad-hoc print, uncaptured by any structured channel today.
                                print(
                                    f"[engine] WARNING: post-commit hibernate failed "
                                    f"for {self._agent.agname}: {exc}"
                                )
                self._commit_pending_session_update()
                failed = False
                return output
            finally:
                if failed:
                    self._pending_session_update = None
                    with agprof.span("teardown:discard"):
                        sandbox.rm_container()
        finally:
            try:
                sandbox_lock.release()
            finally:
                # _execute_harness() closes services before commit.  This
                # outer idempotent pass retries only a component whose first
                # close/stop raised, while successful components stay detached.
                self.close()

    @staticmethod
    def _controlled_error(invocation, destroyed: bool) -> "agdata":
        from ..agdata import agerror

        if destroyed or invocation.is_destroyed():
            return agerror("agent destroyed")
        return agerror("agent invocation cancelled")

    def _execute_harness(
        self,
        context: "agcontext",
        skill: "agskill",
        skill_input: "agdata",
        resource_pool: "agResourcePool",
        sandbox: agSandbox,
        max_steps: "int | None" = None,
        *,
        invocation=None,
    ) -> "agdata":
        """Run host services and the sandbox-side harness while locked."""

        self._agent.data_logger.record_event(
            type="agent_state",
            payload={"state": "running_harness"},
            update_latest_snapshot=True,
            flush=True,
        )

        # Start connections
        manager = HostServerManager(
            self._agent,
            sandbox,
            skill,
            resource_pool,
            invocation=invocation,
        )
        with self._services_lock:
            self._host_server_manager = manager
            self._services_closed = False
        try:
            host_uds_path = manager.start()
            # start harness manager daemon
            engine_name = str(
                getattr(self._agent, "agname", getattr(self._agent, "harness", "agent"))
            )
            handle = ensure_harness_daemon(
                sandbox,
                host_uds_path,
                engine_name,
                self._agent.harness,
                agconfig=self._agent.agconfig,
            )

            # Obtain Host -> Sandbox handle
            with self._services_lock:
                # Invocation pause is intentionally unbounded. Readiness probes
                # retain their short timeout, while the execution-owned outer
                # RPC must not retire its attempt token merely because a valid
                # safe-boundary pause lasts longer than five minutes.
                self._sandbox_interaction_client = handle.client(timeout_s=None)

            # Prefix retained host-only context that this harness session has
            # not incorporated yet.  Stateless harnesses have no advancing
            # cursor, so the same retained context remains available on later
            # calls by design.
            pending_retained = getattr(context, "pending_retained_messages", None)
            retained_messages = (
                pending_retained(self._agent.harness) if callable(pending_retained) else []
            )
            prompt = (
                self._build_prompt_payload(
                    skill,
                    skill_input,
                    retained_messages=retained_messages,
                )
                if retained_messages
                else self._build_prompt_payload(skill, skill_input)
            )
            # Captured before the loop may reassign `prompt` to a retry prompt --
            # the needle must stay the original user turn, not a retry prompt.
            initial_prompt = prompt
            retries_left = skill.max_output_schema_retries
            attempt: "HarnessAttemptResult | None" = None
            prior_session = context.harness_sessions.get(self._agent.harness)
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
                if self._recover_structured_output(skill, attempt, sandbox) is not None:
                    break
                missing = self._missing_output_fields(skill)
                if not missing or retries_left <= 0:
                    break
                retries_left -= 1
                prompt = self._build_retry_prompt(
                    missing, system_instruction=prompt.system_instruction
                )
            result = self._build_execution_result(
                context,
                skill,
                attempt,
                sandbox,
                initial_prompt,
            )
            from ..agdata import agerror

            if (
                not isinstance(result, agerror)
                and attempt is not None
                and attempt.ok
                and attempt.session_id
                and attempt.session_blob_b64 is not None
            ):
                retained_sequence = (
                    max(int(entry["sequence"]) for entry in retained_messages)
                    if retained_messages
                    else None
                )
                self._pending_session_update = (
                    context,
                    str(self._agent.harness),
                    attempt.session_id,
                    attempt.session_blob_b64,
                    retained_sequence,
                )
            return result
        finally:
            self.close()

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _build_prompt_payload(
        self,
        skill: "agskill",
        skill_input: "agdata",
        *,
        retained_messages: "list[dict] | None" = None,
    ) -> PromptPayload:
        from ..harness import agharness

        user_content = agharness.build_user_turn_prompt(skill, skill_input)
        if retained_messages:
            retained = self._render_retained_messages(retained_messages)
            if isinstance(user_content, str):
                user_content = f"{retained}\n\n{user_content}"
            else:
                user_content = [{"type": "text", "text": retained}, *user_content]
        return PromptPayload(
            system_instruction=skill._build_system_prompt(),
            user_content=user_content,
            output_instruction=agharness.build_output_format_instruction(skill),
        )

    @staticmethod
    def _render_retained_messages(messages: "list[dict]") -> str:
        parts = ["[AGENCY RETAINED CONTEXT]"]
        for message in messages:
            role = str(message.get("role", "user")).upper()
            parts.append(f"[{role}]\n{message.get('content', '')}")
        return "\n\n".join(parts)

    def _commit_pending_session_update(self) -> None:
        """Publish a restorable session only after its sandbox checkpoint."""
        update = self._pending_session_update
        self._pending_session_update = None
        if update is None:
            return
        context, harness, session_id, blob_b64, retained_sequence = update
        context.harness_sessions[harness] = {
            "session_id": session_id,
            "blob_b64": blob_b64,
        }
        if retained_sequence is not None:
            context.advance_retained_cursor(harness, retained_sequence)

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
        client = self._sandbox_interaction_client
        manager = self._host_server_manager
        if client is None:
            raise RuntimeError("Harness Manager client is not configured")
        if manager is None:
            raise RuntimeError("Host Server Manager is not configured")

        attempt_token = uuid.uuid4().hex
        manager.bind_attempt_token(attempt_token)
        try:
            request = HarnessAttemptRequest(
                prompt=prompt,
                harness=self._agent.harness,
                max_steps=max_steps,
                resume_session_id=resume_session_id,
                prior_session_blob_b64=prior_session_blob_b64,
                attempt_token=attempt_token,
            )
            return client.run_harness_attempt(request)
        finally:
            manager.clear_attempt_token(attempt_token)

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
        """Handle the plain JSON response contract used by non-MCP harnesses."""
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
        initial_prompt: "PromptPayload | None",
    ) -> "agdata":
        from ..agdata import agdata, agerror

        needle = initial_prompt.user_content if initial_prompt is not None else None
        context.recent_transcript = (
            self._host_server_manager.llm_handler_server.get_main_transcript(needle)
        )

        if attempt is None or not attempt.ok:
            message = attempt.error_message if attempt is not None else "no attempt was made"
            return agerror(message)

        output_schema = skill.output_schema
        if output_schema is not None and output_schema.raw_key() is None:
            collected = self._host_server_manager.host_mcp_server.collected_output()
            missing = sorted(set(output_schema._data) - set(collected))
            recovered = self._recover_structured_output(skill, attempt, sandbox)
            if not missing:
                return agdata(**collected)
            if recovered is not None:
                return recovered
            message = (
                "structured output incomplete after retries -- response was not valid JSON "
                "and submit_output was never called for: " + ", ".join(missing)
            )
            return agerror(message)

        output_key = output_schema.raw_key() if output_schema is not None else "result"
        return agdata(**{output_key: attempt.final_text})
