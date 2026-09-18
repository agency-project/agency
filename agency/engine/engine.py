from __future__ import annotations

import base64
import threading
import uuid
from typing import TYPE_CHECKING, Callable

import cloudpickle

from ..harness.protocol import HarnessAttemptRequest, HarnessAttemptResult, PromptPayload
from ..observability.profiler import agprof
from ..sandbox.agsandbox import agSandbox
from .harness_daemon_launcher import ensure_harness_daemon
from .host_servers.host_server_manager import HostServerManager

if TYPE_CHECKING:
    from ..configs.agconfig import agconfig as agconfig_cls
    from ..agcontext import agcontext
    from ..agdata import agdata
    from ..agent import agent
    from ..agpolicy import agpolicy
    from ..orchestrator.agresources import agResourcePool
    from ..agskill import agskill
    from ..agtool import agtool
    from .clients import HarnessInteractionClient
    from .harness_daemon_launcher import DaemonHandle

_HEALTH_CHECK_INTERVAL_S = 30.0
_HEALTH_CHECK_TIMEOUT_S = 10.0
_HEALTH_CHECK_FAILURE_LIMIT = 3


class AgentEngine:
    """Host-side execution owner for one agent."""

    def __init__(self, agent: "agent") -> None:
        self._agent = agent
        self.agconfig: "agconfig_cls" = agent.agconfig.clone()
        self._host_server_manager: "HostServerManager | None" = None
        self._sandbox_interaction_client: "HarnessInteractionClient | None" = None
        self._daemon_handle: "DaemonHandle | None" = None
        self._services_lock = threading.RLock()
        self._services_closed = True
        self._request_id: "str | None" = None
        self._pending_session_update: "tuple[agcontext, str, str, str, int | None] | None" = None
        self._agtype_cleanup_paths: set[str] = set()

    def change_config(self, agconfig: "agconfig_cls") -> None:
        self.agconfig = agconfig.clone() if agconfig is not None else agconfig_cls()
        if self._host_server_manager is not None:
            self._host_server_manager.change_config(self.agconfig)

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
            self._daemon_handle = None
            self._services_closed = (
                self._sandbox_interaction_client is None and self._host_server_manager is None
            )

        if client_error is not None:
            raise client_error
        if manager_error is not None:
            raise manager_error

    def redirect(self, message: str) -> bool:
        with self._services_lock:
            client = self._sandbox_interaction_client
            if self._services_closed or client is None:
                return False
            # Service teardown is serialized with this call. In particular,
            # a late redirect never contacts a hibernated persistent daemon.
            return client.redirect_harness(self._request_id, message)

    def cancel(self) -> None:
        with self._services_lock:
            client = self._sandbox_interaction_client
            if not self._services_closed and client is not None:
                client.cancel_harness(self._request_id)

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
        is_cancelled: "Callable[[], bool]" = lambda: False,
        request_id: "str | None" = None,
        claim_completion: "Callable[[], bool]" = lambda: True,
    ) -> "agdata":
        """Execute one request and own its complete sandbox transaction."""

        from ..agdata import agerror

        sandbox_lock = sandbox._lock
        sandbox_lock.acquire()
        try:
            failed = True
            self._pending_session_update = None
            try:
                if is_cancelled():
                    return self._controlled_error()

                output = self._execute_harness(
                    context,
                    skill,
                    skill_input,
                    resource_pool,
                    sandbox,
                    max_steps=max_steps,
                    is_cancelled=is_cancelled,
                    request_id=request_id,
                )
                if is_cancelled():
                    return self._controlled_error()
                if isinstance(output, agerror):
                    return output
                if not claim_completion():
                    return self._controlled_error()
                cow_zfs = (
                    getattr(sandbox, "agconfig", self.agconfig).sandbox.checkpoint_backend
                    == "cow_zfs"
                )
                with agprof.span("teardown:commit"):
                    try:
                        sandbox.commit()
                    finally:
                        # A COW checkpoint intentionally ends process state at
                        # the completed-run boundary, including background jobs.
                        pending = False if cow_zfs else sandbox._has_pending_background_work()
                        diagnostic = None
                        if getattr(self.agconfig.sandbox, "hibernation_diagnostics", False):
                            from ..sandbox.pid_diagnostics import decision_snapshot

                            diagnostic = decision_snapshot(sandbox._backend, pending)
                            diagnostic["request_id"] = request_id
                            self._agent.data_logger.record_event(
                                type="hibernation_decision", payload=diagnostic, flush=True
                            )
                        hibernation = "skipped_pending_work" if pending else "performed"
                        if not pending:
                            try:
                                sandbox.stop()
                            except Exception as exc:
                                hibernation = "failed"
                                # DATACOLLECTOR: append -- ad-hoc print, uncaptured by any structured channel today.
                                print(
                                    f"[engine] WARNING: post-commit hibernate failed "
                                    f"for {self._agent.agname}: {exc}"
                                )
                        if diagnostic is not None:
                            self._agent.data_logger.record_event(
                                type="hibernation_outcome",
                                payload={
                                    "request_id": diagnostic["request_id"],
                                    "outcome": hibernation,
                                    "decision_epoch": diagnostic["decision_epoch"],
                                },
                                flush=True,
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
    def _controlled_error() -> "agdata":
        from ..agdata import agcanceled

        return agcanceled()

    def _execute_harness(
        self,
        context: "agcontext",
        skill: "agskill",
        skill_input: "agdata",
        resource_pool: "agResourcePool",
        sandbox: agSandbox,
        max_steps: "int | None" = None,
        *,
        is_cancelled: "Callable[[], bool]" = lambda: False,
        request_id: "str | None" = None,
    ) -> "agdata":
        """Run host services and the sandbox-side harness while locked."""

        self._request_id = request_id
        self._agtype_cleanup_paths = set()
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
            is_cancelled=is_cancelled,
            request_id=request_id,
            recent_transcript=context.recent_transcript,
        )
        with self._services_lock:
            self._host_server_manager = manager
            self._services_closed = False
        try:
            input_schema = getattr(skill, "input_schema", None)
            if input_schema is not None:
                prepared_paths, _offloaded = input_schema.prepare_inputs_in_sandbox(
                    skill_input,
                    sandbox,
                    skill.name,
                    context_limit=self.agconfig.llm.context_limit,
                    agconfig=self.agconfig,
                )
                self._agtype_cleanup_paths.update(prepared_paths)
            with agprof.span("agency:services_start"):
                host_uds_path = manager.start()
            # start harness manager daemon
            engine_name = str(
                getattr(self._agent, "agname", getattr(self._agent, "harness", "agent"))
            )
            with agprof.span("sandbox:ensure_daemon"), agprof.span("runtime.harness_start"):
                handle = ensure_harness_daemon(
                    sandbox,
                    host_uds_path,
                    engine_name,
                    self._agent.harness,
                    agconfig=self.agconfig,
                    progress_source=manager.interaction_server,
                )
            self._daemon_handle = handle

            # Sync current pause state to the daemon before this attempt
            # starts -- closes the race where pause() was requested before
            # the sandbox/daemon existed at all, or between attempts (the
            # daemon's own sticky state, see HarnessManager._agent_paused,
            # is what actually holds it; agent.pause()/resume() reach the
            # SAME daemon directly whenever one already exists). No
            # blocking wait, no Event -- the daemon is what stays paused.
            with self._agent._control_lock:
                try:
                    with handle.client(timeout_s=10) as client:
                        if self._agent.is_paused():
                            client.pause_harness()
                        else:
                            client.resume_harness()
                except Exception as exc:
                    print(
                        f"[engine] WARNING: pause state sync failed for {self._agent.agname}: {exc}"
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
                    sandbox_mcp_tools=skill.sandbox_mcp_tools,
                    policy=skill.policy,
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
            try:
                with agprof.span("agency:services_close"):
                    self.close()
            finally:
                if self._agtype_cleanup_paths:
                    sandbox.remove_files(sorted(self._agtype_cleanup_paths))

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
        user_content = skill.build_user_content(skill_input)
        if retained_messages:
            retained = self._render_retained_messages(retained_messages)
            if isinstance(user_content, str):
                user_content = f"{retained}\n\n{user_content}"
            else:
                user_content = [{"type": "text", "text": retained}, *user_content]
        return PromptPayload(
            system_instruction=skill._build_prompt(),
            user_content=user_content,
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
                f"fields. Still missing: {missing}. Do not answer with text. You must "
                "call the Agency MCP server's submit_output tool once for each missing field."
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
        sandbox_mcp_tools: "list[agtool] | None" = None,
        policy: "agpolicy | None" = None,
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
            sandbox_mcp_tools_b64 = None
            if sandbox_mcp_tools:
                try:
                    sandbox_mcp_tools_b64 = base64.b64encode(
                        cloudpickle.dumps(sandbox_mcp_tools)
                    ).decode("ascii")
                except Exception as exc:
                    # Exception text can contain callable state or payload contents.
                    return HarnessAttemptResult(
                        ok=False,
                        error_message=(
                            f"sandbox MCP setup failed during serialization ({type(exc).__name__})"
                        ),
                    )
            hooked_syscall_names = (
                sorted(policy.syscall_hooks)
                if policy is not None and policy.syscall_hooks
                else None
            )
            request = HarnessAttemptRequest(
                prompt=prompt,
                harness=self._agent.harness,
                request_id=self._request_id,
                max_steps=max_steps,
                resume_session_id=resume_session_id,
                prior_session_blob_b64=prior_session_blob_b64,
                attempt_token=attempt_token,
                sandbox_mcp_tools_b64=sandbox_mcp_tools_b64,
                syscall_default_to_deny=bool(policy.default_to_deny)
                if policy is not None
                else False,
                syscall_hooked_names=hooked_syscall_names,
            )
            return self._run_attempt_with_watchdog(client, request)
        finally:
            manager.clear_attempt_token(attempt_token)

    def _run_attempt_with_watchdog(
        self, client: "HarnessInteractionClient", request: HarnessAttemptRequest
    ) -> HarnessAttemptResult:
        """Run the attempt RPC on a background thread, unbounded exactly as
        before (a real, safe-boundary pause can legitimately run far longer
        than any fixed timeout on the RPC itself) -- but while it's in
        flight, periodically ping the daemon's own /health endpoint on a
        separate, short-timeout connection. A daemon that's merely busy or
        paused still answers /health immediately; one that's died (crashed,
        killed, or otherwise gone) won't. After enough consecutive health
        check failures, give up waiting and report the attempt as failed
        instead of blocking forever on a connection nothing will ever answer.

        The abandoned background thread (still blocked on the dead
        connection) is left to leak until the process exits -- there's no
        way to safely force-cancel an in-flight blocking socket read from
        another thread, and this path should be rare.
        """
        handle = self._daemon_handle
        outcome: "list[HarnessAttemptResult | BaseException]" = []
        done = threading.Event()

        def worker() -> None:
            try:
                outcome.append(client.run_harness_attempt(request))
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread below
                outcome.append(exc)
            finally:
                done.set()

        threading.Thread(target=worker, name="harness-attempt-rpc", daemon=True).start()

        consecutive_failures = 0
        while not done.wait(_HEALTH_CHECK_INTERVAL_S):
            if handle is None:
                continue
            try:
                with handle.client(timeout_s=_HEALTH_CHECK_TIMEOUT_S) as health_client:
                    health_client.is_ready()
                consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
                if consecutive_failures >= _HEALTH_CHECK_FAILURE_LIMIT:
                    return HarnessAttemptResult(
                        ok=False,
                        error_message=(
                            "harness daemon stopped responding to health checks after "
                            f"{consecutive_failures * _HEALTH_CHECK_INTERVAL_S:.0f}s -- "
                            "attempt abandoned"
                        ),
                    )

        result = outcome[0]
        if isinstance(result, BaseException):
            raise result
        return result

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
        """Recover structured output from a model's final text, in case it
        stated its answer as JSON without being asked to (the model is never
        instructed to do this -- see agschema._lenient_json_object) --
        structured output is normally submitted via the return_<field>/
        submit_output MCP tools instead."""
        from ..agdata import agerror

        output_schema = skill.output_schema
        if (
            output_schema is None
            or getattr(output_schema, "raw_key", lambda: None)() is not None
            or not hasattr(output_schema, "validate_and_recover")
        ):
            return None
        recovered, paths = output_schema.validate_and_recover(
            attempt.final_text,
            sandbox,
            exec_timeout=self.agconfig.skill.agbinary_validate_exec_timeout,
        )
        self._agtype_cleanup_paths.update(paths)
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
                result = agdata(**collected)
                errors = output_schema.validate_outputs(
                    result,
                    sandbox,
                    exec_timeout=self.agconfig.skill.agbinary_validate_exec_timeout,
                )
                if errors:
                    return agerror(f"output schema error: {errors}")
                self._agtype_cleanup_paths.update(output_schema.recover_outputs(result, sandbox))
                return result
            if recovered is not None:
                return recovered
            message = (
                "structured output incomplete after retries -- response was not valid JSON "
                "and submit_output was never called for: " + ", ".join(missing)
            )
            return agerror(message)

        output_key = output_schema.raw_key() if output_schema is not None else "result"
        return agdata(**{output_key: attempt.final_text})
