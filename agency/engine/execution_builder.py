from __future__ import annotations

from typing import TYPE_CHECKING

from .types import CompletedResult, HarnessAttemptResult

if TYPE_CHECKING:
    from ..agcontext import agcontext
    from ..agdata import agdata
    from ..agent import agent
    from ..agresources import agResourcePool
    from ..agskill import agskill
    from ..sandbox.agsandbox import agSandbox
    from .harness_bridge import HarnessManagerBridge
    from .host_servers.host_server_manager import HostServerManager
    from .sandbox_provisioner import SandboxLease, SandboxProvisioner


class ExecutionBuilder:
    """Sequence one complete host/sandbox execution transaction.

    Sandbox implementation and lease ownership stay in ``SandboxProvisioner``;
    this class only expresses the ordering between provisioning, host services,
    prompt construction, the harness, cleanup, and sandbox finalization.
    """

    def __init__(
        self,
        sandbox_provisioner: "SandboxProvisioner",
        harness_bridge: "HarnessManagerBridge",
    ) -> None:
        self._sandbox_provisioner = sandbox_provisioner
        self._harness_bridge = harness_bridge

    def execute(
        self,
        agent: "agent",
        context: "agcontext",
        skill: "agskill",
        skill_input: "agdata",
        resource_pool: "agResourcePool",
    ) -> CompletedResult:
        """Run and finalize one execution, preserving the first failure.

        Cleanup proceeds in reverse dependency order. A service-cleanup
        failure invalidates an otherwise successful execution before the
        provisioner is told whether it may checkpoint the sandbox.
        """
        lease: "SandboxLease | None" = None
        host_server_manager: "HostServerManager | None" = None
        harness_manager_pid: "int | None" = None
        harness_launch_attempted = False
        result: "CompletedResult | None" = None
        result_ok = False
        output_failed = False
        output_error: "BaseException | None" = None
        primary_error: "BaseException | None" = None

        try:
            lease = self._sandbox_provisioner.acquire(agent)
            host_server_manager = self._build_host_side_server(
                agent,
                skill,
                lease.sandbox,
                resource_pool,
            )
            host_uds_path = host_server_manager.start()
            prompt = skill.build_prompt_payload(skill_input)

            # From this point onward a failed launch may already have changed
            # sandbox state, even if it never returns a manager PID.
            self._sandbox_provisioner.mark_execution_attempted(lease)
            harness_launch_attempted = True

            # Launch Harness Manager
            harness_manager_pid = self._harness_bridge.ensure_launched(
                lease.sandbox,
                host_uds_path,
            )

            # Attempt skill invocation
            attempt = self._harness_bridge.run_attempt(
                agent.agconfig,
                skill,
                harness_manager_pid,
                prompt,
            )

            # not done yet
            result = self._wait_for_completion(harness_manager_pid, attempt)
            output_failed = self._validate_completed_result(result)
            result_ok = result.ok
            if output_failed:
                output_error = RuntimeError(
                    f"execution result reported an error: {result.output._data['error']}"
                )
            if not result_ok and not output_failed:
                raise RuntimeError(result.error_message or "agent engine execution failed")
        except BaseException as exc:
            primary_error = exc

        # A launch that raised can still have created a partial manager. The
        # bridge receives None in that case and owns whatever recovery its
        # implementation needs for a launch without a returned PID.
        if harness_launch_attempted:
            try:
                self._harness_bridge.stop(harness_manager_pid)
            except BaseException as exc:
                primary_error = self._record_cleanup_error(
                    primary_error or output_error,
                    exc,
                    "stopping the harness manager failed",
                )

        # A HostServerManager may fail after starting only some children, so
        # stop it whenever construction succeeded, not only when start()
        # returned its UDS. It retains failed component/thread handles for a
        # bounded retry, which gives every successfully started service a
        # second cleanup attempt before the manager leaves this local scope.
        if host_server_manager is not None:
            host_cleanup_error = self._stop_host_side_server(host_server_manager)
            if host_cleanup_error is not None:
                primary_error = self._record_cleanup_error(
                    primary_error or output_error,
                    host_cleanup_error,
                    "stopping host-side services failed",
                )

        if lease is not None:
            succeeded = primary_error is None and result_ok and not output_failed
            try:
                self._sandbox_provisioner.finalize(lease, succeeded=succeeded)
            except BaseException as exc:
                primary_error = self._record_cleanup_error(
                    primary_error or output_error,
                    exc,
                    "sandbox finalization failed",
                )

        if primary_error is not None:
            raise primary_error
        if result is None:  # Defensive: every non-exception path sets it.
            raise RuntimeError("execution finished without a CompletedResult")
        return result

    def _build_host_side_server(
        self,
        agent: "agent",
        skill: "agskill",
        sandbox: "agSandbox",
        resource_pool: "agResourcePool",
    ) -> "HostServerManager":
        from .host_servers.host_server_manager import HostServerManager

        return HostServerManager(agent, sandbox, skill, resource_pool)

    def _wait_for_completion(
        self,
        harness_manager_pid: "int | None",
        attempt: HarnessAttemptResult,
    ) -> CompletedResult:
        """Wait for the manager, then recover a host-only completed result.

        The concrete harness bridge is still being migrated to return Agency
        domain objects. Subclasses may override this recovery hook during
        that migration; the transaction and cleanup ordering stays here.
        """
        self._harness_bridge.wait(harness_manager_pid)
        raise NotImplementedError("completed-result recovery is not implemented")

    @staticmethod
    def _stop_host_side_server(
        host_server_manager: "HostServerManager",
    ) -> "BaseException | None":
        first_error: "BaseException | None" = None
        for _attempt in range(2):
            try:
                host_server_manager.stop()
                return None
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                else:
                    retry_notes = tuple(getattr(exc, "__notes__", ()))
                    first_error.add_note(f"retrying host-side service cleanup failed: {exc}")
                    if first_error is not exc:
                        for note in retry_notes:
                            first_error.add_note(f"host cleanup retry: {note}")
        return first_error

    @classmethod
    def _validate_completed_result(cls, result: object) -> bool:
        if not isinstance(result, CompletedResult):
            raise TypeError("ExecutionBuilder._wait_for_completion() must return CompletedResult")
        if not isinstance(result.ok, bool):
            raise TypeError("CompletedResult.ok must be bool")
        if not isinstance(result.delta, list):
            raise TypeError("CompletedResult.delta must be a list")
        return cls._output_represents_error(result)

    @staticmethod
    def _output_represents_error(result: CompletedResult) -> bool:
        output_data = getattr(result.output, "_data", None)
        if not isinstance(output_data, dict):
            raise TypeError("CompletedResult.output must be an agdata-like object")
        return bool(output_data.get("error"))

    @staticmethod
    def _record_cleanup_error(
        primary_error: "BaseException | None",
        cleanup_error: BaseException,
        label: str,
    ) -> BaseException:
        if primary_error is None:
            return cleanup_error
        cleanup_notes = tuple(getattr(cleanup_error, "__notes__", ()))
        primary_error.add_note(f"{label}: {cleanup_error}")
        if primary_error is not cleanup_error:
            for note in cleanup_notes:
                primary_error.add_note(f"{label}: {note}")
        return primary_error
