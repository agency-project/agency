from __future__ import annotations

from typing import TYPE_CHECKING

from ..profiler import agprof
from .host_servers.host_server_manager import HostServerManager
from .types import CompletedResult

if TYPE_CHECKING:
    from ..agcontext import agcontext
    from ..agdata import agdata
    from ..agent import agent
    from ..agresources import agResourcePool
    from ..agskill import agskill
    from ..sandbox.agsandbox import agSandbox
    from .execution_builder import ExecutionBuilder
    from .sandbox_provisioner import SandboxProvisioner


class agentEngine:
    def __init__(
        self,
        agent: "agent",
        context: "agcontext",
        skill: "agskill",
        skill_input: "agdata",
        resource_pool: "agResourcePool",
        *,
        execution_builder: "ExecutionBuilder | None" = None,
        sandbox_provisioner: "SandboxProvisioner | None" = None,
    ) -> None:
        self._agent = agent
        self._context = context
        self._skill = skill
        self._skill_input = skill_input
        self._resource_pool = resource_pool

        if sandbox_provisioner is None:
            from .sandbox_provisioner import SandboxProvisioner

            sandbox_provisioner = SandboxProvisioner()
        self._sandbox_provisioner = sandbox_provisioner

        if execution_builder is None:
            from .execution_builder import ExecutionBuilder
            from .harness_bridge import HarnessManagerBridge

            execution_builder = ExecutionBuilder(
                harness_bridge=HarnessManagerBridge(),
            )
        self._execution_builder = execution_builder
        self._sandbox: "agSandbox | None" = None
        self._sandbox_lock_acquired = False
        self._run_succeeded: "bool | None" = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return

        try:
            with agprof.span("sandbox:provision"):
                sandbox = self._sandbox_provisioner.get_or_create(self._agent)
            sandbox._lock.acquire()
            self._sandbox_lock_acquired = True
            self._sandbox = sandbox

            provisioned_sandbox = self._sandbox_provisioner.provision(sandbox)
            if provisioned_sandbox is not sandbox:
                raise RuntimeError(
                    "SandboxProvisioner.provision() must preserve the sandbox facade"
                )

            self._execution_builder.prepare_execution(
                self._agent,
                self._skill,
                self._resource_pool,
                sandbox=sandbox,
            )
            self._started = True
        except BaseException as execution_error:
            # A preparation failure is a failed sandbox transaction just like
            # a harness exception.  Clean it up here as well as in execute()
            # so callers using start()/run()/stop() directly never leak the
            # sandbox lease.
            self._run_succeeded = False
            try:
                self.stop()
            except BaseException as cleanup_error:
                execution_error.add_note(
                    f"agentEngine cleanup after start failure also failed: {cleanup_error}"
                )
            raise

    def stop(self) -> None:
        sandbox = self._sandbox
        # If this engine does not currently own a sandbox, there is no external
        # cleanup to perform; reset its bookkeeping and finish.
        if sandbox is None:
            self._sandbox_lock_acquired = False
            self._run_succeeded = None
            self._started = False
            return

        # Keep the first cleanup error, but continue cleaning up so the sandbox
        # lease has the best chance of being released.
        first_error: "BaseException | None" = None

        # Execution-scoped services must be gone before the filesystem is
        # checkpointed or discarded.  Keep going on cleanup failures so the
        # sandbox lease is always released.
        try:
            self._execution_builder.close()
        except BaseException as exc:
            first_error = exc

        try:
            self._sandbox_provisioner.teardown(sandbox)
        except BaseException as exc:
            # If closing execution services did not already fail, report this
            # teardown failure as the main cleanup error.
            if first_error is None:
                first_error = exc
            else:
                # Otherwise, preserve the teardown failure as extra context.
                first_error.add_note(f"sandbox provisioner teardown also failed: {exc}")

        # If the run succeeded and earlier cleanup succeeded, save this
        # sandbox's changes as the next checkpoint.
        if self._run_succeeded is True and first_error is None:
            try:
                self._commit_sandbox(sandbox)
            except BaseException as exc:
                first_error = exc
                # A failed checkpoint must not leave the live dirty container
                # available to the next skill.  Drop it so the next access
                # starts from whichever lifecycle image was last committed.
                try:
                    self._discard_sandbox(sandbox)
                except BaseException as discard_error:
                    first_error.add_note(
                        f"sandbox discard after commit failure also failed: {discard_error}"
                    )
        # If a run was attempted but it did not complete successfully, discard
        # its sandbox changes instead of making them available to the next run.
        elif self._run_succeeded is not None:
            try:
                self._discard_sandbox(sandbox)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                else:
                    first_error.add_note(f"sandbox finalization also failed: {exc}")
        try:
            # Only release the lock when this engine successfully acquired it.
            if self._sandbox_lock_acquired:
                sandbox._lock.release()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
            else:
                first_error.add_note(f"sandbox lock release also failed: {exc}")
        finally:
            # Regardless of cleanup errors, do not leave this engine claiming
            # that it owns or has started a sandbox.
            self._sandbox_lock_acquired = False
            self._sandbox = None
            self._run_succeeded = None
            self._started = False

        # Once every cleanup step has had a chance to run, surface the first
        # cleanup failure to the caller.
        if first_error is not None:
            raise first_error

    def run(self) -> CompletedResult:
        if not self._started:
            raise RuntimeError("agentEngine.start() must be called before run()")
        # Pessimistic until the builder returns a well-formed successful
        # result. This also makes malformed builder results roll back.
        self._run_succeeded = False
        try:
            result = self._execution_builder.build_and_run(
                agent=self._agent,
                context=self._context,
                skill=self._skill,
                skill_input=self._skill_input,
                resource_pool=self._resource_pool,
            )
        except BaseException:
            raise

        output_failed = bool(result.output._data.get("error"))
        self._run_succeeded = result.ok and not output_failed
        if not result.ok and not output_failed:
            raise RuntimeError(result.error_message or "agent engine execution failed")
        return result

    def execute(self) -> CompletedResult:
        """Run one complete engine-owned sandbox transaction."""
        try:
            self.start()
            result = self.run()
        except BaseException as execution_error:
            try:
                self.stop()
            except BaseException as cleanup_error:
                execution_error.add_note(
                    f"agentEngine cleanup after execution failure also failed: {cleanup_error}"
                )
            raise

        self.stop()
        return result

    def _commit_sandbox(self, sandbox: "agSandbox") -> None:
        with agprof.span("teardown:commit"):
            sandbox.commit()
            try:
                # Output recovery may have restarted a hibernating sandbox,
                # while commit() deliberately leaves it running.
                if not sandbox._has_pending_background_work():
                    try:
                        sandbox.stop()
                    except Exception as exc:
                        print(
                            "[agentEngine] WARNING: post-commit hibernate "
                            f"failed for {self._agent.agname}: {exc}"
                        )
            except Exception as exc:
                print(
                    "[agentEngine] WARNING: post-commit pending-work check "
                    f"failed for {self._agent.agname}: {exc}"
                )

    def _discard_sandbox(self, sandbox: "agSandbox") -> None:
        with agprof.span("teardown:discard"):
            sandbox.rm_container()
        self._agent.inbox.put(
            "Note: the previous skill call failed. Its sandbox workspace "
            "changes have been discarded and the workspace has been "
            "reverted to the last successful checkpoint."
        )

    @property
    def host_server_manager(self) -> HostServerManager:
        manager = self._execution_builder.host_server_manager
        if manager is None:
            raise RuntimeError("agentEngine.start() has not prepared the host server")
        return manager

    @property
    def sandbox(self) -> "agSandbox | None":
        return self._sandbox
