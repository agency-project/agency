from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ..profiler import agprof

if TYPE_CHECKING:
    from ..agent import agent
    from ..sandbox.agsandbox import agSandbox


@dataclass
class SandboxLease:
    """One provisioner-owned lease over an agent's durable sandbox.

    The flags record which lifecycle boundaries were actually crossed so an
    exception can unwind only work this lease owns. ``finalized`` means that
    finalization has been attempted; it is set before cleanup starts so a
    retry cannot commit, discard, or release the same lease twice.
    """

    agent: "agent"
    sandbox: "agSandbox"
    lock_acquired: bool = False
    physical_start_attempted: bool = False
    provisioned: bool = False
    execution_attempted: bool = False
    committed: bool = False
    discard_attempted: bool = False
    discarded: bool = False
    hibernated: bool = False
    finalized: bool = False


class SandboxProvisioner:
    """Own the complete sandbox lease for one engine execution transaction."""

    _REVERT_NOTICE = (
        "Note: the previous skill call failed. Its sandbox workspace "
        "changes have been discarded and the workspace has been "
        "reverted to the last successful checkpoint."
    )

    # ------------------------------------------------------------------
    # Construction and facade resolution
    # ------------------------------------------------------------------

    def __init__(
        self,
        sandbox_factory: "Callable[..., agSandbox] | None" = None,
    ) -> None:
        self._sandbox_factory = sandbox_factory

    def get_or_create(self, agent: "agent") -> "agSandbox":
        """Return ``agent.sandbox``, creating and attaching it when absent."""
        if agent.sandbox is not None:
            return agent.sandbox

        from ..agconfig import agConfig
        from ..sandbox.agsandbox import agSandbox, agSandboxConfig

        output_dir = (
            agent.agconfig.get("agent", "output_dir", type(agent).output_dir)
            if agent.agconfig is not None
            else type(agent).output_dir
        )
        output_path = Path(output_dir) / agent.agname if output_dir else None

        sandbox_config = agent.agconfig
        if output_path is not None:
            sandbox_config = sandbox_config.clone() if sandbox_config else agConfig()
            agSandboxConfig(sandbox_config).add_mount("agent_output", output_path, "/agent_output")

        sandbox_factory = self._sandbox_factory or agSandbox
        sandbox = sandbox_factory(agent.agname, agconfig=sandbox_config)
        agent.sandbox = sandbox
        return sandbox

    # ------------------------------------------------------------------
    # Lease acquisition and physical startup
    # ------------------------------------------------------------------

    def acquire(self, agent: "agent") -> SandboxLease:
        """Resolve, exclusively lease, and physically prepare ``agent``'s sandbox.

        Facade resolution must precede locking because the lock belongs to the
        facade. Once acquired, that same lock covers physical startup and is
        held until :meth:`finalize` completes. Startup failures are unwound
        here because no lease is returned to the caller in that case.
        """
        lease: "SandboxLease | None" = None
        try:
            with agprof.span("sandbox:provision"):
                # Create the facade and its transaction record.
                sandbox = self.get_or_create(agent)
                lease = SandboxLease(agent=agent, sandbox=sandbox)

                # Exclusively hold the sandbox for the full transaction.
                acquired = sandbox._lock.acquire()
                if acquired is False:
                    raise RuntimeError("sandbox lock acquisition returned False")
                lease.lock_acquired = True

                # A prior transaction whose physical discard failed must
                # never be resumed with its dirty writable layer intact.
                # Retry that discard while holding the same facade lock,
                # before the backend is allowed to start again.
                self._finish_pending_sandbox_cleanup(lease)

                lease.physical_start_attempted = True
                sandbox.ensure_started()
                lease.provisioned = True
        except BaseException as acquisition_error:
            # This also catches a profiler span's __exit__ failure after the
            # physical sandbox was prepared. No instrumentation failure may
            # strand a provisioner-owned lock without returning its lease.
            if lease is not None and lease.lock_acquired:
                self._unwind_failed_acquisition(lease, acquisition_error)
            raise
        assert lease is not None
        return lease

    def _finish_pending_sandbox_cleanup(self, lease: SandboxLease) -> None:
        """Finish cleanup left unsafe by an earlier lease before startup."""
        sandbox_state = getattr(lease.sandbox, "__dict__", {})
        if sandbox_state.get("_provisioner_discard_required", False):
            lease.physical_start_attempted = True
            self._discard(lease, notify_agent=False)
            # The recovered discard belongs to the earlier transaction, not
            # to the new lease now being prepared.
            lease.discard_attempted = False
            lease.discarded = False
        else:
            # Removal may have succeeded while delivery of its inbox notice
            # failed. Retrying the notice does not require another discard.
            self._flush_revert_notice(lease.sandbox)

    def _unwind_failed_acquisition(
        self,
        lease: SandboxLease,
        acquisition_error: BaseException,
    ) -> None:
        """Clean up a failed acquisition while preserving its first error."""
        lease.finalized = True

        if lease.physical_start_attempted:
            try:
                self._discard(lease, notify_agent=False)
            except BaseException as discard_error:
                self._add_cleanup_note(
                    acquisition_error,
                    "discarding partially provisioned sandbox state",
                    discard_error,
                )

        try:
            self._teardown(lease)
        except BaseException as teardown_error:
            self._add_cleanup_note(
                acquisition_error,
                "tearing down the partially provisioned sandbox",
                teardown_error,
            )

        try:
            self._release_lock(lease)
        except BaseException as release_error:
            self._add_cleanup_note(
                acquisition_error,
                "releasing the sandbox lock after provisioning failure",
                release_error,
            )

    # ------------------------------------------------------------------
    # Execution-attempt boundary
    # ------------------------------------------------------------------

    def mark_execution_attempted(self, lease: SandboxLease) -> None:
        """Record the point immediately before the harness is launched."""
        if lease.finalized:
            raise RuntimeError("cannot mark a finalized sandbox lease as attempted")
        if not lease.lock_acquired or not lease.provisioned:
            raise RuntimeError("sandbox execution cannot start before provisioning completes")
        lease.execution_attempted = True

    # ------------------------------------------------------------------
    # Commit, discard, and lease finalization
    # ------------------------------------------------------------------

    def finalize(self, lease: SandboxLease, *, succeeded: bool) -> None:
        """Commit or discard the lease, tear it down, then release its lock.

        Finalization is idempotent. Every cleanup stage gets a chance to run;
        the first failure remains primary and later failures are attached as
        exception notes. The sandbox lock is always the final attempted step.
        """
        if lease.finalized:
            return
        lease.finalized = True

        if not lease.lock_acquired:
            raise RuntimeError("cannot finalize a sandbox lease whose lock is not held")

        first_error: "BaseException | None" = None

        if succeeded:
            if not lease.execution_attempted:
                first_error = RuntimeError(
                    "cannot commit a sandbox when harness execution was not attempted"
                )
            else:
                try:
                    self._commit(lease)
                except BaseException as commit_error:
                    first_error = commit_error
                    try:
                        self._discard(lease, notify_agent=True)
                    except BaseException as discard_error:
                        self._add_cleanup_note(
                            first_error,
                            "discarding dirty sandbox state after commit failure",
                            discard_error,
                        )
        elif lease.execution_attempted:
            try:
                self._discard(lease, notify_agent=True)
            except BaseException as discard_error:
                first_error = discard_error

        try:
            self._teardown(lease)
        except BaseException as teardown_error:
            if first_error is None:
                first_error = teardown_error
            else:
                self._add_cleanup_note(
                    first_error,
                    "tearing down the provisioned sandbox",
                    teardown_error,
                )

        try:
            self._release_lock(lease)
        except BaseException as release_error:
            if first_error is None:
                first_error = release_error
            else:
                self._add_cleanup_note(
                    first_error,
                    "releasing the sandbox lock",
                    release_error,
                )

        if first_error is not None:
            raise first_error

    def _commit(self, lease: SandboxLease) -> None:
        with agprof.span("teardown:commit"):
            committed = lease.sandbox.commit()
        if not committed:
            raise RuntimeError("sandbox commit reported that no physical sandbox existed")
        lease.committed = True

    def _discard(self, lease: SandboxLease, *, notify_agent: bool) -> None:
        # Record the unsafe state before calling the backend. If removal
        # raises after doing only part of its work, a later lease will retry
        # the idempotent discard before it may start the sandbox again.
        lease.sandbox._provisioner_discard_required = True
        if notify_agent:
            lease.sandbox._provisioner_revert_notice_agent = lease.agent
        lease.discard_attempted = True
        with agprof.span("teardown:discard"):
            lease.sandbox.rm_container()
        lease.sandbox._provisioner_discard_required = False
        lease.discarded = True
        self._flush_revert_notice(lease.sandbox)

    def _flush_revert_notice(self, sandbox: "agSandbox") -> None:
        sandbox_state = getattr(sandbox, "__dict__", {})
        notice_agent = sandbox_state.get("_provisioner_revert_notice_agent")
        if notice_agent is None:
            return
        notice_agent.inbox.put(self._REVERT_NOTICE)
        sandbox._provisioner_revert_notice_agent = None

    def _teardown(self, lease: SandboxLease) -> None:
        """Hibernate live state after commit or before execution begins.

        A successful discard already removed the live state. If discard fails,
        stopping is still a useful physical safety fallback: it terminates
        sandbox processes and releases runtime resources, but does not mark
        the dirty state as discarded or suppress the discard error. Successful
        commits and preparation-only exits can also safely release physical
        resources while retaining the durable facade.
        """
        try:
            if lease.discarded or not lease.physical_start_attempted:
                return
            if lease.discard_attempted:
                # A failed physical discard is never a safe "background work
                # is pending" case. Stop it best-effort even during failed
                # acquisition, before the lock is released, and leave the
                # recovery marker for the next lease to retry removal.
                lease.sandbox.stop()
                lease.hibernated = True
                return
            if lease.execution_attempted and not lease.committed:
                return
            if lease.sandbox._has_pending_background_work():
                return
            lease.sandbox.stop()
            lease.hibernated = True
        except BaseException:
            # A backend that could not finish hibernating may retain live
            # processes or an ambiguous writable layer. Force the next lease
            # to remove that physical state before it may start again.
            if not lease.discarded and lease.physical_start_attempted:
                lease.sandbox._provisioner_discard_required = True
            raise

    # ------------------------------------------------------------------
    # Lock release and error aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _release_lock(lease: SandboxLease) -> None:
        if not lease.lock_acquired:
            return
        try:
            lease.sandbox._lock.release()
        finally:
            # A failed release has ambiguous lock state. Retrying could
            # double-release, so this lease must never attempt it again.
            lease.lock_acquired = False

    @staticmethod
    def _add_cleanup_note(
        primary_error: BaseException,
        action: str,
        cleanup_error: BaseException,
    ) -> None:
        cleanup_notes = tuple(getattr(cleanup_error, "__notes__", ()))
        primary_error.add_note(f"{action} also failed: {cleanup_error}")
        if primary_error is not cleanup_error:
            for note in cleanup_notes:
                primary_error.add_note(f"{action}: {note}")
