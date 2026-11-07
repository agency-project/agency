"""In-process ptrace policy bridge to Agency's shared interaction service."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .host_interaction_server import HostInteractionServer


class InteractionSyscallPolicy:
    """Use the same admission/completion path without a daemon RPC hop."""

    def __init__(self, server: HostInteractionServer) -> None:
        self.server = server

    def check(self, _agent, syscall):
        admission = self.server.admit_syscall(syscall)
        return (
            admission["allowed"],
            admission["reason"],
            admission["call_id"],
            admission["env_overrides"],
        )

    def check_completion(self, _agent, call_id, return_value):
        if call_id is not None:
            self.server.complete_syscall(call_id, return_value)

    def record_file_access(self, payload):
        self.server.record_event("checkpoint_file_access", payload, print_to_terminal=False)
