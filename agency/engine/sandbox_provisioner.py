from __future__ import annotations

from typing import TYPE_CHECKING

from .types import ProvisionedSandbox

if TYPE_CHECKING:
    from ..sandbox.agsandbox import agSandbox


class SandboxProvisioner:
    def provision(self, sandbox: "agSandbox") -> ProvisionedSandbox:
        raise NotImplementedError

    def teardown(self, provisioned: ProvisionedSandbox) -> None:
        raise NotImplementedError
