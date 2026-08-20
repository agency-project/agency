from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..sandbox.agsandbox import agSandbox


class SandboxProvisioner:
    def provision(self, sandbox: "agSandbox") -> "agSandbox":
        raise NotImplementedError

    def teardown(self, sandbox: "agSandbox") -> None:
        raise NotImplementedError
