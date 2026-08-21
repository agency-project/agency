from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from ...agconfig import agConfig


class HostServerBase:
    def set_config(self, agconfig: "agConfig") -> None:
        self._agconfig = agconfig

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def build_app(self) -> "Starlette":
        raise NotImplementedError

    def lifespan_context(self, app: "Starlette") -> "AbstractAsyncContextManager[None] | None":
        return None
