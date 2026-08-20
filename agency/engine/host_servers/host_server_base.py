from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

    from ...agconfig import agConfig


class HostServerBase:
    def set_config(self, agconfig: "agConfig") -> None:
        raise NotImplementedError

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def build_app(self) -> "FastAPI":
        raise NotImplementedError
