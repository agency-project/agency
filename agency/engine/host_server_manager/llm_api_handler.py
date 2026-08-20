from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...agconfig import agConfig


class LlmApiHandler:
    def __init__(self, agconfig: "agConfig") -> None:
        raise NotImplementedError

    def dispatch(self, request: dict) -> dict:
        raise NotImplementedError

    def resolve_model(self) -> str:
        raise NotImplementedError

    def context_limit(self) -> int:
        raise NotImplementedError
