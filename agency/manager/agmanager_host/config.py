"""Config surface for `agmanager_host` -- its own module since both
`agmanager_host.py` (`bind_host`, for the TCP/UDS listeners) and
`llm_dispatch.py` (`request_timeout_s`, for the LLM client's own timeout)
need it, and neither should import the other just for this."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...agconfig import GlobalConfigParam, _AgConfigViewBase

if TYPE_CHECKING:
    from ...agconfig import agConfig


class AgHostAgentManagerFields:
    bind_host = GlobalConfigParam("agmanager_host", default="127.0.0.1")
    request_timeout_s = GlobalConfigParam("agmanager_host", default=300)

    def __init__(self, agconfig: "agConfig | None" = None) -> None:
        self._agconfig = agconfig


class agHostAgentManagerConfig(_AgConfigViewBase):
    _OWNER = "agmanager_host"


__all__ = ["AgHostAgentManagerFields", "agHostAgentManagerConfig"]
