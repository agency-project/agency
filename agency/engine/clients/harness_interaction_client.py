"""Host-side client for the sandbox Harness Manager interaction server."""

from __future__ import annotations

from dataclasses import asdict

import httpx

from ...harness.protocol import HarnessAttemptRequest, HarnessAttemptResult
from ...observability.profiler import agprof


_LIFECYCLE_TIMEOUT_S = 30
_PAUSE_RPC_MARGIN_S = 5.0


class HarnessInteractionClient:
    def __init__(self, uds_path: str, timeout_s: "float | None" = 300.0) -> None:
        self._client = httpx.Client(
            transport=httpx.HTTPTransport(uds=uds_path),
            base_url="http://agency-sandbox",
            timeout=timeout_s,
        )

    def run_harness_attempt(self, request: HarnessAttemptRequest) -> HarnessAttemptResult:
        # Envelope: includes daemon/CLI/model/tool work. Never attribute its
        # whole duration to socket overhead or idle time.
        with agprof.span("harness:attempt_rpc"):
            response = self._client.post("/harness_attempt", json=asdict(request))
            response.raise_for_status()
        with agprof.span("agency:decode_harness_result"):
            return HarnessAttemptResult(**response.json())

    def pause_harness(self, wait_timeout_s: "float | None" = None) -> dict:
        """Pause the harness. With ``wait_timeout_s`` this returns only once
        it is actually stopped: ``{"ok": True, "parked": bool}`` (``parked``
        False if it did not stop in time). Without it the call just requests
        the pause and returns ``{"ok": True}``."""
        if wait_timeout_s is None:
            response = self._client.post("/control/pause")
        else:
            # The RPC must outlive the wait it asks the daemon to perform.
            response = self._client.post(
                "/control/pause",
                json={"wait_timeout_s": wait_timeout_s},
                timeout=wait_timeout_s + _PAUSE_RPC_MARGIN_S,
            )
        response.raise_for_status()
        return response.json()

    def redirect_harness(self, request_id: str, message: str) -> bool:
        response = self._client.post(
            "/redirect", json={"request_id": request_id, "message": message}
        )
        response.raise_for_status()
        return response.json()["delivered"] is True

    def resume_harness(self) -> None:
        self._client.post("/control/resume").raise_for_status()

    def cancel_harness(self, request_id: str) -> None:
        self._client.post(
            "/control/cancel", json={"request_id": request_id}, timeout=10
        ).raise_for_status()

    def prepare_fast_checkpoint(self) -> dict:
        response = self._client.post(
            "/lifecycle/prepare_fast_checkpoint", timeout=_LIFECYCLE_TIMEOUT_S
        )
        response.raise_for_status()
        return response.json()

    def complete_fast_restore(self) -> dict:
        response = self._client.post(
            "/lifecycle/complete_fast_restore", timeout=_LIFECYCLE_TIMEOUT_S
        )
        response.raise_for_status()
        return response.json()

    def seize_fast_restore(self) -> dict:
        response = self._client.post("/lifecycle/seize_fast_restore", timeout=_LIFECYCLE_TIMEOUT_S)
        response.raise_for_status()
        return response.json()

    def abort_fast_checkpoint(self) -> dict:
        response = self._client.post(
            "/lifecycle/abort_fast_checkpoint", timeout=_LIFECYCLE_TIMEOUT_S
        )
        response.raise_for_status()
        return response.json()

    def is_ready(self) -> bool:
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json() == {"ready": True}

    def daemon_identity(self) -> "tuple[int, int] | None":
        response = self._client.get("/health")
        response.raise_for_status()
        value = response.headers.get("X-Agency-Daemon-Pid")
        start = response.headers.get("X-Agency-Daemon-Start-Ticks")
        if (
            value is None
            or not value.isdecimal()
            or int(value) <= 0
            or start is None
            or not start.isdecimal()
        ):
            return None
        return int(value), int(start)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HarnessInteractionClient":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


__all__ = ["HarnessInteractionClient"]
