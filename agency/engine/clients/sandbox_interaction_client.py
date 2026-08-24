"""Host-side client for the sandbox Harness Manager interaction server."""

from __future__ import annotations

from dataclasses import asdict

import httpx

from ...harness.protocol import HarnessAttemptRequest, HarnessAttemptResult


class SandboxInteractionClient:
    def __init__(self, uds_path: str, timeout_s: float = 300.0) -> None:
        self._client = httpx.Client(
            transport=httpx.HTTPTransport(uds=uds_path),
            base_url="http://agency-sandbox",
            timeout=timeout_s,
        )

    def run_harness_attempt(self, request: HarnessAttemptRequest) -> HarnessAttemptResult:
        response = self._client.post("/harness_attempt", json=asdict(request))
        response.raise_for_status()
        return HarnessAttemptResult(**response.json())

    def is_ready(self) -> bool:
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json() == {"ready": True}

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SandboxInteractionClient":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


__all__ = ["SandboxInteractionClient"]
