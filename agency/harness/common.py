"""Small helpers shared by more than one route module in this package."""

from __future__ import annotations

from fastapi import Request


def extract_bearer_token(request: Request) -> "str | None":
    auth = request.headers.get("authorization") or request.headers.get("x-api-key")
    if not auth:
        return None
    if auth.lower().startswith("bearer "):
        return auth[len("Bearer ") :].strip()
    return auth.strip()


__all__ = ["extract_bearer_token"]
