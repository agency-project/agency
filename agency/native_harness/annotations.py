"""Opt-in, non-enforcing agent-stated tool purposes (stdlib only)."""

from __future__ import annotations

import copy
import json
import os
import time
import uuid
from pathlib import Path

ARMS = ("baseline", "schema_only", "purpose", "purpose_workstreams")
VERSION = "agency-purpose-v1"
PURPOSE = "For each tool call, state its immediate purpose in one short sentence."
WORKSTREAMS = (
    "Assign stable workstream IDs for the subtasks this call advances. "
    "Reuse IDs when continuing a subtask. A call may advance multiple workstreams."
)


def instruction(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"Unknown annotation arm: {arm}")
    if arm == "purpose_workstreams":
        return PURPOSE + "\n" + WORKSTREAMS
    return PURPOSE if arm == "purpose" else ""


def augment_schemas(schemas: list[dict], arm: str) -> list[dict]:
    instruction(arm)
    if arm == "baseline":
        return schemas
    result = copy.deepcopy(schemas)
    for schema in result:
        function = schema["function"]
        parameters = function["parameters"]
        properties = parameters.setdefault("properties", {})
        if "_agency" in properties or "_agency" in parameters.get("required", []):
            raise ValueError(f"Reserved _agency argument collision in {function['name']}")
        if parameters.get("type", "object") != "object" or "$ref" in parameters:
            raise ValueError(f"Tool {function['name']} needs a concrete object schema")
        annotation_properties = {"purpose": {"type": "string"}}
        if arm == "purpose_workstreams":
            annotation_properties["workstream_ids"] = {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            }
        properties["_agency"] = {
            "type": "object",
            "properties": annotation_properties,
            "additionalProperties": False,
        }
        # Strict provider schemas often require every property. This intervention
        # deliberately leaves metadata optional rather than enabling strict mode.
        function.pop("strict", None)
    return result


def extract(arguments: str, arm: str) -> tuple[str, dict]:
    """Return executable JSON and metadata; malformed labels never cause retries."""
    record = {"source": "executing_model", "version": VERSION, "status": "missing", "raw": None}
    if arm == "baseline":
        record["status"] = "not_requested"
        return arguments, record
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        record["status"] = "malformed_arguments"
        return arguments, record
    if not isinstance(parsed, dict):
        record["status"] = "malformed_arguments"
        return arguments, record
    if "_agency" not in parsed:
        return arguments, record
    metadata = parsed.pop("_agency")
    record["raw"] = metadata
    valid = isinstance(metadata, dict)
    if valid:
        purpose = metadata.get("purpose")
        valid = isinstance(purpose, str) and bool(purpose.strip())
        allowed = {"purpose", "workstream_ids"} if arm == "purpose_workstreams" else {"purpose"}
        valid = valid and not (set(metadata) - allowed)
        if arm == "purpose_workstreams":
            ids = metadata.get("workstream_ids")
            valid = valid and isinstance(ids, list) and bool(ids)
            if isinstance(ids, list):
                valid = valid and all(isinstance(i, str) and bool(i.strip()) for i in ids)
                valid = valid and len(set(map(str, ids))) == len(ids)
    record["status"] = "valid" if valid else "malformed"
    return json.dumps(parsed), record


class TraceWriter:
    """Append and flush each event so interrupted executions retain their prefix.

    Credentials belong to transport configuration, never to this writer. Only
    conversation/tool payloads are accepted, not request headers or environment.
    """

    def __init__(self, path: str, *, agent_id: str, run_id: str, secrets=()):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = {"agent_id": agent_id, "run_id": run_id}
        self.secrets = tuple(secret for secret in secrets if secret)

    def __call__(self, kind: str, payload: dict) -> None:
        event = redact(
            {"kind": kind, "timestamp_ns": time.time_ns(), **self.identity, **payload}, self.secrets
        )
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()


def event_id() -> str:
    return uuid.uuid4().hex


def redact(value, secrets=()):
    """Scrub transport secrets even if an error or tool output echoes them."""
    protected = list(secrets)
    protected.extend(
        item
        for key, item in os.environ.items()
        if any(name in key.upper() for name in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))
    )
    if isinstance(value, str):
        for secret in protected:
            if secret and len(secret) >= 8:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {key: redact(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, secrets) for item in value]
    return value
