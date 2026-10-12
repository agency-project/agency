"""Bounded, read-only evidence for a particular local harness control request."""

from collections import OrderedDict
import time
import uuid


class ControlObservations:
    def __init__(self):
        self.rows = OrderedDict()

    def add(self, action, handle, request_id, attempt_token):
        for row in self.rows.values():
            if row["handle"] is handle and row["result"]["state"] == "pending":
                row["result"].update(
                    state="unknown", detail="A later control superseded this observation."
                )
                row["handle"] = None
        identifier = uuid.uuid4().hex
        self.rows[identifier] = {
            "handle": handle,
            "result": {
                "id": identifier,
                "action": action,
                "request_id": request_id,
                "attempt_token": attempt_token,
                "state": "pending",
                "scope": "local_harness_tasks",
            },
        }
        while len(self.rows) > 64:
            self.rows.popitem(last=False)
        return self.get(identifier)

    def get(self, identifier):
        row = self.rows.get(identifier)
        if row is None:
            return {
                "id": identifier,
                "state": "unknown",
                "detail": "Observation is no longer available in this daemon.",
            }
        result = row["result"]
        if result["state"] != "pending":
            return dict(result)
        snapshot = getattr(row["handle"], "_control_snapshot", None)
        if not callable(snapshot):
            result.update(
                state="unavailable",
                detail="No observable local harness handle was registered at dispatch.",
            )
        else:
            try:
                facts = snapshot()
            except Exception as error:
                result.update(state="unknown", detail=f"Local observation failed: {error}")
                row["handle"] = None
                return dict(result)
            if not isinstance(facts, dict):
                result.update(
                    state="unavailable", detail="Local observation is unsupported by this handle."
                )
            else:
                local = facts.get("local_state", "unknown")
                result.update(local_state=local, observed_at=time.time())
                for key in ("task_count", "parked_count", "returncode"):
                    if isinstance(facts.get(key), int) and not isinstance(facts[key], bool):
                        result[key] = facts[key]
                desired = {"pause": "suspended", "resume": "continued", "cancel": "exited"}
                if local == desired.get(result["action"]):
                    result.update(
                        state="confirmed",
                        detail="Observed local task state; remote work and cleanup are not established.",
                    )
                elif local == "exited":
                    result.update(
                        state="target_exited",
                        detail="The local task tree exited before the requested state was observed.",
                    )
                elif local == "unknown":
                    result.update(
                        state="unknown", detail="The local task state cannot be established."
                    )
        if result["state"] != "pending":
            row["handle"] = None  # Keep the evidence, release completed process/output references.
        return dict(result)
