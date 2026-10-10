"""Execution-owned, narrow profiler command relay. No browser-supplied callables."""

from __future__ import annotations

import hashlib
import json
import time
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from collections import OrderedDict
from pathlib import Path


def command_key(identifier: str) -> str:
    return hashlib.sha256(identifier.encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, default=str), encoding="utf-8")
    temporary.replace(path)


class ProfilerRelay:
    """One UUID-fenced execution process; receipts acknowledge only observed facts."""

    def __init__(self, directory: Path, tasks: dict | None = None):
        from ...agskill import agskill

        self.directory = directory
        self.identifier = uuid.uuid4().hex
        self.tasks = dict(tasks or {})
        for key, skill in self.tasks.items():
            if not isinstance(key, str) or not key.strip() or not isinstance(skill, agskill):
                raise ValueError("ui_tasks must map nonblank task IDs to existing agskill objects")
        self.receipts: dict[str, dict] = {}
        self.seen: set[str] = set()
        self.lock = threading.RLock()
        self._observers = OrderedDict()
        self._observation_pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="profiler-observation"
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "receipts").mkdir(exist_ok=True)

    def publish(self):
        from ...agent import agent

        self._poll_observations()

        task_specs = [
            {
                "id": key,
                "label": skill.name,
                "input_schema": skill.input_schema.to_json() if skill.input_schema else None,
            }
            for key, skill in self.tasks.items()
        ]
        capabilities = {}
        for target in agent.all():
            orchestrator = target._orchestrator
            with orchestrator._event_cond:
                invocation = orchestrator._active_by_agent.get(target)
            # These adapters all use the verified PTY redirect seam. Native
            # has no interrupting redirect handler; future context still works.
            redirect = (
                [invocation]
                if invocation
                and target.harness in {"claude_code", "codex", "opencode", "kimi", "grok"}
                and not target.is_paused()
                else []
            )
            capabilities[str(target.agname)] = {
                "queue_message": True,
                "redirect": redirect,
                "run": task_specs,
            }
        atomic_json(
            self.directory / ".session",
            {
                "relay_id": self.identifier,
                "capabilities": capabilities,
                "control_observations": True,
            },
        )

    def _save(self, receipt):
        with self.lock:
            self.receipts[receipt["id"]] = receipt
            atomic_json(self.directory / "receipts" / f"{command_key(receipt['id'])}.json", receipt)
            while len(self.receipts) > 200:
                oldest = next(iter(self.receipts))
                del self.receipts[oldest]
                (self.directory / "receipts" / f"{command_key(oldest)}.json").unlink(
                    missing_ok=True
                )

    def _event(self, command, target, state, result=None):
        from ...orchestrator import peek_orchestrator

        orchestrator = peek_orchestrator()
        if orchestrator is not None:
            orchestrator.data_logger.record_event(
                "ui_intervention",
                {
                    "command_id": command["id"],
                    "requested_action": command["type"],
                    "target_agent": target,
                    "state": state,
                    "message": command.get("message"),
                    "invocation": command.get("invocation"),
                    "task": command.get("task"),
                    "input": command.get("input"),
                    "result": result or {},
                    "relay_id": self.identifier,
                },
                name=target,
                object="agwebui",
                flush=True,
            )

    def _record_event(self, receipt, command, target, state, result=None):
        """Recording failure is separate from an already-known SDK outcome."""
        try:
            self._event(command, target, state, result)
        except Exception as error:
            receipt["recording_error"] = str(error)

    @staticmethod
    def _read_observations(observations):
        from ...engine.clients import HarnessInteractionClient

        evidence = []
        for expected in observations:
            # A previously observed fact is historical evidence. Losing the
            # socket later must not erase a confirmed dispatch-time outcome.
            if expected.get("state") in {"confirmed", "target_exited", "unknown", "unavailable"}:
                evidence.append(
                    {key: value for key, value in expected.items() if key != "uds_path"}
                )
                continue
            try:
                with HarnessInteractionClient(expected["uds_path"], timeout_s=0.5) as client:
                    actual = client._control_observation(expected["id"])
                if any(
                    actual.get(key) != expected.get(key)
                    for key in ("id", "request_id", "attempt_token", "action")
                ):
                    raise ValueError("Harness observation identity changed or expired")
                evidence.append(actual)
            except Exception as error:
                evidence.append(
                    {"id": expected.get("id"), "state": "unknown", "detail": str(error)}
                )
        return ProfilerRelay._application_evidence(evidence)

    @staticmethod
    def _application_evidence(evidence):
        states = [item.get("state", "unknown") for item in evidence]
        state = (
            "confirmed"
            if states and all(s == "confirmed" for s in states)
            else "pending"
            if "pending" in states
            else "partial"
            if "confirmed" in states
            else "unavailable"
            if states and all(s == "unavailable" for s in states)
            else "unknown"
        )
        return {"state": state, "scope": "local_harness_tasks", "observations": evidence}

    def _watch_control(self, receipt, command, result, detail):
        observations = detail.get("observations", [])
        if (
            not observations
            or detail.get("observation_expected", len(observations)) != len(observations)
            or len(observations) > 16
            or len(self._observers) >= 128
        ):
            result["application"] = {
                "state": "unavailable",
                "scope": "local_harness_tasks",
                "detail": "Not all targeted local harnesses have observable dispatch evidence.",
            }
            return
        result["application"] = {"state": "pending", "scope": "local_harness_tasks"}
        initial = None
        if all(
            item.get("state") in {"confirmed", "unknown", "unavailable", "target_exited"}
            for item in observations
        ):
            initial = Future()
            initial.set_result(
                self._application_evidence(
                    [
                        {key: value for key, value in item.items() if key != "uds_path"}
                        for item in observations
                    ]
                )
            )
        self._observers[(receipt["id"], result["agname"])] = {
            "receipt": receipt,
            "command": command,
            "result": result,
            "observations": observations,
            "deadline": time.monotonic() + 20,
            "future": initial,
            "signature": None,
        }

    def _poll_observations(self, *, schedule=True):
        running = sum(
            row["future"] is not None and not row["future"].done()
            for row in self._observers.values()
        )
        for key, row in list(self._observers.items()):
            future = row["future"]
            application = None
            if future is not None and future.done():
                try:
                    application = future.result()
                except Exception as error:
                    application = {
                        "state": "unknown",
                        "detail": str(error),
                        "scope": "local_harness_tasks",
                    }
                row["future"] = None
            if (
                application is None or application["state"] == "pending"
            ) and time.monotonic() >= row["deadline"]:
                evidence = [
                    {key: value for key, value in item.items() if key != "uds_path"}
                    if item.get("state") == "confirmed"
                    else {
                        "id": item.get("id"),
                        "state": "unknown",
                        "detail": "Observation deadline expired.",
                    }
                    for item in row["observations"]
                ]
                application = {
                    **self._application_evidence(evidence),
                    "detail": "Timed out observing local application; no mutation retry was sent.",
                }
            if application is not None:
                row["result"]["application"] = application
                known = {
                    item.get("id"): item
                    for item in application.get("observations", [])
                    if item.get("state") != "pending"
                }
                row["observations"] = [
                    {**item, **known.get(item.get("id"), {})} for item in row["observations"]
                ]
                # Measurement times change on each read; record substantive transitions only.
                meaningful = json.dumps(
                    {
                        **application,
                        "observations": [
                            {k: v for k, v in item.items() if k != "observed_at"}
                            for item in application.get("observations", [])
                        ],
                    },
                    sort_keys=True,
                )
                if meaningful != row["signature"]:
                    row["signature"] = meaningful
                    self._record_event(
                        row["receipt"],
                        row["command"],
                        row["result"]["agname"],
                        "observed"
                        if application["state"] == "confirmed"
                        else "unknown"
                        if application["state"] != "pending"
                        else "accepted",
                        row["result"],
                    )
                    self._save(row["receipt"])
                if application["state"] != "pending":
                    del self._observers[key]
                    continue
            if schedule and row["future"] is None and running < 4:
                row["future"] = self._observation_pool.submit(
                    self._read_observations, row["observations"]
                )
                running += 1
                self._observers.move_to_end(key)

    def dispatch(self, command):
        from ...agent import agent
        from ...agdata import agdata

        identifier = command.get("id")
        if not isinstance(identifier, str) or not identifier or len(identifier) > 128:
            raise ValueError("A bounded command ID is required")
        if identifier in self.seen:
            return self.receipts.get(
                identifier,
                {
                    "id": identifier,
                    "state": "unknown",
                    "detail": "Already processed; older receipt is no longer retained.",
                },
            )
        if len(self.seen) >= 2000:
            raise ValueError("This relay session reached its 2000-command limit")
        self.seen.add(identifier)
        receipt = {
            "id": identifier,
            "command": command.get("type"),
            "relay_id": self.identifier,
            "state": "unknown",
            "targets": [],
            "time": time.time(),
        }
        self._save(receipt)  # A crash after submission must never cause a blind retry.
        if command.get("relay_id") != self.identifier:
            receipt.update(
                state="failed", detail="Execution process changed; command was not applied."
            )
            self._save(receipt)
            return receipt
        targets = command.get("targets") or []
        by_name = {str(target.agname): target for target in agent.all()}
        for name in targets:
            self._record_event(receipt, command, name, "requested")
            result = {"agname": name, "state": "failed"}
            try:
                target = by_name.get(name)
                if target is None:
                    raise ValueError("The targeted agent no longer exists")
                kind = command["type"].removesuffix("_all")
                previous_detail = getattr(target, "_last_control_receipt", None)
                detail = {}
                if kind in {"pause", "resume"}:
                    (target.pause if kind == "pause" else target.resume)()
                    fresh = getattr(target, "_last_control_receipt", None)
                    detail = (
                        fresh if isinstance(fresh, dict) and fresh is not previous_detail else {}
                    )
                    result.update(
                        {
                            key: detail[key]
                            for key in ("requested", "rpc_acknowledged", "error", "detail")
                            if key in detail
                        },
                        state="accepted",
                        scope="recorded_agent_intent" if detail else "agent_api_accepted",
                    )
                    if detail.get("error"):
                        result.update(state="failed", detail=detail["error"])
                elif kind in {"stop", "kill"}:
                    target.stop(force=kind == "kill")
                    fresh = getattr(target, "_last_control_receipt", None)
                    detail = (
                        fresh if isinstance(fresh, dict) and fresh is not previous_detail else {}
                    )
                    result.update(state="accepted", scope="stop_requested_not_process_exit")
                elif kind == "queue_message":
                    result.update(target._invoke_queue_message_receipt(command["message"]))
                elif kind == "redirect":
                    invocation = command["invocation"]
                    with target._orchestrator._event_cond:
                        request = target._orchestrator._requests.get(invocation)
                        if (
                            request is None
                            or request.agent is not target
                            or request.kind != "skill"
                            or request.state != "running"
                            or target._orchestrator._active_by_agent.get(target) != invocation
                        ):
                            raise ValueError(
                                "Target invocation ended or changed; redirect was not submitted"
                            )
                        if target.harness not in {
                            "claude_code",
                            "codex",
                            "opencode",
                            "kimi",
                            "grok",
                        }:
                            raise ValueError(
                                "Current harness does not support interrupting redirects"
                            )
                        handle = agdata(_future=request.result_future)
                        object.__setattr__(handle, "_execution_id", invocation)
                        object.__setattr__(handle, "_execution_agent", target)
                    result.update(target._redirect_receipt(handle, command["message"]))
                elif kind == "run":
                    skill = self.tasks.get(command.get("task"))
                    if skill is None:
                        raise ValueError("Task is not registered for this profiler session")
                    if not isinstance(command.get("input"), dict):
                        raise ValueError("Task input must be a JSON object")
                    if any(
                        not isinstance(key, str) or key.startswith("_") for key in command["input"]
                    ):
                        raise ValueError("Task input keys must be public data fields")
                    data = agdata(**command["input"])
                    if skill.input_schema and (error := skill.input_schema.validate_input(data)):
                        raise ValueError(error)
                    handle = target.run(skill, data)
                    result.update(
                        state="submitted",
                        invocation=object.__getattribute__(handle, "_execution_id"),
                        scope="scheduler_submission_not_completion",
                    )
                else:
                    raise ValueError("Unsupported profiler operation")
            except Exception as error:
                result.update(state="failed", detail=str(error))
            future = result.pop("future", None)
            if (
                command["type"].removesuffix("_all") in {"pause", "resume", "stop", "kill"}
                and result["state"] != "failed"
            ):
                self._watch_control(receipt, command, result, detail)
            receipt["targets"].append(result)
            self._record_event(receipt, command, name, result["state"], result)
            if future is not None:

                def completed(done, result=result, name=name):
                    try:
                        done.result()
                        result["context_state"] = "committed"
                    except Exception as error:
                        result.update(context_state="failed", detail=str(error))
                    event_state = (
                        "context_committed" if result["context_state"] == "committed" else "failed"
                    )
                    self._record_event(receipt, command, name, event_state, result)
                    self._save(receipt)

                future.add_done_callback(completed)
        failures = sum(item["state"] == "failed" for item in receipt["targets"])
        receipt["state"] = (
            "failed"
            if not targets or failures == len(targets)
            else "partial"
            if failures
            else "accepted"
        )
        self._save(receipt)
        return receipt

    def close(self):
        self._poll_observations(schedule=False)
        for row in self._observers.values():
            evidence = [
                {key: value for key, value in item.items() if key != "uds_path"}
                if item.get("state") == "confirmed"
                else {"id": item.get("id"), "state": "unknown"}
                for item in row["observations"]
            ]
            row["result"]["application"] = {
                **self._application_evidence(evidence),
                "detail": "Execution relay closed before all local outcomes were observed.",
            }
            self._record_event(
                row["receipt"], row["command"], row["result"]["agname"], "unknown", row["result"]
            )
            self._save(row["receipt"])
        self._observers.clear()
        self._observation_pool.shutdown(wait=False, cancel_futures=True)
        try:
            owned = (
                json.loads((self.directory / ".session").read_text()).get("relay_id")
                == self.identifier
            )
        except (OSError, ValueError):
            owned = False
        if owned:
            (self.directory / ".session").unlink(missing_ok=True)
            (self.directory / ".heartbeat").unlink(missing_ok=True)
