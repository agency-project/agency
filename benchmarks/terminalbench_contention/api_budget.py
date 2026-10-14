"""Shared, agent-agnostic API admission budget with a global ledger.

Admission policy (unchanged from the previous benchmark, deliberately naive):
every request, from any agent process, polls one shared rolling-window budget
and is admitted the first time it fits. There is no queue, no priority, no
ordering by agent, size or age. Whoever polls first after budget frees wins.

What changed is the accounting:

* the prompt charge comes from ``tokens.TokenEstimator`` (a tokenizer count of
  the real request, with a recorded safety margin) instead of bytes / 2;
* the completion reservation, the prompt estimate and the model-reported usage
  are tracked as separate numbers;
* a reservation can be reconciled against reported usage under an explicit,
  configurable rule;
* every admission, reconciliation, failure and 429 is appended to one ledger
  shared by all processes.

All state lives in one directory guarded by one ``flock``. Timestamps are
``CLOCK_MONOTONIC`` seconds, which is host-wide on Linux, so entries written by
different worker processes are directly comparable; ``epoch_offset_ns`` in the
state maps them to wall-clock time.

The local budget is not a provider limit. ``tokens_per_window`` defaults to the
historical 1.2M and must be overridden by the operator once the gateway's real
limits are verified. Model-reported usage is never assumed to equal what an
upstream provider charged against its own rate limit.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .tokens import EstimatorConfig, TokenEstimator, completion_cap, legacy_estimate

STATE_VERSION = 2
RECONCILIATION_MODES = ("conservative", "reported_total", "litellm_v3", "off")
COMPLETION_RESERVATIONS = ("max_completion_tokens", "none")
# A waiter that stops refreshing itself (its process died) is forgotten after this.
STALE_WAITER_SECONDS = 15.0


@dataclass(frozen=True)
class BudgetPolicy:
    """The single admission policy every agent in every condition shares."""

    version: int = STATE_VERSION
    route: str = "litellm"
    admission_order: str = "opportunistic"  # documented constant; not configurable
    window_seconds: float = 60.0
    # Historical local budget. Not a verified provider limit.
    tokens_per_window: int = 1_200_000
    # No request-rate cap unless the operator supplies a verified one.
    requests_per_window: "int | None" = None
    minimum_request_spacing_seconds: float = 0.05
    rate_limit_cooldown_seconds: float = 60.0
    poll_interval_seconds: float = 0.25
    # "max_completion_tokens": reserve the request's declared output cap up
    # front, as OpenAI-style and LiteLLM (>=1.7x) TPM limiters do. "none":
    # charge the prompt estimate only.
    completion_reservation: str = "max_completion_tokens"
    default_completion_tokens: int = 4096
    # How a reservation is settled once the model reports usage:
    #   conservative   final = max(admitted charge, reported total). Never refunds.
    #   reported_total final = reported prompt + completion tokens.
    #   litellm_v3     final = reported total minus cached prompt tokens, and a
    #                  failed request is refunded (LiteLLM proxy v3 limiter).
    #   off            the admitted charge is never changed.
    reconciliation: str = "conservative"
    # Whether a request that produced no usage (error, 429, cancellation)
    # releases its charge. Forced True by litellm_v3.
    release_on_failure: bool = False
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)

    def validate(self) -> None:
        if self.admission_order != "opportunistic":
            raise ValueError("This benchmark only implements opportunistic admission")
        if self.reconciliation not in RECONCILIATION_MODES:
            raise ValueError(f"reconciliation must be one of {RECONCILIATION_MODES}")
        if self.completion_reservation not in COMPLETION_RESERVATIONS:
            raise ValueError(f"completion_reservation must be one of {COMPLETION_RESERVATIONS}")
        if self.window_seconds <= 0 or self.tokens_per_window <= 0:
            raise ValueError("window_seconds and tokens_per_window must be positive")
        if self.requests_per_window is not None and self.requests_per_window <= 0:
            raise ValueError("requests_per_window must be positive or null")
        if min(self.minimum_request_spacing_seconds, self.rate_limit_cooldown_seconds) < 0:
            raise ValueError("spacing and cooldown cannot be negative")
        if self.poll_interval_seconds <= 0 or self.default_completion_tokens <= 0:
            raise ValueError("poll_interval_seconds and default_completion_tokens must be > 0")
        self.estimator.validate()

    @property
    def refunds_failures(self) -> bool:
        return self.release_on_failure or self.reconciliation == "litellm_v3"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BudgetPolicy":
        data = dict(data)
        estimator = data.pop("estimator", None)
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown API budget policy field(s): {sorted(unknown)}")
        policy = cls(**data)
        if estimator is not None:
            unknown = set(estimator) - set(EstimatorConfig.__dataclass_fields__)
            if unknown:
                raise ValueError(f"Unknown estimator field(s): {sorted(unknown)}")
            policy = replace(policy, estimator=EstimatorConfig(**estimator))
        policy.validate()
        return policy

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def duration(value):
    """Parse a provider duration header (``1m12.5s``) or numeric Retry-After."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        parts = re.findall(r"([\d.]+)(ms|s|m|h)", value)
        if not parts or "".join(n + u for n, u in parts) != value:
            return None
        return sum(float(n) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[u] for n, u in parts)


class DeadlineReached(TimeoutError):
    """The run deadline passed while a request was waiting for admission."""


class AdmissionCancelled(RuntimeError):
    """The owning agent was cancelled while a request was waiting."""


class LedgerCorrupted(RuntimeError):
    """Shared admission state is unreadable or inconsistent. Fail closed."""


class RequestTooLarge(RuntimeError):
    """A single request's charge exceeds the whole window budget."""


@dataclass
class Admission:
    request_id: str
    agent_id: "str | None"
    attempt: int
    ready_mono: float
    admitted_mono: float
    wait_seconds: float
    polls: int
    estimated_prompt_tokens: int
    raw_prompt_tokens: int
    reserved_completion_tokens: int
    charge: int
    legacy_charge: "int | None"
    estimate_method: str
    window_used_before: int
    window_requests_before: int
    available_before: int
    available_after: int
    waiters_at_ready: int
    waiters_at_admission: int
    overtaken_waiters: int
    blocked_by: list
    sequence: int

    def to_dict(self) -> dict:
        return asdict(self)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


class SharedBudget:
    """One client of the shared budget. Create one per process (or thread)."""

    def __init__(
        self,
        directory,
        policy: BudgetPolicy,
        *,
        clock=time.monotonic,
        sleep=time.sleep,
        deadline=None,
        cancelled=None,
        estimator: "TokenEstimator | None" = None,
    ) -> None:
        policy.validate()
        self.directory = Path(directory)
        self.policy = policy
        self.clock, self.sleep = clock, sleep
        self._deadline = deadline  # callable -> monotonic seconds or None
        self._cancelled = cancelled  # callable -> bool
        self.estimator = estimator or TokenEstimator(policy.estimator)
        self.state_path = self.directory / "state.json"
        self.ledger_path = self.directory / "ledger.jsonl"
        self.lock_path = self.directory / "state.lock"

    # ------------------------------------------------------------------ setup

    @classmethod
    def create(cls, directory, policy: BudgetPolicy, *, run_id: str, **kwargs) -> "SharedBudget":
        """Initialize a fresh ledger. Refuses to reuse an existing one."""
        policy.validate()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        leftovers = [p.name for p in directory.iterdir()]
        if leftovers:
            raise FileExistsError(
                f"API budget directory {directory} is not empty ({leftovers[:4]}); "
                "every run needs a fresh admission ledger"
            )
        budget = cls(directory, policy, **kwargs)
        now = budget.clock()
        state = dict(
            version=STATE_VERSION,
            run_id=run_id,
            policy=policy.to_dict(),
            policy_fingerprint=policy.fingerprint(),
            created_mono=now,
            # wall-clock ns = monotonic ns + epoch_offset_ns (captured once).
            epoch_offset_ns=time.time_ns() - time.monotonic_ns(),
            sequence=0,
            entries=[],
            cooldown_until=0.0,
            next_request=0.0,
            waiters={},
            totals=dict(
                admitted=0,
                admitted_charge=0,
                estimated_prompt_tokens=0,
                reserved_completion_tokens=0,
                reconciled=0,
                reported_prompt_tokens=0,
                reported_completion_tokens=0,
                reported_cached_tokens=0,
                final_charge=0,
                failed=0,
                released_charge=0,
                rate_limited=0,
                wait_censored=0,
                wait_seconds=0.0,
            ),
        )
        with budget._locked():
            _atomic_write(budget.state_path, json.dumps(state))
            budget.ledger_path.write_text("")
        return budget

    # ------------------------------------------------------------- primitives

    class _Lock:
        def __init__(self, path: Path) -> None:
            self.path = path

        def __enter__(self):
            self.handle = self.path.open("a")
            fcntl.flock(self.handle, fcntl.LOCK_EX)
            return self

        def __exit__(self, *exc):
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()

    def _locked(self):
        return SharedBudget._Lock(self.lock_path)

    def _corrupt(self, reason: str):
        """Preserve the unreadable state for diagnosis and refuse to continue."""
        try:
            if self.state_path.exists():
                copy = self.directory / f"state.corrupt-{time.time_ns()}.json"
                shutil.copy2(self.state_path, copy)
        except OSError:
            pass  # diagnostics are best effort; the failure below is what matters
        raise LedgerCorrupted(f"Shared API admission state is unusable: {reason}")

    def _load(self) -> dict:
        try:
            state = json.loads(self.state_path.read_text())
        except FileNotFoundError:
            self._corrupt("state.json is missing (ledger was never initialized or was removed)")
        except (ValueError, UnicodeDecodeError) as exc:
            self._corrupt(f"state.json is not valid JSON ({exc})")
        if not isinstance(state, dict) or state.get("version") != STATE_VERSION:
            self._corrupt("state.json has an unexpected schema version")
        for key, kind in (
            ("entries", list),
            ("waiters", dict),
            ("totals", dict),
            ("sequence", int),
            ("cooldown_until", (int, float)),
            ("next_request", (int, float)),
        ):
            if not isinstance(state.get(key), kind):
                self._corrupt(f"state.json field {key!r} is missing or malformed")
        if state.get("policy_fingerprint") != self.policy.fingerprint():
            raise LedgerCorrupted(
                "This client's API policy differs from the ledger's; all agents must "
                "share one admission policy"
            )
        return state

    def _save(self, state: dict) -> None:
        _atomic_write(self.state_path, json.dumps(state))

    def _append(self, state: dict, event: dict) -> int:
        state["sequence"] += 1
        record = dict(seq=state["sequence"], mono=self.clock(), pid=os.getpid(), **event)
        with self.ledger_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        return state["sequence"]

    def _live(self, state: dict, now: float) -> list:
        window = self.policy.window_seconds
        return [e for e in state["entries"] if e["t"] + window > now]

    # -------------------------------------------------------------- admission

    def charge_for(self, body: bytes):
        """(prompt estimate, reserved completion tokens, legacy charge, payload)."""
        estimate = self.estimator.estimate_bytes(body)
        try:
            payload = json.loads(body)
            if not isinstance(payload, dict):
                payload = {}
        except (ValueError, UnicodeDecodeError):
            payload = {}
        reserved = (
            completion_cap(payload, self.policy.default_completion_tokens)
            if self.policy.completion_reservation == "max_completion_tokens"
            else 0
        )
        legacy = legacy_estimate(body, payload) if payload else None
        return estimate, reserved, legacy

    def admit(
        self,
        body: bytes,
        *,
        request_id: "str | None" = None,
        agent_id: "str | None" = None,
        attempt: int = 0,
    ) -> Admission:
        """Block until this request fits the shared budget, then record it.

        ``agent_id`` is recorded for analysis only. It is never read when
        deciding whether or when to admit.
        """
        policy = self.policy
        request_id = request_id or uuid.uuid4().hex
        estimate, reserved, legacy = self.charge_for(body)
        charge = estimate.estimated_tokens + reserved
        if charge > policy.tokens_per_window:
            raise RequestTooLarge(
                f"Request charge {charge} exceeds the shared window budget "
                f"{policy.tokens_per_window}"
            )
        ready = self.clock()
        polls = 0
        waiters_at_ready = None
        blocked_by: set = set()
        registered = False
        try:
            while True:
                now = self.clock()
                deadline = self._deadline() if self._deadline else None
                if deadline is not None and now >= deadline:
                    raise DeadlineReached("Run deadline reached while waiting for API budget")
                if self._cancelled is not None and self._cancelled():
                    raise AdmissionCancelled("Agent cancelled while waiting for API budget")
                polls += 1
                with self._locked():
                    state = self._load()
                    entries = self._live(state, now)
                    used = sum(e["charge"] for e in entries)
                    waiters = {
                        key: value
                        for key, value in state["waiters"].items()
                        if value["seen"] + STALE_WAITER_SECONDS > now
                    }
                    others = {k: v for k, v in waiters.items() if k != request_id}
                    if waiters_at_ready is None:
                        waiters_at_ready = len(others)
                    wait_until = 0.0
                    if state["cooldown_until"] > now:
                        wait_until = max(wait_until, state["cooldown_until"])
                        blocked_by.add("rate_limit_cooldown")
                    if state["next_request"] > now:
                        wait_until = max(wait_until, state["next_request"])
                        blocked_by.add("request_spacing")
                    if used + charge > policy.tokens_per_window:
                        # Earliest moment enough charge has left the window.
                        freed, needed = 0, used + charge - policy.tokens_per_window
                        release = entries[-1]["t"] + policy.window_seconds
                        for entry in entries:
                            freed += entry["charge"]
                            if freed >= needed:
                                release = entry["t"] + policy.window_seconds
                                break
                        wait_until = max(wait_until, release)
                        blocked_by.add("token_budget")
                    if (
                        policy.requests_per_window is not None
                        and len(entries) + 1 > policy.requests_per_window
                    ):
                        wait_until = max(wait_until, entries[0]["t"] + policy.window_seconds)
                        blocked_by.add("request_budget")
                    if wait_until <= now:
                        waiters.pop(request_id, None)
                        overtaken = sum(v["since"] < ready for v in waiters.values())
                        entry = dict(t=now, charge=charge, id=request_id, admitted=charge)
                        entries.append(entry)
                        totals = state["totals"]
                        totals["admitted"] += 1
                        totals["admitted_charge"] += charge
                        totals["estimated_prompt_tokens"] += estimate.estimated_tokens
                        totals["reserved_completion_tokens"] += reserved
                        totals["wait_seconds"] += now - ready
                        state.update(
                            entries=entries,
                            waiters=waiters,
                            next_request=now + policy.minimum_request_spacing_seconds,
                        )
                        admission = Admission(
                            request_id=request_id,
                            agent_id=agent_id,
                            attempt=attempt,
                            ready_mono=ready,
                            admitted_mono=now,
                            wait_seconds=now - ready,
                            polls=polls,
                            estimated_prompt_tokens=estimate.estimated_tokens,
                            raw_prompt_tokens=estimate.raw_tokens,
                            reserved_completion_tokens=reserved,
                            charge=charge,
                            legacy_charge=legacy,
                            estimate_method=estimate.method,
                            window_used_before=used,
                            window_requests_before=len(entries) - 1,
                            available_before=policy.tokens_per_window - used,
                            available_after=policy.tokens_per_window - used - charge,
                            waiters_at_ready=waiters_at_ready,
                            waiters_at_admission=len(waiters),
                            overtaken_waiters=overtaken,
                            blocked_by=sorted(blocked_by),
                            sequence=0,
                        )
                        admission.sequence = self._append(
                            state, dict(kind="admit", **admission.to_dict())
                        )
                        self._save(state)
                        registered = False
                        return admission
                    # Not admitted: make this request visible as a waiter.
                    mine = waiters.get(request_id)
                    if mine is None or now - mine["seen"] > 2.0:
                        waiters[request_id] = dict(
                            since=ready, seen=now, pid=os.getpid(), charge=charge
                        )
                        state["waiters"] = waiters
                        state["entries"] = entries
                        self._save(state)
                    registered = True
                limit = max(0.001, wait_until - now)
                if deadline is not None:
                    limit = min(limit, max(0.001, deadline - now))
                self.sleep(min(policy.poll_interval_seconds, limit))
        except (DeadlineReached, AdmissionCancelled) as exc:
            self._abandon(request_id, agent_id, ready, charge, type(exc).__name__, registered)
            raise

    def _abandon(self, request_id, agent_id, ready, charge, reason, registered) -> None:
        with self._locked():
            state = self._load()
            state["waiters"].pop(request_id, None)
            state["totals"]["wait_censored"] += 1
            self._append(
                state,
                dict(
                    kind="wait_censored",
                    request_id=request_id,
                    agent_id=agent_id,
                    ready_mono=ready,
                    wait_seconds=self.clock() - ready,
                    charge=charge,
                    reason=reason,
                ),
            )
            self._save(state)

    # --------------------------------------------------------- reconciliation

    def reconcile(
        self,
        request_id: str,
        *,
        prompt_tokens: "int | None",
        completion_tokens: "int | None",
        cached_tokens: int = 0,
        total_tokens: "int | None" = None,
        agent_id: "str | None" = None,
    ) -> dict:
        """Settle a reservation against model-reported usage."""
        policy = self.policy
        prompt = int(prompt_tokens or 0)
        completion = int(completion_tokens or 0)
        cached = int(cached_tokens or 0)
        reported_total = int(total_tokens) if total_tokens else prompt + completion
        with self._locked():
            state = self._load()
            entry = next((e for e in state["entries"] if e["id"] == request_id), None)
            admitted = entry["admitted"] if entry else None
            if entry is not None and entry.get("settled"):
                raise LedgerCorrupted(f"Request {request_id} was settled twice")
            if admitted is None or policy.reconciliation == "off":
                final = admitted
            elif policy.reconciliation == "conservative":
                final = max(admitted, reported_total)
            elif policy.reconciliation == "reported_total":
                final = reported_total
            else:  # litellm_v3
                final = max(0, reported_total - cached)
            if entry is not None:
                entry["charge"] = final
                entry["settled"] = "reconciled"
            totals = state["totals"]
            totals["reconciled"] += 1
            totals["reported_prompt_tokens"] += prompt
            totals["reported_completion_tokens"] += completion
            totals["reported_cached_tokens"] += cached
            if final is not None:
                totals["final_charge"] += final
            record = dict(
                kind="reconcile",
                request_id=request_id,
                agent_id=agent_id,
                mode=policy.reconciliation,
                admitted_charge=admitted,
                final_charge=final,
                adjustment=None if final is None or admitted is None else final - admitted,
                reported_prompt_tokens=prompt,
                reported_completion_tokens=completion,
                reported_cached_tokens=cached,
                reported_total_tokens=reported_total,
                # The reservation left the rolling window before usage arrived.
                entry_expired=entry is None,
            )
            record["seq"] = self._append(state, record)
            self._save(state)
        return record

    def fail(
        self,
        request_id: str,
        *,
        reason: str,
        status_code: "int | None" = None,
        agent_id: "str | None" = None,
    ) -> dict:
        """Close a reservation that produced no usage (error, 429, cancel)."""
        release = self.policy.refunds_failures
        with self._locked():
            state = self._load()
            entry = next((e for e in state["entries"] if e["id"] == request_id), None)
            if entry is not None and entry.get("settled"):
                raise LedgerCorrupted(f"Request {request_id} was settled twice")
            admitted = entry["admitted"] if entry else None
            final = admitted
            if entry is not None:
                if release:
                    final = 0
                    entry["charge"] = 0
                entry["settled"] = "failed"
            totals = state["totals"]
            totals["failed"] += 1
            if final is not None:
                totals["final_charge"] += final
                totals["released_charge"] += admitted - final
            record = dict(
                kind="fail",
                request_id=request_id,
                agent_id=agent_id,
                reason=reason,
                status_code=status_code,
                admitted_charge=admitted,
                final_charge=final,
                released=bool(release and entry is not None),
                entry_expired=entry is None,
            )
            record["seq"] = self._append(state, record)
            self._save(state)
        return record

    def observe_rate_limit(
        self,
        *,
        retry_after=None,
        reset_tokens=None,
        reset_requests=None,
        request_id: "str | None" = None,
        agent_id: "str | None" = None,
    ) -> dict:
        """A 429 pauses admission for every agent, not just the one that saw it."""
        delay = max(
            self.policy.rate_limit_cooldown_seconds,
            duration(retry_after) or 0,
            duration(reset_tokens) or 0,
            duration(reset_requests) or 0,
        )
        with self._locked():
            state = self._load()
            until = max(state["cooldown_until"], self.clock() + delay)
            state["cooldown_until"] = until
            state["totals"]["rate_limited"] += 1
            record = dict(
                kind="rate_limited",
                request_id=request_id,
                agent_id=agent_id,
                cooldown_seconds=delay,
                cooldown_until=until,
                retry_after=retry_after,
                reset_tokens=reset_tokens,
                reset_requests=reset_requests,
            )
            record["seq"] = self._append(state, record)
            self._save(state)
        return record

    # ------------------------------------------------------------- inspection

    def snapshot(self) -> dict:
        """Current window occupancy, for time-series sampling by the runner."""
        with self._locked():
            state = self._load()
        now = self.clock()
        entries = self._live(state, now)
        used = sum(e["charge"] for e in entries)
        waiters = [v for v in state["waiters"].values() if v["seen"] + STALE_WAITER_SECONDS > now]
        return dict(
            mono=now,
            window_used=used,
            window_requests=len(entries),
            window_unsettled=sum(1 for e in entries if not e.get("settled")),
            available=self.policy.tokens_per_window - used,
            utilization=used / self.policy.tokens_per_window,
            waiters=len(waiters),
            waiting_charge=sum(v["charge"] for v in waiters),
            cooldown_remaining=max(0.0, state["cooldown_until"] - now),
            sequence=state["sequence"],
            totals=state["totals"],
        )


def read_ledger(directory) -> list:
    """Parse the ledger. A torn final line (crash mid-append) is reported, not hidden."""
    path = Path(directory) / "ledger.jsonl"
    events = []
    lines = path.read_text().split("\n")
    for number, line in enumerate(lines, 1):
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except ValueError as exc:
            raise LedgerCorrupted(f"ledger.jsonl line {number} is unreadable: {exc}") from exc
    return events


def verify_ledger(directory, policy: "BudgetPolicy | None" = None) -> dict:
    """Replay the ledger and check the invariants the experiment relies on.

    Returns a report with ``ok`` and a list of ``problems``; never raises for an
    invariant violation so a failed run still yields diagnostics.
    """
    directory = Path(directory)
    problems = []
    try:
        state = json.loads((directory / "state.json").read_text())
        events = read_ledger(directory)
        if not isinstance(state, dict) or not isinstance(state.get("totals"), dict):
            raise ValueError("state.json is malformed")
        if not isinstance(state.get("sequence"), int):
            raise ValueError("state.json has no sequence counter")
        policy = policy or BudgetPolicy.from_dict(state["policy"])
    except (OSError, ValueError, TypeError, KeyError, LedgerCorrupted) as exc:
        return dict(ok=False, problems=[f"unreadable ledger: {exc}"], events=0)
    if state.get("policy_fingerprint") != policy.fingerprint():
        problems.append("policy fingerprint differs from the expected policy")
    sequences = [e.get("seq") for e in events]
    if sequences != list(range(1, len(events) + 1)):
        problems.append("ledger sequence numbers are not contiguous from 1")
    if state["sequence"] != len(events):
        problems.append("state sequence does not match the number of ledger events")
    times = [e["mono"] for e in events]
    if any(b < a for a, b in zip(times, times[1:])):
        problems.append("ledger timestamps go backwards")
    admits, settled = {}, {}
    window = []  # [time, charge, id]
    peak_used, peak_requests = 0, 0
    totals = dict(admitted=0, admitted_charge=0, reconciled=0, failed=0, rate_limited=0)
    cooldown_until, last_admit = 0.0, None
    for event in events:
        kind = event.get("kind")
        if kind == "admit":
            now, request_id = event["admitted_mono"], event["request_id"]
            if request_id in admits:
                problems.append(f"request {request_id} admitted twice")
            admits[request_id] = event
            window = [w for w in window if w[0] + policy.window_seconds > now]
            used = sum(w[1] for w in window)
            if used != event["window_used_before"]:
                problems.append(
                    f"request {request_id}: recorded window usage disagrees with replay"
                )
            if used + event["charge"] > policy.tokens_per_window:
                problems.append(f"request {request_id} admitted over the token budget")
            if (
                policy.requests_per_window is not None
                and len(window) + 1 > policy.requests_per_window
            ):
                problems.append(f"request {request_id} admitted over the request budget")
            if now < cooldown_until - 1e-9:
                problems.append(f"request {request_id} admitted during a 429 cooldown")
            if (
                last_admit is not None
                and now - last_admit < policy.minimum_request_spacing_seconds - 1e-9
            ):
                problems.append(f"request {request_id} violates minimum request spacing")
            if event["charge"] != (
                event["estimated_prompt_tokens"] + event["reserved_completion_tokens"]
            ):
                problems.append(f"request {request_id}: charge is not prompt + completion")
            if event["wait_seconds"] < 0 or event["admitted_mono"] < event["ready_mono"]:
                problems.append(f"request {request_id}: negative admission wait")
            last_admit = now
            window.append([now, event["charge"], request_id])
            peak_used = max(peak_used, used + event["charge"])
            peak_requests = max(peak_requests, len(window))
            totals["admitted"] += 1
            totals["admitted_charge"] += event["charge"]
        elif kind in ("reconcile", "fail"):
            request_id = event["request_id"]
            if request_id not in admits:
                problems.append(f"{kind} for unknown request {request_id}")
            if request_id in settled:
                problems.append(f"request {request_id} settled twice")
            settled[request_id] = event
            for item in window:
                if item[2] == request_id and event.get("final_charge") is not None:
                    item[1] = event["final_charge"]
            totals["reconciled" if kind == "reconcile" else "failed"] += 1
        elif kind == "rate_limited":
            cooldown_until = max(cooldown_until, event["cooldown_until"])
            totals["rate_limited"] += 1
        elif kind != "wait_censored":
            problems.append(f"unknown ledger event kind {kind!r}")
    for key, value in totals.items():
        if state["totals"].get(key) != value:
            problems.append(
                f"state total {key!r}={state['totals'].get(key)} but ledger has {value}"
            )
    return dict(
        ok=not problems,
        problems=problems,
        events=len(events),
        admitted=totals["admitted"],
        settled=len(settled),
        unsettled=sorted(set(admits) - set(settled)),
        peak_window_tokens=peak_used,
        peak_window_requests=peak_requests,
        budget=policy.tokens_per_window,
        totals=state["totals"],
    )


def waiting_statistics(events: list) -> dict:
    """Queueing statistics over admitted requests (seconds)."""
    waits = sorted(e["wait_seconds"] for e in events if e.get("kind") == "admit")
    if not waits:
        return dict(n=0)

    def quantile(q):
        return waits[min(len(waits) - 1, max(0, math.ceil(q * len(waits)) - 1))]

    return dict(
        n=len(waits),
        mean=sum(waits) / len(waits),
        p50=quantile(0.5),
        p90=quantile(0.9),
        p99=quantile(0.99),
        maximum=waits[-1],
        waited=sum(w > 0.001 for w in waits),
        total=sum(waits),
    )
