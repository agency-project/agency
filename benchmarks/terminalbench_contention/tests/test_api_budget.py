"""Shared admission budget: accounting, windows, concurrency, failure handling."""

import json
import multiprocessing
import threading
import time
from dataclasses import replace

import pytest

from benchmarks.terminalbench_contention import api_budget
from benchmarks.terminalbench_contention.api_budget import (
    AdmissionCancelled,
    BudgetPolicy,
    DeadlineReached,
    LedgerCorrupted,
    RequestTooLarge,
    SharedBudget,
    duration,
    read_ledger,
    verify_ledger,
    waiting_statistics,
)
from benchmarks.terminalbench_contention.tokens import EstimatorConfig, TokenEstimator

EXACT = EstimatorConfig(safety_factor=1.0, safety_tokens=0)


def body(words: int = 100, max_completion_tokens: int = 100) -> bytes:
    return json.dumps(
        dict(
            model="m",
            max_completion_tokens=max_completion_tokens,
            messages=[dict(role="user", content="alpha " * words)],
        )
    ).encode()


def policy(**changes) -> BudgetPolicy:
    defaults = dict(tokens_per_window=1000, minimum_request_spacing_seconds=0.0, estimator=EXACT)
    return BudgetPolicy(**{**defaults, **changes})


def new_budget(tmp_path, clock, **changes):
    return SharedBudget.create(
        tmp_path / "ledger", policy(**changes), run_id="t", clock=clock.read, sleep=clock.sleep
    )


def client(tmp_path, clock, budget, **kwargs):
    return SharedBudget(
        tmp_path / "ledger", budget.policy, clock=clock.read, sleep=clock.sleep, **kwargs
    )


# ------------------------------------------------------- reservation accounting


def test_charge_is_prompt_estimate_plus_declared_completion_cap(tmp_path, clock):
    budget = new_budget(tmp_path, clock)
    admission = budget.admit(body(100, 250), request_id="r1", agent_id="A01")
    assert admission.reserved_completion_tokens == 250
    assert admission.charge == admission.estimated_prompt_tokens + 250
    assert admission.estimated_prompt_tokens == admission.raw_prompt_tokens > 100
    assert admission.available_before == 1000
    assert admission.available_after == 1000 - admission.charge
    assert admission.wait_seconds == 0 and admission.blocked_by == []
    # The old rule is recorded beside it for comparison only.
    assert admission.legacy_charge > admission.charge


def test_completion_reservation_can_be_disabled_and_defaults_when_absent(tmp_path, clock):
    none = SharedBudget.create(
        tmp_path / "a",
        policy(completion_reservation="none"),
        run_id="t",
        clock=clock.read,
        sleep=clock.sleep,
    )
    assert none.admit(body(50, 400)).reserved_completion_tokens == 0
    default = SharedBudget.create(
        tmp_path / "b",
        policy(default_completion_tokens=321),
        run_id="t",
        clock=clock.read,
        sleep=clock.sleep,
    )
    payload = json.dumps(dict(messages=[dict(role="user", content="hi")])).encode()
    assert default.admit(payload).reserved_completion_tokens == 321


def test_request_larger_than_the_window_is_rejected(tmp_path, clock):
    budget = new_budget(tmp_path, clock, tokens_per_window=50)
    with pytest.raises(RequestTooLarge):
        budget.admit(body(500))
    assert read_ledger(tmp_path / "ledger") == []


# -------------------------------------------------------------- rolling window


def test_window_blocks_then_admits_exactly_when_enough_charge_expires(tmp_path, clock):
    charge = TokenEstimator(EXACT).estimate_bytes(body(100, 100)).estimated_tokens + 100
    limit = 2 * charge + 10  # room for two requests, not three
    budget = new_budget(tmp_path, clock, tokens_per_window=limit, poll_interval_seconds=0.25)
    first = budget.admit(body(100, 100), request_id="a")
    clock.now += 10
    second = budget.admit(body(100, 100), request_id="b")
    assert first.charge == second.charge == charge
    third = budget.admit(body(100, 100), request_id="c")
    # The third request fits only once the first entry (admitted at t=1000) leaves.
    assert third.admitted_mono == pytest.approx(1060.0, abs=0.26)
    assert third.wait_seconds == pytest.approx(50.0, abs=0.26)
    assert third.blocked_by == ["token_budget"]
    assert third.window_used_before == second.charge
    report = verify_ledger(tmp_path / "ledger")
    assert report["ok"], report["problems"]
    assert report["peak_window_tokens"] <= limit


def test_entries_expire_after_the_window(tmp_path, clock):
    budget = new_budget(tmp_path, clock)
    budget.admit(body(100), request_id="a")
    assert budget.snapshot()["window_requests"] == 1
    clock.now += 60.01
    snapshot = budget.snapshot()
    assert snapshot["window_requests"] == 0 and snapshot["window_used"] == 0
    assert snapshot["available"] == 1000 and snapshot["utilization"] == 0


def test_minimum_spacing_and_request_rate_limit(tmp_path, clock):
    spaced = SharedBudget.create(
        tmp_path / "s",
        policy(tokens_per_window=10**6, minimum_request_spacing_seconds=0.5),
        run_id="t",
        clock=clock.read,
        sleep=clock.sleep,
    )
    first = spaced.admit(body(5))
    second = spaced.admit(body(5))
    assert second.admitted_mono - first.admitted_mono >= 0.5
    assert second.blocked_by == ["request_spacing"]
    limited = SharedBudget.create(
        tmp_path / "r",
        policy(tokens_per_window=10**6, requests_per_window=2),
        run_id="t",
        clock=clock.read,
        sleep=clock.sleep,
    )
    start = clock.now
    limited.admit(body(5))
    limited.admit(body(5))
    third = limited.admit(body(5))
    assert third.admitted_mono - start >= 60 and third.blocked_by == ["request_budget"]
    assert verify_ledger(tmp_path / "r")["ok"]


# --------------------------------------------------------------- reconciliation


def test_conservative_reconciliation_never_refunds_but_raises_on_underestimate(tmp_path, clock):
    budget = new_budget(tmp_path, clock, tokens_per_window=100000)
    low = budget.admit(body(100, 500), request_id="low")
    record = budget.reconcile("low", prompt_tokens=50, completion_tokens=10)
    assert record["final_charge"] == low.charge and record["adjustment"] == 0
    high = budget.admit(body(100, 500), request_id="high")
    record = budget.reconcile("high", prompt_tokens=high.charge + 400, completion_tokens=100)
    assert record["final_charge"] == high.charge + 500 and record["adjustment"] == 500
    assert budget.snapshot()["window_used"] == low.charge + high.charge + 500
    totals = budget.snapshot()["totals"]
    # Estimates and reported usage are separate numbers.
    assert totals["estimated_prompt_tokens"] == (
        low.estimated_prompt_tokens + high.estimated_prompt_tokens
    )
    assert totals["reported_prompt_tokens"] == 50 + high.charge + 400
    assert totals["reported_completion_tokens"] == 110
    assert totals["reserved_completion_tokens"] == 1000


def test_reported_total_reconciliation_replaces_the_reservation(tmp_path, clock):
    budget = new_budget(tmp_path, clock, reconciliation="reported_total")
    admission = budget.admit(body(100, 400), request_id="r")
    record = budget.reconcile("r", prompt_tokens=120, completion_tokens=30, cached_tokens=100)
    assert record["final_charge"] == 150
    assert record["adjustment"] == 150 - admission.charge
    assert budget.snapshot()["window_used"] == 150


def test_litellm_reconciliation_excludes_cached_tokens_and_refunds_failures(tmp_path, clock):
    budget = new_budget(tmp_path, clock, reconciliation="litellm_v3")
    budget.admit(body(100, 400), request_id="ok")
    assert (
        budget.reconcile("ok", prompt_tokens=120, completion_tokens=30, cached_tokens=100)[
            "final_charge"
        ]
        == 50
    )
    budget.admit(body(100, 400), request_id="bad")
    failed = budget.fail("bad", reason="http_500", status_code=500)
    assert failed["released"] and failed["final_charge"] == 0
    assert budget.snapshot()["window_used"] == 50


def test_reconciliation_off_leaves_the_charge_untouched(tmp_path, clock):
    budget = new_budget(tmp_path, clock, reconciliation="off")
    admission = budget.admit(body(100, 400), request_id="r")
    assert (
        budget.reconcile("r", prompt_tokens=5000, completion_tokens=5000)["final_charge"]
        == admission.charge
    )


def test_reconciling_after_the_window_expired_is_recorded_without_effect(tmp_path, clock):
    budget = new_budget(tmp_path, clock, reconciliation="reported_total")
    budget.admit(body(100), request_id="slow")
    clock.now += 120
    budget.admit(body(10), request_id="next")  # prunes the expired entry
    record = budget.reconcile("slow", prompt_tokens=99, completion_tokens=1)
    assert record["entry_expired"] and record["final_charge"] is None
    assert verify_ledger(tmp_path / "ledger")["ok"]


def test_double_settlement_is_refused(tmp_path, clock):
    budget = new_budget(tmp_path, clock)
    budget.admit(body(10), request_id="r")
    budget.reconcile("r", prompt_tokens=10, completion_tokens=1)
    with pytest.raises(LedgerCorrupted):
        budget.reconcile("r", prompt_tokens=10, completion_tokens=1)
    with pytest.raises(LedgerCorrupted):
        budget.fail("r", reason="late")


# ------------------------------------------------------- failures, 429, cancel


def test_failed_request_keeps_its_charge_by_default(tmp_path, clock):
    budget = new_budget(tmp_path, clock)
    admission = budget.admit(body(100), request_id="r")
    record = budget.fail("r", reason="http_500", status_code=500)
    assert not record["released"] and record["final_charge"] == admission.charge
    assert budget.snapshot()["window_used"] == admission.charge
    assert budget.snapshot()["totals"]["failed"] == 1


def test_release_on_failure_is_configurable(tmp_path, clock):
    budget = new_budget(tmp_path, clock, release_on_failure=True)
    budget.admit(body(100), request_id="r")
    assert budget.fail("r", reason="cancelled")["released"]
    assert budget.snapshot()["window_used"] == 0
    assert budget.snapshot()["totals"]["released_charge"] > 0


def test_429_cooldown_pauses_every_client(tmp_path, clock):
    budget = new_budget(tmp_path, clock, rate_limit_cooldown_seconds=30)
    budget.observe_rate_limit(retry_after="45", request_id="x", agent_id="A01")
    other = client(tmp_path, clock, budget)
    admission = other.admit(body(10), agent_id="A02")
    assert admission.wait_seconds == pytest.approx(45, abs=0.3)
    assert admission.blocked_by == ["rate_limit_cooldown"]
    events = read_ledger(tmp_path / "ledger")
    assert [e["kind"] for e in events] == ["rate_limited", "admit"]
    assert verify_ledger(tmp_path / "ledger")["ok"]


def test_429_uses_the_longest_of_cooldown_and_provider_headers(tmp_path, clock):
    budget = new_budget(tmp_path, clock, rate_limit_cooldown_seconds=5)
    record = budget.observe_rate_limit(
        retry_after="2", reset_tokens="1m12.5s", reset_requests="12ms"
    )
    assert record["cooldown_seconds"] == 72.5
    assert duration("1m12.5s") == 72.5 and duration("12ms") == 0.012
    assert duration("0") == 0 and duration("garbage") is None and duration(None) is None


def test_deadline_stops_a_waiting_request_and_is_recorded(tmp_path, clock):
    budget = new_budget(tmp_path, clock, tokens_per_window=300)
    budget.admit(body(100, 100), request_id="holder")
    waiter = client(tmp_path, clock, budget, deadline=lambda: clock.now_start + 5)
    clock.now_start = clock.now
    with pytest.raises(DeadlineReached):
        waiter.admit(body(100, 100), request_id="late", agent_id="A02")
    events = read_ledger(tmp_path / "ledger")
    assert events[-1]["kind"] == "wait_censored" and events[-1]["reason"] == "DeadlineReached"
    assert events[-1]["wait_seconds"] == pytest.approx(5, abs=0.3)
    snapshot = budget.snapshot()
    assert snapshot["waiters"] == 0 and snapshot["totals"]["wait_censored"] == 1
    assert snapshot["totals"]["admitted"] == 1  # the censored request was never charged


def test_cancellation_stops_a_waiting_request(tmp_path, clock):
    budget = new_budget(tmp_path, clock, tokens_per_window=300)
    budget.admit(body(100, 100), request_id="holder")
    polls = []

    def cancelled():
        polls.append(clock.now)
        return len(polls) > 3

    waiter = client(tmp_path, clock, budget, cancelled=cancelled)
    with pytest.raises(AdmissionCancelled):
        waiter.admit(body(100, 100), request_id="cancelled")
    assert read_ledger(tmp_path / "ledger")[-1]["reason"] == "AdmissionCancelled"
    assert verify_ledger(tmp_path / "ledger")["ok"]


# -------------------------------------------------------- queueing statistics


def test_waiters_and_overtaking_are_visible_but_never_used_for_ordering(tmp_path, clock):
    budget = new_budget(tmp_path, clock, tokens_per_window=1000)
    big, small = body(250, 200), body(20, 20)
    budget.admit(big, request_id="one")  # ~453 of 1000
    budget.admit(big, request_id="two")  # ~906 of 1000
    # A large request waits (and gives up after 3 s); it stays registered as a waiter
    # only while it is waiting.
    start = clock.now
    waiting = client(tmp_path, clock, budget, deadline=lambda: start + 3)
    with pytest.raises(DeadlineReached):
        waiting.admit(big, request_id="large", agent_id="A-large")
    # A small request from another agent arrives later and is admitted at once:
    # admission is opportunistic, with no queue order and no regard for the agent.
    quick = client(tmp_path, clock, budget).admit(small, request_id="small", agent_id="A-small")
    assert quick.wait_seconds == 0 and quick.charge < 100
    policy_fields = BudgetPolicy().to_dict()
    assert policy_fields["admission_order"] == "opportunistic"
    assert not any("priority" in key or "agent" in key for key in policy_fields)


def test_agent_identity_does_not_change_admission(tmp_path, clock):
    outcomes = []
    for index, agent in enumerate(("A01", "A24")):
        local = type(clock)(1000.0)
        budget = SharedBudget.create(
            tmp_path / f"l{index}",
            policy(tokens_per_window=700),
            run_id="t",
            clock=local.read,
            sleep=local.sleep,
        )
        outcomes.append(
            [round(budget.admit(body(100, 100), agent_id=agent).admitted_mono, 6) for _ in range(4)]
        )
    assert outcomes[0] == outcomes[1]


def test_waiting_statistics():
    events = [dict(kind="admit", wait_seconds=w) for w in (0, 0, 1, 3)] + [dict(kind="fail")]
    stats = waiting_statistics(events)
    assert stats["n"] == 4 and stats["maximum"] == 3 and stats["total"] == 4
    assert stats["waited"] == 2 and stats["mean"] == 1
    assert waiting_statistics([]) == dict(n=0)


# --------------------------------------------------------- concurrent admission


def test_threads_sharing_one_ledger_never_exceed_the_budget(tmp_path):
    shared = SharedBudget.create(
        tmp_path / "ledger",
        policy(tokens_per_window=1500, window_seconds=0.6, poll_interval_seconds=0.01),
        run_id="t",
    )
    errors = []

    def work(agent):
        try:
            mine = SharedBudget(tmp_path / "ledger", shared.policy)
            for number in range(4):
                request = f"{agent}-{number}"
                mine.admit(body(100, 100), request_id=request, agent_id=agent)
                mine.reconcile(request, prompt_tokens=100, completion_tokens=10, agent_id=agent)
        except Exception as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(f"A{i}",)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert not errors
    report = verify_ledger(tmp_path / "ledger")
    assert report["ok"], report["problems"]
    assert report["admitted"] == 32 and report["settled"] == 32 and not report["unsettled"]
    assert report["peak_window_tokens"] <= 1500
    assert (
        sum(e["wait_seconds"] > 0 for e in read_ledger(tmp_path / "ledger") if e["kind"] == "admit")
        > 0
    )


def _process_worker(directory, policy_fields, agent, count):
    budget = SharedBudget(directory, BudgetPolicy.from_dict(policy_fields))
    for number in range(count):
        request = f"{agent}-{number}"
        budget.admit(body(100, 100), request_id=request, agent_id=agent)
        budget.reconcile(request, prompt_tokens=100, completion_tokens=10, agent_id=agent)


def test_separate_processes_share_one_global_budget(tmp_path):
    shared = SharedBudget.create(
        tmp_path / "ledger",
        policy(tokens_per_window=1200, window_seconds=0.5, poll_interval_seconds=0.01),
        run_id="t",
    )
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_process_worker,
            args=(str(tmp_path / "ledger"), shared.policy.to_dict(), f"A{i}", 3),
        )
        for i in range(6)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(90)
        assert process.exitcode == 0
    report = verify_ledger(tmp_path / "ledger")
    assert report["ok"], report["problems"]
    assert report["admitted"] == 18 and report["peak_window_tokens"] <= 1200
    events = read_ledger(tmp_path / "ledger")
    assert len({e["pid"] for e in events}) == 6
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))


# ----------------------------------------------------------- ledger integrity


def test_a_new_ledger_is_required_for_every_run(tmp_path, clock):
    new_budget(tmp_path, clock)
    with pytest.raises(FileExistsError):
        new_budget(tmp_path, clock)


def test_clients_must_share_one_policy(tmp_path, clock):
    budget = new_budget(tmp_path, clock)
    other = SharedBudget(
        tmp_path / "ledger",
        replace(budget.policy, tokens_per_window=2000),
        clock=clock.read,
        sleep=clock.sleep,
    )
    with pytest.raises(LedgerCorrupted, match="share one admission policy"):
        other.admit(body(10))


@pytest.mark.parametrize("damage", ["garbage", "truncate", "delete", "schema"])
def test_corrupted_state_fails_closed_and_is_preserved(tmp_path, clock, damage):
    budget = new_budget(tmp_path, clock)
    budget.admit(body(10), request_id="before")
    state = tmp_path / "ledger" / "state.json"
    if damage == "garbage":
        state.write_text("{not json")
    elif damage == "truncate":
        state.write_text(state.read_text()[:25])
    elif damage == "delete":
        state.unlink()
    else:
        state.write_text(json.dumps(dict(version=2, entries="oops")))
    with pytest.raises(LedgerCorrupted):
        budget.admit(body(10), request_id="after")
    with pytest.raises(LedgerCorrupted):
        budget.snapshot()
    # Nothing was admitted after the damage, and the evidence is kept.
    assert [e["request_id"] for e in read_ledger(tmp_path / "ledger")] == ["before"]
    if damage != "delete":
        assert list((tmp_path / "ledger").glob("state.corrupt-*.json"))
    assert not verify_ledger(tmp_path / "ledger")["ok"]


def test_verifier_detects_tampering_and_torn_writes(tmp_path, clock):
    budget = new_budget(tmp_path, clock, tokens_per_window=100000)
    for number in range(3):
        budget.admit(body(100), request_id=f"r{number}")
        budget.reconcile(f"r{number}", prompt_tokens=100, completion_tokens=1)
    ledger = tmp_path / "ledger" / "ledger.jsonl"
    assert verify_ledger(tmp_path / "ledger")["ok"]
    original = ledger.read_text()
    lines = original.splitlines()
    # A removed event breaks sequence continuity and the totals.
    ledger.write_text("\n".join(lines[:2] + lines[3:]) + "\n")
    assert not verify_ledger(tmp_path / "ledger")["ok"]
    # An inflated charge disagrees with the replayed window.
    event = json.loads(lines[0])
    event["charge"] += 5000
    ledger.write_text("\n".join([json.dumps(event)] + lines[1:]) + "\n")
    problems = verify_ledger(tmp_path / "ledger")["problems"]
    assert any("prompt + completion" in p or "disagrees" in p for p in problems)
    # A torn final line (crash mid-append) is reported, never silently dropped.
    ledger.write_text(original + '{"seq": 7, "kind": "adm')
    report = verify_ledger(tmp_path / "ledger")
    assert not report["ok"] and "unreadable" in report["problems"][0]
    with pytest.raises(LedgerCorrupted):
        read_ledger(tmp_path / "ledger")


def test_verifier_flags_over_budget_admission(tmp_path, clock):
    budget = new_budget(tmp_path, clock, tokens_per_window=100000)
    budget.admit(body(100), request_id="a")
    budget.admit(body(100), request_id="b")
    tight = replace(budget.policy, tokens_per_window=250)
    problems = verify_ledger(tmp_path / "ledger", tight)["problems"]
    assert any("over the token budget" in p for p in problems)


def test_unsettled_requests_are_listed(tmp_path, clock):
    budget = new_budget(tmp_path, clock)
    budget.admit(body(10), request_id="open")
    report = verify_ledger(tmp_path / "ledger")
    assert report["ok"] and report["unsettled"] == ["open"]


# -------------------------------------------------------- policy configuration


def test_policy_round_trips_and_defaults_preserve_the_historical_budget():
    default = BudgetPolicy()
    assert default.tokens_per_window == 1_200_000 and default.window_seconds == 60
    assert default.requests_per_window is None  # no invented provider limit
    assert default.reconciliation == "conservative" and not default.refunds_failures
    assert default.completion_reservation == "max_completion_tokens"
    assert default.minimum_request_spacing_seconds == 0.05
    assert default.rate_limit_cooldown_seconds == 60
    restored = BudgetPolicy.from_dict(json.loads(json.dumps(default.to_dict())))
    assert restored == default and restored.fingerprint() == default.fingerprint()
    assert replace(default, tokens_per_window=2_000_000).fingerprint() != default.fingerprint()


@pytest.mark.parametrize(
    "fields",
    [
        dict(reconciliation="magic"),
        dict(completion_reservation="half"),
        dict(tokens_per_window=0),
        dict(window_seconds=-1),
        dict(requests_per_window=0),
        dict(admission_order="fifo"),
        dict(poll_interval_seconds=0),
        dict(unknown_field=1),
        dict(estimator=dict(nope=1)),
        dict(estimator=dict(safety_factor=0.5)),
    ],
)
def test_policy_rejects_invalid_or_unknown_settings(fields):
    with pytest.raises((ValueError, TypeError)):
        BudgetPolicy.from_dict({**BudgetPolicy().to_dict(), **fields})


def test_timestamps_are_monotonic_and_carry_a_wall_clock_anchor(tmp_path):
    budget = SharedBudget.create(tmp_path / "ledger", policy(), run_id="t")
    before = time.monotonic()
    admission = budget.admit(body(10))
    assert before <= admission.ready_mono <= admission.admitted_mono <= time.monotonic()
    state = json.loads((tmp_path / "ledger" / "state.json").read_text())
    wall = (time.monotonic_ns() + state["epoch_offset_ns"]) / 1e9
    assert abs(wall - time.time()) < 1.0
    assert api_budget.STATE_VERSION == state["version"]
