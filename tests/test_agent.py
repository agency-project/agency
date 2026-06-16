import json
import threading
import pytest
from unittest.mock import MagicMock, patch
from agency.agdata import agdata
from agency.agskill import agskill
from agency.agtool import agtool
from agency.agent import agent

# ---------------------------------------------------------------------------
# Streaming mock helpers (agskill uses stream=True)
# ---------------------------------------------------------------------------

class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.model_extra = {}
        self.reasoning_content = None

class _Choice:
    def __init__(self, delta): self.delta = delta

class _Usage:
    prompt_tokens = 5

class _Chunk:
    def __init__(self, content=None, tool_calls=None, usage=None):
        self.usage = usage
        self.choices = [_Choice(_Delta(content, tool_calls))] if (content is not None or tool_calls) else []

class _TCDelta:
    def __init__(self, name, args_json, call_id):
        self.id = call_id; self.index = 0
        self.function = _TCFnDelta(name, args_json)

class _TCFnDelta:
    def __init__(self, name, args): self.name = name; self.arguments = args


def _noop(arg: agdata) -> agdata:
    return agdata()


def _direct(content: str) -> list:
    return [_Chunk(content=content), _Chunk(usage=_Usage())]


def _tool_resp(name: str, args: dict, call_id: str = "c1") -> list:
    tc = _TCDelta(name, json.dumps(args), call_id)
    return [_Chunk(tool_calls=[tc]), _Chunk(usage=_Usage())]


def make_agent() -> agent:
    return agent(llm_config={"api_key": "k", "model": "gpt-4o"})


# ---------------------------------------------------------------------------
# Basic routing
# ---------------------------------------------------------------------------

def test_run_returns_pending_agdata():
    """run() is non-blocking — result fields resolve lazily."""
    skill = agskill(name="s", system_prompt="")
    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        return agdata(done=True), agdata(messages=[]), [], (0, 0)
    skill.run = fake_run

    ag = make_agent()
    result = ag.run(skill, agdata())
    assert isinstance(result, agdata)
    assert result.done is True   # field access blocks until task finishes


def test_run_calls_named_agskill():
    called = []
    def fake_run(llm_cfg, inp, hist, sandbox, pool, max_steps, **_):
        called.append(inp.to_dict())
        return agdata(done=True), agdata(messages=[]), [], (0, 0)

    skill = agskill(name="dowork", system_prompt="")
    skill.run = fake_run

    ag = make_agent()
    result = ag.run(skill, agdata(task="go"))
    assert result.done is True   # blocks until done
    assert called == [{"task": "go"}]


def test_repr():
    skill_a = agskill("a", "")
    skill_b = agskill("b", "")
    ag = make_agent()
    assert repr(ag) is not None  # agent repr works without owned skills


# ---------------------------------------------------------------------------
# History is updated and serialized on the same agent
# ---------------------------------------------------------------------------

def test_history_updated_after_run():
    skill = agskill(name="s", system_prompt="")
    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        existing = list(hist._data.get("messages", []))
        new_hist = agdata(messages=existing + [
            {"role": "user", "content": inp.to_json()},
            {"role": "assistant", "content": "{}"},
        ])
        return agdata(ok=True), new_hist, [], (0, 0)

    skill.run = fake_run
    ag = make_agent()

    ag.run(skill, agdata(turn=1))
    assert len(ag.history.messages) == 2   # ag.history blocks until done

    ag.run(skill, agdata(turn=2))
    assert len(ag.history.messages) == 4   # chain: step2 waited for step1


def test_sequential_calls_serialize_via_history_chain():
    """Two calls on the same agent must run in order even though both are non-blocking."""
    order = []
    lock = threading.Lock()

    def make_skill(name):
        sk = agskill(name, "")
        def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
            with lock:
                order.append(name)
            return agdata(name=name), agdata(messages=list(hist._data.get("messages", [])) + [
                {"role": "user", "content": name}
            ]), [], (0, 0)
        sk.run = fake_run
        return sk

    skill_first = make_skill("first")
    skill_second = make_skill("second")
    ag = make_agent()
    ag.run(skill_first, agdata())
    ag.run(skill_second, agdata())
    _ = ag.history   # wait for both to finish

    assert order == ["first", "second"]


def test_history_passed_to_agskill():
    skill = agskill("s", "")
    ag = make_agent()
    ag.history = agdata(messages=[{"role": "user", "content": "prior"}])

    received = {}
    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        received["hist"] = hist
        return agdata(), agdata(messages=[]), [], (0, 0)
    skill.run = fake_run

    ag.run(skill, agdata(x=1))
    _ = ag.history   # sync
    assert received["hist"].messages[0]["content"] == "prior"


# ---------------------------------------------------------------------------
# Tools live on skills, not agents
# ---------------------------------------------------------------------------

def test_skill_replace_tools_used_in_run():
    """replace_tools on the skill replaces the full tool list."""
    t = agtool(name="t1", description="", fn=_noop)
    skill = agskill("s", "", replace_tools=[t])
    captured = {}
    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        captured["replace_tools"] = skill.replace_tools
        return agdata(), agdata(messages=[]), [], (0, 0)
    skill.run = fake_run

    ag = make_agent()
    ag.run(skill, agdata())
    _ = ag.history
    assert captured["replace_tools"] == [t]


# ---------------------------------------------------------------------------
# End-to-end with mocked OpenAI
# ---------------------------------------------------------------------------

def test_end_to_end_direct_answer():
    skill = agskill(name="qa", system_prompt="Answer questions.")
    ag = make_agent()

    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct('{"answer": "Paris"}')
        result = ag.run(skill, agdata(question="Capital of France?"))
        assert result.answer == "Paris"   # resolve inside the patch context


def test_end_to_end_with_tool():
    def calc_fn(arg: agdata) -> agdata:
        return agdata(result=arg.a + arg.b)  # type: ignore[operator]

    calc = agtool(
        name="add",
        description="Add two numbers.",
        fn=calc_fn,
        params={"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]},
    )
    skill = agskill(name="math", system_prompt="You are a calculator.", add_tools=[calc])
    ag = make_agent()

    responses = [_tool_resp("add", {"a": 3, "b": 4}), _direct('{"result": 7}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result = ag.run(skill, agdata(task="add 3 and 4"))
        assert result.result == 7


def test_multiple_agskills_coexist():
    skill_a = agskill("a", "")
    skill_b = agskill("b", "")

    def fake_a(llm, inp, hist, sandbox, pool, ms, **_):
        return agdata(from_skill="a"), agdata(messages=[]), [], (0, 0)
    def fake_b(llm, inp, hist, sandbox, pool, ms, **_):
        return agdata(from_skill="b"), agdata(messages=[]), [], (0, 0)

    skill_a.run = fake_a
    skill_b.run = fake_b

    ag = make_agent()
    results = {}
    results["a"] = ag.run(skill_a, agdata())
    results["b"] = ag.run(skill_b, agdata())
    assert results["a"].from_skill == "a"
    assert results["b"].from_skill == "b"


# ---------------------------------------------------------------------------
# Copy constructor
# ---------------------------------------------------------------------------

def test_copy_constructor_inherits_config():
    ag = make_agent()

    copy_ag = agent(ag)
    assert copy_ag.llm_config == ag.llm_config


def test_copy_constructor_deep_copies_history():
    ag = make_agent()
    ag.history = agdata(messages=[{"role": "user", "content": "original"}])

    copy_ag = agent(ag)
    copy_ag.history.messages.append({"role": "assistant", "content": "new"})

    assert len(ag.history.messages) == 1   # parent unaffected


def test_copy_constructor_copies_history_and_config():
    """Copy constructor copies history and llm_config from the source agent."""
    ag = make_agent()
    ag.history = agdata(messages=[{"role": "user", "content": "prior"}])

    copy_ag = agent(ag)
    assert copy_ag.llm_config == ag.llm_config
    assert len(copy_ag.history.messages) == 1
    assert copy_ag.history.messages[0]["content"] == "prior"


def test_fork_is_alias_for_copy_constructor():
    ag = make_agent()
    ag.history = agdata(messages=[{"role": "user", "content": "x"}])
    forked = ag.fork()
    assert forked.llm_config == ag.llm_config
    assert len(forked.history.messages) == 1


def test_copy_constructor_waits_for_inflight_task():
    """agent(src) blocks until src's current task finishes before copying history."""
    skill = agskill("s", "")
    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        return agdata(v=inp.v), agdata(messages=[{"role": "user", "content": str(inp.v)}]), [] , (0, 0)
    skill.run = fake_run

    ag = make_agent()
    ag.run(skill, agdata(v=42))        # non-blocking, in-flight

    fork = agent(ag)                   # blocks until the run completes
    assert len(fork.history.messages) == 1
    assert fork.history.messages[0]["content"] == "42"


# ---------------------------------------------------------------------------
# Parallel execution via copy constructor
# ---------------------------------------------------------------------------

def test_fork_runs_do_not_update_parent_history():
    skill = agskill("s", "")
    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        return agdata(ok=True), agdata(messages=[{"role": "user", "content": "fork_msg"}]), [] , (0, 0)
    skill.run = fake_run

    ag = make_agent()
    ag.history = agdata(messages=[{"role": "user", "content": "original"}])

    r = agent(ag).run(skill, agdata())
    _ = r.ok   # wait for fork to finish

    assert len(ag.history.messages) == 1
    assert ag.history.messages[0]["content"] == "original"


def test_fork_runs_in_parallel():
    """Multiple forks reach the barrier together, proving concurrent execution."""
    barrier = threading.Barrier(3)
    skill = agskill("s", "")
    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        barrier.wait(timeout=5)
        return agdata(n=inp.n), agdata(messages=[]), [], (0, 0)
    skill.run = fake_run

    ag = make_agent()
    results = [agent(ag).run(skill, agdata(n=i)) for i in range(3)]
    assert sorted(r.n for r in results) == [0, 1, 2]


def test_fork_sees_parent_history_at_fork_time():
    seen = {}
    skill = agskill("s", "")
    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        seen["hist"] = list(hist.messages)
        return agdata(), agdata(messages=[]), [], (0, 0)
    skill.run = fake_run

    ag = make_agent()
    ag.history = agdata(messages=[{"role": "user", "content": "seed"}])

    r = agent(ag).run(skill, agdata())
    r._resolve()
    assert seen["hist"][0]["content"] == "seed"


# ---------------------------------------------------------------------------
# Pending agdata as input — auto-resolved before skill runs
# ---------------------------------------------------------------------------

def test_run_accepts_pending_agdata_as_input():
    from concurrent.futures import Future
    received = {}

    skill = agskill("s", "")
    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        received["inp"] = inp.to_dict()
        return agdata(ok=True), agdata(messages=[]), [], (0, 0)
    skill.run = fake_run

    ag = make_agent()

    f: Future[agdata] = Future()
    f.set_result(agdata(resolved=True, value=99))
    pending_input = agdata(_future=f)

    result = ag.run(skill, pending_input)
    _ = result.ok
    assert received["inp"] == {"resolved": True, "value": 99}


def test_run_resolves_list_of_pending_in_input():
    from concurrent.futures import Future
    received = {}

    skill = agskill("s", "")
    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        received["items"] = inp.items
        return agdata(ok=True), agdata(messages=[]), [], (0, 0)
    skill.run = fake_run

    ag = make_agent()

    futures = []
    for i in range(3):
        f: Future[agdata] = Future()
        f.set_result(agdata(text=f"item {i}"))
        futures.append(agdata(_future=f))

    result = ag.run(skill, agdata(items=futures))
    _ = result.ok
    assert [r.text for r in received["items"]] == ["item 0", "item 1", "item 2"]


def test_chained_run_output_as_next_input():
    """Output of one run() passed directly as input to the next — resolved automatically."""
    skill = agskill("s", "")
    received_inputs = []
    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        received_inputs.append(dict(inp._data))
        return agdata(done=True), agdata(messages=[]), [], (0, 0)
    skill.run = fake_run

    ag1 = make_agent()
    ag2 = make_agent()

    r1 = ag1.run(skill, agdata(x=5))  # pending agdata, resolves to agdata(done=True)
    r2 = ag2.run(skill, r1)           # r1 passed as input; resolved before ag2's skill runs
    r2._resolve()

    # ag2 received the resolved r1 as its input
    assert received_inputs[1] == {"done": True}


# ---------------------------------------------------------------------------
# Outer monitoring loop
# ---------------------------------------------------------------------------

def _make_skill_with_pid_side_effect(ag, calls, inject_pid_on_call=0, on_first_call=None):
    """Build a fake agskill.run that records each call's input.

    On the iteration numbered *inject_pid_on_call*, it plants a fake PID into
    the sandbox's _watched_pids dict to simulate a background process having
    been launched.  On subsequent calls it does not add any PIDs, so the loop
    can break cleanly.

    If *on_first_call* is provided it is invoked with the agent on the first
    call (idx == 0), when the sandbox is guaranteed to be live.
    """
    skill = agskill(name="s", system_prompt="")
    import time as _time

    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        idx = len(calls)
        calls.append(dict(inp._data))
        if idx == 0 and on_first_call is not None:
            on_first_call(ag)
        if idx == inject_pid_on_call:
            ag.sandbox._watched_pids[99999] = _time.monotonic()
        else:
            # Clear stale fake PID so the next pids_at_end snapshot is empty
            ag.sandbox._watched_pids.clear()
        return agdata(result="ok"), agdata(messages=[]), [], (0, 0)

    skill.run = fake_run
    return skill


def test_outer_loop_no_background_process_exits_immediately():
    """When no background processes are started, the skill resolves in one iteration."""
    calls = []
    skill = agskill(name="s", system_prompt="")

    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        calls.append(dict(inp._data))
        return agdata(result="ok"), agdata(messages=[]), [], (0, 0)

    skill.run = fake_run
    ag = make_agent()
    agent.poll_interval_s = 0

    ag.run(skill, agdata()).result  # block until done

    assert len(calls) == 1


def test_outer_loop_re_enters_with_completed_event_when_process_finishes():
    """When a background process finishes before ping_interval_s elapses, the
    agent gets a process_completed re-entry so it can act on the output."""
    calls = []
    ag = make_agent()
    skill = _make_skill_with_pid_side_effect(
        ag, calls, inject_pid_on_call=0,
        on_first_call=lambda a: setattr(a.sandbox, 'get_live_pids', lambda: set()),
    )

    agent.poll_interval_s = 0

    ag.run(skill, agdata()).result

    assert len(calls) == 2
    assert calls[1].get("_event") == "process_completed"


def test_outer_loop_re_enters_with_update_event_when_process_still_running():
    """When a process is still running after ping_interval_s elapses, the
    agent gets a process_update re-entry with a status summary."""
    import time as _time

    calls = []
    ag = make_agent()
    skill = agskill(name="s", system_prompt="")

    # Patch get_live_pids: process still alive on first check, gone on second
    check_count = [0]
    def fake_live_pids():
        check_count[0] += 1
        if check_count[0] == 1:
            return {99999}   # still running → triggers process_update path
        ag.sandbox._watched_pids.clear()
        return set()

    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        idx = len(calls)
        calls.append(dict(inp._data))
        if idx == 0:
            ag.sandbox._watched_pids[99999] = _time.monotonic()
            ag.sandbox.get_live_pids = fake_live_pids       # ← moved here
            ag.sandbox.pid_status_summary = lambda: "PID 99999 (running 0m 0s)"  # ← moved here
        else:
            ag.sandbox._watched_pids.clear()
        return agdata(result="ok"), agdata(messages=[]), [], (0, 0)

    skill.run = fake_run

    agent.poll_interval_s = 0
    agent.ping_interval_s = 0

    ag.run(skill, agdata()).result

    assert len(calls) == 2
    assert calls[1].get("_event") == "process_update"
    assert "still running" in calls[1].get("message", "")


def test_outer_loop_completed_event_precedes_update_event():
    """process_completed fires before process_update: if a process finishes
    it always gets a completion re-entry, never an update."""
    calls = []
    ag = make_agent()
    skill = _make_skill_with_pid_side_effect(
        ag, calls, inject_pid_on_call=0,
        on_first_call=lambda a: setattr(a.sandbox, 'get_live_pids', lambda: set()),
    )

    agent.poll_interval_s = 0

    ag.run(skill, agdata()).result

    events = [c.get("_event") for c in calls if "_event" in c]
    assert "process_completed" in events
    assert "process_update" not in events


def test_outer_loop_max_iters_cap():
    """The loop never runs more than max_outer_iters iterations."""
    import time as _time

    calls = []
    ag = make_agent()
    skill = agskill(name="s", system_prompt="")

    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        calls.append(1)
        ag.sandbox._watched_pids[99999] = _time.monotonic()
        ag.sandbox.get_live_pids = lambda: set()  # set here, before monitoring runs
        return agdata(result="ok"), agdata(messages=[]), [], (0, 0)

    skill.run = fake_run

    agent.poll_interval_s = 0
    agent.max_outer_iters = 3

    ag.run(skill, agdata()).result

    assert len(calls) <= 3


def test_outer_loop_long_job_multiple_update_cycles():
    """A single long-running job stays alive across multiple ping intervals.
    The agent gets a process_update on each cycle and a final process_completed
    once the job exits.

    Sequence:
      iter 0  initial call     → starts PID 1
      check 1 get_live_pids()  → {1}  alive  → process_update
      iter 1  process_update   → PID 1 still in _watched_pids
      check 2 get_live_pids()  → {1}  alive  → process_update
      iter 2  process_update   → PID 1 still in _watched_pids
      check 3 get_live_pids()  → {}   done   → process_completed
      iter 3  process_completed→ clears _watched_pids
      pids_at_end = {}  → break
    """
    import time as _time

    ag = make_agent()
    calls = []

    live_seq = [{1}, {1}, set()]
    live_idx = [0]

    def fake_live_pids():
        resp = live_seq[min(live_idx[0], len(live_seq) - 1)]
        live_idx[0] += 1
        if not resp:
            ag.sandbox._watched_pids.clear()
        return resp

    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        event = inp._data.get("_event")
        calls.append(event)
        if event is None:                       # initial call
            ag.sandbox._watched_pids[1] = _time.monotonic()
            ag.sandbox.get_live_pids = fake_live_pids       # ← moved here
            ag.sandbox.pid_status_summary = lambda: "PID 1 (running 0m 0s)"  # ← moved here
        elif event == "process_completed":
            ag.sandbox._watched_pids.clear()
        return agdata(result="ok"), agdata(messages=[]), [], (0, 0)

    skill = agskill(name="s", system_prompt="")
    skill.run = fake_run
    agent.poll_interval_s = 0
    agent.ping_interval_s = 0

    ag.run(skill, agdata()).result

    update_count = calls.count("process_update")
    assert update_count == 2
    assert calls[-1] == "process_completed"
    assert len(calls) == 4   # initial + 2 updates + 1 completed


def test_outer_loop_two_jobs_different_end_times():
    """Two background jobs started together; the fast one finishes first.
    The loop should fire process_update while the slow job is still alive,
    then process_completed once both are gone.

    Sequence:
      iter 0  initial call     → starts PID 1 (fast) and PID 2 (slow)
      check 1 get_live_pids()  → {2}  (1 already done)  → process_update
      iter 1  process_update   → no new PIDs
      check 2 get_live_pids()  → {}   (2 now done)       → process_completed
      iter 2  process_completed→ clears _watched_pids
      pids_at_end = {}  → break
    """
    import time as _time

    ag = make_agent()
    calls = []

    live_seq = [{2}, set()]
    live_idx = [0]

    def fake_live_pids():
        resp = live_seq[min(live_idx[0], len(live_seq) - 1)]
        live_idx[0] += 1
        if not resp:
            ag.sandbox._watched_pids.clear()
        return resp

    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        event = inp._data.get("_event")
        calls.append(event)
        if event is None:
            ag.sandbox._watched_pids[1] = _time.monotonic()   # fast job
            ag.sandbox._watched_pids[2] = _time.monotonic()   # slow job
            ag.sandbox.get_live_pids = fake_live_pids       # ← moved here
            ag.sandbox.pid_status_summary = lambda: "PID 2 (running 0m 0s)"  # ← moved here
        elif event == "process_completed":
            ag.sandbox._watched_pids.clear()
        return agdata(result="ok"), agdata(messages=[]), [], (0, 0)

    skill = agskill(name="s", system_prompt="")
    skill.run = fake_run
    agent.poll_interval_s = 0
    agent.ping_interval_s = 0

    ag.run(skill, agdata()).result

    assert calls.count("process_update") == 1
    assert calls[-1] == "process_completed"
    assert len(calls) == 3


def test_outer_loop_agent_starts_new_job_on_completed_reentry():
    """Agent reacts to process_completed by launching another background job.
    The loop must track the new PID and re-enter again when it finishes.

    Sequence:
      iter 0  initial call      → starts PID 1
      check 1 get_live_pids()   → {}  → process_completed
      iter 1  process_completed → starts PID 2 (new job)
      check 2 get_live_pids()   → {}  → process_completed
      iter 2  process_completed → no new PIDs, clears _watched_pids
      pids_at_end = {}  → break
    """
    import time as _time

    ag = make_agent()
    calls = []
    live_idx = [0]

    def fake_live_pids():
        live_idx[0] += 1
        ag.sandbox._watched_pids.clear()
        return set()   # every check: process already done

    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        event = inp._data.get("_event")
        calls.append(event)
        if event is None:
            ag.sandbox._watched_pids[1] = _time.monotonic()
            ag.sandbox.get_live_pids = fake_live_pids       # ← moved here
        elif event == "process_completed" and len(calls) == 2:
            # React to first completion by kicking off a second job
            ag.sandbox._watched_pids[2] = _time.monotonic()
        return agdata(result="ok"), agdata(messages=[]), [], (0, 0)

    skill = agskill(name="s", system_prompt="")
    skill.run = fake_run
    agent.poll_interval_s = 0

    ag.run(skill, agdata()).result

    assert calls.count("process_completed") == 2
    assert "process_update" not in calls
    assert len(calls) == 3


def test_outer_loop_jobs_added_at_different_points_with_different_latencies():
    """Agent dynamically adds jobs during re-entries; each has a different
    lifetime.  Tests that newly added PIDs are tracked across outer iterations
    and that the loop does not resolve until all work is truly done.

    Sequence:
      iter 0  initial call      → starts PID 1 (slow) and PID 2 (fast)
      check 1 get_live_pids()   → {1}  (2 done)  → process_update
      iter 1  process_update    → agent adds PID 3 (medium)
      check 2 get_live_pids()   → {3}  (1 done)  → process_update
      iter 2  process_update    → no new PIDs
      check 3 get_live_pids()   → {}   (3 done)  → process_completed
      iter 3  process_completed → clears _watched_pids
      pids_at_end = {}  → break
    """
    import time as _time

    ag = make_agent()
    calls = []

    live_seq = [{1}, {3}, set()]
    live_idx = [0]

    def fake_live_pids():
        resp = live_seq[min(live_idx[0], len(live_seq) - 1)]
        live_idx[0] += 1
        if not resp:
            ag.sandbox._watched_pids.clear()
        return resp

    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        event = inp._data.get("_event")
        calls.append(event)
        if event is None:                          # initial: start slow + fast
            ag.sandbox._watched_pids[1] = _time.monotonic()
            ag.sandbox._watched_pids[2] = _time.monotonic()
            ag.sandbox.get_live_pids = fake_live_pids       # ← moved here
            ag.sandbox.pid_status_summary = lambda: "PID ? (running 0m 0s)"  # ← moved here
        elif event == "process_update" and len(calls) == 2:
            # First update: fast job done, slow still running; add a medium job
            ag.sandbox._watched_pids[3] = _time.monotonic()
        elif event == "process_completed":
            ag.sandbox._watched_pids.clear()
        return agdata(result="ok"), agdata(messages=[]), [], (0, 0)

    skill = agskill(name="s", system_prompt="")
    skill.run = fake_run
    agent.poll_interval_s = 0
    agent.ping_interval_s = 0

    ag.run(skill, agdata()).result

    assert calls.count("process_update") == 2
    assert calls[-1] == "process_completed"
    assert len(calls) == 4   # initial + 2 updates + 1 completed


def test_outer_loop_process_completed_fires_when_new_batch_exits_quickly():
    """When the agent starts a new batch of PIDs during a re-entry and they
    finish before ping_interval_s, process_completed fires promptly.

    Sequence:
      iter 0  initial call     → starts PID 1; PID 1 still alive → process_update
      iter 1  process_update   → starts PID 2; PID 2 already done → process_completed
      iter 2  process_completed→ clears _watched_pids → break
    """
    import time as _time

    ag = make_agent()
    calls = []

    live_seq = [{1}, set()]
    live_idx = [0]

    def fake_live_pids():
        resp = live_seq[min(live_idx[0], len(live_seq) - 1)]
        live_idx[0] += 1
        if not resp:
            ag.sandbox._watched_pids.clear()
        return resp

    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        event = inp._data.get("_event")
        calls.append(event)
        if event is None:
            ag.sandbox._watched_pids[1] = _time.monotonic()
            ag.sandbox.get_live_pids = fake_live_pids       # ← moved here
            ag.sandbox.pid_status_summary = lambda: "PID ? (running 0m 0s)"  # ← moved here
        elif event == "process_update":
            ag.sandbox._watched_pids.clear()
            ag.sandbox._watched_pids[2] = _time.monotonic()
        elif event == "process_completed":
            ag.sandbox._watched_pids.clear()
        return agdata(result="ok"), agdata(messages=[]), [], (0, 0)

    skill = agskill(name="s", system_prompt="")
    skill.run = fake_run
    agent.poll_interval_s = 0
    agent.ping_interval_s = 0

    ag.run(skill, agdata()).result

    assert calls[1] == "process_update"
    assert calls[2] == "process_completed"
    assert len(calls) == 3


def test_outer_loop_daemon_release_unblocks_skill():
    """When the agent calls daemon_release during a process_update re-entry,
    _watched_pids becomes empty and the outer loop breaks immediately —
    the skill resolves without waiting for the daemon to exit.

    Sequence:
      iter 0  initial call    → starts PID 1 (daemon server)
              PID 1 still alive after poll window → process_update
      iter 1  process_update  → agent calls daemon_release(1)
                                _watched_pids now empty
              pids_at_end = {} → break
    """
    import time as _time

    ag = make_agent()
    calls = []
    sandbox_state = {}

    def fake_live_pids():
        # PID 1 is always alive — it's a daemon that never exits
        return {1} if ag.sandbox._watched_pids else set()

    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        event = inp._data.get("_event")
        calls.append(event)
        if event is None:
            # Initial call: start a daemon process
            ag.sandbox._watched_pids[1] = _time.monotonic()
            ag.sandbox.get_live_pids = fake_live_pids       # ← moved here
            ag.sandbox.pid_status_summary = lambda: "PID 1 (running 0m 5s)"  # ← moved here
        elif event == "process_update":
            # Agent recognises PID 1 as a daemon and releases it
            ag.sandbox.release_daemon(1)
            sandbox_state['daemon_pids'] = set(ag.sandbox._daemon_pids)
            sandbox_state['watched_pids'] = dict(ag.sandbox._watched_pids)
        return agdata(result="ok"), agdata(messages=[]), [], (0, 0)

    skill = agskill(name="s", system_prompt="")
    skill.run = fake_run
    agent.poll_interval_s = 0
    agent.ping_interval_s = 0

    ag.run(skill, agdata()).result

    assert calls == [None, "process_update"]   # no process_completed — daemon was released
    assert 1 in sandbox_state['daemon_pids']   # PID moved to daemon set
    assert sandbox_state['watched_pids'] == {} # nothing left to monitor


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

def test_agent_all_tracks_live_agents():
    ag1 = make_agent()
    ag2 = make_agent()
    live = agent.all()
    assert ag1 in live
    assert ag2 in live


def test_agent_all_excludes_destroyed():
    import gc
    ag1 = make_agent()
    ag2 = make_agent()
    ag2_name = ag2.agname
    del ag2
    gc.collect()
    assert ag2_name not in [a.agname for a in agent.all()]


# ---------------------------------------------------------------------------
# Checkpointing (requires Docker)
# ---------------------------------------------------------------------------

import subprocess as _subprocess

_docker_ok = pytest.mark.skipif(
    not (lambda: __import__("subprocess").run(
        ["docker", "info"], capture_output=True, timeout=10
    ).returncode == 0)(),
    reason="Docker not available",
)


@_docker_ok
def test_save_and_load_restores_history_and_filesystem(tmp_path):
    from agency.agent import _allocated_agnames

    file_content = [None]

    skill_write = agskill(name="write", system_prompt="")
    def fake_write(cfg, inp, hist, sandbox, pool, ms, **_):
        sandbox.write_file("/workspace/state.txt", "hello\n")
        return agdata(answer="42"), agdata(messages=[{"role": "assistant", "content": "42"}]), [] , (0, 0)
    skill_write.run = fake_write

    skill_read = agskill(name="read", system_prompt="")
    def fake_read(cfg, inp, hist, sandbox, pool, ms, **_):
        file_content[0] = sandbox.read_file("/workspace/state.txt")
        return agdata(ok=True), agdata(messages=[]), [], (0, 0)
    skill_read.run = fake_read

    ag = agent(llm_config={"api_key": "k", "model": "m"})
    ag.run(skill_write, agdata(q="test")).answer

    ckpt = tmp_path / "agent.ckpt"
    ag.save(ckpt)
    assert ckpt.exists()
    saved_agname = ag.agname
    del ag
    _allocated_agnames.discard(saved_agname)

    ag2 = agent.load(ckpt, llm_config={"api_key": "k", "model": "m"})
    assert ag2.agname == saved_agname
    assert len(ag2._history._data.get("messages", [])) > 0
    assert ag2 in agent.all()

    ag2.run(skill_read, agdata()).ok
    assert file_content[0] == "hello\n"

    events = ag2.log.events
    assert any(e.get("event") == "loaded" for e in events)
    del ag2
    _allocated_agnames.discard(saved_agname)


@_docker_ok
def test_save_all_and_load_all(tmp_path):
    import gc
    from agency.agent import _allocated_agnames

    saved_names = set()
    file_content = {}

    skill_write = agskill(name="write", system_prompt="")
    def fake_write(cfg, inp, hist, sandbox, pool, ms, **_):
        sandbox.write_file("/workspace/id.txt", f"{inp.agname}\n")
        return agdata(ok=True), agdata(messages=[]), [], (0, 0)
    skill_write.run = fake_write

    skill_read = agskill(name="read", system_prompt="")
    def fake_read(cfg, inp, hist, sandbox, pool, ms, **_):
        file_content[inp.name] = sandbox.read_file("/workspace/id.txt").strip()
        return agdata(ok=True), agdata(messages=[]), [], (0, 0)
    skill_read.run = fake_read

    def _create_and_save():
        ag1 = agent(llm_config={"api_key": "k", "model": "m"})
        ag2 = agent(llm_config={"api_key": "k", "model": "m"})
        ag1.run(skill_write, agdata(agname=ag1.agname)).ok
        ag2.run(skill_write, agdata(agname=ag2.agname)).ok
        saved_names.update([ag1.agname, ag2.agname])
        agent.save_all(tmp_path)

    _create_and_save()
    gc.collect()
    _allocated_agnames.difference_update(saved_names)

    restored = agent.load_all(tmp_path, llm_config={"api_key": "k", "model": "m"})
    assert len(restored) == 2
    names = {a.agname for a in restored}
    assert names == saved_names
    for a in restored:
        a.run(skill_read, agdata(name=a.agname)).ok
    for a in restored:
        assert file_content[a.agname] == a.agname
    _allocated_agnames.difference_update(saved_names)


@_docker_ok
def test_load_all_skips_already_live_agent(tmp_path):
    import gc
    from agency.agent import _allocated_agnames

    skill = agskill(name="s", system_prompt="")
    def fake_run(cfg, inp, hist, sb, pool, ms, **_):
        return agdata(ok=True), agdata(messages=[]), [], (0, 0)
    skill.run = fake_run

    ag1 = agent(llm_config={"api_key": "k", "model": "m"})
    ag1.run(skill, agdata()).ok  # needs a checkpoint for save
    ag1_name = ag1.agname
    ag2_name = [None]

    def _create_save_ag2():
        ag2 = agent(llm_config={"api_key": "k", "model": "m"})
        ag2.run(skill, agdata()).ok
        ag2_name[0] = ag2.agname
        agent.save_all(tmp_path)

    _create_save_ag2()
    gc.collect()
    _allocated_agnames.discard(ag2_name[0])

    result = agent.load_all(tmp_path, llm_config={"api_key": "k", "model": "m"})
    assert len(result) == 2
    assert ag1 in result
    restored_ag2 = next(a for a in result if a.agname == ag2_name[0])
    assert restored_ag2 is not None
    _allocated_agnames.discard(ag2_name[0])
    _allocated_agnames.discard(ag1_name)


@_docker_ok
def test_load_raises_if_agname_already_live(tmp_path):
    from agency.agent import _allocated_agnames
    skill = agskill(name="s", system_prompt="")
    def fake_run(cfg, inp, hist, sb, pool, ms, **_):
        return agdata(done=True), agdata(messages=[]), [], (0, 0)
    skill.run = fake_run

    ag = agent(llm_config={"api_key": "k", "model": "m"})
    ag.run(skill, agdata()).done  # must run a skill to get a checkpoint
    ckpt = tmp_path / "ag.ckpt"
    ag.save(ckpt)

    with pytest.raises(ValueError, match="already in use"):
        agent.load(ckpt, llm_config={"api_key": "k", "model": "m"})
    _allocated_agnames.discard(ag.agname)
