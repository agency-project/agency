import json
import threading
import pytest
from unittest.mock import MagicMock, patch
from src.agdata import agdata
from src.agskill import agskill
from src.agtool import agtool
from src.agent import agent


def _direct(content: str):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


def _tool_resp(name: str, args: dict, call_id: str = "c1"):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    msg = MagicMock()
    msg.content = None
    msg.tool_calls = [tc]
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


def make_agent(**kwargs) -> agent:
    return agent(llm_config={"api_key": "k", "model": "gpt-4o"}, **kwargs)


# ---------------------------------------------------------------------------
# Basic routing
# ---------------------------------------------------------------------------

def test_run_returns_pending_agdata():
    """run() is non-blocking — result fields resolve lazily."""
    skill = agskill(name="s", system_prompt="")
    def fake_run(llm_cfg, inp, hist, tools, ms, **_):
        return agdata(done=True), agdata(messages=[]), []
    skill.run = fake_run

    ag = make_agent(agskills=[skill])
    result = ag.run("s", agdata())
    assert isinstance(result, agdata)
    assert result.done is True   # field access blocks until task finishes


def test_run_calls_named_agskill():
    called = []
    def fake_run(llm_cfg, inp, hist, tools, max_steps, **_):
        called.append(inp.to_dict())
        return agdata(done=True), agdata(messages=[]), []

    skill = agskill(name="dowork", system_prompt="")
    skill.run = fake_run

    ag = make_agent(agskills=[skill])
    result = ag.run("dowork", agdata(task="go"))
    assert result.done is True   # blocks until done
    assert called == [{"task": "go"}]


def test_run_unknown_agskill_returns_error():
    ag = make_agent()
    result = ag.run("nonexistent", agdata(x=1))
    assert result.error is not None
    assert "nonexistent" in result.error


def test_repr():
    ag = make_agent(agskills=[agskill("a", ""), agskill("b", "")])
    assert "a" in repr(ag) and "b" in repr(ag)


# ---------------------------------------------------------------------------
# History is updated and serialized on the same agent
# ---------------------------------------------------------------------------

def test_history_updated_after_run():
    skill = agskill(name="s", system_prompt="")
    def fake_run(llm_cfg, inp, hist, tools, ms, **_):
        existing = list(hist._data.get("messages", []))
        new_hist = agdata(messages=existing + [
            {"role": "user", "content": inp.to_json()},
            {"role": "assistant", "content": "{}"},
        ])
        return agdata(ok=True), new_hist, []

    skill.run = fake_run
    ag = make_agent(agskills=[skill])

    ag.run("s", agdata(turn=1))
    assert len(ag.history.messages) == 2   # ag.history blocks until done

    ag.run("s", agdata(turn=2))
    assert len(ag.history.messages) == 4   # chain: step2 waited for step1


def test_sequential_calls_serialize_via_history_chain():
    """Two calls on the same agent must run in order even though both are non-blocking."""
    order = []
    lock = threading.Lock()

    def make_skill(name):
        sk = agskill(name, "")
        def fake_run(llm_cfg, inp, hist, tools, ms, **_):
            with lock:
                order.append(name)
            return agdata(name=name), agdata(messages=list(hist._data.get("messages", [])) + [
                {"role": "user", "content": name}
            ]), []
        sk.run = fake_run
        return sk

    ag = make_agent(agskills=[make_skill("first"), make_skill("second")])
    ag.run("first", agdata())
    ag.run("second", agdata())
    _ = ag.history   # wait for both to finish

    assert order == ["first", "second"]


def test_history_passed_to_agskill():
    ag = make_agent(agskills=[agskill("s", "")])
    ag.history = agdata(messages=[{"role": "user", "content": "prior"}])

    received = {}
    def fake_run(llm_cfg, inp, hist, tools, ms, **_):
        received["hist"] = hist
        return agdata(), agdata(messages=[]), []
    ag.agskills[0].run = fake_run

    ag.run("s", agdata(x=1))
    _ = ag.history   # sync
    assert received["hist"].messages[0]["content"] == "prior"


# ---------------------------------------------------------------------------
# Tools are passed to agskill
# ---------------------------------------------------------------------------

def test_agent_tools_passed_to_agskill():
    t = agtool(name="t1", description="", fn=lambda a: agdata())
    ag = make_agent(agskills=[agskill("s", "")], tools=[t])

    received = {}
    def fake_run(llm_cfg, inp, hist, tools, ms, **_):
        received["tools"] = tools
        return agdata(), agdata(messages=[]), []
    ag.agskills[0].run = fake_run

    ag.run("s", agdata())
    _ = ag.history   # sync
    assert received["tools"] == [t]


# ---------------------------------------------------------------------------
# End-to-end with mocked OpenAI
# ---------------------------------------------------------------------------

def test_end_to_end_direct_answer():
    skill = agskill(name="qa", system_prompt="Answer questions.")
    ag = make_agent(agskills=[skill])

    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct('{"answer": "Paris"}')
        result = ag.run("qa", agdata(question="Capital of France?"))
        assert result.answer == "Paris"   # resolve inside the patch context


def test_end_to_end_with_tool():
    called = []
    def calc_fn(arg: agdata) -> agdata:
        called.append(arg.to_dict())
        return agdata(result=arg.a + arg.b)  # type: ignore[operator]

    calc = agtool(
        name="add",
        description="Add two numbers.",
        fn=calc_fn,
        params={"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]},
    )
    skill = agskill(name="math", system_prompt="You are a calculator.")
    ag = make_agent(agskills=[skill], tools=[calc])

    responses = [_tool_resp("add", {"a": 3, "b": 4}), _direct('{"result": 7}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result = ag.run("math", agdata(task="add 3 and 4"))
        assert result.result == 7         # resolve inside the patch context

    assert called == [{"a": 3, "b": 4}]


def test_multiple_agskills_coexist():
    ag = make_agent(agskills=[agskill("a", ""), agskill("b", "")])
    results = {}
    for af in ag.agskills:
        def fake(llm, inp, hist, tools, ms, _name=af.name, **_):
            return agdata(from_skill=_name), agdata(messages=[]), []
        af.run = fake

    results["a"] = ag.run("a", agdata())
    results["b"] = ag.run("b", agdata())
    assert results["a"].from_skill == "a"
    assert results["b"].from_skill == "b"


# ---------------------------------------------------------------------------
# Copy constructor
# ---------------------------------------------------------------------------

def test_copy_constructor_inherits_config_and_skills():
    skill = agskill("s", "")
    ag = make_agent(agskills=[skill], tools=[])

    copy_ag = agent(ag)
    assert copy_ag.llm_config == ag.llm_config
    assert copy_ag.agskills is not ag.agskills   # new list
    assert copy_ag.agskills[0] is skill          # same skill objects


def test_copy_constructor_deep_copies_history():
    ag = make_agent()
    ag.history = agdata(messages=[{"role": "user", "content": "original"}])

    copy_ag = agent(ag)
    copy_ag.history.messages.append({"role": "assistant", "content": "new"})

    assert len(ag.history.messages) == 1   # parent unaffected


def test_copy_constructor_overrides_agskills():
    skill_a = agskill("a", "")
    skill_b = agskill("b", "")
    ag = make_agent(agskills=[skill_a])

    copy_ag = agent(ag, agskills=[skill_b])
    assert [s.name for s in copy_ag.agskills] == ["b"]


def test_fork_is_alias_for_copy_constructor():
    ag = make_agent(agskills=[agskill("s", "")])
    ag.history = agdata(messages=[{"role": "user", "content": "x"}])
    forked = ag.fork()
    assert forked.llm_config == ag.llm_config
    assert len(forked.history.messages) == 1


def test_copy_constructor_waits_for_inflight_task():
    """agent(src) blocks until src's current task finishes before copying history."""
    skill = agskill("s", "")
    def fake_run(llm_cfg, inp, hist, tools, ms, **_):
        return agdata(v=inp.v), agdata(messages=[{"role": "user", "content": str(inp.v)}]), []
    skill.run = fake_run

    ag = make_agent(agskills=[skill])
    ag.run("s", agdata(v=42))          # non-blocking, in-flight

    fork = agent(ag)                   # blocks until the run completes
    assert len(fork.history.messages) == 1
    assert fork.history.messages[0]["content"] == "42"


# ---------------------------------------------------------------------------
# Parallel execution via copy constructor
# ---------------------------------------------------------------------------

def test_fork_runs_do_not_update_parent_history():
    skill = agskill("s", "")
    def fake_run(llm_cfg, inp, hist, tools, ms, **_):
        return agdata(ok=True), agdata(messages=[{"role": "user", "content": "fork_msg"}]), []
    skill.run = fake_run

    ag = make_agent(agskills=[skill])
    ag.history = agdata(messages=[{"role": "user", "content": "original"}])

    r = agent(ag).run("s", agdata())
    _ = r.ok   # wait for fork to finish

    assert len(ag.history.messages) == 1
    assert ag.history.messages[0]["content"] == "original"


def test_fork_runs_in_parallel():
    """Multiple forks reach the barrier together, proving concurrent execution."""
    barrier = threading.Barrier(3)
    skill = agskill("s", "")
    def fake_run(llm_cfg, inp, hist, tools, ms, **_):
        barrier.wait(timeout=5)
        return agdata(n=inp.n), agdata(messages=[]), []
    skill.run = fake_run

    ag = make_agent(agskills=[skill])
    results = [agent(ag).run("s", agdata(n=i)) for i in range(3)]
    assert sorted(r.n for r in results) == [0, 1, 2]


def test_fork_sees_parent_history_at_fork_time():
    seen = {}
    skill = agskill("s", "")
    def fake_run(llm_cfg, inp, hist, tools, ms, **_):
        seen["hist"] = list(hist.messages)
        return agdata(), agdata(messages=[]), []
    skill.run = fake_run

    ag = make_agent(agskills=[skill])
    ag.history = agdata(messages=[{"role": "user", "content": "seed"}])

    r = agent(ag).run("s", agdata())
    r._resolve()
    assert seen["hist"][0]["content"] == "seed"


# ---------------------------------------------------------------------------
# Pending agdata as input — auto-resolved before skill runs
# ---------------------------------------------------------------------------

def test_run_accepts_pending_agdata_as_input():
    from concurrent.futures import Future
    received = {}

    skill = agskill("s", "")
    def fake_run(llm_cfg, inp, hist, tools, ms, **_):
        received["inp"] = inp.to_dict()
        return agdata(ok=True), agdata(messages=[]), []
    skill.run = fake_run

    ag = make_agent(agskills=[skill])

    f: Future[agdata] = Future()
    f.set_result(agdata(resolved=True, value=99))
    pending_input = agdata(_future=f)

    result = ag.run("s", pending_input)
    _ = result.ok
    assert received["inp"] == {"resolved": True, "value": 99}


def test_run_resolves_list_of_pending_in_input():
    from concurrent.futures import Future
    received = {}

    skill = agskill("s", "")
    def fake_run(llm_cfg, inp, hist, tools, ms, **_):
        received["items"] = inp.items
        return agdata(ok=True), agdata(messages=[]), []
    skill.run = fake_run

    ag = make_agent(agskills=[skill])

    futures = []
    for i in range(3):
        f: Future[agdata] = Future()
        f.set_result(agdata(text=f"item {i}"))
        futures.append(agdata(_future=f))

    result = ag.run("s", agdata(items=futures))
    _ = result.ok
    assert [r.text for r in received["items"]] == ["item 0", "item 1", "item 2"]


def test_chained_run_output_as_next_input():
    """Output of one run() passed directly as input to the next — resolved automatically."""
    skill = agskill("s", "")
    received_inputs = []
    def fake_run(llm_cfg, inp, hist, tools, ms, **_):
        received_inputs.append(dict(inp._data))
        return agdata(done=True), agdata(messages=[]), []
    skill.run = fake_run

    ag1 = make_agent(agskills=[skill])
    ag2 = make_agent(agskills=[skill])

    r1 = ag1.run("s", agdata(x=5))   # pending agdata, resolves to agdata(done=True)
    r2 = ag2.run("s", r1)            # r1 passed as input; resolved before ag2's skill runs
    r2._resolve()

    # ag2 received the resolved r1 as its input
    assert received_inputs[1] == {"done": True}
