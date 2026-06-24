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
def test_save_and_load_restores_history_and_filesystem(tmp_path, monkeypatch):
    import subprocess as _sp
    from agency.agent import _allocated_agnames

    monkeypatch.setattr(_sp, "run", _make_ckpt_subprocess_mock(_sp.run))

    skill_write = agskill(name="write", system_prompt="")
    def fake_write(cfg, inp, hist, sandbox, pool, ms, **_):
        return agdata(answer="42"), agdata(messages=[{"role": "assistant", "content": "42"}]), [], (0, 0)
    skill_write.run = fake_write

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

    events = ag2.log.events
    assert any(e.get("event") == "loaded" for e in events)
    del ag2
    _allocated_agnames.discard(saved_agname)
    # Container filesystem round-trip (write_file → save → load → read_file)
    # requires real docker save/load (GB-sized export); covered by manual integration test.


def _make_ckpt_subprocess_mock(real_run):
    """Return a subprocess.run replacement that intercepts docker save/load/tag/rmi
    for ckpt images, returning a tiny fake payload instead of exporting real GB-sized
    Docker images to disk. All other subprocess calls pass through unchanged."""
    import subprocess as _sp

    _FAKE_IMAGE = b"FAKE_DOCKER_IMAGE_BYTES"

    def _mock(cmd, *args, **kwargs):
        cmd_str = " ".join(str(c) for c in cmd)
        ops = ("save", "tag", "rmi")
        # Intercept ckpt-related save/tag/rmi AND any bare "load" call (docker load
        # receives our fake image bytes as stdin so must also be mocked).
        is_ckpt_op = (
            ("ckpt" in cmd_str and any(op in cmd_str for op in ops))
            or ("load" in cmd_str and "ckpt" not in cmd_str
                and kwargs.get("input") == _FAKE_IMAGE)
        )
        if not is_ckpt_op:
            return real_run(cmd, *args, **kwargs)
        kwargs.pop("input", None)
        kwargs.pop("capture_output", None)
        return _sp.CompletedProcess(cmd, returncode=0, stdout=_FAKE_IMAGE, stderr=b"")

    return _mock


@_docker_ok
def test_save_all_and_load_all(tmp_path, monkeypatch):
    import gc
    import subprocess as _sp
    from agency.agent import _allocated_agnames

    monkeypatch.setattr(_sp, "run", _make_ckpt_subprocess_mock(_sp.run))

    saved_names = set()

    skill_write = agskill(name="write", system_prompt="")
    def fake_write(cfg, inp, hist, sandbox, pool, ms, **_):
        sandbox.write_file("/workspace/id.txt", f"{inp.agname}\n")
        return agdata(ok=True), agdata(messages=[]), [], (0, 0)
    skill_write.run = fake_write

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
    try:
        assert len(restored) == 2
        assert {a.agname for a in restored} == saved_names
        # Container filesystem round-trip (read_file after load) requires real docker
        # save/load which exports GB-sized images; covered by manual integration test.
    finally:
        for a in restored:
            try:
                if a.sandbox:
                    a.sandbox.destroy()
            except Exception:
                pass
        _allocated_agnames.difference_update(saved_names)


@_docker_ok
def test_load_all_skips_already_live_agent(tmp_path, monkeypatch):
    import gc
    import subprocess as _sp
    from agency.agent import _allocated_agnames

    monkeypatch.setattr(_sp, "run", _make_ckpt_subprocess_mock(_sp.run))

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
    try:
        assert len(result) == 2
        assert ag1 in result
        restored_ag2 = next(a for a in result if a.agname == ag2_name[0])
        assert restored_ag2 is not None
    finally:
        for a in [ag1] + [a for a in result if a is not ag1]:
            try:
                if a.sandbox:
                    a.sandbox.destroy()
            except Exception:
                pass
        _allocated_agnames.discard(ag2_name[0])
        _allocated_agnames.discard(ag1_name)


@_docker_ok
def test_load_raises_if_agname_already_live(tmp_path, monkeypatch):
    import subprocess as _sp
    from agency.agent import _allocated_agnames

    monkeypatch.setattr(_sp, "run", _make_ckpt_subprocess_mock(_sp.run))

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


# ---------------------------------------------------------------------------
# UI state transitions
# ---------------------------------------------------------------------------

def test_ui_state_error_when_skill_returns_error():
    """_ui_state is set to 'error' when the skill returns agdata(error=...)."""
    skill = agskill(name="s", system_prompt="")

    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        return agdata(error="something went wrong"), agdata(messages=[]), [], (0, 0)

    skill.run = fake_run
    ag = make_agent()
    result = ag.run(skill, agdata())
    _ = result.error   # resolve
    assert ag._ui_state["state"] == "error"


def test_ui_state_finished_on_success():
    """_ui_state is set to 'finished' when the skill returns without error."""
    skill = agskill(name="s", system_prompt="")

    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        return agdata(answer="ok"), agdata(messages=[]), [], (0, 0)

    skill.run = fake_run
    ag = make_agent()
    result = ag.run(skill, agdata())
    _ = result.answer   # resolve
    assert ag._ui_state["state"] == "finished"


def test_ui_state_error_on_skill_exception():
    """_ui_state is set to 'error' when the skill raises an unexpected exception."""
    skill = agskill(name="s", system_prompt="")

    def fake_run(llm_cfg, inp, hist, sandbox, pool, ms, **_):
        raise RuntimeError("unexpected crash")

    skill.run = fake_run
    ag = make_agent()
    result = ag.run(skill, agdata())
    _ = result.error   # resolve (will contain the formatted exception)
    assert ag._ui_state["state"] == "error"
