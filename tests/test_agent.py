import json
import os
import threading
import pytest
from unittest.mock import MagicMock, patch
from agency.agdata import agdata, agerror
from agency.agcontext import agcontext
from agency.agskill import agskill
from agency.agtool import agtool
from agency.agent import agent
from agency.agname import agname as _agname
from agency.agconfig import agConfig

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
    def __init__(self, delta):
        self.delta = delta


class _Usage:
    prompt_tokens = 5


class _Chunk:
    def __init__(self, content=None, tool_calls=None, usage=None):
        self.usage = usage
        self.choices = (
            [_Choice(_Delta(content, tool_calls))] if (content is not None or tool_calls) else []
        )


class _TCDelta:
    def __init__(self, name, args_json, call_id):
        self.id = call_id
        self.index = 0
        self.function = _TCFnDelta(name, args_json)


class _TCFnDelta:
    def __init__(self, name, args):
        self.name = name
        self.arguments = args


def _noop(arg: agdata) -> agdata:
    return agdata()


def _direct(content: str) -> list:
    return [_Chunk(content=content), _Chunk(usage=_Usage())]


def _tool_resp(name: str, args: dict, call_id: str = "c1") -> list:
    tc = _TCDelta(name, json.dumps(args), call_id)
    return [_Chunk(tool_calls=[tc]), _Chunk(usage=_Usage())]


def _llm_agconfig(d: dict) -> agConfig:
    # Force the docker sandbox backend for any test that ends up constructing
    # a real sandbox -- agsandbox_backends' "auto" selection now prefers
    # podman over docker when both are usable, but CI's images/build.sh only
    # builds/tags agency-sandbox:latest for docker, so podman has no local
    # image and would try (and fail) to pull one from a registry. This has
    # no effect on the many tests here that never touch ag.sandbox at all.
    from agency.agsandbox_backends import agSandboxBackendConfig

    return agConfig(agSandboxBackendConfig(backend="docker"), {"agllm_backend": dict(d)})


def make_agent() -> agent:
    return agent(agconfig=_llm_agconfig({"api_key": "k", "model": ""}))


# ---------------------------------------------------------------------------
# Basic routing
# ---------------------------------------------------------------------------


def test_run_returns_pending_agdata():
    """run() is non-blocking — result fields resolve lazily."""
    skill = agskill(name="s", system_prompt="")

    def fake_execute_react(ag, prev_ctx, inp, max_steps=None, **_):
        return agdata(done=True), prev_ctx, []

    skill.execute_react = fake_execute_react

    ag = make_agent()
    result = ag.run(skill, agdata())
    assert isinstance(result, agdata)
    assert result.done is True  # field access blocks until task finishes


def test_run_calls_named_agskill():
    called = []

    def fake_execute_react(ag, prev_ctx, inp, max_steps=None, **_):
        called.append(inp.to_dict())
        return agdata(done=True), prev_ctx, []

    skill = agskill(name="dowork", system_prompt="")
    skill.execute_react = fake_execute_react

    ag = make_agent()
    result = ag.run(skill, agdata(task="go"))
    assert result.done is True  # blocks until done
    assert called == [{"task": "go"}]


def test_repr():
    ag = make_agent()
    assert repr(ag) is not None  # agent repr works without owned skills


# ---------------------------------------------------------------------------
# History is updated and serialized on the same agent
# ---------------------------------------------------------------------------


def test_history_updated_after_run():
    skill = agskill(name="s", system_prompt="")

    def fake_execute_react(ag, prev_ctx, inp, max_steps=None, **_):
        new_msgs = list(prev_ctx.messages) + [
            {"role": "user", "content": inp.to_json()},
            {"role": "assistant", "content": "{}"},
        ]
        return agdata(ok=True), agcontext(messages=new_msgs), []

    skill.execute_react = fake_execute_react
    ag = make_agent()

    ag.run(skill, agdata(turn=1))
    assert len(ag.history.messages) == 2  # ag.history blocks until done

    ag.run(skill, agdata(turn=2))
    assert len(ag.history.messages) == 4  # chain: step2 waited for step1


def test_sequential_calls_serialize_via_history_chain():
    """Two calls on the same agent must run in order even though both are non-blocking."""
    order = []
    lock = threading.Lock()

    def make_skill(name):
        sk = agskill(name, "")

        def fake_execute_react(ag, prev_ctx, inp, max_steps=None, **_):
            with lock:
                order.append(name)
            new_msgs = list(prev_ctx.messages) + [{"role": "user", "content": name}]
            return agdata(name=name), agcontext(messages=new_msgs), []

        sk.execute_react = fake_execute_react
        return sk

    skill_first = make_skill("first")
    skill_second = make_skill("second")
    ag = make_agent()
    ag.run(skill_first, agdata())
    ag.run(skill_second, agdata())
    _ = ag.history  # wait for both to finish

    assert order == ["first", "second"]


def test_history_passed_to_agskill():
    skill = agskill("s", "")
    ag = make_agent()
    ag.history = agdata(messages=[{"role": "user", "content": "prior"}])

    received = {}

    def fake_execute_react(ag, prev_ctx, inp, max_steps=None, **_):
        received["hist"] = prev_ctx
        return agdata(), prev_ctx, []

    skill.execute_react = fake_execute_react

    ag.run(skill, agdata(x=1))
    _ = ag.history  # sync
    assert received["hist"].messages[0]["content"] == "prior"


# ---------------------------------------------------------------------------
# Tools live on skills, not agents
# ---------------------------------------------------------------------------


def test_skill_replace_tools_used_in_run():
    """replace_tools on the skill replaces the full tool list."""
    t = agtool(name="t1", description="", fn=_noop)
    skill = agskill("s", "", replace_tools=[t])
    captured = {}

    def fake_execute_react(ag, prev_ctx, inp, max_steps=None, **_):
        captured["replace_tools"] = skill.replace_tools
        return agdata(), prev_ctx, []

    skill.execute_react = fake_execute_react

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
        MockClient.return_value.chat.completions.create.return_value = _direct(
            '{"answer": "Paris"}'
        )
        result = ag.run(skill, agdata(question="Capital of France?"))
        assert result.result == '{"answer": "Paris"}'  # resolve inside the patch context


def test_end_to_end_with_tool():
    def calc_fn(arg: agdata) -> agdata:
        return agdata(result=arg.a + arg.b)  # type: ignore[operator]

    calc = agtool(
        name="add",
        description="Add two numbers.",
        fn=calc_fn,
        params={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    )
    skill = agskill(name="math", system_prompt="You are a calculator.", add_tools=[calc])
    ag = make_agent()

    responses = [_tool_resp("add", {"a": 3, "b": 4}), _direct('{"result": 7}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result = ag.run(skill, agdata(task="add 3 and 4"))
        assert result.result == '{"result": 7}'


def test_multiple_agskills_coexist():
    skill_a = agskill("a", "")
    skill_b = agskill("b", "")

    def fake_a(ag, prev_ctx, inp, max_steps=None, **_):
        return agdata(from_skill="a"), prev_ctx, []

    def fake_b(ag, prev_ctx, inp, max_steps=None, **_):
        return agdata(from_skill="b"), prev_ctx, []

    skill_a.execute_react = fake_a
    skill_b.execute_react = fake_b

    ag = make_agent()
    results = {}
    results["a"] = ag.run(skill_a, agdata())
    results["b"] = ag.run(skill_b, agdata())
    assert results["a"].from_skill == "a"
    assert results["b"].from_skill == "b"


# ---------------------------------------------------------------------------
# Copy constructor
# ---------------------------------------------------------------------------


def test_fork_inherits_config():
    ag = make_agent()

    forked = agent.fork(ag)
    assert forked.llm.backend.as_dict() == ag.llm.backend.as_dict()


def test_fork_deep_copies_history():
    ag = make_agent()
    ag.history = agdata(messages=[{"role": "user", "content": "original"}])

    forked = agent.fork(ag)
    forked.history.messages.append({"role": "assistant", "content": "new"})

    assert len(ag.history.messages) == 1  # parent unaffected


def test_fork_copies_history_and_config():
    ag = make_agent()
    ag.history = agdata(messages=[{"role": "user", "content": "prior"}])

    forked = agent.fork(ag)
    assert forked.llm.backend.as_dict() == ag.llm.backend.as_dict()
    assert len(forked.history.messages) == 1
    assert forked.history.messages[0]["content"] == "prior"


def test_fork_waits_for_inflight_task():
    """agent.fork(src) blocks until src's current task finishes before copying history."""
    skill = agskill("s", "")

    def fake_execute_react(ag, prev_ctx, inp, max_steps=None, **_):
        new_ctx = agcontext(messages=[{"role": "user", "content": str(inp.v)}])
        return agdata(v=inp.v), new_ctx, []

    skill.execute_react = fake_execute_react

    ag = make_agent()
    ag.run(skill, agdata(v=42))  # non-blocking, in-flight

    forked = agent.fork(ag)  # blocks until the run completes
    assert len(forked.history.messages) == 1
    assert forked.history.messages[0]["content"] == "42"


# ---------------------------------------------------------------------------
# Parallel execution via copy constructor
# ---------------------------------------------------------------------------


def test_fork_runs_do_not_update_parent_history():
    skill = agskill("s", "")

    def fake_execute_react(ag, prev_ctx, inp, max_steps=None, **_):
        new_ctx = agcontext(messages=[{"role": "user", "content": "fork_msg"}])
        return agdata(ok=True), new_ctx, []

    skill.execute_react = fake_execute_react

    ag = make_agent()
    ag.history = agdata(messages=[{"role": "user", "content": "original"}])

    r = agent.fork(ag).run(skill, agdata())
    _ = r.ok  # wait for fork to finish

    assert len(ag.history.messages) == 1
    assert ag.history.messages[0]["content"] == "original"


def test_fork_runs_in_parallel():
    """Multiple forks reach the barrier together, proving concurrent execution."""
    barrier = threading.Barrier(3)
    skill = agskill("s", "")

    def fake_execute_react(ag, prev_ctx, inp, max_steps=None, **_):
        barrier.wait(timeout=5)
        return agdata(n=inp.n), prev_ctx, []

    skill.execute_react = fake_execute_react

    ag = make_agent()
    results = [agent.fork(ag).run(skill, agdata(n=i)) for i in range(3)]
    assert sorted(r.n for r in results) == [0, 1, 2]


def test_fork_sees_parent_history_at_fork_time():
    seen = {}
    skill = agskill("s", "")

    def fake_execute_react(ag, prev_ctx, inp, max_steps=None, **_):
        seen["hist"] = list(prev_ctx.messages)
        return agdata(), prev_ctx, []

    skill.execute_react = fake_execute_react

    ag = make_agent()
    ag.history = agdata(messages=[{"role": "user", "content": "seed"}])

    r = agent.fork(ag).run(skill, agdata())
    r._resolve()
    assert seen["hist"][0]["content"] == "seed"


# ---------------------------------------------------------------------------
# Pending agdata as input — auto-resolved before skill runs
# ---------------------------------------------------------------------------


def test_run_accepts_pending_agdata_as_input():
    from concurrent.futures import Future

    received = {}

    skill = agskill("s", "")

    def fake_execute_react(ag, prev_ctx, inp, max_steps=None, **_):
        received["inp"] = inp.to_dict()
        return agdata(ok=True), prev_ctx, []

    skill.execute_react = fake_execute_react

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

    def fake_execute_react(ag, prev_ctx, inp, max_steps=None, **_):
        received["items"] = inp.items
        return agdata(ok=True), prev_ctx, []

    skill.execute_react = fake_execute_react

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

    def fake_execute_react(ag, prev_ctx, inp, max_steps=None, **_):
        received_inputs.append(dict(inp._data))
        return agdata(done=True), prev_ctx, []

    skill.execute_react = fake_execute_react

    ag1 = make_agent()
    ag2 = make_agent()

    r1 = ag1.run(skill, agdata(x=5))  # pending agdata, resolves to agdata(done=True)
    r2 = ag2.run(skill, r1)  # r1 passed as input; resolved before ag2's skill runs
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
    live_names = [a.agname for a in agent.all()]
    assert ag2_name not in live_names
    assert ag1.agname in live_names


# ---------------------------------------------------------------------------
# Sandbox garbage collection
#
# agent has no __del__ logic for sandboxes anymore (no is_external_sandbox,
# no custom sandbox getter/setter) — cleanup relies entirely on Python
# refcounting plus agSandbox's own __del__/destroy(). These tests use a
# lightweight stand-in that mirrors agSandbox's destroy-on-GC contract
# (idempotent destroy() invoked from __del__) without touching Docker.
# ---------------------------------------------------------------------------


class _GCSandbox:
    def __init__(self, on_destroy):
        self._on_destroy = on_destroy
        self._destroyed = False

    def destroy(self):
        if self._destroyed:
            return
        self._destroyed = True
        self._on_destroy()

    def __del__(self):
        self.destroy()


def test_agent_internal_sandbox_destroyed_only_after_agent_is_gone():
    """A sandbox the agent owns outright must not be destroyed while the
    agent is still alive, but must be cleaned up once it isn't."""
    import gc
    import weakref

    destroyed = []
    sb = _GCSandbox(lambda: destroyed.append(True))
    ref = weakref.ref(sb)

    ag = make_agent()
    ag.sandbox = sb
    del sb
    gc.collect()

    assert ref() is not None  # still alive — ag.sandbox holds it
    assert destroyed == []  # not destroyed just because ag exists

    del ag
    gc.collect()

    assert ref() is None  # collected once its last owner is gone
    assert destroyed == [True]


def test_external_sandbox_survives_agent_deletion_while_still_referenced():
    """A caller-provided sandbox shared beyond the agent must outlive that
    agent's deletion — nothing in agent.__del__ should force-destroy it."""
    import gc
    import weakref

    destroyed = []
    sb = _GCSandbox(lambda: destroyed.append(True))
    ref = weakref.ref(sb)

    ag = agent(agconfig=_llm_agconfig({"api_key": "k", "model": ""}), sandbox=sb)
    del ag
    gc.collect()

    assert ref() is not None  # our own reference keeps it alive
    assert destroyed == []

    del sb
    gc.collect()

    assert ref() is None  # only now, with no references left
    assert destroyed == [True]


# ---------------------------------------------------------------------------
# Checkpointing (requires Docker)
# ---------------------------------------------------------------------------

_docker_ok = pytest.mark.skipif(
    not (
        lambda: (
            __import__("subprocess")
            .run(["docker", "info"], capture_output=True, timeout=10)
            .returncode
            == 0
        )
    )(),
    reason="Docker not available",
)


@_docker_ok
def test_save_and_load_restores_history_and_filesystem(tmp_path, monkeypatch):
    import subprocess as _sp

    monkeypatch.setattr(_sp, "run", _make_ckpt_subprocess_mock(_sp.run))

    skill_write = agskill(name="write", system_prompt="")

    def fake_write(ag, prev_ctx, inp, max_steps=None, **_):
        new_ctx = agcontext(messages=[{"role": "assistant", "content": "42"}])
        return agdata(answer="42"), new_ctx, []

    skill_write.execute_react = fake_write

    ag = agent(agconfig=_llm_agconfig({"api_key": "k", "model": "m"}))
    ag.run(skill_write, agdata(q="test")).answer

    ckpt = tmp_path / "agent.ckpt"
    ag.save(ckpt)
    assert ckpt.exists()
    saved_agname = ag.agname
    del ag
    _agname._allocated.discard(saved_agname)

    ag2 = agent.load(ckpt, agconfig=_llm_agconfig({"api_key": "k", "model": "m"}))
    assert ag2.agname == saved_agname
    assert len(ag2.ctx.messages) > 0
    assert ag2 in agent.all()

    events = ag2.log.events
    assert any(e.get("event") == "loaded" for e in events)
    del ag2
    _agname._allocated.discard(saved_agname)
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
        # create/commit: relabel_owner_pid()'s scrub-on-save/restamp-on-load
        # dance (docker create <tag> && docker commit --change ... <cid> <tag>).
        ops = ("save", "tag", "rmi", "create", "commit")
        # Intercept ckpt-related save/tag/rmi/create/commit AND any bare "load"
        # call (docker load receives our fake image bytes as stdin so must
        # also be mocked).
        is_ckpt_op = ("ckpt" in cmd_str and any(op in cmd_str for op in ops)) or (
            "load" in cmd_str and "ckpt" not in cmd_str and kwargs.get("input") == _FAKE_IMAGE
        )
        if not is_ckpt_op:
            return real_run(cmd, *args, **kwargs)
        kwargs.pop("input", None)
        kwargs.pop("capture_output", None)
        return _sp.CompletedProcess(cmd, returncode=0, stdout=_FAKE_IMAGE, stderr=b"")

    return _mock


@_docker_ok
def test_save_scrubs_and_load_restamps_owner_pid_label(tmp_path, monkeypatch):
    """save() must scrub the owning process's PID (pass None to
    relabel_owner_pid()) before embedding the sandbox image -- a foreign
    PID from this process is meaningless, and potentially misleading,
    once restored by an unrelated process possibly on a different host.
    load() must then restamp the actually-current restoring process's own
    PID afterward. See relabel_owner_pid()'s docstring for why:
    reap_orphaned_containers()'s image scan trusts this label to decide
    whether an image's owner is still alive, and a stale/foreign PID
    could make it act on wrong evidence."""
    import subprocess as _sp
    from agency.agsandbox_backends.container import _ContainerBackendBase

    monkeypatch.setattr(_sp, "run", _make_ckpt_subprocess_mock(_sp.run))

    skill_write = agskill(name="write", system_prompt="")

    def fake_write(ag, prev_ctx, inp, max_steps=None, **_):
        ag.sandbox.write_file("/workspace/id.txt", f"{inp.agname}\n")
        return agdata(ok=True), prev_ctx, []

    skill_write.execute_react = fake_write

    ag = agent(agconfig=_llm_agconfig({"api_key": "k", "model": "m"}))
    ag.run(skill_write, agdata(agname=ag.agname)).ok
    assert ag.sandbox is not None and ag.sandbox._checkpoint_image is not None

    ckpt = tmp_path / "agent.ckpt"
    with patch.object(_ContainerBackendBase, "relabel_owner_pid") as mock_relabel:
        ag.save(ckpt)
    assert mock_relabel.call_count == 1
    assert mock_relabel.call_args.args[1] is None, "save() must scrub, not preserve, the PID"

    saved_agname = ag.agname
    ag.sandbox.destroy()
    del ag
    _agname._allocated.discard(saved_agname)

    with patch.object(_ContainerBackendBase, "relabel_owner_pid") as mock_relabel2:
        ag2 = agent.load(ckpt, agconfig=_llm_agconfig({"api_key": "k", "model": "m"}))
    assert mock_relabel2.call_count == 1
    assert mock_relabel2.call_args.args[1] == os.getpid(), (
        "load() must restamp the CURRENT restoring process's own PID"
    )

    if ag2.sandbox:
        ag2.sandbox.destroy()
    del ag2
    _agname._allocated.discard(saved_agname)


@_docker_ok
def test_save_all_and_load_all(tmp_path, monkeypatch):
    import gc
    import subprocess as _sp

    monkeypatch.setattr(_sp, "run", _make_ckpt_subprocess_mock(_sp.run))

    saved_names = set()

    skill_write = agskill(name="write", system_prompt="")

    def fake_write(ag, prev_ctx, inp, max_steps=None, **_):
        ag.sandbox.write_file("/workspace/id.txt", f"{inp.agname}\n")
        return agdata(ok=True), prev_ctx, []

    skill_write.execute_react = fake_write

    def _create_and_save():
        ag1 = agent(agconfig=_llm_agconfig({"api_key": "k", "model": "m"}))
        ag2 = agent(agconfig=_llm_agconfig({"api_key": "k", "model": "m"}))
        ag1.run(skill_write, agdata(agname=ag1.agname)).ok
        ag2.run(skill_write, agdata(agname=ag2.agname)).ok
        saved_names.update([ag1.agname, ag2.agname])
        agent.save_all(tmp_path)

    _create_and_save()
    gc.collect()
    _agname._allocated.difference_update(saved_names)

    restored = agent.load_all(tmp_path, agconfig=_llm_agconfig({"api_key": "k", "model": "m"}))
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
            except Exception as _e:
                print(f"test cleanup: destroy() failed for {a.agname}: {_e}")
        _agname._allocated.difference_update(saved_names)


@_docker_ok
def test_load_all_skips_already_live_agent(tmp_path, monkeypatch):
    import gc
    import subprocess as _sp

    monkeypatch.setattr(_sp, "run", _make_ckpt_subprocess_mock(_sp.run))

    skill = agskill(name="s", system_prompt="")

    def fake_execute_react(ag, prev_ctx, inp, max_steps=None, **_):
        return agdata(ok=True), prev_ctx, []

    skill.execute_react = fake_execute_react

    ag1 = agent(agconfig=_llm_agconfig({"api_key": "k", "model": "m"}))
    ag1.run(skill, agdata()).ok  # needs a checkpoint for save
    ag1_name = ag1.agname
    ag2_name = [None]

    def _create_save_ag2():
        ag2 = agent(agconfig=_llm_agconfig({"api_key": "k", "model": "m"}))
        ag2.run(skill, agdata()).ok
        ag2_name[0] = ag2.agname
        agent.save_all(tmp_path)

    _create_save_ag2()
    gc.collect()
    _agname._allocated.discard(ag2_name[0])

    result = agent.load_all(tmp_path, agconfig=_llm_agconfig({"api_key": "k", "model": "m"}))
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
            except Exception as _e:
                print(f"test cleanup: destroy() failed for {a.agname}: {_e}")
        _agname._allocated.discard(ag2_name[0])
        _agname._allocated.discard(ag1_name)


@_docker_ok
def test_load_raises_if_agname_already_live(tmp_path, monkeypatch):
    import subprocess as _sp

    monkeypatch.setattr(_sp, "run", _make_ckpt_subprocess_mock(_sp.run))

    skill = agskill(name="s", system_prompt="")

    def fake_execute_react(ag, prev_ctx, inp, max_steps=None, **_):
        return agdata(done=True), prev_ctx, []

    skill.execute_react = fake_execute_react

    ag = agent(agconfig=_llm_agconfig({"api_key": "k", "model": "m"}))
    ag.run(skill, agdata()).done  # must run a skill to get a checkpoint
    ckpt = tmp_path / "ag.ckpt"
    ag.save(ckpt)

    with pytest.raises(ValueError, match="already in use"):
        agent.load(ckpt, agconfig=_llm_agconfig({"api_key": "k", "model": "m"}))
    _agname._allocated.discard(ag.agname)


# ---------------------------------------------------------------------------
# UI state transitions
# ---------------------------------------------------------------------------


def test_ui_state_error_when_skill_returns_error():
    """_ui_state is set to 'error' when the skill returns agerror(...)."""
    skill = agskill(name="s", system_prompt="")

    def fake_execute_react(ag, prev_ctx, inp, max_steps=None, **_):
        return agerror("something went wrong"), prev_ctx, []

    skill.execute_react = fake_execute_react
    ag = make_agent()
    result = ag.run(skill, agdata())
    _ = result.error  # resolve
    assert ag._state.state == "error"


def test_ui_state_finished_on_success():
    """_ui_state is set to 'finished' when the skill returns without error."""
    skill = agskill(name="s", system_prompt="")

    def fake_execute_react(ag, prev_ctx, inp, max_steps=None, **_):
        return agdata(answer="ok"), prev_ctx, []

    skill.execute_react = fake_execute_react
    ag = make_agent()
    result = ag.run(skill, agdata())
    _ = result.answer  # resolve
    assert ag._state.state == "finished"


def test_ui_state_error_on_skill_exception():
    """_ui_state is set to 'error' when the skill raises an unexpected exception."""
    skill = agskill(name="s", system_prompt="")

    def fake_execute_react(ag, prev_ctx, inp, max_steps=None, **_):
        raise RuntimeError("unexpected crash")

    skill.execute_react = fake_execute_react
    ag = make_agent()
    result = ag.run(skill, agdata())
    _ = result.error  # resolve (will contain the formatted exception)
    assert ag._state.state == "error"


# ---------------------------------------------------------------------------
# Input offloading (_offload_large_fields)
# (moved from test_agtype.py — tests agent.py functions)
# ---------------------------------------------------------------------------

from agency.agtype import agtype, agfile, agimage, agbinary, agrawstring
from agency.agschema import agschema as _agschema


def test_prepare_inputs_in_sandbox_replaces_long_string():
    from agency.agschema import _AgSchemaFields

    sandbox = MagicMock()
    long_val = "x" * (_AgSchemaFields.input_offload_chars.default + 1)
    inp = agdata(text=long_val, small="hi")
    paths, fields = _agschema(agdata(text=str, small=str)).prepare_inputs_in_sandbox(
        inp, sandbox, "mskill"
    )
    sandbox.write_file.assert_called_once()
    assert "text" in fields
    assert "small" not in fields
    assert len(paths) == 1
    assert "mskill_text" in paths[0]
    assert "content saved to" in inp._data["text"]
    assert inp._data["small"] == "hi"


def test_prepare_inputs_in_sandbox_skips_short_strings():
    sandbox = MagicMock()
    inp = agdata(text="short")
    paths, fields = _agschema(agdata(text=str)).prepare_inputs_in_sandbox(inp, sandbox, "skill")
    sandbox.write_file.assert_not_called()
    assert paths == []
    assert fields == []


def test_prepare_inputs_in_sandbox_skips_non_string_scalars():
    sandbox = MagicMock()
    inp = agdata(n=42)
    paths, fields = _agschema(agdata(n=int)).prepare_inputs_in_sandbox(inp, sandbox, "skill")
    sandbox.write_file.assert_not_called()
    assert paths == []


def test_prepare_inputs_in_sandbox_sandbox_failure_leaves_field_unchanged():
    from agency.agschema import _AgSchemaFields

    sandbox = MagicMock()
    sandbox.write_file.side_effect = OSError("no space")
    long_val = "y" * (_AgSchemaFields.input_offload_chars.default + 1)
    inp = agdata(text=long_val)
    paths, fields = _agschema(agdata(text=str)).prepare_inputs_in_sandbox(inp, sandbox, "skill")
    assert paths == []
    assert fields == []
    assert inp._data["text"] == long_val


def test_prepare_inputs_in_sandbox_list_large_strings_replaced_with_paths():
    from agency.agschema import _AgSchemaFields

    sandbox = MagicMock()
    long_a = "a" * (_AgSchemaFields.input_offload_chars.default + 1)
    long_b = "b" * (_AgSchemaFields.input_offload_chars.default + 1)
    inp = agdata(items=[long_a, long_b])
    paths, fields = _agschema(agdata(items=list)).prepare_inputs_in_sandbox(inp, sandbox, "sk")
    assert sandbox.write_file.call_count == 2
    assert "items" in fields
    assert len(paths) == 2
    assert "sk_items_0" in paths[0]
    assert "sk_items_1" in paths[1]
    result = inp._data["items"]
    assert result[0] == paths[0]
    assert result[1] == paths[1]


def test_prepare_inputs_in_sandbox_list_short_strings_unchanged():
    sandbox = MagicMock()
    inp = agdata(items=["short", "also short"])
    paths, fields = _agschema(agdata(items=list)).prepare_inputs_in_sandbox(inp, sandbox, "sk")
    sandbox.write_file.assert_not_called()
    assert paths == []
    assert fields == []
    assert inp._data["items"] == ["short", "also short"]


def test_prepare_inputs_in_sandbox_list_mixed_only_large_replaced():
    from agency.agschema import _AgSchemaFields

    sandbox = MagicMock()
    long_val = "x" * (_AgSchemaFields.input_offload_chars.default + 1)
    inp = agdata(items=["short", long_val])
    paths, fields = _agschema(agdata(items=list)).prepare_inputs_in_sandbox(inp, sandbox, "sk")
    sandbox.write_file.assert_called_once()
    result = inp._data["items"]
    assert result[0] == "short"
    assert result[1] == paths[0]


def test_prepare_inputs_in_sandbox_list_non_string_elements_skipped():
    sandbox = MagicMock()
    inp = agdata(items=[42, None, {"key": "val"}])
    paths, fields = _agschema(agdata(items=list)).prepare_inputs_in_sandbox(inp, sandbox, "sk")
    sandbox.write_file.assert_not_called()
    assert paths == []


def test_prepare_inputs_in_sandbox_list_sandbox_failure_leaves_element_unchanged():
    from agency.agschema import _AgSchemaFields

    sandbox = MagicMock()
    sandbox.write_file.side_effect = OSError("no space")
    long_val = "x" * (_AgSchemaFields.input_offload_chars.default + 1)
    inp = agdata(items=[long_val])
    paths, fields = _agschema(agdata(items=list)).prepare_inputs_in_sandbox(inp, sandbox, "sk")
    assert paths == []
    assert fields == []
    assert inp._data["items"] == [long_val]


def test_prepare_inputs_in_sandbox_skips_agtype_list_fields():
    from agency.agschema import _AgSchemaFields

    sandbox = MagicMock()
    data_url = "data:image/jpeg;base64," + "A" * (_AgSchemaFields.input_offload_chars.default + 1)
    inp = agdata(frames=[data_url, data_url])
    schema = agdata(frames=list[agimage])
    paths, fields = _agschema(schema).prepare_inputs_in_sandbox(inp, sandbox, "sk")
    sandbox.write_file.assert_not_called()
    assert paths == []
    assert inp._data["frames"] == [data_url, data_url]


def test_prepare_inputs_in_sandbox_skips_single_agtype_field():
    from agency.agschema import _AgSchemaFields

    sandbox = MagicMock()
    data_url = "data:image/jpeg;base64," + "A" * (_AgSchemaFields.input_offload_chars.default + 1)
    inp = agdata(photo=data_url)
    schema = agdata(photo=agimage)
    paths, fields = _agschema(schema).prepare_inputs_in_sandbox(inp, sandbox, "sk")
    sandbox.write_file.assert_not_called()
    assert inp._data["photo"] == data_url


def test_prepare_inputs_in_sandbox_skips_single_agbinary_field():
    from agency.agschema import _AgSchemaFields

    sandbox = MagicMock()
    # After agbinary.prepare() the value is a short sandbox path, but the skip
    # should fire on the schema hint alone — verify with a long string too.
    long_path = "/workspace/inputs/" + "a" * (_AgSchemaFields.input_offload_chars.default + 1)
    inp = agdata(audio=long_path)
    schema = agdata(audio=agbinary)
    paths, fields = _agschema(schema).prepare_inputs_in_sandbox(inp, sandbox, "sk")
    sandbox.write_file.assert_not_called()
    assert paths == []
    assert inp._data["audio"] == long_path


def test_prepare_inputs_in_sandbox_skips_agbinary_list_fields():
    from agency.agschema import _AgSchemaFields

    sandbox = MagicMock()
    long_path = "/workspace/inputs/" + "b" * (_AgSchemaFields.input_offload_chars.default + 1)
    inp = agdata(clips=[long_path, long_path])
    schema = agdata(clips=list[agbinary])
    paths, fields = _agschema(schema).prepare_inputs_in_sandbox(inp, sandbox, "sk")
    sandbox.write_file.assert_not_called()
    assert paths == []
    assert inp._data["clips"] == [long_path, long_path]


def test_prepare_inputs_in_sandbox_processes_agfile_field_via_prepare():
    sandbox = MagicMock()
    inp = agdata(doc="the file contents")
    schema = agdata(doc=agfile)
    paths, fields = _agschema(schema).prepare_inputs_in_sandbox(inp, sandbox, "sk")
    sandbox.write_file.assert_called_once()
    assert len(paths) == 1
    assert "doc" in paths[0]


def test_prepare_inputs_in_sandbox_offloads_agrawstring_when_long():
    from agency.agschema import _AgSchemaFields

    sandbox = MagicMock()
    long_text = "x" * (_AgSchemaFields.input_offload_chars.default + 1)
    inp = agdata(prompt=long_text)
    schema = agdata(prompt=agrawstring)
    paths, fields = _agschema(schema).prepare_inputs_in_sandbox(inp, sandbox, "sk")
    sandbox.write_file.assert_called_once()
    assert len(paths) == 1
    assert "prompt" in fields
    assert "sk_prompt" in inp._data["prompt"]


def test_prepare_inputs_in_sandbox_preserves_short_agrawstring():
    sandbox = MagicMock()
    inp = agdata(prompt="short")
    schema = agdata(prompt=agrawstring)
    paths, fields = _agschema(schema).prepare_inputs_in_sandbox(inp, sandbox, "sk")
    sandbox.write_file.assert_not_called()
    assert paths == []
    assert inp._data["prompt"] == "short"


def test_prepare_inputs_in_sandbox_skips_dict_agtype_field():
    from agency.agschema import _AgSchemaFields

    sandbox = MagicMock()
    long_val = "data:image/jpeg;base64," + "A" * (_AgSchemaFields.input_offload_chars.default + 1)
    inp = agdata(images={"a": long_val})
    schema = agdata(images=dict[str, agimage])
    paths, fields = _agschema(schema).prepare_inputs_in_sandbox(inp, sandbox, "sk")
    sandbox.write_file.assert_not_called()
    assert paths == []
    assert inp._data["images"] == {"a": long_val}


def test_prepare_inputs_in_sandbox_skips_tuple_agtype_field():
    from agency.agschema import _AgSchemaFields

    sandbox = MagicMock()
    long_val = "data:image/jpeg;base64," + "A" * (_AgSchemaFields.input_offload_chars.default + 1)
    inp = agdata(pair=(long_val, "label"))
    schema = agdata(pair=tuple[agimage, str])
    paths, fields = _agschema(schema).prepare_inputs_in_sandbox(inp, sandbox, "sk")
    sandbox.write_file.assert_not_called()
    assert paths == []
    assert inp._data["pair"] == (long_val, "label")


# ---------------------------------------------------------------------------
# _prepare_agtype_inputs — dict and tuple
# ---------------------------------------------------------------------------


def test_prepare_agtype_inputs_dict_agimage_encodes_values(tmp_path):
    import base64

    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    img_a.write_bytes(b"bytes_a")
    img_b.write_bytes(b"bytes_b")
    inp = agdata(images={"x": str(img_a), "y": str(img_b)})
    schema = agdata(images=dict[str, agimage])
    paths, _ = _agschema(schema).prepare_inputs_in_sandbox(inp, MagicMock(), "sk")
    assert paths == []
    assert inp._data["images"]["x"].startswith("data:image/jpeg;base64,")
    assert inp._data["images"]["y"].startswith("data:image/jpeg;base64,")
    assert base64.b64decode(inp._data["images"]["x"].split(",", 1)[1]) == b"bytes_a"


def test_prepare_agtype_inputs_tuple_encodes_agtype_positions(tmp_path):
    import base64

    img = tmp_path / "img.png"
    img.write_bytes(b"png_bytes")
    inp = agdata(pair=(str(img), "label"))
    schema = agdata(pair=tuple[agimage, str])
    _agschema(schema).prepare_inputs_in_sandbox(inp, MagicMock(), "sk")
    result = inp._data["pair"]
    assert result[0].startswith("data:image/png;base64,")
    assert base64.b64decode(result[0].split(",", 1)[1]) == b"png_bytes"
    assert result[1] == "label"


# ---------------------------------------------------------------------------
# _recover_agtype_outputs — list, dict, tuple
# ---------------------------------------------------------------------------


def test_recover_agtype_outputs_list_agfile_reads_each():
    from agency.agdata import agdata as _agdata

    sandbox = MagicMock()
    sandbox.read_file.side_effect = ["content_a", "content_b"]
    result = _agdata(docs=["/workspace/a.txt", "/workspace/b.txt"])
    schema = _agdata(docs=list[agfile])
    _agschema(schema).recover_outputs(result, sandbox)
    assert result._data["docs"] == ["content_a", "content_b"]
    assert sandbox.read_file.call_count == 2


def test_recover_agtype_outputs_dict_agfile_reads_values():
    from agency.agdata import agdata as _agdata

    sandbox = MagicMock()
    sandbox.read_file.side_effect = ["content_x", "content_y"]
    result = _agdata(docs={"x": "/workspace/x.txt", "y": "/workspace/y.txt"})
    schema = _agdata(docs=dict[str, agfile])
    _agschema(schema).recover_outputs(result, sandbox)
    assert result._data["docs"] == {"x": "content_x", "y": "content_y"}


def test_recover_agtype_outputs_tuple_recovers_agtype_positions():
    from agency.agdata import agdata as _agdata

    sandbox = MagicMock()
    sandbox.read_file.return_value = "file_content"
    result = _agdata(pair=["/workspace/out.txt", 42])
    schema = _agdata(pair=tuple[agfile, int])
    _agschema(schema).recover_outputs(result, sandbox)
    assert result._data["pair"][0] == "file_content"
    assert result._data["pair"][1] == 42
    sandbox.read_file.assert_called_once()


# ---------------------------------------------------------------------------
# Deep nesting — _prepare_agtype_inputs and _recover_agtype_outputs
# ---------------------------------------------------------------------------


def test_prepare_agtype_inputs_nested_list_agimage(tmp_path):
    import base64

    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    img_a.write_bytes(b"img_a")
    img_b.write_bytes(b"img_b")
    inp = agdata(batches=[[str(img_a)], [str(img_b)]])
    schema = agdata(batches=list[list[agimage]])
    _agschema(schema).prepare_inputs_in_sandbox(inp, MagicMock(), "sk")
    assert inp._data["batches"][0][0].startswith("data:image/jpeg;base64,")
    assert inp._data["batches"][1][0].startswith("data:image/jpeg;base64,")
    assert base64.b64decode(inp._data["batches"][0][0].split(",", 1)[1]) == b"img_a"


def test_prepare_agtype_inputs_dict_of_list_agimage(tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"img_x")
    inp = agdata(groups={"g": [str(img)]})
    schema = agdata(groups=dict[str, list[agimage]])
    _agschema(schema).prepare_inputs_in_sandbox(inp, MagicMock(), "sk")
    assert inp._data["groups"]["g"][0].startswith("data:image/jpeg;base64,")


def test_recover_agtype_outputs_nested_list_agfile():
    from agency.agdata import agdata as _agdata

    sandbox = MagicMock()
    sandbox.read_file.side_effect = ["content_a", "content_b"]
    result = _agdata(batches=[["/workspace/a.txt"], ["/workspace/b.txt"]])
    schema = _agdata(batches=list[list[agfile]])
    _agschema(schema).recover_outputs(result, sandbox)
    assert result._data["batches"] == [["content_a"], ["content_b"]]


def test_recover_agtype_outputs_dict_of_list_agfile():
    from agency.agdata import agdata as _agdata

    sandbox = MagicMock()
    sandbox.read_file.side_effect = ["content_x", "content_y"]
    result = _agdata(groups={"g": ["/workspace/x.txt", "/workspace/y.txt"]})
    schema = _agdata(groups=dict[str, list[agfile]])
    _agschema(schema).recover_outputs(result, sandbox)
    assert result._data["groups"] == {"g": ["content_x", "content_y"]}


# ---------------------------------------------------------------------------
# Deep nesting — _offload_large_fields skip
# ---------------------------------------------------------------------------


def test_offload_skips_nested_list_agimage():
    from agency.agschema import _AgSchemaFields

    sandbox = MagicMock()
    long_url = "data:image/jpeg;base64," + "A" * (_AgSchemaFields.input_offload_chars.default + 1)
    inp = agdata(batches=[[long_url], [long_url]])
    schema = agdata(batches=list[list[agimage]])
    paths, fields = _agschema(schema).prepare_inputs_in_sandbox(inp, sandbox, "sk")
    sandbox.write_file.assert_not_called()
    assert paths == []


def test_offload_skips_dict_of_list_agimage():
    from agency.agschema import _AgSchemaFields

    sandbox = MagicMock()
    long_url = "data:image/jpeg;base64," + "A" * (_AgSchemaFields.input_offload_chars.default + 1)
    inp = agdata(groups={"g": [long_url]})
    schema = agdata(groups=dict[str, list[agimage]])
    paths, fields = _agschema(schema).prepare_inputs_in_sandbox(inp, sandbox, "sk")
    sandbox.write_file.assert_not_called()
    assert paths == []


# ---------------------------------------------------------------------------
# Fuzz: _prepare_agtype_inputs — random nested schemas
# ---------------------------------------------------------------------------


def test_random_prepare_agtype_inputs_fuzz():
    """100 randomly generated nested schemas: _prepare_agtype_inputs must not
    crash, must preserve plain-Python values unchanged, and must call the
    correct sandbox method for each agfile/agbinary leaf it encounters.
    """
    import random
    from typing import get_origin, get_args

    rng = random.Random(20240628)

    # agimage uses URL passthrough (no sandbox); agfile/agbinary use sandbox writes;
    # agrawstring is a no-op prepare.
    AGTYPE_LEAVES = [agimage, agfile, agbinary, agrawstring]
    PLAIN_LEAVES = [str, int, float, bool]
    ALL_LEAVES = AGTYPE_LEAVES + PLAIN_LEAVES

    def rand_hint(depth):
        if depth >= 3 or (depth > 0 and rng.random() < 0.4 * depth):
            return rng.choice(ALL_LEAVES)
        kind = rng.choice(("list", "dict", "tuple"))
        if kind == "list":
            return list[rand_hint(depth + 1)]
        if kind == "dict":
            return dict[str, rand_hint(depth + 1)]
        n = rng.randint(1, 3)
        inners = tuple(rand_hint(depth + 1) for _ in range(n))
        return tuple[inners] if len(inners) > 1 else tuple[inners[0]]

    def rand_value(hint):
        if hint is bool:
            return rng.choice([True, False])
        if hint is int:
            return rng.randint(-9, 9)
        if hint is float:
            return round(rng.uniform(-9.0, 9.0), 1)
        if hint is str:
            return rng.choice(["hello", "world"])
        if hint is agimage:
            return f"https://example.com/img{rng.randint(0, 9)}.jpg"
        if hint is agfile:
            return rng.choice(["some text", "more text"])
        if hint is agbinary:
            return b"raw bytes"
        if hint is agrawstring:
            return rng.choice(["raw", "text"])
        origin, args = get_origin(hint), get_args(hint)
        if origin is list:
            return [rand_value(args[0]) for _ in range(rng.randint(1, 3))]
        if origin is dict:
            return {f"k{i}": rand_value(args[1]) for i in range(rng.randint(1, 3))}
        if origin is tuple:
            return [rand_value(t) for t in args]
        return None

    def count_leaves(hint, value, *leaf_types):
        """Count how many values at agtype leaf positions match leaf_types."""
        if isinstance(hint, type) and issubclass(hint, tuple(leaf_types)):
            return 1
        origin, args = get_origin(hint), get_args(hint)
        if origin is list and args and isinstance(value, list):
            return sum(count_leaves(args[0], v, *leaf_types) for v in value)
        if origin is dict and len(args) == 2 and isinstance(value, dict):
            return sum(count_leaves(args[1], v, *leaf_types) for v in value.values())
        if origin is tuple and args and isinstance(value, (list, tuple)):
            return sum(count_leaves(ta, v, *leaf_types) for ta, v in zip(args, value))
        return 0

    failures = []
    for trial in range(100):
        hint = rand_hint(0)
        value = rand_value(hint)
        sandbox = MagicMock()

        inp = agdata(f=value)
        schema = agdata(f=hint)

        try:
            paths, _ = _agschema(schema).prepare_inputs_in_sandbox(inp, sandbox, "sk")
        except Exception as exc:
            failures.append(f"[{trial}] raised {exc!r} for hint={hint!r}")
            continue

        # agfile leaves each produce one sandbox.write_file call and one path
        n_agfile = count_leaves(hint, value, agfile)
        if sandbox.write_file.call_count != n_agfile:
            failures.append(
                f"[{trial}] write_file called {sandbox.write_file.call_count}x, "
                f"expected {n_agfile} for hint={hint!r}"
            )

        # agbinary leaves each produce one sandbox.write_file_bytes call and one path
        n_agbinary = count_leaves(hint, value, agbinary)
        if sandbox.write_file_bytes.call_count != n_agbinary:
            failures.append(
                f"[{trial}] write_file_bytes called {sandbox.write_file_bytes.call_count}x, "
                f"expected {n_agbinary} for hint={hint!r}"
            )

        # agimage (URL) and agrawstring and plain types produce no sandbox calls
        if len(paths) != n_agfile + n_agbinary:
            failures.append(
                f"[{trial}] paths={paths!r}, expected {n_agfile + n_agbinary} entries "
                f"for hint={hint!r}"
            )

    assert not failures, f"{len(failures)}/100 trials failed:\n" + "\n".join(failures[:20])


# ---------------------------------------------------------------------------
# Fuzz: _offload_large_fields — agtype fields skipped, plain/agrawstring offloaded
# ---------------------------------------------------------------------------


def test_random_offload_agtype_skip_fuzz():
    """100 randomly generated single-field schemas: _offload_large_fields must
    skip fields whose hint contains any non-agrawstring agtype at any nesting
    depth, and must offload plain str and agrawstring fields when the value
    exceeds _AgSchemaFields.input_offload_chars.default.
    """
    import random
    from typing import get_origin, get_args
    from agency.agschema import _AgSchemaFields

    rng = random.Random(20240629)

    NON_RAW_AGTYPES = [agimage, agfile, agbinary]
    ALL_LEAVES = NON_RAW_AGTYPES + [agrawstring, str, int, float, bool]

    def rand_hint(depth):
        if depth >= 3 or (depth > 0 and rng.random() < 0.4 * depth):
            return rng.choice(ALL_LEAVES)
        kind = rng.choice(("list", "dict", "tuple"))
        if kind == "list":
            return list[rand_hint(depth + 1)]
        if kind == "dict":
            return dict[str, rand_hint(depth + 1)]
        n = rng.randint(1, 3)
        inners = tuple(rand_hint(depth + 1) for _ in range(n))
        return tuple[inners] if len(inners) > 1 else tuple[inners[0]]

    def hint_has_non_raw_agtype(hint) -> bool:
        if isinstance(hint, type) and issubclass(hint, agtype):
            return not issubclass(hint, agrawstring)
        origin, args = get_origin(hint), get_args(hint)
        if origin is list and args:
            return hint_has_non_raw_agtype(args[0])
        if origin is dict and len(args) == 2:
            return hint_has_non_raw_agtype(args[1])
        if origin is tuple and args:
            return any(hint_has_non_raw_agtype(a) for a in args)
        return False

    long_str = "x" * (_AgSchemaFields.input_offload_chars.default + 1)

    failures = []
    for trial in range(100):
        hint = rand_hint(0)
        sandbox = MagicMock()
        inp = agdata(f=long_str)
        schema = agdata(f=hint)

        try:
            paths, fields = _agschema(schema).prepare_inputs_in_sandbox(inp, sandbox, "sk")
        except Exception as exc:
            failures.append(f"[{trial}] raised {exc!r} for hint={hint!r}")
            continue

        has_non_raw = hint_has_non_raw_agtype(hint)

        if has_non_raw:
            # Field uses the agtype.prepare() path, NOT size-offload.
            # It must NOT appear in auto_offloaded_fields (the size-offload list).
            if "f" in fields:
                failures.append(
                    f"[{trial}] field 'f' was size-offloaded for non-raw agtype hint={hint!r}"
                )
            # Value must not have been replaced with a "content saved to" placeholder.
            if isinstance(inp._data.get("f"), str) and "content saved to" in inp._data["f"]:
                failures.append(
                    f"[{trial}] value replaced by size-offload for non-raw agtype hint={hint!r}"
                )
        else:
            # The offloader checks the actual value type, not the hint — our value
            # IS a long string, so write_file must always be called here.
            if not sandbox.write_file.called:
                failures.append(f"[{trial}] write_file NOT called for hint={hint!r}")

    assert not failures, f"{len(failures)}/100 trials failed:\n" + "\n".join(failures[:20])


# ---------------------------------------------------------------------------
# change_config / get_config_copy
# ---------------------------------------------------------------------------


def test_agent_change_config_reaches_llm():
    ag = make_agent()
    ag.change_config(_llm_agconfig({"api_key": "k", "model": "", "temperature": 0.2}))
    assert ag.llm.backend.temperature == 0.2


def test_agent_change_config_clones_given_agconfig():
    ag = make_agent()
    new_cfg = _llm_agconfig({"api_key": "k", "model": "", "temperature": 0.2})
    ag.change_config(new_cfg)
    new_cfg.agllm_backend.temperature = 0.9
    assert ag.llm.backend.temperature == 0.2


def test_agent_change_config_updates_agconfig_attr():
    ag = make_agent()
    new_cfg = _llm_agconfig({"api_key": "k", "model": "", "temperature": 0.2})
    ag.change_config(new_cfg)
    assert ag.agconfig.get("agllm_backend", "temperature") == 0.2
    assert ag.agconfig is not new_cfg  # cloned, not aliased


def test_agent_get_config_copy_returns_clone_not_same_object():
    ag = make_agent()
    copy = ag.get_config_copy()
    assert copy is not ag.agconfig


def test_agent_get_config_copy_reflects_current_values():
    ag = agent(agconfig=_llm_agconfig({"api_key": "k", "model": "", "temperature": 0.7}))
    assert ag.get_config_copy().agllm_backend.temperature == 0.7


def test_agent_get_config_copy_after_change_config_reflects_new_values():
    ag = make_agent()
    ag.change_config(_llm_agconfig({"api_key": "k", "model": "", "temperature": 0.2}))
    assert ag.get_config_copy().agllm_backend.temperature == 0.2


def test_mutating_agent_get_config_copy_does_not_affect_agent():
    ag = agent(agconfig=_llm_agconfig({"api_key": "k", "model": "", "temperature": 0.7}))
    copy = ag.get_config_copy()
    copy.agllm_backend.temperature = 0.1
    assert ag.agconfig.get("agllm_backend", "temperature") == 0.7
