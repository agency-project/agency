"""Unit and integration tests for the chroot sandbox backend.

Tests that actually chroot are marked with @chroot and skipped automatically
when unprivileged user namespaces aren't usable on this host (see
agency.agsandbox_backends.chroot.chroot_available())."""

from __future__ import annotations

import shutil
import uuid

import pytest

from agency.agsandbox_backends.chroot import (
    chroot_available,
    _ChrootBackend,
    _CHROOT_STATE_ROOT,
    _sanitize_tag,
)

chroot = pytest.mark.skipif(
    not chroot_available(), reason="unprivileged user namespaces not usable"
)


def _make_backend(**kwargs):
    name = f"chroot-test-{uuid.uuid4().hex[:8]}"
    defaults = dict(
        agname=name,
        name=name,
        checkpoint_image=None,
        mounts={},
        agconfig=None,
    )
    defaults.update(kwargs)
    return _ChrootBackend(defaults.pop("agname"), **defaults)


@pytest.fixture(autouse=True)
def _clean_state_root():
    shutil.rmtree(_CHROOT_STATE_ROOT, ignore_errors=True)
    yield
    shutil.rmtree(_CHROOT_STATE_ROOT, ignore_errors=True)


# ---------------------------------------------------------------------------
# _sanitize_tag
# ---------------------------------------------------------------------------


class TestSanitizeTag:
    def test_replaces_slash(self):
        assert "/" not in _sanitize_tag("agency/lifecycle-foo")

    def test_replaces_colon(self):
        assert ":" not in _sanitize_tag("agency/lifecycle-foo:v1")

    def test_distinct_tags_stay_distinct(self):
        assert _sanitize_tag("a/b") != _sanitize_tag("a-b")


# ---------------------------------------------------------------------------
# _own_host_pids -- scopes release_gpu()'s straggler wait to this sandbox's
# own processes (see agresources._wait_for_gpu_clear's own_pids param).
# Chroot processes run directly on the host (no PID namespace to translate
# through, unlike _ContainerBackendBase's version), so _watched_pids is
# already the right PID space -- no real chroot jail needed to test this.
# ---------------------------------------------------------------------------


class TestOwnHostPids:
    def test_matches_watched_pids(self):
        sb = _make_backend()
        sb._watched_pids = {111: 0.0, 222: 0.0}
        assert sb._own_host_pids() == {111, 222}

    def test_empty_when_no_watched_pids(self):
        sb = _make_backend()
        assert sb._own_host_pids() == set()

    def test_returns_a_copy_not_a_live_view(self):
        """Mutating _watched_pids afterward must not retroactively change an
        already-returned snapshot out from under a caller mid-wait."""
        sb = _make_backend()
        sb._watched_pids = {111: 0.0}
        result = sb._own_host_pids()
        sb._watched_pids[222] = 0.0
        assert result == {111}


# ---------------------------------------------------------------------------
# _ChrootBackend -- functional (requires unprivileged userns + chroot)
# ---------------------------------------------------------------------------


@chroot
class TestChrootBackendExec:
    def test_exec_runs_and_returns_output(self):
        sb = _make_backend()
        try:
            out, rc = sb.exec("echo hello-chroot")
            assert rc == 0
            assert "hello-chroot" in out
        finally:
            sb.destroy()

    def test_exec_default_workdir_is_workspace(self):
        sb = _make_backend()
        try:
            out, rc = sb.exec("pwd")
            assert rc == 0
            assert out.strip() == "/workspace"
        finally:
            sb.destroy()

    def test_cannot_see_host_home_directory(self):
        """The jail must not expose the host's real filesystem outside the
        bind-mounted base dirs and workspace."""
        sb = _make_backend()
        try:
            out, rc = sb.exec("ls /home 2>&1; echo RC=$?")
            assert "No such file or directory" in out or "RC=2" in out
        finally:
            sb.destroy()

    def test_readonly_base_dir_rejects_writes(self):
        sb = _make_backend()
        try:
            out, rc = sb.exec("touch /bin/should-not-exist 2>&1; echo RC=$?")
            assert "RC=0" not in out or rc != 0
        finally:
            sb.destroy()

    def test_python_is_usable_inside_jail(self):
        sb = _make_backend()
        try:
            out, rc = sb.exec("python3 -c 'print(1+1)' 2>&1 || echo NO_PYTHON")
            assert "NO_PYTHON" not in out
        finally:
            sb.destroy()


@chroot
class TestChrootBackendFileIO:
    def test_write_then_read_file_round_trips(self):
        sb = _make_backend()
        try:
            sb.write_file("/workspace/hello.txt", "hello world\n")
            assert sb.read_file("/workspace/hello.txt") == "hello world\n"
        finally:
            sb.destroy()

    def test_write_file_bytes_round_trips(self):
        sb = _make_backend()
        try:
            data = bytes([0x00, 0x01, 0xFF, 0x10])
            sb.write_file_bytes("/workspace/bin.dat", data)
            assert sb.read_file_bytes("/workspace/bin.dat") == data
        finally:
            sb.destroy()

    def test_read_missing_file_raises_file_not_found(self):
        sb = _make_backend()
        try:
            with pytest.raises(FileNotFoundError):
                sb.read_file("/workspace/does-not-exist.txt")
        finally:
            sb.destroy()


@chroot
class TestChrootBackendLifecycle:
    def test_ensure_started_does_not_wipe_workspace_across_worker_processes(self):
        """Regression test: tool calls with run_in_subprocess=True (the
        default) each get a fresh cloudpickled copy of the backend, so
        _started is False in every worker's own copy regardless of what an
        earlier worker already did. _ensure_started() must use the
        workspace directory's existence on disk as ground truth, not
        self._started -- otherwise every worker's first touch re-runs
        _materialize_workspace() and wipes out whatever a *different*
        worker already wrote (this exact bug shipped and was caught against
        a real chroot-backed run: a file written by one tool dispatch was
        gone by the time the very next dispatch tried to read it back)."""
        import pickle

        orig = _make_backend()
        try:
            worker1 = pickle.loads(pickle.dumps(orig))
            worker1.write_file("/workspace/inputs/full_text_123.txt", "important content\n")

            worker2 = pickle.loads(pickle.dumps(orig))
            assert worker2._started is False  # fresh copy, exactly like a real worker process
            content = worker2.read_file("/workspace/inputs/full_text_123.txt")
            assert content == "important content\n"
        finally:
            orig.destroy()

    def test_stop_commit_true_commits_even_when_this_process_never_started(self):
        """Same root cause as above, for stop(): the orchestrating process
        calls stop(commit=True) on a sandbox whose actual workspace content
        was written entirely by worker-process tool calls, which never
        touch the orchestrator's own _started flag. stop() must still find
        and commit that real work rather than silently skip committing."""
        import pickle

        orig = _make_backend()
        restored = None
        try:
            worker = pickle.loads(pickle.dumps(orig))
            worker.write_file("/workspace/data.txt", "from-worker\n")
            assert orig._started is False  # orchestrator's own copy never ran an exec

            orig.stop(commit=True)
            assert orig._checkpoint_image is not None, "stop(commit=True) must have committed"

            restored = _make_backend(checkpoint_image=orig._checkpoint_image)
            assert restored.read_file("/workspace/data.txt") == "from-worker\n"
        finally:
            if restored is not None:
                restored.destroy()
            if orig._checkpoint_image:
                _ChrootBackend.delete_image(orig._checkpoint_image, force=True)
            orig.destroy()

    def test_commit_then_new_backend_restores_content(self):
        tag = f"agency/lifecycle-test-{uuid.uuid4().hex[:8]}"
        sb1 = _make_backend()
        try:
            sb1.write_file("/workspace/marker.txt", "checkpoint\n")
            assert sb1.commit(tag) is True
        finally:
            sb1.destroy()

        sb2 = _make_backend(checkpoint_image=tag)
        try:
            assert sb2.read_file("/workspace/marker.txt") == "checkpoint\n"
        finally:
            sb2.destroy()
            _ChrootBackend.delete_image(tag, force=True)

    def test_commit_returns_false_when_never_started(self):
        sb = _make_backend()
        assert sb.commit("agency/never-started") is False
        sb.destroy()

    def test_stop_commit_false_discards_dirty_state(self):
        tag = f"agency/lifecycle-test-{uuid.uuid4().hex[:8]}"
        sb = _make_backend()
        try:
            sb.write_file("/workspace/good.txt", "good\n")
            sb.stop(commit=True)
            sb._ensure_started()
            sb.write_file("/workspace/dirty.txt", "dirty\n")
            sb.stop(commit=False)
            sb._ensure_started()
            content = sb.read_file("/workspace/good.txt")
            assert content == "good\n"
            with pytest.raises(FileNotFoundError):
                sb.read_file("/workspace/dirty.txt")
        finally:
            sb.destroy()
            _ChrootBackend.delete_image(sb._checkpoint_image or tag, force=True)

    def test_restore_materializes_snapshot(self):
        tag = f"agency/lifecycle-test-{uuid.uuid4().hex[:8]}"
        sb = _make_backend()
        try:
            sb.write_file("/workspace/a.txt", "aaa\n")
            sb.commit(tag)
            sb.write_file("/workspace/b.txt", "bbb\n")
            sb.restore(tag)
            assert sb.read_file("/workspace/a.txt") == "aaa\n"
            with pytest.raises(FileNotFoundError):
                sb.read_file("/workspace/b.txt")
        finally:
            sb.destroy()
            _ChrootBackend.delete_image(tag, force=True)

    def test_destroy_removes_jail_directory(self):
        sb = _make_backend()
        sb.exec("true")
        root = sb._root
        assert root.exists()
        sb.destroy()
        assert not root.exists()

    def test_cross_agent_workspace_isolation(self):
        sb1 = _make_backend()
        sb2 = _make_backend()
        try:
            sb1.write_file("/workspace/only-in-1.txt", "secret\n")
            out, rc = sb2.exec("ls /workspace")
            assert "only-in-1.txt" not in out
        finally:
            sb1.destroy()
            sb2.destroy()

    def test_update_limits_is_a_no_op(self):
        sb = _make_backend()
        try:
            sb.update_limits(cpus=2.0, memory="4g")  # must not raise
        finally:
            sb.destroy()


@chroot
class TestChrootImageHelpers:
    def test_tag_image_copies_snapshot(self):
        src_tag = f"agency/src-{uuid.uuid4().hex[:8]}"
        dest_tag = f"agency/dest-{uuid.uuid4().hex[:8]}"
        sb = _make_backend()
        try:
            sb.write_file("/workspace/x.txt", "x\n")
            sb.commit(src_tag)
            _ChrootBackend.tag_image(src_tag, dest_tag)
            sb2 = _make_backend(checkpoint_image=dest_tag)
            try:
                assert sb2.read_file("/workspace/x.txt") == "x\n"
            finally:
                sb2.destroy()
        finally:
            sb.destroy()
            _ChrootBackend.delete_image(src_tag, force=True)
            _ChrootBackend.delete_image(dest_tag, force=True)

    def test_export_then_import_round_trips(self):
        tag = f"agency/export-{uuid.uuid4().hex[:8]}"
        sb = _make_backend()
        try:
            sb.write_file("/workspace/e.txt", "exported\n")
            sb.commit(tag)
            blob = _ChrootBackend.export_image(tag, 30)
            assert isinstance(blob, bytes) and len(blob) > 0
            _ChrootBackend.delete_image(tag, force=True)
            _ChrootBackend.import_image(blob, 30)
            sb2 = _make_backend(checkpoint_image=tag)
            try:
                assert sb2.read_file("/workspace/e.txt") == "exported\n"
            finally:
                sb2.destroy()
        finally:
            sb.destroy()
            _ChrootBackend.delete_image(tag, force=True)

    def test_export_missing_tag_raises(self):
        with pytest.raises(FileNotFoundError):
            _ChrootBackend.export_image(f"agency/no-such-{uuid.uuid4().hex[:8]}", 30)


# ---------------------------------------------------------------------------
# Facade integration -- agSandbox(backend="chroot")
# ---------------------------------------------------------------------------


@chroot
class TestFacadeWithChrootBackend:
    def _make_sandbox(self, **kwargs):
        from agency.agconfig import agConfig
        from agency.agsandbox import agSandbox
        from agency.agsandbox_backends import agSandboxBackendConfig

        cfg = agConfig(agSandboxBackendConfig(backend="chroot"))
        uid = str(uuid.uuid4())
        return agSandbox(uid, agconfig=cfg, **kwargs)

    def test_facade_selects_chroot_backend(self):
        sb = self._make_sandbox()
        try:
            assert isinstance(sb._backend, _ChrootBackend)
        finally:
            sb.destroy()

    def test_facade_exec_and_file_io(self):
        sb = self._make_sandbox()
        try:
            out, rc = sb.exec("echo via-facade")
            assert rc == 0 and "via-facade" in out
            sb.write_file("/workspace/f.txt", "f-content\n")
            assert sb.read_file("/workspace/f.txt") == "f-content\n"
        finally:
            sb.destroy()

    def test_facade_stop_commit_true_sets_checkpoint_image(self):
        sb = self._make_sandbox()
        try:
            sb.exec("true")
            sb.stop(commit=True)
            assert sb._checkpoint_image is not None
        finally:
            sb.destroy()

    def test_facade_fork_preserves_checkpoint_content(self):
        sb = self._make_sandbox()
        fork_sb = None
        try:
            sb.write_file("/workspace/parent.txt", "parent-data\n")
            sb.stop(commit=True)
            fork_sb = sb.fork(str(uuid.uuid4()))
            assert isinstance(fork_sb._backend, _ChrootBackend)
            assert fork_sb.read_file("/workspace/parent.txt") == "parent-data\n"
        finally:
            sb.destroy()
            if fork_sb is not None:
                fork_sb.destroy()


# ---------------------------------------------------------------------------
# agent.py save()/load() wiring -- a checkpoint must round-trip through the
# same backend kind that produced it, not be silently assumed to be a
# docker/podman image tag.
# ---------------------------------------------------------------------------


@chroot
class TestAgentSaveLoadWithChrootBackend:
    def _make_agconfig(self):
        from agency.agconfig import agConfig
        from agency.agsandbox_backends import agSandboxBackendConfig

        return agConfig(
            agSandboxBackendConfig(backend="chroot"),
            {"agllm_backend": {"api_key": "k", "model": "m"}},
        )

    def test_save_records_chroot_image_kind(self, tmp_path):
        import json
        import tarfile
        from agency.agsandbox import agSandbox
        from agency.agent import agent
        from agency.agname import agname

        cfg = self._make_agconfig()
        ag = agent(agconfig=cfg)
        try:
            ag.sandbox = agSandbox(ag.agname, agconfig=cfg)
            ag.sandbox.write_file("/workspace/marker.txt", "data\n")
            ag.sandbox.stop(commit=True)

            ckpt = tmp_path / "agent.ckpt"
            ag.save(ckpt)

            with tarfile.open(ckpt, "r:gz") as tar:
                state = json.loads(tar.extractfile("state.json").read())
            assert state["sandbox_image_kind"] == "chroot"
        finally:
            if ag.sandbox is not None:
                ag.sandbox.destroy()
            agname._allocated.discard(str(ag.agname))

    def test_save_then_load_restores_chroot_workspace(self, tmp_path):
        from agency.agsandbox import agSandbox
        from agency.agent import agent
        from agency.agname import agname

        cfg = self._make_agconfig()
        ag = agent(agconfig=cfg)
        ag2 = None
        try:
            ag.sandbox = agSandbox(ag.agname, agconfig=cfg)
            ag.sandbox.write_file("/workspace/marker.txt", "checkpointed-via-agent\n")
            ag.sandbox.stop(commit=True)

            ckpt = tmp_path / "agent.ckpt"
            ag.save(ckpt)
            saved_agname = str(ag.agname)
            ag.sandbox.destroy()
            agname._allocated.discard(saved_agname)

            ag2 = agent.load(ckpt, agconfig=cfg)
            assert isinstance(ag2.sandbox._backend, _ChrootBackend)
            content = ag2.sandbox.read_file("/workspace/marker.txt")
            assert content == "checkpointed-via-agent\n"
        finally:
            if ag2 is not None and ag2.sandbox is not None:
                ag2.sandbox.destroy()
                agname._allocated.discard(str(ag2.agname))


# ---------------------------------------------------------------------------
# Real ProcessPoolExecutor dispatch -- the actual code path a live agent run
# uses (run_in_subprocess=True, the default), as opposed to calling a
# backend's methods directly in-process. This is what surfaced the
# _ensure_started()/stop() worker-vs-main-process bugs fixed above: calling
# a tool's .fn() directly, or driving the backend object in one process,
# never exercises cloudpickle sending a fresh copy of the sandbox to a
# ProcessPoolExecutor worker for every single call.
# ---------------------------------------------------------------------------


@chroot
class TestChrootSandboxedToolsDispatch:
    def _make_sandbox(self):
        from agency.agconfig import agConfig
        from agency.agsandbox import agSandbox
        from agency.agsandbox_backends import agSandboxBackendConfig

        cfg = agConfig(agSandboxBackendConfig(backend="chroot"))
        return agSandbox(str(uuid.uuid4()), agconfig=cfg)

    def test_files_persist_across_process_pool_tool_calls(self):
        """Files written by the write tool in one worker process must be
        readable by the read tool in a subsequent, separately-dispatched
        worker process call -- the chroot-backend analogue of
        test_docker.py's identically-named container-backend test."""
        from agency.agdata import agdata, agerror
        from agency.tools import make_sandboxed_tools

        sb = self._make_sandbox()
        tools = {t.name: t for t in make_sandboxed_tools(sb)}
        try:
            w = tools["write"](agdata(file_path="/workspace/cross.txt", content="cross-worker\n"))
            assert not isinstance(w, agerror), f"write failed: {w}"
            r = tools["read"](agdata(file_path="/workspace/cross.txt"))
            assert not isinstance(r, agerror), f"read failed after cross-worker write: {r}"
            assert "cross-worker" in r.content
        finally:
            sb.destroy()

    def test_bash_then_read_across_process_pool_tool_calls(self):
        """A file created by the bash tool in one worker process must be
        readable by the read tool in the next, separately-dispatched call --
        matches the exact real-world shape of the bug (agfile.prepare()'s
        sandbox.write_file() in one dispatch, the read tool in the next)."""
        from agency.agdata import agdata, agerror
        from agency.tools import make_sandboxed_tools

        sb = self._make_sandbox()
        tools = {t.name: t for t in make_sandboxed_tools(sb)}
        try:
            b = tools["bash"](
                agdata(
                    command="mkdir -p /workspace/inputs && echo hi > /workspace/inputs/full_text_1.txt"
                )
            )
            assert not isinstance(b, agerror), f"bash failed: {b}"
            r = tools["read"](agdata(file_path="/workspace/inputs/full_text_1.txt"))
            assert not isinstance(r, agerror), f"read failed after cross-worker bash write: {r}"
            assert "hi" in r.content
        finally:
            sb.destroy()

    def test_agent_run_offloads_and_reads_back_large_input(self):
        """End-to-end: a real agskill run whose input schema triggers
        agschema's size-based offload (sandbox.write_file in the prepare
        step, executed in the calling thread) followed by the LLM calling
        the read tool (a separate ProcessPoolExecutor dispatch) to read it
        back -- the exact real-world flow that surfaced this bug."""
        from agency.agconfig import agConfig
        from agency.agdata import agdata
        from agency.agskill import agskill
        from agency.agschema import agSchemaConfig
        from agency.agsandbox_backends import agSandboxBackendConfig
        from agency.agent import agent

        cfg = agConfig(
            agSandboxBackendConfig(backend="chroot"),
            agSchemaConfig(input_offload_chars=10),  # force offload for a short string
            {"agllm_backend": {"api_key": "k", "model": "m"}},
        )

        skill = agskill(name="repro", system_prompt="", input_schema=agdata(text=str))

        def fake_execute_react(ag, prev_ctx, skill_input, max_steps=None, **_):
            # Replicate execute_react()'s real step 2 (input prep) explicitly,
            # since replacing execute_react wholesale also removes that step
            # -- it isn't called automatically just because ag.sandbox exists.
            skill.input_schema.prepare_inputs_in_sandbox(
                skill_input,
                ag.sandbox,
                skill.name,
                context_limit=ag.llm.context_limit,
                agconfig=ag.agconfig,
            )
            # skill_input.text has now been offloaded to a path reference --
            # read it back via the same tool-dispatch path a real ReAct loop
            # (running against an LLM) would use.
            from agency.tools import make_sandboxed_tools

            tools = {t.name: t for t in make_sandboxed_tools(ag.sandbox)}
            path = skill_input.text.split("saved to ")[1].split(" —")[0]
            r = tools["read"](agdata(file_path=path))
            return agdata(answer=r.content), prev_ctx, []

        skill.execute_react = fake_execute_react

        ag = agent(agconfig=cfg)
        try:
            long_text = "x" * 100
            result = ag.run(skill, agdata(text=long_text)).answer
            assert long_text in result
        finally:
            if ag.sandbox is not None:
                ag.sandbox.destroy()
