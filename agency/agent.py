from __future__ import annotations
import io
import json
import os
import tarfile
import threading
import uuid as _uuid_mod
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from .utils.agutil import _DEFAULT_LOG_DIR

# Global weak registry of all live agent instances.
_live_agents: "weakref.WeakSet[agent]" = weakref.WeakSet()


def _llm_config_snapshot(agconfig: "agConfig") -> dict:
    """Backend config fields for logging/checkpointing, minus the secret
    api_key -- a fresh, cheap (no network I/O) construction each call, not a
    persisted instance."""
    return {k: v for k, v in agllm.for_config(agconfig).as_dict().items() if k != "api_key"}


from .agdata import agdata
from .agcontext import agcontext
from .observability.agdatalogger import agDataLogger, agDataLoggerConfigs, _ts
from .orchestrator import get_orchestrator
from .sandbox.agsandbox import agSandbox, agSandboxConfig
from .sandbox import agSandboxBackendConfig
from .llm.agllm import agllm
from .agconfig import agConfig, DynamicConfigParam, _AgConfigViewBase

from .agname import agname as _agname  # [REFACTOR] Why underscore?
from .observability.profiler import agprof
from ._agent_control import AgentControl
from ._submission import CloseHandle, Invocation, MessageSubmission, Submission

if TYPE_CHECKING:
    from .engine import AgentEngine


# Exists only to register agent's config fields (via __set_name__ at import
# time). Reads use a throwaway instance -- _AgAgentFields(agconfig) -- since
# these values are needed in a classmethod (load()) and an instance method
# (save()) that doesn't otherwise inherit from this class.
class _AgAgentFields:
    checkpoint_save_timeout_s = DynamicConfigParam("agent", default=600)
    checkpoint_load_timeout_s = DynamicConfigParam("agent", default=600)
    harness = DynamicConfigParam(
        "agent", default="native"
    )  # Looked up via agharness_backend.for_config() by AgentEngine. "native" runs
    # agency's own react loop as a persistent in-container process
    # (agharness_backends/native.py); any other value names an external
    # harness engine (claude_code/codex/opencode/grok).

    def __init__(self, agconfig=None) -> None:
        self._agconfig = agconfig


class agAgentConfig(_AgConfigViewBase):
    """View over an agConfig for pre-setting agent tunables in one call::

        cfg = agConfig(agAgentConfig(checkpoint_save_timeout_s=300))

    See `_AgConfigViewBase` in agconfig.py for the shared mechanics.
    """

    _OWNER = "agent"


# [REFACTOR] Remove
def _classvar_or_agconfig(agconfig: "agConfig | None", name: str, classvar_default):
    """Resolve one of agent's own knobs (log_dir, output_dir, ...): the plain
    ClassVar default, optionally overridden by agconfig.

    Deliberately NOT a ConfigParam descriptor: agent's docstring documents
    ``agent.log_dir = Path(...)`` as a supported class-level override, and
    assigning to a class attribute that holds a descriptor replaces the
    descriptor itself (silently breaking it for every future instance) --
    so these fields stay plain ClassVars, resolved via this helper instead.
    """
    return classvar_default if agconfig is None else agconfig.get("agent", name, classvar_default)


# [REFACTOR] Check how it works
class agent:
    """Orchestrator that maintains shared history and delegates to named agskills.

    An agent holds all runtime state (LLM config, conversation context, sandbox,
    logging infrastructure) and delegates execution to agskill objects.

    agent.run(skill, input)
        Always non-blocking.  Returns an Invocation immediately.  Calls on the
        same agent are serialized through the context chain.  Calls on
        different agents (forks) run concurrently.

    agent.fork(existing_agent)
        Blocks until the source agent's in-flight task completes, then
        deep-copies the resolved history and snapshots the parent's container.

    Class-level configuration (set once before creating agents)::

        agent.log_dir        = Path("runs/logs")
        agent.ping_interval_s = 300
        agent.poll_interval_s = 5
        agent.max_outer_iters = 144

    The GPU/CPU/memory pool is no longer a class-level override on ``agent``
    -- it's owned by the process-wide orchestrator, constructed eagerly at
    import time. Override it via ``get_orchestrator().agresource_pool = ...``
    instead.
    """

    # [REFACTOR] Move to agconfig
    log_dir: ClassVar[Path | None] = None
    output_dir: ClassVar[Path | None] = None
    ping_interval_s: ClassVar[int] = 300
    poll_interval_s: ClassVar[int] = 5
    max_outer_iters: ClassVar[int] = 144

    # Tier-1-style fallback: agent(agconfig=...) not given -> use this if set.
    # Same "set once before creating agents" convention as the ClassVars
    # above, so scripts that construct agents directly (agent(agname=...),
    # with no agconfig= kwarg) still pick up a run-wide agConfig.
    default_agconfig: "ClassVar[agConfig | None]" = None  # [REFACTOR] Remove

    def __init__(
        self,
        agname: str | None = None,
        *,
        sandbox: "agSandbox | None" = None,
        agconfig: "agConfig | None" = None,
        harness: "str | None" = None,
    ):
        with agprof.span("agent:create"):
            self._initialize(agname, sandbox, agconfig, harness)

    # [REFACTOR] Why separate?
    def _initialize(
        self,
        agname: "str | None",
        sandbox: "agSandbox | None",
        agconfig: "agConfig | None",
        harness: "str | None",
    ) -> None:
        _src_agconfig = agconfig if agconfig is not None else agent.default_agconfig

        if _src_agconfig is None or not _src_agconfig.data.get("agllm_backend"):
            from .agteam import _active_team as _at

            _t = _at.get(None)
            if (
                _t is not None and _t.agconfig is not None and _t.agconfig.data.get("agllm_backend")
            ):  # [REFACTOR] Why do we have auto team-config inheritance only when agllm_backend exists?
                # Adopt the team's agconfig outright (not just for the LLM
                # fields) -- log_dir/output_dir/sandbox settings etc. should
                # also come from it, matching "agents inherit the team's
                # agconfig automatically" (see agteam's docstring).
                _src_agconfig = _t.agconfig
            else:
                raise TypeError(
                    "agent() requires an agconfig with LLM fields set "
                    "(e.g. cfg.agllm_backend.model = ...) when called "
                    "outside an agteam context"
                )

        # Cloned so this agent's own agconfig is independent of whatever
        # source it was built from (an explicit agconfig=, agent.default_agconfig,
        # or the active agteam's agconfig) -- mutating that source afterward
        # must not silently change an already-constructed agent. Use
        # ag.change_config(new_cfg) to change it live -- see that method.
        self.agconfig: "agConfig | None" = (
            _src_agconfig.clone() if _src_agconfig is not None else None
        )  # [REFACTOR] Should use a single setter method, also no default agconfig

        self.agname: _agname = _agname.allocate_agname(agname, prefix="agent")
        self._parent_agent_id: "str | None" = (
            None  # [REFACTOR]  Why do we need to keep reference of parent agent id?
        )

        self.harness: str = (
            harness if harness is not None else _AgAgentFields(self.agconfig).harness
        )  # [REFACTOR] Change to config only
        self.context: agcontext = agcontext()
        # Sandbox is created lazily on first skill run; container provisioning
        # is expensive and agents may be constructed without ever running a skill.
        self.sandbox: "agSandbox | None" = sandbox
        self._owns_sandbox = sandbox is None
        self.engine: "AgentEngine | None" = None

        from .agteam import _active_team

        _team = _active_team.get(None)
        if _team is not None:
            _team._agents.add(self)

        team_name = _team.team_name if _team is not None else None

        _llm_config = _llm_config_snapshot(self.agconfig)
        team_tag = f"  team={team_name}" if team_name else ""
        self._finish_construction(
            event_type="agent_created",
            event_payload={
                "agname": self.agname,
                "team": team_name,
                "llm_config": _llm_config,
            },
            term_message=(
                f"[{self.agname}] CREATED  model={_llm_config.get('model') or '?'}{team_tag}"
            ),
            reuse_data_logger_configs=True,
        )

    def _finish_construction(
        self,
        *,
        event_type: str,
        event_payload: dict,
        term_message: str,
        reuse_data_logger_configs: bool = False,
    ) -> None:
        """Shared tail of _initialize()/fork()/load(): data logger setup,
        initial runtime state, live registry, and the construction-event log
        -- everything that only needs agname/agconfig already resolved,
        regardless of how they were resolved."""
        _log_dir_val = _classvar_or_agconfig(self.agconfig, "log_dir", agent.log_dir)
        log_dir = Path(_log_dir_val) if _log_dir_val is not None else _DEFAULT_LOG_DIR

        data_logger_configs = (
            self.agconfig.__dict__.get("agDataLoggerConfigs") if reuse_data_logger_configs else None
        )
        if data_logger_configs is None:
            self.agconfig.agDataLoggerConfigs = agDataLoggerConfigs(
                db_path=str(log_dir / f"{self.agname}_data.sqlite3")
            )
        self.data_logger = agDataLogger(
            self.agconfig, default_name=str(self.agname), default_object="agent"
        )
        self.data_logger.start()
        from .observability.agdatalogger import resolve_global_db_path

        self._orchestrator = get_orchestrator(
            self.agconfig,
            default_db_path=resolve_global_db_path(log_dir),
        )
        self._submission_lock = threading.RLock()
        initial_sequence = max(
            (
                int(entry.get("sequence", 0))
                for entry in self.context.retained_messages
                if isinstance(entry, dict)
            ),
            default=0,
        )
        self._control = AgentControl(initial_sequence=initial_sequence)
        self._submissions: set[Submission] = set()
        self._next_submission_id = 1
        self._active_operations = 0
        self._cleanup_lock = threading.Lock()
        self._cleanup_scheduled = False
        self._cleanup_done = False
        self._close_handle = CloseHandle()
        self._current_state = "agent_idle"
        _live_agents.add(self)

        self.data_logger.record_event(
            type=event_type, payload=event_payload, term_message=term_message
        )
        self._register_global_catalog(event_payload.get("team"))
        self.record_state("agent_idle")
        self.change_config(self.agconfig)

    def _register_global_catalog(self, team_name: "str | None") -> None:
        """Publish only agent identity and its detailed database location globally."""
        try:
            agent_db_path = Path(self.data_logger._configs.db_path)
            self._orchestrator.data_logger.record_event(
                "agent_registered",
                {"db_path": str(agent_db_path.resolve()), "team": team_name},
                name=str(self.agname),
                object="agent",
                update_latest_snapshot=True,
            )
        except Exception as exc:
            print(f"[agent] WARNING: global catalog registration failed for {self.agname}: {exc}")

    def _submission_finished(self, submission: Submission) -> None:
        with self._orchestrator._event_cond:
            with self._submission_lock:
                self._submissions.discard(submission)
        self._maybe_cleanup_destroyed()

    @contextmanager
    def _operation_lease(self, operation: str, *, allow_destroyed: bool = False):
        """Keep runtime resources alive for one already-admitted host operation."""
        with self._orchestrator._event_cond:
            with self._submission_lock:
                if not allow_destroyed:
                    self._control.assert_alive(operation)
                self._active_operations += 1
        try:
            yield
        finally:
            with self._orchestrator._event_cond:
                with self._submission_lock:
                    self._active_operations -= 1
            self._maybe_cleanup_destroyed()

    def _maybe_cleanup_destroyed(self) -> None:
        schedule = False
        with self._orchestrator._event_cond:
            with self._submission_lock:
                if (
                    self._control.is_destroyed()
                    and self._control.active_invocation() is None
                    and self._active_operations == 0
                    and not self._submissions
                    and not self._cleanup_scheduled
                ):
                    self._cleanup_scheduled = True
                    schedule = True
        if schedule:
            try:
                thread = agprof.spawn_traced(self._finalize_destroy, daemon=True)
                thread.name = f"agency-destroy-{self.agname}"
                thread.start()
            except BaseException:
                self._finalize_destroy()

    def _finalize_destroy(self) -> None:
        with self._cleanup_lock:
            if self._cleanup_done:
                return
            self._cleanup_done = True
        try:
            self.data_logger.record_event(
                type="agent_destroyed",
                payload={"agname": self.agname},
                term_message=f"[{self.agname}] DESTROYED",
                flush=True,
            )
        except Exception as exc:
            print(f"[agent] WARNING: destroy log failed for {self.agname}: {exc}")
        if self._owns_sandbox and self.sandbox is not None:
            try:
                self.sandbox.destroy()
            except Exception as exc:
                print(f"[agent] WARNING: sandbox cleanup failed for {self.agname}: {exc}")
        try:
            self.data_logger.stop()
        except Exception as exc:
            print(f"[agent] WARNING: logger cleanup failed for {self.agname}: {exc}")
        self._control.mark_destroyed()
        self._close_handle._settle()

    def change_config(self, agconfig: "agConfig") -> None:
        with self._operation_lease("change config"):
            self.agconfig = agconfig.clone()
            self.data_logger.set_config(self.agconfig)
            if self.sandbox is not None:
                self.sandbox.change_config(self.agconfig)
            if self.engine is not None:
                self.engine.set_config(self.agconfig)
            self.data_logger.record_event(
                type="agent_config",
                payload=self.agconfig.dynamic_snapshot(),
                update_latest_snapshot=True,
            )

    def get_config_copy(self) -> "agConfig | None":
        """Return a clone of this agent's agconfig, or None if it has none."""
        return self.agconfig.clone() if self.agconfig is not None else None

    # ------------------------------------------------------------------
    # Properties # [REFACTOR] Why as properties?
    # ------------------------------------------------------------------

    @property
    def ctx(self) -> agcontext:
        """Compatibility alias for the one authoritative ``context`` chain."""
        return self.context

    @ctx.setter
    def ctx(self, value: agcontext) -> None:
        self.context = value

    @property
    def output_path(self) -> Path | None:
        out_dir = _classvar_or_agconfig(self.agconfig, "output_dir", agent.output_dir)
        if out_dir is None:
            return None
        return Path(out_dir) / self.agname

    @property
    def container_output_path(self) -> str | None:
        out_dir = _classvar_or_agconfig(self.agconfig, "output_dir", agent.output_dir)
        if out_dir is None:
            return None
        return f"/agent_output/{self.agname}"

    @property
    def history(self) -> agdata:
        """Return the committed transcript after this agent becomes idle."""
        with self._operation_lease("read history", allow_destroyed=True):
            return agdata(messages=self.context.get_resolved_transcript())

    @history.setter
    def history(self, value: agdata) -> None:
        with self._operation_lease("replace history"):
            messages = value.to_dict().get("messages", [])
            while True:
                with self._orchestrator._event_cond:
                    with self._submission_lock:
                        current = self.context
                current.resolve_prev_dependencies()
                with self._orchestrator._event_cond:
                    with self._submission_lock:
                        if self.context is current:
                            current.set_transcript(messages)
                            return

    def record_state(
        self, state: str, skill: "str | None" = None, tool: "str | None" = None
    ) -> None:
        self._current_state = state
        self.data_logger.record_event(
            type="agent_state",
            payload={"state": state, "skill": skill, "tool": tool},
            update_latest_snapshot=True,
            flush=True,
        )

    # ------------------------------------------------------------------
    # Pause / resume
    # ------------------------------------------------------------------

    def _notify_invocation_control(self, invocation: Invocation) -> None:
        self._orchestrator.notify_invocation_control(invocation)

    def _record_context_notice(self, context: agcontext) -> int:
        """Append the typed rollback notice retained after an ordinary failure."""
        last_retained_sequence = max(
            (
                int(entry.get("sequence", 0))
                for entry in context.retained_messages
                if isinstance(entry, dict)
            ),
            default=0,
        )
        self._control.ensure_sequence_at_least(last_retained_sequence)
        sequence = self._control.next_sequence(allow_destroyed=True)
        context.append_retained_message(
            {
                "sequence": sequence,
                "type": "message",
                "role": "system",
                "content": "Note: the previous skill call failed. Its sandbox workspace "
                "changes have been discarded and the workspace has been reverted "
                "to the last successful checkpoint.",
                "source": "context_notice",
            }
        )
        return sequence

    def suspend(self) -> None:
        """Close the independent agent-wide scheduler/execution gate."""
        self._orchestrator.suspend_agent(self)
        self.data_logger.record_event(
            type="agent_suspend_requested",
            payload={"agname": self.agname},
            term_message=f"[{self.agname}] SUSPEND ▶  requested",
        )

    def resume(self) -> None:
        """Reopen only the agent-wide suspension gate."""
        self._orchestrator.resume_agent(self)
        self.data_logger.record_event(
            type="agent_resumed",
            payload={"agname": self.agname},
            term_message=f"[{self.agname}] SUSPEND ✓  resumed",
        )

    def destroy(self) -> CloseHandle:
        """Reject new work and asynchronously drain and clean up this agent."""
        first = self._orchestrator.destroy_agent(self)
        if first:
            _live_agents.discard(self)
            try:
                self.data_logger.record_event(
                    type="agent_destroying",
                    payload={"agname": self.agname},
                    term_message=f"[{self.agname}] DESTROY ▶  cleanup scheduled",
                    flush=True,
                )
            except Exception as exc:
                print(f"[agent] WARNING: destroy request logging failed for {self.agname}: {exc}")
            self._maybe_cleanup_destroyed()
        return self._close_handle

    def is_suspended(self) -> bool:
        return self._control.is_suspended()

    def is_paused(self) -> bool:
        return self._control.is_paused_actual()

    @property
    def lifecycle_state(self) -> str:
        return self._control.lifecycle_state().upper()

    def is_settled(self) -> bool:
        if self._control.is_fully_destroyed():
            return True
        if self._control.is_destroyed():
            return False
        if self._control.is_suspended():
            active = self._control.active_invocation()
            return active is None or self._control.is_paused_actual()
        with self._orchestrator._event_cond:
            return not self._submissions or self._control.is_paused_actual()

    # ------------------------------------------------------------------
    # Execution — delegates to agskill
    # ------------------------------------------------------------------

    def _ensure_sandbox(self) -> agSandbox:
        """Create the sandbox lazily on the admitted engine thread."""
        if self.sandbox is not None:
            return self.sandbox
        sandbox_config = self.agconfig
        agent_output_dir = self.output_path
        if agent_output_dir is not None:
            sandbox_config = sandbox_config.clone() if sandbox_config else agConfig()
            agSandboxConfig(sandbox_config).add_mount(
                "agent_output", agent_output_dir, "/agent_output"
            )
        self.sandbox = agSandbox(self.agname, agconfig=sandbox_config)
        return self.sandbox

    def queue_message(self, message: str) -> MessageSubmission:
        """Append one ordered retained message without starting infrastructure."""
        if not isinstance(message, str):
            raise TypeError("message must be a string")
        if not message.strip():
            raise ValueError("message must be a non-empty string")
        return self._orchestrator.submit_context_message(self, message)

    def run(self, skill, skill_input: agdata, max_steps: "int | None" = None) -> Invocation:
        """Submit ready work and immediately return its exact Invocation."""
        if max_steps is None:
            return skill.run(self, skill_input)
        return skill.run(self, skill_input, max_steps=max_steps)

    async def asyncio_run(
        self,
        skill,
        skill_input: "agdata",
        max_steps: "int | None" = None,
    ) -> "agdata":
        """Async wrapper returning the resolved invocation output."""
        return await self.run(skill, skill_input, max_steps)

    # ------------------------------------------------------------------
    # Destructor
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        """Best-effort: log destruction. The sandbox (if any) cleans itself up
        via agSandbox.__del__ once this agent's reference to it is gone."""
        _live_agents.discard(self)
        try:
            self.data_logger.record_event(
                type="agent_state",
                payload={"state": "agent_exit"},
                update_latest_snapshot=True,
                flush=True,
            )
            self.data_logger.record_event(
                type="agent_destroyed",
                payload={"agname": self.agname},
                term_message=f"[{self.agname}] DESTROYED",
                flush=True,
            )
        except Exception as _e:
            print(f"[agent] WARNING: __del__ log failed for {getattr(self, 'agname', '?')}: {_e}")

    # ------------------------------------------------------------------
    # Fork
    # ------------------------------------------------------------------
    @classmethod
    def fork(cls, src: "agent", agname: str | None = None) -> "agent":
        """Return an independent agent forked from *src*."""
        with src._operation_lease("fork"):
            return cls._fork_leased(src, agname)

    @classmethod
    def _fork_leased(cls, src: "agent", agname: str | None = None) -> "agent":
        """Construct a fork while the source agent's resources are leased."""
        ag: agent = cls.__new__(cls)
        ag.agname = _agname.allocate_agname(agname, prefix="agent")
        ag._parent_agent_id = str(src.agname)
        # Cloned so the fork's own agconfig is independent of src's -- see
        # the matching comment in __init__.
        ag.agconfig = src.agconfig.clone() if src.agconfig is not None else None
        ag.harness = src.harness
        ag.engine = None
        with src._orchestrator._event_cond:
            with src._submission_lock:
                source_context = src.context
        source_context.resolve_prev_dependencies()
        with src._orchestrator._event_cond:
            with src._submission_lock:
                ag.context = source_context.copy()
        _out_dir = _classvar_or_agconfig(ag.agconfig, "output_dir", cls.output_dir)
        _out = Path(_out_dir) / ag.agname if _out_dir else None
        sb_cfg = ag.agconfig
        if _out is not None:
            sb_cfg = sb_cfg.clone() if sb_cfg else agConfig()
            agSandboxConfig(sb_cfg).add_mount("agent_output", _out, "/agent_output")
        ag.sandbox = (
            src.sandbox.fork(ag.agname, agconfig=sb_cfg) if src.sandbox is not None else None
        )
        ag._owns_sandbox = True

        from .agteam import _active_team

        _team = _active_team.get(None)
        if _team is not None:
            _team._agents.add(ag)
        team_name = _team.team_name if _team is not None else None

        ag._finish_construction(
            event_type="agent_forked",
            event_payload={
                "agname": ag.agname,
                "parent_agname": src.agname,
                "team": team_name,
                "llm_config": _llm_config_snapshot(ag.agconfig),
            },
            term_message=f"[{ag.agname}] FORKED   from {src.agname}",
        )
        return ag

    # ------------------------------------------------------------------
    # Live agent registry
    # ------------------------------------------------------------------

    @classmethod
    def all(cls) -> "list[agent]":
        return list(_live_agents)

    @classmethod
    def save_all(cls, directory: "Path | str") -> "list[Path]":
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for ag in cls.all():
            path = directory / f"{ag.agname}.ckpt"
            ag.save(path)
            paths.append(path)
        return paths

    @classmethod
    def load_all(
        cls,
        directory: "Path | str",
        agconfig: "agConfig | None" = None,
    ) -> "list[agent]":
        directory = Path(directory)
        live_names = {ag.agname: ag for ag in cls.all()}
        restored: list[agent] = []

        for ckpt in sorted(directory.glob("*.ckpt")):
            with tarfile.open(ckpt, "r:gz") as tar:
                state = json.loads(tar.extractfile("state.json").read())
            agname = state["agname"]

            if agname in live_names:
                existing = live_names[agname]
                existing.data_logger.record_event(
                    type="agent_load_skipped",
                    payload={"agname": agname, "ckpt": ckpt.name},
                    term_message=(
                        f"[{agname}] CKPT     load_all: already live, skipping {ckpt.name}"
                    ),
                )
                restored.append(existing)
            else:
                ag = cls.load(ckpt, agconfig=agconfig)
                restored.append(ag)

        return restored

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: "Path | str") -> None:
        """Checkpoint this agent to a single .ckpt file."""
        with self._operation_lease("save"):
            self._save_leased(path)

    def _save_leased(self, path: "Path | str") -> None:
        """Write one checkpoint while explicit destruction waits for this lease."""
        path = Path(path)
        image_tag = f"agency/ckpt-{self.agname}"

        with self._orchestrator._event_cond:
            with self._submission_lock:
                checkpoint_context = self.context

        if checkpoint_context.is_pending():
            self.data_logger.record_event(
                type="agent_checkpoint_waiting",
                payload={"agname": self.agname},
                term_message=f"[{self.agname}] CKPT ⏳  waiting for in-flight task to complete...",
            )
        checkpoint_context.resolve_prev_dependencies()
        with self._orchestrator._event_cond:
            with self._submission_lock:
                checkpoint_context = checkpoint_context.copy()

        state = {
            "agname": self.agname,
            "parent_agent_id": self._parent_agent_id,
            "harness": self.harness,
            "llm_config": _llm_config_snapshot(self.agconfig),
            "history": checkpoint_context.recent_transcript,
            "ts": _ts(),
        }
        if self.sandbox is not None and self.sandbox._checkpoint_image is not None:
            # Recorded so load() knows which backend's image format
            # container.tar is in -- a chroot snapshot directory and a
            # docker/podman image tag are unrelated formats.
            state["sandbox_image_kind"] = self.sandbox.image_kind
        if checkpoint_context.harness_sessions:
            # Session continuity travels with the agent's own checkpoint, not
            # with container.tar, so it remains available regardless of which
            # sandbox this checkpoint is later restored onto.
            state["harness_sessions"] = checkpoint_context.harness_sessions
        if checkpoint_context.retained_messages:
            state["retained_messages"] = checkpoint_context.retained_messages
        if checkpoint_context.harness_message_cursors:
            state["harness_message_cursors"] = checkpoint_context.harness_message_cursors
        state_bytes = json.dumps(state, indent=2).encode()

        path.parent.mkdir(parents=True, exist_ok=True)

        if self.sandbox is not None and self.sandbox._checkpoint_image is not None:
            backend_cls = type(self.sandbox._backend)
            backend_cls.tag_image(self.sandbox._checkpoint_image, image_tag)
            try:
                _save_timeout = _AgAgentFields(self.agconfig).checkpoint_save_timeout_s
                # Scrub the owning process's PID before embedding -- it's
                # meaningless (and, since a .ckpt file can be restored by
                # an unrelated process on a different host entirely,
                # potentially misleading) once outside this process's own
                # lifetime. load() re-stamps the actually-current
                # restoring process's PID after import. See
                # relabel_owner_pid()'s docstring.
                backend_cls.relabel_owner_pid(image_tag, None, _save_timeout)
                image_bytes = backend_cls.export_image(image_tag, _save_timeout)
                with tarfile.open(path, "w:gz") as tar:
                    for name, data in [("state.json", state_bytes), ("container.tar", image_bytes)]:
                        info = tarfile.TarInfo(name=name)
                        info.size = len(data)
                        tar.addfile(info, io.BytesIO(data))
            finally:
                backend_cls.delete_image(image_tag, force=True)
        else:
            with tarfile.open(path, "w:gz") as tar:
                info = tarfile.TarInfo(name="state.json")
                info.size = len(state_bytes)
                tar.addfile(info, io.BytesIO(state_bytes))

        size_kb = path.stat().st_size // 1024
        self.data_logger.record_event(
            type="agent_saved",
            payload={"agname": self.agname, "path": str(path), "size_kb": size_kb},
            term_message=f"[{self.agname}] CKPT ✓   saved → {path}  ({size_kb} KB)",
        )

    @classmethod
    def load(
        cls,
        path: "Path | str",
        agconfig: "agConfig | None" = None,
    ) -> "agent":
        """Restore an agent from a checkpoint file created by agent.save().

        The checkpointed LLM config (everything except ``api_key``, which
        ``save()`` strips) is merged into ``agconfig``'s ``agllm_backend``
        fields -- a field already set explicitly on ``agconfig`` (e.g.
        ``cfg.agllm_backend.api_key = ...``, to restore the secret ``save()``
        dropped) wins over the checkpointed value.
        """
        path = Path(path)
        image_tag = f"agency/ckpt-restore-{_uuid_mod.uuid4().hex[:8]}"

        with tarfile.open(path, "r:gz") as tar:
            state = json.loads(tar.extractfile("state.json").read())
            container_member = next(
                (m for m in tar.getmembers() if m.name == "container.tar"), None
            )
            image_bytes = tar.extractfile(container_member).read() if container_member else None

        checkpoint: str | None = None
        image_kind = state.get("sandbox_image_kind", "container")
        if image_bytes is not None:
            _load_timeout = _AgAgentFields(agconfig).checkpoint_load_timeout_s
            backend_cls = agSandbox.backend_for_image_kind(image_kind)
            backend_cls.import_image(image_bytes, _load_timeout)
            original_tag = f"agency/ckpt-{state['agname']}"
            backend_cls.tag_image(original_tag, image_tag)
            backend_cls.delete_image(original_tag)
            # Stamp the actually-current restoring process's own PID --
            # save() scrubbed whatever PID this image carried before
            # embedding it (see relabel_owner_pid()'s docstring), so
            # without this the restored image would carry no owner
            # evidence at all, same as a never-labelled image.
            backend_cls.relabel_owner_pid(image_tag, os.getpid(), _load_timeout)
            checkpoint = image_tag

        ag: agent = cls.__new__(cls)
        ag.agname = _agname.claim_unique_agname(state["agname"])
        ag._parent_agent_id = state.get("parent_agent_id")
        _base_agconfig = agconfig if agconfig is not None else agent.default_agconfig
        ag.agconfig = _base_agconfig.clone() if _base_agconfig is not None else agConfig()
        _already_set = (
            _base_agconfig.data.get("agllm_backend", {}) if _base_agconfig is not None else {}
        )
        for k, v in state.get("llm_config", {}).items():
            if k not in _already_set:
                ag.agconfig.set("agllm_backend", k, v)
        # Accept the old checkpoint key so existing snapshots remain loadable.
        ag.harness = state.get("harness", state.get("engine", "native"))
        ag.engine = None
        ag.context = agcontext(
            recent_transcript=list(state.get("history", [])),
            harness_sessions=state.get("harness_sessions", {}),
            retained_messages=state.get("retained_messages", []),
            harness_message_cursors=state.get("harness_message_cursors", {}),
        )
        _out_dir = _classvar_or_agconfig(ag.agconfig, "output_dir", cls.output_dir)
        _out = Path(_out_dir) / ag.agname if _out_dir else None
        sb_cfg = ag.agconfig
        if _out is not None:
            sb_cfg = sb_cfg.clone() if sb_cfg else agConfig()
            agSandboxConfig(sb_cfg).add_mount("agent_output", _out, "/agent_output")
        if checkpoint and image_kind == "chroot":
            # Force the matching backend -- auto-detection (podman/docker
            # preferred when usable) would otherwise reconstruct this
            # sandbox with a backend that can't make sense of a chroot
            # snapshot tag. Container-kind checkpoints don't need this: auto
            # picking podman vs. docker for them was already safe before
            # chroot existed.
            sb_cfg = sb_cfg.clone() if sb_cfg else agConfig()
            agSandboxBackendConfig(sb_cfg).update(backend="chroot")
        ag.sandbox = (
            agSandbox(ag.agname, checkpoint_image=checkpoint, agconfig=sb_cfg)
            if checkpoint
            else None
        )
        ag._owns_sandbox = True

        ag._finish_construction(
            event_type="agent_loaded",
            event_payload={
                "agname": ag.agname,
                "source": str(path),
                "checkpoint_ts": state.get("ts"),
            },
            term_message=f"[{ag.agname}] LOADED   from {path}",
        )

        return ag

    def __repr__(self) -> str:
        return f"agent(agname={self.agname!r})"
