"""Cross-cutting regression guards for harness profiling.

The end-to-end golden described by M8 depends on a deterministic mock/replay
endpoint for the five harness engines.  That endpoint is deliberately owned
outside the profiler roadmap and is not present in this repository yet.  Keep
the independently useful interruption and static guards here now; add the
engine golden beside them once the replay endpoint has a concrete protocol and
fixture.
"""

from __future__ import annotations

import ast
import json
import threading
from collections import Counter
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_AGENCY_ROOT = _REPO_ROOT / "agency"


# A raw Thread is correct for plumbing that must not inherit a task span.  The
# two agprof.py calls are the reviewed implementation of spawn_traced itself
# (including its profiling-disabled fast path).  An identity is
# (repository-relative path, lexical class/function scope); counts distinguish
# multiple reviewed calls in one scope without relying on brittle line numbers.
_BARE_THREAD_ALLOWLIST = {
    # Harness transcript/service plumbing.
    (
        "agency/agharness_internal/agharness_backends/native.py",
        "_NativeBackend.execute",
    ): (1, "transcript polling"),
    (
        "agency/agharness_internal/agharness_messenger.py",
        "agHarnessMessenger.start",
    ): (1, "TCP server"),
    (
        "agency/agharness_internal/agharness_messenger.py",
        "agHarnessMessenger.ensure_uds_started",
    ): (1, "UDS server"),
    (
        "agency/agharness_internal/agllm_terminus.py",
        "agLLMTerminus.start",
    ): (1, "TCP server"),
    (
        "agency/agharness_internal/agllm_terminus.py",
        "agLLMTerminus.ensure_uds_started",
    ): (1, "UDS server"),
    (
        "agency/agharness_internal/agmcp_server.py",
        "agMCPServer.start",
    ): (1, "TCP server"),
    (
        "agency/agharness_internal/agmcp_server.py",
        "agMCPServer.ensure_uds_started",
    ): (1, "UDS server"),
    (
        "agency/agharness_internal/agprof_ingest.py",
        "agProfilerIngest.ensure_uds_started",
    ): (1, "UDS ingest server"),
    (
        "agency/agharness_internal/agproxy_llm.py",
        "agProxyLLM.start",
    ): (1, "TCP server"),
    (
        "agency/agharness_internal/agproxy_llm.py",
        "agProxyLLM.ensure_uds_started",
    ): (1, "UDS server"),
    # ptrace transport and subprocess pipe plumbing.
    (
        "agency/agharness_internal/agproxy_ptrace_internal/_in_container_entrypoint.py",
        "_Tracer.run",
    ): (3, "stdin writer and subprocess pipe drainers"),
    (
        "agency/agharness_internal/agproxy_ptrace_internal/_in_container_launcher.py",
        "InContainerRelay.start",
    ): (2, "diagnostic and relay readers"),
    (
        "agency/agharness_internal/agproxy_ptrace_internal/_in_container_launcher.py",
        "InContainerRelay._drain_diagnostics",
    ): (2, "subprocess pipe drainers"),
    (
        "agency/agharness_internal/agproxy_ptrace_internal/_tcp_to_uds_relay.py",
        "_handle_connection",
    ): (2, "bidirectional relay pumps"),
    (
        "agency/agharness_internal/agproxy_ptrace_internal/_tcp_to_uds_relay.py",
        "main",
    ): (1, "per-connection handler"),
    (
        "agency/agharness_internal/agproxy_ptrace_internal/_tracer_loop.py",
        "TracerLoop.start",
    ): (1, "tracer loop"),
    (
        "agency/agharness_internal/agproxy_ptrace_internal/_tracer_loop.py",
        "TracerLoop._fork_and_exec",
    ): (3, "stdin writer and subprocess pipe readers"),
    # General background I/O and UI maintenance.
    ("agency/agutil.py", "_iter_batched"): (1, "stream iterator drainer"),
    ("agency/agwebui/__init__.py", "agwebui.run"): (1, "UI command relay"),
    ("agency/agwebui/emitter.py", "agwebui_emitter.emit"): (1, "event pruning"),
    ("agency/profiler/agprof.py", "spawn_traced"): (2, "spawn_traced implementation"),
    (
        "agency/profiler/agprof_emit.py",
        "RemoteProfilerEmitter.__init__",
    ): (1, "bounded telemetry sender"),
    ("agency/tools/human.py", "ask_human_and_wait"): (1, "stdin reader"),
}


class _BareThreadVisitor(ast.NodeVisitor):
    def __init__(
        self,
        relative_path: str,
        *,
        threading_aliases: set[str],
        thread_aliases: set[str],
    ) -> None:
        self.relative_path = relative_path
        self.threading_aliases = threading_aliases
        self.thread_aliases = thread_aliases
        self.scope: list[str] = []
        self.calls: Counter[tuple[str, str]] = Counter()

    def _visit_scope(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node)

    def visit_Call(self, node: ast.Call) -> None:
        is_bare_thread = False
        if isinstance(node.func, ast.Attribute):
            owner = node.func.value
            is_bare_thread = (
                node.func.attr == "Thread"
                and isinstance(owner, ast.Name)
                and owner.id in self.threading_aliases
            )
        elif isinstance(node.func, ast.Name):
            is_bare_thread = node.func.id in self.thread_aliases
        if is_bare_thread:
            scope = ".".join(self.scope) or "<module>"
            self.calls[(self.relative_path, scope)] += 1
        self.generic_visit(node)


def _thread_import_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    """Return aliases that can construct ``threading.Thread`` directly.

    Collect imports in a separate pass so a conditional or function-local
    import cannot evade the guard merely by appearing after a call in AST
    traversal order.  A rare alias later rebound to an unrelated object is
    intentionally conservative: it must still be reviewed and classified.
    """
    threading_aliases: set[str] = set()
    thread_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == "threading":
                    threading_aliases.add(imported.asname or "threading")
        elif isinstance(node, ast.ImportFrom) and node.module == "threading":
            for imported in node.names:
                if imported.name == "Thread":
                    thread_aliases.add(imported.asname or "Thread")
    return threading_aliases, thread_aliases


def _bare_thread_calls() -> Counter[tuple[str, str]]:
    calls: Counter[tuple[str, str]] = Counter()
    for path in _AGENCY_ROOT.rglob("*.py"):
        # macOS rsync/tar exports can leave AppleDouble sidecars such as
        # ``._module.py``. They are binary metadata, not Python sources or
        # executable thread sites.
        if path.name.startswith("._"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        threading_aliases, thread_aliases = _thread_import_aliases(tree)
        visitor = _BareThreadVisitor(
            path.relative_to(_REPO_ROOT).as_posix(),
            threading_aliases=threading_aliases,
            thread_aliases=thread_aliases,
        )
        visitor.visit(tree)
        calls.update(visitor.calls)
    return calls


def test_bare_thread_visitor_recognizes_import_aliases():
    tree = ast.parse(
        """
import threading as th
from threading import Thread as WorkerThread

def launch():
    th.Thread(target=lambda: None)
    WorkerThread(target=lambda: None)
"""
    )
    threading_aliases, thread_aliases = _thread_import_aliases(tree)
    visitor = _BareThreadVisitor(
        "agency/example.py",
        threading_aliases=threading_aliases,
        thread_aliases=thread_aliases,
    )
    visitor.visit(tree)

    assert visitor.calls == Counter({("agency/example.py", "launch"): 2})


def test_session_stop_preserves_an_inflight_run_as_interrupted(monkeypatch, tmp_path):
    """An interrupted harness run must not disappear or look successful."""
    pytest.importorskip("opentelemetry.sdk.trace")
    from agency.profiler import agprof

    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    started = threading.Event()
    release = threading.Event()

    def run() -> None:
        with agprof.span("run0:claude_code:golden-probe"):
            started.set()
            release.wait(timeout=10)

    agprof.start(tmp_path, sample_hz=0, sample_gpu=False)
    worker = agprof.spawn_traced(run)
    worker.start()
    assert started.wait(timeout=10)
    agprof.stop()
    release.set()
    worker.join(timeout=10)

    summary = json.loads((tmp_path / "summary.json").read_text())
    runs = summary["run_metrics"]
    assert runs["started"] == 1
    assert runs["completed"] == 0
    assert runs["succeeded"] == 0
    assert runs["failed"] == 0
    assert runs["interrupted"] == 1
    assert runs["p50_ms"] is None
    assert summary["incomplete_spans"][0]["label"] == "run0:claude_code:golden-probe"
    assert summary["incomplete_spans"][0]["outcome"] == "interrupted"


def test_bare_thread_sites_match_reviewed_allowlist():
    """A new bare task thread must be routed through agprof.spawn_traced()."""
    expected = Counter(
        {identity: count for identity, (count, _classification) in _BARE_THREAD_ALLOWLIST.items()}
    )

    assert _bare_thread_calls() == expected
