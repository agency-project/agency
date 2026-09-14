"""Model-free span bridge smoke; no container or paid API call."""

from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest

from agency.engine.host_servers.host_interaction_server import HostInteractionServer
from agency.native_harness.profiling import NativeProfiler
from agency.observability.profiler import agprof


def test_closing_payload_preserves_opening_timestamp_and_parent():
    payloads = []

    def report(payload):
        payloads.append(payload)
        return {"ok": True}

    profiler = NativeProfiler(SimpleNamespace(record_profiler_span=report))
    profiler.enabled = True
    with profiler.span("outer"):
        with profiler.span("child"):
            pass
    start_outer, start_child, end_child, end_outer = payloads
    assert end_child["start_ts"] == start_child["start_ts"]
    assert end_child["parent"] == start_child["parent"] == start_outer["span_id"]
    assert end_outer["start_ts"] == start_outer["start_ts"]
    assert end_child["end_ts"] >= end_child["start_ts"]


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="agprof requires Linux /proc/cgroups"
)
def test_remote_lifecycle_spans_keep_start_parent_and_outcome(tmp_path, failure):
    logger = Mock()
    server = HostInteractionServer(SimpleNamespace(policy=SimpleNamespace()), logger, "smoke")

    def report(payload):
        server.record_span(
            payload["name"],
            payload.get("start_ts"),
            payload.get("end_ts"),
            payload["attributes"],
            span_id=payload["span_id"],
            parent=payload.get("parent"),
        )
        return {"ok": True}

    profiler = NativeProfiler(SimpleNamespace(record_profiler_span=report))
    profiler.enabled = True
    with agprof.session(tmp_path / "profile", sample_gpu=False):
        try:
            with profiler.span("harness:await_cli"):
                with profiler.span("harness:snapshot"):
                    if failure:
                        raise RuntimeError("synthetic")
        except RuntimeError:
            assert failure
    records = {r[1]: r for r in agprof.profile_records()}
    assert records["harness:snapshot"][8] == records["harness:await_cli"][7]
    assert records["harness:snapshot"][6]["outcome"] == ("failure" if failure else "success")
    assert not server._open_spans
    calls = logger.record_span.call_args_list
    assert len(calls) == 2
    for call in calls:
        assert call.args[1] is not None and call.args[2] >= call.args[1]
    assert calls[0].kwargs["parent"] is not None
