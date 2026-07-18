"""Tests for agpolicy -- the mediation interface for externally-driven
(harness) and, eventually, native tool-call execution."""

from __future__ import annotations

import pytest

from agency.agpolicy import agAllowAllPolicy, agdecision, agpolicy


def test_agdecision_allow_factory():
    d = agdecision.allow()
    assert d.kind == "allow"
    assert d.reason is None
    assert d.new_args is None


def test_agdecision_deny_factory():
    d = agdecision.deny("blocked for testing")
    assert d.kind == "deny"
    assert d.reason == "blocked for testing"
    assert d.new_args is None


def test_agdecision_rewrite_factory():
    d = agdecision.rewrite(["/bin/echo", "x"])
    assert d.kind == "rewrite"
    assert d.new_args == ["/bin/echo", "x"]
    assert d.reason is None


def test_agpolicy_base_check_not_implemented():
    policy = agpolicy()
    with pytest.raises(NotImplementedError):
        policy.check(ag=None, event=None)


def test_allow_all_policy_always_allows():
    policy = agAllowAllPolicy()
    decision = policy.check(ag=None, event=None)
    assert decision.kind == "allow"


def test_custom_policy_subclass():
    class DenyEverything(agpolicy):
        def check(self, ag, event):
            return agdecision.deny("nope")

    policy = DenyEverything()
    decision = policy.check(ag=None, event=None)
    assert decision.kind == "deny"
    assert decision.reason == "nope"
