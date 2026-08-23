# Tests for harness_bridge.py -- construction is side-effect free; launching
# the sandbox-side process remains the next implementation step.

from __future__ import annotations

import pytest

from agency.engine.harness_bridge import HarnessManagerBridge


def test_init_takes_no_arguments_without_launching_anything():
    assert isinstance(HarnessManagerBridge(), HarnessManagerBridge)


def test_init_rejects_a_positional_argument():
    with pytest.raises(TypeError):
        HarnessManagerBridge(object())
