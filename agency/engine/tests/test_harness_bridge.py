# Tests for harness_bridge.py -- currently just the constructor contract; the
# rest of HarnessManagerBridge is still an unimplemented skeleton.

from __future__ import annotations

import pytest

from agency.engine.harness_bridge import HarnessManagerBridge


def test_init_takes_no_arguments_and_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        HarnessManagerBridge()


def test_init_rejects_a_positional_argument():
    with pytest.raises(TypeError):
        HarnessManagerBridge(object())
