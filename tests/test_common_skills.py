"""Tests for agency/common_skills/ (agplan and agbuild)."""

from agency.common_skills.agplan import agplan
from agency.common_skills.agbuild import agbuild


def test_agplan_is_importable():
    assert agplan is not None


def test_agbuild_is_importable():
    assert agbuild is not None
