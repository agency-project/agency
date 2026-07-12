"""Tests for agname — unique agent name allocation."""

import threading
import pytest

from agency.agname import agname, _NOUNS
from agency.agutil import _b36_suffix


# ---------------------------------------------------------------------------
# _b36_suffix
# ---------------------------------------------------------------------------


def test_b36_suffix_zero():
    assert _b36_suffix(0) == "0000"


def test_b36_suffix_nine():
    assert _b36_suffix(9) == "0009"


def test_b36_suffix_ten_is_a():
    assert _b36_suffix(10) == "000a"


def test_b36_suffix_35_is_z():
    assert _b36_suffix(35) == "000z"


def test_b36_suffix_36_is_0010():
    assert _b36_suffix(36) == "0010"


def test_b36_suffix_fixed_width():
    for n in range(200):
        assert len(_b36_suffix(n)) == 4


def test_b36_suffix_custom_width():
    assert len(_b36_suffix(0, width=6)) == 6
    assert _b36_suffix(0, width=6) == "000000"


def test_b36_suffix_monotone_ordering():
    prev = _b36_suffix(0)
    for n in range(1, 100):
        cur = _b36_suffix(n)
        assert cur > prev, f"{cur!r} should sort after {prev!r}"
        prev = cur


# ---------------------------------------------------------------------------
# _NOUNS list
# ---------------------------------------------------------------------------


def test_nouns_list_nonempty():
    assert len(_NOUNS) > 0


def test_nouns_list_no_duplicates():
    assert len(_NOUNS) == len(set(_NOUNS))


def test_nouns_all_lowercase_alpha():
    for noun in _NOUNS:
        assert noun.isalpha() and noun == noun.lower(), noun


def test_nouns_all_four_chars():
    for noun in _NOUNS:
        assert len(noun) == 4, noun


# ---------------------------------------------------------------------------
# agname is a str subclass
# ---------------------------------------------------------------------------


def test_agname_is_str_subclass():
    assert issubclass(agname, str)


def test_agname_instance_is_str():
    assert isinstance(agname("arch_0000"), str)


def test_agname_equals_its_string_value():
    assert agname("arch_0000") == "arch_0000"


def test_agname_hash_matches_str():
    assert hash(agname("arch_0000")) == hash("arch_0000")


def test_agname_works_in_set_with_plain_str():
    s = {"arch_0000"}
    assert agname("arch_0000") in s


def test_agname_plain_str_found_in_set_of_agnames():
    n = agname.allocate_agname("pine")
    assert str(n) in agname._allocated


# ---------------------------------------------------------------------------
# agname.claim_unique_agname
# ---------------------------------------------------------------------------


def test_claim_unique_agname_adds_to_allocated():
    agname.claim_unique_agname("test_name")
    assert "test_name" in agname._allocated


def test_claim_unique_agname_returns_agname_instance():
    result = agname.claim_unique_agname("my_agent")
    assert isinstance(result, agname)
    assert result == "my_agent"


def test_claim_unique_agname_duplicate_raises():
    agname.claim_unique_agname("dup_agent")
    with pytest.raises(ValueError, match="already in use"):
        agname.claim_unique_agname("dup_agent")


def test_claim_unique_agname_different_names_both_succeed():
    agname.claim_unique_agname("agent_a")
    agname.claim_unique_agname("agent_b")
    assert "agent_a" in agname._allocated
    assert "agent_b" in agname._allocated


# ---------------------------------------------------------------------------
# agname.allocate_agname — with explicit base name
# ---------------------------------------------------------------------------


def test_allocate_agname_format():
    name = agname.allocate_agname("arch")
    parts = name.split("_")
    assert len(parts) == 2
    assert parts[0] == "arch"
    assert len(parts[1]) == 4


def test_allocate_agname_first_call_is_0000():
    name = agname.allocate_agname("bolt")
    assert name == "bolt_0000"


def test_allocate_agname_second_call_increments():
    first = agname.allocate_agname("crab")
    second = agname.allocate_agname("crab")
    assert first == "crab_0000"
    assert second == "crab_0001"


def test_allocate_agname_adds_to_allocated():
    name = agname.allocate_agname("dart")
    assert name in agname._allocated


def test_allocate_agname_different_bases_independent():
    a1 = agname.allocate_agname("frog")
    b1 = agname.allocate_agname("gale")
    a2 = agname.allocate_agname("frog")
    assert a1 == "frog_0000"
    assert b1 == "gale_0000"
    assert a2 == "frog_0001"


def test_allocate_agname_uniqueness_across_many():
    names = [agname.allocate_agname("hare") for _ in range(50)]
    assert len(names) == len(set(names))


def test_allocate_agname_returns_agname_instance():
    result = agname.allocate_agname("kite")
    assert isinstance(result, agname)


# ---------------------------------------------------------------------------
# agname.allocate_agname — with name=None (auto-picks from noun list)
# ---------------------------------------------------------------------------


def test_allocate_agname_none_format():
    name = agname.allocate_agname()
    parts = name.split("_")
    assert len(parts) == 2
    noun, suffix = parts
    assert noun in _NOUNS
    assert len(suffix) == 4


def test_allocate_agname_none_adds_to_allocated():
    name = agname.allocate_agname()
    assert name in agname._allocated


def test_allocate_agname_none_sequential_nouns():
    n1 = agname.allocate_agname()
    n2 = agname.allocate_agname()
    noun1, noun2 = n1.split("_")[0], n2.split("_")[0]
    idx1 = _NOUNS.index(noun1)
    idx2 = _NOUNS.index(noun2)
    assert (idx2 - idx1) % len(_NOUNS) == 1


def test_allocate_agname_none_all_unique_across_many():
    names = [agname.allocate_agname() for _ in range(100)]
    assert len(names) == len(set(names))


def test_allocate_agname_none_cycles_through_nouns():
    n = len(_NOUNS)
    first = agname.allocate_agname().split("_")[0]
    for _ in range(n - 1):
        agname.allocate_agname()
    wrapped = agname.allocate_agname().split("_")[0]
    assert first == wrapped


def test_allocate_agname_none_same_noun_increments_suffix():
    n = len(_NOUNS)
    first_noun = agname.allocate_agname().split("_")[0]
    for _ in range(n - 1):
        agname.allocate_agname()
    second = agname.allocate_agname()
    assert second == f"{first_noun}_0001"


def test_allocate_agname_none_returns_agname_instance():
    result = agname.allocate_agname()
    assert isinstance(result, agname)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_allocate_agname_thread_safe():
    results = []
    errors = []

    def worker():
        try:
            results.append(agname.allocate_agname("wave"))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(results) == 50
    assert len(set(results)) == 50


def test_allocate_agname_none_thread_safe():
    results = []
    errors = []

    def worker():
        try:
            results.append(agname.allocate_agname())
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(results) == 50
    assert len(set(results)) == 50
