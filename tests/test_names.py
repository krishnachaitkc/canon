from pathlib import Path

import pytest

from canon.mentions import build_index
from canon.names import MAX_KEYS, build_candidates, string_rule
from canon.parse import parse_show

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "ravenscar"


@pytest.fixture(scope="module")
def candidates():
    found, log = build_candidates(build_index(parse_show(SAMPLE)))
    assert log == []
    return found


def holding(candidates, key):
    return next(c for c in candidates if key in c.keys)


def test_the_three_ravens_spellings_form_one_candidate(candidates):
    candidate = holding(candidates, "ravenscar")

    assert {"ravenscar", "ravenscourt", "ravensgar"} <= set(candidate.keys)
    assert candidate.spellings["Ravenscar"] == 5
    assert candidate.spoken["Ravenscar"] == 2


def test_thomson_joins_only_through_the_context_rule(candidates):
    candidate = holding(candidates, "thorne")
    rules = [rule for a, b, rule in candidate.links if "thomson" in (a, b)]

    assert {"thorne", "thorn", "thomson"} <= set(candidate.keys)
    assert rules
    assert all(rule.startswith("same first name before") for rule in rules)
    assert string_rule("thomson", "thorne", 0.85) is None


def test_the_one_off_slip_joins_its_name(candidates):
    assert set(holding(candidates, "margaret").keys) == {"margaret", "margarett"}


def test_the_decoy_family_and_village_share_a_candidate(candidates):
    # Code cannot tell a family from a village; the model must split them.
    assert {"hartwell", "hartley"} <= set(holding(candidates, "hartwell").keys)


def test_names_that_look_nothing_alike_stay_out(candidates):
    grouped = {key for c in candidates for key in c.keys}

    assert "edmund" not in grouped
    assert "crown" not in grouped
    assert all(len(c.keys) <= MAX_KEYS for c in candidates)


def test_ids_follow_mention_counts(candidates):
    totals = [sum(c.spellings.values()) for c in candidates]

    assert [c.id for c in candidates] == [f"N{n:03d}" for n in range(1, len(candidates) + 1)]
    assert totals == sorted(totals, reverse=True)
