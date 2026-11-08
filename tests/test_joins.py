from pathlib import Path

import pytest

from canon.joins import CONTINUOUS_NOTES, TIME_SKIP, find_joins
from canon.parse import parse_show

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "ravenscar"


@pytest.fixture(scope="module")
def findings():
    return find_joins(parse_show(SAMPLE))


def test_time_flip_on_a_continuous_episode_join(findings):
    time = [f for f in findings if f.group == "time"]

    assert [f.where for f in time] == ["E01 S02 -> E02 S01"]
    assert time[0].source == "rule"
    assert time[0].found.startswith("NIGHT")
    assert "-> DAY" in time[0].found
    assert 'notes say "Picks up immediately"' in time[0].suggestion


def test_the_next_morning_is_not_continuous(findings):
    assert not any(f.where.startswith("E02 S03 -> E03") for f in findings)


def test_a_new_scene_at_night_is_not_a_flip(findings):
    # E2 S1 is DAY and S2 is NIGHT, but S2 is a new scene, not CONTINUOUS.
    assert not any(f.where == "E02 S01 -> E02 S02" for f in findings)


def test_one_building_with_two_names_is_a_check(findings):
    building = {f.where: f for f in findings if f.group == "building"}

    assert sorted(building) == ["E01 S01 -> E01 S02", "E03 S02 -> E04 S01"]
    assert building["E03 S02 -> E04 S01"].found == "cathedral -> chapel"
    assert building["E01 S01 -> E01 S02"].found == "cottage -> cabin"   # from the descriptions
    assert all(f.source == "check" for f in building.values())


def test_rooms_of_one_building_are_not_flagged(findings):
    assert not any(f.where == "E02 S02 -> E02 S03" for f in findings)


@pytest.mark.parametrize(
    "notes, continuous",
    [
        ("Picks up immediately from Episode 1's cliffhanger.", True),
        ("Follows directly from Episode 2's confrontation.", True),
        ("Directly follows Episode 47's public disgrace.", True),
        ("Picks up right after Edmund commands his men.", True),
        ("Picks up immediately following Margaret's ordeal.", True),
        ("Episode 34 transitions directly into the courtroom trial.", False),
        ("Follows directly on from Episode 2, the next morning.", False),
        ("Picks up the morning after Episode 16.", False),
        ("Following her encounter with Edmund in the forest.", False),
    ],
)
def test_which_notes_mean_real_time(notes, continuous):
    assert (bool(CONTINUOUS_NOTES.search(notes)) and not TIME_SKIP.search(notes)) == continuous
