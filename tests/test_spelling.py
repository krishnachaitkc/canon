from pathlib import Path

import pytest

from canon.doctor import format_findings
from canon.mentions import build_index
from canon.parse import parse_show
from canon.spelling import spelling_candidates

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "ravenscar"


@pytest.fixture(scope="module")
def rows():
    show = parse_show(SAMPLE)
    return show, spelling_candidates(show, build_index(show))


def test_spoken_typos_become_candidates(rows):
    _, found = rows
    words = {r.found for r in found}
    seizure = next(r for r in found if r.found == "seizzure")

    assert {"seizzure", "gentelman"} <= words
    assert seizure.suggestion == "seizure"
    assert seizure.ep == 2 and seizure.where.startswith("E02.B")
    assert all(r.source == "check" and r.severity == "audible" for r in found)


def test_names_are_left_to_d1(rows):
    _, found = rows

    assert not {"ravensgar", "margarett", "thomson", "hartwells", "edmund"} & {r.found for r in found}


def test_only_dialogue_is_checked(rows):
    show, found = rows
    kinds = {b.id: b.kind for e in show.episodes for b in e.beats}

    assert found
    assert all(kinds[r.where] == "line" for r in found)


def test_format_findings_name_episode_4():
    found = format_findings(SAMPLE)

    assert [(f.ep, f.where, f.found, f.suggestion) for f in found] == [
        (4, "Scene heading", "SCENE HEADING", "Scene heading")
    ]
