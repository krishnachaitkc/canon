from collections import Counter
from pathlib import Path

import pytest

from canon.mentions import build_index, title_case
from canon.parse import parse_show

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "ravenscar"


@pytest.fixture(scope="module")
def index():
    return build_index(parse_show(SAMPLE))


def test_plural_and_possessive_share_one_key(index):
    surfaces = {m.surface for m in index.mentions if m.key == "hartwell"}

    assert {"Hartwell", "Hartwell's", "Hartwells"} <= surfaces
    assert "hartwells" not in index.names
    assert index.plural_map["hartwells"] == "hartwell"


def test_sentence_start_and_title_words_are_never_mentions(index):
    keys = {m.key for m in index.mentions}

    assert not {"the", "she", "then", "lord", "manor", "hall", "cathedral"} & keys


def test_planted_spellings_are_found_where_planted(index):
    spoken = Counter((m.key, m.ep) for m in index.mentions if m.kind == "spoken")

    assert spoken[("ravenscar", 1)] == 2
    assert spoken[("ravenscourt", 2)] == 1
    assert spoken[("ravensgar", 3)] == 1
    assert spoken[("thorne", 1)] == 2
    assert spoken[("thorne", 4)] == 1
    assert spoken[("thorn", 2)] == 1
    assert spoken[("thomson", 3)] == 1
    assert spoken[("margarett", 4)] == 1
    assert not any(m.key == "margaret" and m.kind == "spoken" for m in index.mentions)


def test_ravenscar_is_also_found_in_a_shot_and_a_heading(index):
    found = {(m.kind, m.ep) for m in index.mentions if m.key == "ravenscar"}

    assert ("shot", 1) in found
    assert ("header", 3) in found


def test_heading_words_count_only_when_the_show_uses_them_as_names(index):
    heading_keys = {m.key for m in index.mentions if m.beat_id.endswith(".heading")}

    # MOOR, ROAD, TIMBER, GATES and ALBAN never appear as names outside a heading.
    assert heading_keys == {"hartwell", "ravenscar"}


def test_a_mention_points_at_its_beat(index):
    mention = next(m for m in index.mentions if m.key == "ravenscourt")

    assert mention.beat_id.startswith("E02.B")
    assert mention.kind == "spoken"
    assert mention.speaker == "EDMUND"
    assert mention.start == 10.0


def test_first_names_and_contexts(index):
    assert {"edmund", "margaret"} <= index.first_names
    assert index.first_before["thomson"]["edmund"] == 1
    assert index.first_before["thorne"]["edmund"] >= 3
    assert index.head_after["ravenscar"]["hall"] >= 2
    assert index.head_after["hartley"]["village"] == 1


def test_title_case_keeps_possessives():
    assert title_case("EDMUND'S COTTAGE") == "Edmund's Cottage"
    assert title_case("ST. ALBAN'S CHAPEL") == "St. Alban's Chapel"
