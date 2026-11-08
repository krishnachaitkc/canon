from pathlib import Path

import pytest

from canon.mentions import build_index
from canon.parse import parse_show
from canon.speakers import core_pairs, core_words, speaker_stats

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "ravenscar"


@pytest.fixture(scope="module")
def stats():
    show = parse_show(SAMPLE)
    return speaker_stats(show, build_index(show))


def test_co_speaking_scenes(stats):
    assert stats["GUARD"].co_speakers.get("ROYAL GUARD", 0) > 0
    assert stats["STRANGER"].co_speakers.get("EDMUND", 0) > 0
    assert "LORD EDMUND" not in stats["EDMUND"].co_speakers
    assert "MARGARET" not in stats["HOODED FIGURE"].co_speakers


def test_episodes_and_lines(stats):
    assert stats["EDMUND"].episodes == [1, 2, 4]
    assert stats["LORD EDMUND"].episodes == [3]
    assert stats["STRANGER"].episodes == [1, 4]
    assert stats["LORD EDMUND"].lines == 3


def test_core_words():
    assert core_words("LORD EDMUND") == ["edmund"]
    assert core_words("LADY MARGARET") == ["margaret"]
    assert core_words("ROYAL GUARD") == []
    assert core_words("EDMUND'S MOTHER") == []      # names the mother, not Edmund
    assert core_words("THE HOODED FIGURE") == ["figure", "hooded"]


def test_a_title_change_shares_a_core_word(stats):
    pairs = {(a, b) for a, b, _ in core_pairs(stats)}

    assert ("EDMUND", "LORD EDMUND") in pairs
    assert ("GUARD", "ROYAL GUARD") not in pairs


def test_samples_are_first_and_longest_and_short(stats):
    lord = stats["LORD EDMUND"]

    assert lord.samples[0] == '"Ravensgar Hall is mine by right."'
    assert all(len(sample) <= 100 for s in stats.values() for sample in s.samples)


def test_names_on_screen_while_a_label_speaks(stats):
    # Planted case 17: STRANGER's first line plays over a shot that names Edmund.
    assert dict(stats["STRANGER"].camera_names).get("Edmund") == 1
    # A label's own name is never its own evidence.
    assert "Edmund" not in dict(stats["EDMUND"].camera_names)


def test_personal_names_come_from_one_name_labels():
    from canon.speakers import personal_names

    labels = ["EDMUND", "LADY MARGARET", "HOODED FIGURE", "THE STRANGER", "GUARD"]
    assert personal_names(labels) == {"edmund", "margaret"}
