import zipfile
from pathlib import Path

import pytest

from canon.parse import natural_key, parse_show, read_input

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples" / "the_lighthouse_heir"


def minimal_script(episode_num: int | None = 1, start: str = "00:00.00", end: str = "00:03.00") -> str:
    header = f"EPISODE NUMBER: Episode {episode_num}\n" if episode_num is not None else ""
    return (
        header
        + "EPISODE LOGLINE: A minimal test episode.\n"
        + "CONTINUITY NOTES: None.\n\n"
        + "--- SCRIPT START ---\n\n"
        + "SCENE 01\n"
        + "Scene heading: INT. TEST ROOM - DAY\n"
        + "Scene start: 00:00.00\n"
        + "Scene end: 01:00.00\n\n"
        + "ACTIONS AND DIALOGUE:\n\n"
        + "**CAMERA:**\n"
        + f"Start: ({start}) | End: ({end})\n"
        + "A single camera shot.\n"
    )


def test_folder_input_parses_the_sample():
    show = parse_show(SAMPLES)

    assert show.title == "the lighthouse heir"
    assert [episode.num for episode in show.episodes] == [1, 2, 3]
    assert show.unconsumed_lines == 0
    for episode in show.episodes:
        assert episode.beats
        assert all(beat.start < beat.end for beat in episode.beats)


def test_zip_and_folder_inputs_match(tmp_path):
    zip_path = tmp_path / "the_lighthouse_heir.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in sorted(SAMPLES.glob("*.txt")):
            zf.write(path, arcname=path.name)

    folder_show = parse_show(SAMPLES)
    zip_show = parse_show(zip_path)

    assert [episode.num for episode in zip_show.episodes] == [1, 2, 3]
    assert [
        [(beat.id, beat.text) for beat in episode.beats]
        for episode in zip_show.episodes
    ] == [
        [(beat.id, beat.text) for beat in episode.beats]
        for episode in folder_show.episodes
    ]


def test_natural_sort_orders_episode_10_after_episode_2(tmp_path):
    for name, num in (("episode-1.txt", 1), ("episode-10.txt", 10), ("episode-2.txt", 2)):
        (tmp_path / name).write_text(minimal_script(num), encoding="utf-8")

    names = [name for name, _ in read_input(tmp_path)]

    assert names == ["episode-1.txt", "episode-2.txt", "episode-10.txt"]
    assert natural_key("episode-10") > natural_key("episode-2")
    assert [episode.num for episode in parse_show(tmp_path).episodes] == [1, 2, 10]


def test_beat_ids_increase_with_time():
    show = parse_show(SAMPLES)

    for episode in show.episodes:
        starts = [beat.start for beat in episode.beats]
        ids = [beat.id for beat in episode.beats]
        assert starts == sorted(starts)
        assert ids == [f"E{episode.num:02d}.B{index:03d}" for index in range(1, len(ids) + 1)]


def test_two_group_tag_sets_offscreen_and_keeps_tone():
    show = parse_show(SAMPLES)
    housekeeper = [beat for beat in show.episodes[0].beats if beat.speaker == "HOUSEKEEPER"]

    firm = [beat for beat in housekeeper if beat.tone == "Firm"]

    assert firm
    assert all(beat.kind == "line" for beat in firm)
    assert all(beat.offscreen for beat in firm)


def test_commanding_resounding_is_a_line_not_sfx():
    show = parse_show(SAMPLES)

    beat = next(
        beat
        for beat in show.episodes[0].beats
        if beat.tone == "Commanding, Resounding"
    )

    assert beat.kind == "line"
    assert beat.speaker == "MIRA"


def test_scene_assignment_by_time():
    show = parse_show(SAMPLES)
    first = show.episodes[0]

    assert {beat.scene for beat in first.beats} == {1, 2}
    assert all(beat.scene == 1 for beat in first.beats if beat.start < 42.0)
    assert all(beat.scene == 2 for beat in first.beats if beat.start >= 42.0)


def test_duplicate_episode_numbers_are_an_error(tmp_path):
    (tmp_path / "a.txt").write_text(minimal_script(1), encoding="utf-8")
    (tmp_path / "b.txt").write_text(minimal_script(1), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate episode number 1"):
        parse_show(tmp_path)


def test_episode_number_falls_back_to_filename(tmp_path):
    (tmp_path / "episode-007-unknown.txt").write_text(
        minimal_script(None), encoding="utf-8"
    )

    show = parse_show(tmp_path)

    assert show.episodes[0].num == 7
    assert show.episodes[0].beats[0].id == "E07.B001"


def test_start_at_or_after_end_is_an_anomaly(tmp_path):
    (tmp_path / "episode-005-broken.txt").write_text(
        minimal_script(5, start="00:06.00", end="00:04.00"), encoding="utf-8"
    )

    show = parse_show(tmp_path)
    beat = show.episodes[0].beats[0]

    assert len(show.anomalies) == 1
    assert "start 6.0 >= end 4.0" in show.anomalies[0]
    assert beat.start == 6.0
    assert beat.end == 6.5
