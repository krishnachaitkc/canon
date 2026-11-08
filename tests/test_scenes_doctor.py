from pathlib import Path

import pytest

from canon.doctor import format_report
from canon.parse import parse_show, split_heading

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples" / "the_lighthouse_heir"


@pytest.mark.parametrize(
    "heading, expected",
    [
        ("INT. CATHEDRAL – DAY", ("INT", "CATHEDRAL", "DAY")),
        (
            "EXT. RAVENSCAR HALL – SERVICE COURTYARD – DAY",
            ("EXT", "RAVENSCAR HALL - SERVICE COURTYARD", "DAY"),
        ),
        ("INT. ABANDONED BEDCHAMBER – PRE-DAWN", ("INT", "ABANDONED BEDCHAMBER", "PRE-DAWN")),
        (
            "INT. COTTAGE - PARLOR / EXT. COTTAGE - CONTINUOUS",
            ("INT", "COTTAGE - PARLOR / EXT. COTTAGE", "CONTINUOUS"),
        ),
        ("INT. NOBLE MANOR", ("INT", "NOBLE MANOR", "")),
        ("INT./EXT. CARRIAGE - NIGHT", ("INT/EXT", "CARRIAGE", "NIGHT")),
        ("ext. harbor - dawn", ("EXT", "HARBOR", "DAWN")),
    ],
)
def test_split_heading(heading, expected):
    assert split_heading(heading) == expected


def test_scenes_are_kept_on_the_episode():
    show = parse_show(SAMPLES)
    first = show.episodes[0]

    assert [(s.num, s.int_ext, s.place, s.time) for s in first.scenes] == [
        (1, "INT", "LIGHTHOUSE STAIRWELL", "DUSK"),
        (2, "INT", "LANTERN ROOM", "NIGHT"),
    ]
    assert first.scenes[1].start == 42.0
    assert all(scene.ep == first.num for scene in first.scenes)


def episode_text(num: int, heading_label: str = "Scene heading") -> str:
    return (
        f"EPISODE NUMBER: Episode {num}\n"
        "EPISODE LOGLINE: A test episode.\n"
        "CONTINUITY NOTES: None.\n\n"
        "--- SCRIPT START ---\n\n"
        "SCENE 01\n"
        f"{heading_label}: INT. TEST ROOM - DAY\n"
        "Scene start: 00:00.00\n"
        "Scene end: 01:00.00\n\n"
        "Actions and dialogue:\n\n"
        "**CAMERA:**\n"
        "Start: (00:00.00) | End: (00:03.00)\n"
        "A single camera shot.\n"
    )


def test_format_report_names_the_episode_that_differs(tmp_path):
    (tmp_path / "episode-1.txt").write_text(episode_text(1), encoding="utf-8")
    (tmp_path / "episode-2.txt").write_text(episode_text(2, "SCENE HEADING"), encoding="utf-8")
    (tmp_path / "episode-3.txt").write_text(episode_text(3), encoding="utf-8")

    report = format_report(tmp_path)

    assert report == [
        {
            "label": "Scene heading",
            "ep": 2,
            "found": "SCENE HEADING",
            "majority": "Scene heading",
            "majority_count": 2,
        }
    ]
    # The uppercase heading label still parses: the scene is not lost.
    show = parse_show(tmp_path)
    assert [len(episode.scenes) for episode in show.episodes] == [1, 1, 1]


def test_format_report_is_quiet_when_every_episode_agrees(tmp_path):
    for num in (1, 2, 3):
        (tmp_path / f"episode-{num}.txt").write_text(episode_text(num), encoding="utf-8")

    assert format_report(tmp_path) == []
