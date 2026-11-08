from pathlib import Path

import pytest

import app as canon_app
from canon.issues import (
    BANNED,
    all_text,
    apart_lines,
    build_cards,
    by_hand_rows,
    change_rows,
    check_places,
    decided_count,
    download_summary,
    is_decided,
    issue_rows,
    metrics,
    progress_text,
    ready_places,
    status,
    tick,
    will_fix_count,
)
from canon.models import Candidate, Edit, Finding, NameGroup
from canon.pipeline import run

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "ravenscar"


@pytest.fixture(scope="module")
def offline(tmp_path_factory):
    return run(str(SAMPLE), str(tmp_path_factory.mktemp("offline")), no_model=True)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    return run(str(SAMPLE), str(tmp_path_factory.mktemp("recorded")), model="fake/recorded")


def card(cards, key):
    return next(item for item in cards if item.id == key)


def test_every_finding_lands_in_exactly_one_card(offline, recorded):
    for result in (offline, recorded):
        cards = build_cards(result)
        placed = [fid for item in cards for fid in item.places]
        ids = [finding.id for finding in result["findings"]]
        assert sorted(placed) == sorted(ids)
        assert len(placed) == len(set(placed))


def test_one_card_per_name_and_voice_set(recorded):
    cards = build_cards(recorded)

    names = [item for item in cards if item.kind == "name"]
    voices = [item for item in cards if item.kind == "voice"]
    assert {item.group_id for item in names} == {"N002.1", "N003.1", "N004.1"}
    assert {item.group_id for item in voices} == {"S001", "S002"}
    assert card(cards, "voice:S001").sentence.startswith("One person under several labels")


def test_every_card_sits_in_the_right_lane(offline, recorded):
    for result in (offline, recorded):
        for item in build_cards(result):
            fixable = [f for f in item.findings if f.edit and f.edit.new != f.edit.old]
            assert item.fixable == tuple(f.id for f in fixable)
            if item.lane == "ready":
                assert fixable
                assert all(f.source == "ai" and f.confidence in ("high", "medium") for f in item.findings)
            elif item.lane == "check":
                assert fixable
                assert not all(
                    f.source == "ai" and f.confidence in ("high", "medium") for f in item.findings
                )
            else:
                assert item.lane == "hand"
                assert not fixable


def test_the_sample_splits_into_the_three_lanes(recorded):
    cards = build_cards(recorded)
    lanes = {item.lane for item in cards}

    assert lanes == {"ready", "check", "hand"}
    assert card(cards, "time:E02.S01.time|DAY|NIGHT").lane == "check"
    assert card(cards, "place:F0012").lane == "hand"
    assert card(cards, "format:4").lane == "hand"
    assert card(cards, "name:N003.1").lane == "ready"


def test_titles_and_sentences_come_from_the_data(recorded):
    cards = build_cards(recorded)
    ravens = card(cards, "name:N003.1")
    lord = card(cards, "voice:S002")
    typo = card(cards, "typo:word:seizzure")
    time = card(cards, "time:E02.S01.time|DAY|NIGHT")
    place = card(cards, "place:F0012")
    fmt = card(cards, "format:4")

    assert ravens.title == "Ravenscourt, Ravensgar → Ravenscar"
    assert ravens.sentence == (
        "'Ravenscourt', 'Ravensgar' look like 'Ravenscar'. Two places, three spellings: "
        "viewers hear all three."
    )
    assert lord.title == "LORD EDMUND → EDMUND"
    assert lord.sentence == (
        "LORD EDMUND speaks 3 lines that belong to EDMUND. Two labels mean two voices for one person."
    )
    assert typo.title == "'seizzure' → 'seizure'"
    assert typo.sentence == "'seizzure' in a spoken line (E02 00:23). Fix: 'seizure'."
    assert time.sentence == (
        "E01 S02 is NIGHT. E02 S01 continues the same moment but says DAY. Fix: E02 S01 → NIGHT."
    )
    assert place.sentence == (
        "The same moment is a cottage before the cut and a cabin after it. "
        "Pick one and fix the heading."
    )
    assert fmt.sentence == "E04 writes 'SCENE HEADING' where other episodes write 'Scene heading'."


def test_the_no_ai_typo_says_it_is_a_dictionary_guess(offline):
    typo = card(build_cards(offline), "typo:word:seizzure")

    assert typo.how_found == "You decide"
    assert typo.sentence.endswith("Dictionary guess. Old words like 'gaol' are real.")
    assert typo.problem == "'seizzure' is not an English word, so the voice would say it wrong."
    assert typo.why == (
        "Closest dictionary word. Old words like 'gaol' are real, so check first."
    )


def test_every_card_says_what_is_wrong_and_why(offline, recorded):
    for result in (offline, recorded):
        for item in build_cards(result):
            assert item.problem, item.id
            assert item.why, item.id


def test_the_name_card_counts_each_spelling(recorded):
    ravens = card(build_cards(recorded), "name:N003.1")

    assert ravens.problem == (
        "One name is written 3 ways: Ravenscar (5), Ravenscourt (1), Ravensgar (1). "
        "Viewers would hear all of them."
    )
    assert ravens.why == "'Ravenscar' is the spelling the script uses most (5 of 7)."


def test_the_voice_card_counts_each_label(recorded):
    cards = build_cards(recorded)
    lord = card(cards, "voice:S002")
    disguise = card(cards, "voice:S001")

    assert lord.problem == (
        "One character speaks under 2 labels: EDMUND (6 lines) and LORD EDMUND (3 lines). "
        "Each label gets its own voice, so one person would sound like two people."
    )
    assert lord.why == "'EDMUND' is the label used most (6 of 9 lines)."
    assert disguise.problem == (
        "The AI says these labels hide one person on purpose (a disguise or a reveal)."
    )
    assert disguise.why == "Nothing to rename. Cast one voice for all of them."


def test_the_time_card_names_the_notes_and_the_caution(recorded):
    time = card(build_cards(recorded), "time:E02.S01.time|DAY|NIGHT")

    assert time.problem == (
        "The notes say E02 picks up immediately after E01, but E01 S2 is NIGHT and E02 S1 "
        "is DAY. The picture would jump from night to day. "
        "If time passes between the two places, this is not a mistake."
    )
    assert time.why == "Canon suggests NIGHT to match the scene before it."


def test_the_place_and_format_cards_say_what_to_do(recorded):
    cards = build_cards(recorded)
    place = card(cards, "place:F0012")
    fmt = card(cards, "format:4")

    assert place.problem == (
        "In one continuous moment the building is a cottage before the cut and a cabin "
        "after it. The video would draw two different buildings."
    )
    assert place.why == "Canon can't tell which name is right. Pick one and edit the heading."
    assert fmt.problem == (
        "E04 writes SCENE HEADING in capitals; the other 3 episodes write 'Scene heading'."
    )
    assert fmt.why == (
        "Tools that read scripts by these labels could miss this episode. It is a one-time edit."
    )


def test_the_typo_card_says_what_the_voice_would_say(recorded):
    typo = card(build_cards(recorded), "typo:word:seizzure")

    assert typo.problem == "'seizzure' is not an English word, so the voice would say it wrong."
    assert typo.why == "The AI read the line and suggests 'seizure'."


def _unsure_name():
    return _finding(
        id="F0001",
        detector="names",
        severity="visible",
        source="check",
        confidence="low",
        ep=0,
        scene=0,
        where="N009",
        found="Alpha / Alfa",
        group="names:N009",
        edit=None,
    )


def _veto_voice():
    return _finding(
        id="F0002",
        detector="speakers",
        severity="voice",
        source="rule",
        confidence="high",
        ep=4,
        scene=1,
        where="E04 S01 00:34.80",
        found="ALPHA matched to ALFA, but both speak in E04 S01 00:34.80",
        group="veto:ALPHA",
        edit=None,
    )


def test_unsure_and_veto_cards_say_what_to_do():
    cards = build_cards(
        {
            "findings": [_unsure_name(), _veto_voice()],
            "name_groups": [],
            "speaker_groups": [],
        }
    )
    unsure = next(item for item in cards if item.kind == "name")
    veto = next(item for item in cards if item.kind == "voice")

    assert unsure.problem == "The AI could not tell whether Alpha and Alfa are one name."
    assert unsure.why == "If they are, give them one name. If not, leave both."
    assert veto.problem == (
        "ALPHA was matched to ALFA, but both speak in E04 S1, so they may be two people."
    )
    assert veto.why == "If so, give one of them a new label."


def test_the_name_card_says_when_the_ai_picked_a_less_used_spelling():
    result = {
        "findings": [
            _finding(
                id="F0001",
                detector="names",
                severity="audible",
                source="ai",
                confidence="high",
                where="E01.B001",
                found="Alfa",
                suggestion="Alpha",
                group="N005.1",
                edit=Edit(at="E01.B001", old="Alfa", new="Alpha"),
            )
        ],
        "name_groups": [
            NameGroup(
                id="N005.1",
                candidate="N005",
                spellings=["Alpha", "Alfa"],
                keys=["alpha", "alfa"],
                canonical="Alpha",
                same_entity=True,
                source="ai",
            )
        ],
        "speaker_groups": [],
        "candidates": [
            Candidate(
                id="N005",
                keys=["alpha", "alfa"],
                spellings={"Alfa": 5, "Alpha": 1},
                spoken={"Alfa": 4, "Alpha": 1},
                links=[],
            )
        ],
    }
    name = build_cards(result)[0]

    assert name.problem == (
        "One name is written 2 ways: Alpha (1), Alfa (5). Viewers would hear all of them."
    )
    assert name.why == (
        "The AI picked 'Alpha' (used 1 of 6 times). You can choose another in Details."
    )


def test_the_format_card_lists_every_label_in_one_episode():
    rows = [
        _finding(id="F0001", detector="format", severity="notes", ep=6, where="Scene heading", found="SCENE HEADING", suggestion="Scene heading", context="49 episodes write 'Scene heading'", group="format:Scene heading", edit=None),
        _finding(id="F0002", detector="format", severity="notes", ep=6, where="Scene start", found="SCENE START", suggestion="Scene start", context="49 episodes write 'Scene start'", group="format:Scene start", edit=None),
        _finding(id="F0003", detector="format", severity="notes", ep=6, where="Scene end", found="SCENE END", suggestion="Scene end", context="49 episodes write 'Scene end'", group="format:Scene end", edit=None),
        _finding(id="F0004", detector="format", severity="notes", ep=6, where="Actions and dialogue", found="ACTIONS AND DIALOGUE", suggestion="Actions and dialogue", context="49 episodes write 'Actions and dialogue'", group="format:Actions and dialogue", edit=None),
    ]
    fmt = build_cards({"findings": rows, "name_groups": [], "speaker_groups": []})[0]

    assert "SCENE HEADING" in fmt.problem
    assert "SCENE START" in fmt.problem
    assert "SCENE END" in fmt.problem
    assert "ACTIONS AND DIALOGUE" in fmt.problem
    assert "the other 49 episodes write them as 'Scene heading', 'Scene start', " in fmt.problem
    assert "'Scene end' and 'Actions and dialogue'" in fmt.problem


def test_kept_apart_pairs_are_one_line_without_ids(recorded):
    lines = apart_lines(recorded)

    assert lines == [
        "Hartwell · Hartley: the AI judged these to be different names. No change."
    ]
    assert not any("N001" in line for line in lines)


def test_kept_apart_drops_spellings_shown_on_a_card():
    result = {
        "findings": [
            _finding(
                id="F0001",
                detector="names",
                severity="audible",
                source="ai",
                confidence="high",
                where="E01.B001",
                found="Ravenscourt",
                suggestion="Ravenscar",
                group="N001.1",
                edit=Edit(at="E01.B001", old="Ravenscourt", new="Ravenscar"),
            )
        ],
        "name_groups": [
            NameGroup(
                id="N001.1",
                candidate="N001",
                spellings=["Ravenscar", "Ravenscourt"],
                keys=["ravenscar", "ravenscourt"],
                canonical="Ravenscar",
                same_entity=True,
                source="ai",
            )
        ],
        "speaker_groups": [],
        "apart": [
            {"candidate": "N001", "spellings": ["Ravenscar"]},
            {"candidate": "N001", "spellings": ["Ravenscourt"]},
            {"candidate": "N002", "spellings": ["Thorne", "Thorn"]},
        ],
    }

    assert apart_lines(result) == [
        "Thorne · Thorn: the AI judged these to be different names. No change."
    ]


def test_no_banned_word_appears_in_card_text(offline, recorded):
    for result in (offline, recorded):
        for text in all_text(build_cards(result)):
            for term in BANNED:
                assert term not in text.lower(), text


def test_status_transitions_on_a_name_card(offline):
    ravens = card(build_cards(offline), "name:N003.1")
    first, second = ravens.fixable

    assert status(ravens) == "Not decided"
    assert not is_decided(ravens)

    assert status(ravens, {first: True}) == "Fixing 1 of 2"
    assert is_decided(ravens, {first: True})
    assert status(ravens, {first: True, second: True}) == "Fixing 2 of 2"
    assert status(ravens, {first: False, second: False}) == "Keep as written"
    assert is_decided(ravens, {first: False, second: False})
    assert status(ravens, {first: False, second: False}, {ravens.id: "by_hand"}) == "Will fix by hand"
    assert status(ravens, hand={ravens.id: "not_a_problem"}) == "Not a problem"


def test_a_by_hand_card_has_no_ticks_and_takes_a_choice(recorded):
    place = card(build_cards(recorded), "place:F0012")

    assert place.fixable == ()
    assert status(place) == "Not decided"
    assert status(place, hand={place.id: "by_hand"}) == "Will fix by hand"
    assert status(place, hand={place.id: "not_a_problem"}) == "Not a problem"


def test_fix_all_ready_ticks_exactly_the_ready_places(recorded):
    cards = build_cards(recorded)
    ready = ready_places(cards)
    checked = tick(cards, {}, "ready")

    assert len(ready) == 10
    assert set(checked) == set(ready)
    assert all(checked.values())
    assert not set(checked) & set(check_places(cards))
    assert will_fix_count(cards, checked) == 10


def test_fix_these_too_ticks_exactly_the_check_lane(recorded):
    cards = build_cards(recorded)
    checked = tick(cards, tick(cards, {}, "ready"), "check")

    assert len(set(checked) - set(ready_places(cards))) == 1
    assert set(check_places(cards)) <= set(checked)
    assert card(cards, "time:E02.S01.time|DAY|NIGHT").fixable[0] in checked


def test_progress_uses_cards_and_ticked_places(recorded):
    cards = build_cards(recorded)
    checked = tick(cards, {}, "ready")

    assert decided_count(cards) == 0
    assert progress_text(cards) == (
        "Ready 0 of 6 · Check first 0 of 1 · Fix by hand 0 of 4 · 0 places will be fixed"
    )
    assert decided_count(cards, checked) == 6
    assert progress_text(cards, checked) == (
        "Ready 6 of 6 · Check first 0 of 1 · Fix by hand 0 of 4 · 10 places will be fixed"
    )
    assert metrics(cards)["issues"] == 11
    assert metrics(cards)["episodes"] == 4
    assert sum(metrics(cards)[key] for key in ("heard", "voice", "seen", "notes")) == 11


def test_the_change_list_holds_every_ticked_place(recorded):
    cards = build_cards(recorded)
    checked = tick(cards, {}, "ready")
    rows = change_rows(cards, checked)

    assert len(rows) == 10
    assert rows[0]["issue"] == "LORD EDMUND → EDMUND"
    assert rows[0]["was"] == "LORD EDMUND" and rows[0]["now"] == "EDMUND"
    assert any(row["was"] == "Ravenscourt" and row["now"] == "Ravenscar" for row in rows)
    assert all(row["episode"] for row in rows)
    assert change_rows(cards, {}) == []


def test_a_changed_spelling_moves_the_change_list(recorded):
    cards = build_cards(recorded)
    checked = tick(cards, {}, "ready")
    rows = change_rows(cards, checked, {"N003.1": "Ravensgar"})

    ravens = [row for row in rows if row["issue"].endswith("→ Ravenscar")]
    assert ravens and {row["now"] for row in ravens} == {"Ravensgar"}


def test_the_by_hand_list_holds_only_cards_marked_by_hand(recorded):
    cards = build_cards(recorded)
    place = card(cards, "place:F0012")

    assert by_hand_rows(cards, {}) == []
    rows = by_hand_rows(cards, {place.id: "by_hand", card(cards, "format:4").id: "not_a_problem"})
    assert len(rows) == 1
    assert rows[0]["title"] == "cottage or cabin?"


def test_the_download_summary_counts_places_episodes_and_issues(recorded):
    cards = build_cards(recorded)
    checked = tick(cards, {}, "ready")
    hand = {card(cards, "place:F0012").id: "by_hand"}

    assert download_summary(cards, checked, hand) == (
        "Fixing 10 places in 3 episodes across 6 issues. "
        "Not changed: 1 check-first and 4 by-hand issues. "
        "Your original files are not changed."
    )


def test_the_download_has_one_row_per_issue(recorded):
    cards = build_cards(recorded)
    checked = tick(cards, {}, "ready")
    issues = issue_rows(cards, checked)

    assert len(issues) == 6
    assert sum(item["places"] for item in issues) == 10
    assert all(item["kind"] and item["episodes"] and item["rows"] for item in issues)
    assert issues[0]["issue"] == "LORD EDMUND → EDMUND"
    assert issues[0]["episodes"] == (3,)
    assert all(row["where_label"] in ("Heard", "Seen", "Voice", "Notes") for item in issues for row in item["rows"])
    assert all("context" in row for item in issues for row in item["rows"])


def test_approvals_from_decisions_equal_the_expected_json(recorded):
    cards = build_cards(recorded)
    checked = tick(cards, {}, "ready")
    defaults = {group.id: group.canonical for group in result_groups(recorded)}
    overrides = {"N003.1": "Ravensgar"}

    approvals = canon_app.build_approvals(
        [finding.model_dump() for finding in recorded["findings"]], checked, overrides, defaults
    )

    expected = {finding.id: False for finding in recorded["findings"]}
    for fid in ready_places(cards):
        expected[fid] = True
    expected["canonical"] = {"N003.1": "Ravensgar"}
    assert approvals == expected
    assert len(approvals) == len(recorded["findings"]) + 1


def _finding(**overrides):
    values = {
        "id": "F0001",
        "detector": "joins",
        "severity": "visible",
        "source": "rule",
        "confidence": "high",
        "ep": 2,
        "scene": 1,
        "start": "00:00.00",
        "where": "E01 S02 -> E02 S01",
        "found": "NIGHT (CONTINUOUS) -> DAY (DAY)",
        "suggestion": "",
        "context": "",
        "group": "time",
        "edit": Edit(at="E02.S01.time", old="DAY", new="NIGHT"),
    }
    values.update(overrides)
    return Finding(**values)


def test_identical_time_fixes_merge_into_one_card():
    cards = build_cards(
        {
            "findings": [
                _finding(id="F0001"),
                _finding(id="F0002", where="E02 S02 -> E02 S03", ep=2, scene=3),
            ],
            "name_groups": [],
            "speaker_groups": [],
        }
    )

    assert len(cards) == 1
    assert cards[0].kind == "time" and cards[0].lane == "check"
    assert cards[0].places == ("F0001", "F0002")
    assert cards[0].sentence.startswith("Two cuts continue the same moment")


def test_format_rows_for_one_episode_share_one_card():
    rows = [
        _finding(id="F0001", detector="format", severity="notes", ep=4, where="Scene heading", found="SCENE HEADING", suggestion="Scene heading", group="format:Scene heading", edit=None),
        _finding(id="F0002", detector="format", severity="notes", ep=4, where="Scene start", found="SCENE START", suggestion="Scene start", group="format:Scene start", edit=None),
        _finding(id="F0003", detector="format", severity="notes", ep=5, where="Scene heading", found="SCENE HEADING", suggestion="Scene heading", group="format:Scene heading", edit=None),
    ]
    cards = build_cards({"findings": rows, "name_groups": [], "speaker_groups": []})

    assert sorted(len(card.places) for card in cards) == [1, 2]
    assert all(card.lane == "hand" for card in cards)
    by_id = {card.id: card for card in cards}
    assert by_id["format:4"].places == ("F0001", "F0002")
    assert "in two places" in by_id["format:4"].sentence
    assert by_id["format:5"].places == ("F0003",)


def test_the_badges_say_how_each_card_was_found(offline, recorded):
    found = {card.how_found for card in build_cards(offline) + build_cards(recorded)}

    assert found == {"No-AI guess", "Rule", "You decide", "AI · high", "AI · medium"}


def result_groups(result):
    return list(result["name_groups"]) + list(result["speaker_groups"])
