import random

from canon.bench import answer_match, answer_partial, score_answer_key, seed_match
from canon.seed import NAME_SUFFIXES, _name_variant


def test_a_vowel_swap_changes_one_character_and_avoids_known_keys():
    rng = random.Random(1)
    taken = {"ravenscar"}
    new = _name_variant("Ravenscar", "vowel_swap", rng, taken)
    assert new is not None
    assert new != "ravenscar"
    assert new not in taken
    assert len(new) == len("ravenscar")
    assert sum(a != b for a, b in zip(new, "ravenscar")) == 1


def test_a_suffix_swap_keeps_the_stem():
    new = _name_variant("Ravensbury", "suffix_swap", random.Random(1), {"ravensbury"})
    assert new is not None
    assert new.startswith("ravens")
    assert new[len("ravens"):] in NAME_SUFFIXES


def test_a_name_variant_never_returns_a_taken_key():
    rng = random.Random(2)
    assert _name_variant("God", "vowel_swap", rng, {"gad", "ged", "gid", "gud"}) is None


def test_answer_key_matches_a_name_term_in_the_right_episode():
    row = {"episodes": "4, 12", "where": "E12 S1 00:46.70", "what": "the frozen briars at Ravensgar Hall", "key_terms": "Ravensgar"}
    right = {"detector": "names", "ep": 12, "scene": 1, "found": "Ravensgar", "suggestion": "Ravenscar"}
    wrong_episode = {"detector": "names", "ep": 13, "scene": 1, "found": "Ravensgar", "suggestion": "Ravenscar"}
    wrong_text = {"detector": "names", "ep": 12, "scene": 1, "found": "Ravenscourt", "suggestion": "Ravenscar"}
    assert answer_match(row, right, "names")
    assert not answer_match(row, wrong_episode, "names")
    assert not answer_match(row, wrong_text, "names")


def test_answer_key_join_matches_the_scene_pair():
    row = {"episodes": "6", "where": "E6 S1 00:24.00 and S2 00:33.00", "what": "stone cottage then wooden cabin", "key_terms": "stone cottage; WOODEN CABIN"}
    right = {"detector": "joins", "ep": 6, "scene": 2, "where": "E06 S01 -> E06 S02", "found": "cottage -> cabin"}
    wrong_scene = {"detector": "joins", "ep": 6, "scene": 3, "where": "E06 S02 -> E06 S03", "found": "cottage -> cabin"}
    assert answer_match(row, right, "joins")
    assert not answer_match(row, wrong_scene, "joins")


SPEAKER_ROW = {
    "id": "4.1",
    "targeted_by": "D2",
    "episodes": "1-4",
    "where": "",
    "what": "Margaret — HOODED FIGURE (E1), MARGARET (E2-3), LADY MARGARET (E4) — Edmund names her in E2",
    "key_terms": "HOODED FIGURE; MARGARET; LADY MARGARET",
}


def line_row(found, suggestion, ep=2, group="S001"):
    return {"detector": "speakers", "ep": ep, "scene": 1, "where": f"E{ep:02d}.B003", "found": found,
            "suggestion": suggestion, "context": "", "group": group}


def test_a_speaker_row_is_not_found_by_a_name_that_only_appears_in_its_prose():
    # The old rule matched EDMUND because "Edmund" is in the row's description.
    assert not answer_match(SPEAKER_ROW, line_row("LORD EDMUND", "EDMUND"), "speakers")


def test_a_label_inside_a_longer_label_is_not_a_match():
    row = {"episodes": "1-4", "what": "Guard — GUARD (E1), STRANGER (E3)", "key_terms": "GUARD; STRANGER"}
    assert not answer_match(row, line_row("ROYAL GUARD", "EDMUND"), "speakers")


def test_a_speaker_row_needs_two_of_its_labels_linked_and_reports_what_is_missing():
    one_link = line_row("HOODED FIGURE", "MARGARET")
    assert answer_match(SPEAKER_ROW, one_link, "speakers")
    assert answer_partial(SPEAKER_ROW, [one_link], "speakers")          # LADY MARGARET is left out
    full = line_row("LADY MARGARET", "MARGARET", ep=4)
    assert not answer_partial(SPEAKER_ROW, [one_link, full], "speakers")
    scored = score_answer_key([dict(SPEAKER_ROW, confirmed="yes")], [one_link])
    assert scored["total"]["found"] == 1 and scored["total"]["partial"] == 1


def test_a_check_counts_by_the_labels_it_lists():
    check = {"detector": "speakers", "ep": 1, "scene": 0, "where": "HOODED FIGURE / MARGARET / STRANGER",
             "found": "HOODED FIGURE and MARGARET and STRANGER may be one character", "suggestion": "", "group": "unsure:x"}
    assert answer_match(SPEAKER_ROW, check, "speakers")
    only_one = dict(check, where="HOODED FIGURE / STRANGER")
    assert not answer_match(SPEAKER_ROW, only_one, "speakers")


def test_a_one_label_row_names_its_character_as_the_second_label():
    row = {"episodes": "3", "what": "Edmund (again?) — STRANGER in the hall (E3)", "key_terms": "STRANGER"}
    assert answer_match(row, line_row("STRANGER", "EDMUND", ep=3), "speakers")
    assert not answer_match(row, line_row("STRANGER", "MARGARET", ep=3), "speakers")


def test_a_one_label_two_people_row_needs_a_veto_check_on_that_label():
    row = {"episodes": "2, 4", "what": "One label, two people: GUARD is Edmund's man in E2 and a stranger in E4",
           "key_terms": "GUARD"}
    veto = {"detector": "speakers", "ep": 4, "scene": 1, "where": "GUARD", "group": "veto:GUARD",
            "found": "GUARD matched to EDMUND, but both speak in E04 S01 00:10.00", "suggestion": ""}
    assert answer_match(row, veto, "speakers")
    assert not answer_match(row, line_row("GUARD", "ROYAL GUARD", ep=4), "speakers")


def test_a_name_term_must_match_whole_words():
    row = {"episodes": "2", "where": "E2 S1", "what": "Hart written once for Hartley", "key_terms": "Hart"}
    longer = {"detector": "names", "ep": 2, "scene": 1, "found": "Hartwell", "suggestion": "Hartley", "where": "E02.B004"}
    assert not answer_match(row, longer, "names")


def test_seed_match_needs_the_injected_word_at_the_injected_episode():
    item = {
        "detector": "names",
        "new": "Ravensgar",
        "episodes": [2],
        "occurrences": [{"ep": 2, "scene": 1}],
    }
    right = {"detector": "names", "ep": 2, "scene": 1, "found": "Ravensgar", "suggestion": "Ravenscar"}
    wrong_episode = {"detector": "names", "ep": 3, "scene": 1, "found": "Ravensgar", "suggestion": "Ravenscar"}
    assert seed_match(item, right)
    assert not seed_match(item, wrong_episode)


def test_seed_speaker_split_is_found_even_when_the_group_is_reported_at_its_first_line():
    item = {
        "detector": "speakers",
        "new": "LADY HARTWELL",
        "episodes": [4],
        "occurrences": [{"ep": 4, "scene": 1}],
    }
    note = {
        "detector": "speakers",
        "ep": 1,
        "scene": 1,
        "found": "EDMUND and LADY HARTWELL are one character, hidden on purpose",
        "suggestion": "Cast one voice for all of them.",
    }
    assert seed_match(item, note)


def test_seed_typo_matches_only_the_injected_spelling():
    item = {"detector": "spelling", "new": "seizzure", "episodes": [2], "occurrences": [{"ep": 2, "scene": 1}]}
    right = {"detector": "spelling", "ep": 2, "scene": 1, "found": "seizzure", "suggestion": "seizure"}
    wrong = {"detector": "spelling", "ep": 2, "scene": 1, "found": "gaol", "suggestion": "goal"}
    assert seed_match(item, right)
    assert not seed_match(item, wrong)
