from pathlib import Path

import pytest

from canon import judge, llm
from canon.mentions import build_index
from canon.names import build_candidates
from canon.parse import parse_show
from canon.pipeline import run
from canon.speakers import speaker_stats

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "ravenscar"
RAVENS = frozenset({"Ravenscar", "Ravenscourt", "Ravensgar"})
THORNE = frozenset({"Thorne", "Thorn", "Thomson"})
MARGARET = frozenset({"Margaret", "Margarett"})


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """The full pipeline on the sample, replaying the live answers in samples/recorded."""
    return run(str(SAMPLE), str(tmp_path_factory.mktemp("sample")), model="fake/recorded")


@pytest.fixture(scope="module")
def sample():
    show = parse_show(SAMPLE)
    index = build_index(show)
    return show, index, speaker_stats(show, index), judge.co_speaking_scenes(show)


def test_the_judges_find_exactly_the_planted_names(recorded):
    names = {frozenset(g.spellings): g for g in recorded["name_groups"]}

    assert set(names) == {RAVENS, THORNE, MARGARET}
    assert names[RAVENS].canonical == "Ravenscar"
    assert names[THORNE].canonical == "Thorne"
    assert names[MARGARET].canonical == "Margaret"
    assert all(g.source == "ai" for g in names.values())
    assert any(set(a["spellings"]) <= {"Hartwell", "Hartley"} for a in recorded["apart"])


def test_the_judges_find_exactly_the_planted_speakers(recorded):
    speakers = {frozenset(g.labels): g for g in recorded["speaker_groups"]}

    assert set(speakers) == {frozenset({"EDMUND", "LORD EDMUND"}), frozenset({"HOODED FIGURE", "MARGARET"})}
    assert speakers[frozenset({"HOODED FIGURE", "MARGARET"})].intentional


def test_the_spelling_judge_keeps_only_typos(recorded):
    verdicts = {v["word"]: v for v in recorded["verdicts"]}
    typos = {f.found: f.suggestion for f in recorded["findings"] if f.detector == "spelling"}

    assert typos == {"seizzure": "seizure", "gentelman": "gentleman"}
    assert verdicts["gaol"]["verdict"] == "valid"
    assert "thee" not in verdicts or verdicts["thee"]["verdict"] == "valid"


def test_every_finding_says_how_it_was_found(recorded):
    assert recorded["findings"]
    assert all(f.source in {"rule", "ai", "check"} for f in recorded["findings"])
    assert recorded["manifest"]["complete"] is True


def test_the_veto_splits_labels_that_speak_in_one_scene(sample):
    _, _, stats, co = sample

    components, vetoed = judge.apply_veto(["GUARD", "ROYAL GUARD"], stats, co)

    assert sorted(map(sorted, components)) == [["GUARD"], ["ROYAL GUARD"]]
    assert vetoed and vetoed[0][2].startswith("E02 S01")


def test_a_confident_merge_the_veto_breaks_becomes_a_check(sample):
    _, _, stats, co = sample

    groups, checks = judge._speaker_groups(
        [(["STRANGER", "EDMUND"], "EDMUND", False, "high", "same man", "ai")], stats, co
    )

    assert groups == []
    assert [(c.source, c.where) for c in checks] == [("check", "STRANGER")]
    assert "E04 S01" in checks[0].found


def test_an_unsure_merge_is_a_check_not_a_group(sample):
    _, _, stats, co = sample

    groups, checks = judge._speaker_groups(
        [(["STRANGER", "ROYAL GUARD"], "STRANGER", False, "low", "both serve the Crown", "ai")],
        stats,
        co,
    )

    assert groups == []
    assert len(checks) == 1 and checks[0].source == "check"


def test_a_bad_answer_is_retried_once_then_the_run_fails(monkeypatch):
    calls = []
    bad = judge.NamesOut(groups=[{"spellings": ["Ravenscar", "Ravenswood"], "same_entity": True, "canonical": "Ravenscar"}])

    def fake_generate(stage, prompt, schema, **kwargs):
        calls.append(prompt)
        return bad

    monkeypatch.setattr(judge.llm, "generate", fake_generate)
    spec = judge.ModelSpec(routes=["deepseek/deepseek-flash"])

    with pytest.raises(llm.LLMFailed):
        judge.ask("names", "PROMPT", judge.NamesOut, spec, judge._check_names(["Ravenscar"]))

    assert len(calls) == 2
    assert '"Ravenswood" is not one of the spellings given' in calls[1]


def test_a_corrected_second_answer_is_accepted(monkeypatch):
    answers = [
        judge.NamesOut(groups=[{"spellings": ["ravenscar"], "same_entity": False}]),
        judge.NamesOut(groups=[{"spellings": ["Ravenscar", "Ravensgar"], "same_entity": True, "canonical": "Ravenscar"}]),
    ]
    monkeypatch.setattr(judge.llm, "generate", lambda *a, **k: answers.pop(0))
    spec = judge.ModelSpec(routes=["deepseek/deepseek-flash"])

    out = judge.ask("names", "PROMPT", judge.NamesOut, spec, judge._check_names(["Ravenscar", "Ravensgar"]))

    assert out.groups[0].spellings == ["Ravenscar", "Ravensgar"]


def test_a_bad_key_is_not_hidden_by_the_majority_rule(sample, monkeypatch):
    show, index, _, _ = sample
    candidates, _ = build_candidates(index)

    def bad_key(*args, **kwargs):
        raise llm.LLMFailed(
            "DeepSeek rejected the API key (HTTP 401).",
            kind="auth",
            hint="Paste a valid DeepSeek key in the sidebar.",
        )

    monkeypatch.setattr(judge.llm, "generate", bad_key)
    spec = judge.ModelSpec(routes=["deepseek/deepseek-flash"])

    with pytest.raises(llm.LLMFailed) as error:
        judge.judge_names(candidates[:1], index, show, spec)

    assert error.value.kind == "auth"
    assert "rejected the API key" in str(error.value)
    assert "0 of 3" not in str(error.value)


def test_the_majority_error_keeps_the_underlying_reason(sample, monkeypatch):
    show, index, _, _ = sample
    candidates, _ = build_candidates(index)

    def bad_json(*args, **kwargs):
        raise llm.LLMFailed(
            "DeepSeek did not return usable JSON.",
            kind="response",
            detail="deepseek/deepseek-flash returned invalid JSON 4 times",
        )

    monkeypatch.setattr(judge.llm, "generate", bad_json)
    spec = judge.ModelSpec(routes=["deepseek/deepseek-flash"])

    with pytest.raises(llm.LLMFailed) as error:
        judge.judge_names(candidates[:1], index, show, spec)

    assert "0 of 3" in str(error.value)
    assert "did not return usable JSON" in str(error.value)
    assert "invalid JSON" in error.value.detail


def test_code_only_mode(sample):
    show, index, stats, _ = sample
    candidates, _ = build_candidates(index)

    names = {frozenset(g.spellings): g for g in judge.code_names(candidates, index)}
    speakers, _ = judge.code_speakers(stats, show)

    assert names[MARGARET].canonical == "Margaret"           # most mentions, not most spoken
    assert all(g.source == "code" for g in names.values())
    assert [sorted(g.labels) for g in speakers] == [["EDMUND", "LORD EDMUND"]]


def _try(*groups):
    return judge.NamesOut(
        groups=[
            {"spellings": list(g), "same_entity": len(g) > 1, "canonical": g[0], "confidence": "high"}
            for g in groups
        ]
    )


def test_majority_vote_groups_what_most_tries_agree_on():
    spellings = ["Ravenscar", "Ravenscourt", "Ravensgar", "Hartwell"]
    tries = [
        _try(("Ravenscar", "Ravenscourt"), ("Ravensgar",), ("Hartwell",)),          # splits one spelling off
        _try(("Ravenscar", "Ravenscourt", "Ravensgar"), ("Hartwell",)),
        _try(("Ravenscar", "Ravenscourt", "Ravensgar"), ("Hartwell",)),
    ]

    out = judge.majority(tries, spellings)
    together = [g for g in out.groups if g.same_entity]

    assert [sorted(g.spellings) for g in together] == [["Ravenscar", "Ravenscourt", "Ravensgar"]]
    assert together[0].confidence == "medium"                 # only 2 of 3 made exactly this group
    assert "2 of 3 tries" in together[0].reason
    assert any(g.spellings == ["Hartwell"] and not g.same_entity for g in out.groups)


def test_a_group_every_try_agrees_on_keeps_its_confidence():
    tries = [_try(("Thorne", "Thorn"),) for _ in range(3)]

    out = judge.majority(tries, ["Thorne", "Thorn"])

    assert out.groups[0].confidence == "high" and out.groups[0].canonical == "Thorne"
