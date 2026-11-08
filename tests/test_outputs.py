import csv
import json
import shutil
from pathlib import Path

import pytest

from canon import audit, edits, joins, report
from canon.models import Episode, Finding, Scene, Show
from canon.parse import parse_show, read_input
from canon.patch import PatchError, locate, patch
from canon.pipeline import input_hash, run

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "ravenscar"


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """The full pipeline on the sample, replaying the live answers in samples/recorded."""
    out = tmp_path_factory.mktemp("run")
    result = run(str(SAMPLE), str(out), model="fake/recorded")
    return out, result


def _by(findings, **want):
    return [f for f in findings if all(getattr(f, k) == v for k, v in want.items())]


def _approve(run_dir: Path, decisions: dict, default: bool = True) -> Path:
    findings = json.loads((run_dir / "findings.json").read_text(encoding="utf-8"))
    approvals = {f["id"]: default for f in findings}
    approvals.update(decisions)
    path = run_dir / "approvals.json"
    path.write_text(json.dumps(approvals), encoding="utf-8")
    return path


def _patched_text(run_dir: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted((run_dir / "patched").glob("*.txt")))


# ---- findings ----

def test_planted_findings_carry_exact_beat_ids_and_times(recorded):
    _, result = recorded
    findings = result["findings"]
    show = parse_show(str(SAMPLE))
    beat = {(b.ep, f"{b.start:05.2f}", b.kind): b.id for e in show.episodes for b in e.beats}

    def at(ep, start):
        return beat[(ep, start, "line")]

    expected = [
        ("names", "Ravenscourt", 2, "10.00"),
        ("names", "Thorn", 2, "10.00"),
        ("names", "Ravensgar", 3, "05.20"),
        ("names", "Thomson", 3, "17.20"),
        ("names", "Margarett", 4, "26.20"),
        ("spelling", "seizzure", 2, "23.60"),
        ("spelling", "gentelman", 3, "21.40"),
    ]
    for detector, found, ep, start in expected:
        rows = _by(findings, detector=detector, found=found)
        assert len(rows) == 1, found
        assert rows[0].where == at(ep, start) and rows[0].start == f"00:{start}", found
    lord = _by(findings, detector="speakers", found="LORD EDMUND")
    assert [f.start for f in lord] == ["00:05.20", "00:21.40", "00:48.20"]
    assert all(f.suggestion == "EDMUND" and f.severity == "voice" for f in lord)
    [flip] = _by(findings, detector="joins", group="time")
    assert flip.where == "E01 S02 -> E02 S01" and flip.edit.at == "E02.S01.time" and flip.edit.new == "NIGHT"


def test_names_keep_apart_what_the_judge_kept_apart(recorded):
    _, result = recorded
    assert not [f for f in result["findings"] if f.found.startswith(("Hartwell", "Hartley"))]
    [casting] = _by(result["findings"], detector="speakers", group="S001")
    assert casting.edit is None and "one voice" in casting.suggestion


def test_findings_are_sorted_by_severity_then_episode_then_time(recorded):
    _, result = recorded
    order = {s: i for i, s in enumerate(["audible", "voice", "visible", "notes"])}
    keys = [(order[f.severity], f.ep, f.start) for f in result["findings"]]
    assert keys == sorted(keys)
    assert [f.id for f in result["findings"]] == [f"F{i:04d}" for i in range(1, len(keys) + 1)]


def test_outputs_are_written(recorded):
    out, result = recorded
    for name in ("findings.csv", "cast.csv", "summary.md", "report.html", "manifest.json", "groups.json"):
        assert (out / name).exists(), name
    with open(out / "findings.csv", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(result["findings"])
    assert rows[0]["fix"].startswith(rows[0]["where"])


def test_the_summary_quotes_no_script_text(recorded):
    out, result = recorded
    summary = (out / "summary.md").read_text(encoding="utf-8")
    words = {f.found for f in result["findings"]} | {s for g in result["name_groups"] for s in g.spellings}
    assert not [w for w in words if w in summary]


def test_the_cast_sheet_has_one_row_per_character(recorded):
    out, _ = recorded
    with open(out / "cast.csv", encoding="utf-8-sig", newline="") as fh:
        cast = {r["character"]: r for r in csv.DictReader(fh)}
    assert cast["EDMUND"]["lines_per_label"] == "EDMUND 6; LORD EDMUND 3"
    assert "one voice" in cast["EDMUND"]["note"]
    assert "on purpose" in cast["MARGARET"]["note"]
    assert {"GUARD", "ROYAL GUARD", "STRANGER"} <= set(cast)


def test_the_report_has_every_tab_and_escapes_text(recorded):
    out, result = recorded
    page = (out / "report.html").read_text(encoding="utf-8")
    for tab in ("Names", "Speakers", "Scene joins", "Spelling", "Format"):
        assert f">{tab} (" in page
    nasty = Finding(detector="format", severity="notes", source="rule", ep=1, scene=0, start="00:00.00",
                    where="x", found="<b>", context="<script>alert(1)</script>")
    page = report.report_html(result["manifest"], [nasty], [], [], [], [], [])
    assert "<script>alert(1)" not in page and "&lt;script&gt;" in page


# ---- edits ----

def test_word_matching_keeps_possessives_and_skips_longer_words():
    assert len(edits.word_spans("Thorne's ledger, the Thornes, Thornewood and Thorne.", "Thorne")) == 2
    assert edits.match_case("RAVENSCOURT", "Ravenscar") == "RAVENSCAR"
    assert edits.match_case("Seizzure", "seizure") == "Seizure"
    assert edits.match_case("seizzure", "Seizure") == "seizure"


def test_locate_agrees_with_the_parser():
    show = parse_show(str(SAMPLE))
    episodes = {e.num: e for e in show.episodes}
    for name, content in read_input(str(SAMPLE)):
        ep, spans = locate(content, name)
        episode = episodes[ep]
        for b in episode.beats:
            start, end = spans[b.id]
            assert " ".join(content[start:end].split()) == b.text
            if b.kind == "line":
                start, end = spans[f"{b.id}.label"]
                assert content[start:end] == b.speaker
        for s in episode.scenes:
            start, end = spans[f"E{ep:02d}.S{s.num:02d}.heading"]
            assert content[start:end] == s.heading
            start, end = spans[f"E{ep:02d}.S{s.num:02d}.time"]
            assert content[start:end].upper() == s.time
        start, end = spans[f"E{ep:02d}.notes"]
        assert content[start:end] == episode.continuity


def test_a_time_flip_in_a_chain_fixes_the_odd_scene():
    def ep(num, time, notes):
        scene = Scene(ep=num, num=1, heading=f"INT. HUT – {time}", start=0, end=10, int_ext="INT", place="HUT", time=time)
        return Episode(num=num, continuity=notes, scenes=[scene])

    show = Show(title="t", episodes=[ep(1, "NIGHT", ""), ep(2, "DAY", "Picks up immediately."), ep(3, "NIGHT", "Picks up immediately.")])
    flips = [f for f in joins.find_joins(show) if f.group == "time"]
    assert len(flips) == 2
    assert {(f.edit.at, f.edit.old, f.edit.new) for f in flips} == {("E02.S01.time", "DAY", "NIGHT")}


# ---- patch ----

def test_patch_applies_approved_fixes_and_leaves_rejected_ones(recorded):
    out, result = recorded
    [thomson] = _by(result["findings"], found="Thomson")
    before = input_hash(str(SAMPLE))
    _approve(out, {thomson.id: False})

    stats = patch(str(out))

    text = _patched_text(out)
    assert stats["problems"] == [] and stats["rejected"] == 1
    assert "ravenscourt" not in text.lower() and "ravensgar" not in text.lower()
    assert "Lord Hartwell's orders" in text
    assert "seizzure" not in text and "Crown's seizure" in text
    assert "gentleman's daughter" in text
    assert "Scene heading: EXT. HALL GATES – NIGHT" in text
    assert "Edmund Thomson" in text                      # rejected, so untouched
    assert "**LORD EDMUND" not in text and "**HOODED FIGURE" in text
    assert input_hash(str(SAMPLE)) == before            # the input is never touched
    assert "-Scene heading: EXT. HALL GATES – DAY" in (out / "patch.diff").read_text(encoding="utf-8")
    patched = parse_show(str(out / "patched"))
    assert sum(len(e.beats) for e in patched.episodes) == sum(len(e.beats) for e in parse_show(str(SAMPLE)).episodes)


def test_nothing_approved_changes_nothing(recorded):
    out, _ = recorded
    _approve(out, {}, default=False)

    stats = patch(str(out))

    assert stats["applied"] == 0 and stats["files_changed"] == 0
    assert (out / "patch.diff").read_text(encoding="utf-8") == ""


def test_a_canonical_override_decides_the_whole_group(recorded):
    out, result = recorded
    group = next(g for g in result["name_groups"] if "Ravensgar" in g.spellings)
    [kept] = _by(result["findings"], found="Ravenscourt")
    approvals = {f.id: False for f in result["findings"] if f.group != group.id}
    approvals[kept.id] = False                              # a row set to false stays as written
    approvals["canonical"] = {group.id: "Ravensgar"}
    (out / "approvals.json").write_text(json.dumps(approvals), encoding="utf-8")

    patch(str(out))

    text = _patched_text(out)
    assert "ravenscar" not in text.lower()                  # the former canonical changes too
    assert "INT. RAVENSGAR HALL – LIBRARY – DAY" in text     # headings keep their capitals
    assert "Thorn has come home to Ravenscourt Hall" in text  # rejected row, and other groups untouched


def test_patch_refuses_a_changed_input(tmp_path):
    source = tmp_path / "ravenscar"
    shutil.copytree(SAMPLE, source)
    out = tmp_path / "run"
    run(str(source), str(out), model="fake/recorded")
    _approve(out, {})
    first = sorted(source.glob("*.txt"))[0]
    first.write_text(first.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(PatchError, match="input changed"):
        patch(str(out))


# ---- audit files ----

def test_the_audit_template_leaves_labels_empty_and_keeps_filled_ones(recorded, tmp_path):
    out, result = recorded
    path = audit.audit_template(str(out), audit_dir=str(tmp_path))

    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert path.name == f"labels_{out.name}.csv"
    assert len(rows) == len(result["findings"])
    assert all(r["correct"] == "" and r["note"] == "" for r in rows)

    rows[0]["correct"] = "yes"
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(audit.AuditError):
        audit.audit_template(str(out), audit_dir=str(tmp_path))


EVIDENCE = """# Evidence

## 1. Spoken name mistakes

| # | What | Where | Confidence |
|---|---|---|---|
| 1.1 | The hall is **Ravensgar** once, Ravenscar elsewhere | E3 S1 00:05.20 | Certain |

## 2. Typos and grammar in spoken lines

| # | Line as written | Should be | Where | Confidence |
|---|---|---|---|---|
| 2.1 | "the Crown's **seizzure**" | seizure | E2 S1 00:23.60 | Certain |
| 2.2 | "he **go** home" | goes | E3 S2 00:48.20 | Likely |

## 4. Speaker labels

None of these pairs ever speak in the same scene.

| Character | Labels | Proof | Confidence |
|---|---|---|---|
| Edmund | EDMUND (E1–E2, E4), LORD EDMUND (E3) | same man | Certain |

**One label, two people:** STRANGER is Edmund in E1 and a messenger in E4. Certain.

## 8. Format

E4 writes `SCENE HEADING:` in capitals. Certain (low impact).

## 9. Name table

| Name | Spoken |
|---|---|
| Ravenscar | 2 |
"""


def test_the_answer_key_copies_the_items_and_leaves_confirmed_empty(tmp_path):
    source = tmp_path / "EVIDENCE.md"
    source.write_text(EVIDENCE, encoding="utf-8")

    path, items = audit.answer_key(str(source), str(tmp_path / "answer_key.csv"))

    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = {r["id"]: r for r in csv.DictReader(fh)}
    assert list(rows) == ["1.1", "2.1", "2.2", "4.1", "4.2", "8.1"]
    assert {k: r["targeted_by"] for k, r in rows.items()} == {
        "1.1": "D1", "2.1": "D4", "2.2": "v2", "4.1": "D2", "4.2": "D2", "8.1": "format",
    }
    assert all(r["confirmed"] == "" and r["note"] == "" for r in rows.values())
    assert rows["4.1"]["episodes"] == "1-4" and "LORD EDMUND" in rows["4.1"]["key_terms"]
    assert rows["4.2"]["confidence"] == "Certain" and rows["2.2"]["confidence"] == "Likely"
    assert rows["2.1"]["key_terms"] == "seizzure"
