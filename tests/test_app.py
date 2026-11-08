import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import app as canon_app
from canon.issues import build_cards, issue_rows
from canon.parse import parse_show

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
SAMPLE = ROOT / "samples" / "ravenscar"


def upload(name: str, data: bytes):
    return SimpleNamespace(name=name, getvalue=lambda: data)


def test_build_approvals_writes_every_row_and_only_changed_canonicals():
    findings = [{"id": "F0001"}, {"id": "F0002"}, {"id": "F0003"}]

    approvals = canon_app.build_approvals(
        findings,
        {"F0001": True},
        {"N001": "Ravensgar", "N002": "Thorne"},
        {"N001": "Ravenscar", "N002": "Thorne"},
    )

    assert approvals == {
        "F0001": True,
        "F0002": False,
        "F0003": False,
        "canonical": {"N001": "Ravensgar"},
    }


def test_build_approvals_omits_canonical_when_nothing_changed():
    approvals = canon_app.build_approvals(
        [{"id": "F0001"}], {"F0001": False}, {"N001": "Ravenscar"}, {"N001": "Ravenscar"}
    )

    assert approvals == {"F0001": False}
    assert "canonical" not in approvals


def test_model_kwargs_maps_each_choice():
    assert canon_app.model_kwargs("Code only") == {"no_model": True}
    assert canon_app.model_kwargs("Gemini free") == {"model": "gemini/gemini-3.5-flash-lite"}
    assert canon_app.model_kwargs("DeepSeek") == {"model": "deepseek/deepseek-flash"}
    assert canon_app.model_kwargs("Custom", "openai/their-model", "https://example.test/v1") == {
        "model": "openai/their-model",
        "base_url": "https://example.test/v1",
    }


def test_model_kwargs_rejects_a_custom_route_without_a_provider():
    with pytest.raises(ValueError):
        canon_app.model_kwargs("Custom", "just-a-name")


def test_key_warning_flags_a_key_from_another_provider():
    assert canon_app.key_warning("DeepSeek", "fb0530f7-78c2-4472-aa9f-cd0ce5aaaaff")
    assert canon_app.key_warning("DeepSeek", "sk-abc") is None
    assert canon_app.key_warning("Gemini free", "AIzaSy-test") is None
    assert canon_app.key_warning("Code only", "anything") is None


def test_a_bad_key_is_shown_as_a_readable_card(monkeypatch):
    from canon import llm

    def bad_key(*args, **kwargs):
        raise llm.LLMFailed(
            "DeepSeek rejected the API key (HTTP 401).",
            kind="auth",
            hint="Paste a valid DeepSeek key in the sidebar.",
            detail="Authentication Fails, Your api key: ****aaff is invalid",
        )

    monkeypatch.setattr("canon.pipeline.run", bad_key)
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    at.radio(key="input_mode").set_value("Sample show")
    at.radio(key="model_choice").set_value("DeepSeek")
    at.button(key="run_button").click().run()

    assert not at.exception
    assert any("rejected the API key" in error.value for error in at.error)
    assert any("What to do next" in markdown.value for markdown in at.markdown)
    assert any("DeepSeek" in markdown.value for markdown in at.markdown)
    assert at.session_state["canon_error"]["title"] == (
        "The model provider rejected the API key"
    )


def test_save_uploads_writes_txt_files_and_keeps_duplicates_apart(tmp_path):
    files = [upload("episode-1.txt", b"one"), upload("episode-1.txt", b"two")]

    folder = Path(canon_app.save_uploads(files, tmp_path / "input"))

    assert (folder / "episode-1.txt").read_bytes() == b"one"
    assert (folder / "episode-1-2.txt").read_bytes() == b"two"


def test_save_uploads_writes_one_zip(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("episode-1.txt", "one")
    path = Path(canon_app.save_uploads([upload("show.zip", buffer.getvalue())], tmp_path / "input"))

    assert path.suffix == ".zip" and zipfile.is_zipfile(path)


def test_save_uploads_rejects_a_zip_with_txt_files(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("episode-1.txt", "one")

    with pytest.raises(ValueError, match="either one .zip"):
        canon_app.save_uploads(
            [upload("show.zip", buffer.getvalue()), upload("episode-2.txt", b"two")], tmp_path
        )


def test_zip_patch_contains_the_patched_folder_diff_and_json(tmp_path):
    run = tmp_path / "run"
    (run / "patched").mkdir(parents=True)
    (run / "patched" / "episode-1.txt").write_text("fixed", encoding="utf-8")
    (run / "patch.diff").write_text("a diff", encoding="utf-8")
    (run / "patch.json").write_text(json.dumps({"applied": 1}), encoding="utf-8")

    with zipfile.ZipFile(io.BytesIO(canon_app.zip_patch(run))) as archive:
        assert sorted(archive.namelist()) == [
            "patch.diff",
            "patch.json",
            "patched/episode-1.txt",
        ]


def test_zip_patch_refuses_a_missing_patch(tmp_path):
    with pytest.raises(FileNotFoundError):
        canon_app.zip_patch(tmp_path)


def test_run_totals_reads_the_manifest():
    manifest = {
        "no_model": False,
        "model": ["gemini/gemini-3.5-flash-lite"],
        "cost_usd": 0.0,
        "llm": {
            "names": {"calls": 2, "cached": 1, "input_tokens": 10, "output_tokens": 5},
            "spelling": {"calls": 1, "cached": 0, "input_tokens": 3, "output_tokens": 2},
        },
    }

    assert canon_app.run_totals(manifest) == {
        "model": "gemini/gemini-3.5-flash-lite",
        "calls": 3,
        "cached": 1,
        "tokens_in": 13,
        "tokens_out": 7,
        "cost_usd": 0.0,
    }


def test_friendly_model_names_hide_the_route():
    assert canon_app.friendly_model("deepseek/deepseek-flash") == "DeepSeek"
    assert canon_app.friendly_model("gemini/gemini-3.5-flash-lite") == "Gemini"
    assert canon_app.friendly_model("fake/recorded") == "recorded AI answers"
    assert canon_app.friendly_model("openai/their-model") == "openai/their-model"


def test_the_input_line_counts_spoken_lines():
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    at.radio(key="input_mode").set_value("Sample show")
    at.radio(key="model_choice").set_value("No AI")
    at.button(key="run_button").click().run()

    show = parse_show(SAMPLE)
    spoken = sum(1 for episode in show.episodes for beat in episode.beats if beat.kind == "line")
    scenes = sum(len(episode.scenes) for episode in show.episodes)
    expected = f"Read {len(show.episodes)} episodes · {spoken:,} spoken lines · {scenes} scenes."
    captions = [item.value for item in at.caption]

    assert expected in captions
    assert all("11181 lines" not in value for value in captions)


def test_the_app_survives_a_build_without_progress_reporting(monkeypatch):
    """A deploy can pair this app.py with an older canon.llm. Progress is cosmetic."""
    from canon import llm

    monkeypatch.delattr(llm, "set_progress", raising=False)

    at = AppTest.from_file(str(APP), default_timeout=180).run()
    at.radio(key="input_mode").set_value("Sample show")
    at.radio(key="model_choice").set_value("No AI")
    at.button(key="run_button").click().run()

    assert not at.exception
    assert not at.error
    assert at.session_state["canon_run"]["manifest"]["complete"] is True


def _click(at, label: str):
    button = next(item for item in at.button if item.label.startswith(label))
    return button.click()


def test_the_home_page_offers_the_one_click_sample():
    at = AppTest.from_file(str(APP), default_timeout=180).run()

    assert "Try the sample now" in [button.label for button in at.button]
    _click(at, "Try the sample now").run(timeout=180)

    assert not at.exception
    assert not at.error
    assert at.session_state["page"] == "Review"
    assert at.session_state["canon_run"]["checker"] == "Recorded AI"
    assert at.session_state["canon_run"]["manifest"]["complete"] is True

    # The demo uses the sample's recorded AI answers, so the one-click fix has work to do.
    fix_all = next(item for item in at.button if item.label.startswith("✓ Fix all ready"))
    assert not fix_all.disabled
    assert fix_all.label != "✓ Fix all ready (0 places)"
    fix_all.click().run(timeout=180)
    assert not at.exception
    assert any(
        "All 6 ready issues are set to be fixed (10 places). Next: look at Check first (1)."
        in item.value
        for item in at.success
    )
    run_id = at.session_state["canon_run"]["run_id"]
    ticked = [key for key in at.session_state if str(key).startswith(f"fix|{run_id}|") and at.session_state[key]]
    assert ticked


def test_the_app_runs_the_sample_offline_and_shows_cards_and_metrics():
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    at.radio(key="input_mode").set_value("Sample show")
    at.radio(key="model_choice").set_value("No AI")
    at.button(key="run_button").click().run()

    assert not at.exception
    assert not at.error
    state = at.session_state["canon_run"]
    cards = build_cards(state["result"])
    assert [(m.label, m.value) for m in at.metric] == [
        ("Issues", str(len(cards))),
        ("Heard", "8"),
        ("Voice", "1"),
        ("Seen", "3"),
        ("Episodes", "4"),
    ]
    assert not at.tabs
    headers = [item.value for item in at.subheader]
    assert any(value.startswith("Ready to fix") for value in headers)
    assert any(value.startswith("Check first") for value in headers)
    assert any(value.startswith("Fix by hand") for value in headers)
    shown = " ".join(markdown.value for markdown in at.markdown)
    assert "Canon is sure and can change these for you. One click: Fix all ready." in shown
    assert "Canon can change these, but no AI checked them. Read each card, then press Fix." in shown
    assert "Canon can't change these for you (building names, header format" in shown
    assert any(value.startswith("Ready 0 of") for value in (item.value for item in at.caption))
    assert len([button for button in at.button if button.label == "Details"]) == len(cards)
    shown = " ".join(markdown.value for markdown in at.markdown)
    assert all(card.title in shown for card in cards)
    assert any("**Problem:**" in markdown.value for markdown in at.markdown)
    assert any("**Why this fix:**" in markdown.value for markdown in at.markdown)
    assert at.checkbox == []


def test_the_download_page_makes_the_fixed_zip():
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    at.radio(key="input_mode").set_value("Sample show")
    at.radio(key="model_choice").set_value("No AI")
    at.button(key="run_button").click().run()
    run_id = at.session_state["canon_run"]["run_id"]

    _click(at, "Fix these too").run()
    _click(at, "Next: download").run()
    _click(at, "Make fixed scripts").run()

    assert not at.exception
    assert not at.error
    packed = at.session_state[f"fixed|{run_id}"]
    with zipfile.ZipFile(io.BytesIO(packed["zip"])) as archive:
        names = archive.namelist()
    assert "CHANGES.csv" in names
    assert "changes.diff" in names
    assert "fix_by_hand.txt" in names
    assert any(name.startswith("fixed_scripts/") for name in names)
    assert any(button.label.startswith("⬇ Download fixed_scripts.zip") for button in at.download_button)


def test_the_download_page_has_one_summary_row_per_issue():
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    _click(at, "Try the sample now").run(timeout=180)
    state = at.session_state["canon_run"]
    run_id = state["run_id"]
    cards = build_cards(state["result"])

    _click(at, "✓ Fix all ready").run()
    fix = {
        str(key)[len(f"fix|{run_id}|"):]: bool(value)
        for key, value in at.session_state.items()
        if str(key).startswith(f"fix|{run_id}|") and value
    }
    issues = issue_rows(cards, fix)
    _click(at, "Next: download").run()

    assert not at.exception
    summary = at.dataframe[0].value
    assert list(summary.columns) == ["Issue", "Kind", "Places", "Episodes"]
    assert len(summary) == len(issues) == 6
    assert summary["Places"].sum() == 10
    assert set(summary["Kind"]) == {"NAME", "VOICE", "SPOKEN TYPO"}
    assert [label.label for label in at.expander if " · " in label.label] == [
        f"{item['issue']} · {item['places']} place{'s' if item['places'] != 1 else ''}"
        for item in issues
    ]


def test_details_dialog_keeps_decisions_when_it_closes():
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    at.radio(key="input_mode").set_value("Sample show")
    at.radio(key="model_choice").set_value("No AI")
    at.button(key="run_button").click().run()
    run_id = at.session_state["canon_run"]["run_id"]

    next(
        button for button in at.button
        if button.label == "Details" and "name:N003.1" in str(button.key)
    ).click().run()
    at.selectbox[0].set_value("Ravensgar").run()
    assert at.session_state[f"spelling|{run_id}|N003.1"] == "Ravensgar"

    next(
        button for button in at.button
        if button.label == "Details" and "name:N003.1" in str(button.key)
    ).click().run()
    at.checkbox[0].set_value(True).run()
    assert at.session_state[f"fix|{run_id}|F0001"] is True

    next(
        button for button in at.button
        if button.label == "Details" and "place:F0015" in str(button.key)
    ).click().run()
    at.radio(key=f"whand|{run_id}|place:F0015").set_value("Not a problem").run()
    assert at.session_state[f"hand|{run_id}|place:F0015"] == "Not a problem"
    assert not at.exception


def test_clear_all_and_a_new_check_clear_the_decisions():
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    at.radio(key="input_mode").set_value("Sample show")
    at.radio(key="model_choice").set_value("No AI")
    at.button(key="run_button").click().run()
    old_run = at.session_state["canon_run"]["run_id"]

    _click(at, "Fix these too").run()
    assert any(str(key).startswith(f"fix|{old_run}|") for key in at.session_state)

    _click(at, "Clear all").run()
    assert not [key for key in at.session_state if str(key).startswith(f"fix|{old_run}|")]

    _click(at, "Fix these too").run()
    at.button(key="run_button").click().run()
    new_run = at.session_state["canon_run"]["run_id"]
    assert new_run != old_run
    assert not [key for key in at.session_state if str(key).startswith(f"fix|{old_run}|")]
    assert not [key for key in at.session_state if str(key).startswith(f"fix|{new_run}|")]


def test_zip_fixed_puts_every_episode_under_fixed_scripts(tmp_path):
    run = tmp_path / "run"
    (run / "patched").mkdir(parents=True)
    (run / "patched" / "episode-1.txt").write_text("fixed", encoding="utf-8")
    (run / "patch.diff").write_text("a diff", encoding="utf-8")
    changes = [
        {"episode": 1, "time": "00:10.00", "where": "E01.B001", "was": "Ravenscourt", "now": "Ravenscar", "issue": "Ravenscourt → Ravenscar"}
    ]

    with zipfile.ZipFile(io.BytesIO(canon_app.zip_fixed(run, changes, []))) as archive:
        assert sorted(archive.namelist()) == [
            "CHANGES.csv",
            "changes.diff",
            "fix_by_hand.txt",
            "fixed_scripts/episode-1.txt",
        ]
        changes_text = archive.read("CHANGES.csv").decode("utf-8")
    assert "Ravenscourt" in changes_text and "Ravenscar" in changes_text
