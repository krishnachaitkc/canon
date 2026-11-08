import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import app as canon_app

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"


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
    at.radio(key="input_mode").set_value("Try the sample show")
    at.radio(key="model_choice").set_value("DeepSeek")
    at.button(key="run_button").click().run()

    assert not at.exception
    assert any("rejected the API key" in error.value for error in at.error)
    assert any("What to do next" in markdown.value for markdown in at.markdown)
    assert any("Code only" in markdown.value for markdown in at.markdown)
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


def test_the_app_survives_a_build_without_progress_reporting(monkeypatch):
    """A deploy can pair this app.py with an older canon.llm. Progress is cosmetic."""
    from canon import llm

    monkeypatch.delattr(llm, "set_progress", raising=False)

    at = AppTest.from_file(str(APP), default_timeout=180).run()
    at.radio(key="input_mode").set_value("Try the sample show")
    at.radio(key="model_choice").set_value("Code only")
    at.button(key="run_button").click().run()

    assert not at.exception
    assert not at.error
    assert at.session_state["canon_run"]["manifest"]["complete"] is True


def test_the_app_runs_the_sample_offline_and_shows_tiles_and_tabs():
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    at.radio(key="input_mode").set_value("Try the sample show")
    at.radio(key="model_choice").set_value("Code only")
    at.button(key="run_button").click().run()

    assert not at.exception
    assert not at.error
    assert [(m.label, m.value) for m in at.metric] == [
        ("Audible", "11"),
        ("Voice", "3"),
        ("Visible", "3"),
        ("Notes", "1"),
        ("Episodes affected", "4"),
    ]
    assert [tab.label for tab in at.tabs] == [
        "Names",
        "Speakers",
        "Scene joins",
        "Spelling",
        "Format",
    ]
    state = at.session_state["canon_run"]
    assert state["manifest"]["complete"] is True
    assert state["result"]["findings"]
    assert len(at.checkbox) == len(state["result"]["findings"])
    assert {button.label for button in at.download_button} >= {
        "findings.csv",
        "cast.csv",
        "report.html",
        "approvals.json",
    }
