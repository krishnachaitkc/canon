import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "leak_check.py"
SPEC = importlib.util.spec_from_file_location("leak_check", SCRIPT)
leak_check = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = leak_check
SPEC.loader.exec_module(leak_check)

EPISODES = {
    "episode-001.txt": "the old miller walked past the silent river at dawn",
    "episode-002.txt": "actions and dialogue follow every scene heading here",
    "episode-003.txt": "actions and dialogue follow every scene heading here",
    "episode-004.txt": "nobody spoke while the horses waited by the broken gate",
}
NAMES = "Vellmoor\n# a comment, not a name\nQuillby Manor\n"


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "repo"
    episodes = root / "data" / "episodes"
    episodes.mkdir(parents=True)
    for name, text in EPISODES.items():
        (episodes / name).write_text(text + "\n", encoding="utf-8")
    names = root / "plan" / "leak_names.txt"
    names.parent.mkdir()
    names.write_text(NAMES, encoding="utf-8")
    return root, episodes, names


def write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_a_name_is_matched_whole_word_and_case_insensitively(repo):
    root, episodes, names = repo
    write(root, "notes.txt", "The VELLMOOR ledger is missing.\nVellmoorish is not the name.\n")

    hits = leak_check.check(root, ["notes.txt"], episodes, names)

    assert [(h.path, h.line, h.rule) for h in hits] == [("notes.txt", 1, "name: Vellmoor")]


def test_a_multiword_name_is_one_rule(repo):
    root, episodes, names = repo
    write(root, "deed.txt", "the quillby manor deed is sealed\n")

    hits = leak_check.check(root, ["deed.txt"], episodes, names)

    assert [h.rule for h in hits] == ["name: Quillby Manor"]


def test_an_eight_word_sequence_is_reported_with_its_episode(repo):
    root, episodes, names = repo
    write(root, "story.txt", "Yesterday the old miller walked past the silent river at dawn.\n")

    hits = leak_check.check(root, ["story.txt"], episodes, names)

    assert [(h.line, h.rule) for h in hits] == [(1, "8-word sequence from E01")]


def test_a_sequence_in_half_the_episodes_is_format_boilerplate(repo):
    root, episodes, names = repo
    write(root, "format.txt", "Actions and dialogue follow every scene heading here.\n")

    assert leak_check.check(root, ["format.txt"], episodes, names) == []


def test_binary_files_are_skipped(repo):
    root, episodes, names = repo
    (root / "blob.bin").write_bytes(b"\x00\x01Vellmoor the old miller walked past the silent river")

    assert leak_check.check(root, ["blob.bin"], episodes, names) == []


def test_a_clean_file_has_no_hits(repo):
    root, episodes, names = repo
    write(root, "clean.txt", "the old miller walked slowly home\n")

    assert leak_check.check(root, ["clean.txt"], episodes, names) == []


def test_a_missing_name_list_cannot_be_checked(repo):
    root, episodes, _ = repo

    with pytest.raises(leak_check.LeakCheckError, match="name list not found"):
        leak_check.check(root, ["clean.txt"], episodes, root / "plan" / "absent.txt")


def test_missing_episodes_cannot_be_checked(repo):
    root, _, names = repo

    with pytest.raises(leak_check.LeakCheckError, match="episodes folder not found"):
        leak_check.check(root, [], root / "data" / "absent", names)


def test_main_exits_one_on_a_hit_and_never_prints_the_sentence(repo, monkeypatch, capsys):
    root, _, _ = repo
    write(root, "story.txt", "Yesterday the old miller walked past the silent river at dawn.\n")
    monkeypatch.setattr(leak_check, "git_files", lambda _root: ["story.txt"])

    code = leak_check.main(["--root", str(root)])

    captured = capsys.readouterr()
    assert code == 1
    assert "8-word sequence from E01" in captured.out
    assert "old miller" not in captured.out
    assert "silent river" not in captured.out


def test_main_exits_zero_when_clean(repo, monkeypatch, capsys):
    root, _, _ = repo
    write(root, "clean.txt", "nothing private lives here\n")
    monkeypatch.setattr(leak_check, "git_files", lambda _root: ["clean.txt"])

    code = leak_check.main(["--root", str(root)])

    assert code == 0
    assert "clean" in capsys.readouterr().out
