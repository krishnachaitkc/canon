"""Fail if a repo file leaks the private pack.

Two rules:
  A. A name from the private list (one name per line, # comments) must not appear in any
     tracked, staged or untracked-not-ignored file. Whole word, case-insensitive.
  B. An 8-word sequence taken from data/episodes must not appear in one of those files.
     Shingles stay inside one line, so the pack's short format labels never combine into
     fake sequences. Sequences found in at least half of the episodes are format
     boilerplate, not story, and are ignored.

The matching sentence is never printed: a hit says the name that matched, or the
episode a sequence came from.

Run from the repo root:  .venv\\Scripts\\python scripts\\leak_check.py
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

WORDS = re.compile(r"[a-z']+")
NAME_BOUNDARY = r"(?<![A-Za-z0-9_]){}(?![A-Za-z0-9_])"
SHINGLE_WORDS = 8
EPISODE_IN_NAME = re.compile(r"(?:episode|ep)[-_ ]?0*(\d+)", re.IGNORECASE)


class LeakCheckError(RuntimeError):
    """The check itself could not run (missing list, missing episodes, no git)."""


@dataclass(frozen=True)
class Hit:
    path: str
    line: int
    rule: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}"


def git_files(root: Path) -> list[str]:
    """Tracked + staged + untracked-not-ignored files, relative to root."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise LeakCheckError(f"cannot run git: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise LeakCheckError(f"git ls-files failed in {root}: {detail}")
    return [line for line in result.stdout.splitlines() if line.strip()]


def read_text(path: Path) -> str | None:
    """File text, or None when the file is binary."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise LeakCheckError(f"cannot read {path}: {exc}") from exc
    if b"\x00" in data[:8192]:
        return None
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


def load_names(path: Path) -> list[str]:
    if not path.is_file():
        raise LeakCheckError(f"name list not found: {path}")
    names = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            names.append(value)
    if not names:
        raise LeakCheckError(f"name list is empty: {path}")
    return names


def name_patterns(names: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    return [
        (name, re.compile(NAME_BOUNDARY.format(re.escape(name)), re.IGNORECASE))
        for name in names
    ]


def episode_label(path: Path, fallback: int) -> str:
    match = EPISODE_IN_NAME.search(path.stem)
    return f"E{int(match.group(1)):02d}" if match else f"episode {fallback}"


def episode_texts(episodes_dir: Path) -> list[tuple[Path, str]]:
    if not episodes_dir.is_dir():
        raise LeakCheckError(f"episodes folder not found: {episodes_dir}")
    files = sorted(episodes_dir.glob("*.txt"))
    if not files:
        raise LeakCheckError(f"no .txt episodes in {episodes_dir}")
    out = []
    for path in files:
        text = read_text(path)
        if text is None:
            raise LeakCheckError(f"episode is not readable text: {path}")
        out.append((path, text))
    return out


def line_words(text: str) -> list[tuple[int, list[str]]]:
    """(line number, lowercase words) for each line; shingles never cross a line break."""
    return [
        (number, [match.group(0) for match in WORDS.finditer(line.lower())])
        for number, line in enumerate(text.splitlines(), 1)
    ]


def story_shingles(episodes_dir: Path) -> dict[str, str]:
    """8-word sequence -> the episode label it came from, excluding format boilerplate."""
    episodes = episode_texts(episodes_dir)
    seen_in: dict[str, set[int]] = {}
    labels: dict[str, str] = {}
    for index, (path, text) in enumerate(episodes):
        label = episode_label(path, index + 1)
        present = set()
        for _, tokens in line_words(text):
            for start in range(len(tokens) - SHINGLE_WORDS + 1):
                key = " ".join(tokens[start:start + SHINGLE_WORDS])
                present.add(key)
                labels.setdefault(key, label)
        for key in present:
            seen_in.setdefault(key, set()).add(index)
    count = len(episodes)
    return {
        key: labels[key]
        for key, where in seen_in.items()
        if len(where) * 2 < count  # in fewer than half the episodes
    }


def check_file(
    path: str,
    text: str,
    patterns: list[tuple[str, re.Pattern[str]]],
    story: dict[str, str],
) -> list[Hit]:
    hits: list[Hit] = []
    seen: set[tuple[int, str]] = set()
    for number, line in enumerate(text.splitlines(), 1):
        for name, pattern in patterns:
            if pattern.search(line) and (number, f"name: {name}") not in seen:
                seen.add((number, f"name: {name}"))
                hits.append(Hit(path, number, f"name: {name}"))
        tokens = [match.group(0) for match in WORDS.finditer(line.lower())]
        for start in range(len(tokens) - SHINGLE_WORDS + 1):
            key = " ".join(tokens[start:start + SHINGLE_WORDS])
            source = story.get(key)
            if source is None:
                continue
            rule = f"8-word sequence from {source}"
            if (number, rule) in seen:
                continue
            seen.add((number, rule))
            hits.append(Hit(path, number, rule))
    return hits


def check(root: Path, files: list[str], episodes_dir: Path, names_file: Path) -> list[Hit]:
    names = load_names(names_file)
    patterns = name_patterns(names)
    story = story_shingles(episodes_dir)
    hits: list[Hit] = []
    for relative in files:
        path = root / relative
        if not path.is_file():
            continue
        text = read_text(path)
        if text is None:
            continue
        hits.extend(check_file(relative, text, patterns, story))
    return sorted(hits, key=lambda hit: (hit.path, hit.line, hit.rule))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail if a repo file leaks the private script pack.")
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--episodes", default="data/episodes", help="private episodes folder")
    parser.add_argument("--names", default="plan/leak_names.txt", help="private name list")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    try:
        files = git_files(root)
        hits = check(root, files, root / args.episodes, root / args.names)
    except LeakCheckError as exc:
        print(f"leak check failed: {exc}", file=sys.stderr)
        return 1
    for hit in hits:
        print(hit.render())
    if hits:
        print(f"leak check: {len(hits)} hit(s) in {len(files)} files", file=sys.stderr)
        return 1
    print(f"leak check: clean ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
