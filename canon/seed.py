"""Seeded benchmark: copy the pack, inject known mistakes, and record the truth.

R1: every injected item is written to truth.json marked `seeded`, and the benchmark scores
against that file. Code never writes a human label. The injections copy the shapes the
detectors claim to catch: a name under two spellings, one character under two labels, a
time-of-day or building change on a continuous join, and a misspelled spoken word.
"""

import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .joins import build_joins, heading_type, time_class
from .mentions import TITLES, build_index, clean_surface, raw_key
from .parse import format_timestamp, parse_show, read_input, sec
from .speakers import core_words, speaker_stats

NAME_RULES = ["vowel_swap", "double_consonant", "undouble", "suffix_swap", "drop_letter", "transpose"]
NAME_SUFFIXES = ["bury", "borough", "ton", "ford", "wick", "ley", "ham", "worth", "field", "wood", "bridge"]
VOWELS = "aeiou"
TITLE_PREFIXES = {"LADY", "LORD", "THE", "SIR", "DUKE", "DUCHESS", "MISS", "MRS", "MR", "CAPTAIN", "FATHER"}
MALE_WORDS = {"LORD", "SIR", "DUKE", "MR", "CAPTAIN", "FATHER", "KING", "PRINCE", "BROTHER", "SON",
              "PRIEST", "BISHOP", "MAN"}
BUILDING_PAIRS = {
    "cathedral": "chapel", "chapel": "cathedral",
    "manor": "castle", "castle": "manor",
    "cottage": "cabin", "cabin": "cottage",
}
WORD = re.compile(r"[A-Za-z]{6,}")
SPEAKER_TAG = re.compile(r"^\s*\*\*(.+?):\*\*\s*$")
HEADING = re.compile(r"^\s*SCENE\s+HEADING:", re.IGNORECASE)
EPISODE_NUMBER = re.compile(r"EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)", re.IGNORECASE)


class SeedError(RuntimeError):
    pass


@dataclass
class SeedEdit:
    file: str
    line: int
    col: int
    length: int
    new: str


def _case_like(original: str, word: str) -> str:
    if original.isupper():
        return word.upper()
    if original[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def _scene_at(episode, when: float) -> int:
    for scene in episode.scenes:
        if scene.start - 0.01 <= when < scene.end:
            return scene.num
    before = [scene for scene in episode.scenes if scene.start <= when + 0.01]
    if before:
        return before[-1].num
    return episode.scenes[0].num if episode.scenes else 1


def _name_variant(word: str, rule: str, rng: random.Random, taken: set[str]) -> str | None:
    low = word.lower()
    if len(low) < 3:
        return None
    candidates: list[str] = []
    if rule == "vowel_swap":
        spots = [i for i in range(1, len(low) - 1) if low[i] in VOWELS]
        rng.shuffle(spots)
        for i in spots:
            for vowel in VOWELS:
                if vowel != low[i]:
                    candidates.append(low[:i] + vowel + low[i + 1:])
    elif rule == "double_consonant":
        spots = [i for i in range(1, len(low) - 1) if low[i].isalpha() and low[i] not in VOWELS]
        rng.shuffle(spots)
        for i in spots:
            candidates.append(low[:i] + low[i] + low[i:])
    elif rule == "undouble":
        spots = [i for i in range(1, len(low) - 1) if low[i] == low[i + 1]]
        rng.shuffle(spots)
        for i in spots:
            candidates.append(low[:i] + low[i + 1:])
    elif rule == "suffix_swap":
        for suffix in NAME_SUFFIXES:
            if low.endswith(suffix) and len(low) > len(suffix) + 2:
                stem = low[:-len(suffix)]
                for other in NAME_SUFFIXES:
                    if other != suffix:
                        candidates.append(stem + other)
    elif rule == "drop_letter":
        spots = list(range(1, len(low) - 1))
        rng.shuffle(spots)
        for i in spots:
            candidates.append(low[:i] + low[i + 1:])
    elif rule == "transpose":
        spots = list(range(1, len(low) - 2))
        rng.shuffle(spots)
        for i in spots:
            if low[i] != low[i + 1]:
                candidates.append(low[:i] + low[i + 1] + low[i] + low[i + 2:])
    rng.shuffle(candidates)
    for candidate in candidates:
        if len(candidate) < 3 or not candidate.isalpha() or candidate == low:
            continue
        if candidate in taken:
            continue
        return candidate
    return None


class Seeder:
    def __init__(self, input_source: str, seed_num: int, final_run: str = "out/final"):
        self.input_source = input_source
        self.seed_num = seed_num
        self.rng = random.Random(seed_num)
        self.files = read_input(input_source)
        self.show = parse_show(input_source)
        self.index = build_index(self.show)
        self.lines = {name: text.split("\n") for name, text in self.files}
        self.used: dict[str, set[int]] = defaultdict(set)
        self.edits: list[SeedEdit] = []
        self.truth: list[dict] = []
        self.beat_text = {b.id: b.text for e in self.show.episodes for b in e.beats}
        self.ep_file = self._episode_files()
        self.accepted = self._accepted_keys(final_run)

    def _episode_files(self) -> dict[int, str]:
        out = {}
        for position, (name, text) in enumerate(self.files, 1):
            match = EPISODE_NUMBER.search(text)
            out[int(match.group(1)) if match else position] = name
        return out

    def _accepted_keys(self, final_run: str) -> set[str]:
        path = Path(final_run) / "groups.json"
        if not path.exists():
            return set()
        data = json.loads(path.read_text(encoding="utf-8"))
        keys = set()
        for group in data.get("names", []):
            if group.get("same_entity"):
                keys.update(group.get("keys", []))
        return keys

    # ---- locating text ----

    def _beat_line(self, filename: str, beat_id: str) -> int:
        text = self.beat_text[beat_id]
        if not text.strip():
            raise SeedError(f"beat has no text: {beat_id}")
        lines = self.lines[filename]
        fallback = None
        for i, line in enumerate(lines):
            if line.strip() != text:
                continue
            if i not in self.used[filename]:
                self.used[filename].add(i)
                return i
            if fallback is None:
                fallback = i
        if fallback is not None:
            return fallback
        raise SeedError(f"beat line not found: {beat_id} {text[:50]!r}")

    def _word_col(self, line: str, word: str, beat_text: str, offset_in_text: int) -> int:
        base_offset = line.find(beat_text)
        approx = (base_offset if base_offset >= 0 else 0) + offset_in_text
        if line[approx:approx + len(word)] == word:
            return approx
        found = line.find(word)
        if found < 0:
            raise SeedError(f"word not found on line: {word!r}")
        return found

    def _label_line(self, filename: str, beat_line: int, label: str) -> int:
        lines = self.lines[filename]
        for i in range(beat_line - 1, max(-1, beat_line - 5), -1):
            match = SPEAKER_TAG.match(lines[i])
            if match and match.group(1).split(" (")[0].strip() == label:
                return i
        raise SeedError(f"label line not found for {label!r} above line {beat_line}")

    def _heading_line(self, filename: str, heading: str) -> int:
        for i, line in enumerate(self.lines[filename]):
            if HEADING.match(line) and heading in line:
                return i
        raise SeedError(f"heading line not found: {heading!r}")

    # ---- injections ----

    def names(self) -> None:
        spoken: dict[str, list] = defaultdict(list)
        for mention in self.index.mentions:
            if mention.kind == "spoken":
                spoken[mention.key].append(mention)
        episodes = {key: {m.ep for m in items} for key, items in spoken.items()}
        eligible = [
            key for key in spoken
            if len(spoken[key]) >= 6 and len(episodes[key]) >= 3 and key not in self.accepted
        ]
        eligible.sort(key=lambda k: (-len(spoken[k]), k))
        if len(eligible) < 10:
            raise SeedError(f"only {len(eligible)} names have 6+ spoken mentions in 3+ episodes")
        taken = set(self.index.names) | set(self.index.display)
        pool = list(eligible)
        self.rng.shuffle(pool)
        made = 0
        for key in pool:
            if made >= 10:
                break
            items = sorted(spoken[key], key=lambda m: (m.ep, m.start, m.beat_id))
            count = 2 if self.rng.random() < 0.5 else 3
            targets = items[-count:]
            display = clean_surface(targets[-1].surface)
            rules = list(NAME_RULES)
            self.rng.shuffle(rules)
            new = rule = None
            for candidate_rule in rules:
                new = _name_variant(display, candidate_rule, self.rng, taken)
                if new:
                    rule = candidate_rule
                    break
            if not new:
                continue
            made += 1
            taken.add(raw_key(new))
            occurrences = []
            for mention in targets:
                filename = self.ep_file[mention.ep]
                line_no = self._beat_line(filename, mention.beat_id)
                line = self.lines[filename][line_no]
                surface = mention.surface
                base = clean_surface(surface)
                suffix = surface[len(base):]
                col = self._word_col(line, base, self.beat_text[mention.beat_id], mention.char_start)
                replacement = _case_like(base, new)
                self.edits.append(SeedEdit(filename, line_no, col, len(base), replacement))
                occurrences.append(
                    {
                        "ep": mention.ep,
                        "scene": mention.scene,
                        "start": format_timestamp(mention.start),
                        "where": mention.beat_id,
                        "old": surface,
                        "new": replacement + suffix,
                    }
                )
            self.truth.append(
                {
                    "id": f"S{len(self.truth) + 1:03d}",
                    "detector": "names",
                    "kind": "name_variant",
                    "rule": rule,
                    "old": display,
                    "new": _case_like(display, new),
                    "episodes": sorted({m.ep for m in targets}),
                    "occurrences": occurrences,
                }
            )
        if made < 10:
            raise SeedError(f"only {made} names could be given a variant")

    def labels(self) -> None:
        stats = speaker_stats(self.show, self.index)
        eligible = [
            label for label, stat in stats.items()
            if stat.lines >= 20 and (core_words(label) or len(label.split()) > 1)
        ]
        eligible.sort(key=lambda label: (-stats[label].lines, label))
        if len(eligible) < 4:
            raise SeedError(f"only {len(eligible)} characters have 20+ lines")
        for label in self.rng.sample(eligible, 4):
            episodes = sorted(stats[label].episodes)
            if len(episodes) < 2:
                continue
            run_length = min(self.rng.choice([2, 3, 4]), len(episodes) - 1)
            last = episodes[-1]
            run = list(range(last - run_length + 1, last + 1))
            new_label = self._label_variant(label, stats)
            if not new_label:
                continue
            occurrences = []
            for episode_num in run:
                filename = self.ep_file[episode_num]
                lines = self.lines[filename]
                episode = next(e for e in self.show.episodes if e.num == episode_num)
                for i, line in enumerate(lines):
                    match = SPEAKER_TAG.match(line)
                    if not match or match.group(1).split(" (")[0].strip() != label:
                        continue
                    start = line.find("**") + 2
                    if line[start:start + len(label)] != label:
                        raise SeedError(f"label not at expected place: {line!r}")
                    start_line = lines[i + 1] if i + 1 < len(lines) else ""
                    time_match = re.search(r"Start:\s*\(?([0-9:.]+)\)?", start_line)
                    when = sec(time_match.group(1)) if time_match else 0.0
                    scene = _scene_at(episode, when)
                    self.edits.append(SeedEdit(filename, i, start, len(label), new_label))
                    occurrences.append(
                        {
                            "ep": episode_num,
                            "scene": scene,
                            "start": format_timestamp(when),
                            "where": f"E{episode_num:02d}.S{scene:02d}.label",
                            "old": label,
                            "new": new_label,
                        }
                    )
            self.truth.append(
                {
                    "id": f"S{len(self.truth) + 1:03d}",
                    "detector": "speakers",
                    "kind": "label_split",
                    "old": label,
                    "new": new_label,
                    "episodes": run,
                    "occurrences": occurrences,
                }
            )

    def _label_variant(self, label: str, stats: dict) -> str | None:
        taken = set(stats)
        words = label.split()
        options = []
        if words[0] in TITLE_PREFIXES and len(words) > 1:
            options.append(" ".join(words[1:]))
        elif len(words) > 1:
            title = "LORD " if any(w in MALE_WORDS for w in words) else "LADY "
            options.append(title + label)
        for i in range(1, len(label) - 2):
            if label[i].isalpha() and label[i + 1].isalpha() and label[i] != label[i + 1]:
                options.append(label[:i] + label[i + 1] + label[i] + label[i + 2:])
                break
        self.rng.shuffle(options)
        for option in options:
            if option and option != label and option not in taken:
                return option
        return None

    def _joins(self):
        joins = build_joins(self.show)
        effective: dict[tuple[int, int], str | None] = {}
        building: dict[tuple[int, int], str | None] = {}
        for join in joins:
            key = (join.next.ep, join.next.num)
            own = time_class(join.next.time)
            inherited = effective.get((join.prev.ep, join.prev.num)) if join.continuous else None
            effective[key] = own or inherited
            after = heading_type(join.next)
            inherited_type = building.get((join.prev.ep, join.prev.num)) if join.continuous else None
            building[key] = after or inherited_type
        return joins, effective, building

    def times(self) -> None:
        joins, effective, _ = self._joins()
        eligible = []
        for join in joins:
            if not join.continuous or join.next.time not in ("DAY", "NIGHT"):
                continue
            before = effective.get((join.prev.ep, join.prev.num))
            after = time_class(join.next.time)
            if before in ("day", "night") and after == before:
                eligible.append(join)
        eligible.sort(key=lambda j: (j.next.ep, j.next.num))
        if len(eligible) < 4:
            raise SeedError(f"only {len(eligible)} continuous joins agree on the time of day")
        for join in self.rng.sample(eligible, 4):
            filename = self.ep_file[join.next.ep]
            line_no = self._heading_line(filename, join.next.heading)
            line = self.lines[filename][line_no]
            old = join.next.time
            new = "NIGHT" if old == "DAY" else "DAY"
            col = line.rfind(old)
            if col < 0:
                raise SeedError(f"time token not found: {old!r} in {line!r}")
            self.edits.append(SeedEdit(filename, line_no, col, len(old), new))
            self.truth.append(
                {
                    "id": f"S{len(self.truth) + 1:03d}",
                    "detector": "joins",
                    "kind": "time_flip",
                    "old": old,
                    "new": new,
                    "episodes": [join.next.ep],
                    "occurrences": [
                        {
                            "ep": join.next.ep,
                            "scene": join.next.num,
                            "start": format_timestamp(join.next.start),
                            "where": f"E{join.prev.ep:02d} S{join.prev.num:02d} -> E{join.next.ep:02d} S{join.next.num:02d}",
                            "old": old,
                            "new": new,
                        }
                    ],
                }
            )

    def buildings(self) -> None:
        joins, _, building = self._joins()
        eligible = []
        for join in joins:
            if not join.continuous:
                continue
            before = building.get((join.prev.ep, join.prev.num))
            after = heading_type(join.next)
            if before and after and before == after and before in BUILDING_PAIRS:
                eligible.append(join)
        eligible.sort(key=lambda j: (j.next.ep, j.next.num))
        if len(eligible) < 3:
            raise SeedError(f"only {len(eligible)} continuous joins share a building type")
        for join in self.rng.sample(eligible, 3):
            filename = self.ep_file[join.next.ep]
            line_no = self._heading_line(filename, join.next.heading)
            line = self.lines[filename][line_no]
            old = heading_type(join.next)
            new = BUILDING_PAIRS[old]
            match = re.search(rf"\b{old}\b", line, re.IGNORECASE)
            if not match:
                raise SeedError(f"building word not found: {old!r} in {line!r}")
            self.edits.append(SeedEdit(filename, line_no, match.start(), match.end() - match.start(), new.upper()))
            self.truth.append(
                {
                    "id": f"S{len(self.truth) + 1:03d}",
                    "detector": "joins",
                    "kind": "building_swap",
                    "old": old,
                    "new": new,
                    "episodes": [join.next.ep],
                    "occurrences": [
                        {
                            "ep": join.next.ep,
                            "scene": join.next.num,
                            "start": format_timestamp(join.next.start),
                            "where": f"E{join.prev.ep:02d} S{join.prev.num:02d} -> E{join.next.ep:02d} S{join.next.num:02d}",
                            "old": old,
                            "new": new,
                        }
                    ],
                }
            )

    def typos(self) -> None:
        from spellchecker import SpellChecker

        spell = SpellChecker()
        candidates = []
        for episode in self.show.episodes:
            for beat in episode.beats:
                if beat.kind != "line" or beat.speaker is None:
                    continue
                for match in WORD.finditer(beat.text):
                    word = match.group(0)
                    low = word.lower()
                    if low in self.index.names or low in TITLES:
                        continue
                    if low in spell.known([low]):
                        candidates.append((beat.ep, beat.id, match.start(), word))
        candidates.sort()
        if len(candidates) < 10:
            raise SeedError(f"only {len(candidates)} common spoken words to misspell")
        by_beat = defaultdict(list)
        for candidate in candidates:
            by_beat[candidate[1]].append(candidate)
        beats = sorted(by_beat)
        self.rng.shuffle(beats)
        chosen = [self.rng.choice(by_beat[beat_id]) for beat_id in beats[:10]]
        if len(chosen) < 10:
            raise SeedError(f"only {len(chosen)} spoken lines to misspell")
        for ep, beat_id, offset, word in chosen:
            low = word.lower()
            mutation = self._typo_variant(low, self.rng.choice(["drop", "double", "swap"]), spell)
            if not mutation:
                continue
            filename = self.ep_file[ep]
            line_no = self._beat_line(filename, beat_id)
            line = self.lines[filename][line_no]
            col = self._word_col(line, word, self.beat_text[beat_id], offset)
            self.edits.append(SeedEdit(filename, line_no, col, len(word), mutation))
            self.truth.append(
                {
                    "id": f"S{len(self.truth) + 1:03d}",
                    "detector": "spelling",
                    "kind": "typo",
                    "old": word,
                    "new": mutation,
                    "episodes": [ep],
                    "occurrences": [
                        {
                            "ep": ep,
                            "scene": next(b.scene for e in self.show.episodes if e.num == ep for b in e.beats if b.id == beat_id),
                            "start": format_timestamp(next(b.start for e in self.show.episodes if e.num == ep for b in e.beats if b.id == beat_id)),
                            "where": beat_id,
                            "old": word,
                            "new": mutation,
                        }
                    ],
                }
            )

    def _typo_variant(self, word: str, rule: str, spell) -> str | None:
        options = []
        if rule == "drop":
            for i in range(1, len(word) - 1):
                options.append(word[:i] + word[i + 1:])
        elif rule == "double":
            for i in range(1, len(word) - 1):
                options.append(word[:i] + word[i] + word[i:])
        else:
            for i in range(1, len(word) - 2):
                if word[i] != word[i + 1]:
                    options.append(word[:i] + word[i + 1] + word[i] + word[i + 2:])
        self.rng.shuffle(options)
        for option in options:
            if option and option not in spell.known([option]):
                return option
        return None

    # ---- writing ----

    def apply(self) -> None:
        by_file: dict[str, list[SeedEdit]] = defaultdict(list)
        for edit in self.edits:
            by_file[edit.file].append(edit)
        for filename, items in by_file.items():
            lines = self.lines[filename]
            items.sort(key=lambda e: (e.line, e.col), reverse=True)
            last_line, last_col = None, None
            for edit in items:
                if edit.line == last_line and last_col is not None and edit.col + edit.length > last_col:
                    raise SeedError(f"overlapping edits on {filename} line {edit.line}")
                line = lines[edit.line]
                lines[edit.line] = line[:edit.col] + edit.new + line[edit.col + edit.length:]
                last_line, last_col = edit.line, edit.col

    def write(self, out_dir: str) -> dict:
        out = Path(out_dir)
        input_dir = out / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        for name, _ in self.files:
            (input_dir / name).write_text("\n".join(self.lines[name]), encoding="utf-8")
        payload = {
            "seeded": True,
            "seed": self.seed_num,
            "input": self.input_source,
            "input_sha256": hashlib.sha256("".join(text for _, text in self.files).encode()).hexdigest(),
            "items": self.truth,
            "counts": {
                "names": sum(1 for item in self.truth if item["detector"] == "names"),
                "speakers": sum(1 for item in self.truth if item["detector"] == "speakers"),
                "joins": sum(1 for item in self.truth if item["detector"] == "joins"),
                "spelling": sum(1 for item in self.truth if item["detector"] == "spelling"),
            },
        }
        (out / "truth.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        return payload


def make_seed(input_source: str, out_dir: str, seed_num: int, final_run: str = "out/final") -> dict:
    seeder = Seeder(input_source, seed_num, final_run)
    seeder.names()
    seeder.labels()
    seeder.times()
    seeder.buildings()
    seeder.typos()
    seeder.apply()
    return seeder.write(out_dir)
