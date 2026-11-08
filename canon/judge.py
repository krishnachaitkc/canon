"""Judges. The model decides which candidates are one thing; code checks every answer and vetoes.

R3: the model only returns spellings, labels and words copied from its input, plus a verdict.
Every answer is checked; a bad answer is retried once with the problems listed, then the run
fails. R4: two labels that speak in one scene are never one character, whatever the model says.
"""

import hashlib
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from . import llm
from .config import WORKERS, parse_route
from .mentions import HEAD_NOUNS, MentionIndex, iter_texts
from .models import Candidate, Finding, NameGroup, Show, SpeakerGroup, SpeakerStat
from .parse import format_timestamp
from .speakers import core_pairs

LOG = logging.getLogger(__name__)
PROMPTS = Path(__file__).resolve().parent / "prompts"
CONFIDENCE = {"high", "medium", "low"}
RANK = {"low": 0, "medium": 1, "high": 2}
ENTITY_TYPES = {"person", "family", "place", "org", "other"}
SNIPPET_CHARS = 160
CO_SPEAKERS_SHOWN = 15


# ---- model answers (lenient strings; code normalizes and checks) ----

class _NameGroupOut(BaseModel):
    spellings: list[str]
    same_entity: bool = False
    entity_type: str = "other"
    canonical: str = ""
    confidence: str = "low"
    reason: str = ""


class _RefersOut(BaseModel):
    spelling: str
    refers_to: str = ""


class NamesOut(BaseModel):
    refers: list[_RefersOut] = []
    groups: list[_NameGroupOut]


class _CharacterOut(BaseModel):
    labels: list[str]
    canonical: str = ""
    intentional: bool = False
    confidence: str = "low"
    reason: str = ""


class SpeakersOut(BaseModel):
    characters: list[_CharacterOut] = []


class _WordOut(BaseModel):
    word: str
    verdict: str
    correction: str = ""
    reason: str = ""


class SpellingOut(BaseModel):
    words: list[_WordOut]


@dataclass
class ModelSpec:
    routes: list[str]
    base_url: str | None = None
    api_key_env: str | None = None
    record_dir: Path | None = None

    def workers(self) -> int:
        provider, _ = parse_route(self.routes[0])
        return WORKERS.get(provider, 1)


def prompt_text(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text(encoding="utf-8").strip()


def _record(spec: ModelSpec, prompt: str, result: BaseModel) -> None:
    if spec.record_dir is None:
        return
    spec.record_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256((prompt + llm.JSON_RULE).encode()).hexdigest()
    (spec.record_dir / f"{digest}.json").write_text(result.model_dump_json(indent=1), encoding="utf-8")


def ask(stage: str, prompt: str, schema, spec: ModelSpec, check) -> BaseModel:
    """Call the model, check the answer, retry once with the problems listed, else fail."""
    current = prompt
    for attempt in (1, 2):
        result = llm.generate(
            stage,
            current,
            schema,
            models=spec.routes,
            base_url=spec.base_url,
            api_key_env=spec.api_key_env,
        )
        _record(spec, current, result)
        problems = check(result)
        if not problems:
            return result
        LOG.warning("%s answer failed its check (attempt %d): %s", stage, attempt, problems)
        current = (
            prompt
            + "\n\nYour previous answer had these problems:\n- "
            + "\n- ".join(problems)
            + "\nReturn the corrected JSON."
        )
    raise llm.LLMFailed(f"{stage}: the answer failed its check twice: {problems}")


def _match(value: str, allowed: list[str]) -> str | None:
    """Exact match, or the one allowed value that differs only in case."""
    if value in allowed:
        return value
    folded = [a for a in allowed if a.casefold() == value.strip().casefold()]
    return folded[0] if len(folded) == 1 else None


def _norm(value: str, allowed: set[str], default: str) -> str:
    value = (value or "").strip().lower()
    return value if value in allowed else default


# ---- D1: names ----

def _window(text: str, start: int, length: int) -> str:
    pad = max(0, (SNIPPET_CHARS - length) // 2)
    left, right = max(0, start - pad), min(len(text), start + length + pad)
    return ("…" if left else "") + text[left:right] + ("…" if right < len(text) else "")


CONTEXT_LINE_CHARS = 120


def dialogue_neighbors(show: Show) -> dict[str, tuple]:
    """For every spoken line, the spoken lines just before and after it in the same scene."""
    out = {}
    for episode in show.episodes:
        lines = [b for b in episode.beats if b.kind == "line"]
        for i, beat in enumerate(lines):
            before = lines[i - 1] if i > 0 and lines[i - 1].scene == beat.scene else None
            after = lines[i + 1] if i + 1 < len(lines) and lines[i + 1].scene == beat.scene else None
            out[beat.id] = (before, after)
    return out


def _context_line(beat) -> str:
    text = beat.text.strip()
    if len(text) > CONTEXT_LINE_CHARS:
        text = text[: CONTEXT_LINE_CHARS - 1].rstrip() + "\u2026"
    return f"{beat.speaker}: {text}"


def name_prompt(
    candidate: Candidate,
    index: MentionIndex,
    texts: dict[str, str],
    neighbors: dict[str, tuple] | None = None,
) -> tuple[str, dict[str, str]]:
    """Build the prompt for one candidate. Returns (prompt, display spelling -> key).

    A spoken snippet also shows the line before and after it: a one-line mention such as
    "The transfer of Ravensgar." only makes sense with its exchange around it.
    """
    by_key = index.by_key()
    display_to_key = {index.display[key]: key for key in candidate.keys}
    lines = [prompt_text("names"), "", "Why code put these together:"]
    for a, b, rule in candidate.links:
        lines.append(f"  {index.display[a]} and {index.display[b]}: {rule}")
    lines += ["", "Spellings:"]
    number = 0
    for display in candidate.spellings:                # most mentioned first
        key = display_to_key[display]
        mentions = sorted(by_key[key], key=lambda m: (m.ep, m.scene, m.start, m.char_start))
        picks = [mentions[0]]
        spoken = [m for m in mentions if m.kind == "spoken"]
        for extra in (spoken[0] if spoken else None, mentions[-1]):
            if extra is not None and extra not in picks:
                picks.append(extra)
        before, after = Counter(), Counter()
        for m in mentions:
            text = texts.get(m.beat_id, "")
            prev = re.search(r"([A-Z][a-z]+)[ \t]+$", text[: m.char_start])
            if prev:
                before[prev.group(1)] += 1
            nxt = re.match(r"[ \t]+([A-Za-z]+)", text[m.char_start + len(m.surface):])
            if nxt and (nxt.group(1)[0].isupper() or nxt.group(1).lower() in HEAD_NOUNS):
                after[nxt.group(1)] += 1
        context = []
        if before:
            context.append("Before it: " + ", ".join(f"{w} x{n}" for w, n in before.most_common(3)) + ".")
        if after:
            context.append("After it: " + ", ".join(f"{w} x{n}" for w, n in after.most_common(3)) + ".")
        lines.append(
            f'Spelling "{display}": {candidate.spellings[display]} mentions, '
            f"{candidate.spoken.get(display, 0)} spoken. " + " ".join(context)
        )
        for m in picks:
            number += 1
            where = f"E{m.ep:02d} {m.kind}" + (f" {m.speaker}" if m.kind == "spoken" and m.speaker else "")
            snippet = _window(texts.get(m.beat_id, ""), m.char_start, len(m.surface))
            lines.append(f"  [{number}] {where}: {snippet}")
            if m.kind == "spoken" and neighbors and m.beat_id in neighbors:
                before, after = neighbors[m.beat_id]
                if before is not None:
                    lines.append(f"      line before: {_context_line(before)}")
                if after is not None:
                    lines.append(f"      line after: {_context_line(after)}")
    return "\n".join(lines), display_to_key


def _check_names(spellings: list[str]):
    def check(out: NamesOut) -> list[str]:
        problems, seen = [], Counter()
        for group in out.groups:
            fixed = []
            for value in group.spellings:
                match = _match(value, spellings)
                if match is None:
                    problems.append(f'"{value}" is not one of the spellings given')
                else:
                    fixed.append(match)
                    seen[match] += 1
            if group.same_entity and len(fixed) >= 2 and _match(group.canonical, fixed) is None:
                problems.append(f'canonical "{group.canonical}" is not in its group {fixed}')
        for value in spellings:
            if seen[value] == 0:
                problems.append(f'"{value}" is missing: every spelling must appear in exactly one group')
            elif seen[value] > 1:
                problems.append(f'"{value}" appears in {seen[value]} groups; it must appear in exactly one')
        return problems

    return check


def _normalize_names(out: NamesOut, spellings: list[str]) -> None:
    """After the check passed: copy each spelling exactly as given (only case can differ)."""
    for group in out.groups:
        group.spellings = [_match(value, spellings) for value in group.spellings]
        if group.same_entity and len(group.spellings) >= 2:
            group.canonical = _match(group.canonical, group.spellings)


def judge_names(
    candidates: list[Candidate], index: MentionIndex, show: Show, spec: ModelSpec
) -> tuple[list[NameGroup], list[dict], list[Finding]]:
    texts = {t.beat_id: t.text for t in iter_texts(show)}
    neighbors = dialogue_neighbors(show)
    prompts = {}
    for candidate in candidates:
        prompts[candidate.id] = name_prompt(candidate, index, texts, neighbors)

    def one(job: tuple[str, int]):
        cid, attempt = job
        prompt, _ = prompts[cid]
        if attempt > 1:
            prompt += f"\n\n(Independent check {attempt} of {NAME_TRIES}.)"
        spellings = list(by_id[cid].spellings)
        try:
            out = ask("names", prompt, NamesOut, spec, _check_names(spellings))
        except llm.LLMFailed as exc:
            LOG.warning("names %s try %d failed: %s", cid, attempt, exc)
            return None
        _normalize_names(out, spellings)
        return out

    by_id = {candidate.id: candidate for candidate in candidates}
    jobs = [(cid, attempt) for cid in sorted(by_id) for attempt in range(1, NAME_TRIES + 1)]
    answers = llm.run_many("names", jobs, one, spec.workers())
    groups, apart, checks = [], [], []
    for candidate in candidates:
        _, display_to_key = prompts[candidate.id]
        tries = [answers[(candidate.id, a)] for a in range(1, NAME_TRIES + 1)]
        tries = [out for out in tries if out is not None]
        if len(tries) < (NAME_TRIES // 2) + 1:
            raise llm.LLMFailed(
                f"names {candidate.id}: only {len(tries)} of {NAME_TRIES} tries answered; need a majority"
            )
        out = majority(tries, list(candidate.spellings))
        number = 0
        for g in out.groups:
            unsure = _norm(g.confidence, CONFIDENCE, "low") == "low"
            if g.same_entity and len(g.spellings) >= 2 and unsure:
                # The model is unsure: a person decides (R5).
                spoken = sum(candidate.spoken.get(s, 0) for s in g.spellings)
                checks.append(
                    Finding(
                        detector="names",
                        severity="audible" if spoken else "visible",
                        source="check",
                        confidence="low",
                        ep=0,
                        scene=0,
                        start="00:00.00",
                        where=candidate.id,
                        found=" / ".join(g.spellings),
                        suggestion="The model is unsure these are one name. Check the snippets in the report.",
                        context=g.reason.strip()[:70],
                        group=f"names:{candidate.id}",
                    )
                )
                continue
            if g.same_entity and len(g.spellings) >= 2:
                number += 1
                groups.append(
                    NameGroup(
                        id=f"{candidate.id}.{number}",
                        candidate=candidate.id,
                        spellings=g.spellings,
                        keys=[display_to_key[s] for s in g.spellings],
                        canonical=g.canonical,
                        same_entity=True,
                        entity_type=_norm(g.entity_type, ENTITY_TYPES, "other"),
                        confidence=_norm(g.confidence, CONFIDENCE, "low"),
                        reason=g.reason.strip(),
                        source="ai",
                    )
                )
            else:
                apart.append(
                    {"candidate": candidate.id, "spellings": g.spellings, "reason": g.reason.strip()}
                )
    return groups, apart, checks


NAME_TRIES = 3


def majority(tries: list[NamesOut], spellings: list[str]) -> NamesOut:
    """Combine independent answers: two spellings are one name only if most tries say so.

    A group every try agrees on keeps the tries' lowest confidence; a group that only a
    majority agrees on is at most medium. Reasons and canonicals come from the tries that
    made exactly that group, else from any try that put its spellings together.
    """
    need = len(tries) // 2 + 1
    together = Counter()
    for out in tries:
        for g in out.groups:
            if g.same_entity and len(g.spellings) >= 2:
                for i, a in enumerate(sorted(g.spellings)):
                    for b in sorted(g.spellings)[i + 1:]:
                        together[(a, b)] += 1
    parent = {s: s for s in spellings}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), votes in sorted(together.items()):
        if votes >= need:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    members: dict[str, list[str]] = defaultdict(list)
    for s in spellings:
        members[find(s)].append(s)

    combined = []
    for group in members.values():
        group_set = set(group)
        if len(group) == 1:
            combined.append(_NameGroupOut(spellings=group, same_entity=False, reason="kept apart by most tries"))
            continue
        exact = [g for out in tries for g in out.groups if set(g.spellings) == group_set and g.same_entity]
        overlapping = [
            g for out in tries for g in out.groups if g.same_entity and len(set(g.spellings) & group_set) >= 2
        ]
        source = exact or overlapping
        canonicals = Counter(g.canonical for g in source if g.canonical in group_set)
        canonical = sorted(canonicals.items(), key=lambda kv: (-kv[1], spellings.index(kv[0])))[0][0] if canonicals else group[0]
        lowest = min((_norm(g.confidence, CONFIDENCE, "low") for g in source), key=RANK.get, default="low")
        confidence = lowest if len(exact) == len(tries) else min(lowest, "medium", key=RANK.get)
        types = Counter(_norm(g.entity_type, ENTITY_TYPES, "other") for g in source)
        combined.append(
            _NameGroupOut(
                spellings=group,
                same_entity=True,
                entity_type=types.most_common(1)[0][0] if types else "other",
                canonical=canonical,
                confidence=confidence,
                reason=(source[0].reason.strip() if source else "")
                + f" ({len(exact)} of {len(tries)} tries made exactly this group)",
            )
        )
    return NamesOut(groups=combined)


def code_names(candidates: list[Candidate], index: MentionIndex) -> list[NameGroup]:
    """No model: each candidate is one group; canonical has the most mentions (then spoken, then earliest)."""
    by_key = index.by_key()
    groups = []
    for candidate in candidates:
        display_to_key = {index.display[key]: key for key in candidate.keys}

        def rank(display):
            mentions = by_key[display_to_key[display]]
            first = min((m.ep, m.start) for m in mentions)
            return (-candidate.spellings[display], -candidate.spoken.get(display, 0), first)

        spellings = sorted(candidate.spellings, key=rank)
        groups.append(
            NameGroup(
                id=f"{candidate.id}.1",
                candidate=candidate.id,
                spellings=spellings,
                keys=[display_to_key[s] for s in spellings],
                canonical=spellings[0],
                same_entity=True,
                confidence="low",
                reason="code only: similar spellings, not judged",
                source="code",
            )
        )
    return groups


# ---- D2: speakers ----

def _ranges(episodes: list[int]) -> str:
    parts, start, prev = [], None, None
    for ep in episodes:
        if start is None:
            start = prev = ep
        elif ep == prev + 1:
            prev = ep
        else:
            parts.append(f"E{start}" if start == prev else f"E{start}-{prev}")
            start = prev = ep
    if start is not None:
        parts.append(f"E{start}" if start == prev else f"E{start}-{prev}")
    return ", ".join(parts)


def speaker_prompt(stats: dict[str, SpeakerStat]) -> str:
    lines = [prompt_text("speakers"), "", "Speaker labels:"]
    for label in sorted(stats, key=lambda l: (-stats[l].lines, l)):
        s = stats[label]
        on_screen = ", ".join(f"{name} x{n}" for name, n in s.camera_names) or "no named person"
        others = sorted(s.co_speakers)
        shared = ", ".join(others[:CO_SPEAKERS_SHOWN]) + (
            f" and {len(others) - CO_SPEAKERS_SHOWN} more" if len(others) > CO_SPEAKERS_SHOWN else ""
        )
        lines.append(
            f'LABEL "{label}": {s.lines} lines in {_ranges(s.episodes)}. On screen while it speaks: {on_screen}.'
        )
        lines.append(f"  Shares scenes with: {shared or 'nobody'}.")
        lines.append("  Lines: " + " | ".join(s.samples))
        for sentence in s.named_in:
            lines.append(f"  A sentence mentioning it with a named person: {sentence}")
    return "\n".join(lines)


def _check_speakers(labels: list[str]):
    def check(out: SpeakersOut) -> list[str]:
        problems, seen = [], Counter()
        for character in out.characters:
            fixed = []
            for value in character.labels:
                match = _match(value, labels)
                if match is None:
                    problems.append(f'"{value}" is not one of the labels given')
                else:
                    fixed.append(match)
                    seen[match] += 1
            if len(fixed) < 2:
                problems.append(f"a character needs two or more labels: {fixed}")
        return problems

    return check


def _normalize_speakers(out: SpeakersOut, labels: list[str]) -> None:
    for character in out.characters:
        character.labels = [_match(value, labels) for value in character.labels]
        canonical = _match(character.canonical, character.labels)
        if canonical is None and character.canonical.strip():
            character.reason = (
                f"{character.reason.strip()} (model suggested the name "
                f"\"{character.canonical.strip()}\"; the label with most lines is kept)"
            ).strip()
        character.canonical = canonical or ""


def co_speaking_scenes(show: Show) -> dict[frozenset, str]:
    """For every pair of labels that speak in one scene, where they first do: 'E41 S01 00:34.80'."""
    first_line: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for episode in show.episodes:
        for beat in episode.beats:
            if beat.kind == "line" and beat.speaker:
                seen = first_line[(beat.ep, beat.scene)]
                seen.setdefault(beat.speaker, beat.start)
    out: dict[frozenset, str] = {}
    for (ep, scene) in sorted(first_line):
        labels = first_line[(ep, scene)]
        names = sorted(labels)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pair = frozenset((a, b))
                if pair not in out:
                    when = max(labels[a], labels[b])
                    out[pair] = f"E{ep:02d} S{scene:02d} {format_timestamp(when)}"
    return out


def apply_veto(
    labels: list[str], stats: dict[str, SpeakerStat], co: dict[frozenset, str]
) -> tuple[list[list[str]], list[tuple[str, str, str]]]:
    """Split a proposed character so no two co-speaking labels stay together."""
    order = sorted(labels, key=lambda l: (-stats[l].lines, l))
    components: list[list[str]] = []
    for label in order:
        for component in components:
            if not any(frozenset((label, other)) in co for other in component):
                component.append(label)
                break
        else:
            components.append([label])
    vetoed = [
        (a, b, co[frozenset((a, b))])
        for i, a in enumerate(order)
        for b in order[i + 1:]
        if frozenset((a, b)) in co
    ]
    return components, vetoed


def _speaker_groups(
    proposals: list[tuple[list[str], str, bool, str, str, str]],
    stats: dict[str, SpeakerStat],
    co: dict[frozenset, str],
) -> tuple[list[SpeakerGroup], list[Finding]]:
    """proposals: (labels, canonical, intentional, confidence, reason, source)."""
    groups, checks = [], []
    for labels, canonical, intentional, confidence, reason, source in proposals:
        if source == "ai" and confidence == "low":
            # The model is unsure: a person decides (R5).
            first = min(labels, key=lambda l: (stats[l].episodes[0], l))
            ep = stats[first].episodes[0]
            checks.append(
                Finding(
                    detector="speakers",
                    severity="voice",
                    source="check",
                    confidence="low",
                    ep=ep,
                    scene=0,
                    start="00:00.00",
                    where=" / ".join(sorted(labels)),
                    found=f"{' and '.join(sorted(labels))} may be one character",
                    suggestion="The model is unsure. If they are one person, give them one voice.",
                    context=reason[:70],
                    group="unsure:" + "+".join(sorted(labels)),
                )
            )
            continue
        components, vetoed = apply_veto(labels, stats, co)
        for component in components:
            if len(component) < 2:
                continue
            chosen = canonical if canonical in component else max(component, key=lambda l: (stats[l].lines, l))
            groups.append(
                SpeakerGroup(
                    id="",
                    labels=component,
                    canonical=chosen,
                    intentional=intentional,
                    confidence=confidence,
                    reason=reason,
                    source=source,
                    vetoed=[v for v in vetoed if v[0] in component or v[1] in component],
                )
            )
        if vetoed and confidence == "high" and source == "ai":
            for a, b, where in vetoed:
                suspect, other = sorted((a, b), key=lambda l: (stats[l].lines, l))
                parts = re.match(r"E(\d+) S(\d+) (\S+)", where)
                ep, scene = int(parts.group(1)), int(parts.group(2))
                checks.append(
                    Finding(
                        detector="speakers",
                        severity="voice",
                        source="check",
                        ep=ep,
                        scene=scene,
                        start=parts.group(3),
                        where=suspect,
                        found=f"{suspect} matched to {other}, but both speak in {where}",
                        suggestion=(
                            f"{suspect} may be two different people. If so, give the one in "
                            f"E{ep:02d} S{scene:02d} its own label."
                        ),
                        context=reason[:70],
                        group=f"veto:{suspect}",
                    )
                )
    groups.sort(key=lambda g: (-sum(stats[l].lines for l in g.labels), g.canonical))
    for number, group in enumerate(groups, 1):
        group.id = f"S{number:03d}"
    return groups, checks


def judge_speakers(
    stats: dict[str, SpeakerStat], show: Show, spec: ModelSpec
) -> tuple[list[SpeakerGroup], list[Finding]]:
    out = ask("speakers", speaker_prompt(stats), SpeakersOut, spec, _check_speakers(sorted(stats)))
    _normalize_speakers(out, sorted(stats))
    return _speaker_groups(merge_overlaps(out.characters), stats, co_speaking_scenes(show))


def merge_overlaps(characters: list[_CharacterOut]) -> list[tuple]:
    """Characters that share a label are one character. Merge them, keeping the least confident
    rating (so an unsure link still goes to a person), any intentional flag, and every reason."""
    parent = list(range(len(characters)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: dict[str, int] = {}
    for i, character in enumerate(characters):
        for label in character.labels:
            if label in owner:
                a, b = find(owner[label]), find(i)
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                owner[label] = i
    merged: dict[int, list[_CharacterOut]] = {}
    for i, character in enumerate(characters):
        merged.setdefault(find(i), []).append(character)
    proposals = []
    for members in merged.values():
        labels = sorted({label for c in members for label in c.labels})
        confidence = min((_norm(c.confidence, CONFIDENCE, "low") for c in members), key=RANK.get)
        canonical = next((c.canonical for c in members if c.canonical), "")
        reason = " / ".join(c.reason.strip() for c in members if c.reason.strip())
        if len(members) > 1:
            reason = f"{reason} (merged {len(members)} overlapping answers)"
        proposals.append(
            (labels, canonical, any(c.intentional for c in members), confidence, reason, "ai")
        )
    return proposals


def code_speakers(stats: dict[str, SpeakerStat], show: Show) -> tuple[list[SpeakerGroup], list[Finding]]:
    """No model: labels that share an identifying word, then the veto."""
    parent = {label: label for label in stats}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    reasons = defaultdict(list)
    for a, b, why in core_pairs(stats):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
        reasons[min(ra, rb)].append(why)
    members = defaultdict(list)
    for label in stats:
        members[find(label)].append(label)
    proposals = [
        (sorted(labels), "", False, "low", "code only: " + "; ".join(sorted(set(reasons[root]))), "code")
        for root, labels in sorted(members.items())
        if len(labels) >= 2
    ]
    return _speaker_groups(proposals, stats, co_speaking_scenes(show))


# ---- D4: spelling ----

def spelling_prompt(findings: list[Finding], show: Show) -> tuple[str, list[str]]:
    beats = {b.id: b for e in show.episodes for b in e.beats}
    loglines = {e.num: e.logline for e in show.episodes}
    order = {b.id: i for i, b in enumerate(b for e in show.episodes for b in e.beats)}
    flat = [b for e in show.episodes for b in e.beats]
    words: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        words[f.found].append(f)
    lines = [prompt_text("spelling"), "", "Words:"]
    for word in sorted(words):
        lines.append(f'Word "{word}"')
        for f in words[word][:2]:
            beat = beats[f.where]
            previous = next(
                (flat[i] for i in range(order[beat.id] - 1, -1, -1) if flat[i].kind == "line"),
                None,
            )
            lines.append(f"  Line ({beat.speaker}, E{beat.ep:02d}): {beat.text}")
            if previous is not None and previous.ep == beat.ep:
                lines.append(f"  Line before ({previous.speaker}): {previous.text}")
            summary = loglines.get(beat.ep, "")
            if summary:
                lines.append(f"  Episode summary: {summary[:240]}")
    return "\n".join(lines), sorted(words)


def _check_spelling(words: list[str]):
    def check(out: SpellingOut) -> list[str]:
        problems, seen = [], Counter()
        for item in out.words:
            match = _match(item.word, words)
            if match is None:
                problems.append(f'"{item.word}" is not one of the words given')
                continue
            seen[match] += 1
            verdict = item.verdict.strip().lower()
            if verdict not in ("typo", "valid"):
                problems.append(f'"{item.word}": verdict must be typo or valid, not "{item.verdict}"')
            if verdict == "typo" and (
                not item.correction.strip() or item.correction.strip().lower() == match.lower()
            ):
                problems.append(f'"{item.word}" is a typo but has no different correction')
        for word in words:
            if seen[word] != 1:
                problems.append(f'"{word}" must be judged exactly once (got {seen[word]})')
        return problems

    return check


def judge_spelling(
    findings: list[Finding], show: Show, spec: ModelSpec
) -> tuple[list[Finding], list[dict]]:
    """Return the findings kept as typos (source ai) and every verdict."""
    if not findings:
        return [], []
    prompt, words = spelling_prompt(findings, show)
    out = ask("spelling", prompt, SpellingOut, spec, _check_spelling(words))
    for item in out.words:
        item.word = _match(item.word, words)
        item.verdict = item.verdict.strip().lower()
    verdicts = {item.word: item for item in out.words}
    kept = []
    for f in findings:
        verdict = verdicts[f.found]
        if verdict.verdict == "typo":
            correction = verdict.correction.strip()
            kept.append(
                f.model_copy(
                    update={
                        "source": "ai",
                        "confidence": "high",
                        "suggestion": correction,
                        "edit": f.edit.model_copy(update={"new": correction}) if f.edit else None,
                    }
                )
            )
    return kept, [
        {"word": w, "verdict": verdicts[w].verdict, "correction": verdicts[w].correction.strip(), "reason": verdicts[w].reason.strip()}
        for w in words
    ]


# ---- comparing two runs (model choice) ----

def agreement(run_a: str, run_b: str) -> dict:
    """Share of D1 candidates split identically, D2 groups identical, D4 verdicts identical."""
    import json

    def load(run):
        folder = Path(run)
        groups = json.loads((folder / "groups.json").read_text(encoding="utf-8"))
        candidates = json.loads((folder / "candidates.json").read_text(encoding="utf-8"))
        return groups, candidates

    (ga, ca), (gb, _) = load(run_a), load(run_b)

    def partitions(groups, candidates):
        out = {}
        for cand in candidates:
            together = [frozenset(g["spellings"]) for g in groups["names"] if g["candidate"] == cand["id"]]
            covered = {s for part in together for s in part}
            singles = [frozenset([s]) for s in cand["spellings"] if s not in covered]
            out[cand["id"]] = frozenset(together + singles)
        return out

    pa, pb = partitions(ga, ca), partitions(gb, ca)
    name_diffs = [
        f"{cid}: A {sorted(sorted(p) for p in pa[cid])} vs B {sorted(sorted(p) for p in pb.get(cid, []))}"
        for cid in sorted(pa)
        if pa[cid] != pb.get(cid)
    ]
    names = (1 - len(name_diffs) / len(pa) if pa else 1.0, len(pa), name_diffs)

    sa = {frozenset(g["labels"]) for g in ga["speakers"]}
    sb = {frozenset(g["labels"]) for g in gb["speakers"]}
    union = sa | sb
    speaker_diffs = [f"only {'A' if g in sa else 'B'}: {sorted(g)}" for g in sorted(union ^ (sa & sb), key=sorted)]
    speakers = (len(sa & sb) / len(union) if union else 1.0, len(union), speaker_diffs)

    va = {v["word"]: v["verdict"] for v in ga["spelling_verdicts"]}
    vb = {v["word"]: v["verdict"] for v in gb["spelling_verdicts"]}
    words = sorted(set(va) | set(vb))
    spell_diffs = [f"'{w}': A {va.get(w)} vs B {vb.get(w)}" for w in words if va.get(w) != vb.get(w)]
    spelling = (1 - len(spell_diffs) / len(words) if words else 1.0, len(words), spell_diffs)
    return {"names": names, "speakers": speakers, "spelling": spelling}
