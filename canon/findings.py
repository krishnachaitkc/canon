"""One findings table for every detector, and the cast sheet. Every location, quote and count
is written by code; the judges only decide which spellings and labels belong together."""

from .edits import nth_before
from .mentions import MentionIndex, clean_surface, iter_texts, raw_key
from .models import Edit, Finding, NameGroup, Show, SpeakerGroup, SpeakerStat
from .parse import format_timestamp, sec

SEVERITY_ORDER = {"audible": 0, "voice": 1, "visible": 2, "notes": 3}
CONTEXT_CHARS = 70


def cut(text: str, start: int, length: int) -> str:
    """At most CONTEXT_CHARS characters around [start, start+length)."""
    pad = max(0, (CONTEXT_CHARS - length) // 2)
    left, right = max(0, start - pad), min(len(text), start + length + pad)
    if left == 0:
        right = min(len(text), max(right, CONTEXT_CHARS))
    return ("…" if left else "") + text[left:right].strip() + ("…" if right < len(text) else "")


def mention_severity(kind: str, beat_id: str) -> str:
    if kind == "spoken":
        return "audible"
    if kind in ("shot", "action", "sfx") or beat_id.endswith(".heading"):
        return "visible"           # the video model reads descriptions and scene headings
    return "notes"                 # logline and continuity notes


def _singular(spelling: str, key: str) -> str:
    """A group's spelling without a plural "s" when its most common form is the plural."""
    return spelling[:-1] if raw_key(spelling) != key and spelling.lower().endswith("s") else spelling


def name_rows(groups: list[NameGroup], index: MentionIndex, show: Show) -> list[Finding]:
    """One row per mention of a non-canonical spelling in an accepted name group.

    The fix keeps the mention's possessive and plural: "Thomson's" -> "Thorne's",
    "the Thomsons" -> "the Thornes".
    """
    texts = {t.beat_id: t.text for t in iter_texts(show)}
    by_key = index.by_key()
    rows = []
    for group in groups:
        if not group.same_entity or group.canonical not in group.spellings:
            continue
        canonical_key = group.keys[group.spellings.index(group.canonical)]
        base = _singular(group.canonical, canonical_key)
        for key in group.keys:
            if key == canonical_key:
                continue
            for m in sorted(by_key.get(key, []), key=lambda m: (m.ep, m.start, m.beat_id, m.char_start)):
                written = clean_surface(m.surface)
                new = base + ("s" if raw_key(written) != m.key else "")
                possessive = m.surface[len(written):]
                text = texts.get(m.beat_id, "")
                rows.append(
                    Finding(
                        detector="names",
                        severity=mention_severity(m.kind, m.beat_id),
                        source=group.source,
                        confidence=group.confidence,
                        ep=m.ep,
                        scene=m.scene,
                        start=format_timestamp(m.start),
                        where=m.beat_id,
                        found=m.surface,
                        suggestion=new + possessive,
                        context=cut(text, m.char_start, len(m.surface)),
                        group=group.id,
                        edit=Edit(at=m.beat_id, old=written, new=new, nth=nth_before(text, written, m.char_start)),
                    )
                )
    return rows


def lines_by_label(show: Show) -> dict[str, list]:
    lines: dict[str, list] = {}
    for episode in show.episodes:
        for beat in episode.beats:
            if beat.kind == "line" and beat.speaker:
                lines.setdefault(beat.speaker, []).append(beat)
    return lines


def speaker_rows(groups: list[SpeakerGroup], show: Show) -> list[Finding]:
    """One row per line under a non-canonical label; one casting note per intentional group."""
    lines = lines_by_label(show)
    rows = []
    for group in groups:
        if group.intentional:
            first = min((b for label in group.labels for b in lines.get(label, [])), key=lambda b: (b.ep, b.start))
            rows.append(
                Finding(
                    detector="speakers",
                    severity="voice",
                    source=group.source,
                    confidence=group.confidence,
                    ep=first.ep,
                    scene=first.scene,
                    start=format_timestamp(first.start),
                    where=" / ".join(group.labels),
                    found=f"{' and '.join(group.labels)} are one character, hidden on purpose",
                    suggestion=f"Cast one voice for all of them; the labels may stay. The character is {group.canonical}.",
                    context=group.reason[:CONTEXT_CHARS],
                    group=group.id,
                )
            )
            continue
        for label in group.labels:
            if label == group.canonical:
                continue
            for beat in sorted(lines.get(label, []), key=lambda b: (b.ep, b.start)):
                rows.append(
                    Finding(
                        detector="speakers",
                        severity="voice",
                        source=group.source,
                        confidence=group.confidence,
                        ep=beat.ep,
                        scene=beat.scene,
                        start=format_timestamp(beat.start),
                        where=beat.id,
                        found=label,
                        suggestion=group.canonical,
                        context=cut(beat.text, 0, 0),
                        group=group.id,
                        edit=Edit(at=f"{beat.id}.label", old=label, new=group.canonical),
                    )
                )
    return rows


def sort_and_number(findings: list[Finding]) -> list[Finding]:
    """Severity (audible, voice, visible, notes), then episode, then time; IDs F0001 on."""

    def key(f: Finding):
        return (SEVERITY_ORDER[f.severity], f.ep, sec(f.start), f.detector, f.where, f.found)

    ordered = sorted(findings, key=key)
    for number, finding in enumerate(ordered, 1):
        finding.id = f"F{number:04d}"
    return ordered


def episode_ranges(episodes) -> str:
    """[1, 2, 3, 5] -> "1-3, 5"."""
    nums = sorted(set(episodes))
    parts, start = [], None
    for i, n in enumerate(nums):
        if start is None:
            start = n
        if i + 1 == len(nums) or nums[i + 1] != n + 1:
            parts.append(f"{start}" if start == n else f"{start}-{n}")
            start = None
    return ", ".join(parts)


def cast_rows(stats: dict[str, SpeakerStat], groups: list[SpeakerGroup], findings: list[Finding]) -> list[dict]:
    """One row per character: every label it speaks under, and what casting must know."""
    checks: dict[str, list[str]] = {}
    for f in findings:
        if f.detector == "speakers" and f.source == "check":
            for label in f.where.split(" / "):
                checks.setdefault(label, []).append(f.id)
    rows, grouped = [], set()
    for group in groups:
        labels = [group.canonical] + [l for l in group.labels if l != group.canonical]
        grouped.update(labels)
        if group.intentional:
            note = "One character under several labels on purpose: use one voice; the labels may stay."
        else:
            note = f"One character: use one voice and label every line {group.canonical}."
        rows.append(_cast_row(group.canonical, labels, stats, note, group, checks))
    for label in stats:
        if label not in grouped:
            rows.append(_cast_row(label, [label], stats, "", None, checks))
    return sorted(rows, key=lambda r: (-r["lines"], r["character"]))


def _cast_row(character, labels, stats, note, group, checks) -> dict:
    episodes = [e for label in labels for e in stats[label].episodes]
    ids = sorted({i for label in labels for i in checks.get(label, [])})
    return {
        "character": character,
        "labels": " / ".join(labels),
        "lines": sum(stats[label].lines for label in labels),
        "lines_per_label": "; ".join(f"{label} {stats[label].lines}" for label in labels),
        "episodes": episode_ranges(episodes),
        "group": group.id if group else "",
        "source": f"{group.source} {group.confidence}" if group else "",
        "note": note,
        "vetoes": "; ".join(f"{a} + {b} both speak in {where}" for a, b, where in (group.vetoed if group else [])),
        "checks": " ".join(ids),
    }
