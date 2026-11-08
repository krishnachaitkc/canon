"""Turn a run result into review cards: one card per issue, one place per fix.

`app.py` draws these; this module owns the lanes, the sentences, the status and the
change list. Every finding lands in exactly one card.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

KIND_LABELS = {
    "name": "NAME",
    "voice": "VOICE",
    "time": "TIME OF DAY",
    "place": "PLACE",
    "typo": "SPOKEN TYPO",
    "format": "FORMAT",
}
KIND_TITLES = {
    "name": "Name",
    "voice": "Voice label",
    "time": "Time of day",
    "place": "Place",
    "typo": "Spoken typo",
    "format": "Format",
}
WHERE_ORDER = {"heard": 0, "voice": 1, "seen": 2, "notes": 3}
WHERE_LABELS = {"heard": "Heard", "voice": "Voice", "seen": "Seen", "notes": "Notes"}
SEVERITY_WHERE = {
    "audible": "heard",
    "voice": "voice",
    "visible": "seen",
    "notes": "notes",
}
LANES = ("ready", "check", "hand")
LANE_TITLES = {
    "ready": "Ready to fix",
    "check": "Check first",
    "hand": "Fix by hand",
}
LANE_RULES = {
    "ready": "Canon is sure and can change these for you. One click: Fix all ready.",
    "check": "Canon can change these, but no AI checked them. Read each card, then press Fix.",
    "hand": (
        "Canon can't change these for you (building names, header format, labels that hide "
        "someone on purpose). Read, decide, edit your script."
    ),
}
HOW_FOUND = {
    "rule": "Rule",
    "check": "You decide",
    "code": "No-AI guess",
}
BANNED = ("canonical", "finding", "approve", "group", "detector", "severity", "patch", "report-only")
_REPLACE = {
    "canonical": "standard",
    "finding": "item",
    "approved": "accepted",
    "approve": "accept",
    "group": "set",
    "detector": "checker",
    "severity": "importance",
    "patch": "fix",
    "report-only": "notes only",
}
_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


@dataclass(frozen=True)
class Card:
    id: str
    kind: str
    lane: str
    title: str
    sentence: str
    where: str
    places: tuple[str, ...]
    fixable: tuple[str, ...]
    episodes: tuple[int, ...]
    how_found: str
    problem: str = ""
    why: str = ""
    options: tuple[str, ...] = ()
    default: str = ""
    reason: str = ""
    group_id: str = ""
    findings: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    @property
    def kind_label(self) -> str:
        return KIND_LABELS[self.kind]

    @property
    def where_label(self) -> str:
        return WHERE_LABELS[self.where]

    @property
    def lane_title(self) -> str:
        return LANE_TITLES[self.lane]

    @property
    def lane_rule(self) -> str:
        return LANE_RULES[self.lane]


def word(number: int) -> str:
    return _WORDS[number] if 0 <= number < len(_WORDS) else str(number)


def safe_text(text: str) -> str:
    """Plain on-screen text: no banned word survives into a card."""
    value = str(text or "")
    for bad, plain in _REPLACE.items():
        value = value.replace(bad, plain).replace(bad.title(), plain.capitalize())
    return value


def is_fixable(finding) -> bool:
    edit = getattr(finding, "edit", None)
    return bool(edit and edit.new and edit.new != edit.old)


def _severity_of(finding) -> str:
    return SEVERITY_WHERE.get(getattr(finding, "severity", ""), "notes")


def _where(findings: Iterable) -> str:
    return min((_severity_of(f) for f in findings), key=lambda w: WHERE_ORDER[w])


def _badge(findings: Iterable) -> str:
    findings = list(findings)
    source = findings[0].source if findings else "check"
    confidence = findings[0].confidence if findings else ""
    if source == "ai":
        return f"AI · {confidence}" if confidence else "AI"
    return HOW_FOUND.get(source, "You decide")


def _label_at(edit_at: str) -> str:
    part = edit_at[:-5] if edit_at.endswith(".time") else edit_at
    return part.replace(".S", " S").replace(".", " ")


def _episode_text(ep: int) -> str:
    return f"E{ep:02d}"


def _join_and(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _quote(items: Iterable[str]) -> str:
    return ", ".join(f"'{item}'" for item in items)


def _article(value: str) -> str:
    return "an" if value[:1].lower() in "aeiou" else "a"


def _time_short(start: str) -> str:
    return (start or "")[:5]


def _scene_short(label: str) -> str:
    """'E02 S01' -> 'E02 S1', the way a person says it."""
    return re.sub(r"\bS0*(\d+)\b", r"S\1", label or "")


def _quoted_list(items: Iterable[str]) -> str:
    return _join_and([f"'{item}'" for item in items])


def _counted(spellings: Iterable[str], counts: dict[str, int], unit: str = "") -> str:
    parts = []
    for spelling in spellings:
        if counts:
            suffix = f" {unit}" if unit else ""
            parts.append(f"{spelling} ({counts.get(spelling, 0)}{suffix})")
        else:
            parts.append(spelling)
    return ", ".join(parts)


def _picked_why(chosen: str, counts: dict[str, int], word_for_total: str) -> str:
    total = sum(counts.values())
    picked = counts.get(chosen, 0)
    return (
        f"The AI picked '{chosen}' (used {picked} of {total} {word_for_total}). "
        "You can choose another in Details."
    )


def _main_episode(findings: list) -> int:
    for finding in findings:
        if getattr(finding, "ep", 0):
            return int(finding.ep)
    return 0


def _name_card(key: str, findings: list, groups: dict, candidates: dict) -> Card:
    spot = key.split(":", 1)[1]
    group = groups.get(spot)
    first = findings[0]
    fixable = tuple(f.id for f in findings if is_fixable(f))
    episodes = tuple(sorted({int(f.ep) for f in findings if f.ep}))
    problem, why = "", ""
    if group is not None:
        spellings = list(group.spellings)
        wrong = [s for s in spellings if s != group.canonical]
        canonical = group.canonical
        title = f"{', '.join(wrong)} → {canonical}" if wrong else canonical
        where = _where(findings)
        verb = {"heard": "viewers hear", "seen": "viewers see", "notes": "the scripts write"}.get(
            where, "viewers hear"
        )
        all_of = "both" if len(spellings) == 2 else f"all {word(len(spellings))}"
        look = "look" if len(wrong) > 1 else "looks"
        sentence = (
            f"{_quote(wrong)} {look} like '{canonical}'. "
            f"{word(len(findings)).capitalize()} place{'s' if len(findings) != 1 else ''}, "
            f"{word(len(spellings))} spellings: {verb} {all_of}."
        )
        options, default, reason = tuple(spellings), canonical, safe_text(group.reason)
        candidate = candidates.get(group.id.split(".", 1)[0])
        # Count only this card's spellings: one look-alike candidate can hold two families.
        counts = (
            {s: n for s, n in dict(candidate.spellings).items() if s in spellings}
            if candidate is not None
            else {}
        )
        if counts:
            problem = (
                f"One name is written {len(spellings)} ways: "
                f"{_counted(spellings, counts)}. Viewers would hear all of them."
            )
            most = max(counts.values())
            if counts.get(canonical, 0) < most:
                why = _picked_why(canonical, counts, "times")
            else:
                total = sum(counts.get(s, 0) for s in spellings)
                why = (
                    f"'{canonical}' is the spelling the script uses most "
                    f"({counts.get(canonical, 0)} of {total})."
                )
        else:
            problem = (
                f"One name is written {len(spellings)} ways: {', '.join(spellings)}. "
                "Viewers would hear all of them."
            )
            why = f"'{canonical}' is the spelling used most often."
    else:
        spellings = [part.strip() for part in first.found.split(" / ") if part.strip()]
        title = f"{_join_and(spellings)}?"
        where = _where(findings)
        sentence = (
            f"The AI was not sure whether {_quote(spellings[:1])[0] if len(spellings) == 1 else _join_and(_quote(spellings))} "
            "are one name. You decide."
        )
        if len(spellings) < 2:
            sentence = f"The AI was not sure about '{first.found}'. You decide."
        options, default, reason = (), "", safe_text(first.context)
        named = _join_and(spellings) if len(spellings) > 1 else f"'{first.found}'"
        problem = f"The AI could not tell whether {named} are one name."
        why = "If they are, give them one name. If not, leave both."
    lane = _lane(findings, fixable)
    return Card(
        id=key,
        kind="name",
        lane=lane,
        title=title,
        sentence=sentence,
        where=where,
        places=tuple(f.id for f in findings),
        fixable=fixable,
        episodes=episodes,
        how_found=_badge(findings),
        problem=safe_text(problem),
        why=safe_text(why),
        options=options,
        default=default,
        reason=reason,
        group_id=getattr(group, "id", ""),
        findings=tuple(findings),
    )


def _voice_card(key: str, findings: list, groups: dict, stats: dict) -> Card:
    spot = key.split(":", 1)[1]
    group = groups.get(spot)
    first = findings[0]
    fixable = tuple(f.id for f in findings if is_fixable(f))
    episodes = tuple(sorted({int(f.ep) for f in findings if f.ep}))
    lane = _lane(findings, fixable)
    problem, why = "", ""
    if group is not None and group.intentional:
        labels = ", ".join(group.labels)
        title = f"{labels} → one voice"
        sentence = (
            "One person under several labels on purpose. Use one voice for all of them. "
            "The labels can stay."
        )
        options, default = tuple(group.labels), group.canonical
        reason = safe_text(group.reason)
        problem = "The AI says these labels hide one person on purpose (a disguise or a reveal)."
        why = "Nothing to rename. Cast one voice for all of them."
    elif group is not None:
        wrong = [label for label in group.labels if label != group.canonical]
        title = f"{', '.join(wrong)} → {group.canonical}"
        count = len(findings)
        speakers = ", ".join(wrong)
        verb = "speaks" if len(wrong) == 1 else "speak"
        voices = word(len(group.labels))
        sentence = (
            f"{speakers} {verb} {count} {'line' if count == 1 else 'lines'} that belong to "
            f"{group.canonical}. {voices.capitalize()} labels mean {voices} voices for one person."
        )
        options, default = tuple(group.labels), group.canonical
        reason = safe_text(group.reason)
        labels = list(group.labels)
        counts = {label: stats[label].lines for label in labels if label in stats}
        if counts:
            parts = [f"{label} ({counts.get(label, 0)} lines)" for label in labels]
            problem = (
                f"One character speaks under {len(labels)} labels: {_join_and(parts)}. "
                f"Each label gets its own voice, so one person would sound like "
                f"{word(len(labels))} people."
            )
            total = sum(counts.values())
            most = max(counts.values())
            if counts.get(group.canonical, 0) < most:
                why = _picked_why(group.canonical, counts, "lines")
            else:
                why = (
                    f"'{group.canonical}' is the label used most "
                    f"({counts.get(group.canonical, 0)} of {total} lines)."
                )
        else:
            problem = (
                f"One character speaks under {len(labels)} labels: {_join_and(labels)}. "
                f"Each label gets its own voice, so one person would sound like "
                f"{word(len(labels))} people."
            )
            why = f"'{group.canonical}' is the label used most."
    elif spot.startswith("unsure:"):
        labels = [part for part in spot[len("unsure:"):].split("+") if part]
        title = f"{' or '.join(labels)}?"
        sentence = (
            f"The AI was not sure whether {_join_and(labels)} are one person. You decide."
        )
        options, default, reason = (), "", safe_text(first.context)
        problem = f"The AI could not tell whether {_join_and(labels)} are one person."
        why = "If they are, give them one label. If not, leave both."
    elif spot.startswith("veto:"):
        labels = spot[len("veto:"):]
        title = f"{labels}?"
        found = first.found.rstrip(".").replace(" matched to ", " was matched to ", 1)
        sentence = f"{found}. They may be two people."
        options, default, reason = (), "", safe_text(first.context)
        match = re.match(r"(.+?) was matched to (.+?), but both speak in (.+)", found + ".")
        if match:
            person, other, when = match.groups()
            scene = _scene_short(re.match(r"(E\d+ S\d+)", when).group(1)) if re.match(r"(E\d+ S\d+)", when) else when
            problem = (
                f"{person} was matched to {other}, but both speak in {scene}, "
                "so they may be two people."
            )
        else:
            problem = found + "."
        why = "If so, give one of them a new label."
    else:
        title = first.where
        sentence = safe_text(first.found)
        options, default, reason = (), "", safe_text(first.context)
        problem = safe_text(first.found).rstrip(".") + "." if first.found else "Canon could not decide this."
        why = "Read the lines, then decide."
    return Card(
        id=key,
        kind="voice",
        lane=lane,
        title=title,
        sentence=sentence,
        where="voice",
        places=tuple(f.id for f in findings),
        fixable=fixable,
        episodes=episodes,
        how_found=_badge(findings),
        problem=safe_text(problem),
        why=safe_text(why),
        options=options,
        default=default,
        reason=reason,
        group_id=getattr(group, "id", ""),
        findings=tuple(findings),
    )


def _time_card(key: str, findings: list) -> Card:
    first = findings[0]
    edit = first.edit
    fixable = tuple(f.id for f in findings if is_fixable(f))
    episodes = tuple(sorted({int(f.ep) for f in findings if f.ep}))
    caution = " If time passes between the two places, this is not a mistake."
    left, right = (first.where.split(" -> ") + ["", ""])[:2]
    left_label, right_label = _scene_short(left), _scene_short(right)
    times = (first.found.split(" -> ") + ["", ""])[:2]
    early = times[0].split(" ")[0] if times[0] else ""
    late = times[1].split(" ")[0] if len(times) > 1 and times[1] else ""
    if edit is not None:
        target = _label_at(edit.at)
        target_short = _scene_short(target)
        before, after = (first.where.split(" -> ") + ["", ""])[:2]
        earlier = first.found.split(" -> ")[0].split(" ")[0]
        sentence = (
            f"{before} is {earlier}. {after} continues the same moment but says {edit.old}. "
            f"Fix: {target} → {edit.new}."
        )
        if len(findings) > 1:
            sentence = (
                f"{word(len(findings)).capitalize()} cuts continue the same moment and flip the time of day. "
                f"Fix: {target} → {edit.new}."
            )
        title = f"{target}: {edit.old} → {edit.new}"
        if len(findings) > 1:
            problem = (
                f"{target_short} is {edit.old} between {edit.new} scenes that continue the same moment. "
                "The picture would jump."
            )
        else:
            match = re.search(r'notes say "([^"]+)"', first.suggestion or "")
            if match:
                phrase = match.group(1).lower()
                next_ep = right.split(" ")[0]
                prev_ep = left.split(" ")[0]
                if phrase.startswith(("immediately follows", "directly follows")):
                    lead = f"The notes say {next_ep} {phrase} {prev_ep}"
                else:
                    lead = f"The notes say {next_ep} {phrase} after {prev_ep}"
            else:
                lead = f"The script calls the cut {left_label} → {right_label} continuous"
            problem = (
                f"{lead}, but {left_label} is {early} and {right_label} is {late}. "
                f"The picture would jump from {early.lower()} to {late.lower()}."
            )
        both = len(findings) > 1 or "also differs" in (first.suggestion or "")
        if both:
            why = f"Canon suggests {edit.new} to match the scenes on both sides."
        elif target_short == right_label:
            why = f"Canon suggests {edit.new} to match the scene before it."
        else:
            why = f"Canon suggests {edit.new} to match the scene after it."
    else:
        sentence = safe_text(first.found)
        title = first.found.replace(" -> ", " → ")
        target = first.where
        problem = (
            f"{left_label} is {early} and {right_label} is {late}, but the cut continues "
            "the same moment. The picture would jump."
        )
        why = "Check the two headings and edit the one that is wrong."
    return Card(
        id=key,
        kind="time",
        lane=_lane(findings, fixable),
        title=title,
        sentence=sentence,
        where="seen",
        places=tuple(f.id for f in findings),
        fixable=fixable,
        episodes=episodes,
        how_found=_badge(findings),
        problem=safe_text(problem + caution),
        why=safe_text(why),
        group_id="time",
        findings=tuple(findings),
    )


def _place_card(key: str, finding) -> Card:
    found = getattr(finding, "found", "")
    before, after = (found.split(" -> ") + ["", ""])[:2]
    sentence = (
        f"The same moment is {_article(before)} {before} before the cut and "
        f"{_article(after)} {after} after it. Pick one and fix the heading."
    )
    problem = (
        f"In one continuous moment the building is {_article(before)} {before} before the cut "
        f"and {_article(after)} {after} after it. The video would draw two different buildings."
    )
    why = "Canon can't tell which name is right. Pick one and edit the heading."
    return Card(
        id=key,
        kind="place",
        lane="hand",
        title=f"{before} or {after}?" if after else found,
        sentence=sentence,
        where="seen",
        places=(finding.id,),
        fixable=(),
        episodes=(int(finding.ep),) if finding.ep else (),
        how_found=_badge([finding]),
        problem=safe_text(problem),
        why=safe_text(why),
        findings=(finding,),
    )


def _typo_card(key: str, findings: list) -> Card:
    first = findings[0]
    edit = first.edit
    fixable = tuple(f.id for f in findings if is_fixable(f))
    episodes = tuple(sorted({int(f.ep) for f in findings if f.ep}))
    new = edit.new if edit is not None else first.suggestion
    if len(findings) == 1:
        sentence = (
            f"'{first.found}' in a spoken line "
            f"({_episode_text(first.ep)} {_time_short(first.start)}). Fix: '{new}'."
        )
    else:
        sentence = (
            f"'{first.found}' in {word(len(findings))} spoken lines "
            f"({_episode_text(episodes[0])}–{_episode_text(episodes[-1])}). Fix: '{new}'."
        )
    if first.source == "check":
        sentence += " Dictionary guess. Old words like 'gaol' are real."
    problem = f"'{first.found}' is not an English word, so the voice would say it wrong."
    if first.source == "ai":
        why = f"The AI read the line and suggests '{new}'."
    else:
        why = "Closest dictionary word. Old words like 'gaol' are real, so check first."
    return Card(
        id=key,
        kind="typo",
        lane=_lane(findings, fixable),
        title=f"'{first.found}' → '{new}'",
        sentence=sentence,
        where="heard",
        places=tuple(f.id for f in findings),
        fixable=fixable,
        episodes=episodes,
        how_found=_badge(findings),
        problem=safe_text(problem),
        why=safe_text(why),
        findings=tuple(findings),
    )


def _format_card(key: str, findings: list) -> Card:
    first = findings[0]
    sentence = (
        f"{_episode_text(first.ep)} writes '{first.found}' where other episodes "
        f"write '{first.suggestion}'."
    )
    if len(findings) > 1:
        sentence = (
            f"{_episode_text(first.ep)} writes '{first.found}' in {word(len(findings))} places "
            f"where other episodes write '{first.suggestion}'."
        )
    labels: list[str] = []
    suggestions: list[str] = []
    for finding in findings:
        if finding.found not in labels:
            labels.append(finding.found)
        if finding.suggestion and finding.suggestion not in suggestions:
            suggestions.append(finding.suggestion)
    other = 0
    for finding in findings:
        match = re.search(r"(\d+)\s+episodes?", finding.context or "")
        if match:
            other = max(other, int(match.group(1)))
    others = f"the other {other} episodes" if other else "the other episodes"
    episode = _episode_text(first.ep)
    caps = all(label.isupper() for label in labels)
    if len(labels) == 1:
        if caps:
            problem = (
                f"{episode} writes {labels[0]} in capitals; {others} "
                f"write '{suggestions[0]}'."
            )
        else:
            problem = (
                f"{episode} writes {_quoted_list(labels)} where {others} "
                f"write {_quoted_list(suggestions)}."
            )
    elif caps:
        problem = (
            f"{episode} writes {_join_and(labels)} in capitals; {others} "
            f"write them as {_quoted_list(suggestions)}."
        )
    else:
        problem = (
            f"{episode} writes {_quoted_list(labels)} where {others} "
            f"write {_quoted_list(suggestions)}."
        )
    why = "Tools that read scripts by these labels could miss this episode. It is a one-time edit."
    return Card(
        id=key,
        kind="format",
        lane="hand",
        title=f"{first.found} → {first.suggestion}",
        sentence=sentence,
        where="notes",
        places=tuple(f.id for f in findings),
        fixable=(),
        episodes=(int(first.ep),) if first.ep else (),
        how_found=_badge(findings),
        problem=safe_text(problem),
        why=safe_text(why),
        findings=tuple(findings),
    )


def _lane(findings: list, fixable: tuple[str, ...]) -> str:
    if not fixable:
        return "hand"
    if all(f.source == "ai" and f.confidence in ("high", "medium") for f in findings):
        return "ready"
    return "check"


def _card_key(finding) -> str:
    group = getattr(finding, "group", "") or ""
    detector = finding.detector
    if detector in ("names", "speakers"):
        if detector == "names":
            return f"name:{group or finding.id}"
        return f"voice:{group or finding.id}"
    if detector == "joins":
        if group == "time" and is_fixable(finding):
            edit = finding.edit
            return f"time:{edit.at}|{edit.old}|{edit.new}"
        if group == "time":
            return f"time:{finding.id}"
        return f"place:{finding.id}"
    if detector == "spelling":
        return f"typo:{group or finding.found}"
    if detector == "format":
        return f"format:{finding.ep}"
    return f"other:{finding.id}"


def build_cards(result: dict) -> list[Card]:
    """Every finding in exactly one card, lanes sorted: ready, check, hand."""
    findings = list(result.get("findings") or [])
    names = {group.id: group for group in result.get("name_groups") or []}
    speakers = {group.id: group for group in result.get("speaker_groups") or []}
    candidates = {candidate.id: candidate for candidate in result.get("candidates") or []}
    stats = result.get("stats") or {}
    buckets: dict[str, list] = {}
    for finding in findings:
        buckets.setdefault(_card_key(finding), []).append(finding)
    cards = []
    for key, grouped in buckets.items():
        first = grouped[0]
        if first.detector == "names":
            cards.append(_name_card(key, grouped, names, candidates))
        elif first.detector == "speakers":
            cards.append(_voice_card(key, grouped, speakers, stats))
        elif first.detector == "joins":
            if key.startswith("time:"):
                cards.append(_time_card(key, grouped))
            else:
                cards.append(_place_card(key, first))
        elif first.detector == "spelling":
            cards.append(_typo_card(key, grouped))
        elif first.detector == "format":
            cards.append(_format_card(key, grouped))
    cards.sort(key=lambda card: (LANES.index(card.lane), -len(card.places), _main_episode(list(card.findings)), card.id))
    return cards


def lanes(cards: list[Card]) -> dict[str, list[Card]]:
    return {lane: [card for card in cards if card.lane == lane] for lane in LANES}


def fixable_places(cards: list[Card], lane: str | None = None) -> list[str]:
    return [
        fid
        for card in cards
        if lane is None or card.lane == lane
        for fid in card.fixable
    ]


def ready_places(cards: list[Card]) -> list[str]:
    return fixable_places(cards, "ready")


def check_places(cards: list[Card]) -> list[str]:
    return fixable_places(cards, "check")


def tick(cards: list[Card], fix: dict[str, bool], lane: str) -> dict[str, bool]:
    """A new decisions map with every fixable place in the lane ticked."""
    updated = dict(fix)
    for card in cards:
        if card.lane == lane:
            for fid in card.fixable:
                updated[fid] = True
    return updated


def status(card: Card, fix: dict[str, bool] | None = None, hand: dict[str, str] | None = None) -> str:
    fix, hand = fix or {}, hand or {}
    choice = hand.get(card.id)
    if choice == "by_hand":
        return "Will fix by hand"
    if choice == "not_a_problem":
        return "Not a problem"
    if not card.fixable:
        return "Not decided"
    ticked = sum(1 for fid in card.fixable if fix.get(fid))
    if ticked:
        return f"Fixing {ticked} of {len(card.fixable)}"
    if all(fid in fix for fid in card.fixable):
        return "Keep as written"
    return "Not decided"


def is_decided(card: Card, fix: dict[str, bool] | None = None, hand: dict[str, str] | None = None) -> bool:
    fix, hand = fix or {}, hand or {}
    if hand.get(card.id):
        return True
    if any(fix.get(fid) for fid in card.fixable):
        return True
    return bool(card.fixable) and all(fid in fix for fid in card.fixable)


def decided_count(cards: list[Card], fix: dict[str, bool] | None = None, hand: dict[str, str] | None = None) -> int:
    return sum(1 for card in cards if is_decided(card, fix, hand))


def will_fix_count(cards: list[Card], fix: dict[str, bool] | None = None) -> int:
    fix = fix or {}
    return sum(1 for card in cards for fid in card.fixable if fix.get(fid))


def progress_text(cards: list[Card], fix: dict[str, bool] | None = None, hand: dict[str, str] | None = None) -> str:
    short = {"ready": "Ready", "check": "Check first", "hand": "Fix by hand"}
    parts = []
    for lane in LANES:
        members = [card for card in cards if card.lane == lane]
        done = sum(1 for card in members if is_decided(card, fix, hand))
        parts.append(f"{short[lane]} {done} of {len(members)}")
    places = will_fix_count(cards, fix)
    parts.append(f"{places} {'place' if places == 1 else 'places'} will be fixed")
    return " · ".join(parts)


def metrics(cards: list[Card]) -> dict[str, int]:
    counts = {"issues": len(cards), "heard": 0, "voice": 0, "seen": 0, "notes": 0}
    for card in cards:
        counts[card.where] += 1
    counts["episodes"] = len({ep for card in cards for ep in card.episodes})
    return counts


def change_rows(
    cards: list[Card],
    fix: dict[str, bool] | None = None,
    choices: dict[str, str] | None = None,
) -> list[dict]:
    """Every ticked place, in card order, as (old -> new) rows for the preview."""
    fix, choices = fix or {}, choices or {}
    rows = []
    for card in cards:
        chosen = choices.get(card.group_id) if card.group_id else None
        for finding in card.findings:
            if finding.id not in card.fixable or not fix.get(finding.id):
                continue
            edit = finding.edit
            new = edit.new
            if chosen and card.default and chosen != card.default:
                tail = edit.new[len(card.default):] if edit.new.startswith(card.default) else ""
                new = chosen + tail
            rows.append(
                {
                    "episode": int(finding.ep),
                    "time": finding.start,
                    "where": finding.where,
                    "where_label": WHERE_LABELS[_severity_of(finding)],
                    "context": finding.context,
                    "was": edit.old,
                    "now": new,
                    "issue": card.title,
                    "kind": card.kind_label,
                    "card": card.id,
                }
            )
    return rows


def issue_rows(
    cards: list[Card],
    fix: dict[str, bool] | None = None,
    choices: dict[str, str] | None = None,
) -> list[dict]:
    """One entry per issue being fixed, with its change rows, in card order."""
    rows = change_rows(cards, fix, choices)
    by_card: dict[str, list[dict]] = {}
    for row in rows:
        by_card.setdefault(row["card"], []).append(row)
    issues = []
    for card in cards:
        selected = by_card.get(card.id)
        if not selected:
            continue
        issues.append(
            {
                "issue": card.title,
                "kind": card.kind_label,
                "places": len(selected),
                "episodes": tuple(sorted({row["episode"] for row in selected if row["episode"]})),
                "rows": selected,
            }
        )
    return issues


def by_hand_rows(cards: list[Card], hand: dict[str, str] | None = None) -> list[dict]:
    hand = hand or {}
    rows = []
    for card in cards:
        if hand.get(card.id) != "by_hand":
            continue
        spot = card.findings[0] if card.findings else None
        rows.append(
            {
                "episode": int(getattr(spot, "ep", 0)),
                "where": getattr(spot, "where", ""),
                "title": card.title,
                "sentence": card.sentence,
                "problem": card.problem,
                "why": card.why,
            }
        )
    return rows


def download_summary(
    cards: list[Card],
    fix: dict[str, bool] | None = None,
    hand: dict[str, str] | None = None,
) -> str:
    rows = change_rows(cards, fix)
    episodes = len({row["episode"] for row in rows if row["episode"]})
    issues = len({row["card"] for row in rows})
    fixed = {row["card"] for row in rows}
    check_left = sum(
        1 for card in cards if card.lane == "check" and card.id not in fixed
    )
    hand_count = sum(1 for card in cards if card.lane == "hand")
    text = (
        f"Fixing {len(rows)} {'place' if len(rows) == 1 else 'places'} in "
        f"{episodes} {'episode' if episodes == 1 else 'episodes'} across "
        f"{issues} {'issue' if issues == 1 else 'issues'}."
    )
    text += (
        f" Not changed: {check_left} check-first and {hand_count} by-hand issues."
    )
    return text + " Your original files are not changed."


def apart_lines(result: dict) -> list[str]:
    """One plain line per kept-apart pair, with no IDs and no spellings shown on a card."""
    shown = {option for card in build_cards(result) for option in card.options}
    grouped: dict[str, list[str]] = {}
    for entry in result.get("apart") or []:
        candidate = entry.get("candidate") or ""
        for spelling in entry.get("spellings") or []:
            if spelling not in grouped.setdefault(candidate, []):
                grouped[candidate].append(spelling)
    lines = []
    for candidate in sorted(grouped):
        spellings = [s for s in grouped[candidate] if s not in shown]
        if len(spellings) < 2:
            continue
        lines.append(
            f"{' · '.join(spellings)}: the AI judged these to be different names. No change."
        )
    return lines


def all_text(cards: list[Card]) -> list[str]:
    """Every displayed card string, for the wording check."""
    text = []
    for card in cards:
        text.extend(
            [
                card.kind_label,
                card.where_label,
                card.title,
                card.sentence,
                card.problem,
                card.why,
                card.how_found,
                card.reason,
            ]
        )
        text.extend(card.options)
        text.append(card.default)
    return [item for item in text if item]
