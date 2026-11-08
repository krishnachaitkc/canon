"""Turn a run result into review cards: one card per issue, one place per fix.

`app.py` draws these; this module owns the lanes, the sentences, the status and the
change list. Every finding lands in exactly one card.
"""

from __future__ import annotations

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
    "ready": "AI-checked. The fix is automatic.",
    "check": "Found by rules or code without AI review. Look before you fix.",
    "hand": "Canon can't edit these. Decide, then fix in your script.",
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


def _main_episode(findings: list) -> int:
    for finding in findings:
        if getattr(finding, "ep", 0):
            return int(finding.ep)
    return 0


def _name_card(key: str, findings: list, groups: dict) -> Card:
    spot = key.split(":", 1)[1]
    group = groups.get(spot)
    first = findings[0]
    fixable = tuple(f.id for f in findings if is_fixable(f))
    episodes = tuple(sorted({int(f.ep) for f in findings if f.ep}))
    if group is not None:
        wrong = [s for s in group.spellings if s != group.canonical]
        canonical = group.canonical
        title = f"{', '.join(wrong)} → {canonical}" if wrong else canonical
        where = _where(findings)
        verb = {"heard": "viewers hear", "seen": "viewers see", "notes": "the scripts write"}.get(
            where, "viewers hear"
        )
        all_of = "both" if len(group.spellings) == 2 else f"all {word(len(group.spellings))}"
        look = "look" if len(wrong) > 1 else "looks"
        sentence = (
            f"{_quote(wrong)} {look} like '{canonical}'. "
            f"{word(len(findings)).capitalize()} place{'s' if len(findings) != 1 else ''}, "
            f"{word(len(group.spellings))} spellings: {verb} {all_of}."
        )
        options, default, reason = tuple(group.spellings), group.canonical, safe_text(group.reason)
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
        options=options,
        default=default,
        reason=reason,
        group_id=getattr(group, "id", ""),
        findings=tuple(findings),
    )


def _voice_card(key: str, findings: list, groups: dict) -> Card:
    spot = key.split(":", 1)[1]
    group = groups.get(spot)
    first = findings[0]
    fixable = tuple(f.id for f in findings if is_fixable(f))
    episodes = tuple(sorted({int(f.ep) for f in findings if f.ep}))
    lane = _lane(findings, fixable)
    if group is not None and group.intentional:
        labels = ", ".join(group.labels)
        title = f"{labels} → one voice"
        sentence = (
            "One person under several labels on purpose. Use one voice for all of them. "
            "The labels can stay."
        )
        options, default = tuple(group.labels), group.canonical
        reason = safe_text(group.reason)
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
    elif spot.startswith("unsure:"):
        labels = [part for part in spot[len("unsure:"):].split("+") if part]
        title = f"{' or '.join(labels)}?"
        sentence = (
            f"The AI was not sure whether {_join_and(labels)} are one person. You decide."
        )
        options, default, reason = (), "", safe_text(first.context)
    elif spot.startswith("veto:"):
        labels = spot[len("veto:"):]
        title = f"{labels}?"
        found = first.found.rstrip(".").replace(" matched to ", " was matched to ", 1)
        sentence = f"{found}. They may be two people."
        options, default, reason = (), "", safe_text(first.context)
    else:
        title = first.where
        sentence = safe_text(first.found)
        options, default, reason = (), "", safe_text(first.context)
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
    if edit is not None:
        target = _label_at(edit.at)
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
    else:
        sentence = safe_text(first.found)
        title = first.found.replace(" -> ", " → ")
        target = first.where
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
    buckets: dict[str, list] = {}
    for finding in findings:
        buckets.setdefault(_card_key(finding), []).append(finding)
    cards = []
    for key, grouped in buckets.items():
        first = grouped[0]
        if first.detector == "names":
            cards.append(_name_card(key, grouped, names))
        elif first.detector == "speakers":
            cards.append(_voice_card(key, grouped, speakers))
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
    places = will_fix_count(cards, fix)
    return (
        f"{decided_count(cards, fix, hand)} of {len(cards)} decided · "
        f"{places} {'place' if places == 1 else 'places'} will be fixed"
    )


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
                    "was": edit.old,
                    "now": new,
                    "issue": card.title,
                }
            )
    return rows


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
    by_hand = sum(1 for card in cards if (hand or {}).get(card.id) == "by_hand")
    text = (
        f"Fixing {len(rows)} {'place' if len(rows) == 1 else 'places'} in "
        f"{episodes} {'episode' if episodes == 1 else 'episodes'}."
    )
    if by_hand:
        text += f" {by_hand} {'issue' if by_hand == 1 else 'issues'} to fix by hand."
    return text + " Your original files are not changed."


def all_text(cards: list[Card]) -> list[str]:
    """Every displayed card string, for the wording check."""
    text = []
    for card in cards:
        text.extend([card.kind_label, card.where_label, card.title, card.sentence, card.how_found, card.reason])
        text.extend(card.options)
        text.append(card.default)
    return [item for item in text if item]
