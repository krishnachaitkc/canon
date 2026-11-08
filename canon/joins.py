"""D3: scene joins. Code only.

T1 (rule): on a join the script itself calls continuous, the time of day flips between day
and night. The script contradicts itself, so this is a rule finding.

L1 (check): on a continuous join, the building is named as two different types of the same
kind of place (cathedral and chapel, cottage and cabin, manor and castle). Moving between two
different places (a barn, then a cottage) is normal, so it is never flagged. Code cannot tell
whether it is one building or two, so a person decides.
"""

import re
from dataclasses import dataclass

from .models import Edit, Finding, Scene, Show
from .parse import format_timestamp

DAY = {"DAY", "MORNING", "NOON", "AFTERNOON"}
NIGHT = {"NIGHT", "MIDNIGHT", "PRE-DAWN"}
EDGE = {"DUSK", "DAWN", "EVENING", "TWILIGHT"}

# The episode notes call a join continuous only with these phrases. "Transitions directly
# into the trial" or "cutting directly to" describe the story's order, not real time.
CONTINUOUS_NOTES = re.compile(
    r"\b(?:picks?\s+up|picking\s+up|follows?|following)\s+(?:immediately|directly|right\s+after)\b"
    r"|\b(?:immediately|directly)\s+follows?\b"
    r"|\bpicks?\s+up\s+right\s+after\b",
    re.IGNORECASE,
)
TIME_SKIP = re.compile(
    r"\b(?:morning|next\s+day|later|hours|dawn|the\s+following\s+(?:day|morning|night)|days|weeks|night\s+in)\b",
    re.IGNORECASE,
)

# Kinds of place. A change inside one kind on a continuous join is the slip worth a look.
FAMILIES = {
    "worship": {"cathedral", "chapel", "church", "abbey"},
    "great house": {"manor", "castle", "palace"},
    "small house": {"cottage", "cabin", "hut"},
    "inn": {"tavern", "inn"},
    "barn": {"barn"},
}
TYPE_FAMILY = {word: family for family, words in FAMILIES.items() for word in words}
TYPE_WORD = re.compile(r"\b(" + "|".join(sorted(TYPE_FAMILY)) + r")\b", re.IGNORECASE)


def time_class(token: str) -> str | None:
    token = token.upper().strip()
    if token in DAY:
        return "day"
    if token in NIGHT:
        return "night"
    if token in EDGE:
        return "edge"
    return None


def heading_type(scene: Scene) -> str | None:
    found = TYPE_WORD.findall(scene.place)
    return found[-1].lower() if found else None


@dataclass
class Join:
    prev: Scene
    next: Scene
    continuous: bool
    why: str                 # "heading CONTINUOUS" or the notes phrase
    notes: str = ""


def scene_order(show: Show) -> list[Scene]:
    return [scene for episode in show.episodes for scene in sorted(episode.scenes, key=lambda s: s.num)]


def build_joins(show: Show) -> list[Join]:
    notes = {episode.num: episode.continuity for episode in show.episodes}
    scenes = scene_order(show)
    joins = []
    for prev, nxt in zip(scenes, scenes[1:]):
        if nxt.time == "CONTINUOUS":
            joins.append(Join(prev, nxt, True, "heading CONTINUOUS"))
            continue
        if nxt.ep != prev.ep:
            text = notes.get(nxt.ep, "")
            match = CONTINUOUS_NOTES.search(text)
            if match and not TIME_SKIP.search(text):
                joins.append(Join(prev, nxt, True, f'notes say "{match.group(0)}"', text))
                continue
            joins.append(Join(prev, nxt, False, "", text))
            continue
        joins.append(Join(prev, nxt, False, ""))
    return joins


def _label(scene: Scene) -> str:
    return f"E{scene.ep:02d} S{scene.num:02d}"


def find_joins(show: Show) -> list[Finding]:
    joins = build_joins(show)
    beats_by_scene: dict[tuple[int, int], list] = {}
    for episode in show.episodes:
        for beat in episode.beats:
            beats_by_scene.setdefault((beat.ep, beat.scene), []).append(beat)

    # Effective time of day: a scene's own token, else the class of the scene it continues.
    effective: dict[tuple[int, int], str | None] = {}
    # Effective building type: heading, else the last one its descriptions name, else inherited.
    building: dict[tuple[int, int], str | None] = {}
    continuous_into: dict[tuple[int, int], Join] = {}
    for join in joins:
        if join.continuous:
            continuous_into[(join.next.ep, join.next.num)] = join

    for scene in scene_order(show):
        key = (scene.ep, scene.num)
        join = continuous_into.get(key)
        own = time_class(scene.time)
        inherited = effective.get((join.prev.ep, join.prev.num)) if join else None
        effective[key] = own or inherited
        described = None
        for beat in beats_by_scene.get(key, []):
            if beat.kind in ("shot", "action"):
                for word in TYPE_WORD.findall(beat.text):
                    described = word.lower()
        inherited_type = building.get((join.prev.ep, join.prev.num)) if join else None
        building[key] = heading_type(scene) or described or inherited_type

    findings: list[Finding] = []
    flips = []
    for join in joins:
        if not join.continuous:
            continue
        a = effective.get((join.prev.ep, join.prev.num))
        b = time_class(join.next.time) or effective.get((join.next.ep, join.next.num))
        if {a, b} == {"day", "night"}:
            flips.append(join)

    flipped_into = {(j.next.ep, j.next.num) for j in flips}
    flipped_from = {(j.prev.ep, j.prev.num) for j in flips}
    for join in flips:
        prev_key, next_key = (join.prev.ep, join.prev.num), (join.next.ep, join.next.num)
        odd = []
        if prev_key in flipped_into:
            odd.append(f"{_label(join.prev)} also differs from the scene before it")
        if next_key in flipped_from:
            odd.append(f"{_label(join.next)} also differs from the scene after it")
        # Which heading to change: in A -> B -> C with A agreeing with C, B is the odd one out;
        # otherwise the later scene follows the earlier one.
        if prev_key in flipped_into:
            target, wanted = join.prev, time_class(join.next.time) or effective.get(next_key)
        else:
            target, wanted = join.next, effective.get(prev_key)
        edit = None
        if time_class(target.time) in ("day", "night") and wanted in ("day", "night"):
            edit = Edit(at=f"E{target.ep:02d}.S{target.num:02d}.time", old=target.time, new=wanted.upper())
        suggestion = (
            f"{join.why}, but the time of day flips. One of these headings is wrong."
            + (" " + "; ".join(odd) + "." if odd else "")
            + (f" Approving sets {_label(target)} to {edit.new}." if edit else "")
        )
        findings.append(
            Finding(
                detector="joins",
                severity="visible",
                source="rule",
                confidence="high",
                ep=join.next.ep,
                scene=join.next.num,
                start=format_timestamp(join.next.start),
                where=f"{_label(join.prev)} -> {_label(join.next)}",
                found=(
                    f"{effective.get(prev_key, '').upper()} ({join.prev.time or 'no time'}) -> "
                    f"{(time_class(join.next.time) or effective.get(next_key) or '').upper()} "
                    f"({join.next.time or 'no time'})"
                ),
                suggestion=suggestion,
                context=f"{join.prev.heading} | {join.next.heading}"[:200],
                group="time",
                edit=edit,
            )
        )

    for join in joins:
        if not join.continuous:
            continue
        before = building.get((join.prev.ep, join.prev.num))
        after = heading_type(join.next)
        if not before or not after or before == after:
            continue
        if TYPE_FAMILY[before] != TYPE_FAMILY[after]:
            continue
        findings.append(
            Finding(
                detector="joins",
                severity="visible",
                source="check",
                ep=join.next.ep,
                scene=join.next.num,
                start=format_timestamp(join.next.start),
                where=f"{_label(join.prev)} -> {_label(join.next)}",
                found=f"{before} -> {after}",
                suggestion=(
                    f"{join.why}: the same moment is a {before} before the cut and a {after} after it. "
                    "If it is one building, use one name."
                ),
                context=f"{join.prev.heading} | {join.next.heading}"[:200],
                group="building",
            )
        )
    return findings
