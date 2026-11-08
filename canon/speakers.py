"""D2: speaker-label statistics. Code only; the model judges later, and code vetoes."""

import re
from collections import Counter, defaultdict

from .mentions import TITLES, MentionIndex, label_words
from .models import Show, SpeakerStat

# Role words that say what someone is, not who. They never identify a character.
GENERIC = {
    "the", "of", "in", "and", "man", "woman", "guard", "guards", "thug", "official", "villager",
    "voice", "fighter", "scout", "sentry", "driver", "cart", "constable", "police", "royal",
    "crown", "elder", "old", "young", "tall", "bald", "third", "noble", "nobleman",
    "noblewoman", "aristocratic", "peasant", "loyalist", "informant", "butler", "footman",
    "housekeeper", "maid", "judge", "priest", "high", "off", "screen", "townsman", "with",
    "cap", "conspirator",
}
# Words that make a one-word label a role rather than a person ("MATRIARCH", "THE STRANGER").
ROLE_WORDS = GENERIC | {"matriarch", "stranger", "groom", "figure", "duchess", "dowager"}

SAMPLE_CHARS = 100
CAMERA_TOP = 3
NAMED_IN_MAX = 2


def core_words(label: str) -> list[str]:
    """The words that identify who a label is.

    For a possessive label ("EDMUND'S MOTHER") only the words after the possessive count:
    the label names the mother, not Edmund. Titles and generic roles are dropped.
    """
    text = label
    if re.search(r"['’]S\b", label, re.IGNORECASE):
        text = re.split(r"['’]S\b", label, flags=re.IGNORECASE)[-1]
    words = label_words(text)
    return sorted({w for w in words if w not in TITLES and w not in GENERIC and len(w) >= 3})


def personal_names(labels: list[str]) -> set[str]:
    """Names of people, taken from labels that are one name plus titles: MARGARET, LORD EDMUND.

    A possessive label (EDMUND'S MOTHER) names a relation, not that person, so it is skipped.
    """
    out = set()
    for label in labels:
        if re.search(r"['’]S\b", label, re.IGNORECASE):
            continue
        words = [w for w in label_words(label) if w not in TITLES and w != "the"]
        if len(words) == 1 and words[0] not in ROLE_WORDS and len(words[0]) >= 3:
            out.add(words[0])
    return out


def label_phrase(label: str) -> str:
    """How descriptions write a label: "THE HOODED FIGURE" -> "hooded figure"."""
    words = label.lower().split()
    if words and words[0] == "the":
        words = words[1:]
    return " ".join(words)


def named_in(label: str, people: set[str], texts: list[str]) -> list[str]:
    """Up to NAMED_IN_MAX sentences that name a descriptive label together with a person.

    This is where a script says who a role is ("the hooded figure lowers her hood: it is
    Margaret", "the stranger Edmund waits by the gate"). Sentences where the name comes right
    after the label come first. Cut by code; the model only reads them.
    """
    phrase = label_phrase(label)
    # One-word roles (MAID, GUARD) match too much: "disguised as a scullery maid, Margaret..."
    # would point MAID at the wrong person. Only descriptive labels of two or more words.
    if len(phrase.split()) < 2 or phrase in people:
        return []
    pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
    scored = []
    for text in texts:
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            hit = pattern.search(sentence)
            if not hit:
                continue
            best = None
            for word in re.finditer(r"[A-Z][a-z]+", sentence):
                if word.group(0).casefold() in people and not (hit.start() <= word.start() < hit.end()):
                    gap = word.start() - hit.end()
                    rank = gap if 0 <= gap <= 40 else 1000 + abs(gap)
                    best = rank if best is None else min(best, rank)
            if best is not None:
                scored.append((best, len(sentence), sentence.strip()))
    out = []
    for _, _, sentence in sorted(scored):
        cut = sentence if len(sentence) <= 200 else sentence[:199].rstrip() + "…"
        if cut not in out:
            out.append(cut)
        if len(out) == NAMED_IN_MAX:
            break
    return out


def _cut(text: str) -> str:
    text = text.strip()
    return text if len(text) <= SAMPLE_CHARS else text[: SAMPLE_CHARS - 1].rstrip() + "…"


def speaker_stats(show: Show, index: MentionIndex) -> dict[str, SpeakerStat]:
    lines_by_label: dict[str, list] = defaultdict(list)
    speakers_in_scene: dict[tuple[int, int], set[str]] = defaultdict(set)
    visuals_in_scene: dict[tuple[int, int], list] = defaultdict(list)
    for episode in show.episodes:
        for beat in episode.beats:
            if beat.kind == "line" and beat.speaker:
                lines_by_label[beat.speaker].append(beat)
                speakers_in_scene[(beat.ep, beat.scene)].add(beat.speaker)
            elif beat.kind in ("shot", "action"):
                visuals_in_scene[(beat.ep, beat.scene)].append(beat)

    # Personal names in each shot or action beat. Camera directions also refer to people by
    # their roles ("the Hooded Figure"); only real names say who someone is.
    people = personal_names(sorted(lines_by_label))
    names_in_beat: dict[str, list[str]] = defaultdict(list)
    for mention in index.mentions:
        if mention.kind in ("shot", "action") and mention.key in people:
            names_in_beat[mention.beat_id].append(index.display[mention.key])

    descriptions = [
        beat.text for episode in show.episodes for beat in episode.beats if beat.kind in ("shot", "action")
    ] + [text for episode in show.episodes for text in (episode.logline, episode.continuity) if text]

    co_speak: dict[str, Counter] = defaultdict(Counter)
    for labels in speakers_in_scene.values():
        for a in labels:
            for b in labels:
                if a != b:
                    co_speak[a][b] += 1

    stats = {}
    for label in sorted(lines_by_label):
        beats = lines_by_label[label]
        scenes = sorted({(b.ep, b.scene) for b in beats})
        first = beats[0].text
        longest = max(beats, key=lambda b: (len(b.text), -b.ep, -b.start)).text
        samples = [_cut(first)] if first == longest else [_cut(first), _cut(longest)]

        # Who is on screen while this label speaks: names in shots that overlap its lines.
        own = set(label_words(label))
        on_screen: Counter = Counter()
        for line in beats:
            for visual in visuals_in_scene[(line.ep, line.scene)]:
                if visual.start < line.end and line.start < visual.end:
                    on_screen.update(
                        name for name in names_in_beat.get(visual.id, []) if name.casefold() not in own
                    )
        stats[label] = SpeakerStat(
            label=label,
            lines=len(beats),
            episodes=sorted({b.ep for b in beats}),
            scenes=[f"E{ep:02d}S{num:02d}" for ep, num in scenes],
            core=core_words(label),
            samples=samples,
            camera_names=sorted(on_screen.items(), key=lambda kv: (-kv[1], kv[0]))[:CAMERA_TOP],
            named_in=named_in(label, people, descriptions),
            co_speakers=dict(sorted(co_speak[label].items())),
        )
    return stats


def core_pairs(stats: dict[str, SpeakerStat]) -> list[tuple[str, str, str]]:
    """Label pairs that share an identifying word: LORD EDMUND and EDMUND."""
    labels = sorted(stats)
    pairs = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            shared = sorted(set(stats[a].core) & set(stats[b].core))
            if shared:
                pairs.append((a, b, f"shared word {shared[0]}"))
    return pairs
