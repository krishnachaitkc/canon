"""Find every proper-noun mention in a show, with its exact location.

Code only. A word counts as a name if the show capitalizes it mid-sentence somewhere, or
never writes it in lowercase. A name that is also a speaker-label word is a first name.
Scene headings are all caps, so their place
words are title-cased and kept only when they are names elsewhere in the show.
"""

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .models import Mention, Show
from .parse import split_heading

TOKEN = re.compile(r"\b[A-Z][a-z]{2,}(?:['’]s)?\b")
LOWER_WORD = re.compile(r"\b[a-z][a-z]+(?:['’]s)?\b")
NEXT_WORD = re.compile(r"^[ \t]+([A-Za-z]+)")
SENTENCE_BREAK = set('.!?:;"“”—…(')

TITLES = {
    "lord", "lady", "sir", "duke", "duchess", "dowager", "earl", "count", "countess", "baron",
    "baroness", "king", "queen", "prince", "princess", "miss", "mrs", "mr", "master", "father",
    "mother", "sister", "brother", "captain", "doctor", "madam", "grace", "majesty", "highness",
}

# Words a show capitalizes for reasons other than being a name.
STOPWORDS = {
    "the", "she", "he", "his", "her", "hers", "they", "them", "their", "this", "that", "then",
    "there", "these", "those", "what", "when", "where", "who", "whom", "why", "how", "and",
    "but", "not", "now", "our", "your", "you", "yes", "one", "its", "all", "every", "each",
    "some", "any", "here", "after", "before", "behind", "beyond", "inside", "outside", "over",
    "under", "with", "without", "from", "into", "upon", "only", "just", "even", "still",
    "perhaps", "please", "enough", "well", "good", "wait", "look", "come", "stop", "let",
    "tell", "take", "give", "make", "get", "see", "know", "think", "say", "said", "will",
    "would", "could", "should", "must", "may", "might", "can", "did", "does", "was", "were",
    "been", "being", "have", "has", "had", "for", "too", "very", "much", "more", "most",
    "many", "few", "never", "always", "again", "also", "back", "down", "off", "out", "away",
    "once", "first", "last", "next", "new", "old", "close", "wide", "medium", "extreme",
    "tight", "low", "high", "slow", "quick", "full", "over-the-shoulder", "ext", "int",
}

# Head nouns: the word after a place name. Used as context, never as a name.
HEAD_NOUNS = {
    "manor", "hall", "estate", "house", "abbey", "castle", "park", "court", "village", "street",
    "cathedral", "chapel",
}

# Building and room words that appear in headings. Never names.
SET_WORDS = HEAD_NOUNS | {
    "church", "cottage", "cabin", "barn", "palace", "tavern", "inn", "tower", "dungeon", "cells",
    "cell", "corridor", "hallway", "study", "library", "nave", "aisle", "altar", "balcony",
    "bedchamber", "chamber", "chambers", "room", "drawing", "parlor", "parlour", "kitchen",
    "cellar", "wing", "stairwell", "garden", "gardens", "grounds", "courtyard", "gates", "gate",
    "entrance", "steps", "vestibule", "road", "path", "lane", "square", "woods", "forest",
    "ruins", "yard", "office", "courtroom",
}

NOT_NAMES = TITLES | STOPWORDS | SET_WORDS

KIND = {"line": "spoken", "shot": "shot", "action": "action", "sfx": "sfx"}


def clean_surface(surface: str) -> str:
    """Drop a possessive: "Ravenscar's" -> "Ravenscar", "EDMUND'S" -> "EDMUND"."""
    if surface[-2:].lower() in ("'s", "’s"):
        return surface[:-2]
    return surface.rstrip("'’")


def raw_key(surface: str) -> str:
    return unicodedata.normalize("NFKC", clean_surface(surface)).casefold()


def title_case(text: str) -> str:
    """Title-case a heading so "EDMUND'S COTTAGE" becomes "Edmund's Cottage"."""
    return re.sub(
        r"[A-Za-z]+(?:['’][A-Za-z]+)?",
        lambda m: m.group(0)[0].upper() + m.group(0)[1:].lower(),
        text,
    )


def label_words(label: str) -> list[str]:
    """Words of a speaker label, lowercased, possessive owner words included."""
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", label)
    return [raw_key(w) for w in words]


@dataclass
class Text:
    """One piece of script text with where it came from."""

    text: str
    ep: int
    scene: int
    beat_id: str
    start: float
    end: float
    kind: str
    speaker: str | None = None
    heading: bool = False


@dataclass
class MentionIndex:
    mentions: list[Mention]
    names: set[str]                          # keys that count as names
    first_names: set[str]                    # names that are also speaker-label words
    display: dict[str, str]                  # key -> most common written form
    first_before: dict[str, Counter] = field(default_factory=dict)   # key -> first names right before it
    head_after: dict[str, Counter] = field(default_factory=dict)     # key -> head nouns right after it
    plural_map: dict[str, str] = field(default_factory=dict)         # "hartwells" -> "hartwell"

    def by_key(self) -> dict[str, list[Mention]]:
        out: dict[str, list[Mention]] = defaultdict(list)
        for mention in self.mentions:
            out[mention.key].append(mention)
        return out


def iter_texts(show: Show) -> list[Text]:
    texts: list[Text] = []
    for episode in show.episodes:
        ep = episode.num
        if episode.logline:
            texts.append(Text(episode.logline, ep, 0, f"E{ep:02d}.logline", 0.0, 0.0, "header"))
        if episode.continuity:
            texts.append(Text(episode.continuity, ep, 0, f"E{ep:02d}.notes", 0.0, 0.0, "header"))
        for scene in episode.scenes:
            _, place, _ = split_heading(scene.heading)
            texts.append(
                Text(
                    title_case(place),
                    ep,
                    scene.num,
                    f"E{ep:02d}.S{scene.num:02d}.heading",
                    scene.start,
                    scene.end,
                    "header",
                    heading=True,
                )
            )
        for beat in episode.beats:
            texts.append(
                Text(beat.text, ep, beat.scene, beat.id, beat.start, beat.end, KIND[beat.kind], beat.speaker)
            )
    return texts


def _is_sentence_start(text: str, pos: int) -> bool:
    before = text[:pos].rstrip()
    return not before or before[-1] in SENTENCE_BREAK


def build_index(show: Show) -> MentionIndex:
    texts = iter_texts(show)
    labels = sorted({b.speaker for e in show.episodes for b in e.beats if b.speaker})
    label_word_set = {w for label in labels for w in label_words(label)}

    # Pass 1: which keys does the show treat as names? Headings are not evidence.
    # A capitalized word is a name if the show capitalizes it mid-sentence somewhere,
    # or never writes it in lowercase at all. The second rule catches a one-off slip that
    # starts its sentence ("Margarett, ma'am."), while "The" and "Then" still fail both.
    mid_sentence: set[str] = set()
    capitalized: set[str] = set()
    lowercase: set[str] = set()
    raw_counts: Counter = Counter()
    for item in texts:
        if item.heading:
            continue
        for word in LOWER_WORD.findall(item.text):
            lowercase.add(raw_key(word))
        for match in TOKEN.finditer(item.text):
            key = raw_key(match.group(0))
            raw_counts[key] += 1
            capitalized.add(key)
            if not _is_sentence_start(item.text, match.start()):
                mid_sentence.add(key)
    # Speaker-label words are not evidence on their own: most are roles (GUARD, STRANGER).
    # Labels only mark which names are first names.
    names = {
        key
        for key in capitalized
        if (key in mid_sentence or key not in lowercase) and key not in NOT_NAMES and len(key) >= 3
    }

    # Plural families: "Hartwells" counts as "Hartwell" when the singular is used twice.
    plural_map = {
        key: key[:-1]
        for key in names
        if key.endswith("s") and key[:-1] in names and raw_counts[key[:-1]] >= 2
    }
    names = {plural_map.get(key, key) for key in names}
    first_names = {key for key in names if key in label_word_set}

    # Pass 2: every mention of a name, plus the words around it.
    mentions: list[Mention] = []
    surfaces: dict[str, Counter] = defaultdict(Counter)
    first_before: dict[str, Counter] = defaultdict(Counter)
    head_after: dict[str, Counter] = defaultdict(Counter)
    for item in texts:
        matches = list(TOKEN.finditer(item.text))
        for index, match in enumerate(matches):
            key = raw_key(match.group(0))
            key = plural_map.get(key, key)
            if key not in names:
                continue
            surface = match.group(0)
            mentions.append(
                Mention(
                    surface=surface,
                    key=key,
                    ep=item.ep,
                    scene=item.scene,
                    beat_id=item.beat_id,
                    start=item.start,
                    end=item.end,
                    kind=item.kind,
                    speaker=item.speaker,
                    char_start=match.start(),
                )
            )
            surfaces[key][clean_surface(surface)] += 1
            if index > 0:
                prev = matches[index - 1]
                between = item.text[prev.end(): match.start()]
                prev_key = raw_key(prev.group(0))
                if between.strip() == "" and prev_key in first_names:
                    first_before[key][prev_key] += 1
            after = NEXT_WORD.match(item.text[match.end():])
            if after and after.group(1).lower() in HEAD_NOUNS:
                head_after[key][after.group(1).lower()] += 1

    display = {key: sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for key, counter in surfaces.items()}
    return MentionIndex(
        mentions=mentions,
        names={key for key in names if key in surfaces},
        first_names=first_names,
        display=display,
        first_before=dict(first_before),
        head_after=dict(head_after),
        plural_map=plural_map,
    )
