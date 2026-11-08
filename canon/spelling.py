"""D4: misspelled words in dialogue. The voice reads every word as written.

Code finds words an English dictionary does not know; the model later judges each one as a
typo or a valid period, regional or coined word. Names are D1's job and are skipped.
"""

import re
from functools import lru_cache

from .mentions import MentionIndex, clean_surface, raw_key
from .edits import nth_before
from .models import Edit, Finding, Show
from .parse import format_timestamp

WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
INTERJECTION = re.compile(r"^(?:a+h+|o+h+|u+g+h+|h+m+|a+r+g+h+|s+h+|m+m+|e+h+|u+h+)$", re.IGNORECASE)
SENTENCE_BREAK = set('.!?:;"“”—…(')
CONTEXT_CHARS = 70


@lru_cache(maxsize=1)
def _checker():
    from spellchecker import SpellChecker

    return SpellChecker()


def _sentence_start(text: str, pos: int) -> bool:
    before = text[:pos].rstrip()
    return not before or before[-1] in SENTENCE_BREAK


def _context(text: str, start: int, end: int) -> str:
    half = (CONTEXT_CHARS - (end - start)) // 2
    left = max(0, start - half)
    right = min(len(text), end + half)
    snippet = text[left:right].strip()
    return (("…" if left > 0 else "") + snippet + ("…" if right < len(text) else ""))[: CONTEXT_CHARS + 2]


def spelling_candidates(show: Show, index: MentionIndex) -> list[Finding]:
    """Every unknown word in a spoken line, one finding per occurrence, source "check"."""
    checker = _checker()
    rows = []
    for episode in show.episodes:
        for beat in episode.beats:
            if beat.kind != "line":
                continue
            for match in WORD.finditer(beat.text):
                surface = match.group(0)
                word = clean_surface(surface)
                if "'" in word or "’" in word:
                    continue                      # contractions: don't, I'm
                if len(word) < 3 or INTERJECTION.match(word):
                    continue
                if word[0].isupper():
                    # A capital mid-sentence, or a known name, is D1's business.
                    if not _sentence_start(beat.text, match.start()) or raw_key(word) in index.names:
                        continue
                lower = word.lower()
                if lower in checker:
                    continue
                rows.append((beat, match, lower))

    findings = []
    for beat, match, lower in rows:
        suggestion = _checker().correction(lower) or ""
        suggestion = suggestion if suggestion != lower else ""
        written = clean_surface(match.group(0))
        findings.append(
            Finding(
                detector="spelling",
                severity="audible",
                source="check",
                ep=beat.ep,
                scene=beat.scene,
                start=format_timestamp(beat.start),
                where=beat.id,
                found=lower,
                suggestion=suggestion,
                context=_context(beat.text, match.start(), match.end()),
                group=f"word:{lower}",
                edit=Edit(at=beat.id, old=written, new=suggestion, nth=nth_before(beat.text, written, match.start())),
            )
        )
    return findings
