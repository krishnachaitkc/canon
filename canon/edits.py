"""Whole-word matching shared by the detectors (which count occurrences) and the patch (which
applies them), so both sides always agree on which occurrence a finding means."""

import re


def word_spans(text: str, word: str) -> list[tuple[int, int]]:
    """Spans of `word` as a whole word, ignoring case. A possessive after it ("Thorne's") still
    counts, so the patch keeps the possessive; a longer word ("Thornes", "Thornewood") does not."""
    pattern = re.compile(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])", re.IGNORECASE)
    return [m.span() for m in pattern.finditer(text)]


def nth_before(text: str, word: str, position: int) -> int:
    """How many whole-word occurrences of `word` start before `position`."""
    return sum(1 for start, _ in word_spans(text, word) if start < position)


def match_case(written: str, new: str) -> str:
    """Write `new` in the case style of `written`: RAVENSCOURT -> RAVENSCAR, Seizzure -> Seizure."""
    letters = [c for c in written if c.isalpha()]
    if len(letters) > 1 and all(c.isupper() for c in letters):
        return new.upper()
    if written[:1].isupper():
        return new[:1].upper() + new[1:]
    if written[:1].islower() and new[:1].isupper() and not any(c.isupper() for c in new[1:]):
        return new[:1].lower() + new[1:]
    return new
