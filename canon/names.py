"""D1: group names that look like spellings of one thing. Code only; the model judges later."""

import logging
from collections import Counter, defaultdict

from rapidfuzz.distance import DamerauLevenshtein, JaroWinkler

from .mentions import MentionIndex
from .models import Candidate

LOG = logging.getLogger(__name__)

MIN_LEN = 5
MAX_KEYS = 12
JW_PREFIX_BAR = 0.85     # string link: common prefix >= 4 and Jaro-Winkler at least this
JW_CONTEXT_BAR = 0.75    # context link: shared prefix of 3, or Jaro-Winkler at least this
JW_STEP = 0.03


def common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def string_rule(a: str, b: str, jw_prefix_bar: float) -> str | None:
    shortest = min(len(a), len(b))
    distance = DamerauLevenshtein.distance(a, b)
    if shortest >= 6 and distance <= 2:
        return f"edit distance {distance}"
    if shortest == 5 and distance == 1:
        return "edit distance 1"
    if common_prefix(a, b) >= 4 and JaroWinkler.similarity(a, b) >= jw_prefix_bar:
        return f"shared prefix, Jaro-Winkler {JaroWinkler.similarity(a, b):.2f}"
    return None


def context_rule(a: str, b: str, index: MentionIndex, jw_context_bar: float) -> str | None:
    shared_first = sorted(
        set(index.first_before.get(a, {})) & set(index.first_before.get(b, {}))
    )
    shared_head = sorted(set(index.head_after.get(a, {})) & set(index.head_after.get(b, {})))
    if not shared_first and not shared_head:
        return None
    if common_prefix(a, b) < 3 and JaroWinkler.similarity(a, b) < jw_context_bar:
        return None
    if shared_first:
        return f"same first name before ({shared_first[0]})"
    return f"same word after ({shared_head[0]})"


class _UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _links(index: MentionIndex, jw_prefix_bar: float, jw_context_bar: float) -> list[tuple[str, str, str]]:
    keys = sorted(key for key in index.names if len(key) >= MIN_LEN)
    by_letter: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_letter[key[0]].append(key)
    links = []
    for group in by_letter.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                rule = string_rule(a, b, jw_prefix_bar) or context_rule(a, b, index, jw_context_bar)
                if rule:
                    links.append((a, b, rule))
    return links


def build_candidates(index: MentionIndex) -> tuple[list[Candidate], list[str]]:
    """Return candidates and a log of any threshold raises."""
    log: list[str] = []
    jw_prefix_bar, jw_context_bar = JW_PREFIX_BAR, JW_CONTEXT_BAR
    while True:
        links = _links(index, jw_prefix_bar, jw_context_bar)
        finder = _UnionFind(sorted({k for a, b, _ in links for k in (a, b)}))
        for a, b, _ in links:
            finder.union(a, b)
        components: dict[str, list[str]] = defaultdict(list)
        for key in finder.parent:
            components[finder.find(key)].append(key)
        biggest = max((len(keys) for keys in components.values()), default=0)
        if biggest <= MAX_KEYS:
            break
        jw_prefix_bar = round(jw_prefix_bar + JW_STEP, 2)
        jw_context_bar = round(jw_context_bar + JW_STEP, 2)
        message = (
            f"a candidate had {biggest} keys; Jaro-Winkler bars raised to "
            f"{jw_prefix_bar} (prefix) and {jw_context_bar} (context)"
        )
        LOG.warning(message)
        log.append(message)

    by_key = index.by_key()
    candidates = []
    for keys in components.values():
        keys = sorted(keys)
        spellings: Counter = Counter()
        spoken: Counter = Counter()
        for key in keys:
            name = index.display[key]
            spellings[name] += len(by_key[key])
            spoken[name] += sum(1 for m in by_key[key] if m.kind == "spoken")
        member_links = sorted((a, b, rule) for a, b, rule in links if a in keys)
        candidates.append(
            Candidate(
                id="",
                keys=keys,
                spellings=dict(sorted(spellings.items(), key=lambda kv: (-kv[1], kv[0]))),
                spoken=dict(sorted(spoken.items(), key=lambda kv: (-kv[1], kv[0]))),
                links=member_links,
            )
        )
    candidates.sort(key=lambda c: (-sum(c.spellings.values()), c.keys))
    for number, candidate in enumerate(candidates, 1):
        candidate.id = f"N{number:03d}"
    return candidates, log
