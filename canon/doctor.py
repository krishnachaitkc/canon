"""Parse a show and print the facts the rest of the pipeline relies on."""

import re
from collections import Counter

from .parse import _episode_number, parse_show, read_input

# Header labels whose spelling should be the same in every episode.
FORMAT_LABELS = {
    "Scene heading": re.compile(r"^[ \t]*(scene[ \t]+heading)[ \t]*:", re.IGNORECASE | re.MULTILINE),
    "Scene start": re.compile(r"^[ \t]*(scene[ \t]+start)[ \t]*:", re.IGNORECASE | re.MULTILINE),
    "Scene end": re.compile(r"^[ \t]*(scene[ \t]+end)[ \t]*:", re.IGNORECASE | re.MULTILINE),
    "Actions and dialogue": re.compile(
        r"^[ \t]*(actions[ \t]+and[ \t]+dialogue)[ \t]*:?[ \t]*$", re.IGNORECASE | re.MULTILINE
    ),
    "EPISODE NUMBER": re.compile(r"^[ \t]*(episode[ \t]+number)[ \t]*:", re.IGNORECASE | re.MULTILINE),
    "EPISODE LOGLINE": re.compile(r"^[ \t]*(episode[ \t]+logline)[ \t]*:", re.IGNORECASE | re.MULTILINE),
    "CONTINUITY NOTES": re.compile(r"^[ \t]*(continuity[ \t]+notes)[ \t]*:", re.IGNORECASE | re.MULTILINE),
}


def format_report(input_source: str) -> list[dict]:
    """Find header labels an episode spells differently from the majority.

    Returns one dict per (label, episode) that differs: label, ep, found, majority, majority_count.
    """
    per_label: dict[str, dict[int, str]] = {label: {} for label in FORMAT_LABELS}
    for name, content in read_input(input_source):
        ep = _episode_number(content, name)
        for label, pattern in FORMAT_LABELS.items():
            spellings = sorted({" ".join(m.group(1).split()) for m in pattern.finditer(content)})
            if spellings:
                per_label[label][ep] = " / ".join(spellings)
    out = []
    for label in FORMAT_LABELS:
        used = per_label[label]
        if not used:
            continue
        counts = Counter(used.values())
        majority, majority_count = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        for ep in sorted(used):
            if used[ep] != majority:
                out.append(
                    {
                        "label": label,
                        "ep": ep,
                        "found": used[ep],
                        "majority": majority,
                        "majority_count": majority_count,
                    }
                )
    return out


def format_findings(input_source: str) -> list:
    """Every minority header spelling as a finding (severity notes, source rule)."""
    from .models import Finding

    return [
        Finding(
            detector="format",
            severity="notes",
            source="rule",
            confidence="high",
            ep=item["ep"],
            scene=0,
            start="00:00.00",
            where=item["label"],
            found=item["found"],
            suggestion=item["majority"],
            context=f"{item['majority_count']} episodes write '{item['majority']}'",
            group=f"format:{item['label']}",
        )
        for item in format_report(input_source)
    ]


def doctor(input_source: str, title: str | None = None) -> int:
    show = parse_show(input_source, title=title)
    kinds = Counter(beat.kind for episode in show.episodes for beat in episode.beats)
    total_beats = sum(len(episode.beats) for episode in show.episodes)
    runtime_min = sum(episode.runtime for episode in show.episodes) / 60.0
    scenes = sum(len(episode.scenes) for episode in show.episodes)
    labels = sorted(
        {beat.speaker for episode in show.episodes for beat in episode.beats if beat.kind == "line"}
    )
    coverage = (
        (show.script_lines - show.unconsumed_lines) / show.script_lines * 100.0
        if show.script_lines
        else 100.0
    )
    problems = format_report(input_source)

    print("Canon doctor")
    print(f"input: {input_source}")
    print(f"title: {show.title}")
    print(f"episodes: {len(show.episodes)}")
    print(
        f"beats: {total_beats} "
        f"(shot={kinds['shot']} line={kinds['line']} action={kinds['action']} sfx={kinds['sfx']})"
    )
    print(f"runtime: {runtime_min:.1f} min")
    print(f"scenes: {scenes}")
    print(f"speaker labels: {len(labels)}")
    print(f"anomalies: {len(show.anomalies)}")
    print(f"unconsumed lines: {show.unconsumed_lines}")
    print(f"coverage: {coverage:.1f}%")
    print(f"format differences: {len(problems)}")
    for item in problems:
        print(
            f"  E{item['ep']:02d} writes '{item['found']}' for {item['label']} "
            f"(majority '{item['majority']}', {item['majority_count']} episodes)"
        )
    return 1 if coverage < 95.0 else 0
