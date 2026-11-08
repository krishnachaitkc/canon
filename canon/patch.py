"""Apply the findings a person approved to a copy of the scripts. The input is never touched.

approvals.json: {"F0003": true, "F0007": false, ..., "canonical": {"N003.1": "Ravenscar"}}.
An approved finding with an edit is applied; a finding without one is report only (locations,
checks, format). A canonical override decides a whole group: every other spelling or label in
it becomes the chosen one, except rows set to false.
"""

import difflib
import json
import os
import re
import shutil
import zipfile
from pathlib import Path

from .edits import match_case, word_spans
from .models import Edit, Finding, NameGroup, SpeakerGroup
from .parse import BEAT, HEADING_SEP, SCENE, SFX_TAGS, _episode_number, natural_key, sec

LOGLINE = re.compile(r"EPISODE LOGLINE:\s*(.+)", re.IGNORECASE)
NOTES = re.compile(r"CONTINUITY NOTES:\s*(.+)", re.IGNORECASE)
MARKER = re.compile(r"---\s*SCRIPT START\s*---", re.IGNORECASE)
EPISODE_OF = re.compile(r"^E(\d+)\.")


class PatchError(ValueError):
    pass


def read_raw(input_source: str) -> list[tuple[str, str]]:
    """(name, text) pairs like parse.read_input, but with line endings exactly as written."""
    if os.path.isdir(input_source):
        names = sorted(
            (n for n in os.listdir(input_source)
             if n.lower().endswith((".txt", ".md")) and os.path.isfile(os.path.join(input_source, n))),
            key=natural_key,
        )
        out = []
        for name in names:
            with open(os.path.join(input_source, name), encoding="utf-8", errors="replace", newline="") as fh:
                out.append((name, fh.read()))
        return out
    if os.path.isfile(input_source) and zipfile.is_zipfile(input_source):
        with zipfile.ZipFile(input_source) as zf:
            names = sorted(
                (n for n in zf.namelist()
                 if n.lower().endswith((".txt", ".md")) and not n.endswith("/") and "__MACOSX" not in n),
                key=natural_key,
            )
            return [(n, zf.read(n).decode("utf-8", errors="replace")) for n in names]
    raise PatchError(f"Input must be a folder or a zip of .txt/.md files: {input_source}")


def _trimmed(start: int, end: int, text: str) -> tuple[int, int]:
    lead = len(text) - len(text.lstrip())
    trail = len(text) - len(text.rstrip())
    return start + lead, end - trail


def locate(content: str, filename: str = "") -> tuple[int, dict[str, tuple[int, int]]]:
    """Map every location ID Canon uses to its character span in the raw file.

    Mirrors parse_episode_text: beats are numbered by start time, then file order.
    """
    ep = _episode_number(content, filename)
    spans: dict[str, tuple[int, int]] = {}
    for name, pattern in (("logline", LOGLINE), ("notes", NOTES)):
        match = pattern.search(content)
        if match:
            spans[f"E{ep:02d}.{name}"] = _trimmed(*match.span(1), match.group(1))
    for match in SCENE.finditer(content):
        num = int(match.group(1))
        start, end = _trimmed(*match.span(2), match.group(2))
        spans[f"E{ep:02d}.S{num:02d}.heading"] = (start, end)
        heading = content[start:end]
        seps = list(HEADING_SEP.finditer(heading))
        if seps:
            spans[f"E{ep:02d}.S{num:02d}.time"] = _trimmed(start + seps[-1].end(), end, heading[seps[-1].end():])

    marker = MARKER.search(content)
    offset = marker.end() if marker else 0
    body = content[offset:]
    rows = []
    for order, match in enumerate(BEAT.finditer(body)):
        rows.append((sec(match.group(2)), order, match))
    rows.sort(key=lambda row: (row[0], row[1]))
    for index, (_, _, match) in enumerate(rows, 1):
        beat_id = f"E{ep:02d}.B{index:03d}"
        spans[beat_id] = _trimmed(offset + match.start(4), offset + match.end(4), match.group(4))
        tag = match.group(1)
        base = re.sub(r"\(.*$", "", tag.strip().rstrip(":").strip()).strip()
        upper = base.upper()
        if base and not upper.startswith(("CAMERA", "ACTION")) and upper not in SFX_TAGS:
            at = tag.find(base)
            spans[f"{beat_id}.label"] = (offset + match.start(1) + at, offset + match.start(1) + at + len(base))
    return ep, spans


def resolve(content: str, spans: dict[str, tuple[int, int]], edit: Edit) -> tuple[int, int, str]:
    """The exact (start, end, replacement) an edit means in this file."""
    if edit.at not in spans:
        raise PatchError(f"{edit.at}: location not found")
    start, end = spans[edit.at]
    segment = content[start:end]
    if edit.at.endswith(".time"):
        if " ".join(segment.split()).upper() != " ".join(edit.old.split()).upper():
            raise PatchError(f"{edit.at}: expected {edit.old!r}, found {segment!r}")
        return start, end, match_case(segment, edit.new)
    if edit.at.endswith(".label"):
        if segment != edit.old:
            raise PatchError(f"{edit.at}: expected label {edit.old!r}, found {segment!r}")
        return start, end, edit.new
    found = word_spans(segment, edit.old)
    if len(found) <= edit.nth:
        raise PatchError(f"{edit.at}: occurrence {edit.nth + 1} of {edit.old!r} not found")
    a, b = found[edit.nth]
    return start + a, start + b, match_case(segment[a:b], edit.new)


def apply_edits(content: str, spans: dict[str, tuple[int, int]], edits: list[Edit]) -> tuple[str, int, list[str]]:
    """Apply edits right to left. Identical edits count once; clashing ones are skipped."""
    planned: dict[tuple[int, int], str] = {}
    problems = []
    for edit in edits:
        try:
            start, end, text = resolve(content, spans, edit)
        except PatchError as exc:
            problems.append(str(exc))
            continue
        if (start, end) in planned:
            if planned[(start, end)] != text:
                problems.append(f"{edit.at}: two approved fixes disagree ({planned[(start, end)]!r} and {text!r})")
            continue
        if any(start < e and s < end for s, e in planned):
            problems.append(f"{edit.at}: overlaps another approved fix")
            continue
        planned[(start, end)] = text
    for (start, end), text in sorted(planned.items(), reverse=True):
        content = content[:start] + text + content[end:]
    return content, len(planned), problems


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_edits(run: Path, approvals: dict, findings: list[Finding]) -> tuple[list[Edit], dict]:
    """Which edits the approvals ask for, plus counts for the report."""
    from .findings import name_rows, speaker_rows

    approvals = dict(approvals)
    overrides = approvals.pop("canonical", None) or {}
    by_id = {f.id: f for f in findings}
    bad = [k for k, v in approvals.items() if not isinstance(v, bool)]
    if bad:
        raise PatchError(f"approvals must be true or false: {', '.join(sorted(bad))}")
    stats = {
        "approved": sum(1 for v in approvals.values() if v),
        "rejected": sum(1 for v in approvals.values() if v is False),
        "unknown_ids": sorted(k for k in approvals if k not in by_id),
        "report_only": 0,
        "no_replacement": 0,
        "overrides": len(overrides),
    }
    edits = []
    for fid, ok in approvals.items():
        f = by_id.get(fid)
        if not ok or f is None or f.group in overrides:
            continue
        if f.edit is None:
            stats["report_only"] += 1
        elif not f.edit.new or f.edit.new == f.edit.old:
            stats["no_replacement"] += 1
        else:
            edits.append(f.edit)

    if overrides:
        from .mentions import build_index
        from .parse import parse_show

        groups = _load_json(run / "groups.json")
        names = {g["id"]: NameGroup(**g) for g in groups["names"]}
        speakers = {g["id"]: SpeakerGroup(**g) for g in groups["speakers"]}
        manifest = _load_json(run / "manifest.json")
        show = parse_show(manifest["input"])
        index = build_index(show) if any(gid in names for gid in overrides) else None
        for gid, value in overrides.items():
            if gid in names:
                group = names[gid]
                if value not in group.spellings:
                    raise PatchError(f"{gid}: {value!r} is not one of its spellings")
                rows = name_rows([group.model_copy(update={"canonical": value})], index, show)
            elif gid in speakers:
                group = speakers[gid]
                if value not in group.labels:
                    raise PatchError(f"{gid}: {value!r} is not one of its labels")
                rows = speaker_rows([group.model_copy(update={"canonical": value})], show)
            else:
                raise PatchError(f"canonical override for unknown group {gid}")
            refused = {
                (f.edit.at, f.edit.nth, f.edit.old)
                for f in findings
                if f.group == gid and f.edit and approvals.get(f.id) is False
            }
            edits += [r.edit for r in rows if r.edit and (r.edit.at, r.edit.nth, r.edit.old) not in refused]
    return edits, stats


def patch(run_dir: str, approvals_path: str | None = None) -> dict:
    """Write <run>/patched/ (every episode, fixed where approved) and <run>/patch.diff."""
    from .pipeline import input_hash

    run = Path(run_dir)
    manifest = _load_json(run / "manifest.json")
    if not manifest.get("complete"):
        raise PatchError(f"{run} is not a complete run")
    source = manifest["input"]
    if input_hash(source) != manifest["input_sha256"]:
        raise PatchError("the input changed since this run; run Canon again before patching")
    findings = [Finding(**f) for f in _load_json(run / "findings.json")]
    approvals = _load_json(Path(approvals_path) if approvals_path else run / "approvals.json")
    edits, stats = collect_edits(run, approvals, findings)

    files = read_raw(source)
    by_episode: dict[int, list[Edit]] = {}
    for edit in edits:
        match = EPISODE_OF.match(edit.at)
        if not match:
            raise PatchError(f"bad location {edit.at}")
        by_episode.setdefault(int(match.group(1)), []).append(edit)

    out_dir = run / "patched"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    diff, problems, applied, changed = [], [], 0, 0
    seen_episodes = set()
    for name, content in files:
        ep, spans = locate(content, name)
        seen_episodes.add(ep)
        new, count, issues = apply_edits(content, spans, by_episode.get(ep, []))
        applied += count
        problems += issues
        target = out_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="") as fh:
            fh.write(new)
        if new != content:
            changed += 1
            diff += difflib.unified_diff(
                content.splitlines(keepends=True), new.splitlines(keepends=True), f"a/{name}", f"b/{name}"
            )
    for ep in sorted(set(by_episode) - seen_episodes):
        problems.append(f"E{ep:02d}: no such episode in the input")
    text = "".join(line if line.endswith(("\n", "\r")) else line + "\n" for line in diff)
    (run / "patch.diff").write_text(text, encoding="utf-8", newline="")
    stats.update({"edits": len(edits), "applied": applied, "files_changed": changed, "problems": problems})
    (run / "patch.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return stats
