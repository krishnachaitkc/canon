"""Files a person fills in. Code writes the rows and leaves every judgment column empty (R1),
and never overwrites a file that already holds someone's labels."""

import csv
import re
from pathlib import Path

from .findings import episode_ranges

LABEL_COLUMNS = ["id", "detector", "severity", "source", "confidence", "episode", "scene", "start",
                 "where", "found", "suggestion", "correct", "note"]
KEY_COLUMNS = ["id", "category", "confidence", "episodes", "where", "what", "key_terms", "targeted_by",
               "confirmed", "note"]
HUMAN_COLUMNS = {"correct", "note", "confirmed"}

# Which part of Canon is meant to catch each section of the evidence file (or one item in it).
# D1 names, D2 speaker labels, D3 scene joins, D4 spoken spelling, format, v2 = the optional AI read.
TARGETS = {
    "1": "D1",
    "2": "v2", "2.1": "D4",          # only a non-word is a spelling finding; real-word slips need a reader
    "3": "v2",
    "4": "D2",
    "5": "D3",
    "6": "D3", "6.3": "v2",          # 6.3 is about dialogue and naming, not one continuous join
    "7": "v2",
    "8": "format",
}

SECTION = re.compile(r"^##\s+(\d+)\.\s+(.+?)\s*$")
BOLD = re.compile(r"\*\*(.+?)\*\*")
LABEL = re.compile(r"\b[A-Z][A-Z']+(?:\s+[A-Z][A-Z']+)*\b")
EPISODE = re.compile(r"\bE(\d+)(?:\s*[–-]\s*E?(\d+))?")
CONFIDENCE = re.compile(r"\b(Certain|Very likely|Likely)\b")


class AuditError(ValueError):
    pass


def _has_labels(path: Path) -> bool:
    if not path.exists():
        return False
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if any((row.get(c) or "").strip() for c in HUMAN_COLUMNS):
                return True
    return False


def _write(path: Path, columns: list[str], rows: list[dict], force: bool) -> None:
    if _has_labels(path) and not force:
        raise AuditError(f"{path} already holds labels; it was not overwritten (use --force to replace it)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: ("" if c in HUMAN_COLUMNS else row.get(c, "")) for c in columns})


def audit_template(run_dir: str, audit_dir: str = "audit", force: bool = False) -> Path:
    """audit/labels_<run>.csv: one row per finding, `correct` and `note` empty."""
    run = Path(run_dir)
    with open(run / "findings.csv", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    path = Path(audit_dir) / f"labels_{run.name}.csv"
    _write(path, LABEL_COLUMNS, rows, force)
    return path


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _episodes(text: str) -> str:
    nums = []
    for match in EPISODE.finditer(text):
        a = int(match.group(1))
        b = int(match.group(2)) if match.group(2) else a
        nums += list(range(min(a, b), max(a, b) + 1))
    return episode_ranges(nums)


def _plain(text: str) -> str:
    return BOLD.sub(r"\1", text).replace("`", "")


def _item(section: str, title: str, item_id: str, cells: dict, raw: str) -> dict:
    where = cells.get("Where", "") or cells.get("Join", "")
    skip = {"#", "Where", "Join", "Confidence"}
    what = " — ".join(_plain(v) for k, v in cells.items() if k not in skip and v)
    conf = cells.get("Confidence") or (CONFIDENCE.findall(raw) or [""])[-1]
    terms = [t for t in BOLD.findall(raw) if not t.endswith(":")]
    if section == "4":
        terms += [m for m in LABEL.findall(cells.get("Labels", raw)) if m not in terms]
    return {
        "id": item_id,
        "category": title,
        "confidence": _plain(conf),
        "episodes": _episodes(raw),
        "where": _plain(where),
        "what": what,
        "key_terms": "; ".join(_plain(t) for t in terms),
        "targeted_by": TARGETS.get(item_id) or TARGETS[section],
    }


def parse_evidence(text: str) -> list[dict]:
    """Every item in the numbered sections of the evidence file that TARGETS covers.

    Table rows are items. A paragraph is an item when its section has no table (Format), or
    when it opens with a bold label ("**One label, two people:**"); other paragraphs are notes.
    """
    sections: list[dict] = []
    current = None
    for line in text.splitlines():
        match = SECTION.match(line)
        if match or line.startswith("## "):
            current = {"num": match.group(1), "title": match.group(2), "header": None, "rows": [], "paras": []} if match else None
            if current and current["num"] in TARGETS:
                sections.append(current)
            else:
                current = None
            continue
        stripped = line.strip()
        if current is None or not stripped or stripped == "---":
            continue
        if stripped.startswith("|"):
            cells = _cells(stripped)
            if current["header"] is None:
                current["header"] = cells
            elif not all(set(c) <= set("-: ") for c in cells):
                current["rows"].append((dict(zip(current["header"], cells)), stripped))
        else:
            current["paras"].append(stripped)

    items = []
    for s in sections:
        count = 0
        for cells, raw in s["rows"]:
            count += 1
            items.append(_item(s["num"], s["title"], cells.get("#") or f"{s['num']}.{count}", cells, raw))
        for para in s["paras"]:
            if s["header"] is None or para.startswith("**"):
                count += 1
                items.append(_item(s["num"], s["title"], f"{s['num']}.{count}", {"What": para}, para))
    return items


def answer_key(evidence: str = "plan/EVIDENCE.md", out: str = "audit/answer_key.csv", force: bool = False) -> tuple[Path, list[dict]]:
    """audit/answer_key.csv from the evidence file, with `confirmed` empty for the user."""
    source = Path(evidence)
    if not source.exists():
        raise AuditError(f"{source} not found")
    items = parse_evidence(source.read_text(encoding="utf-8"))
    if not items:
        raise AuditError(f"no items found in {source}")
    path = Path(out)
    _write(path, KEY_COLUMNS, items, force)
    return path, items
