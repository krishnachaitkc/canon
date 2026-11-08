"""Phase 6: score Canon and a whole-show model baseline.

R1: every metric is computed from findings plus either the confirmed answer key or
truth.json marked `seeded`. Nothing here writes a human label.
"""

import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

from . import llm
from .config import price_for
from .mentions import clean_surface
from .parse import read_input
from .pipeline import cost_usd, input_hash, run
from .seed import make_seed

PROMPTS = Path(__file__).resolve().parent / "prompts"
BASELINE_ROUTE = "gemini/gemini-3.5-flash-lite"
TARGET_DETECTOR = {"D1": "names", "D2": "speakers", "D3": "joins", "D4": "spelling", "format": "format"}
CONFIRMED_TRUE = {"yes", "y", "true", "1"}
EPISODE_RANGE = re.compile(r"(\d+)\s*[–-]\s*(\d+)")
EPISODE_ANY = re.compile(r"\b(\d+)\b")
EPISODE_MARK = re.compile(r"\bE(\d+)\b", re.IGNORECASE)
LOC = re.compile(r"(?:E(\d+)\s*)?\bS(\d+)\b", re.IGNORECASE)


class BaselineItem(BaseModel):
    kind: str = ""
    episode: str | int | None = None
    scene: str | int | None = None
    time: str = ""
    found: str = ""
    suggestion: str = ""
    detail: str = ""


class BaselineOut(BaseModel):
    findings: list[BaselineItem] = []


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def as_int(value) -> int:
    if isinstance(value, int):
        return value
    digits = re.search(r"\d+", str(value or ""))
    return int(digits.group(0)) if digits else 0


def split_terms(text: str) -> list[str]:
    return [norm(part) for part in re.split(r"[;]", str(text or "")) if norm(part)]


def row_episodes(row: dict) -> list[int]:
    episodes = set()
    column = str(row.get("episodes", ""))
    for a, b in EPISODE_RANGE.findall(column):
        episodes.update(range(int(a), int(b) + 1))
    without_ranges = EPISODE_RANGE.sub(" ", column)
    for number in EPISODE_ANY.findall(without_ranges):
        episodes.add(int(number))
    for number in EPISODE_MARK.findall(str(row.get("where", ""))):
        episodes.add(int(number))
    return sorted(episodes)


def row_locations(row: dict) -> set[tuple[int, int]]:
    """(episode, scene) pairs stated in the row's Where text, e.g. "E6 S1 ... and S2"."""
    text = str(row.get("where", ""))
    found, last_episode = set(), None
    for match in LOC.finditer(text):
        if match.group(1):
            last_episode = int(match.group(1))
        if last_episode is not None:
            found.add((last_episode, int(match.group(2))))
    return found


def finding_locations(finding: dict) -> set[tuple[int, int]]:
    found = row_locations(finding)
    ep, scene = as_int(finding.get("ep")), as_int(finding.get("scene"))
    if ep and scene:
        found.add((ep, scene))
    return found


LABEL_TERM = re.compile(r"^[A-Z][A-Z' ]*[A-Z]$")
TWO_PEOPLE = "one label, two people"


def has_word(text: str, term: str) -> bool:
    """`term` as whole words in `text` (both already normalized): "hart" is not in "hartwell"."""
    return bool(term) and re.search(r"(?<![a-z'])" + re.escape(term) + r"(?![a-z'])", text) is not None


def row_labels(row: dict) -> set[str]:
    """The speaker labels a D2 row names: its upper-case key terms."""
    return {norm(term) for term in str(row.get("key_terms", "")).split(";") if LABEL_TERM.match(term.strip())}


def row_people(row: dict) -> set[str]:
    """row_labels plus the character the row is about when it is written as a label
    ("Edmund (again?)" -> edmund), so a one-label row can still need two labels linked."""
    who = re.sub(r"\(.*?\)", "", str(row.get("what", "")).split(" — ")[0]).strip()
    return row_labels(row) | ({norm(who)} if who else set())


def finding_labels(finding: dict) -> set[str]:
    """The labels a speaker finding links: the list in `where` for a check or casting note,
    the suspect of a veto check, or the label and its canonical for a per-line row."""
    where = str(finding.get("where", ""))
    if " / " in where:
        return {norm(label) for label in where.split(" / ")}
    group = str(finding.get("group", ""))
    if group.startswith("veto:"):
        return {norm(group[len("veto:"):])}
    return {norm(finding.get(key, "")) for key in ("found", "suggestion")} - {""}


def answer_match(row: dict, finding: dict, detector: str) -> bool:
    episodes = row_episodes(row)
    if episodes and as_int(finding.get("ep")) not in episodes:
        return False
    if detector == "joins":
        stated = row_locations(row)
        if stated:
            return (as_int(finding.get("ep")), as_int(finding.get("scene"))) in stated
        return True
    if detector == "format":
        return True
    if detector == "speakers":
        # A speaker row is found only by a finding that links two of the row's own labels,
        # compared as whole labels: GUARD is not ROYAL GUARD. A one-label-two-people row
        # needs a finding that says that label is two people (a veto check).
        linked = finding_labels(finding)
        if TWO_PEOPLE in norm(row.get("what", "")):
            return str(finding.get("group", "")).startswith("veto:") and bool(row_labels(row) & linked)
        return len(row_people(row) & linked) >= 2
    text = norm(" ".join(str(finding.get(key, "")) for key in ("found", "suggestion", "where", "context")))
    for term in split_terms(row.get("key_terms", "")):
        if has_word(text, term):
            return True
    found = norm(clean_surface(str(finding.get("found", ""))))
    return has_word(norm(row.get("what", "")), found)


def answer_partial(row: dict, matches: list[dict], detector: str) -> bool:
    """Found, but some of the row's labels are in no matching finding (speakers only)."""
    if detector != "speakers" or not matches or TWO_PEOPLE in norm(row.get("what", "")):
        return False
    covered = set().union(*(finding_labels(f) for f in matches))
    return not row_labels(row) <= covered


def answer_exact(row: dict, finding: dict) -> bool | None:
    stated = row_locations(row)
    if stated:
        return (as_int(finding.get("ep")), as_int(finding.get("scene"))) in stated
    episodes = row_episodes(row)
    if episodes:
        return as_int(finding.get("ep")) in episodes
    return None


def score_answer_key(rows: list[dict], findings: list[dict]) -> dict:
    per = defaultdict(lambda: {"rows": 0, "found": 0, "partial": 0, "located": 0, "exact": 0})
    detail = []
    for row in rows:
        detector = TARGET_DETECTOR.get(str(row.get("targeted_by", "")))
        if not detector:
            continue
        per[detector]["rows"] += 1
        matches = [f for f in findings if f.get("detector") == detector and answer_match(row, f, detector)]
        found = bool(matches)
        partial = answer_partial(row, matches, detector)
        exact = None
        if found:
            per[detector]["found"] += 1
            per[detector]["partial"] += 1 if partial else 0
            known = [answer_exact(row, f) for f in matches]
            known = [value for value in known if value is not None]
            if known:
                exact = any(known)
                per[detector]["located"] += 1
                per[detector]["exact"] += 1 if exact else 0
        detail.append(
            {
                "id": row.get("id", ""),
                "detector": detector,
                "episodes": row.get("episodes", ""),
                "where": row.get("where", ""),
                "found": found,
                "partial": partial,
                "exact": exact,
                "matched": matches[0].get("where", "") if matches else "",
            }
        )
    total = {
        "rows": sum(value["rows"] for value in per.values()),
        "found": sum(value["found"] for value in per.values()),
        "partial": sum(value["partial"] for value in per.values()),
        "located": sum(value["located"] for value in per.values()),
        "exact": sum(value["exact"] for value in per.values()),
    }
    if total["rows"]:
        total["recall"] = round(total["found"] / total["rows"], 4)
    if total["located"]:
        total["location_exact"] = round(total["exact"] / total["located"], 4)
    return {"by_detector": dict(per), "total": total, "rows": detail}


def seed_match(item: dict, finding: dict) -> bool:
    if finding.get("detector") != item.get("detector"):
        return False
    text = norm(str(finding.get("found", "")) + " " + str(finding.get("suggestion", "")))
    if item.get("detector") == "joins":
        return any(
            as_int(finding.get("ep")) == int(occ["ep"])
            and as_int(finding.get("scene")) == int(occ["scene"])
            and norm(item.get("new", "")) in text
            for occ in item.get("occurrences", [])
        )
    new = norm(clean_surface(str(item.get("new", ""))))
    if item.get("detector") == "speakers":
        # An intentional group is reported once, at its first line, so the episode may be
        # earlier than the injected run. The injected label is unique, so the text is enough.
        return new in text
    if as_int(finding.get("ep")) not in item.get("episodes", []):
        return False
    if item.get("detector") == "spelling":
        return norm(clean_surface(str(finding.get("found", "")))) == new
    return new == norm(clean_surface(str(finding.get("found", "")))) or new in text


def seed_exact(item: dict, finding: dict) -> bool:
    locations = {(int(occ["ep"]), int(occ["scene"])) for occ in item.get("occurrences", [])}
    return (as_int(finding.get("ep")), as_int(finding.get("scene"))) in locations


def score_seed(truth: dict, findings: list[dict], baseline: list[dict]) -> dict:
    per = defaultdict(lambda: {"items": 0, "found": 0, "exact": 0, "new_false": 0})
    baseline_keys = {
        (f.get("detector"), as_int(f.get("ep")), as_int(f.get("scene")), norm(f.get("found", "")))
        for f in baseline
    }
    baseline_by_detector = defaultdict(list)
    for finding in baseline:
        baseline_by_detector[(finding.get("detector"), as_int(finding.get("ep")))].append(
            norm(finding.get("found", ""))
        )

    def in_baseline(finding: dict) -> bool:
        detector, ep = finding.get("detector"), as_int(finding.get("ep"))
        value = norm(finding.get("found", ""))
        if (detector, ep, as_int(finding.get("scene")), value) in baseline_keys:
            return True
        if not value:
            return False
        return any(value in other or other in value for other in baseline_by_detector.get((detector, ep), []) if other)
    related = []
    for item in truth.get("items", []):
        if item.get("detector") != "joins":
            related.append((item.get("detector"), norm(item.get("new", "")), set(item.get("episodes", []))))
    matched = set()

    def is_seed_related(finding: dict) -> bool:
        for detector, token, episodes in related:
            if finding.get("detector") == detector and as_int(finding.get("ep")) in episodes and token:
                text = norm(str(finding.get("found", "")) + " " + str(finding.get("suggestion", "")))
                if token in text:
                    return True
        if finding.get("detector") == "joins":
            for item in truth.get("items", []):
                if item.get("detector") != "joins":
                    continue
                for occ in item.get("occurrences", []):
                    if (
                        as_int(finding.get("ep")) == int(occ["ep"])
                        and norm(item.get("new", ""))
                        in norm(str(finding.get("found", "")) + " " + str(finding.get("suggestion", "")))
                    ):
                        return True
        return False

    for item in truth.get("items", []):
        detector = item.get("detector", "")
        per[detector]["items"] += 1
        hits = [f for f in findings if seed_match(item, f)]
        if hits:
            per[detector]["found"] += 1
            if any(seed_exact(item, f) for f in hits):
                per[detector]["exact"] += 1
            matched.update(id(f) for f in hits)
    for finding in findings:
        if id(finding) in matched:
            continue
        if in_baseline(finding) or is_seed_related(finding):
            continue
        per[finding.get("detector", "")]["new_false"] += 1
    for value in per.values():
        if value["items"]:
            value["recall"] = round(value["found"] / value["items"], 4)
        if value["found"]:
            value["location_exact"] = round(value["exact"] / value["found"], 4)
    return {"by_detector": dict(per), "total": {
        "items": sum(v["items"] for v in per.values()),
        "found": sum(v["found"] for v in per.values()),
        "exact": sum(v["exact"] for v in per.values()),
        "new_false": sum(v["new_false"] for v in per.values()),
    }}


def stats_block(stats: dict, seconds: float, cost: float | None, cached: int = 0) -> dict:
    calls = sum(int(value.get("calls", 0)) for value in stats.values())
    cached_total = cached or sum(int(value.get("cached", 0)) for value in stats.values())
    return {
        "calls": calls,
        "cached": cached_total,
        "tokens_in": sum(int(value.get("input_tokens", 0)) for value in stats.values()),
        "tokens_out": sum(int(value.get("output_tokens", 0)) for value in stats.values()),
        "reasoning_tokens": sum(int(value.get("reasoning_tokens", 0)) for value in stats.values()),
        "seconds": round(seconds, 2),
        "cost_usd": cost,
    }


def load_findings(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / "findings.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run Canon first")
    return json.loads(path.read_text(encoding="utf-8"))


def as_dicts(items) -> list[dict]:
    return [item.model_dump() if hasattr(item, "model_dump") else item for item in items]


def load_manifest(run_dir: str | Path) -> dict:
    path = Path(run_dir) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_answer_key(path: str | Path) -> list[dict]:
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig", newline="")))
    return [row for row in rows if str(row.get("confirmed", "")).strip().lower() in CONFIRMED_TRUE]


def baseline_detector(kind: str) -> str:
    text = str(kind).lower()
    if any(word in text for word in ("speaker", "label", "character", "voice")):
        return "speakers"
    if any(word in text for word in ("join", "time", "day", "night", "building", "location", "set")):
        return "joins"
    if any(word in text for word in ("spell", "typo", "misspell", "word")):
        return "spelling"
    if "name" in text:
        return "names"
    return ""


def baseline_findings(answer: BaselineOut) -> list[dict]:
    out = []
    for item in answer.findings:
        detector = baseline_detector(item.kind)
        if not detector:
            continue
        ep, scene = as_int(item.episode), as_int(item.scene)
        out.append(
            {
                "detector": detector,
                "ep": ep,
                "scene": scene,
                "where": f"E{ep:02d} S{scene:02d}",
                "start": item.time,
                "found": item.found,
                "suggestion": item.suggestion,
                "context": item.detail,
            }
        )
    return out


def run_baseline(input_source: str, out_dir: str | Path) -> dict:
    instructions = (PROMPTS / "baseline_show.txt").read_text(encoding="utf-8")
    files = read_input(input_source)
    prompt = instructions + "".join(f"\n\n===== FILE: {name} =====\n{text}" for name, text in files)
    out = Path(out_dir)
    llm.reset_stats()
    llm.set_run_dir(out / "baseline")
    started = time.time()
    try:
        answer = llm.generate("baseline", prompt, BaselineOut, models=[BASELINE_ROUTE])
    except Exception as exc:
        stats = llm.get_stats().get("baseline", {})
        return {
            "complete": False,
            "error": f"{type(exc).__name__}: {exc}",
            "findings": [],
            "stats": stats_block({"baseline": stats}, time.time() - started, None),
        }
    finally:
        llm.set_run_dir(None)
    stats = llm.get_stats().get("baseline", {})
    return {
        "complete": True,
        "findings": baseline_findings(answer),
        "stats": stats_block({"baseline": stats}, time.time() - started, cost_usd({"baseline": stats})),
        "prompt_tokens_estimate": len(prompt) // 4,
    }


def bench(
    input_source: str = "data/episodes",
    out_dir: str = "out/bench",
    seeds: list[int] | None = None,
    baseline: str | None = "show",
    answer_key_path: str = "audit/answer_key.csv",
    final_run: str = "out/final",
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    final = Path(final_run)

    if not (final / "findings.json").exists():
        run(input_source, str(final))
    canon_findings = load_findings(final)
    canon_manifest = load_manifest(final)

    code_dir = out / "canon_code"
    code_result = run(input_source, str(code_dir), no_model=True)
    code_findings = as_dicts(code_result["findings"])
    code_manifest = code_result["manifest"]

    seed_results = {}
    for number in seeds or []:
        seed_dir = Path(f"out/seed_{number}")
        truth_path = seed_dir / "truth.json"
        if not truth_path.exists():
            make_seed(input_source, str(seed_dir), number, str(final))
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        run_dir = seed_dir / "run"
        manifest = load_manifest(run_dir)
        expected = input_hash(str(seed_dir / "input"))
        if not manifest.get("complete") or manifest.get("input_sha256") != expected:
            run_result = run(str(seed_dir / "input"), str(run_dir))
            manifest = run_result["manifest"]
            seeded_findings = as_dicts(run_result["findings"])
        else:
            seeded_findings = load_findings(run_dir)
        seed_results[number] = {
            "truth": truth,
            "findings": seeded_findings,
            "stats": stats_block(manifest.get("llm", {}), manifest.get("seconds", 0.0), manifest.get("cost_usd")),
        }

    baseline_result = None
    if baseline == "show":
        baseline_result = run_baseline(input_source, out)

    rows = load_answer_key(answer_key_path)
    answer_key = {
        "rows": len(rows),
        "canon": score_answer_key(rows, canon_findings),
        "canon_no_model": score_answer_key(rows, code_findings),
    }
    if baseline_result and baseline_result.get("complete"):
        answer_key["b2"] = score_answer_key(rows, baseline_result["findings"])

    seeded = {}
    for number, result in seed_results.items():
        seeded[str(number)] = {
            "counts": result["truth"].get("counts", {}),
            "canon": score_seed(result["truth"], result["findings"], canon_findings),
        }
        if baseline_result and baseline_result.get("complete") and number == (seeds or [None])[0]:
            seeded[str(number)]["b2"] = score_seed(result["truth"], baseline_result["findings"], canon_findings)

    methods = {
        "canon": stats_block(
            canon_manifest.get("llm", {}),
            canon_manifest.get("seconds", 0.0),
            canon_manifest.get("cost_usd"),
        ),
        "canon_no_model": stats_block(
            code_manifest.get("llm", {}),
            code_manifest.get("seconds", 0.0),
            0.0,
        ),
    }
    for number, result in seed_results.items():
        methods[f"canon_seed_{number}"] = result["stats"]
    if baseline_result:
        methods["b2"] = baseline_result["stats"]

    missing = {}
    if baseline == "show" and baseline_result and not baseline_result.get("complete"):
        missing["b2"] = baseline_result.get("error", "unknown failure")
    if not rows:
        missing["answer_key"] = "no confirmed rows"

    metrics = {
        "complete": not missing,
        "input": input_source,
        "answer_key": answer_key,
        "seeds": seeded,
        "methods": methods,
        "missing": missing,
        "notes": {
            "answer_key_rows_confirmed": len(rows),
            "baseline_route": BASELINE_ROUTE if baseline == "show" else None,
            "match_rule": "detector, episode, scene when stated, and the found text or a key term",
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "bench.md").write_text(render(metrics), encoding="utf-8")
    return metrics


def _pct(found: int, total: int) -> str:
    return f"{found}/{total} ({found / total:.0%})" if total else "0/0"


def render(metrics: dict) -> str:
    lines = ["# Canon benchmark", ""]
    lines.append(f"Input: `{metrics['input']}`. Confirmed answer-key rows: **{metrics['notes']['answer_key_rows_confirmed']}**.")
    lines.append("")
    lines.append("## Answer key (confirmed rows only)")
    lines.append("")
    lines.append("| Method | Rows | Found | Of those, in part | Recall | Location exact |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, title in (("canon", "Canon (model)"), ("canon_no_model", "Canon (--no-model)"), ("b2", "B2 (whole show, one call)")):
        if name not in metrics["answer_key"]:
            continue
        total = metrics["answer_key"][name]["total"]
        located = total.get("located", 0)
        exact = f"{total.get('exact', 0)}/{located}" if located else "n/a"
        lines.append(
            f"| {title} | {total['rows']} | {total['found']} | {total.get('partial', 0)} | "
            f"{_pct(total['found'], total['rows'])} | {exact} |"
        )
    lines.append("")

    lines.append("### By detector, Canon (model)")
    lines.append("")
    lines.append("| Detector | Rows | Found | Of those, in part | Recall | Location exact |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for detector, value in sorted(metrics["answer_key"]["canon"]["by_detector"].items()):
        located = value.get("located", 0)
        exact = f"{value['exact']}/{located}" if located else "n/a"
        lines.append(
            f"| {detector} | {value['rows']} | {value['found']} | {value.get('partial', 0)} | "
            f"{_pct(value['found'], value['rows'])} | {exact} |"
        )
    lines.append("")

    missed = [row for row in metrics["answer_key"]["canon"]["rows"] if not row["found"]]
    if missed:
        lines.append("### Missed answer-key rows, Canon (model)")
        lines.append("")
        lines.append("| Row | Detector | Where |")
        lines.append("|---|---|---|")
        for row in missed:
            lines.append(f"| {row['id']} | {row['detector']} | {row['where'] or row['episodes']} |")
        lines.append("")
    partial = [row for row in metrics["answer_key"]["canon"]["rows"] if row.get("partial")]
    if partial:
        lines.append("### Found only in part, Canon (model)")
        lines.append("")
        lines.append("A speaker row is found in part when Canon links two of its labels but leaves at least one out.")
        lines.append("")
        lines.append("| Row | Detector | Episodes |")
        lines.append("|---|---|---|")
        for row in partial:
            lines.append(f"| {row['id']} | {row['detector']} | {row['episodes']} |")
        lines.append("")

    if metrics["seeds"]:
        lines.append("## Seeded benchmark")
        lines.append("")
        for seed, result in metrics["seeds"].items():
            lines.append(f"### Seed {seed}")
            lines.append("")
            lines.append(f"Injected: {result['counts']}.")
            lines.append("")
            for method, title in (("canon", "Canon (model)"), ("b2", "B2 (whole show)")):
                if method not in result:
                    continue
                lines.append(f"**{title}**")
                lines.append("")
                lines.append("| Detector | Items | Found | Recall | Location exact | New false findings |")
                lines.append("|---|---:|---:|---:|---:|---:|")
                for detector, value in sorted(result[method]["by_detector"].items()):
                    lines.append(
                        f"| {detector} | {value['items']} | {value['found']} | "
                        f"{_pct(value['found'], value['items'])} | {value.get('location_exact', 0):.0%} | "
                        f"{value['new_false']} |"
                    )
                total = result[method]["total"]
                lines.append(
                    f"| **all** | **{total['items']}** | **{total['found']}** | "
                    f"**{_pct(total['found'], total['items'])}** | | **{total['new_false']}** |"
                )
                lines.append("")

    lines.append("## Calls, tokens, time, cost")
    lines.append("")
    lines.append("| Method | Calls | Cached | Tokens in | Tokens out | Seconds | Cost USD |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, value in metrics["methods"].items():
        cost = "" if value.get("cost_usd") is None else f"${value['cost_usd']:.4f}"
        lines.append(
            f"| {name} | {value.get('calls', 0)} | {value.get('cached', 0)} | "
            f"{value.get('tokens_in', 0)} | {value.get('tokens_out', 0)} | "
            f"{value.get('seconds', 0)} | {cost} |"
        )
    lines.append("")

    if metrics["missing"]:
        lines.append("## Missing")
        lines.append("")
        for name, reason in metrics["missing"].items():
            lines.append(f"- **{name}**: {reason}")
        lines.append("")

    lines.append("## How rows are matched")
    lines.append("")
    lines.append("- Answer key: a row is found when a finding has the row's detector and its episode is in the row's episode list. Names and spelling also need one of the row's key terms as whole words in the finding (or the finding's word as whole words in the row). A speaker row needs a finding that links two of the row's own labels, compared as whole labels, so a label never matches a longer label that contains it; a one-label-two-people row needs a veto check on that label. Joins need the stated scene pair. Location exactness compares the finding's episode and scene with the location the row states.")
    lines.append("- Seeds: an item is found when a finding has the item's detector, the right episode, and the injected spelling, label or word (or is the join at the injected scene with the new day/night or building value). A speaker split also counts when the injected label appears in an intentional group, which is reported once at the group's first line.")
    lines.append("- New false findings are findings in the seeded run that do not match a seeded item, do not appear in `out/final`, and are not otherwise attributable to a seed. `out/final` is one cached run and the judge is stochastic, so a label grouped differently in a live seeded run can appear as new even when the seed did not touch it.")
    lines.append("- `v2` answer-key rows are not scored: the optional AI read is not part of v1.")
    return "\n".join(lines) + "\n"
