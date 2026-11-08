"""Run Canon on a show and write every output to one run folder."""

import hashlib
import json
import time
from pathlib import Path

from . import llm
from .config import ROUTES, is_peak, price_for
from .errors import describe
from .doctor import format_findings
from .findings import cast_rows, name_rows, sort_and_number, speaker_rows
from .joins import find_joins
from .judge import ModelSpec, code_names, code_speakers, judge_names, judge_speakers, judge_spelling
from .mentions import build_index
from .names import build_candidates
from .parse import parse_show, read_input
from .report import tallies, write_outputs
from .speakers import core_pairs, speaker_stats
from .spelling import spelling_candidates

STOP_POINTS = ["candidates", "all"]
DETECTORS = ["names", "speakers", "joins", "spelling", "format"]


def input_hash(input_source: str) -> str:
    digest = hashlib.sha256()
    for name, content in read_input(input_source):
        digest.update(name.encode("utf-8"))
        digest.update(content.encode("utf-8"))
    return digest.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def cost_usd(stats: dict) -> float | None:
    """Cost from token counts and the price table; None if a route has no known price."""
    total, peak = 0.0, is_peak()
    for stage in stats.values():
        for route, tokens in stage.get("tokens_by_model", {}).items():
            price = price_for(route, peak)
            if price is None:
                return None
            fresh = max(0, tokens["input"] - tokens["cache_hit"])
            total += fresh * price["input"] / 1e6
            total += tokens["cache_hit"] * price["cached"] / 1e6
            total += tokens["output"] * price["output"] / 1e6
    return round(total, 6)


def _phase(name: str) -> None:
    """Report a code phase if this build of llm supports it.

    A deploy can pair this module with an older `canon.llm`; progress is cosmetic and
    must never stop a run.
    """
    reporter = getattr(llm, "report_phase", None)
    if reporter is not None:
        reporter(name)


def run(
    input_source: str,
    out_dir: str,
    title: str | None = None,
    stop_after: str = "all",
    only: list[str] | None = None,
    no_model: bool = False,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    record_dir: str | None = None,
) -> dict:
    started = time.time()
    selected = [d for d in DETECTORS if d in (only or DETECTORS)]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    judging = stop_after == "all" and not no_model
    routes = [model] if model else list(ROUTES["names"])
    spec = ModelSpec(
        routes=routes,
        base_url=base_url,
        api_key_env=api_key_env,
        record_dir=Path(record_dir) if record_dir else None,
    )
    llm.reset_stats()
    llm.set_run_dir(out if judging else None)

    manifest = {
        "complete": False,
        "stopped_after": stop_after,
        "detectors": selected,
        "no_model": no_model,
        "model": routes if judging else None,
        "input": str(input_source),
        "input_sha256": input_hash(input_source),
    }
    try:
        _phase("Reading the scripts")
        show = parse_show(input_source, title=title)
        index = build_index(show)
        _write_json(out / "mentions.json", [m.model_dump() for m in index.mentions])
        manifest["title"] = show.title

        candidates, threshold_log, stats, pairs = [], [], {}, []
        name_groups, apart, speaker_groups, verdicts = [], [], [], []
        findings = []
        if "names" in selected:
            _phase("Finding name variants")
            candidates, threshold_log = build_candidates(index)
            _write_json(out / "candidates.json", [c.model_dump() for c in candidates])
        if "speakers" in selected:
            _phase("Reading speaker labels")
            stats = speaker_stats(show, index)
            pairs = core_pairs(stats)
            _write_json(
                out / "speakers.json",
                {"labels": [s.model_dump() for s in stats.values()], "core_pairs": pairs},
            )

        if stop_after == "all":
            if "names" in selected:
                if no_model:
                    name_groups = code_names(candidates, index)
                else:
                    _phase("Judging name groups")
                    name_groups, apart, name_checks = judge_names(candidates, index, show, spec)
                    findings += name_checks
                findings += name_rows(name_groups, index, show)
            if "speakers" in selected:
                if no_model:
                    speaker_groups, checks = code_speakers(stats, show)
                else:
                    _phase("Judging speaker labels")
                    speaker_groups, checks = judge_speakers(stats, show, spec)
                findings += checks
                findings += speaker_rows(speaker_groups, show)

        if "joins" in selected:
            _phase("Checking scene joins")
            findings += find_joins(show)
        if "spelling" in selected:
            _phase("Checking spoken spelling")
            candidates_d4 = spelling_candidates(show, index)
            if judging:
                _phase("Judging spelling")
                kept, verdicts = judge_spelling(candidates_d4, show, spec)
                findings += kept
            else:
                findings += candidates_d4
        if "format" in selected:
            _phase("Checking header format")
            findings += format_findings(input_source)
        findings = sort_and_number(findings)

        _write_json(
            out / "groups.json",
            {
                "names": [g.model_dump() for g in name_groups],
                "names_kept_apart": apart,
                "speakers": [g.model_dump() for g in speaker_groups],
                "spelling_verdicts": verdicts,
            },
        )
        _write_json(out / "findings.json", [f.model_dump() for f in findings])

        counts = {
            "episodes": len(show.episodes),
            "mentions": len(index.mentions),
            "names": len(index.names),
            "candidates": len(candidates),
            "name_groups": len(name_groups),
            "speaker_labels": len(stats),
            "core_pairs": len(pairs),
            "speaker_groups": len(speaker_groups),
        }
        for detector in DETECTORS:
            counts[f"findings_{detector}"] = sum(1 for f in findings if f.detector == detector)
        tally = tallies(findings)
        counts["findings"] = len(findings)
        counts["by_severity"] = tally["by_severity"]
        counts["by_source"] = tally["by_source"]
        counts["episodes_affected"] = len(tally["episodes"])
        counts["fixes_ready"] = tally["with_fix"]
        llm_stats = llm.get_stats()
        manifest.update(
            {
                "complete": True,
                "counts": counts,
                "threshold_log": threshold_log,
                "llm": llm_stats,
                "model_switches": llm.get_switches(),
                "cost_usd": cost_usd(llm_stats) if judging else 0.0,
            }
        )
        cast = cast_rows(stats, speaker_groups, findings) if stats else []
        write_outputs(out, manifest, findings, name_groups, apart, speaker_groups, verdicts, cast)
        return {
            "manifest": manifest,
            "candidates": candidates,
            "stats": stats,
            "pairs": pairs,
            "name_groups": name_groups,
            "apart": apart,
            "speaker_groups": speaker_groups,
            "verdicts": verdicts,
            "findings": findings,
            "cast": cast,
        }
    except Exception as exc:
        info = describe(exc)
        manifest["error"] = info.cause
        manifest["error_title"] = info.title
        manifest["error_hints"] = list(info.hints)
        manifest["error_detail"] = info.detail
        manifest["llm"] = llm.get_stats()
        manifest["model_switches"] = llm.get_switches()
        raise
    finally:
        manifest["seconds"] = round(time.time() - started, 2)
        _write_json(out / "manifest.json", manifest)
        llm.set_run_dir(None)
        clear = getattr(llm, "set_progress", None)
        if clear is not None:
            clear(None)

