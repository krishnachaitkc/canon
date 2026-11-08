import argparse
from pathlib import Path

from pydantic import BaseModel

from .doctor import doctor
from .llm import generate, get_stats
from .pipeline import DETECTORS, STOP_POINTS, run


class SmokeOut(BaseModel):
    status: str


def smoke(route: str) -> int:
    try:
        result = generate(
            "smoke",
            'Return exactly one JSON object and nothing else: {"status": "OK"}',
            SmokeOut,
            models=[route],
            use_cache=False,
        )
    except Exception as exc:
        print(f"model: {route}")
        print(f"status: FAILED: {exc}")
        return 1
    stats = get_stats().get("smoke", {})
    ok = result.status.strip().upper() == "OK"
    print(f"model: {route}")
    print(f"status: {'OK' if ok else result.status}")
    print(f"tokens: in={stats.get('input_tokens', 0)} out={stats.get('output_tokens', 0)}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="canon")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor_parser = sub.add_parser("doctor", help="parse a show and print its facts")
    doctor_parser.add_argument("--input", "-i", required=True, help="folder or zip of screenplays")
    doctor_parser.add_argument("--title", default=None)

    smoke_parser = sub.add_parser("smoke", help="one tiny JSON call to check a model route")
    smoke_parser.add_argument("--model", required=True, help="provider/name")

    run_parser = sub.add_parser("run", help="run Canon on a show")
    run_parser.add_argument("--input", "-i", required=True, help="folder or zip of screenplays")
    run_parser.add_argument("--out", "-o", required=True, help="run folder")
    run_parser.add_argument("--title", default=None)
    run_parser.add_argument("--stop-after", choices=STOP_POINTS, default="all")
    run_parser.add_argument("--no-model", action="store_true", help="code only; nothing leaves the machine")
    run_parser.add_argument(
        "--only", default=None, help=f"comma list of detectors: {','.join(DETECTORS)}"
    )
    run_parser.add_argument("--model", default=None, help="provider/name for every judge, e.g. gemini/gemini-3.5-flash-lite")
    run_parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint for --model openai/...")
    run_parser.add_argument("--api-key-env", default=None, help="environment variable holding the key")
    run_parser.add_argument("--record", default=None, help="save every model answer here (for tests)")
    run_parser.add_argument("--details", action="store_true", help="also list groups and findings (quotes the script)")

    patch_parser = sub.add_parser("patch", help="apply approved findings to a copy of the scripts")
    patch_parser.add_argument("--run", required=True, help="run folder")
    patch_parser.add_argument("--approvals", default=None, help="approvals.json (default: <run>/approvals.json)")

    audit_parser = sub.add_parser("audit-template", help="write audit/labels_<run>.csv for a person to fill")
    audit_parser.add_argument("--run", required=True, help="run folder")
    audit_parser.add_argument("--force", action="store_true", help="replace a file that already holds labels")

    key_parser = sub.add_parser("answer-key", help="write audit/answer_key.csv from plan/EVIDENCE.md")
    key_parser.add_argument("--evidence", default="plan/EVIDENCE.md")
    key_parser.add_argument("--out", default="audit/answer_key.csv")
    key_parser.add_argument("--force", action="store_true", help="replace a file that already holds labels")

    compare_parser = sub.add_parser("compare", help="how far two runs' judges agree")
    compare_parser.add_argument("--a", required=True)
    compare_parser.add_argument("--b", required=True)

    seed_parser = sub.add_parser("seed", help="inject known mistakes and write truth.json")
    seed_parser.add_argument("--input", "-i", required=True, help="folder or zip of screenplays")
    seed_parser.add_argument("--out", "-o", required=True, help="seed folder")
    seed_parser.add_argument("--seed", type=int, required=True)
    seed_parser.add_argument("--final-run", default="out/final", help="run whose groups are not seeded again")

    bench_parser = sub.add_parser("bench", help="score against the answer key and seeded packs")
    bench_parser.add_argument("--input", "-i", default="data/episodes", help="folder or zip of screenplays")
    bench_parser.add_argument("--out", "-o", required=True, help="bench folder")
    bench_parser.add_argument("--seeds", nargs="+", type=int, default=None)
    bench_parser.add_argument("--baseline", choices=["show", "none"], default=None)
    bench_parser.add_argument("--answer-key", default="audit/answer_key.csv")
    bench_parser.add_argument("--final-run", default="out/final")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor(args.input, title=args.title)
    if args.command == "smoke":
        return smoke(args.model)
    if args.command == "run":
        return run_command(args)
    if args.command == "compare":
        return compare_command(args.a, args.b)
    if args.command == "patch":
        return patch_command(args.run, args.approvals)
    if args.command == "audit-template":
        return audit_command(args.run, args.force)
    if args.command == "answer-key":
        return answer_key_command(args.evidence, args.out, args.force)
    if args.command == "seed":
        return seed_command(args)
    if args.command == "bench":
        return bench_command(args)
    parser.error(f"unknown command: {args.command}")
    return 2


def seed_command(args) -> int:
    from .seed import SeedError, make_seed

    try:
        truth = make_seed(args.input, args.out, args.seed, args.final_run)
    except SeedError as exc:
        print(f"seed: {exc}")
        return 1
    print(f"seed {args.seed}: {args.input} -> {args.out}")
    print("injected: " + "  ".join(f"{name} {count}" for name, count in truth["counts"].items()))
    print(f"truth: {Path(args.out) / 'truth.json'}")
    return 0


def bench_command(args) -> int:
    from .bench import bench

    metrics = bench(args.input, args.out, args.seeds, args.baseline, args.answer_key, args.final_run)
    answer_key = metrics["answer_key"]
    print(f"bench: {args.out}")
    print(f"answer key: {metrics['notes']['answer_key_rows_confirmed']} confirmed rows")
    for name in ("canon", "canon_no_model", "b2"):
        if name in answer_key:
            total = answer_key[name]["total"]
            print(f"  {name}: {total['found']}/{total['rows']} found")
    for seed, result in metrics["seeds"].items():
        for name in ("canon", "b2"):
            if name in result:
                total = result[name]["total"]
                print(f"  seed {seed} {name}: {total['found']}/{total['items']} found, {total['new_false']} new false")
    for name, reason in metrics["missing"].items():
        print(f"missing {name}: {reason}")
    print(f"wrote: {Path(args.out) / 'metrics.json'}  {Path(args.out) / 'bench.md'}")
    return 0


def run_command(args) -> int:
    only = [d.strip() for d in args.only.split(",")] if args.only else None
    unknown = [d for d in only or [] if d not in DETECTORS]
    if unknown:
        raise SystemExit(f"unknown detector(s): {', '.join(unknown)}")
    result = run(
        args.input,
        args.out,
        title=args.title,
        stop_after=args.stop_after,
        only=only,
        no_model=args.no_model,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        record_dir=args.record,
    )
    manifest = result["manifest"]
    counts = manifest["counts"]
    print(f"Canon run: {manifest['title']} -> {args.out}  detectors: {','.join(manifest['detectors'])}")
    print(
        f"episodes: {counts['episodes']}  mentions: {counts['mentions']}  names: {counts['names']}  "
        f"candidates: {counts['candidates']}  speaker labels: {counts['speaker_labels']}  "
        f"label pairs sharing a word: {counts['core_pairs']}"
    )
    groups_n, speakers_n = result["name_groups"], result["speaker_groups"]
    print(
        f"name groups: {len(groups_n)} (spellings merged: {sum(len(g.spellings) - 1 for g in groups_n)})  "
        f"speaker groups: {len(speakers_n)} (intentional: {sum(1 for g in speakers_n if g.intentional)})"
    )
    sev, src = counts["by_severity"], counts["by_source"]
    print(f"findings: {counts['findings']}  " + "  ".join(f"{k} {v}" for k, v in sev.items()))
    print("by detector: " + "  ".join(f"{d} {counts[f'findings_{d}']}" for d in DETECTORS))
    print("by source: " + "  ".join(f"{k} {v}" for k, v in src.items()))
    print(
        f"episodes affected: {counts['episodes_affected']} of {counts['episodes']}  "
        f"fixes ready to approve: {counts['fixes_ready']}  report only: {counts['findings'] - counts['fixes_ready']}"
    )
    print("wrote: findings.csv cast.csv summary.md report.html groups.json findings.json manifest.json")
    if args.details:
        print_details(result)
    print_cost(manifest)
    return 0 if manifest["complete"] else 1


def print_details(result) -> None:
    manifest = result["manifest"]
    for line in manifest["threshold_log"]:
        print(f"note: {line}")
    if result["candidates"]:
        print("candidates (spelling: mentions/spoken):")
        for candidate in result["candidates"]:
            parts = [
                f"{name} {total}/{candidate.spoken.get(name, 0)}"
                for name, total in candidate.spellings.items()
            ]
            print(f"  {candidate.id}: " + ", ".join(parts))
    if result["name_groups"] or result["apart"]:
        print("name groups (canonical <- other spellings):")
        for g in result["name_groups"]:
            others = [s for s in g.spellings if s != g.canonical]
            print(f"  {g.id} [{g.source} {g.confidence}] {g.entity_type}: {g.canonical} <- {', '.join(others)}")
        for a in result["apart"]:
            print(f"  kept apart in {a['candidate']}: {', '.join(a['spellings'])}")
    if result["speaker_groups"]:
        print("speaker groups (canonical <- other labels):")
        for g in result["speaker_groups"]:
            others = [l for l in g.labels if l != g.canonical]
            flag = " intentional" if g.intentional else ""
            print(f"  {g.id} [{g.source} {g.confidence}{flag}] {g.canonical} <- {', '.join(others)}")
            for a, b, where in g.vetoed:
                print(f"      vetoed {a} + {b}: both speak in {where}")
    if result["verdicts"]:
        print("spelling verdicts:")
        for v in result["verdicts"]:
            print(f"  '{v['word']}': {v['verdict']}" + (f" -> '{v['correction']}'" if v["correction"] else ""))
    for detector in DETECTORS:
        rows = [f for f in result["findings"] if f.detector == detector]
        if not rows:
            continue
        print(f"{detector} findings: {len(rows)}")
        if detector in ("names", "speakers"):
            per_group = {}
            for f in rows:
                per_group[f.group] = per_group.get(f.group, 0) + 1
            print("  rows per group: " + ", ".join(f"{g} {n}" for g, n in per_group.items()))
            rows = [f for f in rows if f.source == "check"]
        for f in rows:
            if detector == "joins":
                print(f"  {f.id} {f.where:22s} {f.group:8s} {f.source:5s} {f.found}  [{f.context}]")
            elif detector == "spelling":
                print(f"  {f.id} [{f.source}] E{f.ep:02d} S{f.scene:02d} {f.start} {f.where}  '{f.found}' -> '{f.suggestion}'")
            elif detector == "speakers":
                print(f"  {f.id} [{f.source}] {f.found}")
            elif detector == "names":
                print(f"  {f.id} [{f.source}] {f.where}: {f.found}")
            else:
                print(f"  {f.id} E{f.ep:02d} {f.where}: '{f.found}' (others: '{f.suggestion}')")


def print_cost(manifest) -> None:
    llm_stats = manifest.get("llm", {})
    calls = sum(s.get("calls", 0) for s in llm_stats.values())
    cached = sum(s.get("cached", 0) for s in llm_stats.values())
    tokens_in = sum(s.get("input_tokens", 0) for s in llm_stats.values())
    tokens_out = sum(s.get("output_tokens", 0) for s in llm_stats.values())
    if manifest.get("model"):
        print(
            f"model: {manifest['model'][0]}  calls: {calls} (+{cached} cached)  "
            f"tokens: {tokens_in} in / {tokens_out} out  cost: ${manifest.get('cost_usd')}"
        )
    for switch in manifest.get("model_switches", []):
        print(f"note: {switch}")
    print(f"seconds: {manifest.get('seconds', 0)}")


def patch_command(run_dir: str, approvals: str | None) -> int:
    from .patch import PatchError, patch

    try:
        stats = patch(run_dir, approvals)
    except (PatchError, FileNotFoundError) as exc:
        print(f"patch failed: {exc}")
        return 1
    print(
        f"approved: {stats['approved']}  rejected: {stats['rejected']}  canonical overrides: {stats['overrides']}  "
        f"edits: {stats['edits']}  applied: {stats['applied']}  report only: {stats['report_only']}  "
        f"no replacement: {stats['no_replacement']}  files changed: {stats['files_changed']}"
    )
    for fid in stats["unknown_ids"]:
        print(f"note: unknown finding {fid}")
    for problem in stats["problems"]:
        print(f"problem: {problem}")
    print(f"wrote: {run_dir}/patched/ {run_dir}/patch.diff (the input is unchanged)")
    return 1 if stats["problems"] else 0


def audit_command(run_dir: str, force: bool) -> int:
    from .audit import AuditError, audit_template

    try:
        path = audit_template(run_dir, force=force)
    except AuditError as exc:
        print(f"audit-template: {exc}")
        return 1
    with open(path, encoding="utf-8-sig") as fh:
        rows = sum(1 for _ in fh) - 1
    print(f"wrote: {path} ({rows} rows; correct and note are empty for you to fill)")
    return 0


def answer_key_command(evidence: str, out: str, force: bool) -> int:
    from collections import Counter

    from .audit import AuditError, answer_key

    try:
        path, items = answer_key(evidence, out, force)
    except AuditError as exc:
        print(f"answer-key: {exc}")
        return 1
    targets = Counter(item["targeted_by"] for item in items)
    levels = Counter(item["confidence"].split(" (")[0] for item in items)
    print(f"wrote: {path} ({len(items)} items; confirmed is empty for you to fill)")
    print("targeted by: " + "  ".join(f"{k} {v}" for k, v in sorted(targets.items())))
    print("confidence: " + "  ".join(f"{k} {v}" for k, v in sorted(levels.items())))
    return 0


def compare_command(run_a: str, run_b: str) -> int:
    from .judge import agreement

    result = agreement(run_a, run_b)
    for part in ("names", "speakers", "spelling"):
        share, total, differences = result[part]
        print(f"{part}: {share:.0%} agree ({total} compared)")
        for line in differences:
            print(f"  differs: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
