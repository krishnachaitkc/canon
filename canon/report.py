"""Write a run's human outputs: findings.csv, cast.csv, summary.md and report.html."""

import csv
import html
from collections import Counter
from pathlib import Path

from .findings import SEVERITY_ORDER, episode_ranges
from .models import Finding

FINDING_COLUMNS = [
    "id", "severity", "detector", "source", "confidence", "episode", "scene", "start",
    "where", "found", "suggestion", "context", "group", "fix",
]
CAST_COLUMNS = ["character", "labels", "lines", "lines_per_label", "episodes", "group", "source", "note", "vetoes", "checks"]
DETECTOR_TITLES = {"names": "Names", "speakers": "Speakers", "joins": "Scene joins", "spelling": "Spelling", "format": "Format"}
SOURCES = ["rule", "ai", "check", "code"]


def fix_text(f: Finding) -> str:
    if f.edit is None:
        return ""
    nth = f" (occurrence {f.edit.nth + 1})" if f.edit.nth else ""
    return f"{f.edit.at}: {f.edit.old} -> {f.edit.new}{nth}"


def finding_row(f: Finding) -> dict:
    return {
        "id": f.id, "severity": f.severity, "detector": f.detector, "source": f.source,
        "confidence": f.confidence, "episode": f.ep, "scene": f.scene, "start": f.start,
        "where": f.where, "found": f.found, "suggestion": f.suggestion, "context": f.context,
        "group": f.group, "fix": fix_text(f),
    }


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    # utf-8-sig so Excel opens the accents and dashes correctly.
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def tallies(findings: list[Finding]) -> dict:
    return {
        "by_severity": {s: sum(1 for f in findings if f.severity == s) for s in SEVERITY_ORDER},
        "by_detector": {
            d: {s: sum(1 for f in findings if f.detector == d and f.severity == s) for s in SEVERITY_ORDER}
            for d in DETECTOR_TITLES
        },
        "by_source": {s: sum(1 for f in findings if f.source == s) for s in SOURCES},
        "episodes": sorted({f.ep for f in findings if f.ep > 0}),
        "with_fix": sum(1 for f in findings if f.edit is not None and f.edit.new and f.edit.new != f.edit.old),
    }


def summary_md(manifest: dict, findings: list[Finding], name_groups, speaker_groups) -> str:
    """Numbers only, so it can be shared without quoting the script."""
    t = tallies(findings)
    total_eps = manifest.get("counts", {}).get("episodes", 0)
    llm = manifest.get("llm", {})
    calls = sum(s.get("calls", 0) for s in llm.values())
    cached = sum(s.get("cached", 0) for s in llm.values())
    lines = [
        "# Canon summary",
        "",
        f"- Episodes: {total_eps}. Episodes with at least one finding: {len(t['episodes'])}"
        + (f" ({episode_ranges(t['episodes'])})" if t["episodes"] else ""),
        f"- Findings: {len(findings)}. With a fix ready to approve: {t['with_fix']}. Report only: {len(findings) - t['with_fix']}.",
        f"- Name groups: {len(name_groups)} (spellings merged: {sum(len(g.spellings) - 1 for g in name_groups)}). "
        f"Speaker groups: {len(speaker_groups)} (intentional: {sum(1 for g in speaker_groups if g.intentional)}).",
        f"- Model: {(manifest.get('model') or ['none (code only)'])[0]}. Calls: {calls} (+{cached} cached). "
        f"Cost: ${manifest.get('cost_usd', 0)}.",
        "",
        "## Findings by detector and severity",
        "",
        "| Detector | audible | voice | visible | notes | total |",
        "|---|---|---|---|---|---|",
    ]
    for d, title in DETECTOR_TITLES.items():
        row = t["by_detector"][d]
        lines.append(f"| {title} | " + " | ".join(str(row[s]) for s in SEVERITY_ORDER) + f" | {sum(row.values())} |")
    sev = t["by_severity"]
    lines.append("| **All** | " + " | ".join(f"**{sev[s]}**" for s in SEVERITY_ORDER) + f" | **{len(findings)}** |")
    lines += [
        "",
        "Severity: audible = a voice says it; voice = a speaker label (casting); visible = a heading or description "
        "the video model draws; notes = episode notes and format.",
        "",
        "## Findings by source",
        "",
        "| Source | count | meaning |",
        "|---|---|---|",
        f"| rule | {t['by_source']['rule']} | the script contradicts itself |",
        f"| ai | {t['by_source']['ai']} | a model judged it; code wrote every location |",
        f"| check | {t['by_source']['check']} | a person decides |",
        f"| code | {t['by_source']['code']} | code-only mode, not judged |",
        "",
    ]
    return "\n".join(lines)


# ---- report.html ----

CSS = """
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1d1d1b;--muted:#66655f;--line:#e2e1dc;--accent:#3b5bdb;
--audible:#c92a2a;--voice:#9c36b5;--visible:#e8590c;--notes:#5c5f66;--rule:#2b8a3e;--ai:#1971c2;--check:#e67700;--code:#5c5f66;--mark:#fff3bf}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--panel:#1d1d1d;--ink:#ececea;--muted:#a3a29c;--line:#333;--accent:#91a7ff;
--audible:#ff8787;--voice:#da77f2;--visible:#ffa94d;--notes:#adb5bd;--rule:#69db7c;--ai:#74c0fc;--check:#ffc078;--code:#adb5bd;--mark:#5c4d00}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header,main{max-width:1200px;margin:0 auto;padding:16px}h1{font-size:22px;margin:8px 0 4px}h1 small{color:var(--muted);font-weight:400}
.meta,.legend{color:var(--muted);margin:4px 0}.tiles{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 12px;min-width:110px}
.tile b{display:block;font-size:22px}.tile span{color:var(--muted);font-size:12px}
nav{display:flex;flex-wrap:wrap;gap:4px;border-bottom:1px solid var(--line);margin-top:8px}
nav button{background:none;border:0;border-bottom:3px solid transparent;color:var(--muted);padding:8px 12px;font:inherit;cursor:pointer}
nav button.on{color:var(--ink);border-color:var(--accent);font-weight:600}
.tools{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:12px 0}
.tools input[type=search]{flex:1;min-width:220px;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);font:inherit}
.panel{display:none}.panel.on{display:block}
details{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin:10px 0;padding:0 12px}
summary{cursor:pointer;padding:10px 0}.reason{color:var(--muted);margin:0 0 8px}
.wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;margin:6px 0 12px;background:var(--panel)}
th,td{text-align:left;vertical-align:top;padding:6px 8px;border-bottom:1px solid var(--line)}th{font-size:12px;color:var(--muted);font-weight:600}
td.id{font-family:ui-monospace,Consolas,monospace;white-space:nowrap}.loc{white-space:nowrap}.loc small{display:block;color:var(--muted)}
.badge{display:inline-block;border:1px solid currentColor;border-radius:10px;padding:0 7px;font-size:11px;line-height:17px;white-space:nowrap}
.b-rule{color:var(--rule)}.b-ai{color:var(--ai)}.b-check{color:var(--check)}.b-code{color:var(--code)}
.s-audible{color:var(--audible)}.s-voice{color:var(--voice)}.s-visible{color:var(--visible)}.s-notes{color:var(--notes)}
.fix{white-space:nowrap}.old{text-decoration:line-through;color:var(--muted)}mark{background:var(--mark);color:inherit}
.muted{color:var(--muted)}h2{font-size:16px;margin:18px 0 6px}.empty{color:var(--muted);padding:12px 0}
"""

JS = """
const buttons=[...document.querySelectorAll('nav button')];
function show(id){buttons.forEach(b=>b.classList.toggle('on',b.dataset.tab===id));
document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('on',p.id===id));}
buttons.forEach(b=>b.addEventListener('click',()=>{show(b.dataset.tab);history.replaceState(null,'','#'+b.dataset.tab);}));
show((location.hash||'#names').slice(1));
const q=document.getElementById('q'),only=document.getElementById('only');
function filter(){const term=q.value.trim().toLowerCase();
document.querySelectorAll('tr[data-q]').forEach(r=>{const ok=(!term||r.dataset.q.includes(term))&&(!only.checked||r.dataset.source==='check');r.style.display=ok?'':'none';});
document.querySelectorAll('details').forEach(d=>{const rows=d.querySelectorAll('tr[data-q]');if(rows.length)d.style.display=[...rows].some(r=>r.style.display!=='none')?'':'none';});}
q.addEventListener('input',filter);only.addEventListener('change',filter);
"""


def _e(text) -> str:
    return html.escape(str(text), quote=True)


def _badge(source: str, confidence: str = "") -> str:
    conf = f' <span class="muted">{_e(confidence)}</span>' if confidence else ""
    return f'<span class="badge b-{_e(source)}">{_e(source)}</span>{conf}'


def _highlight(context: str, found: str) -> str:
    low, key = context.lower(), found.lower()
    i = low.find(key) if key else -1
    if i < 0:
        return _e(context)
    return _e(context[:i]) + "<mark>" + _e(context[i:i + len(found)]) + "</mark>" + _e(context[i + len(found):])


def _loc(f: Finding) -> str:
    if f.ep == 0:
        return '<td class="loc">whole show</td>'
    scene = f" S{f.scene:02d}" if f.scene else ""
    return f'<td class="loc">E{f.ep:02d}{scene} {_e(f.start)}<small>{_e(f.where)}</small></td>'


def _row(f: Finding, cells: str) -> str:
    q = " ".join([f.id, f"e{f.ep:02d}", f.where, f.found, f.suggestion, f.context, f.group, f.source]).lower()
    return f'<tr data-q="{_e(q)}" data-source="{_e(f.source)}"><td class="id">{_e(f.id)}</td>{cells}</tr>'


def _occurrence_table(rows: list[Finding]) -> str:
    body = "".join(
        _row(
            f,
            _loc(f)
            + f'<td class="s-{f.severity}">{_e(f.severity)}</td>'
            + f'<td class="fix"><span class="old">{_e(f.found)}</span> → <b>{_e(f.suggestion)}</b></td>'
            + f"<td>{_highlight(f.context, f.found)}</td>",
        )
        for f in rows
    )
    return (
        '<div class="wrap"><table><thead><tr><th>ID</th><th>Where</th><th>Severity</th><th>Found → fix</th>'
        f"<th>Context</th></tr></thead><tbody>{body}</tbody></table></div>"
    )


def _check_table(rows: list[Finding]) -> str:
    if not rows:
        return ""
    body = "".join(
        _row(
            f,
            _loc(f)
            + f"<td>{_badge(f.source, f.confidence)}</td><td><b>{_e(f.found)}</b><br>{_e(f.suggestion)}</td>"
            + f'<td class="muted">{_e(f.context)}</td>',
        )
        for f in rows
    )
    return (
        "<h2>For a person to decide</h2>"
        '<div class="wrap"><table><thead><tr><th>ID</th><th>Where</th><th>Source</th><th>What</th><th>Model note</th>'
        f"</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _group_block(title: str, meta: str, reason: str, rows: list[Finding]) -> str:
    table = _occurrence_table(rows) if rows else '<p class="empty">No occurrences to change.</p>'
    return (
        f'<details open><summary>{title} {meta} <span class="muted">· {len(rows)} rows</span></summary>'
        f'<p class="reason">{_e(reason)}</p>{table}</details>'
    )


def _names_panel(findings, name_groups, apart) -> str:
    rows = [f for f in findings if f.detector == "names"]
    by_group: dict[str, list[Finding]] = {}
    for f in rows:
        by_group.setdefault(f.group, []).append(f)
    parts = []
    for g in sorted(name_groups, key=lambda g: (-len(by_group.get(g.id, [])), g.id)):
        others = ", ".join(s for s in g.spellings if s != g.canonical)
        title = f"<b>{_e(g.canonical)}</b> ← {_e(others)}"
        meta = f"{_badge(g.source, g.confidence)} <span class='muted'>{_e(g.entity_type)} · {_e(g.id)}</span>"
        parts.append(_group_block(title, meta, g.reason, by_group.get(g.id, [])))
    parts.append(_check_table([f for f in rows if f.source == "check"]))
    if apart:
        items = "".join(f"<li>{_e(', '.join(a['spellings']))} <span class='muted'>({_e(a['candidate'])})</span></li>" for a in apart)
        parts.append(f"<h2>Kept apart by the judge</h2><ul>{items}</ul>")
    return "".join(parts) or '<p class="empty">No name findings.</p>'


def _speakers_panel(findings, speaker_groups) -> str:
    rows = [f for f in findings if f.detector == "speakers"]
    by_group: dict[str, list[Finding]] = {}
    for f in rows:
        by_group.setdefault(f.group, []).append(f)
    parts = []
    for g in sorted(speaker_groups, key=lambda g: (-len(by_group.get(g.id, [])), g.id)):
        others = ", ".join(l for l in g.labels if l != g.canonical)
        flag = " <span class='badge b-check'>intentional</span>" if g.intentional else ""
        title = f"<b>{_e(g.canonical)}</b> ← {_e(others)}{flag}"
        meta = f"{_badge(g.source, g.confidence)} <span class='muted'>{_e(g.id)}</span>"
        vetoes = "".join(f" Vetoed: {a} + {b} both speak in {w}." for a, b, w in g.vetoed)
        group_rows = by_group.get(g.id, [])
        if g.intentional:
            parts.append(
                f'<details open><summary>{title} {meta}</summary><p class="reason">{_e(g.reason + vetoes)}</p>'
                + "".join(f'<p>{_e(f.id)}: {_e(f.suggestion)}</p>' for f in group_rows)
                + "</details>"
            )
        else:
            parts.append(_group_block(title, meta, g.reason + vetoes, group_rows))
    parts.append(_check_table([f for f in rows if f.source == "check"]))
    return "".join(parts) or '<p class="empty">No speaker findings.</p>'


def _simple_table(rows: list[Finding], headers: list[str], cells) -> str:
    if not rows:
        return '<p class="empty">None found.</p>'
    head = "".join(f"<th>{_e(h)}</th>" for h in ["ID"] + headers)
    body = "".join(_row(f, cells(f)) for f in rows)
    return f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _joins_panel(findings) -> str:
    rows = [f for f in findings if f.detector == "joins"]
    return _simple_table(
        rows,
        ["Join", "Kind", "Source", "What changes", "Suggestion", "Headings"],
        lambda f: f'<td class="loc">{_e(f.where)}<small>{_e(f.start)}</small></td><td>{_e(f.group)}</td>'
        f"<td>{_badge(f.source, f.confidence)}</td><td><b>{_e(f.found)}</b></td><td>{_e(f.suggestion)}</td>"
        f'<td class="muted">{_e(f.context)}</td>',
    )


def _spelling_panel(findings, verdicts) -> str:
    rows = [f for f in findings if f.detector == "spelling"]
    table = _simple_table(
        rows,
        ["Where", "Source", "Found → fix", "Line"],
        lambda f: _loc(f) + f"<td>{_badge(f.source, f.confidence)}</td>"
        f'<td class="fix"><span class="old">{_e(f.found)}</span> → <b>{_e(f.suggestion or "?")}</b></td>'
        f"<td>{_highlight(f.context, f.found)}</td>",
    )
    cleared = [v for v in verdicts if v.get("verdict") != "typo"]
    if cleared:
        items = "".join(f"<li><b>{_e(v['word'])}</b>: {_e(v['verdict'])}. <span class='muted'>{_e(v.get('reason', ''))}</span></li>" for v in cleared)
        table += f"<h2>Unknown to the dictionary, cleared by the judge</h2><ul>{items}</ul>"
    return table


def _format_panel(findings) -> str:
    rows = [f for f in findings if f.detector == "format"]
    return _simple_table(
        rows,
        ["Episode", "Label", "Found", "Most episodes write", "Note"],
        lambda f: f"<td>E{f.ep:02d}</td><td>{_e(f.where)}</td><td><b>{_e(f.found)}</b></td>"
        f'<td>{_e(f.suggestion)}</td><td class="muted">{_e(f.context)}</td>',
    )


def _cast_panel(cast: list[dict]) -> str:
    if not cast:
        return '<p class="empty">No speaker labels.</p>'
    head = "".join(f"<th>{_e(h)}</th>" for h in ["Character", "Labels (lines)", "Episodes", "Note", "Vetoes and checks"])
    body = "".join(
        f'<tr data-q="{_e((r["character"] + " " + r["labels"]).lower())}" data-source="{_e(r["source"].split(" ")[0] if r["source"] else "")}">'
        f"<td><b>{_e(r['character'])}</b><br><span class='muted'>{_e(r['source'])}</span></td><td>{_e(r['lines_per_label'])}</td>"
        f"<td>{_e(r['episodes'])}</td><td>{_e(r['note'])}</td><td class='muted'>{_e(' '.join(x for x in (r['vetoes'], r['checks']) if x))}</td></tr>"
        for r in cast
    )
    return f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def report_html(manifest, findings, name_groups, apart, speaker_groups, verdicts, cast) -> str:
    t = tallies(findings)
    llm = manifest.get("llm", {})
    calls = sum(s.get("calls", 0) for s in llm.values())
    cached = sum(s.get("cached", 0) for s in llm.values())
    model = (manifest.get("model") or ["code only"])[0]
    counts = {d: sum(1 for f in findings if f.detector == d) for d in DETECTOR_TITLES}
    checks = t["by_source"]["check"]
    tiles = "".join(
        f'<div class="tile"><b class="s-{s}">{t["by_severity"][s]}</b><span>{s}</span></div>' for s in SEVERITY_ORDER
    ) + (
        f'<div class="tile"><b>{len(t["episodes"])}</b><span>episodes affected of {manifest.get("counts", {}).get("episodes", 0)}</span></div>'
        f'<div class="tile"><b class="b-check">{checks}</b><span>checks for a person</span></div>'
        f'<div class="tile"><b>{t["with_fix"]}</b><span>fixes ready to approve</span></div>'
    )
    tabs = [
        ("names", f"Names ({counts['names']})", _names_panel(findings, name_groups, apart)),
        ("speakers", f"Speakers ({counts['speakers']})", _speakers_panel(findings, speaker_groups)),
        ("joins", f"Scene joins ({counts['joins']})", _joins_panel(findings)),
        ("spelling", f"Spelling ({counts['spelling']})", _spelling_panel(findings, verdicts)),
        ("format", f"Format ({counts['format']})", _format_panel(findings)),
        ("cast", f"Cast ({len(cast)})", _cast_panel(cast)),
    ]
    nav = "".join(f'<button data-tab="{tid}">{_e(label)}</button>' for tid, label, _ in tabs)
    panels = "".join(f'<section class="panel" id="{tid}">{body}</section>' for tid, _, body in tabs)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Canon report</title><style>{CSS}</style></head><body><header>"
        f"<h1>Canon report <small>{_e(manifest.get('title', ''))}</small></h1>"
        f'<p class="meta">{manifest.get("counts", {}).get("episodes", 0)} episodes · {len(findings)} findings · '
        f"model {_e(model)} · {calls} calls (+{cached} cached) · ${_e(manifest.get('cost_usd', 0))}</p>"
        f'<div class="tiles">{tiles}</div>'
        '<p class="legend">Source: <span class="badge b-rule">rule</span> the script contradicts itself · '
        '<span class="badge b-ai">ai</span> a model judged it, code wrote every location · '
        '<span class="badge b-check">check</span> a person decides · <span class="badge b-code">code</span> code only, not judged.</p>'
        f"</header><main><nav>{nav}</nav>"
        '<div class="tools"><input id="q" type="search" placeholder="Filter by name, label, episode (e22), word or ID">'
        '<label><input id="only" type="checkbox"> only checks</label></div>'
        f"{panels}</main><script>{JS}</script></body></html>"
    )


def write_outputs(out: Path, manifest, findings, name_groups, apart, speaker_groups, verdicts, cast) -> None:
    write_csv(out / "findings.csv", FINDING_COLUMNS, [finding_row(f) for f in findings])
    write_csv(out / "cast.csv", CAST_COLUMNS, cast)
    (out / "summary.md").write_text(summary_md(manifest, findings, name_groups, speaker_groups), encoding="utf-8")
    (out / "report.html").write_text(
        report_html(manifest, findings, name_groups, apart, speaker_groups, verdicts, cast), encoding="utf-8"
    )
