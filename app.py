"""Streamlit front end for Canon: run the pipeline, review findings, approve, download.

The page logic lives in small functions. `build_approvals`, `save_uploads`, `zip_patch`,
`model_kwargs` and `run_totals` are pure (or filesystem-only) so tests cover them without a
browser.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Sequence

import streamlit as st

from canon.config import parse_route
from canon.errors import ErrorInfo, describe
from canon.patch import patch
from canon.pipeline import run
from canon import llm

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "samples" / "ravenscar"
RUN_LOCK = threading.Lock()
MODEL_CHOICES = ["Code only", "Gemini free", "DeepSeek", "Custom"]
SOURCE_COLORS = {"rule": "green", "ai": "blue", "check": "orange", "code": "gray"}
SOURCE_LABELS = {"rule": "rule", "ai": "AI", "check": "check", "code": "code"}
SOURCE_WORDS = {
    "rule": "found by a pattern rule",
    "ai": "found by the AI judge, then checked by code",
    "check": "a person should decide",
    "code": "found by code only, not judged by AI",
}
KEY_HELP = {
    "Code only": "No key is used, and nothing leaves this machine.",
    "Gemini free": (
        "Paste a Google AI Studio key (normally starts with AIza), or leave it empty "
        "to use the server key."
    ),
    "DeepSeek": (
        "Paste a DeepSeek key (normally starts with sk-), or leave it empty to use the "
        "server key."
    ),
    "Custom": "Paste the key for your endpoint, if it needs one. Leave empty to use the server key.",
}
PRIVACY_LINE = (
    "Uploads are processed on this server and, in a model mode, sent to the chosen model "
    "provider; Code only sends nothing."
)
FIRST_RUN = (
    "Canon finds continuity slips before a script goes to voice and video generation."
)
STEPS = [
    ("1", "Pick an input", "The bundled sample, or upload your own `.zip` or `.txt` scripts."),
    ("2", "Pick a model", "**Code only** needs no key and sends nothing. Gemini free costs $0."),
    ("3", "Run Canon", "Watch the stage and its progress. A full 50-episode show with a model takes a while."),
    ("4", "Review", "Each finding shows its location, what was found, and the suggested fix."),
    ("5", "Approve", "Tick the findings you agree with, then download the patched zip."),
]
SLOW_MODEL_NOTE = (
    "A reasoning model can take several minutes on a full show. Each finished call is cached, "
    "so a stopped run resumes where it left off."
)


def key_warning(choice: str, key: str) -> str | None:
    """A hint when a pasted key does not look like the chosen provider's key. Never blocks."""
    value = (key or "").strip()
    if not value or choice == "Code only":
        return None
    if choice == "DeepSeek" and not value.startswith("sk-"):
        return (
            "This does not look like a DeepSeek key (they normally start with sk-). "
            "If it belongs to another provider, choose Custom and enter that provider's base URL."
        )
    if choice == "Gemini free" and not value.startswith("AIza"):
        return (
            "This does not look like a Google AI Studio key (they normally start with AIza). "
            "Check the key, or choose Custom for another provider."
        )
    return None


def _upload_error(exc: Exception) -> ErrorInfo:
    return ErrorInfo(
        "That upload cannot be used",
        str(exc),
        ("Upload either one .zip file or one or more .txt files.",),
    )


def _show_error(info: ErrorInfo) -> None:
    """One short card: what failed, what to do, and the raw reply only on request."""
    st.error(f"**{info.title}**\n\n{info.cause}")
    if info.hints:
        st.markdown("**What to do next**")
        for hint in info.hints:
            st.markdown(f"- {hint}")
    detail = (info.detail or "").strip()
    if detail and detail != (info.cause or "").strip():
        with st.expander("Technical details"):
            st.code(detail, language="text")


def _fail(info: ErrorInfo) -> None:
    st.session_state["canon_run"] = None
    st.session_state["canon_error"] = info.as_dict()
    st.rerun()


def _bridge_secrets() -> None:
    """Community Cloud stores secrets in st.secrets; the pipeline reads environment variables."""
    try:
        secrets = st.secrets
    except Exception:
        return
    for name in (
        "GEMINI_API_KEY",
        "LLM_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        try:
            value = secrets[name]
        except Exception:
            continue
        if value and not os.environ.get(name):
            os.environ[name] = str(value)


_bridge_secrets()


def model_kwargs(choice: str, custom_route: str = "", base_url: str = "") -> dict[str, Any]:
    """The run() arguments for a model choice."""
    if choice == "Code only":
        return {"no_model": True}
    if choice == "Gemini free":
        return {"model": "gemini/gemini-3.5-flash-lite"}
    if choice == "DeepSeek":
        return {"model": "deepseek/deepseek-flash"}
    if choice != "Custom":
        raise ValueError(f"Unknown model choice: {choice}")
    route = custom_route.strip()
    parse_route(route)
    return {"model": route, "base_url": base_url.strip() or None}


def _upload_name(file: Any) -> str:
    return Path(str(getattr(file, "name", ""))).name


def _upload_bytes(file: Any) -> bytes:
    value = file.getvalue() if hasattr(file, "getvalue") else file.read()
    return value.encode("utf-8") if isinstance(value, str) else value


def save_uploads(files: Sequence[Any], destination: str | Path) -> str:
    """Save uploaded scripts outside the repo. Returns a folder or one zip path."""
    if not files:
        raise ValueError("Upload one .zip or one or more .txt files.")
    names = [_upload_name(file) for file in files]
    if any(not name for name in names):
        raise ValueError("Every upload needs a file name.")
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    if any(name.lower().endswith(".zip") for name in names):
        if len(files) != 1 or not names[0].lower().endswith(".zip"):
            raise ValueError("Upload either one .zip or one or more .txt files, not both.")
        path = dest / "upload.zip"
        path.write_bytes(_upload_bytes(files[0]))
        return str(path)
    if not all(name.lower().endswith(".txt") for name in names):
        raise ValueError("Only .zip and .txt scripts are accepted.")
    used: set[str] = set()
    for file, name in zip(files, names):
        stem, suffix = Path(name).stem, Path(name).suffix
        candidate, counter = name, 2
        while candidate in used:
            candidate = f"{stem}-{counter}{suffix}"
            counter += 1
        used.add(candidate)
        (dest / candidate).write_bytes(_upload_bytes(file))
    return str(dest)


def run_totals(manifest: dict) -> dict:
    """Model, call, token and cost totals from a run manifest."""
    llm = manifest.get("llm") or {}
    model = "code only" if manifest.get("no_model") or not manifest.get("model") else manifest["model"][0]
    return {
        "model": model,
        "calls": sum(int(stage.get("calls", 0)) for stage in llm.values()),
        "cached": sum(int(stage.get("cached", 0)) for stage in llm.values()),
        "tokens_in": sum(int(stage.get("input_tokens", 0)) for stage in llm.values()),
        "tokens_out": sum(int(stage.get("output_tokens", 0)) for stage in llm.values()),
        "cost_usd": manifest.get("cost_usd", 0.0),
    }


def build_approvals(
    findings: list[dict],
    checked: dict[str, bool],
    overrides: dict[str, str],
    defaults: dict[str, str],
) -> dict:
    """The approvals.json patch.py reads. Every finding is true or false."""
    approvals: dict[str, Any] = {
        finding["id"]: bool(checked.get(finding["id"], False)) for finding in findings
    }
    canonical = {
        group: value
        for group, value in sorted(overrides.items())
        if value != defaults.get(group)
    }
    if canonical:
        approvals["canonical"] = canonical
    return approvals


def approvals_json(approvals: dict) -> str:
    return json.dumps(approvals, ensure_ascii=False, indent=1)


def zip_patch(run_dir: str | Path) -> bytes:
    """In-memory zip of patched/, patch.diff and patch.json, as patch.py wrote them."""
    run = Path(run_dir)
    patched = run / "patched"
    if not patched.is_dir() or not (run / "patch.diff").is_file() or not (run / "patch.json").is_file():
        raise FileNotFoundError("the patch output is missing")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(patched.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(run).as_posix())
        archive.write(run / "patch.diff", "patch.diff")
        archive.write(run / "patch.json", "patch.json")
    return buffer.getvalue()


def _approve_key(run_id: str, finding_id: str) -> str:
    return f"approve|{run_id}|{finding_id}"


def _location(finding) -> str:
    if finding.ep == 0:
        return f"whole show · {finding.where}"
    scene = f" S{finding.scene:02d}" if finding.scene else ""
    return f"E{finding.ep:02d}{scene} · {finding.start} · {finding.where}"


def _render_finding(finding, run_id: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{finding.id}** · {_location(finding)}")
        st.badge(SOURCE_LABELS.get(finding.source, finding.source), color=SOURCE_COLORS.get(finding.source, "gray"))
        if finding.confidence:
            st.caption(f"confidence: {finding.confidence}")
        if finding.found:
            st.markdown(f"**Found:** {finding.found}")
        if finding.suggestion:
            st.markdown(f"**Suggestion:** {finding.suggestion}")
        if finding.context:
            st.caption(finding.context)
        if finding.source == "check":
            st.info("A person decides this one.")
        st.checkbox("Approve this finding", key=_approve_key(run_id, finding.id))


def _render_group(group, rows, run_id: str, options: list[str], selections: dict, defaults: dict) -> None:
    with st.container(border=True):
        st.subheader(f"{group.canonical} ← {', '.join(o for o in options if o != group.canonical)}")
        st.caption(
            f"Group {group.id} · {SOURCE_WORDS.get(group.source, group.source)} · "
            f"confidence: {group.confidence}"
        )
        st.markdown(f"**Canonical:** {group.canonical}")
        st.markdown(f"**Spellings or labels:** {', '.join(options)}")
        if group.reason:
            st.caption(group.reason)
        canonical_key = f"canonical|{run_id}|{group.id}"
        index = options.index(group.canonical) if group.canonical in options else 0
        selections[group.id] = st.selectbox(
            "Canonical form", options, index=index, key=canonical_key
        )
        defaults[group.id] = group.canonical
        st.caption(
            "Picking a different canonical rewrites every other spelling or label in this "
            "group to it, including the former canonical. Rows left unticked stay as written."
        )
        if st.button("Tick every row in this group", key=f"tick|{run_id}|{group.id}"):
            for finding in rows:
                st.session_state[_approve_key(run_id, finding.id)] = True
        for finding in rows:
            _render_finding(finding, run_id)


def _render_check_rows(rows: list, run_id: str) -> None:
    if not rows:
        return
    st.subheader("For a person to decide")
    for finding in rows:
        _render_finding(finding, run_id)


def _render_findings(rows: list, run_id: str) -> None:
    if not rows:
        st.caption("None found.")
        return
    for finding in rows:
        _render_finding(finding, run_id)


def _names_panel(result, run_id: str, selections: dict, defaults: dict) -> None:
    findings = [f for f in result["findings"] if f.detector == "names"]
    groups = result["name_groups"]
    for group in groups:
        rows = [f for f in findings if f.group == group.id]
        _render_group(group, rows, run_id, group.spellings, selections, defaults)
    _render_check_rows([f for f in findings if f.source == "check"], run_id)
    if result["apart"]:
        st.subheader("Kept apart by the judge")
        for item in result["apart"]:
            st.caption(f"{', '.join(item['spellings'])} ({item['candidate']})")
    if not groups and not result["apart"] and not any(f.source == "check" for f in findings):
        st.caption("None found.")


def _speakers_panel(result, run_id: str, selections: dict, defaults: dict) -> None:
    findings = [f for f in result["findings"] if f.detector == "speakers"]
    for group in result["speaker_groups"]:
        rows = [f for f in findings if f.group == group.id]
        _render_group(group, rows, run_id, group.labels, selections, defaults)
    _render_check_rows([f for f in findings if f.source == "check"], run_id)
    if not result["speaker_groups"] and not any(f.source == "check" for f in findings):
        st.caption("None found.")


def _render_run(state: dict) -> None:
    result = state["result"]
    manifest = state["manifest"]
    findings = result["findings"]
    run_id = state["run_id"]
    counts = manifest.get("counts", {})
    severity = counts.get("by_severity", {})

    columns = st.columns(5)
    for column, (key, label) in zip(
        columns,
        [("audible", "Audible"), ("voice", "Voice"), ("visible", "Visible"), ("notes", "Notes")],
    ):
        column.metric(label, int(severity.get(key, 0)))
    columns[4].metric("Episodes affected", int(counts.get("episodes_affected", 0)))

    totals = run_totals(manifest)
    cost = "unknown" if totals["cost_usd"] is None else f"${totals['cost_usd']}"
    st.caption(
        f"Model: {totals['model']} · calls: {totals['calls']} (+{totals['cached']} cached) · "
        f"tokens: {totals['tokens_in']} in / {totals['tokens_out']} out · "
        f"cost: {cost}"
    )

    st.caption(
        "Each card shows what code found, what Canon suggests, and the quote it used. "
        "Nothing is changed until you tick Approve and build the patched zip."
    )
    selections: dict[str, str] = {}
    defaults: dict[str, str] = {}
    names_tab, speakers_tab, joins_tab, spelling_tab, format_tab = st.tabs(
        ["Names", "Speakers", "Scene joins", "Spelling", "Format"]
    )
    with names_tab:
        _names_panel(result, run_id, selections, defaults)
    with speakers_tab:
        _speakers_panel(result, run_id, selections, defaults)
    with joins_tab:
        _render_findings([f for f in findings if f.detector == "joins"], run_id)
    with spelling_tab:
        _render_findings([f for f in findings if f.detector == "spelling"], run_id)
    with format_tab:
        _render_findings([f for f in findings if f.detector == "format"], run_id)

    checked = {
        finding.id: bool(st.session_state.get(_approve_key(run_id, finding.id), False))
        for finding in findings
    }
    approvals = build_approvals(
        [finding.model_dump() for finding in findings], checked, selections, defaults
    )

    st.divider()
    st.subheader("Downloads")
    run_dir = Path(state["run_dir"])
    download_columns = st.columns(4)
    download_columns[0].download_button(
        "findings.csv",
        (run_dir / "findings.csv").read_bytes(),
        file_name="findings.csv",
        mime="text/csv",
        key=f"dl_findings|{run_id}",
    )
    download_columns[1].download_button(
        "cast.csv",
        (run_dir / "cast.csv").read_bytes(),
        file_name="cast.csv",
        mime="text/csv",
        key=f"dl_cast|{run_id}",
    )
    download_columns[2].download_button(
        "report.html",
        (run_dir / "report.html").read_bytes(),
        file_name="report.html",
        mime="text/html",
        key=f"dl_report|{run_id}",
    )
    download_columns[3].download_button(
        "approvals.json",
        approvals_json(approvals).encode("utf-8"),
        file_name="approvals.json",
        mime="application/json",
        key=f"dl_approvals|{run_id}",
    )

    patch_key = f"patch|{run_id}"
    if st.button("Build patched zip", key=f"build_patch|{run_id}"):
        approvals_path = run_dir / "approvals.json"
        approvals_path.write_text(approvals_json(approvals), encoding="utf-8")
        try:
            stats = patch(str(run_dir), str(approvals_path))
            data = zip_patch(run_dir)
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
        else:
            st.session_state[patch_key] = {
                "stats": stats,
                "zip": data,
                "digest": hashlib.sha256(approvals_json(approvals).encode()).hexdigest(),
            }
    packed = st.session_state.get(patch_key)
    if packed:
        stats = packed["stats"]
        st.write(
            f"Patch: approved {stats.get('approved', 0)} · rejected {stats.get('rejected', 0)} · "
            f"overrides {stats.get('overrides', 0)} · edits {stats.get('edits', 0)} · "
            f"applied {stats.get('applied', 0)} · report only {stats.get('report_only', 0)} · "
            f"files changed {stats.get('files_changed', 0)}"
        )
        current = hashlib.sha256(approvals_json(approvals).encode()).hexdigest()
        if packed["digest"] != current:
            st.warning("Approvals changed since this zip was built. Build it again.")
        problems = stats.get("problems") or []
        if problems:
            st.warning(f"{len(problems)} problem(s) while patching")
            for problem in problems:
                st.caption(problem)
        st.download_button(
            "patched.zip (patched/, patch.diff, patch.json)",
            packed["zip"],
            file_name="canon_patched.zip",
            mime="application/zip",
            key=f"dl_patch|{run_id}",
        )
    st.caption("The patch writes a copy in the run folder; the input scripts are never changed.")


def _start_run(input_mode, uploads, model_choice, custom_route, custom_base, api_key) -> None:
    api_key = (api_key or "").strip()
    try:
        kwargs = model_kwargs(model_choice, custom_route, custom_base)
    except ValueError as exc:
        _fail(describe(exc))
        return

    if input_mode == "Try the sample show":
        input_path = str(SAMPLE)
    else:
        if not uploads:
            _fail(
                ErrorInfo(
                    "No scripts were uploaded",
                    "Upload one .zip file or one or more .txt files in the sidebar.",
                    ("Only .zip and .txt scripts are accepted.",),
                )
            )
            return
        folder = tempfile.mkdtemp(prefix="canon_input_")
        try:
            input_path = save_uploads(uploads, folder)
        except Exception as exc:
            shutil.rmtree(folder, ignore_errors=True)
            _fail(_upload_error(exc))
            return

    out_dir = tempfile.mkdtemp(prefix="canon_run_")
    key_env = st.session_state.get("canon_key_env")
    if not key_env:
        key_env = f"CANON_APP_KEY_{uuid.uuid4().hex}"
        st.session_state["canon_key_env"] = key_env
    api_key_env = key_env if model_choice != "Code only" and api_key else None
    if api_key_env:
        os.environ[api_key_env] = api_key

    previous = st.session_state.get("canon_run")
    if previous:
        st.session_state.pop(f"patch|{previous['run_id']}", None)
    problem = None
    started = time.time()
    try:
        with st.status("Running Canon...", expanded=True) as status:
            line = st.empty()
            bar = st.progress(0.0)

            def elapsed() -> str:
                return f"{int(time.time() - started)}s"

            def progress(stage: str, done: int, total: int) -> None:
                if total:
                    bar.progress(min(1.0, done / total))
                    line.write(f"**{stage}** · {done} of {total} done · {elapsed()}")
                else:
                    line.write(f"**{stage}...** · {elapsed()}")

            llm.set_progress(progress)
            try:
                with RUN_LOCK:
                    result = run(input_path, out_dir, api_key_env=api_key_env, **kwargs)
            finally:
                llm.set_progress(None)
            status.update(
                label=f"Done in {int(time.time() - started)}s", state="complete", expanded=False
            )
    except Exception as exc:
        problem = describe(exc)
    finally:
        if api_key_env:
            os.environ.pop(api_key_env, None)

    if problem is not None:
        _fail(problem)
        return
    if not result["manifest"].get("complete"):
        manifest = result["manifest"]
        _fail(
            ErrorInfo(
                manifest.get("error_title") or "The run did not finish",
                manifest.get("error") or "The run stopped before it finished.",
                tuple(
                    manifest.get("error_hints")
                    or ("Run Canon again. Finished model calls are cached, so a retry skips them.",)
                ),
                manifest.get("error_detail") or "",
            )
        )
        return

    st.session_state["canon_error"] = None
    st.session_state["canon_run"] = {
        "result": result,
        "manifest": result["manifest"],
        "run_dir": out_dir,
        "input_path": input_path,
        "run_id": uuid.uuid4().hex,
    }


def _sidebar() -> None:
    with st.sidebar:
        st.title("Canon")
        st.caption("Continuity check for AI microdrama scripts.")
        input_mode = st.radio(
            "Input", ["Try the sample show", "Upload scripts"], key="input_mode"
        )
        uploads = []
        if input_mode == "Upload scripts":
            uploads = st.file_uploader(
                "Scripts", type=["zip", "txt"], accept_multiple_files=True, key="uploads"
            )
        st.caption(PRIVACY_LINE)
        model_choice = st.radio("Model", MODEL_CHOICES, index=0, key="model_choice")
        custom_route, custom_base = "", ""
        if model_choice == "Custom":
            custom_route = st.text_input(
                "Provider/name", placeholder="openai/your-model", key="custom_route"
            )
            custom_base = st.text_input("Base URL (optional)", key="custom_base_url")
        api_key = st.text_input(
            "API key (optional)",
            type="password",
            key="api_key",
            help=KEY_HELP[model_choice],
        )
        warning = key_warning(model_choice, api_key)
        if warning:
            st.warning(warning)
        if model_choice != "Code only":
            st.caption(SLOW_MODEL_NOTE)
        if st.button("Run Canon", key="run_button", type="primary"):
            _start_run(input_mode, uploads, model_choice, custom_route, custom_base, api_key)


def main() -> None:
    st.set_page_config(page_title="Canon", layout="wide")
    _sidebar()
    error = st.session_state.get("canon_error")
    if error:
        info = (
            ErrorInfo.from_dict(error)
            if isinstance(error, dict)
            else ErrorInfo("Canon stopped", str(error))
        )
        st.title("Canon")
        _show_error(info)
        st.caption("No scripts were changed. Fix the item above, then run Canon again.")
        return
    state = st.session_state.get("canon_run")
    if not state:
        st.title("Canon")
        st.write(FIRST_RUN)
        st.subheader("How to use it")
        for number, title, body in STEPS:
            st.markdown(f"**{number}. {title}.** {body}")
        st.divider()
        st.subheader("What Canon looks for")
        st.markdown(
            "- **Names:** one person, family or place spelled more than one way.\n"
            "- **Speakers:** one character under several labels, or one label for two people.\n"
            "- **Scene joins:** a day/night or building change across a cut the notes call continuous.\n"
            "- **Spelling:** misspellings inside spoken lines.\n"
            "- **Format:** a header label written differently from the rest of the show."
        )
        st.info(SLOW_MODEL_NOTE)
        return
    _render_run(state)


if __name__ == "__main__":
    main()
