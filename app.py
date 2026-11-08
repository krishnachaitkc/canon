"""Streamlit front end for Canon: pick a script, check it, decide, download the fixed copy.

`build_approvals`, `zip_patch`, `zip_fixed`, `save_uploads`, `model_kwargs`, `run_totals`
and `key_warning` are pure (or filesystem-only) so tests cover them without a browser.
"""

from __future__ import annotations

import csv
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

from canon import llm
from canon.config import parse_route
from canon.errors import ErrorInfo, describe
from canon.issues import (
    KIND_TITLES,
    LANE_RULES,
    LANE_TITLES,
    LANES,
    SEVERITY_WHERE,
    WHERE_LABELS,
    apart_lines,
    build_cards,
    by_hand_rows,
    change_rows,
    check_places,
    decided_count,
    download_summary,
    issue_rows,
    metrics,
    progress_text,
    ready_places,
    safe_text,
    status,
)
from canon.patch import patch
from canon.parse import parse_show
from canon.pipeline import run

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "samples" / "ravenscar"
RUN_LOCK = threading.Lock()
PAGES = ["Home", "Review", "Download"]
MODEL_CHOICES = ["No AI", "Gemini", "DeepSeek", "Other model"]
MODEL_ALIASES = {"No AI": "Code only", "Gemini": "Gemini free", "Other model": "Custom"}
HAND_OPTIONS = ["I'll fix it by hand", "Not a problem"]
HAND_VALUES = {"I'll fix it by hand": "by_hand", "Not a problem": "not_a_problem"}
KIND_BY_LABEL = {label: kind for kind, label in KIND_TITLES.items()}
PRIVACY_LINE = "Files go only to the AI you choose. No AI sends nothing."
HOME_LINE = (
    "Canon finds script mistakes that AI voice and video would turn into errors viewers "
    "hear and see. Run it before you generate."
)
SLOW_MODEL_NOTE = (
    "A reasoning model can take several minutes on a full show. Each finished call is cached, "
    "so a stopped run resumes where it left off."
)
KEY_HELP = {
    "No AI": "No key is used, and nothing leaves this machine.",
    "Gemini": (
        "Paste a Google AI Studio key (normally starts with AIza), or leave it empty "
        "to use the server key."
    ),
    "DeepSeek": (
        "Paste a DeepSeek key (normally starts with sk-), or leave it empty to use the "
        "server key."
    ),
    "Other model": "Paste the key for your endpoint, if it needs one. Leave empty to use the server key.",
}
MODEL_CAPTIONS = [
    "Instant · free · sends nothing",
    "Free · about 1 min per 50 episodes",
    "Best results · needs your key",
    "Any OpenAI-compatible model",
]
SCRIPT_CAPTIONS = ["The bundled sample show", "Your .zip or .txt files"]
FILES_HELP = [
    "One `.zip` of episode files, or one or more `.txt` files.",
    "One episode per file. The number comes from `EPISODE NUMBER:` or the file name, "
    "e.g. `episode-001-....txt`.",
    "The script format: `EPISODE LOGLINE:`, `CONTINUITY NOTES:`, `--- SCRIPT START ---`, "
    "then per scene `SCENE 01`, `Scene heading: EXT. MOOR ROAD – NIGHT`, "
    "`Scene start:`/`Scene end:`, then beats such as `**CAMERA:**` or "
    "`**EDMUND (Tense):**` with `Start: (00:05.00) | End: (00:09.00)` and the text.",
]
SAMPLE_LINES = """EPISODE NUMBER: Episode 1
EPISODE LOGLINE: On a storm-dark moor road, a stranger asks the way ...
CONTINUITY NOTES: First episode. Establishes Edmund Thorne's return ...
--- SCRIPT START ---
SCENE 01
Scene heading: EXT. MOOR ROAD – NIGHT
Scene start: 00:00.00
Scene end: 00:38.00
**CAMERA:**
Start: (00:00.00) | End: (00:05.00)
Wide shot of a low stone cottage on the black moor ..."""
WHAT_IT_FINDS = [
    {"Kind": "Name", "Example": "Ravenscourt → Ravenscar", "Fixed for you?": "yes"},
    {"Kind": "Voice label", "Example": "LORD EDMUND → EDMUND", "Fixed for you?": "yes"},
    {"Kind": "Time of day", "Example": "NIGHT → DAY across one moment", "Fixed for you?": "yes"},
    {"Kind": "Place", "Example": "cabin vs cottage in one moment", "Fixed for you?": "you fix it"},
    {"Kind": "Spoken typo", "Example": "seizzure → seizure", "Fixed for you?": "yes"},
    {"Kind": "Format", "Example": "SCENE HEADING vs Scene heading", "Fixed for you?": "you fix it"},
]


def key_warning(choice: str, key: str) -> str | None:
    """A hint when a pasted key does not look like the chosen provider's key. Never blocks."""
    value = (key or "").strip()
    choice = MODEL_ALIASES.get(choice, choice)
    if not value or choice == "Code only":
        return None
    if choice == "DeepSeek" and not value.startswith("sk-"):
        return (
            "This does not look like a DeepSeek key (they normally start with sk-). "
            "If it belongs to another provider, choose Other model and enter that provider's base URL."
        )
    if choice == "Gemini free" and not value.startswith("AIza"):
        return (
            "This does not look like a Google AI Studio key (they normally start with AIza). "
            "Check the key, or choose Other model for another provider."
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


def _set_progress(callback) -> None:
    """Register the live progress callback, on any build of the package.

    Streamlit reruns app.py from disk but keeps imported modules in memory, so a deploy
    can pair this file with an older `canon.llm` for a while. Progress is cosmetic; the
    run itself must never fail because the function is missing.
    """
    setter = getattr(llm, "set_progress", None)
    if setter is not None:
        setter(callback)


def _fail(info: ErrorInfo) -> None:
    st.session_state["canon_run"] = None
    st.session_state["canon_error"] = info.as_dict()
    st.session_state["goto"] = "Home"
    st.rerun()


def _go(page: str) -> None:
    st.session_state["goto"] = page
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
    choice = MODEL_ALIASES.get(choice, choice)
    if choice == "Code only":
        return {"no_model": True}
    if choice == "Recorded AI":
        # The sample show's AI answers, recorded from DeepSeek and replayed: instant, free, offline.
        return {"model": "fake/recorded"}
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


def friendly_model(route: str) -> str:
    """A model name a person reading the screen understands."""
    value = str(route or "")
    if value.startswith("deepseek/"):
        return "DeepSeek"
    if value.startswith("gemini/"):
        return "Gemini"
    if value.startswith("fake/"):
        return "recorded AI answers"
    return value


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


def _changes_csv(changes: Sequence[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["episode", "time", "where", "was", "now", "issue"])
    for row in changes:
        ep = int(row.get("episode") or 0)
        writer.writerow(
            [
                f"E{ep:02d}" if ep else "whole show",
                row.get("time", ""),
                row.get("where", ""),
                row.get("was", ""),
                row.get("now", ""),
                row.get("issue", ""),
            ]
        )
    return buffer.getvalue()


def _by_hand_txt(rows: Sequence[dict]) -> str:
    lines = ["These places need a hand edit. The original files are not changed."]
    for row in rows:
        ep = int(row.get("episode") or 0)
        where = row.get("where", "")
        label = f"E{ep:02d}" if ep else "whole show"
        lines.append(f"{label} · {where} · {row.get('title', '')}")
    return "\n".join(lines) + "\n"


def zip_fixed(
    run_dir: str | Path,
    changes: Sequence[dict] | None = None,
    by_hand: Sequence[dict] | None = None,
) -> bytes:
    """The download: fixed_scripts/, CHANGES.csv, changes.diff and fix_by_hand.txt."""
    run = Path(run_dir)
    patched = run / "patched"
    if not patched.is_dir() or not (run / "patch.diff").is_file():
        raise FileNotFoundError("the fixed scripts are missing")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(patched.rglob("*")):
            if path.is_file():
                archive.write(path, f"fixed_scripts/{path.relative_to(patched).as_posix()}")
        archive.writestr("CHANGES.csv", _changes_csv(changes or []))
        archive.write(run / "patch.diff", "changes.diff")
        archive.writestr("fix_by_hand.txt", _by_hand_txt(by_hand or []))
    return buffer.getvalue()


def _decisions_digest(approvals: dict, changes: Sequence[dict], by_hand: Sequence[dict]) -> str:
    payload = json.dumps(
        {"approvals": approvals, "changes": list(changes), "by_hand": list(by_hand)},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sample_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(SAMPLE.glob("*.txt")):
            archive.write(path, path.name)
    return buffer.getvalue()


def _fix_key(run_id: str, finding_id: str) -> str:
    return f"fix|{run_id}|{finding_id}"


def _hand_key(run_id: str, card_id: str) -> str:
    return f"hand|{run_id}|{card_id}"


def _spell_key(run_id: str, group_id: str) -> str:
    return f"spelling|{run_id}|{group_id}"


def _fix_widget(run_id: str, finding_id: str) -> str:
    return f"wfix|{run_id}|{finding_id}"


def _hand_widget(run_id: str, card_id: str) -> str:
    return f"whand|{run_id}|{card_id}"


def _spell_widget(run_id: str, group_id: str) -> str:
    return f"wspell|{run_id}|{group_id}"


def _sync_fix(run_id: str, finding_id: str) -> None:
    st.session_state[_fix_key(run_id, finding_id)] = bool(
        st.session_state.get(_fix_widget(run_id, finding_id))
    )


def _sync_choice(run_id: str, group_id: str) -> None:
    value = st.session_state.get(_spell_widget(run_id, group_id))
    if value:
        st.session_state[_spell_key(run_id, group_id)] = value


def _sync_hand(run_id: str, card_id: str) -> None:
    value = st.session_state.get(_hand_widget(run_id, card_id))
    if value in HAND_OPTIONS:
        st.session_state[_hand_key(run_id, card_id)] = value


def _set_fixes(run_id: str, finding_ids: Sequence[str], value: bool) -> None:
    for finding_id in finding_ids:
        st.session_state[_fix_key(run_id, finding_id)] = bool(value)
        st.session_state[_fix_widget(run_id, finding_id)] = bool(value)


def _prefixed(prefix: str) -> dict[str, Any]:
    return {
        str(key)[len(prefix):]: st.session_state[key]
        for key in list(st.session_state)
        if str(key).startswith(prefix)
    }


def _fix_map(run_id: str) -> dict[str, bool]:
    return {key: bool(value) for key, value in _prefixed(f"fix|{run_id}|").items()}


def _hand_map(run_id: str) -> dict[str, str]:
    return {
        key: HAND_VALUES[value]
        for key, value in _prefixed(f"hand|{run_id}|").items()
        if value in HAND_VALUES
    }


def _choices(run_id: str, result: dict) -> dict[str, str]:
    choices = {}
    for group in list(result.get("name_groups") or []) + list(result.get("speaker_groups") or []):
        choices[group.id] = st.session_state.get(_spell_key(run_id, group.id), group.canonical)
    return choices


def _clear_decisions(run_id: str) -> None:
    for key in [str(key) for key in list(st.session_state)]:
        if f"|{run_id}|" in key and key.split("|", 1)[0] in (
            "fix",
            "hand",
            "spelling",
            "wfix",
            "whand",
            "wspell",
            "fixall",
            "details",
            "dialog",
        ):
            del st.session_state[key]


def _cut(text: str, limit: int = 70) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _episodes_text(episodes: tuple[int, ...]) -> str:
    shown = [f"E{ep}" for ep in episodes[:4]]
    if len(episodes) > 4:
        shown.append(f"+{len(episodes) - 4}")
    return ", ".join(shown)


def _episode_ranges(episodes: tuple[int, ...]) -> str:
    """'E1–3, E21–24' for the download summary."""
    numbers = sorted({int(ep) for ep in episodes if ep})
    if not numbers:
        return ""
    parts = []
    start = previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        parts.append(f"E{start}" if start == previous else f"E{start}–{previous}")
        start = previous = number
    parts.append(f"E{start}" if start == previous else f"E{start}–{previous}")
    return ", ".join(parts)


def _start_run(
    input_mode: str,
    uploads,
    model_choice: str,
    custom_route: str,
    custom_base: str,
    api_key: str,
) -> bool:
    api_key = (api_key or "").strip()
    try:
        kwargs = model_kwargs(model_choice, custom_route, custom_base)
    except ValueError as exc:
        _fail(describe(exc))
        return False

    label = "Sample show"
    if input_mode == "Sample show":
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
            return False
        folder = tempfile.mkdtemp(prefix="canon_input_")
        try:
            input_path = save_uploads(uploads, folder)
        except Exception as exc:
            shutil.rmtree(folder, ignore_errors=True)
            _fail(_upload_error(exc))
            return False
        label = _upload_name(uploads[0])

    out_dir = tempfile.mkdtemp(prefix="canon_run_")
    key_env = st.session_state.get("canon_key_env")
    if not key_env:
        key_env = f"CANON_APP_KEY_{uuid.uuid4().hex}"
        st.session_state["canon_key_env"] = key_env
    api_key_env = key_env if model_choice != "No AI" and api_key else None
    if api_key_env:
        os.environ[api_key_env] = api_key

    previous = st.session_state.get("canon_run")
    if previous:
        _clear_decisions(previous["run_id"])
        st.session_state.pop(f"fixed|{previous['run_id']}", None)
    problem = None
    started = time.time()
    try:
        with st.status("Checking the script...", expanded=True) as box:
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

            _set_progress(progress)
            try:
                with RUN_LOCK:
                    result = run(input_path, out_dir, api_key_env=api_key_env, **kwargs)
            finally:
                _set_progress(None)
            box.update(
                label=f"Done in {int(time.time() - started)}s", state="complete", expanded=False
            )
    except Exception as exc:
        problem = describe(exc)
    finally:
        if api_key_env:
            os.environ.pop(api_key_env, None)

    if problem is not None:
        _fail(problem)
        return False
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
        return False

    input_stats = ""
    try:
        show = parse_show(input_path)
        scenes = sum(len(episode.scenes) for episode in show.episodes)
        spoken = sum(
            1 for episode in show.episodes for beat in episode.beats if beat.kind == "line"
        )
        input_stats = (
            f"Read {len(show.episodes)} episodes · {spoken:,} spoken lines · {scenes} scenes."
        )
        if show.unconsumed_lines:
            input_stats += f" {show.unconsumed_lines} lines were not understood."
    except Exception:
        input_stats = f"Read {int(result['manifest'].get('counts', {}).get('episodes', 0))} episodes."

    st.session_state["canon_error"] = None
    st.session_state["canon_run"] = {
        "result": result,
        "manifest": result["manifest"],
        "run_dir": out_dir,
        "input_path": input_path,
        "run_id": uuid.uuid4().hex,
        "label": label,
        "checker": model_choice,
        "input_stats": input_stats,
    }
    _go("Review")
    return True


def _files_help() -> None:
    for line in FILES_HELP:
        st.markdown(f"- {line}")


def _sample_lines() -> str:
    path = SAMPLE / "episode-001-the-road.txt"
    try:
        return "\n".join(path.read_text(encoding="utf-8").splitlines()[:10])
    except OSError:
        return SAMPLE_LINES


def _sidebar() -> None:
    with st.sidebar:
        st.title("Canon")
        st.caption("Script continuity checker")

        st.markdown("**1 Script**")
        input_mode = st.radio(
            "Script",
            ["Sample show", "My files"],
            captions=SCRIPT_CAPTIONS,
            key="input_mode",
            label_visibility="collapsed",
        )
        uploads = []
        if input_mode == "My files":
            uploads = st.file_uploader(
                "Upload .zip or .txt", type=["zip", "txt"], accept_multiple_files=True, key="uploads"
            )
        with st.popover("What files work?"):
            _files_help()

        st.markdown("**2 Check with**")
        model_choice = st.radio(
            "Check with",
            MODEL_CHOICES,
            captions=MODEL_CAPTIONS,
            key="model_choice",
            label_visibility="collapsed",
        )
        custom_route, custom_base, api_key = "", "", ""
        if model_choice == "Other model":
            custom_route = st.text_input(
                "Model name", placeholder="openai/your-model", key="custom_route"
            )
            custom_base = st.text_input("Base URL", key="custom_base_url")
        if model_choice != "No AI":
            api_key = st.text_input(
                "API key",
                type="password",
                key="api_key",
                help=KEY_HELP[model_choice],
            )
            warning = key_warning(model_choice, api_key)
            if warning:
                st.warning(warning)
            st.caption(SLOW_MODEL_NOTE)

        if st.button("Check script", key="run_button", type="primary", width="stretch"):
            _start_run(input_mode, uploads, model_choice, custom_route, custom_base, api_key)

        state = st.session_state.get("canon_run")
        if state and state.get("manifest", {}).get("complete"):
            cards = build_cards(state["result"])
            fix = _fix_map(state["run_id"])
            hand = _hand_map(state["run_id"])
            decided = decided_count(cards, fix, hand)
            st.divider()
            st.caption(
                f"✓ {state.get('label', 'Sample show')} · {state.get('checker', 'No AI')} · "
                f"{len(cards)} issues · {decided} decided"
            )
            if st.button("Clear", key="clear_run"):
                _clear_decisions(state["run_id"])
                st.session_state["canon_run"] = None
                st.session_state["canon_error"] = None
                _go("Home")
        st.caption(PRIVACY_LINE)


def _empty_state(page: str) -> None:
    st.subheader("Check a script first")
    st.write("Pick a script in the sidebar, then press Check script.")
    if st.button("Back to Home", key=f"empty|{page}"):
        _go("Home")


def _home() -> None:
    st.title("Canon")
    st.write(HOME_LINE)
    if st.button("Try the sample now", key="sample_now", type="primary"):
        _start_run("Sample show", [], "Recorded AI", "", "", "")
    st.caption(
        "Checks the 4-episode sample with AI answers recorded earlier: instant and free. "
        "To check your own script, use the sidebar."
    )

    st.divider()
    st.subheader("Why it matters")
    columns = st.columns(3)
    points = [
        (
            "Heard",
            "The voice reads every word as written. 'Ravenscourt' is said aloud, even when "
            "the place is Ravenscar.",
        ),
        (
            "Voice",
            "Every speaker label gets its own voice. LORD EDMUND and EDMUND become two voices "
            "for one person.",
        ),
        (
            "Seen",
            "Video is drawn from the scene heading. NIGHT, then DAY in the same moment, gives "
            "two different pictures.",
        ),
    ]
    for column, (title, body) in zip(columns, points):
        with column, st.container(border=True):
            st.subheader(title)
            st.write(body)

    st.subheader("How it works")
    steps = [
        ("1", "Pick a script", "The sample show, a .zip, or one or more .txt files."),
        ("2", "Check", "Code finds every case, AI judges, code double-checks."),
        ("3", "Decide", "One click per issue, or Fix all ready."),
        ("4", "Download", "Get the fixed copy. Your files are never changed."),
    ]
    columns = st.columns(4)
    for column, (number, title, body) in zip(columns, steps):
        with column, st.container(border=True):
            st.markdown(f"**{number}. {title}**")
            st.caption(body)

    st.subheader("What it finds")
    st.dataframe(WHAT_IT_FINDS, hide_index=True, width="stretch")

    st.subheader("What files work")
    _files_help()
    st.code(_sample_lines(), language="text")
    st.download_button(
        "Download the sample show (.zip)",
        _sample_zip(),
        file_name="ravenscar-sample.zip",
        mime="application/zip",
        key="dl_sample_zip",
    )

    st.divider()
    st.caption("It does not check grammar, plot facts, costumes or props.")
    st.caption("AI can be wrong, so you decide every fix.")


def _lane_guide() -> None:
    """Three small boxes that explain the lanes before the cards."""
    columns = st.columns(3)
    for column, lane in zip(columns, LANES):
        with column, st.container(border=True):
            st.markdown(f"**{LANE_TITLES[lane]}:** {LANE_RULES[lane]}")


def _lane_header(lane: str, cards: list, run_id: str) -> None:
    st.subheader(f"{LANE_TITLES[lane]} ({len(cards)})")
    st.caption(LANE_RULES[lane])
    if lane == "check":
        places = check_places(cards)
        if st.button(
            f"Fix these too ({len(places)} places)",
            key=f"fixcheck|{run_id}",
            disabled=not places,
        ):
            _set_fixes(run_id, [fid for card in cards for fid in card.fixable], True)
            st.rerun()


def _draw_card(card, run_id: str, fix: dict, hand: dict) -> None:
    with st.container(border=True):
        top = st.columns([3, 2])
        where = card.where_label.lower()
        top[0].caption(card.kind_label if where in card.kind_label.lower() else f"{card.kind_label} · {where}")
        chip = status(card, fix, hand)
        color = "gray"
        if chip.startswith("Fixing"):
            color = "green"
        elif chip in ("Keep as written", "Not a problem"):
            color = "blue"
        elif chip == "Will fix by hand":
            color = "orange"
        top[1].badge(chip, color=color)
        st.markdown(f"**{card.title}**")
        st.markdown(f"**Problem:** {card.problem}")
        st.markdown(f"**{_why_label(card)}:** {card.why}")
        count = f"{len(card.places)} place{'s' if len(card.places) != 1 else ''}"
        episodes = _episodes_text(card.episodes)
        st.caption(f"{count} · {episodes}" if episodes else count)
        example = next((item for item in card.findings if item.context), None)
        if example is not None and card.kind in ("time", "place"):
            st.caption("Headings: " + _cut(example.context))
        elif example is not None:
            change = ""
            if example.edit is not None and example.edit.new != example.edit.old:
                change = f"  → {example.edit.new}"
            if card.kind == "voice" and example.edit is not None:
                st.caption(f"{example.found}: {_cut(example.context)}  → label {example.edit.new}")
            else:
                st.caption(_cut(example.context) + change)
        st.caption(card.how_found)
        buttons = st.columns(2)
        if card.fixable:
            all_on = all(fix.get(fid) for fid in card.fixable)
            label = "✓ Fixing · Undo" if all_on else f"Fix all {len(card.fixable)}"
            if buttons[0].button(label, key=f"fixall|{run_id}|{card.id}"):
                _set_fixes(run_id, card.fixable, not all_on)
                st.rerun()
        if buttons[1].button("Details", key=f"details|{run_id}|{card.id}"):
            _details(card)


def _why_label(card) -> str:
    return "What to do" if card.lane == "hand" else "Why this fix"


@st.dialog("Details", width="large")
def _details(card) -> None:
    state = st.session_state.get("canon_run")
    if not state:
        return
    run_id = state["run_id"]
    st.subheader(card.title)
    st.markdown(f"**Problem:** {card.problem}")
    st.markdown(f"**{_why_label(card)}:** {card.why}")

    if card.options and card.group_id:
        current = st.session_state.get(_spell_key(run_id, card.group_id), card.default)
        index = card.options.index(current) if current in card.options else 0
        label = "Use this spelling:" if card.kind == "name" else "Use this label:"
        chosen = st.selectbox(
            label,
            list(card.options),
            index=index,
            key=_spell_widget(run_id, card.group_id),
            on_change=_sync_choice,
            args=(run_id, card.group_id),
        )
        if chosen != card.default:
            st.info(
                f"Places already written '{card.default}' will also change to '{chosen}'."
            )

    fix = _fix_map(run_id)
    if card.fixable:
        for finding in card.findings:
            if finding.id not in card.fixable:
                continue
            row = st.columns([1, 4])
            row[0].checkbox(
                "Fix",
                value=bool(fix.get(finding.id)),
                key=_fix_widget(run_id, finding.id),
                on_change=_sync_fix,
                args=(run_id, finding.id),
            )
            change = f"{finding.edit.old} → {finding.edit.new}" if finding.edit else ""
            where_word = WHERE_LABELS[SEVERITY_WHERE.get(finding.severity, "notes")]
            row[1].caption(
                f"E{finding.ep:02d} · {finding.start} · {where_word} · {finding.where} · "
                f"{_cut(finding.context, 80)} {change}".strip()
            )
        all_col, none_col = st.columns(2)
        all_col.button(
            "Fix all",
            key=f"dialog_all|{run_id}|{card.id}",
            on_click=_set_fixes,
            args=(run_id, card.fixable, True),
        )
        none_col.button(
            "Fix none",
            key=f"dialog_none|{run_id}|{card.id}",
            on_click=_set_fixes,
            args=(run_id, card.fixable, False),
        )

    if card.lane == "hand" and not card.fixable:
        for finding in card.findings:
            where_word = WHERE_LABELS[SEVERITY_WHERE.get(finding.severity, "notes")]
            change = ""
            if finding.edit is not None:
                change = f" · {finding.edit.old} → {finding.edit.new}"
            elif finding.suggestion:
                change = f" · {finding.found} → {finding.suggestion}"
            st.caption(
                f"E{finding.ep:02d} · {finding.start} · {where_word} · {finding.where}{change}"
            )
        current = st.session_state.get(_hand_key(run_id, card.id))
        choice = st.radio(
            "What will you do?",
            HAND_OPTIONS,
            index=HAND_OPTIONS.index(current) if current in HAND_OPTIONS else None,
            key=_hand_widget(run_id, card.id),
            on_change=_sync_hand,
            args=(run_id, card.id),
        )
        if choice is None:
            st.caption("Pick one so Review can show this issue as decided.")

    sources = {finding.source for finding in card.findings}
    if card.reason and sources & {"ai", "check"}:
        st.caption("AI's reason")
        st.write(card.reason)
    ids = [card.group_id, " ".join(card.places)]
    beat = next((finding.where for finding in card.findings if finding.where), "")
    if beat:
        ids.append(beat)
    st.caption(" · ".join(part for part in ids if part))
    if st.button("Done", key=f"done|{run_id}|{card.id}", type="primary"):
        st.rerun()


def _visible_cards(cards: list, query: str, kinds: list[str], hide: bool, fix: dict, hand: dict) -> list:
    shown = []
    for card in cards:
        if kinds and card.kind not in kinds:
            continue
        if hide and (status(card, fix, hand) != "Not decided"):
            continue
        if query:
            haystack = " ".join(
                [card.title, card.sentence, card.kind_label, card.how_found]
                + [f"E{ep}" for ep in card.episodes]
            ).lower()
            if query.lower() not in haystack:
                continue
        shown.append(card)
    return shown


def _review(state: dict) -> None:
    result = state["result"]
    run_id = state["run_id"]
    cards = build_cards(result)
    fix = _fix_map(run_id)
    hand = _hand_map(run_id)
    counts = metrics(cards)

    columns = st.columns(5)
    for column, (label, value) in zip(
        columns,
        [
            ("Issues", counts["issues"]),
            ("Heard", counts["heard"]),
            ("Voice", counts["voice"]),
            ("Seen", counts["seen"]),
            ("Episodes", counts["episodes"]),
        ],
    ):
        column.metric(label, value)

    if state.get("input_stats"):
        st.caption(state["input_stats"])
    totals = run_totals(state["manifest"])
    if totals["model"] == "code only":
        st.caption("Checked with No AI: nothing was sent anywhere. Cost $0.")
    elif str(totals["model"]).startswith("fake/"):
        st.caption("Checked with AI answers recorded earlier for this sample. Cost $0.")
    else:
        cost = "unknown" if totals["cost_usd"] is None else f"${float(totals['cost_usd']):.4f}"
        st.caption(
            f"Checked with {friendly_model(totals['model'])} · {totals['calls']} AI calls "
            f"({totals['cached']} reused) · cost {cost}"
        )

    _lane_guide()

    total = len(cards)
    st.progress(decided_count(cards, fix, hand) / total if total else 0.0)
    st.caption(progress_text(cards, fix, hand))

    ready = ready_places(cards)
    ready_issues = sum(1 for card in cards if card.lane == "ready")
    check_issues = sum(1 for card in cards if card.lane == "check")
    actions = st.columns([2, 1, 3])
    if actions[0].button(
        f"✓ Fix all ready ({len(ready)} places)",
        key=f"fixready|{run_id}",
        type="primary",
        disabled=not ready,
    ):
        _set_fixes(run_id, ready, True)
        st.session_state[f"ready_note|{run_id}"] = (
            f"All {ready_issues} ready {'issue' if ready_issues == 1 else 'issues'} are set "
            f"to be fixed ({len(ready)} {'place' if len(ready) == 1 else 'places'}). "
            f"Next: look at Check first ({check_issues})."
        )
        st.rerun()
    note = st.session_state.pop(f"ready_note|{run_id}", None)
    if note:
        st.success(note)
    if not ready:
        actions[0].caption("Nothing here was checked by AI. Look at Check first, or check again with Gemini (free).")
    if actions[1].button("Clear all", key=f"clearall|{run_id}"):
        _clear_decisions(run_id)
        st.rerun()

    filters = st.columns([2, 3, 1])
    query = filters[0].text_input(
        "Search", placeholder="Name, label or episode, e.g. E22", key=f"search|{run_id}"
    )
    picked = filters[1].pills(
        "Kind",
        list(KIND_TITLES.values()),
        selection_mode="multi",
        key=f"kinds|{run_id}",
    )
    hide = filters[2].toggle("Hide decided", key=f"hide|{run_id}")
    kinds = [KIND_BY_LABEL[label] for label in (picked or [])]

    for lane in LANES:
        lane_cards = [card for card in cards if card.lane == lane]
        _lane_header(lane, lane_cards, run_id)
        shown = _visible_cards(lane_cards, query, kinds, hide, fix, hand)
        if not shown:
            st.caption("None here.")
            continue
        columns = st.columns(3)
        for index, card in enumerate(shown):
            with columns[index % 3]:
                _draw_card(card, run_id, fix, hand)

    kept_apart = apart_lines(result)
    if kept_apart:
        with st.expander("Looked alike, kept apart"):
            for line in kept_apart:
                st.caption(line)

    st.divider()
    if st.button("Next: download →", key=f"next|{run_id}", type="primary"):
        _go("Download")


def _download(state: dict) -> None:
    result = state["result"]
    run_id = state["run_id"]
    run_dir = Path(state["run_dir"])
    cards = build_cards(result)
    fix = _fix_map(run_id)
    hand = _hand_map(run_id)
    choices = _choices(run_id, result)
    rows = change_rows(cards, fix, choices)
    hand_rows = by_hand_rows(cards, hand)

    st.subheader("Download")
    if not rows:
        st.info("Nothing to fix yet. Tick fixes in Review.")
        if st.button("Back to Review", key=f"back|{run_id}"):
            _go("Review")
        return

    st.write(download_summary(cards, fix, hand))
    st.caption("A voice label change renames who speaks a line; the words stay the same.")
    issues = issue_rows(cards, fix, choices)
    summary = [
        {
            "Issue": item["issue"],
            "Kind": item["kind"],
            "Places": item["places"],
            "Episodes": _episode_ranges(item["episodes"]),
        }
        for item in issues
    ]
    st.dataframe(summary, hide_index=True, width="stretch")
    for item in issues:
        label = f"{item['issue']} · {item['places']} place{'s' if item['places'] != 1 else ''}"
        with st.expander(label):
            preview = [
                {
                    "Episode": f"E{row['episode']}" if row["episode"] else "Whole show",
                    "Time": row["time"],
                    "Heard/Seen/Voice/Notes": row["where_label"],
                    "Line": _cut(row.get("context", ""), 80),
                    "Was → Now": f"{row['was']} → {row['now']}",
                }
                for row in item["rows"]
            ]
            st.dataframe(preview, hide_index=True, width="stretch")

    defaults = {
        group.id: group.canonical
        for group in list(result.get("name_groups") or []) + list(result.get("speaker_groups") or [])
    }
    checked = {finding.id: bool(fix.get(finding.id)) for finding in result["findings"]}
    approvals = build_approvals(
        [finding.model_dump() for finding in result["findings"]], checked, choices, defaults
    )
    packed_key = f"fixed|{run_id}"
    if st.button("Make fixed scripts", key=f"make|{run_id}", type="primary"):
        approvals_path = run_dir / "approvals.json"
        approvals_path.write_text(approvals_json(approvals), encoding="utf-8")
        try:
            stats = patch(str(run_dir), str(approvals_path))
            data = zip_fixed(run_dir, rows, hand_rows)
        except Exception as exc:
            st.error(safe_text(f"{type(exc).__name__}: {exc}"))
        else:
            st.session_state[packed_key] = {
                "zip": data,
                "stats": stats,
                "digest": _decisions_digest(approvals, rows, hand_rows),
            }

    packed = st.session_state.get(packed_key)
    if packed:
        problems = packed["stats"].get("problems") or []
        if problems:
            st.warning(f"{len(problems)} places could not be changed.")
            for problem in problems:
                st.caption(safe_text(problem))
        current = _decisions_digest(approvals, rows, hand_rows)
        if packed["digest"] != current:
            st.warning("Your decisions changed since this copy was made. Make the fixed scripts again.")
        st.download_button(
            "⬇ Download fixed_scripts.zip",
            packed["zip"],
            file_name="fixed_scripts.zip",
            mime="application/zip",
            key=f"dl_fixed|{run_id}",
            type="primary",
        )

    with st.expander("More files"):
        columns = st.columns(4)
        files = [
            ("full report (HTML)", run_dir / "report.html", "text/html"),
            ("all issues (CSV)", run_dir / "findings.csv", "text/csv"),
            ("voice cast list (CSV)", run_dir / "cast.csv", "text/csv"),
        ]
        for column, (label, path, mime) in zip(columns, files):
            if path.is_file():
                column.download_button(
                    label,
                    path.read_bytes(),
                    file_name=path.name,
                    mime=mime,
                    key=f"dl_{path.stem}|{run_id}",
                )
        columns[3].download_button(
            "my decisions (JSON)",
            approvals_json(approvals).encode("utf-8"),
            file_name="approvals.json",
            mime="application/json",
            key=f"dl_decisions|{run_id}",
        )


def main() -> None:
    st.set_page_config(page_title="Canon", layout="wide")
    goto = st.session_state.pop("goto", None)
    if goto in PAGES:
        st.session_state["page"] = goto
    st.session_state.setdefault("page", "Home")
    _sidebar()

    page = st.segmented_control(
        "Go to", PAGES, key="page", label_visibility="collapsed"
    )
    if page not in PAGES:
        page = "Home"

    error = st.session_state.get("canon_error")
    if error:
        info = (
            ErrorInfo.from_dict(error)
            if isinstance(error, dict)
            else ErrorInfo("Canon stopped", str(error))
        )
        st.title("Canon")
        _show_error(info)
        st.caption("No scripts were changed. Fix the item above, then check again.")
        return

    state = st.session_state.get("canon_run")
    if page == "Home":
        _home()
    elif not state:
        _empty_state(page)
    elif page == "Review":
        _review(state)
    else:
        _download(state)


if __name__ == "__main__":
    main()
