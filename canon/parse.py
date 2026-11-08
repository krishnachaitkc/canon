import os
import re
import zipfile

from .models import Beat, Episode, Scene, Show

BEAT = re.compile(
    r"\*\*([^*]+)\*\*[ \t]*\r?\n"
    r"[ \t]*Start:\s*\(?([0-9:.]+)\)?\s*\|\s*End:\s*\(?([0-9:.]+)\)?[^\r\n]*\r?\n"
    r"((?:(?!\*\*)[^\r\n]+\r?\n?)*)"
)

SCENE = re.compile(
    r"^[ \t]*SCENE[ \t]+(\d+)[ \t]*\r?\n"
    r"[ \t]*SCENE HEADING:[ \t]*(.*?)[ \t]*\r?\n"
    r"[ \t]*SCENE START:[ \t]*\(?([0-9:.]+)\)?[^\r\n]*\r?\n"
    r"[ \t]*SCENE END:[ \t]*\(?([0-9:.]+)\)?",
    re.IGNORECASE | re.MULTILINE,
)

STRUCTURAL = re.compile(
    r"^(?:"
    r"SCENE[ \t]+\d+[ \t]*$"
    r"|SCENE[ \t]+HEADING:.*$"
    r"|SCENE[ \t]+START:.*$"
    r"|SCENE[ \t]+END:.*$"
    r"|ACTIONS[ \t]+AND[ \t]+DIALOGUE:?[ \t]*$"
    r"|---.*---[ \t]*$"
    r")",
    re.IGNORECASE,
)

SFX_TAGS = {"SOUND FX", "SFX", "SOUND", "SOUND EFFECT", "SOUND EFFECTS", "MUSIC", "AUDIO"}
OFFSCREEN_GROUPS = {"O.S.", "V.O.", "OS", "VO", "OFF-SCREEN"}

# A heading separator is a dash with spaces on both sides, so "PRE-DAWN" stays whole.
HEADING_SEP = re.compile(r"\s+[–—-]\s+")
INT_EXT = re.compile(r"^(INT\.?/EXT\.?|EXT\.?/INT\.?|INT\.?|EXT\.?)\s+", re.IGNORECASE)


def split_heading(heading: str) -> tuple[str, str, str]:
    """Split a scene heading into (int_ext, place, time).

    The time is the text after the last spaced dash, e.g. DAY, NIGHT or CONTINUOUS.
    A heading with no spaced dash has no time.
    """
    text = " ".join(heading.split())
    parts = HEADING_SEP.split(text)
    if len(parts) > 1:
        place, time = " - ".join(parts[:-1]), parts[-1]
    else:
        place, time = text, ""
    int_ext = ""
    match = INT_EXT.match(place)
    if match:
        int_ext = match.group(1).upper().replace(".", "")
        place = place[match.end():]
    return int_ext, place.strip().upper(), time.strip().upper()


def natural_key(name: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def sec(ts: str) -> float:
    value = 0.0
    for part in ts.strip("() \t\r\n").split(":"):
        value = value * 60 + float(part)
    return value


def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    remainder = seconds % 3600
    minutes = int(remainder // 60)
    secs = remainder % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes:02d}:{secs:05.2f}"


def default_title(input_source: str) -> str:
    base = os.path.splitext(os.path.basename(os.path.normpath(input_source)))[0]
    base = re.sub(r"[-_ ]*scripts?$", "", base, flags=re.IGNORECASE)
    base = base.replace("_", " ")
    return re.sub(r"\s+", " ", base).strip()


def read_input(input_source: str) -> list[tuple[str, str]]:
    """Return (filename, text) pairs from a folder or zip, in natural order."""
    if os.path.isdir(input_source):
        names = [
            name
            for name in os.listdir(input_source)
            if name.lower().endswith((".txt", ".md"))
            and os.path.isfile(os.path.join(input_source, name))
        ]
        names.sort(key=natural_key)
        out = []
        for name in names:
            with open(os.path.join(input_source, name), encoding="utf-8", errors="replace") as fh:
                out.append((name, fh.read()))
        return out
    if os.path.isfile(input_source) and zipfile.is_zipfile(input_source):
        with zipfile.ZipFile(input_source) as zf:
            names = [
                name
                for name in zf.namelist()
                if name.lower().endswith((".txt", ".md"))
                and not name.endswith("/")
                and "__MACOSX" not in name
            ]
            names.sort(key=natural_key)
            return [
                (name, zf.read(name).decode("utf-8", errors="replace"))
                for name in names
            ]
    raise ValueError(f"Input must be a folder or a zip of .txt/.md files: {input_source}")


def _episode_number(content: str, filename: str) -> int:
    match = re.search(r"EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)", content, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"episode[^0-9]*(\d+)", filename, re.IGNORECASE)
    if not match:
        match = re.search(r"(\d+)", filename)
    if not match:
        raise ValueError(f"Cannot determine episode number from {filename}")
    return int(match.group(1))


def _single_line_header(content: str, name: str) -> str:
    match = re.search(rf"{name}:\s*(.+)", content, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _scene_for(t: float, scenes: list[dict]) -> int:
    for scene in scenes:
        if scene["start"] - 0.01 <= t < scene["end"]:
            return scene["num"]
    before = [scene for scene in scenes if scene["start"] <= t + 0.01]
    if before:
        return before[-1]["num"]
    return scenes[0]["num"] if scenes else 1


def parse_episode_text(content: str, filename: str = "") -> tuple[Episode, int, int, list[str]]:
    """Parse one screenplay. Returns (episode, unconsumed_lines, script_lines)."""
    ep_num = _episode_number(content, filename)
    logline = _single_line_header(content, "EPISODE LOGLINE")
    continuity = _single_line_header(content, "CONTINUITY NOTES")

    scenes = [
        {
            "num": int(match.group(1)),
            "heading": match.group(2).strip(),
            "start": sec(match.group(3)),
            "end": sec(match.group(4)),
        }
        for match in SCENE.finditer(content)
    ]

    marker = re.search(r"---\s*SCRIPT START\s*---", content, re.IGNORECASE)
    body = content[marker.end():] if marker else content

    anomalies: list[str] = []
    rows = []
    spans = []
    for order, match in enumerate(BEAT.finditer(body)):
        tag = match.group(1).strip().rstrip(":").strip()
        raw_start, raw_end = match.group(2), match.group(3)
        start, end = sec(raw_start), sec(raw_end)
        text = " ".join(match.group(4).split())

        base = re.sub(r"\(.*$", "", tag).strip()
        upper = base.upper()
        speaker = None
        tone = None
        offscreen = False
        if upper.startswith("CAMERA"):
            kind = "shot"
        elif upper.startswith("ACTION"):
            kind = "action"
        elif upper in SFX_TAGS:
            kind = "sfx"
        else:
            kind = "line"
            speaker = base
            groups = re.findall(r"\(([^)]*)\)", tag)
            offscreen = any(group.strip().upper() in OFFSCREEN_GROUPS for group in groups)
            tone_parts = [
                group.strip()
                for group in groups
                if group.strip().upper() not in OFFSCREEN_GROUPS
            ]
            tone = ", ".join(tone_parts) if tone_parts else None

        if start >= end:
            anomalies.append(
                f"E{ep_num:02d} beat {order + 1}: start {start} >= end {end}; "
                f"end set to start + 0.5"
            )
            end = start + 0.5

        rows.append(
            {
                "order": order,
                "kind": kind,
                "speaker": speaker,
                "tone": tone,
                "offscreen": offscreen,
                "start": start,
                "end": end,
                "text": text,
            }
        )
        spans.append((match.start(), match.end()))

    rows.sort(key=lambda row: (row["start"], row["order"]))
    beats = [
        Beat(
            id=f"E{ep_num:02d}.B{index:03d}",
            ep=ep_num,
            scene=_scene_for(row["start"], scenes),
            kind=row["kind"],
            start=row["start"],
            end=row["end"],
            text=row["text"],
            speaker=row["speaker"],
            tone=row["tone"],
            offscreen=row["offscreen"],
        )
        for index, row in enumerate(rows, 1)
    ]

    unconsumed = 0
    script_lines = 0
    offset = 0
    for line in body.splitlines(keepends=True):
        line_start, line_end = offset, offset + len(line)
        offset = line_end
        stripped = line.strip()
        if not stripped or STRUCTURAL.match(stripped):
            continue
        script_lines += 1
        if any(start < line_end and line_start < end for start, end in spans):
            continue
        unconsumed += 1

    runtime = max((beat.end for beat in beats), default=0.0)
    scene_models = []
    for scene in scenes:
        int_ext, place, time = split_heading(scene["heading"])
        scene_models.append(
            Scene(
                ep=ep_num,
                num=scene["num"],
                heading=scene["heading"],
                start=scene["start"],
                end=scene["end"],
                int_ext=int_ext,
                place=place,
                time=time,
            )
        )
    episode = Episode(
        num=ep_num,
        logline=logline,
        continuity=continuity,
        beats=beats,
        scenes=scene_models,
        runtime=runtime,
    )
    return episode, unconsumed, script_lines, anomalies


def parse_show(input_source: str, title: str | None = None) -> Show:
    files = read_input(input_source)
    if not files:
        raise ValueError(f"No .txt or .md screenplays found in {input_source}")

    episodes: list[Episode] = []
    anomalies: list[str] = []
    unconsumed = 0
    script_lines = 0
    seen: set[int] = set()

    for filename, content in files:
        episode, missing, lines, problems = parse_episode_text(content, filename)
        if episode.num in seen:
            raise ValueError(f"Duplicate episode number {episode.num} ({filename})")
        seen.add(episode.num)
        episodes.append(episode)
        anomalies.extend(problems)
        unconsumed += missing
        script_lines += lines

    episodes.sort(key=lambda episode: episode.num)
    return Show(
        title=title or default_title(input_source),
        episodes=episodes,
        anomalies=anomalies,
        unconsumed_lines=unconsumed,
        script_lines=script_lines,
    )
