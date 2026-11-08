from typing import Literal

from pydantic import BaseModel, Field


class Beat(BaseModel):
    id: str
    ep: int
    scene: int
    kind: Literal["shot", "line", "action", "sfx"]
    start: float
    end: float
    text: str
    speaker: str | None = None
    tone: str | None = None
    offscreen: bool = False


class Scene(BaseModel):
    ep: int
    num: int
    heading: str
    start: float
    end: float
    int_ext: str = ""
    place: str = ""
    time: str = ""


class Episode(BaseModel):
    num: int
    logline: str = ""
    continuity: str = ""
    beats: list[Beat] = Field(default_factory=list)
    scenes: list[Scene] = Field(default_factory=list)
    runtime: float = 0.0


class Show(BaseModel):
    title: str
    episodes: list[Episode] = Field(default_factory=list)
    anomalies: list[str] = Field(default_factory=list)
    unconsumed_lines: int = 0
    script_lines: int = 0


Kind = Literal["spoken", "shot", "action", "sfx", "header"]


class Mention(BaseModel):
    surface: str          # as written (headings title-cased), e.g. "Ravenscar's"
    key: str              # normalized, e.g. "ravenscar"
    ep: int
    scene: int            # 0 for the logline and continuity notes
    beat_id: str          # beat ID, or E05.logline / E05.notes / E05.S02.heading
    start: float
    end: float
    kind: Kind
    speaker: str | None = None
    char_start: int = 0


class Candidate(BaseModel):
    id: str
    keys: list[str]
    spellings: dict[str, int]       # display spelling -> mentions
    spoken: dict[str, int]          # display spelling -> spoken mentions
    links: list[tuple[str, str, str]]  # (key_a, key_b, rule)


Source = Literal["rule", "ai", "check", "code"]
Severity = Literal["audible", "voice", "visible", "notes"]


class NameGroup(BaseModel):
    id: str                         # N003.1
    candidate: str
    spellings: list[str]            # display spellings, as the script writes them
    keys: list[str]
    canonical: str
    same_entity: bool
    entity_type: str = "other"
    confidence: str = "low"
    reason: str = ""
    source: Source


class SpeakerGroup(BaseModel):
    id: str                         # S001
    labels: list[str]
    canonical: str
    intentional: bool = False
    confidence: str = "low"
    reason: str = ""
    source: Source
    vetoed: list[tuple[str, str, str]] = Field(default_factory=list)  # (label_a, label_b, "E41 S01 00:34.80")


class Edit(BaseModel):
    """What approving a finding changes: the nth whole word `old` at location `at` becomes `new`.

    `at` is a beat ID, E05.logline, E05.notes, E05.S02.heading, E05.S02.time (the heading's
    time of day) or a beat ID plus ".label" (its speaker label).
    """

    at: str
    old: str
    new: str
    nth: int = 0


class Finding(BaseModel):
    """One row of findings.csv. Every location and quote is written by code."""

    id: str = ""
    detector: Literal["names", "speakers", "joins", "spelling", "format"]
    severity: Severity
    source: Source
    confidence: str = ""
    ep: int
    scene: int
    start: str                      # MM:SS.ss
    where: str                      # beat ID, label, or "E27 S1 -> E28 S1" for a join
    found: str
    suggestion: str = ""
    context: str = ""               # at most 70 characters, cut by code
    group: str = ""                 # links rows that share one fix
    edit: Edit | None = None        # None: report only, a person makes the change


class SpeakerStat(BaseModel):
    label: str
    lines: int
    episodes: list[int]
    scenes: list[str]               # "E01S02"
    core: list[str]                 # identifying words left after titles and generic roles
    samples: list[str]              # first line and longest line, cut to 100 characters
    camera_names: list[tuple[str, int]]  # personal names in shots that play during its lines
    named_in: list[str] = Field(default_factory=list)  # sentences naming a descriptive label with a person
    co_speakers: dict[str, int]     # other label -> scenes shared
