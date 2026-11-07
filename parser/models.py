"""Structured screenplay data. These models are the single source of truth that every
later stage (agents, custom checks, ranking, UI) reads from."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


def format_ts(seconds: float) -> str:
    """Seconds -> 'MM:SS.ff' (the display format used in outputs)."""
    cs = round(seconds * 100)
    m, rem = divmod(cs, 6000)
    s, f = divmod(rem, 100)
    return f"{m:02d}:{s:02d}.{f:02d}"


class Item(BaseModel):
    """One timed block: a camera shot, action line, sound cue or line of dialogue."""

    item_id: str                      # e01_s01_i023: stable, used in traces
    episode: int
    scene: int
    index: int                        # position within the scene
    kind: str                         # camera | action | sound | dialogue (from parser.yaml)
    label: str                        # raw block label, e.g. "MATRIARCH (O.S.) (Firm)"
    character: str | None = None      # dialogue only
    voice_tags: list[str] = []        # O.S., V.O. ...
    emotions: list[str] = []
    text: str
    start: float                      # seconds from episode start
    end: float
    start_ts: str                     # original strings, kept for exact matching
    end_ts: str
    end_inferred: bool = False        # True when End was missing and back-filled
    source: Literal["pattern", "llm_fallback"] = "pattern"
    line_no: int                      # line of the block header in the source file

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def render(self) -> str:
        """Compact one-line form for prompts."""
        who = self.character or self.kind.upper()
        tone = f" ({', '.join(self.emotions)})" if self.emotions else ""
        return f"[{format_ts(self.start)}-{format_ts(self.end)}] {who}{tone}: {self.text}"


class Scene(BaseModel):
    number: int
    heading: str | None = None
    start: float
    end: float
    items: list[Item] = []


class UnparsedLine(BaseModel):
    line_no: int
    text: str
    reason: str


class Coverage(BaseModel):
    content_lines: int                # non-blank lines in the file
    parsed_lines: int                 # matched by a pattern (incl. ignored/marker lines)
    unparsed_lines: int
    recovered_by_fallback: int = 0    # unparsed lines the LLM fallback turned into items
    percent: float                    # parsed / content * 100


class Episode(BaseModel):
    number: int
    source_file: str
    logline: str | None = None
    continuity_notes: str | None = None
    scenes: list[Scene] = []
    duration: float = 0.0
    coverage: Coverage
    unparsed: list[UnparsedLine] = []
    warnings: list[str] = []

    # ---- helpers used by custom checks and agents ---------------------------------
    def all_items(self) -> list[Item]:
        return [it for sc in self.scenes for it in sc.items]

    def boundaries(self) -> list[float]:
        """Every valid cut point: each item's start and end, sorted and unique.
        Promo start/end points must land on one of these."""
        pts = {self.scenes[0].start} if self.scenes else set()
        for it in self.all_items():
            pts.add(it.start)
            pts.add(it.end)
        return sorted(pts)

    def timestamp_index(self) -> set[str]:
        """Every timestamp string that appears in the screenplay (for rejecting invented ones)."""
        out: set[str] = set()
        for it in self.all_items():
            out.add(it.start_ts)
            out.add(it.end_ts)
        return out

    def items_between(self, start: float, end: float, kinds: list[str] | None = None) -> list[Item]:
        """Items that start inside [start, end)."""
        return [
            it for it in self.all_items()
            if start <= it.start < end and (kinds is None or it.kind in kinds)
        ]


class Show(BaseModel):
    name: str | None = None
    episodes: list[Episode] = Field(default_factory=list)

    def episode(self, number: int) -> Episode:
        for ep in self.episodes:
            if ep.number == number:
                return ep
        raise KeyError(f"episode {number} not parsed")
