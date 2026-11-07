"""Custom checks that run after Pydantic validation (spec §4).

They catch what a schema can't: invented timestamps, cuts in the middle of a line,
promos that are too short or too long. Each problem message names the nearest valid
timestamps so the retry prompt gives the model something concrete to fix.

A check is `fn(output) -> list[str]`; an empty list means it passed.
"""
from __future__ import annotations

import bisect
from typing import Any, Callable, Iterable

from config.loader import ParserConfig, Settings
from parser.models import Episode, format_ts
from parser.screenplay_parser import ParseError, parse_ts


class ScreenplayIndex:
    """Fast lookups over parsed episodes: valid cut points and dialogue spans."""

    def __init__(self, episodes: Iterable[Episode], parser_cfg: ParserConfig, tolerance_s: float,
                 dialogue_kind: str | None = None):
        self.pats = parser_cfg.compiled
        self.tol = tolerance_s
        self.dialogue_kind = dialogue_kind or parser_cfg.dialogue_kind
        self.episodes = {ep.number: ep for ep in episodes}
        self.points = {n: ep.boundaries() for n, ep in self.episodes.items()}
        self.starts = {n: sorted({it.start for it in ep.all_items()}) for n, ep in self.episodes.items()}
        self.ends = {n: sorted({it.end for it in ep.all_items()}) for n, ep in self.episodes.items()}
        self.lines = {n: [(it.start, it.end) for it in ep.all_items() if it.kind == self.dialogue_kind]
                      for n, ep in self.episodes.items()}

    def seconds(self, ts: str) -> float | None:
        try:
            return parse_ts(str(ts), self.pats)
        except ParseError:
            return None

    def _near(self, values: list[float], t: float) -> bool:
        """Is any value within tolerance of t? Checks the closest value on each side of t: a
        boundary exactly one tolerance away can land just outside it through float rounding,
        and must not hide an exact match next to it (real case: 00:09.10 / 00:09.15)."""
        i = bisect.bisect_left(values, t)
        return any(abs(values[j] - t) <= self.tol for j in (i - 1, i) if 0 <= j < len(values))

    def nearest(self, values: list[float], t: float, k: int = 3) -> str:
        best = sorted(values, key=lambda v: abs(v - t))[:k]
        return ", ".join(format_ts(v) for v in sorted(best))

    def exists(self, episode: int, t: float) -> bool:
        return self._near(self.points.get(episode, []), t)

    def splits_line(self, episode: int, t: float) -> tuple[float, float] | None:
        """The dialogue line that a cut at t would split, if any."""
        for s, e in self.lines.get(episode, []):
            if s + self.tol < t < e - self.tol:
                return s, e
        return None


# ------------------------------------------------------------------- timestamp checks
def check_timestamp(index: ScreenplayIndex, episode: int, ts: str, what: str) -> list[str]:
    """One timestamp: parseable, in a known episode, and actually in the screenplay."""
    if episode not in index.episodes:
        return [f"{what}: episode {episode} does not exist (have {sorted(index.episodes)})"]
    t = index.seconds(ts)
    if t is None:
        return [f"{what}: '{ts}' is not a timestamp in MM:SS.ff form"]
    if not index.exists(episode, t):
        return [f"{what}: {ts} is not a timestamp in episode {episode}; "
                f"nearest real ones: {index.nearest(index.points[episode], t)}"]
    return []


def check_segment(index: ScreenplayIndex, seg: Any, what: str) -> tuple[list[str], float]:
    """Returns (problems, duration in seconds)."""
    ep = seg.episode
    probs = check_timestamp(index, ep, seg.start_timestamp, f"{what} start") + \
        check_timestamp(index, ep, seg.end_timestamp, f"{what} end")
    s, e = index.seconds(seg.start_timestamp), index.seconds(seg.end_timestamp)
    if s is None or e is None or ep not in index.episodes:
        return probs, 0.0
    if e <= s:
        return probs + [f"{what}: end {seg.end_timestamp} is not after start {seg.start_timestamp}"], 0.0
    # Line boundaries: a promo may not open or close in the middle of a spoken line.
    if (line := index.splits_line(ep, s)) is not None:
        probs.append(f"{what}: start {seg.start_timestamp} cuts into a line spoken "
                     f"{format_ts(line[0])}-{format_ts(line[1])}; start at {format_ts(line[0])} or a line boundary")
    if (line := index.splits_line(ep, e)) is not None:
        probs.append(f"{what}: end {seg.end_timestamp} cuts off a line spoken "
                     f"{format_ts(line[0])}-{format_ts(line[1])}; end at {format_ts(line[1])} or a line boundary")
    return probs, round(e - s, 3)


def check_promo_segments(index: ScreenplayIndex, segments: list, settings: Settings, what: str) -> list[str]:
    """A whole promo: every segment valid, no overlaps, total duration within limits."""
    pc = settings.promo
    if not segments:
        return [f"{what}: has no segments"]
    probs: list[str] = []
    if len(segments) > pc.max_segments:
        probs.append(f"{what}: {len(segments)} segments, max is {pc.max_segments}")
    total = 0.0
    spans: dict[int, list[tuple[float, float]]] = {}
    for i, seg in enumerate(segments, start=1):
        label = f"{what} segment {i}" if len(segments) > 1 else what
        p, dur = check_segment(index, seg, label)
        probs += p
        total += dur
        s, e = index.seconds(seg.start_timestamp), index.seconds(seg.end_timestamp)
        if s is not None and e is not None and e > s:
            for os_, oe in spans.get(seg.episode, []):
                if s < oe - index.tol and os_ < e - index.tol:
                    probs.append(f"{label}: overlaps another segment of the same promo in episode {seg.episode}")
            spans.setdefault(seg.episode, []).append((s, e))
    total = round(total, 2)
    if total > 0:   # skip when no segment could be measured (already reported above)
        # Models are bad at MM:SS arithmetic, so say by how much, and for a single clip list
        # real line ends that would fit.
        if total < pc.min_duration_s:
            probs.append(f"{what}: total duration {total}s is under the {pc.min_duration_s:g}s minimum "
                         f"(short by {pc.min_duration_s - total:.1f}s); extend it{_fitting_ends(index, segments, pc)}")
        elif total > pc.max_duration_s:
            probs.append(f"{what}: total duration {total}s is over the {pc.max_duration_s:g}s maximum "
                         f"(over by {total - pc.max_duration_s:.1f}s); tighten it{_fitting_ends(index, segments, pc)}")
    return probs


def _fitting_ends(index: ScreenplayIndex, segments: list, pc, k: int = 4) -> str:
    """For a one-clip promo: line-end times that would give a legal duration from its start."""
    if len(segments) != 1:
        return ""
    seg = segments[0]
    s = index.seconds(seg.start_timestamp)
    if s is None or seg.episode not in index.ends:
        return ""
    ok = [e for e in index.ends[seg.episode]
          if pc.min_duration_s <= e - s <= pc.max_duration_s and index.splits_line(seg.episode, e) is None]
    if not ok:
        return f"; no line end gives {pc.min_duration_s:g}-{pc.max_duration_s:g}s from {seg.start_timestamp}, pick another start"
    return f"; from start {seg.start_timestamp}, valid end times include: {', '.join(format_ts(e) for e in ok[-k:])}"


# --------------------------------------------------------------------- check factories
def promo_checks(index: ScreenplayIndex, settings: Settings,
                 get_promos: Callable[[Any], list], name: Callable[[Any, int], str] | None = None):
    """Check for outputs holding a list of promos, each with `.segments`."""
    name = name or (lambda p, i: f"promo {getattr(p, 'promo_id', None) or i}")

    def check_promos(output) -> list[str]:
        probs: list[str] = []
        for i, promo in enumerate(get_promos(output), start=1):
            probs += check_promo_segments(index, promo.segments, settings, name(promo, i))
        return probs
    return check_promos


def timestamp_checks(index: ScreenplayIndex, get_refs: Callable[[Any], list[tuple[int, str, str]]]):
    """Check for outputs that cite single timestamps (e.g. Reader moments).
    get_refs returns (episode, timestamp, label) tuples."""
    def check_timestamps(output) -> list[str]:
        probs: list[str] = []
        for ep, ts, label in get_refs(output):
            probs += check_timestamp(index, ep, ts, label)
        return probs
    return check_timestamps
