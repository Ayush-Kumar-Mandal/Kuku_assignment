"""Config-driven screenplay parser.

A line-by-line state machine. Every pattern comes from config/parser.yaml; the code only
knows the *shape* of a screenplay (header -> scenes -> timed blocks), not its spelling.

Lines no pattern can place are collected as `unparsed` with a reason. An optional
`fallback` callable (wired to an LLM in Step 2) may turn them into Items.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from config.loader import CompiledPatterns, ParserConfig, load_parser_config
from parser.models import Coverage, Episode, Item, Scene, Show, UnparsedLine, format_ts

# fallback(unparsed_lines, partially_parsed_episode) -> recovered items
FallbackFn = Callable[[list[UnparsedLine], Episode], list[Item]]


class ParseError(ValueError):
    pass


# ------------------------------------------------------------------------- helpers
def parse_ts(text: str, pats: CompiledPatterns) -> float:
    m = pats.timestamp.match(text)
    if not m:
        raise ParseError(f"not a timestamp: {text!r}")
    g = m.groupdict()
    secs = int(g.get("h") or 0) * 3600 + int(g["m"]) * 60 + int(g["s"])
    frac = g.get("f")
    if frac:
        secs += int(frac) / 10 ** len(frac)
    return round(secs, 3)


def _split_label(label: str, cfg: ParserConfig, pats: CompiledPatterns):
    """'MATRIARCH (O.S.) (Firm, Cold)' -> kind, character, voice_tags, emotions."""
    bare = pats.parenthetical.sub("", label).strip().upper()
    kind = pats.block_kinds.get(bare)
    if kind is not None:
        return kind, None, [], []

    m = pats.speaker_name.match(label)
    character = (m.group("name") if m else label).strip()
    voice_tags, emotions = [], []
    for piece in pats.parenthetical.findall(label):
        piece = piece.strip()
        if piece.upper().rstrip(".") in pats.voice_tags:
            voice_tags.append(piece)
        else:
            emotions += [e.strip() for e in piece.split(cfg.speaker.emotion_separator) if e.strip()]
    return cfg.dialogue_kind, character, voice_tags, emotions


def _clean_text(lines: list[str], kind: str, cfg: ParserConfig) -> str:
    text = " ".join(lines).strip()
    q = cfg.dialogue_quotes
    if kind == cfg.dialogue_kind and text and text[0] in q:
        text = text.strip(q).strip()
    return text


# -------------------------------------------------------------------------- parser
class _Block:
    def __init__(self, label: str, line_no: int):
        self.label = label
        self.line_no = line_no
        self.start_ts: str | None = None
        self.end_ts: str | None = None
        self.body: list[str] = []
        self.raw: list[tuple[int, str]] = []   # every source line belonging to this block
        self.closed = False                     # body ended (blank line seen after it)


class _SceneBuf:
    def __init__(self, number: int, line_no: int):
        self.number = number
        self.line_no = line_no
        self.heading: str | None = None
        self.start: float | None = None
        self.end: float | None = None
        self.items: list[dict] = []


class _EpisodeParser:
    def __init__(self, path: Path, cfg: ParserConfig):
        self.path = path
        self.cfg = cfg
        self.p = cfg.compiled
        self.header: dict[str, str] = {}
        self.last_header: str | None = None
        self.scenes: list[_SceneBuf] = []
        self.scene: _SceneBuf | None = None
        self.block: _Block | None = None
        self.unparsed: list[UnparsedLine] = []
        self.content = 0
        self.parsed = 0
        self.seen_body = False   # a scene or block has appeared (header is over)
        self.ended = False       # past the script-end marker

    # -- small utilities ------------------------------------------------------------
    def _miss(self, line_no: int, text: str, reason: str):
        self.unparsed.append(UnparsedLine(line_no=line_no, text=text, reason=reason))

    def _ts(self, text: str, line_no: int, raw: str) -> float | None:
        try:
            return parse_ts(text, self.p)
        except ParseError as e:
            self._miss(line_no, raw, str(e))
            return None

    def _ensure_scene(self, line_no: int) -> _SceneBuf:
        # Formats without SCENE markers still parse: open an implicit scene.
        if self.scene is None:
            n = self.scenes[-1].number + 1 if self.scenes else 1
            self.scene = _SceneBuf(n, line_no)
            self.scenes.append(self.scene)
        return self.scene

    # -- block lifecycle ------------------------------------------------------------
    def _flush_block(self):
        b, self.block = self.block, None
        if b is None:
            return
        if b.start_ts is None:
            for ln, txt in b.raw:
                self._miss(ln, txt, "block without a timing line")
            return
        start = self._ts(b.start_ts, b.line_no, b.start_ts)
        end = self._ts(b.end_ts, b.line_no, b.end_ts) if b.end_ts else None
        if start is None:
            return
        kind, character, tags, emotions = _split_label(b.label, self.cfg, self.p)
        scene = self._ensure_scene(b.line_no)
        scene.items.append(dict(
            label=b.label, kind=kind, character=character, voice_tags=tags,
            emotions=emotions, text=_clean_text(b.body, kind, self.cfg),
            start=start, end=end, start_ts=b.start_ts, end_ts=b.end_ts, line_no=b.line_no,
        ))
        self.parsed += len(b.raw)

    def _close_scene(self):
        self._flush_block()
        self.scene = None

    # -- main loop ------------------------------------------------------------------
    def feed(self, line_no: int, raw: str):
        s = raw.strip()
        if not s:
            if self.block is not None and self.block.body and self.cfg.block_ends_on_blank_line:
                self.block.closed = True
            return
        self.content += 1
        p = self.p

        if p.script_start.match(s):
            self._flush_block()
            self.parsed += 1
            return
        if p.script_end.match(s):
            self._close_scene()
            self.ended = True
            self.parsed += 1
            return

        for key, rx in p.header.items():
            m = rx.match(s)
            if m:
                self.header[key] = (m.groupdict().get("num") or m.groupdict().get("text") or "").strip()
                self.last_header = key
                self.parsed += 1
                return

        m = p.scene["number"].match(s)
        if m:
            self._close_scene()
            self.scene = _SceneBuf(int(m.group("num")), line_no)
            self.scenes.append(self.scene)
            self.seen_body = True
            self.parsed += 1
            return
        for key in ("heading", "start", "end"):
            m = p.scene[key].match(s)
            if m:
                self._flush_block()
                sc = self._ensure_scene(line_no)
                if key == "heading":
                    sc.heading = m.group("text").strip()
                else:
                    val = self._ts(m.group("ts"), line_no, s)
                    if val is None:
                        return
                    setattr(sc, key, val)
                self.seen_body = True
                self.parsed += 1
                return

        if any(rx.match(s) for rx in p.ignore):
            self.parsed += 1
            return

        m = p.block_header.match(s)
        if m:
            self._flush_block()
            self.block = _Block(m.group("label").strip(), line_no)
            self.block.raw.append((line_no, s))
            body = (m.groupdict().get("body") or "").strip()
            if body:
                self.block.body.append(body)
            self.seen_body = True
            return

        m = p.timing_line.match(s)
        if m:
            if self.block is None:
                self._miss(line_no, s, "timing line outside a block")
            elif self.block.start_ts is not None:
                self._miss(line_no, s, "second timing line in one block")
            else:
                self.block.start_ts = m.group("start")
                self.block.end_ts = m.group("end")
                self.block.raw.append((line_no, s))
            return

        # Plain text: block body, or a wrapped header value, or unknown.
        if self.block is not None and not self.block.closed:
            self.block.body.append(s)
            self.block.raw.append((line_no, s))
        elif not self.seen_body and self.last_header in ("logline", "continuity_notes"):
            self.header[self.last_header] += " " + s
            self.parsed += 1
        elif self.ended:
            self._miss(line_no, s, "text after script end")
        else:
            self._miss(line_no, s, "unrecognised line")

    # -- assembly -------------------------------------------------------------------
    def build(self) -> Episode:
        self._close_scene()
        number = self._episode_number()
        tol = self.cfg.warning_tolerance_s
        warnings: list[str] = []
        scenes: list[Scene] = []

        for sb in self.scenes:
            if not sb.items and sb.start is None:
                warnings.append(f"scene {sb.number}: empty, dropped")
                continue
            starts = [it["start"] for it in sb.items]
            sc_start = sb.start if sb.start is not None else (min(starts) if starts else 0.0)
            known_ends = [it["end"] for it in sb.items if it["end"] is not None]
            sc_end = sb.end if sb.end is not None else max(known_ends + starts + [sc_start])
            if sb.start is None:
                warnings.append(f"scene {sb.number}: no start time, using first item ({format_ts(sc_start)})")
            if sb.end is None:
                warnings.append(f"scene {sb.number}: no end time, using last item ({format_ts(sc_end)})")

            items: list[Item] = []
            for i, it in enumerate(sb.items):
                inferred = it["end"] is None
                if inferred:
                    # Spec: end time = next item's start; the last item runs to scene end.
                    nxt = next((o["start"] for o in sb.items[i + 1:] if o["start"] > it["start"]), None)
                    it["end"] = nxt if nxt is not None else sc_end
                    it["end_ts"] = format_ts(it["end"])
                items.append(Item(
                    item_id=f"e{number:02d}_s{sb.number:02d}_i{i + 1:03d}",
                    episode=number, scene=sb.number, index=i + 1, end_inferred=inferred, **it,
                ))
            scenes.append(Scene(number=sb.number, heading=sb.heading, start=sc_start, end=sc_end, items=items))

        warnings += _sanity_checks(scenes, tol)
        all_ends = [it.end for sc in scenes for it in sc.items] + [sc.end for sc in scenes]
        cov = Coverage(
            content_lines=self.content, parsed_lines=self.parsed, unparsed_lines=len(self.unparsed),
            percent=round(100 * self.parsed / self.content, 2) if self.content else 0.0,
        )
        return Episode(
            number=number, source_file=self.path.name,
            logline=self.header.get("logline"), continuity_notes=self.header.get("continuity_notes"),
            scenes=scenes, duration=max(all_ends, default=0.0),
            coverage=cov, unparsed=self.unparsed, warnings=warnings,
        )

    def _episode_number(self) -> int:
        if self.header.get("episode_number", "").isdigit():
            return int(self.header["episode_number"])
        m = self.p.filename_episode.search(self.path.stem)
        if m:
            return int(m.group("num"))
        raise ParseError(f"{self.path.name}: no episode number in header or filename")


def _sanity_checks(scenes: list[Scene], tol: float) -> list[str]:
    out: list[str] = []
    prev_end = None
    for sc in scenes:
        if prev_end is not None and sc.start < prev_end - tol:
            out.append(f"scene {sc.number}: starts {format_ts(sc.start)} before previous scene ends {format_ts(prev_end)}")
        prev_end = sc.end
        last_start: dict[str, float] = {}
        for it in sc.items:
            if it.end < it.start:
                out.append(f"{it.item_id}: end {it.end_ts} before start {it.start_ts}")
            if it.start < sc.start - tol or it.end > sc.end + tol:
                out.append(f"{it.item_id}: {it.start_ts}-{it.end_ts} outside scene {format_ts(sc.start)}-{format_ts(sc.end)}")
            if it.start < last_start.get(it.kind, -1.0) - tol:
                out.append(f"{it.item_id}: {it.kind} starts {it.start_ts}, earlier than previous {it.kind}")
            last_start[it.kind] = it.start
            if not it.text:
                out.append(f"{it.item_id}: empty text")
    return out


# ------------------------------------------------------------------------ public API
def parse_text(text: str, cfg: ParserConfig | None = None, name: str = "episode.txt") -> Episode:
    """Parse screenplay text that isn't on disk yet (`name` stands in for the file name,
    e.g. for the episode number fallback). Used to validate generated episodes."""
    cfg = cfg or load_parser_config()
    ep_parser = _EpisodeParser(Path(name), cfg)
    for line_no, line in enumerate(text.splitlines(), start=1):
        ep_parser.feed(line_no, line)
    return ep_parser.build()


def parse_episode(path: str | Path, cfg: ParserConfig | None = None, fallback: FallbackFn | None = None) -> Episode:
    cfg = cfg or load_parser_config()
    path = Path(path)
    episode = parse_text(path.read_text(encoding=cfg.encoding), cfg, str(path))
    if fallback and episode.unparsed:
        _apply_fallback(episode, fallback)
    return episode


def _apply_fallback(episode: Episode, fallback: FallbackFn) -> None:
    recovered = fallback(list(episode.unparsed), episode)
    claimed: set[int] = set()
    for n, it in enumerate(recovered, start=1):
        scene = next((s for s in episode.scenes if s.start <= it.start < s.end), episode.scenes[-1] if episode.scenes else None)
        if scene is None:
            continue
        it.source, it.episode, it.scene = "llm_fallback", episode.number, scene.number
        it.item_id = f"e{episode.number:02d}_s{scene.number:02d}_f{n:03d}"
        scene.items.append(it)
        scene.items.sort(key=lambda x: (x.start, x.index))
        claimed.add(it.line_no)
    episode.coverage.recovered_by_fallback = len(claimed & {u.line_no for u in episode.unparsed})


def parse_show(folder: str | Path, cfg: ParserConfig | None = None, fallback: FallbackFn | None = None,
               name: str | None = None) -> Show:
    cfg = cfg or load_parser_config()
    folder = Path(folder)
    files = sorted(folder.glob(cfg.file_glob))
    if not files:
        raise FileNotFoundError(f"no files matching {cfg.file_glob!r} in {folder}")
    episodes = [parse_episode(f, cfg, fallback) for f in files]
    episodes.sort(key=lambda e: e.number)
    dupes = {e.number for e in episodes if sum(o.number == e.number for o in episodes) > 1}
    if dupes:
        raise ParseError(f"duplicate episode numbers: {sorted(dupes)}")
    return Show(name=name, episodes=episodes)
