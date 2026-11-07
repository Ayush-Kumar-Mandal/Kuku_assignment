"""Small helpers shared by the evaluation scripts."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from config.loader import Settings
from parser.models import Episode


def eval_dir(settings: Settings) -> Path:
    d = settings.resolve(settings.evaluation.dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "show"


def run_path(settings: Settings, run_id: str) -> Path:
    return settings.resolve(settings.paths.output_dir) / "runs" / run_id


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def run_episodes(settings: Settings, run_id: str) -> dict[int, Episode]:
    """The run's own parsed episodes (written once by the parse node)."""
    d = run_path(settings, run_id) / "parsed"
    eps = [Episode.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(d.glob("episode_*.json"))]
    if not eps:
        raise SystemExit(f"no parsed episodes in {d}: is '{run_id}' a run id?")
    return {e.number: e for e in eps}


def spans(segments: list[dict], to_s) -> list[tuple[int, float, float]]:
    return [(s["episode"], to_s(s["start_timestamp"]), to_s(s["end_timestamp"])) for s in segments]


def overlap(a: list[tuple[int, float, float]], b: list[tuple[int, float, float]]) -> float:
    """Shared screen time / the shorter side's duration (same measure as ranking)."""
    shared = sum(max(0.0, min(e1, e2) - max(s1, s2)) for ep1, s1, e1 in a for ep2, s2, e2 in b if ep1 == ep2)
    da, db = sum(e - s for _, s, e in a), sum(e - s for _, s, e in b)
    shorter = min(da, db)
    return shared / shorter if shorter > 0 else 0.0
