"""Typed LangGraph state.

Everything in the state is plain JSON-able data (dicts/lists), so the SQLite checkpointer
can always restore it, even after code changes. Pydantic models are rebuilt from it
where needed (see graph/context.py).

Reducers: dict channels merge (a node returns only the keys it changed); `moments`
and `trace` are append-only; everything else is replaced.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict


def merge(left: dict | None, right: dict | None) -> dict:
    out = dict(left or {})
    out.update(right or {})
    return out


Maker = Literal["cutter", "joiner"]


class Revision(TypedDict):
    promo_id: str
    maker: Maker                       # node that made the promo and must fix it
    instruction: str
    source: Literal["evaluator", "editor"]


class PromoState(TypedDict, total=False):
    # ---- run inputs (CLI / UI) -----------------------------------------------------
    run_id: str
    show: str
    genre: str
    input_dir: str
    switches: dict[str, bool]          # memory / joiner / editor_review for this run
    episode_filter: list[int] | None   # only these episode numbers (cheap trial runs); None = all
    agents_kind: str                   # "llm" or "stub": a resume must use the same agents
    model_overrides: dict[str, str]    # {agent: "provider:model"} chosen at start (CLI --model / UI)

    # ---- parse ---------------------------------------------------------------------
    # Compact index only: {number, source_file, logline, duration, items, parsed_path}.
    # The full parsed episodes live in output/runs/<run_id>/parsed/ (written once by parse)
    # because the SQLite checkpointer stores a full copy of the state at EVERY step:
    # keeping ~1.5 MB of screenplay in state made one 50-episode run's DB 143 MB.
    episodes: list[dict]
    parse_report: dict

    # ---- memory_read ---------------------------------------------------------------
    memory_used: dict                  # enabled, guidelines_version, rules, examples

    # ---- reader (one episode per step) ---------------------------------------------
    reader_cursor: int                 # index into episodes of the next one to read
    story_summary: str                 # running summary carried episode to episode
    episode_notes: Annotated[dict[str, dict], merge]   # ep -> context a new viewer needs, summary after it
    moments: Annotated[list[dict], operator.add]       # hook + cliffhanger moments

    # ---- cutter / joiner -----------------------------------------------------------
    cutter_cursor: int
    joiner_done: bool
    candidates: Annotated[dict[str, dict], merge]      # promo_id -> current version of promo
    versions: Annotated[dict[str, list[dict]], merge]  # promo_id -> every version, oldest first
    pending_eval: list[str]            # promo_ids waiting for the evaluator

    # ---- evaluator loop ------------------------------------------------------------
    evaluations: Annotated[dict[str, list[dict]], merge]   # promo_id -> one entry per version
    revision_counts: Annotated[dict[str, int], merge]
    revision_queue: list[Revision]
    status: Annotated[dict[str, str], merge]   # promo_id -> passed | kept_best | revising | ...

    # ---- rank ----------------------------------------------------------------------
    promos: list[dict]                 # ranked final list
    ranking_report: dict               # considered / ranked / dropped with reasons

    # ---- editor review / feedback --------------------------------------------------
    review_round: int
    editor_decisions: Annotated[dict[str, dict], merge]   # promo_id -> {decision, comment, round}
    fix_attempts: Annotated[dict[str, int], merge]
    lessons: list[dict]                # FeedbackAgent output for memory_write
    review_complete: bool               # editor_review found nothing pending (or hit max rounds)
    review_summary: dict               # rounds, stopped_by_max_rounds, unreviewed promo ids
    review_notes: Annotated[dict, merge]   # promo_id -> note for the editor (e.g. a fix that failed)

    # ---- memory_write / bookkeeping ------------------------------------------------
    memory_written: dict
    trace: Annotated[list[dict], operator.add]          # every event, see graph/nodes.py:_event
    errors: Annotated[list[str], operator.add]


def trace_id(run_id: str, promo_id: str) -> str:
    return f"{run_id}:{promo_id}"


def promo_trace(state: dict[str, Any], promo_id: str) -> list[dict]:
    """All events for one promo, in order (Reader moments it used included)."""
    promo = (state.get("candidates") or {}).get(promo_id, {})
    moment_ids = {promo.get("hook_moment_id"), promo.get("cliffhanger_moment_id"), *promo.get("source_moment_ids", [])}
    return [e for e in state.get("trace", [])
            if e.get("promo_id") == promo_id or (e.get("moment_id") and e.get("moment_id") in moment_ids)]
