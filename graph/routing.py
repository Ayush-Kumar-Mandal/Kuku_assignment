"""Conditional-edge functions. Pure functions of state, so they are easy to test and
the diagram shows every possible target (path maps in graph/build.py)."""
from __future__ import annotations

from graph.state import PromoState


def _n_episodes(state: PromoState) -> int:
    return len(state.get("episodes") or [])


def _queued(state: PromoState, maker: str) -> bool:
    return any(r["maker"] == maker for r in state.get("revision_queue", []))


def after_reader(state: PromoState) -> str:
    return "reader" if state.get("reader_cursor", 0) < _n_episodes(state) else "cutter"


def after_cutter(state: PromoState) -> str:
    if state.get("cutter_cursor", 0) < _n_episodes(state):
        return "cutter"                                   # next episode
    if _queued(state, "joiner"):
        return "joiner"                                   # joiner also has fixes this round
    if state.get("switches", {}).get("joiner") and not state.get("joiner_done"):
        return "joiner"                                   # first pass
    return "evaluator"


def after_evaluator(state: PromoState) -> str:
    """Score every pending promo first (batch by batch), then send failed ones back to
    whoever made them; when nothing is queued, rank."""
    if state.get("pending_eval"):
        return "evaluator"
    if _queued(state, "cutter"):
        return "cutter"
    if _queued(state, "joiner"):
        return "joiner"
    return "rank"


def after_rank(state: PromoState) -> str:
    return "editor_review" if state.get("switches", {}).get("editor_review") else "memory_write"


def after_editor_review(state: PromoState) -> str:
    """Decisions go to feedback; when review is complete (nothing pending / max rounds), finish."""
    return "memory_write" if state.get("review_complete") else "feedback"


def after_feedback(state: PromoState) -> str:
    """Editor fixes go to the maker (then evaluator -> rank -> review); otherwise rank again so
    rejected slots are back-filled and any newcomers get reviewed."""
    if _queued(state, "cutter"):
        return "cutter"
    if _queued(state, "joiner"):
        return "joiner"
    return "rank"
