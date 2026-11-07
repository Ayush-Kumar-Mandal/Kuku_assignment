"""The interface graph nodes use to reach agents.

Nodes never call an LLM directly; they call these methods. Step 3 runs the graph with
StubAgents (agents/stubs.py); agents/llm_agents.py is the LLM-backed implementation with the same
signatures. All inputs/outputs are plain dicts, the same shape stored in graph state.
"""
from __future__ import annotations

from typing import Protocol

from parser.models import Episode


class AgentError(RuntimeError):
    """An agent could not produce a valid answer (e.g. retries exhausted). Nodes catch it,
    record it in state["errors"] and carry on; rate-limit stops and config errors are NOT
    AgentErrors: they stop the run so it can be resumed from the checkpoint."""


class Agents(Protocol):
    def read_episode(self, episode: Episode, story_summary: str, memory: dict) -> dict:
        """-> {"moments": [moment], "new_viewer_context": str, "story_summary": str}
        moment = {moment_id, episode, kind: hook|cliffhanger, timestamp, description,
                  strength 1-10, moment_type}"""

    def cut_episode(self, episode: Episode, moments: list[dict], memory: dict) -> list[dict]:
        """-> single-segment promos (see promo shape in agents/stubs.py)."""

    def join(self, episodes: dict[int, Episode], moments: list[dict], candidates: list[dict],
             memory: dict) -> list[dict]:
        """-> multi-segment promos, each with `join_rationale`."""

    def revise(self, maker: str, promo: dict, instruction: str, episodes: dict[int, Episode],
               memory: dict, source: str = "evaluator") -> dict:
        """-> the same promo_id with a new version applying `instruction`.
        source: "evaluator" (automatic loop) or "editor" (human Fix in review)."""

    def interpret_fix(self, promo: dict, comment: str, episodes: dict[int, Episode], memory: dict) -> dict:
        """FeedbackAgent, job 1: the editor's Fix comment -> {interpretation, instruction}:
        what the editor wants, and a precise edit (real timestamps) for the maker."""

    def learn(self, new_feedback: list[dict], older_feedback: list[dict], rules: list[dict], context: dict) -> dict:
        """FeedbackAgent, job 2: editor feedback -> {relations: [{feedback_id, rule_id, relation:
        supports|contradicts}], new_rules: [{text, source_feedback_ids, conflicts_with}],
        conflicts: [{rule_a, rule_b, why}]}. Code decides scope, status and counts."""

    def evaluate(self, promo_text: str, promo: dict, memory: dict) -> dict:
        """-> {scores: {hook, cliffhanger, standalone}, passed, fix_instruction}.
        `promo_text` is ONLY what a viewer would see; nothing else about the show."""
