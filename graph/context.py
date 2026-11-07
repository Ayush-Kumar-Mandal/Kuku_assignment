"""Run context: the non-serialisable things nodes need (settings, agents, LLM client).

Passed to LangGraph via `context_schema` and `graph.invoke(..., context=ctx)`, so it is
NOT checkpointed: on resume the caller builds a fresh one. Graph state stays plain data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agents.base import Agents
from config.loader import MomentTypes, ParserConfig, Settings
from parser.models import Episode


@dataclass
class RunContext:
    settings: Settings
    parser_cfg: ParserConfig
    moment_types: MomentTypes
    agents: Agents
    client: Any = None                           # llm.client.LLMClient when real agents run
    parser_fallback: Callable | None = None      # LLM fallback for unparsable lines
    memory: Any = None                           # memory.store.MemoryStore (None: no long-term memory)
    _episodes: dict = field(default_factory=dict, repr=False)

    def run_dir(self, run_id: str) -> Path:
        return self.settings.resolve(self.settings.paths.output_dir) / "runs" / run_id

    def episodes(self, state: dict) -> dict[int, Episode]:
        """Full Episode models for the run, loaded from the parse node's files (cached)."""
        index = state.get("episodes") or []
        key = (state.get("run_id"), tuple(e["number"] for e in index))
        if key not in self._episodes:
            self._episodes = {key: {
                e["number"]: Episode.model_validate_json(Path(e["parsed_path"]).read_text(encoding="utf-8"))
                for e in index}}
        return self._episodes[key]
