"""Assemble the StateGraph.

  parse -> memory_read -> reader (loops per episode) -> cutter (loops per episode)
        -> joiner -> evaluator -> rank -> editor_review -> feedback -> memory_write

Loops and back-edges:
  reader    -> reader                  until every episode is read (summary carries over)
  cutter    -> cutter                  until every episode is cut
  evaluator -> cutter | joiner         failed promo back to its maker with the fix
  feedback  -> cutter | joiner         editor "fix" back to the maker, then re-evaluated
  rank      -> memory_write            when editor review is switched off
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from graph import nodes, routing
from graph.context import RunContext
from graph.state import PromoState

NODES = ["parse", "memory_read", "reader", "cutter", "joiner", "evaluator",
         "rank", "editor_review", "feedback", "memory_write"]


def build_graph(checkpointer=None):
    g = StateGraph(PromoState, context_schema=RunContext)
    for name in NODES:
        g.add_node(name, getattr(nodes, name))

    g.add_edge(START, "parse")
    g.add_edge("parse", "memory_read")
    g.add_edge("memory_read", "reader")
    # Path maps list every target explicitly so the exported diagram is complete.
    g.add_conditional_edges("reader", routing.after_reader, ["reader", "cutter"])
    g.add_conditional_edges("cutter", routing.after_cutter, ["cutter", "joiner", "evaluator"])
    g.add_edge("joiner", "evaluator")
    g.add_conditional_edges("evaluator", routing.after_evaluator, ["evaluator", "cutter", "joiner", "rank"])
    g.add_conditional_edges("rank", routing.after_rank, ["editor_review", "memory_write"])
    g.add_conditional_edges("editor_review", routing.after_editor_review, ["feedback", "memory_write"])
    g.add_conditional_edges("feedback", routing.after_feedback, ["cutter", "joiner", "rank"])
    g.add_edge("memory_write", END)
    return g.compile(checkpointer=checkpointer)
