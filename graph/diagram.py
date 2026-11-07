"""Export the graph diagram for the report.

    python -m graph.diagram [--out docs] [--notes docs/graph_notes.md]

Writes graph.mmd (Mermaid source, renders on GitHub), graph.md (the Mermaid graph plus the
hand-written notes in docs/graph_notes.md) and graph.png. The graph itself always comes from
LangGraph, so it can't drift from the code; edit the notes file, not graph.md. The PNG is
rendered by the mermaid.ink web service that LangGraph uses by default; if that's unreachable
the .mmd is still written and the PNG is skipped with a message.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from graph.build import build_graph

DEFAULT_NOTES = "docs/graph_notes.md"
MARKER = "<!-- PIPELINE_GRAPH -->"            # in the notes file: put the generated graph here


def _png(g) -> bytes:
    return g.draw_mermaid_png()


def export(out_dir: str | Path = "docs", notes: str | Path = DEFAULT_NOTES) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g = build_graph().get_graph()
    mmd = g.draw_mermaid()
    (out / "graph.mmd").write_text(mmd, encoding="utf-8")
    graph_block = "```mermaid\n" + mmd + "\n```"
    notes = Path(notes)
    text = notes.read_text(encoding="utf-8").strip() if notes.is_file() else ""
    if MARKER in text:                       # the notes say where the generated graph goes
        page = "# Pipeline graph\n\n" + text.replace(MARKER, graph_block) + "\n"
    else:                                    # otherwise: graph first, notes after it
        page = "# Pipeline graph\n\n" + graph_block + "\n" + ("\n" + text + "\n" if text else "")
    (out / "graph.md").write_text(page, encoding="utf-8")
    result = {"mermaid": str(out / "graph.mmd"), "markdown": str(out / "graph.md")}
    try:
        (out / "graph.png").write_bytes(_png(g))
        result["png"] = str(out / "graph.png")
    except Exception as e:  # network / service problem: the Mermaid source is enough
        result["png_error"] = f"{type(e).__name__}: {e}"
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs")
    ap.add_argument("--notes", default=DEFAULT_NOTES)
    args = ap.parse_args()
    for k, v in export(args.out, args.notes).items():
        print(f"{k}: {v}")
