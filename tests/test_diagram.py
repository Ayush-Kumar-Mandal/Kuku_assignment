"""graph.diagram: the exported page = generated Mermaid graph + the hand-written notes file."""
from graph import diagram


def test_export_appends_the_notes_and_keeps_the_generated_graph(tmp_path, monkeypatch):
    monkeypatch.setattr(diagram, "_png", lambda g: b"")          # no network in tests
    notes = tmp_path / "notes.md"
    notes.write_text("## What each step does\n\nNOTES BODY", encoding="utf-8")
    out = diagram.export(tmp_path / "docs", notes=notes)
    md = (tmp_path / "docs" / "graph.md").read_text(encoding="utf-8")
    assert md.startswith("# Pipeline graph")
    assert "```mermaid" in md and "editor_review" in md          # the real graph, from LangGraph
    assert md.index("```mermaid") < md.index("NOTES BODY")       # notes come after the graph
    assert out["markdown"].endswith("graph.md")


def test_notes_can_place_the_generated_graph_with_a_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(diagram, "_png", lambda g: b"")
    notes = tmp_path / "notes.md"
    notes.write_text(f"## Big picture\n\nOVERVIEW\n\n{diagram.MARKER}\n\n## Steps\n\nSTEPS", encoding="utf-8")
    diagram.export(tmp_path / "docs", notes=notes)
    md = (tmp_path / "docs" / "graph.md").read_text(encoding="utf-8")
    assert md.index("OVERVIEW") < md.index("editor_review") < md.index("STEPS")   # graph lands at the marker
    assert diagram.MARKER not in md and md.count("```mermaid") == 1


def test_export_works_without_a_notes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(diagram, "_png", lambda g: b"")
    diagram.export(tmp_path / "docs", notes=tmp_path / "missing.md")
    assert "```mermaid" in (tmp_path / "docs" / "graph.md").read_text(encoding="utf-8")
