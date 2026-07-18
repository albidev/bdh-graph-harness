import asyncio
from types import SimpleNamespace

import bdh_graph_harness.api.routes as routes
from bdh_graph_harness.neurogenesis.merge import assimilate_evidence, looks_conflicting


def test_assimilate_evidence_is_append_only(tmp_path):
    note = tmp_path / "wiki" / "concepts" / "canonical.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        "---\ntitle: Canonical Concept\nupdated: 2026-01-01\n---\n\nOriginal definition.\n",
        encoding="utf-8",
    )
    result = assimilate_evidence(
        tmp_path,
        "wiki/concepts/canonical",
        {"absolute_path": str(note), "title": "Canonical Concept"},
        "A second evidence statement.",
        source_notes=["Session A"],
        source_node_ids=["vault:source-a.md", "external:projects/demo.md"],
        query="recovery query",
    )
    content = note.read_text(encoding="utf-8")
    assert result["status"] == "merged"
    assert "Original definition." in content
    assert "A second evidence statement." in content
    assert "source: Session A" in content
    assert 'activated_from_ids: ["vault:source-a.md", "external:projects/demo.md"]' in content
    assert "updated: 2026-01-01" not in content

    again = assimilate_evidence(
        tmp_path,
        "wiki/concepts/canonical",
        {"absolute_path": str(note), "title": "Canonical Concept"},
        "A second evidence statement.",
    )
    assert again["status"] == "already_present"
    assert content == note.read_text(encoding="utf-8")


def test_conflict_guard_is_conservative():
    assert looks_conflicting("Instead, replace the previous architecture.")
    assert looks_conflicting("Invece usare il counterpart edge.")
    assert not looks_conflicting("Adds a second evidence statement.")


def test_run_neurogenesis_merges_exact_title_without_creating_note(monkeypatch, tmp_path):
    note = tmp_path / "wiki" / "concepts" / "canonical.md"
    note.parent.mkdir(parents=True)
    note.write_text("---\ntitle: Canonical Concept\n---\n\nOriginal.\n", encoding="utf-8")
    node = {
        "title": "Canonical Concept",
        "absolute_path": str(note),
        "relative_path": "wiki/concepts/canonical.md",
    }
    ctx = SimpleNamespace(
        nodes={"wiki/concepts/canonical": node},
        config=SimpleNamespace(
            path=str(tmp_path),
            settings={"neurogenesis_enabled": True, "neurogenesis_dir": "wiki/concepts"},
        ),
    )
    monkeypatch.setattr(
        routes,
        "extract_new_concepts",
        lambda *args, **kwargs: [{
            "title": "Canonical Concept",
            "definition": "New evidence from a session.",
        }],
    )
    monkeypatch.setattr(
        routes,
        "create_note",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must merge, not create")),
    )

    result = routes.run_neurogenesis(
        "response", "query", {"wiki/concepts/canonical": 0.9}, ctx
    )
    assert result[0]["merged"] is True
    assert "New evidence from a session." in note.read_text(encoding="utf-8")


def test_run_neurogenesis_merges_high_similarity_candidate(monkeypatch, tmp_path):
    note = tmp_path / "wiki" / "concepts" / "canonical.md"
    note.parent.mkdir(parents=True)
    note.write_text("---\ntitle: Canonical Concept\n---\n\nOriginal.\n", encoding="utf-8")
    node = {"title": "Canonical Concept", "absolute_path": str(note)}
    ctx = SimpleNamespace(
        nodes={"wiki/concepts/canonical": node},
        config=SimpleNamespace(
            path=str(tmp_path),
            settings={"neurogenesis_enabled": True, "neurogenesis_dir": "wiki/concepts"},
        ),
    )
    monkeypatch.setattr(
        routes,
        "extract_new_concepts",
        lambda *args, **kwargs: [{
            "title": "Equivalent Concept",
            "definition": "Evidence compatible with the canonical concept.",
        }],
    )
    monkeypatch.setattr(
        routes,
        "find_semantic_match",
        lambda *args, **kwargs: {
            "node_id": "wiki/concepts/canonical",
            "title": "Canonical Concept",
            "similarity": 0.91,
        },
    )
    monkeypatch.setattr(
        routes,
        "create_note",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must assimilate")),
    )

    result = routes.run_neurogenesis("response", "query", {}, ctx)
    assert result[0]["merged"] is True
    assert result[0]["similarity"] == 0.91
    assert "Evidence compatible" in note.read_text(encoding="utf-8")
