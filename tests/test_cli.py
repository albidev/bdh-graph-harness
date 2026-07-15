"""CLI regression tests."""

from __future__ import annotations

import sys

import bdh_graph_harness.__main__ as cli


def test_serve_mode_uses_module_graph_builder_after_list_vaults_support(monkeypatch, tmp_path):
    """--serve must build the graph instead of shadowing build_graph in main()."""
    config = {
        "vault_path": str(tmp_path),
        "hybrid_search": False,
        "online_plasticity": False,
    }
    captured = {}

    monkeypatch.setattr(sys, "argv", ["bdh", "--serve"])
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr(cli, "build_graph", lambda _path, use_cache: ({"note": {}}, {"note": []}))
    monkeypatch.setattr(cli, "compute_all_embeddings", lambda _nodes, _path: "collection")
    monkeypatch.setattr(cli, "load_state", lambda _path: {"synapses": {}, "queries": 0})
    monkeypatch.setattr(
        cli,
        "start_api_server",
        lambda cfg, nodes, edges, collection, state: captured.update(
            cfg=cfg, nodes=nodes, edges=edges, collection=collection, state=state
        ),
    )

    cli.main()

    assert captured["nodes"] == {"note": {}}
    assert captured["collection"] == "collection"


def test_scan_sources_is_read_only_and_skips_runtime_pipeline(monkeypatch, tmp_path, capsys):
    vault = tmp_path / "vault"
    projects = tmp_path / "projects"
    vault.mkdir()
    projects.mkdir()
    (vault / "note.md").write_text("# Vault\n[[external:projects/demo.md]]", encoding="utf-8")
    (projects / "demo.md").write_text("# Demo\n[[vault:note]]", encoding="utf-8")
    (projects / "excluded.md").write_text("# Excluded", encoding="utf-8")

    config = {
        "vault_path": str(vault),
        "external_sources": [{
            "id": "projects",
            "path": str(projects),
            "exclude": ["excluded.md"],
        }],
        "graph_ignore": [],
    }
    monkeypatch.setattr(sys, "argv", ["bdh", "--scan-sources"])
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr(
        cli,
        "compute_all_embeddings",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("scan-only must not initialize embeddings")
        ),
    )

    cli.main()
    output = capsys.readouterr().out

    assert "external:projects: 1 Markdown" in output
    assert "vault:vault: 1 Markdown" in output
    assert "✓ Nodes: 2" in output
    assert "✓ Resolved wikilinks: 2" in output
    assert "✓ Counterpart edges: 0" in output
    assert not (vault / ".bdh-graph-cache.json").exists()
    assert not (projects / ".bdh-graph-cache.json").exists()
