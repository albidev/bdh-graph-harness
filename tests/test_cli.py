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
