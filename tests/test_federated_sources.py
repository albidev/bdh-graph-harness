"""Tests for source-aware Markdown ingestion and federated graph construction."""

from __future__ import annotations

from pathlib import Path

from bdh_graph_harness.graph.federated import build_federated_graph, migrate_legacy_state_ids
from bdh_graph_harness.graph.sources import (
    CounterpartSpec,
    ExternalMarkdownSource,
    VaultMarkdownSource,
    counterpart_specs_from_config,
    sources_from_config,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_external_source_include_exclude_and_cross_source_links(tmp_path):
    vault = tmp_path / "vault"
    projects = tmp_path / "projects"
    _write(
        vault / "wiki/concepts/bdh.md",
        "# BDH\nSee [[external:projects/demo/README]].",
    )
    _write(
        projects / "demo/README.md",
        "# Demo\nSee [[docs/design]] and [[vault:wiki/concepts/bdh]].",
    )
    _write(
        projects / "demo/docs/design.md",
        "# Design\nSee [[missing-note]].",
    )
    _write(projects / "demo/skip.md", "# Skip\nNot part of the graph.")
    _write(projects / ".hidden.md", "# Hidden\nNever scanned.")

    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    nodes, edges, unresolved = build_federated_graph(
        [
            VaultMarkdownSource(str(vault)),
            ExternalMarkdownSource(
                str(projects),
                source_id="projects",
                exclude=["demo/skip.md"],
            ),
        ]
    )
    after = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))

    assert before == after, "source scan must not copy or generate files"
    assert set(nodes) == {
        "vault:wiki/concepts/bdh.md",
        "external:projects/demo/README.md",
        "external:projects/demo/docs/design.md",
    }
    assert nodes["external:projects/demo/README.md"]["source_type"] == "external"
    assert nodes["external:projects/demo/README.md"]["source_id"] == "projects"
    assert nodes["external:projects/demo/README.md"]["relative_path"] == "demo/README.md"
    assert nodes["external:projects/demo/README.md"]["writable"] is False

    vault_targets = [link["target"] for link in edges["vault:wiki/concepts/bdh.md"]]
    readme_targets = [link["target"] for link in edges["external:projects/demo/README.md"]]
    assert vault_targets == ["external:projects/demo/README.md"]
    assert set(readme_targets) == {
        "external:projects/demo/docs/design.md",
        "vault:wiki/concepts/bdh.md",
    }
    design_edges = edges["external:projects/demo/docs/design.md"]
    assert any(
        edge["target"] == "external:projects/demo/README.md"
        and edge["type"] == "project_context"
        and edge["relation"] == "same_project"
        and edge["generated"] is True
        for edge in design_edges
    )
    assert nodes["external:projects/demo/docs/design.md"]["project_group"] == "demo"
    assert unresolved == [{
        "source": "external:projects/demo/docs/design.md",
        "target": "missing-note",
        "display": "missing-note",
        "source_path": "demo/docs/design.md",
    }]


def test_counterpart_edges_are_reciprocal_without_parent_node(tmp_path):
    vault = tmp_path / "vault"
    projects = tmp_path / "projects"
    _write(vault / "projects/demo/overview.md", "# Vault overview")
    _write(projects / "demo/README.md", "# Repository README")

    nodes, edges, unresolved = build_federated_graph(
        [
            VaultMarkdownSource(str(vault)),
            ExternalMarkdownSource(str(projects), source_id="projects"),
        ],
        counterparts=[CounterpartSpec(
            source_id="projects",
            group_id="demo",
            vault_path="projects/demo/overview.md",
            external_path="demo/README.md",
        )],
    )

    assert unresolved == []
    assert set(nodes) == {
        "vault:projects/demo/overview.md",
        "external:projects/demo/README.md",
    }
    assert nodes["vault:projects/demo/overview.md"]["project_group"] == "demo"
    assert nodes["external:projects/demo/README.md"]["project_group"] == "demo"
    assert edges["vault:projects/demo/overview.md"] == [{
        "target": "external:projects/demo/README.md",
        "type": "counterpart",
        "relation": "same_project",
        "group_id": "demo",
        "weight": 1.0,
        "explicit": False,
        "generated": True,
        "traversable": True,
    }]
    assert edges["external:projects/demo/README.md"][0]["target"] == "vault:projects/demo/overview.md"
    assert all(not node_id.startswith("project:") for node_id in nodes)


def test_all_external_project_notes_bridge_to_enriched_vault_counterpart(tmp_path):
    vault = tmp_path / "vault"
    projects = tmp_path / "projects"
    _write(vault / "projects/demo/overview.md", "# Enriched vault overview\nMore project context.")
    _write(projects / "demo/README.md", "# Repository README")
    _write(projects / "demo/docs/design.md", "# Design")

    nodes, edges, unresolved = build_federated_graph(
        [
            VaultMarkdownSource(str(vault)),
            ExternalMarkdownSource(str(projects), source_id="projects"),
        ],
        counterparts=[CounterpartSpec(
            source_id="projects",
            group_id="demo",
            vault_path="projects/demo/overview.md",
            external_path="demo/README.md",
        )],
    )

    assert unresolved == []
    design_edges = edges["external:projects/demo/docs/design.md"]
    assert any(
        edge["target"] == "vault:projects/demo/overview.md"
        and edge["type"] == "project_context"
        and edge["group_id"] == "demo"
        for edge in design_edges
    )


def test_counterpart_specs_from_config_supports_singular_and_plural_forms(tmp_path):
    vault = tmp_path / "vault"
    projects = tmp_path / "projects"
    vault.mkdir()
    projects.mkdir()
    config = {
        "vault_path": str(vault),
        "external_sources": [{
            "id": "projects",
            "path": str(projects),
            "counterparts": [{
                "group_id": "demo",
                "vault_path": "projects/demo/overview.md",
                "external_path": "demo/README.md",
            }],
        }, {
            "id": "docs",
            "path": str(projects),
            "counterpart": {
                "group_id": "docs",
                "vault_path": "projects/docs/overview.md",
                "external_path": "docs/README.md",
            },
        }],
    }

    specs = counterpart_specs_from_config(config)

    assert [(spec.source_id, spec.group_id) for spec in specs] == [
        ("projects", "demo"),
        ("docs", "docs"),
    ]
    assert specs[0].external_path == "demo/README.md"


def test_sources_from_config_supports_include_exclude_and_read_only_defaults(tmp_path):
    vault = tmp_path / "vault"
    projects = tmp_path / "projects"
    vault.mkdir()
    projects.mkdir()
    _write(projects / "docs/root.md", "# Root")
    _write(projects / "docs/private.md", "# Private")
    _write(projects / "other.md", "# Other")
    sources = sources_from_config({
        "vault_path": str(vault),
        "external_sources": [{
            "id": "selected",
            "path": str(projects),
            "include": ["docs/*.md"],
            "exclude": ["docs/private.md"],
        }],
    })

    assert len(sources) == 2
    assert sources[0].source_type == "vault"
    assert sources[0].writable is True
    assert sources[1].source_id == "selected"
    assert sources[1].source_type == "external"
    assert sources[1].writable is False
    scanned = [document.relative_path for document in sources[1].scan()]
    assert scanned == ["docs/root.md"]


def test_sources_from_config_rejects_duplicate_ids(tmp_path):
    vault = tmp_path / "vault"
    projects = tmp_path / "projects"
    vault.mkdir()
    projects.mkdir()
    config = {
        "vault_path": str(vault),
        "external_sources": [
            {"id": "projects", "path": str(projects)},
            {"id": "projects", "path": str(projects)},
        ],
    }

    import pytest
    with pytest.raises(ValueError, match="Duplicate document source id"):
        sources_from_config(config)


def test_migrate_legacy_state_ids_to_federated_vault_ids():
    state = {
        "synapses": {
            "wiki/concepts/bdh|wiki/concepts/other": {
                "weight": 0.72,
                "frequency": 4,
            }
        },
        "node_quality": {"wiki/concepts/bdh": {"score": 0.8}},
        "dormant_nodes": ["wiki/concepts/other"],
    }
    nodes = {
        "vault:wiki/concepts/bdh.md": {},
        "vault:wiki/concepts/other.md": {},
    }

    migrated = migrate_legacy_state_ids(state, nodes)

    assert "vault:wiki/concepts/bdh.md|vault:wiki/concepts/other.md" in migrated["synapses"]
    assert "vault:wiki/concepts/bdh.md" in migrated["node_quality"]
    assert migrated["dormant_nodes"] == ["vault:wiki/concepts/other.md"]
