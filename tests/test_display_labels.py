"""Tests for contextual display labels."""

from bdh_graph_harness.graph.display import (
    add_display_label,
    compose_display_label,
    infer_context_label,
)


def test_compose_display_label_preserves_title_and_adds_context():
    assert compose_display_label("README", "BDH Graph Harness") == "BDH Graph Harness — README"
    assert compose_display_label("BDH Graph Harness", "BDH Graph Harness") == "BDH Graph Harness"
    assert compose_display_label("Hebbian Learning", None) == "Hebbian Learning"


def test_external_context_uses_project_path_or_source_id():
    assert infer_context_label(
        source_type="external",
        source_id="projects",
        relative_path="bdh-graph-harness/README.md",
    ) == "BDH Graph Harness"
    assert infer_context_label(
        source_type="external",
        source_id="bdh-graph-harness",
        relative_path="README.md",
    ) == "BDH Graph Harness"


def test_vault_context_is_explicit_or_projects_scoped_only():
    assert infer_context_label(
        source_type="vault",
        source_id="vault",
        relative_path="projects/privacy-guard/overview.md",
    ) == "Privacy Guard"
    assert infer_context_label(
        source_type="vault",
        source_id="vault",
        relative_path="wiki/concepts/hebbian-learning.md",
    ) is None
    assert infer_context_label(
        source_type="vault",
        source_id="vault",
        relative_path="wiki/concepts/memory.md",
        frontmatter={"project": "Hermes Agent"},
    ) == "Hermes Agent"


def test_add_display_label_does_not_change_identity_or_title():
    node = {
        "id": "external:projects/demo/README.md",
        "title": "README",
        "source_type": "external",
        "source_id": "projects",
        "relative_path": "demo/README.md",
    }
    add_display_label(node)
    assert node["id"] == "external:projects/demo/README.md"
    assert node["title"] == "README"
    assert node["context_label"] == "Demo"
    assert node["display_label"] == "Demo — README"


def test_project_group_wins_for_vault_and_external_nodes():
    for source_type in ("vault", "external"):
        node = {
            "id": "node",
            "title": "Overview",
            "source_type": source_type,
            "source_id": "projects",
            "relative_path": "some/path.md",
            "project_group": "bdh-graph-harness",
        }
        add_display_label(node)
        assert node["display_label"] == "BDH Graph Harness — Overview"
