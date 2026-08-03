"""Contract tests for the BDH OKF compatibility layer."""

from pathlib import Path

from bdh_graph_harness.graph.builder import build_graph
from bdh_graph_harness.graph.federated import build_federated_graph
from bdh_graph_harness.graph.okf import (
    extract_markdown_links,
    is_reserved_filename,
    parse_okf_frontmatter,
)
from bdh_graph_harness.graph.sources import VaultMarkdownSource


def test_parse_okf_frontmatter_preserves_typed_nested_metadata():
    content = """---
type: Concept
title: Hebbian dynamic edges
description: Learned relations strengthened by repeated co-activation.
tags: [bdh, graph-memory]
generated:
  by: process:bdh-neurogenesis
  at: 2026-08-02T12:00:00Z
verified:
  by: human:reviewer
  at: 2026-08-02T13:00:00Z
sources:
  - id: source-a
    resource: /concepts/source-a.md
    title: Source A
custom:
  nested:
    keep: true
---
# Body
"""

    metadata = parse_okf_frontmatter(content)

    assert metadata["type"] == "Concept"
    assert metadata["tags"] == ["bdh", "graph-memory"]
    assert metadata["generated"]["by"] == "process:bdh-neurogenesis"
    assert metadata["sources"][0]["resource"] == "/concepts/source-a.md"
    assert metadata["verified"] == [{
        "by": "human:reviewer",
        "at": "2026-08-02T13:00:00Z",
    }]
    assert metadata["custom"] == {"nested": {"keep": True}}


def test_parse_okf_frontmatter_returns_empty_mapping_without_frontmatter():
    assert parse_okf_frontmatter("# No metadata\n") == {}


def test_extract_markdown_links_ignores_external_urls_and_code_examples():
    content = """See [Target](/concepts/target.md) and [Sibling](./sibling.md).

[External](https://example.com/docs)
`[Inline](/ignored.md)`

```markdown
[Fenced](/ignored-too.md)
```
"""

    assert extract_markdown_links(content) == [
        ("/concepts/target.md", "Target"),
        ("./sibling.md", "Sibling"),
    ]


def test_okf_reserved_filenames_are_reserved_at_any_directory_level():
    assert is_reserved_filename("index.md") is True
    assert is_reserved_filename("nested/index.md") is True
    assert is_reserved_filename("nested/log.md") is True
    assert is_reserved_filename("nested/concept.md") is False


def test_federated_okf_read_path_keeps_metadata_and_resolves_markdown_links(tmp_path):
    vault = Path(tmp_path)
    (vault / "index.md").write_text(
        "# Bundle index\n* [Source](/concepts/source.md) - source\n",
        encoding="utf-8",
    )
    (vault / "log.md").write_text(
        "# Update Log\n## 2026-08-02\n* Added concepts.\n",
        encoding="utf-8",
    )
    concepts = vault / "concepts"
    concepts.mkdir()
    (concepts / "source.md").write_text(
        """---
type: Concept
title: Source
description: Source concept.
status: stable
generated:
  by: process:fixture
  at: 2026-08-02T12:00:00Z
verified:
  by: human:reviewer
  at: 2026-08-02T13:00:00Z
---
See [Target](./target.md) and [External](https://example.com).
""",
        encoding="utf-8",
    )
    (concepts / "target.md").write_text(
        "---\ntype: Reference\ntitle: Target\n---\n# Target\n",
        encoding="utf-8",
    )

    nodes, edges, unresolved = build_federated_graph(
        [VaultMarkdownSource(str(vault))],
        okf_mode=True,
    )

    source_id = "vault:concepts/source.md"
    target_id = "vault:concepts/target.md"
    assert set(nodes) == {source_id, target_id}
    assert nodes[source_id]["title"] == "Source"
    assert nodes[source_id]["okf"]["generated"]["by"] == "process:fixture"
    assert nodes[source_id]["okf"]["verified"] == [{
        "by": "human:reviewer",
        "at": "2026-08-02T13:00:00Z",
    }]
    assert any(edge["target"] == target_id for edge in edges[source_id])
    assert unresolved == []


def test_legacy_okf_read_path_keeps_legacy_ids_and_resolves_markdown_links(tmp_path):
    vault = Path(tmp_path)
    (vault / "index.md").write_text("# Bundle index\n", encoding="utf-8")
    concepts = vault / "concepts"
    concepts.mkdir()
    (concepts / "source.md").write_text(
        """---
 type: Concept
 title: Source
 status: stable
 generated:
   by: process:fixture
 ---
 See [Target](./target.md).
 """.replace("\n ", "\n"),
        encoding="utf-8",
    )
    (concepts / "target.md").write_text(
        "---\ntype: Reference\ntitle: Target\n---\n# Target\n",
        encoding="utf-8",
    )

    nodes, edges = build_graph(str(vault), use_cache=False, okf_mode=True)

    assert set(nodes) == {"concepts/source", "concepts/target"}
    assert nodes["concepts/source"]["okf"]["type"] == "Concept"
    assert any(edge["target"] == "concepts/target" for edge in edges["concepts/source"])
