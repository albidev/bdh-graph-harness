"""Contract tests for the BDH OKF validator and exporter."""

from pathlib import Path

from bdh_graph_harness.graph.okf import parse_okf_frontmatter
from bdh_graph_harness.okf import export_okf_bundle, validate_okf_bundle


def test_validator_reports_conformance_errors_without_rejecting_unknown_types(tmp_path):
    (tmp_path / "missing-type.md").write_text(
        "---\ntitle: Missing type\n---\nBody\n",
        encoding="utf-8",
    )
    (tmp_path / "index.md").write_text(
        "---\ntype: Concept\n---\n# Index\n",
        encoding="utf-8",
    )
    (tmp_path / "log.md").write_text(
        "# Log\n## 2026-08-01\n- Older\n## 2026-08-02\n- Newer\n",
        encoding="utf-8",
    )
    (tmp_path / "unknown-type.md").write_text(
        "---\ntype: VendorSpecificThing\n---\n# Valid\n",
        encoding="utf-8",
    )

    result = validate_okf_bundle(tmp_path)

    assert result.valid is False
    codes = {issue.code for issue in result.errors}
    assert "missing_frontmatter_type" in codes
    assert "reserved_frontmatter" in codes
    assert "log_dates_not_descending" in codes
    assert not any(
        issue.path == "unknown-type.md" and issue.code == "unknown_type"
        for issue in result.errors
    )


def test_exporter_writes_sanitized_bundle_with_index_log_and_static_links(tmp_path):
    nodes = {
        "concepts/alpha": {
            "id": "concepts/alpha",
            "title": "Alpha",
            "text": "Alpha explains the exported concept.\n",
            "okf": {
                "type": "Concept",
                "description": "A public alpha concept.",
                "tags": ["public", "okf"],
                "generated": {
                    "by": "process:bdh-okf-export",
                    "at": "2026-08-02T12:00:00Z",
                },
                "verified": {
                    "by": "human:reviewer",
                    "at": "2026-08-02T13:00:00Z",
                },
                "status": "stable",
                "stale_after": "2027-01-01",
                "sources": [
                    {
                        "id": "local-private",
                        "resource": "/Users/albi/Documents/private/source.md",
                        "title": "Private source",
                        "author": "human:reviewer",
                        "token": "must-not-leak",
                    },
                    {
                        "id": "public-doc",
                        "resource": "https://example.com/public-doc",
                        "title": "Public documentation",
                    },
                ],
                "private_extension": {"api_key": "must-not-leak"},
            },
            "activated_from_ids": ["private/session/source"],
            "path": "/Users/albi/Documents/private/alpha.md",
        },
        "concepts/beta": {
            "id": "concepts/beta",
            "title": "Beta",
            "text": "Beta body.\n",
            "okf": {"type": "Reference"},
        },
    }
    edges = {
        "concepts/alpha": [
            {"target": "concepts/beta", "display": "Beta"},
            {
                "target": "concepts/beta",
                "display": "learned relation",
                "type": "hebbian",
                "weight": 0.91,
            },
        ],
    }

    result = export_okf_bundle(
        nodes,
        edges,
        tmp_path / "bundle",
        generated_at="2026-08-02T14:00:00Z",
        log_entries=[
            {
                "date": "2026-08-02",
                "kind": "Export",
                "message": "Exported the public OKF bundle.",
            }
        ],
    )

    assert result.files == [
        "concepts/alpha.md",
        "concepts/beta.md",
        "index.md",
        "log.md",
    ]
    assert result.warnings

    validation = validate_okf_bundle(tmp_path / "bundle")
    assert validation.valid is True, validation.errors

    alpha_path = tmp_path / "bundle" / "concepts/alpha.md"
    alpha_content = alpha_path.read_text(encoding="utf-8")
    alpha_metadata = parse_okf_frontmatter(alpha_content)
    assert alpha_metadata["type"] == "Concept"
    assert alpha_metadata["sources"][0]["resource"].startswith("redacted://local/")
    assert alpha_metadata["sources"][1]["resource"] == "https://example.com/public-doc"
    assert "must-not-leak" not in alpha_content
    assert "/Users/albi" not in alpha_content
    assert "activated_from_ids" not in alpha_content
    assert "[Beta](beta.md)" in alpha_content
    assert "hebbian" not in alpha_content

    index_content = (tmp_path / "bundle" / "index.md").read_text(encoding="utf-8")
    assert parse_okf_frontmatter(index_content) == {"okf_version": "0.2"}
    assert "[Alpha](concepts/alpha.md)" in index_content
    assert "A public alpha concept." in index_content

    log_content = (tmp_path / "bundle" / "log.md").read_text(encoding="utf-8")
    assert "## 2026-08-02" in log_content
    assert "**Export**: Exported the public OKF bundle." in log_content


def test_exporter_redacts_absolute_node_ids(tmp_path):
    result = export_okf_bundle(
        {
            "/Users/albi/Documents/private/note": {
                "id": "/Users/albi/Documents/private/note",
                "text": "Private path must not become a public bundle path.",
                "okf": {"type": "Concept"},
            }
        },
        {},
        tmp_path / "bundle",
        generated_at="2026-08-02T14:00:00Z",
    )

    assert len(result.files) == 3
    assert result.files[0].startswith("concepts/redacted-")
    assert "Users" not in result.files[0]
    assert validate_okf_bundle(tmp_path / "bundle").valid
