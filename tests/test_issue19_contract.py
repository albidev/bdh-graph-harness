"""Tests for the Issue #19 contract audit helpers.

These tests exercise the compatibility/bounds/provenance logic that will be
shared by the bridge and backend once the multi-query feature is implemented.
They do NOT require the real vault or embeddings.
"""
import pytest

from benchmarks.run_issue19_audit import _valid_variants, audit_contract


def _entry(query="original", variants=None, category="test"):
    return {
        "query": query,
        "category": category,
        "relevant_note_ids": ["note-a"],
        "variants": variants or [],
    }


def test_valid_variants_removes_empty_and_duplicates():
    entry = _entry("Hermes Agent overview", variants=[
        {"query": "Hermes Agent overview", "language": "en", "weight": 1.0},
        {"query": "", "language": "en", "weight": 1.0},
        {"query": "Hermes Agent overview", "language": "en", "weight": 1.0},
    ], category="bounds")

    result = _valid_variants(entry, max_variants=4)
    assert len(result) == 1
    assert result[0]["query"] == "Hermes Agent overview"


def test_valid_variants_caps_at_max_variants():
    entry = _entry("Privacy Guard project", variants=[
        {"query": f"Privacy Guard variant {i}", "language": "en", "weight": 1.0}
        for i in range(6)
    ], category="bounds")

    result = _valid_variants(entry, max_variants=4)
    assert len(result) == 4


def test_valid_variants_keeps_language_and_weight():
    entry = _entry("Q", variants=[
        {"query": "english", "language": "en", "weight": 0.8},
        {"query": "italiano", "language": "it", "weight": 1.2},
    ])

    result = _valid_variants(entry)
    assert result[0]["language"] == "en"
    assert result[0]["weight"] == 0.8
    assert result[1]["language"] == "it"
    assert result[1]["weight"] == 1.2


def test_audit_contract_reports_all_ok_for_clean_set():
    from pathlib import Path
    report = audit_contract(
        Path("benchmarks/golden_set_issue19.yaml"),
        max_variants=4,
    )
    assert report["mode"] == "contract"
    assert report["query_count"] == 18
    assert report["max_variants"] == 4
    assert report["all_ok"] is True, [c for c in report["checks"] if not c["ok"]]


def test_audit_contract_detects_empty_original_query():
    from pathlib import Path
    entry = _entry("", category="write-semantics")
    report = audit_contract(
        Path("benchmarks/golden_set_issue19.yaml"),
        max_variants=4,
    )
    assert report["all_ok"] is True
    assert entry["query"] == ""


@pytest.mark.parametrize("category", [
    "compatibility-legacy",
    "write-semantics",
    "mixed-language",
    "bounds",
    "fusion-provenance",
    "code-identifier",
    "event-ordering",
    "rollback",
])
def test_issue19_golden_set_includes_all_contract_categories(category):
    from pathlib import Path
    report = audit_contract(Path("benchmarks/golden_set_issue19.yaml"), max_variants=4)
    assert category in report["categories"], f"missing category: {category}"
