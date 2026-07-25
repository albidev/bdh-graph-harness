"""Contract test for legacy/single-query retrieval diagnostics in JS."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBSOCKET = ROOT / "bdh_graph_harness/visualization/templates/websocket.js"


def test_renderRetrievalDiagnostics_handles_single_query_payload():
    websocket = WEBSOCKET.read_text()
    # Ensure the diagnostics renderer references query_variants without
    # assuming the field is always present or multi-query.
    assert "payload.query_variants" in websocket
    assert "Array.isArray(payload.query_variants)" in websocket
    assert "payload.query_variants.length > 1" in websocket


def test_renderRetrievalDiagnostics_shows_no_evidence_for_empty_notes():
    websocket = WEBSOCKET.read_text()
    assert "No direct evidence" in websocket
    assert "'no-evidence'" in websocket
    assert "0 notes activated" in websocket


def test_renderRetrievalDiagnostics_displays_multi_query_provenance():
    websocket = WEBSOCKET.read_text()
    assert "Fused" in websocket
    assert "variants" in websocket
    assert "notes matched by multiple variants" in websocket
