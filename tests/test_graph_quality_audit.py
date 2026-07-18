from datetime import datetime, timedelta

from scripts.audit_graph_quality import audit_graph


def test_audit_graph_reports_structural_and_hebbian_invariants():
    now = datetime(2026, 7, 18, 12, 0, 0)
    graph = {
        "nodes": [
            {"id": "a", "source_id": "vault"},
            {"id": "b", "source_id": "projects"},
        ],
        "edges": [
            {"source": "a", "target": "a", "type": "wikilink"},
            {"source": "a", "target": "b", "type": "counterpart", "generated": True},
        ],
        "hebbian": [
            {
                "note_a": "a",
                "note_b": "b",
                "weight": 0.08,
                "frequency": 0.3,
                "last_coactivated": (now - timedelta(hours=49)).isoformat(),
            },
            {
                "note_a": "a",
                "note_b": "missing",
                "weight": 0.9,
                "frequency": 3,
                "last_coactivated": now.isoformat(),
            },
        ],
    }

    report = audit_graph(graph, {"queries_processed": 4}, now=now)

    assert report["structural_self_loops"] == 1
    assert report["structural_invalid_endpoints"] == 0
    assert report["hebbian_invalid_endpoints"] == 1
    assert report["hebbian_stale_weak_synapses"] == 1
    assert report["generated_edge_types"] == {"counterpart": 1}
