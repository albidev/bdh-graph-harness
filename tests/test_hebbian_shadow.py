import hashlib

from bdh_graph_harness.retrieval.shadow import build_dynamic_shadow


def test_dynamic_shadow_exposes_only_dynamic_edge_provenance_without_raw_query():
    query = "keep this private"
    routing = {
        "activation_details": [
            {"id": "seed", "role": "seed", "final_score": 1.0},
            {
                "id": "learned",
                "role": "hebbian_neighbor",
                "matched_by": "hebbian_edge",
                "parent_id": "seed",
                "hebbian_edge_weight": 0.8,
                "hebbian_edge_trust": 0.75,
                "final_score": 0.6,
            },
        ]
    }

    shadow = build_dynamic_shadow(query, routing)

    assert shadow["query_fingerprint"] == hashlib.sha256(query.encode()).hexdigest()[:16]
    assert "query" not in shadow
    assert shadow["dynamic_only_count"] == 1
    assert shadow["dynamic_only"][0] == {
        "id": "learned",
        "rank": 2,
        "parent_id": "seed",
        "weight": 0.8,
        "trust": 0.75,
        "score": 0.6,
    }
