"""Tests for conservative, read-only legacy Hebbian curation."""

from bdh_graph_harness.memory.curation import audit_legacy_synapses


def test_audit_rejects_strong_but_semantically_distant_synapse():
    state = {
        "synapses": {
            "alpha|beta": {
                "weight": 0.9,
                "frequency": 12,
                "last_coactivated": "2026-07-26T12:00:00",
                "consolidation_candidate_cycles": 3,
            }
        }
    }
    embeddings = {"alpha": [1.0, 0.0], "beta": [0.0, 1.0]}

    audit = audit_legacy_synapses(
        state,
        valid_node_ids={"alpha", "beta"},
        embeddings=embeddings,
        min_similarity=0.60,
    )

    assert audit["summary"]["rejected"] == 1
    assert audit["rejected"][0]["reason"] == "semantic_similarity_below_threshold"


def test_build_curated_state_promotes_only_confident_keep_verdicts():
    from bdh_graph_harness.memory.curation import build_curated_state

    legacy = {"synapses": {"alpha|beta": {"weight": 0.7}, "alpha|gamma": {"weight": 0.8}}}
    reviews = [
        {"edge_key": "alpha|beta", "verdict": "keep", "confidence": 0.9, "relation_type": "implementation"},
        {"edge_key": "alpha|gamma", "verdict": "quarantine", "confidence": 0.99, "relation_type": "none"},
    ]

    curated = build_curated_state(legacy, reviews, min_confidence=0.8)

    assert set(curated["synapses"]) == {"alpha|beta"}
    assert curated["synapses"]["alpha|beta"]["curation"]["verdict"] == "keep"
    assert legacy["synapses"] == {"alpha|beta": {"weight": 0.7}, "alpha|gamma": {"weight": 0.8}}
