"""Read-only, conservative audit helpers for legacy Hebbian synapses."""

from __future__ import annotations

import math


def _cosine(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _pair_from_key(key: str) -> tuple[str, str] | None:
    if key.count("|") != 1:
        return None
    source, target = key.split("|", 1)
    return (source, target) if source and target else None


def audit_legacy_synapses(state: dict, *, valid_node_ids: set[str], embeddings: dict[str, list[float]], min_similarity: float) -> dict:
    """Classify legacy edges without mutating state or promoting any edge.

    Only valid edges whose endpoint embeddings meet ``min_similarity`` become
    candidates for later LLM adjudication. Everything else remains in the
    original state and is returned as a rejection record for auditability.
    """
    candidates, rejected = [], []
    for key, synapse in state.get("synapses", {}).items():
        pair = _pair_from_key(key)
        if pair is None:
            rejected.append({"key": key, "reason": "malformed_synapse_key"})
            continue
        source, target = pair
        if source not in valid_node_ids or target not in valid_node_ids:
            rejected.append({"key": key, "reason": "missing_graph_node"})
            continue
        if source not in embeddings or target not in embeddings:
            rejected.append({"key": key, "reason": "missing_embedding"})
            continue
        similarity = _cosine(embeddings[source], embeddings[target])
        record = {"key": key, "source": source, "target": target, "similarity": similarity, "synapse": synapse}
        if similarity < min_similarity:
            record["reason"] = "semantic_similarity_below_threshold"
            rejected.append(record)
        else:
            candidates.append(record)
    candidates.sort(key=lambda item: (item["similarity"], item["synapse"].get("weight", 0.0)), reverse=True)
    return {
        "summary": {"total": len(state.get("synapses", {})), "candidates": len(candidates), "rejected": len(rejected)},
        "candidates": candidates,
        "rejected": rejected,
    }


def build_curated_state(legacy_state: dict, reviews: list[dict], *, min_confidence: float) -> dict:
    """Create a new provenance-rich state from confident explicit keeps only."""
    review_by_key = {review.get("edge_key"): review for review in reviews}
    curated = {}
    for key, synapse in legacy_state.get("synapses", {}).items():
        review = review_by_key.get(key)
        if not review or review.get("verdict") != "keep" or review.get("confidence", 0.0) < min_confidence:
            continue
        promoted = dict(synapse)
        promoted["curation"] = {
            "source": "legacy_semantic_review_v1",
            "verdict": "keep",
            "confidence": review["confidence"],
            "relation_type": review.get("relation_type", "none"),
            "reason": review.get("reason", ""),
        }
        curated[key] = promoted
    return {
        "schema_version": 1,
        "kind": "legacy_curated",
        "source_state": "legacy",
        "synapses": curated,
        "curation_policy": {"verdict": "keep", "min_confidence": min_confidence},
    }
