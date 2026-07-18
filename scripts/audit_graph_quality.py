#!/usr/bin/env python3
"""Read-only audit for BDH graph structural and Hebbian invariants."""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections import Counter
from datetime import datetime, timedelta


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.load(response)


def _parse_timestamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def audit_graph(
    graph: dict,
    stats: dict | None = None,
    *,
    now: datetime | None = None,
    weak_threshold: float = 0.15,
    weak_max_frequency: float = 1.0,
    weak_min_age_hours: float = 48,
) -> dict:
    """Return invariant/quality metrics without mutating the supplied payload."""
    stats = stats or {}
    nodes = {node.get("id") for node in graph.get("nodes", [])}
    structural = graph.get("edges", [])
    hebbian = graph.get("hebbian", [])

    structural_invalid = [
        edge for edge in structural
        if edge.get("source") not in nodes or edge.get("target") not in nodes
    ]
    structural_self_loops = [
        edge for edge in structural
        if edge.get("source") == edge.get("target")
    ]
    structural_keys = Counter(
        (edge.get("source"), edge.get("target"), edge.get("type", "wikilink"))
        for edge in structural
    )
    hebbian_invalid = [
        synapse for synapse in hebbian
        if synapse.get("note_a") not in nodes or synapse.get("note_b") not in nodes
    ]
    hebbian_self_loops = [
        synapse for synapse in hebbian
        if synapse.get("note_a") == synapse.get("note_b")
    ]

    effective_now = now or datetime.now()
    stale_weak = 0
    weak = 0
    strong = 0
    malformed_timestamps = 0
    for synapse in hebbian:
        try:
            is_weak = float(synapse.get("weight", 0.0)) < weak_threshold
        except (TypeError, ValueError):
            is_weak = False
        if not is_weak:
            strong += 1
            continue
        weak += 1
        try:
            low_frequency = float(synapse.get("frequency", 0.0)) <= weak_max_frequency
        except (TypeError, ValueError):
            low_frequency = False
        timestamp = _parse_timestamp(synapse.get("last_coactivated"))
        if timestamp is None:
            malformed_timestamps += 1
            continue
        comparison_now = effective_now
        if timestamp.tzinfo is None and comparison_now.tzinfo is not None:
            timestamp = timestamp.replace(tzinfo=comparison_now.tzinfo)
        elif timestamp.tzinfo is not None and comparison_now.tzinfo is None:
            comparison_now = comparison_now.replace(tzinfo=timestamp.tzinfo)
        if low_frequency and timestamp <= comparison_now - timedelta(hours=weak_min_age_hours):
            stale_weak += 1

    source_counts = Counter(
        node.get("source_id", "unknown") for node in graph.get("nodes", [])
    )
    generated_counts = Counter(
        edge.get("type", "unknown")
        for edge in structural
        if edge.get("generated")
    )

    return {
        "stats": stats,
        "nodes": len(nodes),
        "structural_edges": len(structural),
        "structural_invalid_endpoints": len(structural_invalid),
        "structural_self_loops": len(structural_self_loops),
        "structural_duplicate_edges": sum(
            count - 1 for count in structural_keys.values() if count > 1
        ),
        "hebbian_edges": len(hebbian),
        "hebbian_invalid_endpoints": len(hebbian_invalid),
        "hebbian_self_loops": len(hebbian_self_loops),
        "hebbian_strong_synapses": strong,
        "hebbian_weak_synapses": weak,
        "hebbian_stale_weak_synapses": stale_weak,
        "hebbian_malformed_timestamps": malformed_timestamps,
        "generated_edge_types": dict(generated_counts),
        "source_counts": dict(source_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8643")
    parser.add_argument("--weak-threshold", type=float, default=0.15)
    parser.add_argument("--weak-max-frequency", type=float, default=1.0)
    parser.add_argument("--weak-min-age-hours", type=float, default=48)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    report = audit_graph(
        _get_json(f"{base}/api/graph"),
        _get_json(f"{base}/api/stats"),
        weak_threshold=args.weak_threshold,
        weak_max_frequency=args.weak_max_frequency,
        weak_min_age_hours=args.weak_min_age_hours,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
