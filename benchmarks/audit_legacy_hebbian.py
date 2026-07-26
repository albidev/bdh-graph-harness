#!/usr/bin/env python3
"""Read-only audit of legacy Hebbian synapses against the current graph/Chroma."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Support direct execution from ``benchmarks/`` without requiring installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
import yaml

from bdh_graph_harness.graph import build_configured_graph, migrate_legacy_state_ids
from bdh_graph_harness.vaults import normalize_vault_configs
from bdh_graph_harness.memory.curation import audit_legacy_synapses


DEFAULT_VAULT = Path("/Users/albi/Documents/Hermes")
DEFAULT_CHROMA = Path("/Users/albi/Documents/Hermes/.bdh-chroma-issue19-prod")
DEFAULT_CONFIG = Path("/Users/albi/Projects/bdh-graph-harness/.worktrees/develop-live/bdh-config.local.yaml")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--chroma", type=Path, default=DEFAULT_CHROMA)
    parser.add_argument("--collection", default="notes")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--min-similarity", type=float, default=0.60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    legacy_path = args.vault / ".bdh-state.json"
    state = json.loads(legacy_path.read_text())
    config = yaml.safe_load(args.config.read_text())
    config.setdefault("vault_path", str(args.vault))
    vault_configs = normalize_vault_configs(config)
    core = next((vault for vault in vault_configs if vault.id == "core"), vault_configs[0])
    nodes, _, _ = build_configured_graph(core.settings, use_cache=True)
    if not nodes:
        raise SystemExit("Federated graph unavailable; refusing to audit without node validation")
    state = migrate_legacy_state_ids(state, nodes)

    collection = chromadb.PersistentClient(path=str(args.chroma)).get_collection(args.collection)
    node_ids = sorted(set(nodes) & set(collection.get()["ids"]))
    payload = collection.get(ids=node_ids, include=["embeddings"])
    embeddings = dict(zip(payload["ids"], payload.get("embeddings", [])))

    audit = audit_legacy_synapses(
        state,
        valid_node_ids=set(nodes),
        embeddings=embeddings,
        min_similarity=args.min_similarity,
    )
    audit["metadata"] = {
        "source_state": str(legacy_path),
        "mode": "read_only",
        "min_similarity": args.min_similarity,
        "graph_nodes": len(nodes),
        "embeddings_loaded": len(embeddings),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
