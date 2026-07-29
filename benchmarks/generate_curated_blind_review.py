#!/usr/bin/env python3
"""Create a local A/B blind pack: curated legacy edge vs embedding neighbor."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import chromadb
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bdh_graph_harness.graph import build_configured_graph
from bdh_graph_harness.vaults import normalize_vault_configs

VAULT = Path('/Users/albi/Documents/Hermes')
CONFIG = Path('/Users/albi/Projects/bdh-graph-harness/.worktrees/develop-live/bdh-config.local.yaml')
CURATED = ROOT / 'benchmarks/results/legacy-curated-state-v1.json'
OUT = ROOT / 'benchmarks/review/legacy-hebbian-curated-blind-v1.json'
KEY = ROOT / 'benchmarks/review/legacy-hebbian-curated-blind-v1.key.json'
CHROMA = VAULT / '.bdh-chroma-issue19-prod'


def excerpt(node: dict) -> str:
    return ' '.join((node.get('content') or node.get('text') or '').split())[:750]


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    config.setdefault('vault_path', str(VAULT))
    vaults = normalize_vault_configs(config)
    core = next(item for item in vaults if item.id == 'core')
    nodes, _, _ = build_configured_graph(core.settings, use_cache=True)
    collection = chromadb.PersistentClient(str(CHROMA)).get_collection('notes')
    rng = random.Random(20260726)
    records, key = [], []
    for index, edge_key in enumerate(json.loads(CURATED.read_text())['synapses'], 1):
        source, curated_target = edge_key.split('|', 1)
        embedding = collection.get(ids=[source], include=['embeddings'])['embeddings'][0]
        query = collection.query(query_embeddings=[embedding], n_results=12, include=[])
        baseline = next((item for item in query['ids'][0] if item not in {source, curated_target} and item in nodes), None)
        if not baseline or source not in nodes or curated_target not in nodes:
            continue
        choices = [('curated', curated_target), ('embedding', baseline)]
        rng.shuffle(choices)
        records.append({'id': f'curated-{index:03d}', 'source': {'title': nodes[source].get('title', source), 'excerpt': excerpt(nodes[source])}, 'candidate_a': {'title': nodes[choices[0][1]].get('title', choices[0][1]), 'excerpt': excerpt(nodes[choices[0][1]])}, 'candidate_b': {'title': nodes[choices[1][1]].get('title', choices[1][1]), 'excerpt': excerpt(nodes[choices[1][1]])}})
        key.append({'id': f'curated-{index:03d}', 'candidate_a': choices[0][0], 'candidate_b': choices[1][0]})
    OUT.write_text(json.dumps({'records': records}, indent=2))
    KEY.write_text(json.dumps({'key': key}, indent=2))
    print(json.dumps({'pairs': len(records)}))

if __name__ == '__main__':
    main()
