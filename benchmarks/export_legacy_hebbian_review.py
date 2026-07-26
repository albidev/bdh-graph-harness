#!/usr/bin/env python3
"""Export high-confidence legacy candidates for local LLM adjudication."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bdh_graph_harness.graph import build_configured_graph
from bdh_graph_harness.vaults import normalize_vault_configs

DEFAULT_CONFIG = Path('/Users/albi/Projects/bdh-graph-harness/.worktrees/develop-live/bdh-config.local.yaml')


def _excerpt(node: dict, limit: int = 900) -> str:
    text = node.get('content') or node.get('text') or ''
    return ' '.join(text.split())[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--min-weight', type=float, default=0.15)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text())
    config = yaml.safe_load(args.config.read_text())
    config.setdefault('vault_path', '/Users/albi/Documents/Hermes')
    vaults = normalize_vault_configs(config)
    core = next((item for item in vaults if item.id == 'core'), vaults[0])
    nodes, _, _ = build_configured_graph(core.settings, use_cache=True)

    records = []
    for candidate in audit['candidates']:
        synapse = candidate['synapse']
        if synapse.get('weight', 0.0) < args.min_weight:
            continue
        source, target = candidate['source'], candidate['target']
        if source not in nodes or target not in nodes:
            continue
        records.append({
            'id': f'legacy-{len(records) + 1:03d}',
            'edge_key': candidate['key'],
            'similarity': candidate['similarity'],
            'weight': synapse.get('weight', 0.0),
            'frequency': synapse.get('frequency', 0.0),
            'consolidation_cycles': synapse.get('consolidation_candidate_cycles', 0),
            'source': {'id': source, 'title': nodes[source].get('title', source), 'excerpt': _excerpt(nodes[source])},
            'target': {'id': target, 'title': nodes[target].get('title', target), 'excerpt': _excerpt(nodes[target])},
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'metadata': {'mode': 'local_only', 'min_weight': args.min_weight}, 'candidates': records}, indent=2))
    print(json.dumps({'exported': len(records)}))


if __name__ == '__main__':
    main()
