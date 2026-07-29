#!/usr/bin/env python3
"""Aggregate independent local review outputs into edge-keyed curation decisions."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path('/Users/albi/Projects/bdh-hebbian-legacy-curation')
PACK = ROOT / 'benchmarks/review/legacy-hebbian-llm-pack-v1.json'
SUMMARIES = [
    Path('/Users/albi/.hermes/cache/delegation/subagent-summary-0-20260726_211648_812315.txt'),
    Path('/Users/albi/.hermes/cache/delegation/subagent-summary-1-20260726_211648_812721.txt'),
    Path('/Users/albi/.hermes/cache/delegation/subagent-summary-2-20260726_211648_812948.txt'),
]
OUTPUT = ROOT / 'benchmarks/review/legacy-hebbian-review-v1.json'


def main() -> None:
    edge_by_id = {item['id']: item['edge_key'] for item in json.loads(PACK.read_text())['candidates']}
    reviews = []
    for path in SUMMARIES:
        match = re.search(r'```json\s*(\[.*?\])\s*```', path.read_text(), flags=re.DOTALL)
        if not match:
            raise SystemExit(f'No JSON review block in {path}')
        for review in json.loads(match.group(1)):
            review['edge_key'] = edge_by_id[review['id']]
            reviews.append(review)
    if len(reviews) != len(edge_by_id) or len({review['id'] for review in reviews}) != len(edge_by_id):
        raise SystemExit('Review coverage is incomplete or duplicated')
    OUTPUT.write_text(json.dumps({'metadata': {'reviewers': 3, 'mode': 'independent_local_review'}, 'reviews': reviews}, indent=2))
    print(json.dumps({'reviews': len(reviews), 'keep': sum(r['verdict'] == 'keep' for r in reviews), 'quarantine': sum(r['verdict'] == 'quarantine' for r in reviews), 'reject': sum(r['verdict'] == 'reject' for r in reviews)}))


if __name__ == '__main__':
    main()
