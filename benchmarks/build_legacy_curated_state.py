#!/usr/bin/env python3
"""Build a local, disabled curated legacy state from reviewed decisions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bdh_graph_harness.memory.curation import build_curated_state


LEGACY = Path('/Users/albi/Documents/Hermes/.bdh-state.json')
REVIEWS = ROOT / 'benchmarks/review/legacy-hebbian-review-v1.json'
OUTPUT = ROOT / 'benchmarks/results/legacy-curated-state-v1.json'


def main() -> None:
    legacy = json.loads(LEGACY.read_text())
    reviews = json.loads(REVIEWS.read_text())['reviews']
    curated = build_curated_state(legacy, reviews, min_confidence=0.80)
    curated['curation_source'] = 'legacy-hebbian-review-v1'
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(curated, indent=2))
    print(json.dumps({'promoted': len(curated['synapses']), 'output': str(OUTPUT)}))


if __name__ == '__main__':
    main()
