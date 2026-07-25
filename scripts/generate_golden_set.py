"""Generate a draft golden set by sampling real vault notes.

Queries are written with reviewed:false and must be human-validated before
being used as authoritative ground truth.
"""
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bdh_graph_harness.graph.builder import build_graph

VAULT_PATH = '/Users/albi/Documents/Hermes'


def main():
    nodes, edges = build_graph(VAULT_PATH, use_cache=True)

    prefix_to_category = {
        'wiki/concepts': 'concept',
        'wiki/entities': 'entity',
        'knowledge/': 'entity',
        'projects/': 'activity',
        'memory/learned': 'activity',
        'works/': 'activity',
    }

    def note_category(note_id: str) -> str:
        for prefix, cat in prefix_to_category.items():
            if note_id.startswith(prefix):
                return cat
        # Notes that link to several concepts become cross-reference candidates.
        if len(edges.get(note_id, [])) >= 2:
            return 'crossref'
        return 'news'  # catch-all for everything else (rare in this vault)

    by_category = {cat: [] for cat in ['concept', 'entity', 'news', 'activity', 'crossref']}

    for nid, node in nodes.items():
        if 'index' in nid or 'log' in nid:
            continue
        text = node.get('text', '')
        if len(text) < 80:
            continue
        cat = note_category(nid)
        by_category[cat].append((nid, node.get('title', nid), text[:400]))

    random.seed(2026)
    queries = []
    targets = [('concept', 10), ('entity', 10), ('news', 3), ('activity', 10), ('crossref', 7)]
    coverage_warnings = []
    for cat, target in targets:
        pool = by_category[cat]
        if len(pool) < target:
            coverage_warnings.append(
                f"{cat}: only {len(pool)} candidates available (target {target})"
            )
        samples = random.sample(pool, min(target, len(pool)))
        for nid, title, _ in samples:
            queries.append({
                'query': title.strip(),
                'category': cat,
                'relevant_note_ids': [nid],
                'reviewed': False,
                'source_note_id': nid,
                'rationale': 'Sampled note title as query; needs human review',
            })

    # Enrich cross-reference queries with outlinks as additional relevant notes.
    enriched = []
    for q in queries:
        if q['category'] != 'crossref':
            enriched.append(q)
            continue
        nid = q['relevant_note_ids'][0]
        outlinks = [
            e['target'] for e in edges.get(nid, [])
            if isinstance(e.get('target'), str)
        ]
        enriched.append({
            **q,
            'query': f"Come si collega {q['query']} agli altri concetti?",
            'relevant_note_ids': [nid] + outlinks[:2],
            'rationale': 'LLM-style cross-reference; needs human review',
        })
    queries = enriched

    counts = {cat: len([q for q in queries if q['category'] == cat]) for cat in sorted({q['category'] for q in queries})}
    print('Counts:', counts)
    print('Total:', len(queries))
    if coverage_warnings:
        print('Coverage warnings:', coverage_warnings)

    output = {
        'version': '2026-07-25-v1-draft',
        'review_status': 'pending',
        'auto_generated': True,
        'vault_path': VAULT_PATH,
        'vault_size': len(nodes),
        'coverage_warnings': coverage_warnings,
        'review_checklist': [
            'Verify each query has a single clear answer in the vault.',
            'Confirm relevant_note_ids are ordered by relevance where meaningful.',
            'Add missing news/activity notes if the vault snapshot is incomplete.',
            'Rewrite title-as-query entries into natural questions where needed.',
            'Promote reviewed:false to reviewed:true only after running a sample query.',
        ],
        'queries': queries,
    }

    out_path = Path(__file__).resolve().parent.parent / 'benchmarks' / 'golden_set.yaml'
    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.dump(output, f, allow_unicode=True, sort_keys=False)
    print('Wrote', out_path)


if __name__ == '__main__':
    main()
