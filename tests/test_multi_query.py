"""Tests for multi-query retrieval module."""
import pytest

from bdh_graph_harness.retrieval.multi_query import (
    QueryVariant,
    _gather_variant_results,
    normalize_query_variants,
    rrb_fuse,
    weighted_fuse,
    count_multivariant_hits,
)


# ---------------------------------------------------------------------------
# normalize_query_variants
# ---------------------------------------------------------------------------

def test_normalize_keeps_original_query():
    variants = normalize_query_variants('original query')
    assert len(variants) == 1
    assert variants[0].query == 'original query'
    assert variants[0].language == 'original'


def test_normalize_accepts_dict_variants():
    variants = normalize_query_variants(
        'original query',
        [
            {'query': 'variant one', 'language': 'en', 'weight': 1.5},
            {'query': 'variant two', 'language': 'it'},
        ],
    )
    assert len(variants) == 3
    assert variants[1].query == 'variant one'
    assert variants[1].language == 'en'
    assert variants[1].weight == 1.5
    assert variants[2].language == 'it'


def test_normalize_dedupes_exact_matches():
    variants = normalize_query_variants(
        'Original Query',
        [{'query': 'original query'}, 'original query'],
    )
    assert len(variants) == 1


def test_normalize_drops_empty_and_invalid():
    variants = normalize_query_variants(
        'base',
        [{'query': ''}, {'query': '  '}, {'foo': 'bar'}, 'valid variant', 123],
    )
    assert len(variants) == 2
    assert variants[1].query == 'valid variant'


def test_normalize_bounds_max_variants():
    variants = normalize_query_variants(
        'base',
        [{'query': f'v{i}'} for i in range(10)],
        max_variants=3,
    )
    assert len(variants) == 3


def test_normalize_coerces_bad_weight():
    variants = normalize_query_variants(
        'base',
        [{'query': 'v', 'weight': 'not-a-number'}],
    )
    assert variants[1].weight == 1.0


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def test_rrb_fuse_ranks_and_normalizes():
    v = QueryVariant
    results = [
        (v('a'), {'x': 0.9, 'y': 0.8}),
        (v('b'), {'x': 0.7, 'z': 0.6}),
    ]
    fused = rrb_fuse(results, k=60)
    # x appears in both variants, so it should be top
    assert fused['x'] == 1.0
    assert fused['y'] == fused['z']
    assert fused['y'] < fused['x']


def test_rrb_fuse_empty():
    assert rrb_fuse([]) == {}


def test_weighted_fuse_prefers_max_score():
    v = QueryVariant
    results = [
        (v('a', weight=1.0), {'x': 0.9}),
        (v('b', weight=2.0), {'x': 0.5}),
    ]
    fused = weighted_fuse(results)
    # x has score 0.9 from variant a and 1.0 (0.5 * 2) from variant b
    assert fused['x'] == 1.0


def test_count_multivariant_hits():
    v = QueryVariant
    results = [
        (v('a'), {'x': 0.9, 'y': 0.8}),
        (v('b'), {'x': 0.7, 'z': 0.6}),
    ]
    assert count_multivariant_hits(results) == {'x': 2, 'y': 1, 'z': 1}


def test_count_multivariant_hits_counts_distinct_notes():
    """multi_query_multivariant_hits counts canonical notes matched by >1 variants."""
    v = QueryVariant
    results = [
        (v('a'), {'x': 0.9, 'y': 0.8}),
        (v('b'), {'x': 0.7, 'z': 0.6}),
        (v('c'), {'x': 0.5, 'y': 0.4, 'z': 0.3}),
    ]
    hits = count_multivariant_hits(results)
    assert hits == {'x': 3, 'y': 2, 'z': 2}
    assert sum(1 for c in hits.values() if c > 1) == 3


def test_variant_retrieval_is_seed_only(monkeypatch):
    """Graph expansion happens once on the canonical query, not per variant."""
    calls = []

    def fake_attention(query, nodes, edges, collection, k, max_hop,
                       bm25_index, hebbian_state, routing_meta=None):
        calls.append((query, max_hop))
        return {query: 1.0}

    monkeypatch.setattr(
        'bdh_graph_harness.retrieval.multi_query.attention', fake_attention
    )
    _gather_variant_results(
        [QueryVariant('original'), QueryVariant('translation', language='it')],
        {}, {}, None, None, None, k=5, max_hop=2,
    )

    assert calls == [('original', 0), ('translation', 0)]
