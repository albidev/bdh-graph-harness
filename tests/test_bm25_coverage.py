"""Boundary coverage for BM25 empty and fallback normalization behavior."""

from bdh_graph_harness.retrieval.bm25 import BM25Index


def test_bm25_empty_query_and_unknown_note_are_zero():
    index = BM25Index({"note": {"text": "retrieval architecture"}})

    assert index.score("the in", "note") == 0.0
    assert index.score("retrieval", "missing") == 0.0


def test_bm25_normalization_paths_and_no_results_search():
    index = BM25Index({"note": {"text": "retrieval retrieval architecture"}})

    raw = index.score("retrieval", "note")
    assert index.score_normalized("retrieval", "note", max_score=raw / 2) == 1.0
    assert 0 < index.score_normalized("retrieval", "note") <= 1.0
    assert index.score_normalized("absent", "note") == 0.0
    assert index.score_batch("absent") == {}
    assert index.search("absent", top_k=1) == []
