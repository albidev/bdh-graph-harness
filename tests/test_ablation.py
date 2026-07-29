from __future__ import annotations

import pytest

from benchmarks import ablation


def _fake_attention(calls):
    def attention(_query, _nodes, _edges, _collection, *, routing_meta=None, **_kwargs):
        calls.append(routing_meta is not None)
        if routing_meta is not None:
            routing_meta["activation_details"] = [{"hop": 0}, {"hop": 1}]
        return {"target": 1.0}

    return attention


def test_cold_pass_never_updates_hebbian_state(monkeypatch):
    calls = []
    monkeypatch.setattr(ablation, "attention", _fake_attention(calls))

    updates = []
    monkeypatch.setattr(
        ablation,
        "hebbian_update",
        lambda active, state: (updates.append((active, state)) or (state, [], [])),
    )

    ablation._run_pass(
        [{"query": "test query", "relevant_note_ids": ["target"]}],
        {"target": {}},
        {},
        object(),
        object(),
        ablation._fresh_state(),
        cold=True,
        collect_hops=False,
    )

    assert updates == []


def test_materialized_graph_uses_configured_federated_builder(monkeypatch):
    monkeypatch.setitem(ablation.CONFIG, "external_sources", [{"id": "projects"}])
    expected_nodes = {"external:projects/readme": {}}
    expected_edges = {"external:projects/readme": []}
    calls = []

    def fake_builder(config, *, use_cache):
        calls.append((config, use_cache))
        return expected_nodes, expected_edges, []

    monkeypatch.setattr(ablation, "build_configured_graph", fake_builder)

    nodes, edges, state_path = ablation._build_materialized_graph()

    assert (nodes, edges, state_path) == (expected_nodes, expected_edges, None)
    assert calls == [(ablation.CONFIG, True)]


def test_instrumented_query_calls_attention_once(monkeypatch):
    calls = []
    monkeypatch.setattr(ablation, "attention", _fake_attention(calls))

    ablation._run_single_query(
        query="test query",
        expected={"target"},
        nodes={"target": {}},
        edges={},
        collection=object(),
        bm25_index=object(),
        state=ablation._fresh_state(),
        category="test",
        cold=True,
        collect_hops=True,
    )

    assert calls == [True]


def test_golden_set_validator_rejects_missing_relevant_note_id():
    with pytest.raises(ValueError, match="missing-note"):
        ablation.validate_golden_set(
            [{"query": "What safeguards state integrity?", "relevant_note_ids": ["missing-note"]}],
            {"present-note": {"title": "Present Note"}},
        )


def test_golden_set_validator_rejects_verbatim_target_title():
    with pytest.raises(ValueError, match="title leakage"):
        ablation.validate_golden_set(
            [{"query": "Explain Graph Memory", "relevant_note_ids": ["graph-memory"]}],
            {"graph-memory": {"title": "Graph Memory"}},
        )


def test_golden_set_validator_accepts_multiple_existing_semantic_targets():
    ablation.validate_golden_set(
        [
            {
                "query": "How do learned connection weights affect traversal through connected documents?",
                "relevant_note_ids": ["hebian", "traversal"],
            }
        ],
        {
            "hebian": {"title": "Hebbian Learning"},
            "traversal": {"title": "Graph Traversal"},
        },
    )


def test_golden_set_validator_accepts_expected_empty_negative_query():
    ablation.validate_golden_set(
        [
            {
                "query": "What is the rainfall forecast for Rome tomorrow?",
                "expected_empty": True,
                "relevant_note_ids": [],
            }
        ],
        {},
    )


def test_run_pass_excludes_expected_empty_queries_from_ranking_metrics(monkeypatch):
    monkeypatch.setattr(ablation, "attention", lambda *_args, **_kwargs: {"target": 1.0})

    metrics, _metadata = ablation._run_pass(
        [
            {"query": "positive", "relevant_note_ids": ["target"]},
            {"query": "negative", "expected_empty": True, "relevant_note_ids": []},
        ],
        {"target": {"title": "Target"}},
        {},
        object(),
        object(),
        ablation._fresh_state(),
        cold=True,
        collect_hops=False,
    )

    assert metrics.mrr == 1.0
    assert metrics.negative_query_count == 1
    assert metrics.negative_nonempty_rate == 1.0
    assert metrics.per_query[1]["is_correct_rejection"] is False


def test_serialize_includes_negative_control_metrics():
    serialized = ablation._serialize(
        ablation.Metrics(negative_query_count=10, negative_nonempty_rate=0.2)
    )

    assert serialized["negative_query_count"] == 10
    assert serialized["negative_nonempty_rate"] == 0.2


def test_hebbian_trajectory_trains_only_on_training_queries(monkeypatch):
    train = [{"query": "train", "relevant_note_ids": ["a"]}]
    holdout = [{"query": "holdout", "relevant_note_ids": ["a"]}]
    calls = []

    def fake_run_pass(queries, _nodes, _edges, _collection, _bm25, state, *, cold, collect_hops):
        calls.append((queries, cold))
        if queries is train:
            assert cold is False
            state["synapses"]["a|b"] = {"weight": 0.2}
            return ablation.Metrics(mrr=0.1), {}
        assert cold is True
        return ablation.Metrics(mrr=0.3 if state["synapses"] else 0.2), {}

    monkeypatch.setattr(ablation, "_run_pass", fake_run_pass)

    result = ablation._evaluate_hebbian_trajectory(
        train,
        holdout,
        nodes={},
        edges=[],
        collection=None,
        bm25_index=None,
        collect_hops=False,
    )

    assert calls == [(holdout, True), (train, False), (holdout, True)]
    assert result["cold"].mrr == 0.2
    assert result["after_training"].mrr == 0.3
    assert result["trained_final_synapses"] == 1
