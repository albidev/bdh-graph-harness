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
