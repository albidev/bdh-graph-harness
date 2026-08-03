"""Contract tests for OKF-aware retrieval ranking."""

from copy import deepcopy
from datetime import datetime, timezone

from bdh_graph_harness import config
from bdh_graph_harness.retrieval.attention import attention
from bdh_graph_harness.retrieval.okf_policy import (
    apply_okf_retrieval_policy,
    evaluate_okf_metadata,
)


NOW = datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc)


def _node(**okf):
    return {"title": okf.get("title", "Concept"), "text": "body", "okf": okf}


def test_okf_policy_keeps_trust_freshness_and_provenance_as_separate_signals():
    decision = evaluate_okf_metadata(
        _node(
            type="Concept",
            status="stable",
            generated={"by": "process:bdh", "at": "2026-08-01T12:00:00Z"},
            verified=[{"by": "human:reviewer", "at": "2026-08-01T12:00:00Z"}],
            stale_after="2999-09-01",
            sources=[
                {"id": "spec", "resource": "https://example.com/spec"},
                {"id": "local", "resource": "/Users/albi/private/source.md"},
            ],
        ),
        now=NOW,
    )

    assert decision["status"]["value"] == "stable"
    assert decision["verified"]["state"] == "verified"
    assert decision["freshness"]["state"] == "fresh"
    assert decision["provenance"]["source_count"] == 2
    assert decision["provenance"]["source_types"] == ["url", "local"]
    assert decision["provenance"]["generated_by_present"] is True
    assert decision["provenance"]["generated_at_present"] is True
    assert "/Users/albi" not in str(decision)
    assert decision["multiplier"] > 1.0


def test_generated_provenance_is_reported_without_evidence_bonus():
    decision = evaluate_okf_metadata(
        _node(generated={"by": "process:bdh", "at": "2026-08-01T12:00:00Z"}),
        now=NOW,
    )

    assert decision["provenance"]["has_provenance"] is True
    assert decision["provenance"]["generated_by_present"] is True
    assert decision["provenance"]["source_count"] == 0
    assert decision["provenance"]["factor"] == 1.0
    assert decision["multiplier"] == 1.0


def test_okf_policy_demotes_stale_draft_and_preserves_legacy_neutrality():
    scores = {"trusted": 1.0, "stale_draft": 1.0, "legacy": 1.0}
    nodes = {
        "trusted": _node(
            type="Concept",
            status="stable",
            verified=True,
            stale_after="2999-09-01",
            sources=[{"resource": "https://example.com"}],
        ),
        "stale_draft": _node(
            type="Concept",
            status="draft",
            verified=False,
            stale_after="2026-07-01",
            sources=[],
        ),
        "legacy": {"title": "Legacy", "text": "body"},
    }

    adjusted, decisions = apply_okf_retrieval_policy(
        scores,
        nodes,
        now=NOW,
        enabled=True,
    )

    assert adjusted["trusted"] > adjusted["legacy"] > adjusted["stale_draft"]
    assert decisions["stale_draft"]["freshness"]["state"] == "stale"
    assert decisions["stale_draft"]["status"]["value"] == "draft"
    assert decisions["legacy"]["applied"] is False
    assert adjusted["legacy"] == 1.0


def test_okf_policy_is_a_noop_when_disabled():
    scores = {"a": 0.4, "b": 0.8}
    nodes = {"a": _node(status="deprecated"), "b": _node(status="stable")}

    adjusted, decisions = apply_okf_retrieval_policy(
        scores,
        nodes,
        now=NOW,
        enabled=False,
    )

    assert adjusted == scores
    assert decisions == {}


class _PolicyCollection:
    def count(self):
        return 2

    def query(self, **_kwargs):
        # stale has the higher raw vector score; OKF should reverse the ranking.
        return {
            "ids": [["trusted", "stale"]],
            "distances": [[0.10, 0.0]],
        }


def test_attention_applies_okf_policy_without_mutating_hebbian_state(monkeypatch):
    nodes = {
        "trusted": _node(
            type="Concept",
            status="stable",
            verified=True,
            stale_after="2999-09-01",
            sources=[{"resource": "https://example.com"}],
        ),
        "stale": _node(
            type="Concept",
            status="deprecated",
            verified=False,
            stale_after="2026-07-01",
        ),
    }
    state = {"queries": 7, "synapses": {"trusted|stale": {"weight": 0.4}}}
    original_state = deepcopy(state)
    routing = {}

    monkeypatch.setattr(
        "bdh_graph_harness.retrieval.attention.get_embeddings",
        lambda _texts: [[1.0, 0.0]],
    )
    monkeypatch.setitem(config.CONFIG, "okf_mode", "read")
    monkeypatch.setitem(config.CONFIG, "okf_retrieval_policy_enabled", True)
    monkeypatch.setitem(config.CONFIG, "hybrid_search", False)
    monkeypatch.setitem(config.CONFIG, "retrieval_abstention_enabled", False)
    monkeypatch.setitem(config.CONFIG, "adaptive_threshold", False)

    active = attention(
        "q",
        nodes,
        {"trusted": [], "stale": []},
        _PolicyCollection(),
        k=2,
        max_hop=0,
        hebbian_state=state,
        routing_meta=routing,
    )

    assert active["trusted"] > active["stale"]
    assert routing["okf_policy"]["enabled"] is True
    details = {item["id"]: item for item in routing["activation_details"]}
    assert details["trusted"]["okf_policy"]["verified"]["state"] == "verified"
    assert details["stale"]["okf_policy"]["freshness"]["state"] == "stale"
    assert state == original_state
