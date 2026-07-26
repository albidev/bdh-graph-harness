"""Deterministic branch coverage for attention helpers and IaF propagation."""

from datetime import datetime, timedelta

from bdh_graph_harness import config
from bdh_graph_harness.retrieval import attention


class FakeCollection:
    def __init__(self, ids=("seed", "peer")):
        self.ids = list(ids)

    def count(self):
        return len(self.ids)

    def query(self, **_kwargs):
        return {"ids": [self.ids], "distances": [[0.0, 0.4][:len(self.ids)]]}


def test_hebbian_helpers_handle_empty_invalid_and_capped_scores(monkeypatch):
    assert attention.compute_adaptive_threshold([], floor=0.33) == 0.33
    assert attention.compute_adaptive_threshold([0.1, 0.2], floor=0.33) == 0.33
    assert attention._compute_hebbian_boost("a", None, {"b"}) == 0.0
    assert attention._compute_hebbian_boost("a", {"synapses": {}}, {"b"}) == 0.0
    state = {
        "synapses": {
            "a|b": {"weight": 5.0},
            "a|c": {"weight": 4.0},
            "a|d": {"weight": 3.0},
        }
    }
    monkeypatch.setitem(config.CONFIG, "hebbian_boost_top_n", 2)
    monkeypatch.setitem(config.CONFIG, "hebbian_boost_weight_factor", 1.0)
    monkeypatch.setitem(config.CONFIG, "hebbian_boost_max", 0.5)
    assert attention._compute_hebbian_boost("a", state, {"a", "b", "c", "d"}) == 0.5

    now = datetime.now()
    recent = now.isoformat()
    old = (now - timedelta(days=1)).isoformat()
    with_validity = {
        "_valid_node_ids": {"a", "b"},
        "synapses": {
            "a|b": {"weight": 1.0, "last_coactivated": recent},
            "a|dead": {"weight": 1.0, "last_coactivated": recent},
            "b|c": {"weight": 0.0, "last_coactivated": recent},
            "c|d": {"weight": 1.0, "last_coactivated": old},
            "broken": {"weight": 1.0, "last_coactivated": "nope"},
        },
    }
    assert attention._get_recently_active_notes(
        with_validity, valid_node_ids={"a", "b"}
    ) == {"a", "b"}
    assert "_valid_node_ids" in with_validity
    assert attention._get_recently_active_notes(None) == set()


def test_attention_dispatches_to_iaf_and_formats_context(monkeypatch):
    nodes = {"seed": {"title": "Seed", "text": "body"}}
    monkeypatch.setitem(config.CONFIG, "experimental_integrate_fire", True)
    monkeypatch.setattr(attention, "integrate_and_fire_attention", lambda *args: {"iaf": 1.0})
    assert attention.attention("q", nodes, {}, FakeCollection()) == {"iaf": 1.0}
    monkeypatch.setitem(config.CONFIG, "experimental_integrate_fire", False)

    assert attention.format_context({"seed": 0.75, "missing": 0.1}, nodes) == "### Seed (activation: 0.750)\nbody\n"


def test_integrate_and_fire_covers_empty_embedding_and_firing(monkeypatch):
    nodes = {
        "seed": {"title": "Seed", "text": "seed"},
        "peer": {"title": "Peer", "text": "peer"},
    }
    edges = {"seed": [{"target": "peer"}], "peer": []}
    collection = FakeCollection()
    monkeypatch.setattr(attention, "get_embeddings", lambda _queries: [[]])
    assert attention.integrate_and_fire_attention("q", nodes, edges, collection) == {}

    monkeypatch.setattr(attention, "get_embeddings", lambda _queries: [[1.0]])
    monkeypatch.setitem(config.CONFIG, "iaf_tau_base", 0.01)
    monkeypatch.setitem(config.CONFIG, "iaf_tau_k", 0.0)
    monkeypatch.setitem(config.CONFIG, "iaf_max_steps", 3)
    monkeypatch.setitem(config.CONFIG, "hybrid_search", False)
    fired = attention.integrate_and_fire_attention("q", nodes, edges, collection, k=1)
    assert set(fired) == {"seed", "peer"}


def test_attention_single_pass_covers_missing_targets_and_threshold(monkeypatch):
    nodes = {"seed": {"title": "Seed", "text": "seed"}, "peer": {"title": "Peer", "text": "peer"}}
    edges = {"seed": [{"target": "missing"}, {"target": "peer"}]}
    monkeypatch.setattr(attention, "get_embeddings", lambda _queries: [[1.0]])
    monkeypatch.setitem(config.CONFIG, "adaptive_threshold", False)
    monkeypatch.setitem(config.CONFIG, "active_threshold", 0.01)
    monkeypatch.setitem(config.CONFIG, "hop_decay", 0.5)
    active = attention.attention("q", nodes, edges, FakeCollection(), k=1, max_hop=1)
    assert set(active) == {"seed", "peer"}


def test_attention_traverses_strong_hebbian_edge_without_static_wikilink(monkeypatch):
    nodes = {
        "seed": {"title": "Seed", "text": "seed"},
        "learned": {"title": "Learned", "text": "learned"},
    }
    state = {
        "synapses": {
            "learned|seed": {"weight": 0.8, "last_coactivated": "2999-01-01T00:00:00+00:00"}
        }
    }
    routing_meta = {}
    monkeypatch.setattr(attention, "get_embeddings", lambda _queries: [[1.0]])
    monkeypatch.setitem(config.CONFIG, "adaptive_threshold", False)
    monkeypatch.setitem(config.CONFIG, "active_threshold", 0.01)
    monkeypatch.setitem(config.CONFIG, "hop_decay", 0.5)
    monkeypatch.setitem(config.CONFIG, "hebbian_dynamic_edges_enabled", True)
    monkeypatch.setitem(config.CONFIG, "hebbian_dynamic_min_weight", 0.15)
    monkeypatch.setitem(config.CONFIG, "hebbian_dynamic_top_n", 3)
    monkeypatch.setitem(config.CONFIG, "hebbian_dynamic_gain", 1.0)
    monkeypatch.setitem(config.CONFIG, "hebbian_dynamic_hop_decay", 0.5)

    active = attention.attention(
        "q",
        nodes,
        {"seed": [], "learned": []},
        FakeCollection(ids=("seed",)),
        k=1,
        max_hop=1,
        hebbian_state=state,
        routing_meta=routing_meta,
    )

    assert active["learned"] == 0.65
    learned = next(item for item in routing_meta["activation_details"] if item["id"] == "learned")
    assert learned["role"] == "hebbian_neighbor"
    assert learned["matched_by"] == "hebbian_edge"
    assert learned["hebbian_edge_weight"] == 0.8


def test_dynamic_hebbian_edge_uses_its_own_hop_decay(monkeypatch):
    nodes = {
        "seed": {"title": "Seed", "text": "seed"},
        "learned": {"title": "Learned", "text": "learned"},
    }
    state = {"synapses": {"learned|seed": {"weight": 0.8}}}
    monkeypatch.setattr(attention, "get_embeddings", lambda _queries: [[1.0]])
    monkeypatch.setitem(config.CONFIG, "adaptive_threshold", False)
    monkeypatch.setitem(config.CONFIG, "active_threshold", 0.01)
    monkeypatch.setitem(config.CONFIG, "hop_decay", 0.5)
    monkeypatch.setitem(config.CONFIG, "hebbian_dynamic_gain", 1.0)
    monkeypatch.setitem(config.CONFIG, "hebbian_dynamic_hop_decay", 0.4)

    active = attention.attention(
        "q", nodes, {"seed": [], "learned": []}, FakeCollection(ids=("seed",)),
        k=1, max_hop=1, hebbian_state=state,
    )

    assert active["learned"] == 0.52


def test_attention_resolves_federated_hebbian_ids_to_core_nodes(monkeypatch):
    nodes = {
        "wiki/seed": {"title": "Seed", "text": "seed"},
        "wiki/learned": {"title": "Learned", "text": "learned"},
    }
    state = {
        "synapses": {
            "vault:wiki/learned.md|vault:wiki/seed.md": {"weight": 0.8}
        }
    }
    monkeypatch.setattr(attention, "get_embeddings", lambda _queries: [[1.0]])
    monkeypatch.setitem(config.CONFIG, "adaptive_threshold", False)
    monkeypatch.setitem(config.CONFIG, "active_threshold", 0.01)
    monkeypatch.setitem(config.CONFIG, "hop_decay", 0.5)
    monkeypatch.setitem(config.CONFIG, "hebbian_dynamic_edges_enabled", True)
    monkeypatch.setitem(config.CONFIG, "hebbian_dynamic_gain", 1.0)
    monkeypatch.setitem(config.CONFIG, "hebbian_dynamic_hop_decay", 0.5)

    active = attention.attention(
        "q",
        nodes,
        {"wiki/seed": [], "wiki/learned": []},
        FakeCollection(ids=("wiki/seed",)),
        k=1,
        max_hop=1,
        hebbian_state=state,
    )

    assert active["wiki/learned"] == 0.65


def test_static_and_hebbian_edge_applies_learned_weight_once(monkeypatch):
    nodes = {
        "seed": {"title": "Seed", "text": "seed"},
        "learned": {"title": "Learned", "text": "learned"},
    }
    state = {"synapses": {"learned|seed": {"weight": 0.8}}}
    routing_meta = {}
    monkeypatch.setattr(attention, "get_embeddings", lambda _queries: [[1.0]])
    monkeypatch.setitem(config.CONFIG, "adaptive_threshold", False)
    monkeypatch.setitem(config.CONFIG, "active_threshold", 0.01)
    monkeypatch.setitem(config.CONFIG, "hop_decay", 0.5)
    monkeypatch.setitem(config.CONFIG, "hebbian_dynamic_edges_enabled", True)
    monkeypatch.setitem(config.CONFIG, "hebbian_dynamic_gain", 1.0)
    monkeypatch.setitem(config.CONFIG, "hebbian_gain", 1.0)

    active = attention.attention(
        "q",
        nodes,
        {"seed": [{"target": "learned"}], "learned": []},
        FakeCollection(ids=("seed",)),
        k=1,
        max_hop=1,
        hebbian_state=state,
        routing_meta=routing_meta,
    )

    assert active["learned"] == 0.9
    learned = next(item for item in routing_meta["activation_details"] if item["id"] == "learned")
    assert learned["matched_by"] == "static_and_hebbian_edge"
