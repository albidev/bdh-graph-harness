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

    def get(self, ids, include):
        return {"ids": list(ids), "embeddings": [[1.0] for _ in ids]}


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


def test_dynamic_hebbian_trust_prefers_recurrent_consolidated_synapses():
    now = datetime.now()
    one_shot = {
        "weight": 0.8,
        "frequency": 0.3,
        "consolidation_candidate_cycles": 0,
        "last_coactivated": now.isoformat(),
    }
    recurrent = {
        "weight": 0.8,
        "frequency": 2.0,
        "consolidation_candidate_cycles": 1,
        "last_coactivated": now.isoformat(),
    }

    one_shot_trust = attention._hebbian_dynamic_trust(one_shot, now=now)
    recurrent_trust = attention._hebbian_dynamic_trust(recurrent, now=now)

    assert 0.0 < one_shot_trust < recurrent_trust <= 1.0


def test_dynamic_hebbian_edge_score_is_attenuated_by_trust(monkeypatch):
    nodes = {
        "seed": {"title": "Seed", "text": "seed"},
        "learned": {"title": "Learned", "text": "learned"},
    }
    state = {"synapses": {"learned|seed": {
        "weight": 0.8,
        "frequency": 0.3,
        "consolidation_candidate_cycles": 0,
    }}}
    monkeypatch.setattr(attention, "get_embeddings", lambda _queries: [[1.0]])
    monkeypatch.setitem(config.CONFIG, "adaptive_threshold", False)
    monkeypatch.setitem(config.CONFIG, "active_threshold", 0.01)
    monkeypatch.setitem(config.CONFIG, "hebbian_dynamic_gain", 1.0)
    monkeypatch.setitem(config.CONFIG, "hebbian_dynamic_hop_decay", 0.5)

    active = attention.attention(
        "q", nodes, {"seed": [], "learned": []}, FakeCollection(ids=("seed",)),
        k=1, max_hop=1, hebbian_state=state,
    )

    assert active["learned"] == 0.1625


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


def test_attention_relevance_batch_size_has_runtime_default():
    assert config.CONFIG["attention_relevance_batch_size"] == 256


def test_query_relevance_batches_collection_reads(monkeypatch):
    class Collection:
        def __init__(self):
            self.get_calls = []

        def get(self, ids, include):
            self.get_calls.append(list(ids))
            return {"ids": list(ids), "embeddings": [[1.0, 0.0] for _ in ids]}

    monkeypatch.setitem(config.CONFIG, "attention_relevance_batch_size", 2)
    collection = Collection()

    scores = attention._query_relevance(
        [1.0, 0.0], collection, ["one", "two", "three", "two"],
    )

    assert collection.get_calls == [["one", "two"], ["three"]]
    assert scores == {"one": 1.0, "two": 1.0, "three": 1.0}


def test_attention_preserves_structural_priority_when_neighbor_embedding_is_missing(monkeypatch):
    class Collection:
        def count(self):
            return 3

        def query(self, **_kwargs):
            return {"ids": [["seed"]], "distances": [[0.0]]}

        def get(self, ids, include):
            # Chroma omits the stale/unembedded target instead of returning a vector.
            return {"ids": ["low_relevance"], "embeddings": [[0.01, 1.0]]}

    nodes = {
        "seed": {"title": "Seed", "text": "seed"},
        "missing_embedding": {"title": "Fallback", "text": "fallback"},
        "low_relevance": {"title": "Weak", "text": "weak"},
    }
    edges = {"seed": [
        {"target": "missing_embedding"},
        {"target": "low_relevance"},
    ]}
    monkeypatch.setattr(attention, "get_embeddings", lambda _queries: [[1.0, 0.0]])
    monkeypatch.setitem(config.CONFIG, "adaptive_threshold", False)
    monkeypatch.setitem(config.CONFIG, "active_threshold", 0.01)
    monkeypatch.setitem(config.CONFIG, "hop_decay", 0.5)
    monkeypatch.setitem(config.CONFIG, "max_neighbors_per_hop", 1)

    active = attention.attention("q", nodes, edges, Collection(), k=1, max_hop=1)

    assert set(active) == {"seed", "missing_embedding"}


def test_attention_batches_only_dynamic_targets_reachable_from_seed(monkeypatch):
    class Collection:
        def __init__(self):
            self.get_calls = []

        def count(self):
            return 4

        def query(self, **_kwargs):
            return {"ids": [["seed"]], "distances": [[0.0]]}

        def get(self, ids, include):
            self.get_calls.append(list(ids))
            embeddings = {"reachable": [1.0, 0.0]}
            return {"ids": list(ids), "embeddings": [embeddings[node_id] for node_id in ids]}

    nodes = {node_id: {"title": node_id, "text": node_id} for node_id in (
        "seed", "reachable", "unreachable_source", "unreachable_target",
    )}
    state = {"synapses": {
        attention.encode_synapse_key("unreachable_source", "unreachable_target"): {
            "weight": 0.8, "frequency": 2.0, "consolidation_candidate_cycles": 1,
        },
    }}
    collection = Collection()
    monkeypatch.setattr(attention, "get_embeddings", lambda _queries: [[1.0, 0.0]])
    monkeypatch.setitem(config.CONFIG, "adaptive_threshold", False)
    monkeypatch.setitem(config.CONFIG, "active_threshold", 0.01)

    attention.attention(
        "q", nodes, {"seed": [{"target": "reachable"}]}, collection,
        k=1, max_hop=1, hebbian_state=state,
    )

    assert collection.get_calls == [["reachable"]]


def test_attention_does_not_let_mixed_edge_weight_override_static_semantic_rank(monkeypatch):
    class Collection:
        def count(self):
            return 3

        def query(self, **_kwargs):
            return {"ids": [["seed"]], "distances": [[0.0]]}

        def get(self, ids, include):
            embeddings = {
                "semantic_static": [1.0, 0.0],
                "weak_mixed": [0.1, 1.0],
            }
            return {"ids": list(ids), "embeddings": [embeddings[node_id] for node_id in ids]}

    nodes = {node_id: {"title": node_id, "text": node_id} for node_id in (
        "seed", "semantic_static", "weak_mixed",
    )}
    state = {"synapses": {
        attention.encode_synapse_key("seed", "weak_mixed"): {
            "weight": 1.0, "frequency": 2.0, "consolidation_candidate_cycles": 1,
        },
    }}
    monkeypatch.setattr(attention, "get_embeddings", lambda _queries: [[1.0, 0.0]])
    monkeypatch.setitem(config.CONFIG, "adaptive_threshold", False)
    monkeypatch.setitem(config.CONFIG, "active_threshold", 0.01)
    monkeypatch.setitem(config.CONFIG, "max_neighbors_per_hop", 1)
    monkeypatch.setitem(config.CONFIG, "hebbian_dynamic_query_relevance_floor", 0.0)

    active = attention.attention(
        "q", nodes,
        {"seed": [{"target": "semantic_static"}, {"target": "weak_mixed"}]},
        Collection(), k=1, max_hop=1, hebbian_state=state,
    )

    assert set(active) == {"seed", "semantic_static"}


def test_attention_prefers_semantically_relevant_static_neighbor_in_batch(monkeypatch):
    """Static expansion must not use insertion order when candidates exceed its cap."""
    class Collection:
        def __init__(self):
            self.get_calls = []

        def count(self):
            return 3

        def query(self, **_kwargs):
            return {"ids": [["seed"]], "distances": [[0.0]]}

        def get(self, ids, include):
            self.get_calls.append(list(ids))
            embeddings = {
                "topologically_first": [0.0, 1.0],
                "semantic_match": [1.0, 0.0],
            }
            return {"ids": list(ids), "embeddings": [embeddings[node_id] for node_id in ids]}

    nodes = {
        "seed": {"title": "Seed", "text": "seed"},
        "topologically_first": {"title": "Weak", "text": "weak"},
        "semantic_match": {"title": "Relevant", "text": "relevant"},
        "unreachable": {"title": "Unreachable", "text": "unreachable"},
        "off_path": {"title": "Off path", "text": "off path"},
    }
    edges = {"seed": [
        {"target": "topologically_first"},
        {"target": "semantic_match"},
    ], "unreachable": [{"target": "off_path"}]}
    collection = Collection()
    monkeypatch.setattr(attention, "get_embeddings", lambda _queries: [[1.0, 0.0]])
    monkeypatch.setitem(config.CONFIG, "adaptive_threshold", False)
    monkeypatch.setitem(config.CONFIG, "active_threshold", 0.01)
    monkeypatch.setitem(config.CONFIG, "hop_decay", 0.5)
    monkeypatch.setitem(config.CONFIG, "max_neighbors_per_hop", 1)

    active = attention.attention("q", nodes, edges, collection, k=1, max_hop=1)

    assert set(active) == {"seed", "semantic_match"}
    assert collection.get_calls == [["topologically_first", "semantic_match"]]


def test_attention_traverses_strong_hebbian_edge_without_static_wikilink(monkeypatch):
    nodes = {
        "seed": {"title": "Seed", "text": "seed"},
        "learned": {"title": "Learned", "text": "learned"},
    }
    state = {
        "synapses": {
            "learned|seed": {
                "weight": 0.8,
                "frequency": 2.0,
                "consolidation_candidate_cycles": 1,
                "last_coactivated": "2999-01-01T00:00:00+00:00",
            }
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
    state = {"synapses": {"learned|seed": {
        "weight": 0.8, "frequency": 2.0, "consolidation_candidate_cycles": 1,
    }}}
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
            "vault:wiki/learned.md|vault:wiki/seed.md": {
                "weight": 0.8,
                "frequency": 2.0,
                "consolidation_candidate_cycles": 1,
            }
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


def test_static_edge_hebbian_gain_reads_v2_synapse_key(monkeypatch):
    """Static traversal must apply Hebbian gain from canonical v2 state."""
    from bdh_graph_harness.memory.hebbian import encode_synapse_key

    seed, learned = "seed|source", "learned|target"
    nodes = {
        seed: {"title": "Seed", "text": "seed"},
        learned: {"title": "Learned", "text": "learned"},
    }
    state = {"synapses": {encode_synapse_key(seed, learned): {"weight": 0.8}}}
    monkeypatch.setattr(attention, "get_embeddings", lambda _queries: [[1.0]])
    monkeypatch.setitem(config.CONFIG, "adaptive_threshold", False)
    monkeypatch.setitem(config.CONFIG, "active_threshold", 0.01)
    monkeypatch.setitem(config.CONFIG, "hop_decay", 0.5)
    monkeypatch.setitem(config.CONFIG, "hebbian_dynamic_edges_enabled", False)
    monkeypatch.setitem(config.CONFIG, "hebbian_gain", 1.0)

    active = attention.attention(
        "q",
        nodes,
        {seed: [{"target": learned}], learned: []},
        FakeCollection(ids=(seed,)),
        k=1,
        max_hop=1,
        hebbian_state=state,
    )

    assert active[learned] == 0.9


def test_static_edge_hebbian_gain_reads_legacy_synapse_key(monkeypatch):
    """Static traversal keeps applying gain while legacy state remains on disk."""
    seed, learned = "seed", "learned"
    nodes = {
        seed: {"title": "Seed", "text": "seed"},
        learned: {"title": "Learned", "text": "learned"},
    }
    state = {"synapses": {"learned|seed": {"weight": 0.8}}}
    monkeypatch.setattr(attention, "get_embeddings", lambda _queries: [[1.0]])
    monkeypatch.setitem(config.CONFIG, "adaptive_threshold", False)
    monkeypatch.setitem(config.CONFIG, "active_threshold", 0.01)
    monkeypatch.setitem(config.CONFIG, "hop_decay", 0.5)
    monkeypatch.setitem(config.CONFIG, "hebbian_dynamic_edges_enabled", False)
    monkeypatch.setitem(config.CONFIG, "hebbian_gain", 1.0)

    active = attention.attention(
        "q", nodes, {seed: [{"target": learned}], learned: []},
        FakeCollection(ids=(seed,)), k=1, max_hop=1, hebbian_state=state,
    )

    assert active[learned] == 0.9


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


def test_dynamic_query_relevance_rejects_semantically_unrelated_neighbor():
    class Collection:
        def get(self, ids, include):
            return {
                "ids": ids,
                "embeddings": [[0.0, 1.0] if note_id == "noise" else [1.0, 0.0] for note_id in ids],
            }

    scores = attention._dynamic_query_relevance([1.0, 0.0], Collection(), {"relevant", "noise"})
    assert scores["relevant"] == 1.0
    assert scores["noise"] == 0.0


def test_associative_context_never_contaminates_primary_and_respects_budget(monkeypatch):
    monkeypatch.setitem(config.CONFIG, "hebbian_associative_context_max_items", 1)
    context = attention._build_associative_context(
        seeds=[("seed", 1.0)],
        adjacency={"seed": [("primary", 0.9, 1.0), ("context", 0.8, 1.0)]},
        primary_ids={"seed", "primary"},
    )
    assert [item["id"] for item in context] == ["context"]
    assert context[0]["matched_by"] == "hebbian_edge"
    assert context[0]["source_seed_id"] == "seed"


def test_associative_lane_preserves_primary_results(monkeypatch):
    nodes = {"seed": {"title": "Seed", "text": "seed"}, "learned": {"title": "Learned", "text": "learned"}}
    state = {"synapses": {"learned|seed": {"weight": 0.8, "frequency": 2.0, "consolidation_candidate_cycles": 1}}}
    monkeypatch.setattr(attention, "get_embeddings", lambda _queries: [[1.0]])
    monkeypatch.setitem(config.CONFIG, "adaptive_threshold", False)
    monkeypatch.setitem(config.CONFIG, "active_threshold", 0.01)
    monkeypatch.setitem(config.CONFIG, "hebbian_dynamic_edges_enabled", True)
    monkeypatch.setitem(config.CONFIG, "hebbian_associative_context_enabled", True)
    routing = {}
    primary = attention.attention("q", nodes, {"seed": [], "learned": []}, FakeCollection(ids=("seed",)), k=1, max_hop=1, hebbian_state=state, routing_meta=routing)
    assert list(primary) == ["seed"]
    assert routing["associative_context"][0]["id"] == "learned"
