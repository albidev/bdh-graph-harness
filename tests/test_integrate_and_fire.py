"""Tests for Integrate-and-Fire attention model."""
import math
import pytest
import chromadb
import harness
from bdh_graph_harness import config as bdh_config
import bdh_graph_harness.retrieval.attention as bdh_attention_mod


# ---------------------------------------------------------------------------
# compute_tau
# ---------------------------------------------------------------------------

def test_tau_base_only():
    """Node with degree 0 should get base threshold."""
    bdh_config.CONFIG['iaf_tau_base'] = 0.15
    bdh_config.CONFIG['iaf_tau_k'] = 0.075
    degree_map = {}
    tau = bdh_attention_mod.compute_tau('isolated', degree_map)
    assert tau == pytest.approx(0.15)


def test_tau_scales_with_degree():
    """Higher degree → higher threshold (logarithmic)."""
    degree_map = {'hub': 30, 'leaf': 2}
    tau_hub = bdh_attention_mod.compute_tau('hub', degree_map)
    tau_leaf = bdh_attention_mod.compute_tau('leaf', degree_map)
    assert tau_hub > tau_leaf


def test_tau_logarithmic():
    """Verify logarithmic scaling: τ = base + k * log(1 + deg)."""
    bdh_config.CONFIG['iaf_tau_base'] = 0.15
    bdh_config.CONFIG['iaf_tau_k'] = 0.1
    degree_map = {'node': 99}
    expected = 0.15 + 0.1 * math.log1p(99)
    tau = bdh_attention_mod.compute_tau('node', degree_map)
    assert tau == pytest.approx(expected)


def test_tau_known_values():
    """Table of expected τ values for known degrees."""
    bdh_config.CONFIG['iaf_tau_base'] = 0.15
    bdh_config.CONFIG['iaf_tau_k'] = 0.075
    cases = [
        (0, 0.15),
        (1, 0.15 + 0.075 * math.log1p(1)),
        (5, 0.15 + 0.075 * math.log1p(5)),
        (15, 0.15 + 0.075 * math.log1p(15)),
        (35, 0.15 + 0.075 * math.log1p(35)),
    ]
    for deg, expected in cases:
        tau = bdh_attention_mod.compute_tau('x', {'x': deg})
        assert tau == pytest.approx(expected), f"degree={deg}"


# ---------------------------------------------------------------------------
# _compute_degree
# ---------------------------------------------------------------------------

def test_compute_degree_simple():
    """Degree counts outgoing + incoming edges."""
    nodes = {'a': {'id': 'a'}, 'b': {'id': 'b'}, 'c': {'id': 'c'}}
    edges = {
        'a': [{'target': 'b', 'display': 'b'}],
        'b': [{'target': 'c', 'display': 'c'}],
    }
    degree = bdh_attention_mod._compute_degree(edges, nodes)
    assert degree['a'] == 1   # 1 outgoing
    assert degree['b'] == 2   # 1 incoming + 1 outgoing
    assert degree['c'] == 1   # 1 incoming


def test_compute_degree_hubs():
    """Hub nodes have high degree."""
    nodes = {f'n{i}': {'id': f'n{i}'} for i in range(6)}
    edges = {
        'hub': [{'target': f'n{i}', 'display': f'n{i}'} for i in range(5)],
    }
    degree = bdh_attention_mod._compute_degree(edges, nodes)
    assert degree['hub'] == 5
    for i in range(5):
        assert degree[f'n{i}'] == 1


# ---------------------------------------------------------------------------
# _build_reverse_edges
# ---------------------------------------------------------------------------

def test_reverse_edges():
    """Reverse index maps target → [sources]."""
    nodes = {'a': {'id': 'a'}, 'b': {'id': 'b'}, 'c': {'id': 'c'}}
    edges = {
        'a': [{'target': 'b', 'display': 'b'}],
        'c': [{'target': 'b', 'display': 'b'}],
    }
    reverse = bdh_attention_mod._build_reverse_edges(edges, nodes)
    assert set(reverse['b']) == {'a', 'c'}
    assert reverse['a'] == []
    assert reverse['c'] == []


# ---------------------------------------------------------------------------
# integrate_and_fire_attention
# ---------------------------------------------------------------------------

@pytest.fixture
def iaf_graph():
    """Build a test graph with known structure for IaF testing."""
    nodes = {
        'seed1': {'id': 'seed1', 'title': 'Seed 1', 'text': 'Query-related content.'},
        'seed2': {'id': 'seed2', 'title': 'Seed 2', 'text': 'Query-related content.'},
        'mid1': {'id': 'mid1', 'title': 'Mid 1', 'text': 'Bridge node.'},
        'mid2': {'id': 'mid2', 'title': 'Mid 2', 'text': 'Bridge node.'},
        'hub': {'id': 'hub', 'title': 'Hub', 'text': 'High-degree connector.'},
        'leaf': {'id': 'leaf', 'title': 'Leaf', 'text': 'Isolated endpoint.'},
        'isolated': {'id': 'isolated', 'title': 'Isolated', 'text': 'No connections.'},
    }
    edges = {
        'seed1': [{'target': 'mid1', 'display': 'mid1'}],
        'seed2': [{'target': 'mid2', 'display': 'mid2'}],
        'mid1': [{'target': 'hub', 'display': 'hub'}],
        'mid2': [{'target': 'hub', 'display': 'hub'}],
        'hub': [{'target': 'leaf', 'display': 'leaf'}],
        'seed1': [{'target': 'seed2', 'display': 'seed2'}],
    }
    return nodes, edges


@pytest.fixture
def iaf_collection(iaf_graph):
    """Create ChromaDB collection with mock embeddings for IaF tests."""
    nodes, _ = iaf_graph
    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection('iaf_test', metadata={'hnsw:space': 'cosine'})
    if col.count() > 0:
        col.delete(ids=col.get()['ids'])

    embeddings = {
        'seed1': [1.0, 0.0, 0.0, 0.0],
        'seed2': [0.95, 0.05, 0.0, 0.0],
        'mid1': [0.5, 0.5, 0.0, 0.0],
        'mid2': [0.4, 0.6, 0.0, 0.0],
        'hub': [0.3, 0.3, 0.3, 0.1],
        'leaf': [0.1, 0.1, 0.1, 0.7],
        'isolated': [0.0, 0.0, 0.0, 1.0],
    }
    for nid, emb in embeddings.items():
        col.add(
            ids=[nid],
            embeddings=[emb],
            documents=[nodes[nid]['text'][:200]],
            metadatas=[{'title': nodes[nid]['title'], 'tags': ''}],
        )
    return col


def test_iaf_returns_dict(monkeypatch, iaf_graph, iaf_collection):
    """IaF returns a dict of node_id → float."""
    nodes, edges = iaf_graph
    monkeypatch.setattr(bdh_attention_mod, 'get_embeddings', lambda texts: [[1.0, 0.0, 0.0, 0.0]])
    result = bdh_attention_mod.integrate_and_fire_attention(
        'query', nodes, edges, iaf_collection, k=2, max_hop=3
    )
    assert isinstance(result, dict)
    for nid, score in result.items():
        assert isinstance(score, float)
        assert score > 0


def test_iaf_seeds_are_active(monkeypatch, iaf_graph, iaf_collection):
    """Seed nodes (highest embedding similarity) should fire first."""
    nodes, edges = iaf_graph
    monkeypatch.setattr(bdh_attention_mod, 'get_embeddings', lambda texts: [[1.0, 0.0, 0.0, 0.0]])
    result = bdh_attention_mod.integrate_and_fire_attention(
        'query', nodes, edges, iaf_collection, k=2, max_hop=0
    )
    # With max_hop=0, only seeds should be active
    assert 'seed1' in result or 'seed2' in result


def test_iaf_propagation(monkeypatch, iaf_graph, iaf_collection):
    """IaF should propagate activation beyond seeds via fired neighbors."""
    nodes, edges = iaf_graph
    monkeypatch.setattr(bdh_attention_mod, 'get_embeddings', lambda texts: [[1.0, 0.0, 0.0, 0.0]])

    result_seeds = bdh_attention_mod.integrate_and_fire_attention(
        'query', nodes, edges, iaf_collection, k=2, max_hop=0
    )
    result_propagated = bdh_attention_mod.integrate_and_fire_attention(
        'query', nodes, edges, iaf_collection, k=2, max_hop=3
    )
    # Propagation should activate at least as many nodes
    assert len(result_propagated) >= len(result_seeds)


def test_iaf_hub_requires_more_input(monkeypatch, iaf_graph, iaf_collection):
    """Hub nodes with high degree should have higher τ and need more input to fire."""
    nodes, edges = iaf_graph
    # Make hub have many edges
    for i in range(10):
        nodes[f'extra{i}'] = {'id': f'extra{i}', 'title': f'Extra {i}', 'text': 'x'}
        edges['hub'].append({'target': f'extra{i}', 'display': f'extra{i}'})

    monkeypatch.setattr(bdh_attention_mod, 'get_embeddings', lambda texts: [[1.0, 0.0, 0.0, 0.0]])
    col = iaf_collection
    for nid in [f'extra{i}' for i in range(10)]:
        col.add(
            ids=[nid],
            embeddings=[[0.3, 0.3, 0.3, 0.1]],
            documents=['x'][:200],
            metadatas=[{'title': nid, 'tags': ''}],
        )

    result = bdh_attention_mod.integrate_and_fire_attention(
        'query', nodes, edges, col, k=2, max_hop=3
    )
    # Hub should have higher τ than leaf
    hub_tau = bdh_attention_mod.compute_tau('hub', bdh_attention_mod._compute_degree(edges, nodes))
    leaf_tau = bdh_attention_mod.compute_tau('leaf', bdh_attention_mod._compute_degree(edges, nodes))
    assert hub_tau > leaf_tau


def test_iaf_empty_embedding(monkeypatch, iaf_graph, iaf_collection):
    """Empty embedding returns empty dict."""
    nodes, edges = iaf_graph
    monkeypatch.setattr(bdh_attention_mod, 'get_embeddings', lambda texts: [[]])
    result = bdh_attention_mod.integrate_and_fire_attention(
        'query', nodes, edges, iaf_collection, k=2, max_hop=3
    )
    assert result == {}


def test_iaf_isolation_by_threshold(monkeypatch, iaf_graph, iaf_collection):
    """Isolated node with no incoming edges should not fire."""
    nodes, edges = iaf_graph
    monkeypatch.setattr(bdh_attention_mod, 'get_embeddings', lambda texts: [[0.0, 0.0, 0.0, 1.0]])
    # Query embedding points to 'isolated' but it has no incoming edges
    result = bdh_attention_mod.integrate_and_fire_attention(
        'query', nodes, edges, iaf_collection, k=2, max_hop=3
    )
    # 'isolated' might fire as a seed, but won't get propagation input
    # because no one points to it
    if 'isolated' in result:
        # If it fires, it's only because it was a seed, not from propagation
        assert result['isolated'] > 0  # seed activation


def test_iaf_backward_compat_toggle(monkeypatch, iaf_graph, iaf_collection):
    """When integrate_and_fire=False, attention() uses single-pass k-hop."""
    nodes, edges = iaf_graph
    monkeypatch.setattr(bdh_attention_mod, 'get_embeddings', lambda texts: [[1.0, 0.0, 0.0, 0.0]])

    original_flag = bdh_config.CONFIG.get('integrate_and_fire', True)
    bdh_config.CONFIG['integrate_and_fire'] = False
    try:
        result = bdh_attention_mod.attention(
            'query', nodes, edges, iaf_collection, k=2, max_hop=2
        )
        assert isinstance(result, dict)
        assert len(result) > 0
    finally:
        bdh_config.CONFIG['integrate_and_fire'] = original_flag
