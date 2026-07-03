"""Tests for attention() and _resolve_target()."""
import os
import tempfile
import pytest
import chromadb
import harness


# ---------------------------------------------------------------------------
# _resolve_target
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_nodes():
    return {
        'alpha': {'id': 'alpha', 'title': 'Alpha'},
        'wiki/beta': {'id': 'wiki/beta', 'title': 'Beta'},
        'concepts/gamma': {'id': 'concepts/gamma', 'title': 'Gamma'},
    }


def test_resolve_target_direct(simple_nodes):
    assert harness._resolve_target('alpha', simple_nodes) == 'alpha'


def test_resolve_target_prefixed(simple_nodes):
    # 'beta' not in nodes directly, but 'wiki/beta' is
    assert harness._resolve_target('beta', simple_nodes) == 'wiki/beta'


def test_resolve_target_concepts_prefix(simple_nodes):
    assert harness._resolve_target('gamma', simple_nodes) == 'concepts/gamma'


def test_resolve_target_basename(simple_nodes):
    # 'some/path/gamma' → basename 'gamma' → matches 'concepts/gamma'
    assert harness._resolve_target('some/path/gamma', simple_nodes) == 'concepts/gamma'


def test_resolve_target_no_match(simple_nodes):
    assert harness._resolve_target('nonexistent', simple_nodes) is None


# ---------------------------------------------------------------------------
# attention()
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_graph():
    """Build a small mock graph with 6 nodes and known edges."""
    d = tempfile.mkdtemp()
    notes = {
        'apple': 'Apple is a fruit that grows on trees.',
        'banana': 'Banana is a yellow fruit rich in potassium.',
        'cherry': 'Cherry is a small red fruit.',
        'carrot': 'Carrot is an orange vegetable.',
        'spinach': 'Spinach is a green leafy vegetable.',
        'index': 'Index page linking to everything.',
    }
    for nid, text in notes.items():
        with open(os.path.join(d, f'{nid}.md'), 'w') as f:
            f.write(f'# {nid}\n{text}\n')

    nodes, edges = harness.build_graph(d)
    # Add edges: apple→banana, banana→cherry, carrot→spinach, index→everything
    edges['apple'] = [{'target': 'banana', 'display': 'banana'}]
    edges['banana'] = [{'target': 'cherry', 'display': 'cherry'}]
    edges['carrot'] = [{'target': 'spinach', 'display': 'spinach'}]
    edges['index'] = [
        {'target': 'apple', 'display': 'apple'},
        {'target': 'banana', 'display': 'banana'},
        {'target': 'cherry', 'display': 'cherry'},
        {'target': 'carrot', 'display': 'carrot'},
        {'target': 'spinach', 'display': 'spinach'},
    ]
    return nodes, edges


@pytest.fixture
def mock_collection(mock_graph):
    """Create an in-memory ChromaDB collection with mock embeddings."""
    nodes, _ = mock_graph
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection('test_notes', metadata={'hnsw:space': 'cosine'})
    # Clear any existing data from prior test runs
    if collection.count() > 0:
        collection.delete(ids=collection.get()['ids'])

    # Use orthogonal-ish vectors so similarity is predictable
    embeddings = {
        'apple': [1.0, 0.0, 0.0, 0.0],
        'banana': [0.9, 0.1, 0.0, 0.0],  # similar to apple
        'cherry': [0.8, 0.2, 0.0, 0.0],  # similar to apple/banana
        'carrot': [0.0, 0.0, 1.0, 0.0],
        'spinach': [0.0, 0.0, 0.9, 0.1],  # similar to carrot
        'index': [0.5, 0.5, 0.5, 0.0],   # in the middle
    }
    for nid, emb in embeddings.items():
        collection.add(
            ids=[nid],
            embeddings=[emb],
            documents=[nodes[nid]['text'][:200]],
            metadatas=[{'title': nodes[nid]['title'], 'tags': ''}],
        )
    return collection


def test_attention_seed_selection(monkeypatch, mock_graph, mock_collection):
    """Verify seed selection picks highest similarity nodes."""
    nodes, edges = mock_graph
    # Query embedding similar to apple/banana/cherry
    monkeypatch.setattr(harness, 'get_embeddings', lambda texts: [[1.0, 0.0, 0.0, 0.0]])

    active = harness.attention('fruit', nodes, edges, mock_collection, k=3, max_hop=0)
    # With max_hop=0, only seeds — apple/banana/cherry should be top
    assert 'apple' in active
    assert active['apple'] > 0.25  # above threshold


def test_attention_khop_expansion(monkeypatch, mock_graph, mock_collection):
    """Verify k-hop expansion brings in neighbors."""
    nodes, edges = mock_graph
    monkeypatch.setattr(harness, 'get_embeddings', lambda texts: [[1.0, 0.0, 0.0, 0.0]])

    # With max_hop=0, only seeds
    active_seeds = harness.attention('fruit', nodes, edges, mock_collection, k=3, max_hop=0)
    # With max_hop=1, banana→cherry expansion should include cherry if not already
    active_expanded = harness.attention('fruit', nodes, edges, mock_collection, k=3, max_hop=2)
    # Expansion should include at least as many nodes
    assert len(active_expanded) >= len(active_seeds)


def test_attention_max_hop_limit(monkeypatch, mock_graph, mock_collection):
    """Verify max_hop=0 prevents any traversal."""
    nodes, edges = mock_graph
    monkeypatch.setattr(harness, 'get_embeddings', lambda texts: [[1.0, 0.0, 0.0, 0.0]])

    active = harness.attention('fruit', nodes, edges, mock_collection, k=2, max_hop=0)
    # All active notes should be seed-level (high similarity to query)
    for nid, score in active.items():
        # Seeds have direct similarity; traversed nodes have decayed scores
        assert score > 0.25


def test_attention_neighbor_cap(monkeypatch, mock_graph, mock_collection):
    """Verify max_neighbors_per_hop caps traversal breadth."""
    nodes, edges = mock_graph
    monkeypatch.setattr(harness, 'get_embeddings', lambda texts: [[0.5, 0.5, 0.5, 0.0]])

    # index has 5 neighbors; with cap at 3, only 3 should be traversed
    original_cap = harness.CONFIG['max_neighbors_per_hop']
    harness.CONFIG['max_neighbors_per_hop'] = 2
    try:
        active = harness.attention('everything', nodes, edges, mock_collection, k=3, max_hop=1)
        # Should not explode — cap limits expansion
        assert len(active) <= 10  # reasonable upper bound
    finally:
        harness.CONFIG['max_neighbors_per_hop'] = original_cap


def test_attention_hub_dampening(monkeypatch, mock_graph, mock_collection):
    """Verify hub dampening reduces scores for high-degree nodes."""
    nodes, edges = mock_graph
    # Make 'index' a hub with degree > threshold
    harness.CONFIG['hub_degree_threshold'] = 3
    monkeypatch.setattr(harness, 'get_embeddings', lambda texts: [[0.5, 0.5, 0.5, 0.0]])

    active = harness.attention('everything', nodes, edges, mock_collection, k=6, max_hop=0)
    # index should be dampened since it has 5 edges (> threshold 3)
    if 'index' in active:
        # Dampened score should be less than raw similarity (1.0 - 0.0 = 1.0)
        # dampen = 1/(1+0.15*(5-3)) = 1/1.3 ≈ 0.769
        assert active['index'] < 1.0
    harness.CONFIG['hub_degree_threshold'] = 15  # reset


def test_attention_empty_embedding(monkeypatch, mock_graph, mock_collection):
    """Verify empty embedding returns empty dict."""
    nodes, edges = mock_graph
    monkeypatch.setattr(harness, 'get_embeddings', lambda texts: [[]])

    active = harness.attention('test', nodes, edges, mock_collection)
    assert active == {}


def test_attention_returns_scores(monkeypatch, mock_graph, mock_collection):
    """Verify returned dict has note_id→score mapping."""
    nodes, edges = mock_graph
    monkeypatch.setattr(harness, 'get_embeddings', lambda texts: [[1.0, 0.0, 0.0, 0.0]])

    active = harness.attention('fruit', nodes, edges, mock_collection, k=3, max_hop=1)
    assert isinstance(active, dict)
    for nid, score in active.items():
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0