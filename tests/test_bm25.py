"""Tests for BM25Index (Phase 3.1 hybrid search)."""
import pytest
import harness


@pytest.fixture
def mock_nodes():
    """Mock nodes with known text for BM25 scoring."""
    return {
        'apple': {
            'id': 'apple', 'title': 'Apple',
            'text': 'Apple is a fruit that grows on trees. Apples are red or green.',
        },
        'banana': {
            'id': 'banana', 'title': 'Banana',
            'text': 'Banana is a yellow fruit rich in potassium. Bananas grow in tropical climates.',
        },
        'cherry': {
            'id': 'cherry', 'title': 'Cherry',
            'text': 'Cherry is a small red fruit. Cherries are used in desserts.',
        },
        'carrot': {
            'id': 'carrot', 'title': 'Carrot',
            'text': 'Carrot is an orange vegetable. Carrots grow underground.',
        },
        'spinach': {
            'id': 'spinach', 'title': 'Spinach',
            'text': 'Spinach is a green leafy vegetable. Spinach is rich in iron.',
        },
        'index': {
            'id': 'index', 'title': 'Index',
            'text': 'Index page linking to apple banana cherry carrot spinach.',
        },
    }


def test_bm25_build(mock_nodes):
    """BM25 index builds correctly with right doc count and terms."""
    idx = harness.BM25Index(mock_nodes)
    assert idx.N == 6
    assert len(idx.df) > 0
    assert idx.avg_dl > 0


def test_bm25_score_exact_match(mock_nodes):
    """BM25 gives higher score for docs containing query terms."""
    idx = harness.BM25Index(mock_nodes)
    # 'apple' query should score the apple doc highest
    score_apple = idx.score('apple', 'apple')
    score_carrot = idx.score('apple', 'carrot')
    assert score_apple > 0
    assert score_carrot == 0.0  # carrot doesn't contain 'apple'


def test_bm25_score_no_match(mock_nodes):
    """BM25 returns 0 for non-matching queries."""
    idx = harness.BM25Index(mock_nodes)
    score = idx.score('quantum physics', 'apple')
    assert score == 0.0


def test_bm25_score_normalized(mock_nodes):
    """BM25 scores are normalized to [0, 1] range."""
    idx = harness.BM25Index(mock_nodes)
    for nid in mock_nodes:
        score = idx.score('fruit apple banana', nid)
        assert 0.0 <= score <= 1.0


def test_bm25_search_returns_sorted(mock_nodes):
    """BM25 search returns results sorted by score descending."""
    idx = harness.BM25Index(mock_nodes)
    results = idx.search('fruit')
    assert len(results) > 0
    # Check descending order
    for i in range(len(results) - 1):
        assert results[i][1] >= results[i + 1][1]


def test_bm25_search_top_k(mock_nodes):
    """BM25 search respects top_k limit."""
    idx = harness.BM25Index(mock_nodes)
    results = idx.search('fruit', top_k=2)
    assert len(results) <= 2


def test_bm25_search_subset_ids(mock_nodes):
    """BM25 search can filter to a subset of note IDs."""
    idx = harness.BM25Index(mock_nodes)
    results = idx.search('fruit', note_ids=['apple', 'carrot'])
    ids = [r[0] for r in results]
    assert 'apple' in ids
    assert 'carrot' not in ids  # carrot doesn't match 'fruit'


def test_bm25_empty_nodes():
    """BM25 handles empty node set gracefully."""
    idx = harness.BM25Index({})
    assert idx.N == 0
    assert idx.score('test', 'any') == 0.0
    assert idx.search('test') == []


def test_bm25_tokenize():
    """Tokenizer lowercases and splits on non-alphanumeric."""
    tokens = harness.BM25Index._tokenize('Hello, World! 123')
    assert tokens == ['hello', 'world', '123']


def test_bm25_idf_positive(mock_nodes):
    """IDF values are positive for all terms."""
    idx = harness.BM25Index(mock_nodes)
    for term, idf in idx.idf.items():
        assert idf > 0.0, f"IDF for '{term}' should be positive, got {idf}"


def test_bm25_rare_term_higher_idf(mock_nodes):
    """Rare terms (in fewer docs) have higher IDF than common terms."""
    idx = harness.BM25Index(mock_nodes)
    # 'fruit' appears in apple, banana, cherry (3 docs)
    # 'potassium' appears only in banana (1 doc)
    idf_rare = idx.idf.get('potassium', 0)
    idf_common = idx.idf.get('fruit', 0)
    if idf_rare and idf_common:
        assert idf_rare > idf_common