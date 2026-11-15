"""Budget-aware context packing: unit behaviour and the /api/query contract."""
import pytest

from bdh_graph_harness.llm.prompt import build_messages, format_context
from bdh_graph_harness.retrieval.context_packing import (
    TRUNCATION_MARKER,
    PackItem,
    build_retrieved_context,
    coerce_budget_chars,
    pack_items,
    truncate_at_boundary,
)

from tests.test_multi_query_api import _capture_app, mock_app_mq_setup  # noqa: F401


def _sentences(word, count):
    return " ".join(f"{word} sentence number {i}." for i in range(count))


def _spent(items, packed):
    overhead = {item.key: item.overhead for item in items}
    return sum(len(entry.text) + overhead[entry.key] for entry in packed)


# ---------------------------------------------------------------------------
# pack_items
# ---------------------------------------------------------------------------

def test_everything_is_kept_verbatim_when_it_fits():
    items = [PackItem("a", 0.9, "alpha body", 10), PackItem("b", 0.5, "beta body", 10)]
    packed, omitted = pack_items(items, 1000)
    assert [(entry.key, entry.text, entry.truncated) for entry in packed] == [
        ("a", "alpha body", False), ("b", "beta body", False),
    ]
    assert omitted == []


def test_long_top_note_no_longer_starves_the_notes_ranked_after_it():
    """Regression: a greedy fill gave the whole budget to the first note."""
    long_text = _sentences("long", 2000)
    items = [
        PackItem("long", 1.0, long_text, 12),
        PackItem("second", 0.9, _sentences("second", 20), 12),
        PackItem("third", 0.8, _sentences("third", 20), 12),
    ]
    packed, omitted = pack_items(items, 4000)
    by_key = {entry.key: entry for entry in packed}
    assert omitted == []
    assert set(by_key) == {"long", "second", "third"}
    assert by_key["second"].truncated is False
    assert by_key["third"].truncated is False
    assert by_key["long"].truncated is True
    assert _spent(items, packed) <= 4000
    # The room short notes did not need went to the long one, not to waste.
    assert _spent(items, packed) > 4000 * 0.95


def test_total_never_exceeds_budget_for_any_budget():
    items = [
        PackItem(f"n{i}", 1.0 - i * 0.1, _sentences(f"note{i}", 40 * (i + 1)), 15)
        for i in range(6)
    ]
    for budget in (0, 1, 50, 200, 399, 400, 1234, 4000, 8000, 50000):
        packed, _ = pack_items(items, budget)
        assert _spent(items, packed) <= budget, budget
        for entry in packed:
            assert len(entry.text) <= entry.full_chars


def test_a_note_that_cannot_fit_is_skipped_without_stopping_the_packing():
    items = [
        PackItem("first", 1.0, "x" * 300, 0),
        PackItem("too-big-for-floor", 0.9, "y" * 5000, 0),
        PackItem("small", 0.8, "z" * 50, 0),
    ]
    packed, omitted = pack_items(items, 400, min_chars=300)
    assert [entry.key for entry in packed] == ["first", "small"]
    assert omitted == ["too-big-for-floor"]


def test_extra_room_follows_the_activation_score():
    text = "w" * 10000
    items = [PackItem("strong", 0.9, text, 0), PackItem("weak", 0.3, text, 0)]
    packed, _ = pack_items(items, 4000, min_chars=100)
    strong, weak = (len(entry.text) for entry in packed)
    assert strong > weak
    assert weak >= 100
    assert strong + weak <= 4000


def test_tiny_budget_still_returns_the_best_note_shortened():
    items = [PackItem("top", 1.0, _sentences("top", 200), 10), PackItem("next", 0.5, "short", 10)]
    packed, omitted = pack_items(items, 120, min_chars=400)
    assert packed[0].key == "top"
    assert packed[0].truncated is True
    assert _spent(items, packed) <= 120
    assert "top" not in omitted


def test_zero_scores_are_treated_as_equal_weights():
    items = [PackItem("a", 0.0, "a" * 5000, 0), PackItem("b", 0.0, "b" * 5000, 0)]
    packed, _ = pack_items(items, 2000, min_chars=10)
    assert abs(len(packed[0].text) - len(packed[1].text)) <= len(TRUNCATION_MARKER) + 1


# ---------------------------------------------------------------------------
# truncate_at_boundary / coerce_budget_chars
# ---------------------------------------------------------------------------

def test_truncation_prefers_a_sentence_end_and_marks_the_cut():
    text = "First sentence here. Second sentence follows. Third one is cut somewhere in the middle of it."
    cut = truncate_at_boundary(text, 60)
    assert len(cut) <= 60
    assert cut == "First sentence here. Second sentence follows." + TRUNCATION_MARKER


def test_truncation_falls_back_to_a_word_boundary_then_a_hard_cut():
    words = truncate_at_boundary("alpha beta gamma delta epsilon zeta eta theta", 30)
    assert len(words) <= 30
    assert words.endswith(TRUNCATION_MARKER)
    assert not words[: -len(TRUNCATION_MARKER)].endswith(" ")
    solid = truncate_at_boundary("x" * 100, 30)
    assert solid == "x" * (30 - len(TRUNCATION_MARKER)) + TRUNCATION_MARKER


def test_truncation_leaves_short_text_alone():
    assert truncate_at_boundary("short", 30) == "short"
    assert truncate_at_boundary("anything", 0) == ""


@pytest.mark.parametrize("value, expected", [
    (None, 8000), ("4000", 8000), (True, 8000), (4000, 4000), (1, 200), (10**9, 64000),
])
def test_budget_coercion_falls_back_and_clamps(value, expected):
    assert coerce_budget_chars(value, 8000) == expected


# ---------------------------------------------------------------------------
# build_retrieved_context
# ---------------------------------------------------------------------------

def test_retrieved_context_keeps_the_legacy_shape_when_it_fits():
    nodes = {"a": {"text": " Alpha content "}, "b": {"text": "Beta content"}, "empty": {"text": "  "}}
    activated = [
        {"id": "a", "title": "Alpha", "score": 0.9},
        {"id": "empty", "title": "Empty", "score": 0.8},
        {"id": "missing", "title": "Missing", "score": 0.7},
        {"id": "b", "title": "Beta", "score": 0.6},
    ]
    context, meta = build_retrieved_context(activated, nodes, 8000)
    assert context == "## Alpha\nAlpha content\n\n## Beta\nBeta content"
    assert meta["used_chars"] == len(context)
    assert meta["omitted"] == []
    assert [note["id"] for note in meta["notes"]] == ["a", "b"]
    assert all(note["truncated"] is False for note in meta["notes"])


def test_retrieved_context_fits_the_callers_budget_and_reports_what_was_cut():
    nodes = {
        "long": {"text": _sentences("long", 3000)},
        "b": {"text": _sentences("beta", 30)},
        "c": {"text": _sentences("gamma", 30)},
    }
    activated = [
        {"id": "long", "title": "Long", "score": 1.0},
        {"id": "b", "title": "Beta", "score": 0.9},
        {"id": "c", "title": "Gamma", "score": 0.8},
    ]
    context, meta = build_retrieved_context(activated, nodes, 4000)
    assert len(context) <= 4000
    assert "## Beta" in context and "## Gamma" in context
    assert "gamma sentence number 29." in context
    by_id = {note["id"]: note for note in meta["notes"]}
    assert by_id["long"]["truncated"] is True
    assert by_id["long"]["full_chars"] == len(nodes["long"]["text"])
    assert by_id["b"]["truncated"] is False


# ---------------------------------------------------------------------------
# Synthesis prompt
# ---------------------------------------------------------------------------

def test_neighbours_get_a_smaller_guaranteed_excerpt_than_seeds():
    long_text = _sentences("body", 400)
    nodes = {nid: {"text": long_text} for nid in ("s1", "s2", "n1", "n2")}
    activated = [
        {"id": "s1", "title": "Seed one", "score": 1.0, "role": "seed"},
        {"id": "s2", "title": "Seed two", "score": 0.8, "role": "seed"},
        {"id": "n1", "title": "Neighbour one", "score": 0.3, "role": "graph_neighbor"},
        {"id": "n2", "title": "Neighbour two", "score": 0.3, "role": "hebbian_neighbor"},
    ]
    budget = 1400
    _, uniform = build_retrieved_context(activated, nodes, budget, min_note_chars=400)
    context, tiered = build_retrieved_context(
        activated, nodes, budget, min_note_chars=400, min_neighbor_chars=150,
    )
    assert len(context) <= budget
    # With one floor for everyone the budget cannot hold all four notes.
    assert uniform["omitted"] == ["n2"]
    assert tiered["omitted"] == []
    chars = {note["id"]: note["chars"] for note in tiered["notes"]}
    assert chars["s1"] >= 400 and chars["s2"] >= 400
    assert 100 <= chars["n1"] < 400 and 100 <= chars["n2"] < 400


def test_notes_without_a_role_keep_the_full_guarantee():
    nodes = {"a": {"text": "x" * 5000}, "b": {"text": "y" * 5000}}
    activated = [{"id": "a", "title": "A", "score": 0.9}, {"id": "b", "title": "B", "score": 0.2}]
    _, meta = build_retrieved_context(activated, nodes, 1000, min_note_chars=400, min_neighbor_chars=50)
    assert all(note["chars"] >= 390 for note in meta["notes"])


def test_prompt_context_is_unchanged_for_short_notes():
    nodes = {"seed": {"title": "Seed", "text": "body"}}
    assert format_context({"seed": 0.75, "missing": 0.1}, nodes) == "### Seed (activation: 0.750)\nbody\n"


def test_prompt_shares_one_budget_instead_of_cutting_every_note_alike():
    nodes = {
        "long": {"title": "Long", "text": _sentences("long", 600)},
        "stub": {"title": "Stub", "text": "A short concept stub."},
    }
    active = {"long": 0.9, "stub": 0.8}
    messages = build_messages("what is long?", active, nodes, context_budget_chars=6000)
    user = messages[1]["content"]
    context = user.split("## Question")[0]
    assert len(context) <= 6000 + len("## Activated Notes Context\n\n\n\n")
    assert "A short concept stub." in context
    # The old fixed cut stopped the long note at 2000 characters.
    long_section = context.split("### Stub")[0]
    assert len(long_section) > 4000
    assert TRUNCATION_MARKER in long_section


# ---------------------------------------------------------------------------
# /api/query contract
# ---------------------------------------------------------------------------

async def _post_query(monkeypatch, setup, body):
    from aiohttp.test_utils import TestClient, TestServer

    nodes, edges, collection, state, config, _ = setup
    app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.post('/api/query', json=body)
        return resp.status, await resp.json()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_api_query_packs_retrieved_context_to_the_requested_budget(mock_app_mq_setup, monkeypatch):  # noqa: F811
    nodes = mock_app_mq_setup[0]
    nodes['alpha']['text'] = _sentences("alpha", 3000)
    nodes['beta']['text'] = _sentences("beta", 20)
    status, data = await _post_query(monkeypatch, mock_app_mq_setup, {
        'query': 'test query', 'learn': False, 'respond': False,
        'include_context': True, 'context_budget_chars': 2000,
    })
    assert status == 200
    context = data['retrieved_context']
    assert len(context) <= 2000
    # The lower-ranked note survives whole next to a long top note.
    assert 'beta sentence number 19.' in context
    meta = data['retrieved_context_meta']
    assert meta['budget_chars'] == 2000
    assert meta['used_chars'] == len(context)
    assert {note['id']: note['truncated'] for note in meta['notes']} == {'alpha': True, 'beta': False}
    assert meta['omitted'] == []


@pytest.mark.asyncio
async def test_api_query_default_budget_is_config_driven(mock_app_mq_setup, monkeypatch):  # noqa: F811
    nodes, _edges, _collection, _state, config, _ = mock_app_mq_setup
    nodes['alpha']['text'] = _sentences("alpha", 3000)
    config['retrieval_context_budget_chars'] = 1500
    status, data = await _post_query(monkeypatch, mock_app_mq_setup, {
        'query': 'test query', 'learn': False, 'respond': False, 'include_context': True,
    })
    assert status == 200
    assert data['retrieved_context_meta']['budget_chars'] == 1500
    assert len(data['retrieved_context']) <= 1500
    assert '## Beta' in data['retrieved_context']


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [0, -5, "4000", 12.5, True])
async def test_api_query_rejects_an_invalid_budget(mock_app_mq_setup, monkeypatch, bad):  # noqa: F811
    status, data = await _post_query(monkeypatch, mock_app_mq_setup, {
        'query': 'test query', 'learn': False, 'respond': False,
        'include_context': True, 'context_budget_chars': bad,
    })
    assert status == 400
    assert data['field'] == 'context_budget_chars'


@pytest.mark.asyncio
async def test_api_query_without_include_context_returns_no_note_bodies(mock_app_mq_setup, monkeypatch):  # noqa: F811
    status, data = await _post_query(monkeypatch, mock_app_mq_setup, {
        'query': 'test query', 'learn': False, 'respond': False,
    })
    assert status == 200
    assert 'retrieved_context' not in data
    assert 'retrieved_context_meta' not in data
