"""Tests for neurogenesis: extract_new_concepts, create_note, slugify, update_vault_index, append_to_vault_log."""
import os
import json
import tempfile
import pytest
import harness
import bdh_graph_harness.config as bdh_config
import bdh_graph_harness.neurogenesis.creator as bdh_creator


@pytest.fixture
def temp_vault():
    d = tempfile.mkdtemp()
    return d


# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------

def test_slugify_simple():
    assert harness.slugify('Hello World') == 'hello-world'


def test_slugify_caps():
    assert harness.slugify('My Cool Title') == 'my-cool-title'


def test_slugify_special_chars():
    assert harness.slugify('Test! @#$% Title') == 'test-title'


def test_slugify_spaces():
    assert harness.slugify('  spaces  here  ') == 'spaces-here'


def test_slugify_underscores():
    assert harness.slugify('foo_bar_baz') == 'foo-bar-baz'


def test_slugify_multiple_dashes():
    assert harness.slugify('a---b') == 'a-b'


def test_slugify_empty():
    assert harness.slugify('') == ''


# ---------------------------------------------------------------------------
# create_note
# ---------------------------------------------------------------------------

def test_create_note_creates_file(temp_vault):
    """Test that create_note creates a file with correct frontmatter and content."""
    note_id = harness.create_note(
        temp_vault, 'Test Concept', 'A test definition.',
        ['Source A', 'Source B'], 'test query'
    )
    assert note_id is not None
    assert note_id.startswith('concepts/') or note_id.startswith('wiki/concepts/')

    note_path = os.path.join(temp_vault, note_id + '.md')
    assert os.path.isfile(note_path)

    with open(note_path, 'r') as f:
        content = f.read()

    assert 'Test Concept' in content  # title present (may be quoted by YAML escape)
    assert 'tags: [neurogenesis, auto-generated]' in content
    assert 'confidence: low' in content
    assert 'A test definition.' in content
    assert 'created_by: bdh-neurogenesis' in content
    assert 'generation_query: "test query"' in content
    assert 'activated_from: "Source A, Source B"' in content
    assert '## Origin' not in content


def test_create_note_does_not_overwrite(temp_vault):
    """Test that create_note does NOT overwrite existing files."""
    harness.create_note(temp_vault, 'Existing', 'First def.', [], 'q1')
    result = harness.create_note(temp_vault, 'Existing', 'Second def.', [], 'q2')
    assert result is None  # should return None for existing

    # Verify original content preserved (path depends on config neurogenesis_dir)
    note_path = os.path.join(temp_vault, 'concepts', 'existing.md')
    if not os.path.isfile(note_path):
        note_path = os.path.join(temp_vault, 'wiki', 'concepts', 'existing.md')
    with open(note_path, 'r') as f:
        content = f.read()
    assert 'First def.' in content
    assert 'Second def.' not in content


def test_create_note_source_links(temp_vault):
    """Test that create_note includes wikilinks to source notes."""
    note_id = harness.create_note(
        temp_vault, 'New Concept', 'Definition.',
        ['Alpha', 'Beta', 'Gamma'], 'query'
    )
    note_path = os.path.join(temp_vault, note_id + '.md')
    with open(note_path, 'r') as f:
        content = f.read()
    assert '[[concepts/alpha|Alpha]]' in content
    assert '[[concepts/beta|Beta]]' in content


def test_create_note_creates_concepts_dir(temp_vault):
    """Test that create_note creates the concepts/ directory if needed."""
    concepts_dir = os.path.join(temp_vault, 'concepts')
    wiki_concepts_dir = os.path.join(temp_vault, 'wiki', 'concepts')
    assert not os.path.isdir(concepts_dir)
    assert not os.path.isdir(wiki_concepts_dir)

    harness.create_note(temp_vault, 'New', 'Def.', [], 'q')
    # Path depends on config neurogenesis_dir (concepts/ or wiki/concepts/)
    assert os.path.isdir(concepts_dir) or os.path.isdir(wiki_concepts_dir)


# ---------------------------------------------------------------------------
# update_vault_index
# ---------------------------------------------------------------------------

def test_update_vault_index_new_section(temp_vault):
    """Test that update_vault_index adds entry to a new section."""
    os.makedirs(os.path.join(temp_vault, 'wiki'))
    harness.update_vault_index(temp_vault, 'concepts/foo', 'Foo', section='concepts')

    index_path = os.path.join(temp_vault, 'wiki', 'index.md')
    assert os.path.isfile(index_path)
    with open(index_path, 'r') as f:
        content = f.read()
    assert '## concepts' in content
    assert '[[concepts/foo|Foo]]' in content


def test_update_vault_index_existing_section(temp_vault):
    """Test that update_vault_index adds entry to an existing section."""
    os.makedirs(os.path.join(temp_vault, 'wiki'))
    index_path = os.path.join(temp_vault, 'wiki', 'index.md')
    with open(index_path, 'w') as f:
        f.write("# Index\n\n## concepts\n- [[concepts/old|Old]]\n")

    harness.update_vault_index(temp_vault, 'concepts/new', 'New', section='concepts')

    with open(index_path, 'r') as f:
        content = f.read()
    assert '[[concepts/old|Old]]' in content
    assert '[[concepts/new|New]]' in content
    # New entry should be after old entry
    assert content.index('old') < content.index('new')


def test_update_vault_index_creates_wiki_dir(temp_vault):
    """Test that update_vault_index creates wiki/ dir if needed."""
    harness.update_vault_index(temp_vault, 'concepts/test', 'Test', section='concepts')
    assert os.path.isfile(os.path.join(temp_vault, 'wiki', 'index.md'))


# ---------------------------------------------------------------------------
# append_to_vault_log
# ---------------------------------------------------------------------------

def test_append_to_vault_log_creates_file(temp_vault):
    """Test that append_to_vault_log creates wiki/log.md."""
    harness.append_to_vault_log(temp_vault, 'Test action')
    log_path = os.path.join(temp_vault, 'wiki', 'log.md')
    assert os.path.isfile(log_path)
    with open(log_path, 'r') as f:
        content = f.read()
    assert '# BDH Graph Harness Log' in content
    assert 'Test action' in content


def test_append_to_vault_log_appends(temp_vault):
    """Test that append_to_vault_log appends to existing log."""
    harness.append_to_vault_log(temp_vault, 'First action')
    harness.append_to_vault_log(temp_vault, 'Second action')

    log_path = os.path.join(temp_vault, 'wiki', 'log.md')
    with open(log_path, 'r') as f:
        content = f.read()
    assert 'First action' in content
    assert 'Second action' in content
    assert content.index('First action') < content.index('Second action')


def test_append_to_vault_log_includes_timestamp(temp_vault):
    """Test that log entry includes a timestamp."""
    harness.append_to_vault_log(temp_vault, 'Timed action')
    log_path = os.path.join(temp_vault, 'wiki', 'log.md')
    with open(log_path, 'r') as f:
        content = f.read()
    # Should have ISO-like timestamp
    assert '202' in content  # year prefix


# ---------------------------------------------------------------------------
# extract_new_concepts (mocked LLM)
# ---------------------------------------------------------------------------

def test_extract_new_concepts_mocked(monkeypatch):
    """Test extract_new_concepts with mocked LLM call."""
    mock_concepts = [
        {'title': 'Quantum Entanglement', 'definition': 'A quantum phenomenon.'},
        {'title': 'Neural Plasticity', 'definition': 'Brain adaptation.'},
    ]

    monkeypatch.setattr(bdh_creator, 'retry_with_backoff', lambda fn: mock_concepts)

    nodes = {'a': {'title': 'Existing'}}
    result = harness.extract_new_concepts('Some response', 'query', {'a': 0.5}, nodes)
    assert len(result) == 2
    assert result[0]['title'] == 'Quantum Entanglement'
    assert result[1]['title'] == 'Neural Plasticity'


def test_extract_new_concepts_empty(monkeypatch):
    """Test extract_new_concepts returns empty list on LLM error."""
    def raise_error(fn):
        raise Exception('LLM unavailable')

    monkeypatch.setattr(bdh_creator, 'retry_with_backoff', raise_error)

    nodes = {'a': {'title': 'Existing'}}
    result = harness.extract_new_concepts('response', 'query', {}, nodes)
    assert result == []


def test_extract_new_concepts_json_with_fences(monkeypatch):
    """Test that JSON wrapped in markdown fences is parsed correctly."""
    raw_content = '```json\n[{"title": "Test", "definition": "Def"}]\n```'

    # Mock urllib.request.urlopen to return our fenced JSON
    class MockResp:
        def __init__(self, data):
            self._data = data
        def read(self):
            return self._data
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    mock_result = json.dumps({'message': {'content': raw_content}}).encode()

    import urllib.request as urlreq
    monkeypatch.setattr(urlreq, 'urlopen', lambda req, timeout=120: MockResp(mock_result))
    # Ensure OLLAMA_LLM_URL is set so the request URL is valid
    monkeypatch.setattr(bdh_config, 'OLLAMA_LLM_URL', 'http://localhost:11434/api/chat')
    # Skip retry delays
    monkeypatch.setattr(bdh_creator, 'retry_with_backoff', lambda fn: fn())

    nodes = {'a': {'title': 'Existing'}}
    result = harness.extract_new_concepts('response', 'query', {}, nodes)
    assert len(result) == 1
    assert result[0]['title'] == 'Test'


def test_extract_new_concepts_openrouter_object_wrapper(monkeypatch):
    """OpenRouter's json_object contract uses a concepts wrapper."""
    raw_content = json.dumps({
        'concepts': [
            {'title': 'Contrastive Learning', 'definition': 'Learns by comparing pairs.'},
        ],
    })

    class MockResp:
        def read(self):
            return json.dumps({'choices': [{'message': {'content': raw_content}}]}).encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    import urllib.request as urlreq
    monkeypatch.setattr(urlreq, 'urlopen', lambda req, timeout=120: MockResp())
    monkeypatch.setattr(bdh_config, 'OLLAMA_LLM_URL', 'https://openrouter.ai/api/v1/chat/completions')
    monkeypatch.setattr(bdh_creator, 'retry_with_backoff', lambda fn: fn())
    monkeypatch.setattr(bdh_creator, 'is_semantic_duplicate', lambda *args: False)
    monkeypatch.setitem(bdh_creator.CONFIG, 'llm_provider', 'openrouter')

    result = harness.extract_new_concepts('Contrastive learning response', 'query', {}, {})
    assert result == [{'title': 'Contrastive Learning', 'definition': 'Learns by comparing pairs.'}]


def test_extract_new_concepts_accepts_openrouter_single_object(monkeypatch):
    """Tolerate the object OpenRouter returned during the live contract test."""
    raw_content = json.dumps({
        'title': 'Contrastive Learning',
        'definition': 'Learns by comparing pairs.',
    })

    class MockResp:
        def read(self):
            return json.dumps({'choices': [{'message': {'content': raw_content}}]}).encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    import urllib.request as urlreq
    monkeypatch.setattr(urlreq, 'urlopen', lambda req, timeout=120: MockResp())
    monkeypatch.setattr(bdh_config, 'OLLAMA_LLM_URL', 'https://openrouter.ai/api/v1/chat/completions')
    monkeypatch.setattr(bdh_creator, 'retry_with_backoff', lambda fn: fn())
    monkeypatch.setattr(bdh_creator, 'is_semantic_duplicate', lambda *args: False)
    monkeypatch.setitem(bdh_creator.CONFIG, 'llm_provider', 'openrouter')

    result = harness.extract_new_concepts('Contrastive learning response', 'query', {}, {})
    assert result == [{'title': 'Contrastive Learning', 'definition': 'Learns by comparing pairs.'}]


def test_extract_new_concepts_prompt_is_signal_first_and_bounded(monkeypatch):
    """Prompt requests explicit evidence, conservative extraction, and max five concepts."""
    captured = {}

    class MockResp:
        def read(self):
            return json.dumps({'message': {'content': '{"concepts": []}'}}).encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    import urllib.request as urlreq
    def capture_request(req, timeout=120):
        captured['payload'] = json.loads(req.data)
        return MockResp()

    monkeypatch.setattr(urlreq, 'urlopen', capture_request)
    monkeypatch.setattr(bdh_config, 'OLLAMA_LLM_URL', 'http://localhost:11434/api/chat')
    monkeypatch.setattr(bdh_creator, 'retry_with_backoff', lambda fn: fn())
    monkeypatch.setitem(bdh_creator.CONFIG, 'llm_provider', 'openrouter')

    harness.extract_new_concepts('response', 'query', {}, {})
    system_prompt = captured['payload']['messages'][0]['content']
    assert 'explicitly present in the response' in system_prompt
    assert 'Return at most 5 concepts' in system_prompt
    assert 'When evidence is weak, return []' in system_prompt
    assert captured['payload']['response_format'] == {'type': 'json_object'}