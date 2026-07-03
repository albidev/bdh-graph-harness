"""Tests for neurogenesis: extract_new_concepts, create_note, slugify, update_vault_index, append_to_vault_log."""
import os
import json
import tempfile
import pytest
import harness


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
    assert note_id.startswith('concepts/')

    note_path = os.path.join(temp_vault, note_id + '.md')
    assert os.path.isfile(note_path)

    with open(note_path, 'r') as f:
        content = f.read()

    assert 'title: Test Concept' in content
    assert 'tags: [neurogenesis, auto-generated]' in content
    assert 'confidence: low' in content
    assert 'A test definition.' in content
    assert 'test query' in content


def test_create_note_does_not_overwrite(temp_vault):
    """Test that create_note does NOT overwrite existing files."""
    harness.create_note(temp_vault, 'Existing', 'First def.', [], 'q1')
    result = harness.create_note(temp_vault, 'Existing', 'Second def.', [], 'q2')
    assert result is None  # should return None for existing

    # Verify original content preserved
    note_path = os.path.join(temp_vault, 'concepts', 'existing.md')
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
    assert not os.path.isdir(concepts_dir)

    harness.create_note(temp_vault, 'New', 'Def.', [], 'q')
    assert os.path.isdir(concepts_dir)


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

    monkeypatch.setattr(harness, 'retry_with_backoff', lambda fn: mock_concepts)

    nodes = {'a': {'title': 'Existing'}}
    result = harness.extract_new_concepts('Some response', 'query', {'a': 0.5}, nodes)
    assert len(result) == 2
    assert result[0]['title'] == 'Quantum Entanglement'
    assert result[1]['title'] == 'Neural Plasticity'


def test_extract_new_concepts_empty(monkeypatch):
    """Test extract_new_concepts returns empty list on LLM error."""
    def raise_error(fn):
        raise Exception('LLM unavailable')

    monkeypatch.setattr(harness, 'retry_with_backoff', raise_error)

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
    monkeypatch.setattr(harness, 'OLLAMA_LLM_URL', 'http://localhost:11434/api/chat')
    # Skip retry delays
    monkeypatch.setattr(harness, 'retry_with_backoff', lambda fn: fn())

    nodes = {'a': {'title': 'Existing'}}
    result = harness.extract_new_concepts('response', 'query', {}, nodes)
    assert len(result) == 1
    assert result[0]['title'] == 'Test'