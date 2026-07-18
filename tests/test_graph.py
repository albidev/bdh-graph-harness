"""Tests for graph extraction: build_graph, extract_wikilinks, extract_text, parse_frontmatter, extract_note_id."""
import os
import tempfile
import textwrap
import pytest
import harness


# ---------------------------------------------------------------------------
# extract_note_id
# ---------------------------------------------------------------------------

def test_extract_note_id_simple():
    fp = '/vault/notes/foo.md'
    assert harness.extract_note_id(fp, '/vault/notes') == 'foo'


def test_extract_note_id_nested():
    fp = '/vault/wiki/concepts/sub/page.md'
    assert harness.extract_note_id(fp, '/vault') == 'wiki/concepts/sub/page'


def test_extract_note_id_no_md_extension():
    fp = '/vault/notes/readme.txt'
    assert harness.extract_note_id(fp, '/vault/notes') == 'readme.txt'


# ---------------------------------------------------------------------------
# parse_frontmatter
# ---------------------------------------------------------------------------

def test_parse_frontmatter_basic():
    content = "---\ntitle: My Note\ntags: [a, b]\n---\nBody text"
    fm = harness.parse_frontmatter(content)
    assert fm['title'] == 'My Note'
    assert fm['tags'] == '[a, b]'


def test_parse_frontmatter_none():
    content = "No frontmatter here\nJust body"
    fm = harness.parse_frontmatter(content)
    assert fm == {}


def test_parse_frontmatter_empty_value():
    content = "---\ntitle: \n---\nbody"
    fm = harness.parse_frontmatter(content)
    assert 'title' in fm


def test_parse_json_frontmatter_list_sanitizes_source_ids():
    fm = {'activated_from_ids': '["vault:source.md", "external:projects/demo.md", 42]'}
    assert harness.parse_json_frontmatter_list(fm, 'activated_from_ids') == [
        'vault:source.md',
        'external:projects/demo.md',
    ]
    assert harness.parse_json_frontmatter_list(
        {'activated_from_ids': 'not-json'}, 'activated_from_ids'
    ) == []


# ---------------------------------------------------------------------------
# extract_wikilinks
# ---------------------------------------------------------------------------

def test_extract_wikilinks_simple():
    links = harness.extract_wikilinks("See [[target]] for more")
    assert links == [('target', 'target')]


def test_extract_wikilinks_with_display():
    links = harness.extract_wikilinks("See [[target|display text]] for more")
    assert links == [('target', 'display text')]


def test_extract_wikilinks_nested_path():
    links = harness.extract_wikilinks("See [[concepts/sub/page]] for more")
    assert links == [('concepts/sub/page', 'concepts/sub/page')]


def test_extract_wikilinks_multiple():
    content = "[[a]] and [[b|Bee]] and [[c/sub/page]]"
    links = harness.extract_wikilinks(content)
    assert len(links) == 3
    assert links[0] == ('a', 'a')
    assert links[1] == ('b', 'Bee')
    assert links[2] == ('c/sub/page', 'c/sub/page')


def test_extract_wikilinks_none():
    links = harness.extract_wikilinks("No wikilinks here")
    assert links == []


def test_extract_wikilinks_ignores_code_examples():
    content = "See [[real-note]]. `[[INLINE_EXAMPLE]]`\n```md\n[[FENCED_EXAMPLE]]\n```"
    assert harness.extract_wikilinks(content) == [('real-note', 'real-note')]

# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------

def test_extract_text_strips_frontmatter():
    content = "---\ntitle: Test\n---\n# Heading\nBody text"
    text = harness.extract_text(content)
    assert 'title' not in text  # frontmatter stripped
    assert 'Body text' in text


def test_extract_text_converts_wikilinks():
    content = "See [[target|display]] here"
    text = harness.extract_text(content)
    assert 'display' in text
    assert '[[' not in text


def test_extract_text_strips_markdown():
    content = "# Heading\n**bold** and *italic* and `code`"
    text = harness.extract_text(content)
    assert '#' not in text
    assert '*' not in text
    assert '`' not in text
    assert 'Heading' in text
    assert 'bold' in text


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_vault():
    d = tempfile.mkdtemp()
    # Note with frontmatter
    with open(os.path.join(d, 'alpha.md'), 'w') as f:
        f.write(textwrap.dedent("""\
            ---
            title: Alpha Note
            tags: [concept, core]
            ---
            # Alpha Note
            This links to [[beta]] and [[concepts/gamma|Gamma]]."""))
    # Note without frontmatter
    with open(os.path.join(d, 'beta.md'), 'w') as f:
        f.write("# Beta\nLinks back to [[alpha]].")
    # Nested note
    os.makedirs(os.path.join(d, 'concepts'))
    with open(os.path.join(d, 'concepts', 'gamma.md'), 'w') as f:
        f.write("---\ntitle: Gamma\ntags: [concept]\n---\nGamma content [[alpha|Alpha]].")
    return d


def test_build_graph_node_count(mock_vault):
    nodes, edges = harness.build_graph(mock_vault)
    assert len(nodes) == 3
    assert 'alpha' in nodes
    assert 'beta' in nodes
    assert 'concepts/gamma' in nodes


def test_build_graph_accepts_per_vault_ignore_patterns(mock_vault):
    nodes, _ = harness.build_graph(mock_vault, use_cache=False, graph_ignore=['beta'])
    assert 'beta' not in nodes
    assert 'alpha' in nodes


def test_build_graph_node_titles(mock_vault):
    nodes, _ = harness.build_graph(mock_vault)
    assert nodes['alpha']['title'] == 'Alpha Note'
    assert nodes['beta']['title'] == 'beta'  # no frontmatter → filename
    assert nodes['concepts/gamma']['title'] == 'Gamma'


def test_build_graph_node_tags(mock_vault):
    nodes, _ = harness.build_graph(mock_vault)
    assert nodes['alpha']['tags'] == '[concept, core]'
    assert nodes['beta']['tags'] == ''  # no frontmatter


def test_build_graph_edges(mock_vault):
    _, edges = harness.build_graph(mock_vault)
    assert 'alpha' in edges
    targets = [e['target'] for e in edges['alpha']]
    assert 'beta' in targets
    assert 'concepts/gamma' in targets


def test_legacy_builder_drops_resolved_self_wikilinks(tmp_path):
    with open(os.path.join(tmp_path, 'self.md'), 'w') as handle:
        handle.write('# Self\nSee [[self]].')

    nodes, edges = harness.build_graph(str(tmp_path), use_cache=False)

    assert 'self' in nodes
    assert edges.get('self', []) == []


def test_build_graph_edge_display(mock_vault):
    _, edges = harness.build_graph(mock_vault)
    for e in edges['alpha']:
        if e['target'] == 'concepts/gamma':
            assert e['display'] == 'Gamma'
            break


def test_build_graph_node_text(mock_vault):
    nodes, _ = harness.build_graph(mock_vault)
    assert 'Alpha Note' in nodes['alpha']['text'] or 'Alpha' in nodes['alpha']['text']
    assert '[[' not in nodes['alpha']['text']  # wikilinks converted


def test_build_graph_skips_hidden_dirs(mock_vault):
    os.makedirs(os.path.join(mock_vault, '.obsidian'))
    with open(os.path.join(mock_vault, '.obsidian', 'config.md'), 'w') as f:
        f.write("config")
    nodes, _ = harness.build_graph(mock_vault)
    assert len(nodes) == 3  # .obsidian skipped