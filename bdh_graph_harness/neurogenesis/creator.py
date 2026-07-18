"""Neurogenesis — extract new concepts, create notes, and update vault index."""

import os
import re
import sys
import json

from bdh_graph_harness.config import CONFIG, retry_with_backoff
import bdh_graph_harness.config as _config
from bdh_graph_harness.neurogenesis.dedupe import is_duplicate, is_semantic_duplicate


# --- Noise filters (deterministic, applied before LLM and after) ---

import re as _re

# Regex patterns for concepts that are ALWAYS noise — never create a note for these.
# Matched against the slugified title (lowercase, hyphenated).
_NOISE_PATTERNS = [
    # LLM model names (glm-5, gemma3, ministral-3b, gpt-oss-20b, mimo-v2.5, etc.)
    _re.compile(r'^(glm|gemma|mistral|ministral|minimax|gpt|gpt-oss|llama|qwen|deepseek|phi|codestral|mimo|command-r|mixtral|yi|solar|dbrx|gemma\d|nemotron)[\w.-]*$'),
    # Infrastructure/tool names that are just labels, not concepts
    _re.compile(r'^(ollama|ollama-cloud|openrouter|opencode|opencode-zen|huggingface|anthropic|together|groq|fireworks|runway)$'),
    # BDH internal plumbing — the graph describing its own operations
    _re.compile(r'^(changed-nodes|graph-refresh|hebbian-update|hebbian-updates|incremental-update|incremental-updates|initnetwork|newconcepts|delta-update|pulse-animation|removed-nodes|added-node-data|diamond-shape|neuron-particles|new-concepts|graph-rebuild|full-rebuild)$'),
    # Honcho internal components
    _re.compile(r'^honcho-(deriver|dialectic|dream|summary|context|search|reasoning)$'),
    # Generic process descriptors (no domain signal)
    _re.compile(r'^(incremental|delta|update|refresh|rebuild|init|startup|shutdown|pulse|animation|changed|removed|added)$'),
]


def _is_noise_title(title: str) -> bool:
    """Deterministic check: is this title pure noise that should never become a note?"""
    slug = slugify(title)
    for pat in _NOISE_PATTERNS:
        if pat.match(slug):
            return True
    # Reject very short slugs (< 4 chars) — almost always noise
    if len(slug) < 4:
        return True
    # Reject pure version numbers or sizes (e.g. "3b", "20b", "v2.5")
    if _re.fullmatch(r'[\d.v-]+', slug):
        return True
    return False


def extract_new_concepts(llm_response, query, active_notes, nodes, *, allow_existing=False):
    """Ask the LLM to identify concepts in its response that aren't in the vault."""
    import urllib.request

    # List existing note titles
    existing_titles = [node['title'] for node in nodes.values()]

    system_prompt = (
        "You are a conservative concept extractor for a persistent knowledge graph. "
        "The knowledge graph may contain information from any domain. Infer the relevant domain from the response "
        "and the existing vault context; never assume a fixed subject area. "
        "Given an LLM response and a list of existing concept titles in the vault, identify only NEW concepts "
        "that are explicitly present in the response and are NOT in the existing list. Never infer or invent a "
        "concept merely because it is related to the query.\n\n"
        "EXTRACT concepts that are:\n"
        "- Algorithms, architectures, or design patterns (e.g. 'sparse attention', 'mixture-of-experts')\n"
        "- Technical methods or techniques with a specific domain (e.g. 'contrastive learning', 'kv-cache')\n"
        "- Lessons learned, error patterns, or debugging insights (e.g. 'b-tree corruption recovery')\n"
        "- Domain concepts that someone would want to look up later\n\n"
        "DO NOT extract:\n"
        "- Names of LLM models (e.g. 'glm-5', 'gemma3', 'mistral', 'llama')\n"
        "- Names of API providers or tools (e.g. 'ollama', 'openrouter', 'huggingface')\n"
        "- Internal operations of the knowledge graph itself (e.g. 'graph refresh', 'hebbian update', 'neurogenesis', 'changed nodes')\n"
        "- Generic process words without domain meaning (e.g. 'incremental update', 'delta', 'refresh', 'init')\n"
        "- Meta-descriptions of what the LLM is doing right now\n"
        "- Concepts that are just synonyms or spelling variants of existing ones\n"
        "- Proper nouns that are just labels (company names, product names) without technical depth\n\n"
        "First decide whether the response contains a durable, reusable concept at all. Set durable=false "
        "for live operational questions, status updates, UI observations, current-session debugging, requests "
        "to proceed, or a description of what the agent is doing right now. Set durable=true only for a reusable "
        "principle, lesson, root cause, architecture pattern, algorithm, or domain concept that remains useful "
        "outside this session. When durable=false, concepts MUST be [].\n\n"
        "For each new concept, provide a short title of 2-5 words and a one-sentence definition. "
        "Return at most 5 concepts. When evidence is weak, return {\"durable\": false, \"concepts\": []}.\n\n"
        'The required JSON shape is exactly: {"durable": true|false, "concepts": [{"title": "...", "definition": "..."}]}.'
    )

    user_prompt = f"""## Existing concept titles in vault
{json.dumps(existing_titles)}

## LLM response
{llm_response}

## Original query
{query}
"""

    data = json.dumps({
        "model": CONFIG['llm_model'],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        **({"format": "json", "options": {"temperature": 0.1, "num_ctx": CONFIG['llm_max_ctx']}}
           if CONFIG.get('llm_provider', 'ollama') == 'ollama'
           else {"temperature": 0.1, "max_tokens": CONFIG['llm_max_ctx'],
                 "response_format": {"type": "json_object"}}),
    }).encode()

    provider = CONFIG.get('llm_provider', 'ollama')
    headers = {'Content-Type': 'application/json'}
    if provider == 'openrouter':
        headers['Authorization'] = f"Bearer {CONFIG.get('openrouter_key', '')}"
        headers['HTTP-Referer'] = 'https://github.com/bdh-graph-harness'
        headers['X-Title'] = 'BDH Graph Harness'
        headers['User-Agent'] = 'BDH-Graph-Harness/1.0'

    def _extract_call():
        req = urllib.request.Request(_config.OLLAMA_LLM_URL, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            if provider == 'openrouter':
                choices = result.get('choices', [])
                content = choices[0].get('message', {}).get('content', '[]') if choices else '[]'
            else:
                content = result.get('message', {}).get('content', '[]')
            # Handle LLM returning text with embedded JSON
            content = content.strip()
            # Aggressively strip markdown code fences and prefixes
            # Remove any ```json or ``` markers
            content = re.sub(r'```(?:json)?', '', content)
            # Remove leading "json" prefix that gemma4 sometimes adds
            content = re.sub(r'^json\s*', '', content)
            content = content.strip()
            # Parse the canonical object wrapper, while accepting legacy arrays
            # and the single-object shape returned by some OpenRouter free models.
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
                if not match:
                    return []
                parsed = json.loads(match.group(1))

            if isinstance(parsed, dict) and parsed.get('durable') is False:
                return []
            if isinstance(parsed, dict) and isinstance(parsed.get('concepts'), list):
                concepts = parsed['concepts']
            elif isinstance(parsed, dict) and 'title' in parsed and 'definition' in parsed:
                concepts = [parsed]
            elif isinstance(parsed, list):
                concepts = parsed
            else:
                concepts = []

            # Filter out:
            # 1. Concepts that already exist (exact title match)
            # 2. Noise titles (regex blocklist — model names, plumbing, etc.)
            # 3. Semantic duplicates (embedding similarity > threshold)
            filtered = []
            for c in concepts:
                if not isinstance(c, dict) or 'title' not in c:
                    continue
                title = c['title']
                definition = c.get('definition', '')
                if not isinstance(title, str) or not isinstance(definition, str):
                    continue
                title = title.strip()
                definition = definition.strip()
                if not title or not definition:
                    continue
                exact_match = is_duplicate(title, existing_titles)
                if exact_match and not allow_existing:
                    continue
                if _is_noise_title(title):
                    print(f"  🚫 Noise filter rejected: '{title}'", file=sys.stderr)
                    continue
                semantic_match = is_semantic_duplicate(title, definition)
                if semantic_match and not allow_existing:
                    print(f"  🔁 Semantic duplicate rejected: '{title}'", file=sys.stderr)
                    continue
                item = {'title': title, 'definition': definition}
                if allow_existing:
                    item['_exact_match'] = exact_match
                    item['_semantic_match'] = semantic_match
                filtered.append(item)
            return filtered[:5]

    try:
        return retry_with_backoff(_extract_call)
    except Exception as e:
        print(f"  ⚠ Neurogenesis extraction error: {e}", file=sys.stderr)
        return []


def slugify(title):
    """Convert a title to a filename-safe slug."""
    slug = title.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[\s_]+', '-', slug)
    slug = re.sub(r'-+', '-', slug)
    return slug.strip('-')


def _yaml_escape(value):
    """Escape a string for safe YAML frontmatter value.
    Wraps in double quotes and escapes special chars to prevent YAML injection."""
    if not value:
        return '""'
    # Escape backslash and double quotes, then wrap in double quotes
    escaped = value.replace('\\', '\\\\').replace('"', '\\"')
    # Also escape newlines (shouldn't happen in titles, but safety first)
    escaped = escaped.replace('\n', '\\n')
    return f'"{escaped}"'


def _sanitize_for_note(text, max_len=200):
    """Sanitize text for inclusion in a vault note body.
    Strips control chars and truncates. Prevents prompt injection from becoming
    executable YAML or markdown frontmatter injection."""
    if not text:
        return ''
    # Remove null bytes and other control chars (except newline/tab)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    return text[:max_len]


def _serialize_source_node_ids(source_node_ids):
    """Serialize validated source IDs as a deterministic JSON list."""
    result = []
    for source_id in source_node_ids or []:
        if not isinstance(source_id, str):
            continue
        source_id = source_id.strip()
        if source_id and '\n' not in source_id and '\r' not in source_id and source_id not in result:
            result.append(source_id)
    return json.dumps(result, ensure_ascii=False)


def create_note(
    vault_root,
    title,
    definition,
    source_notes,
    query,
    neurogenesis_dir=None,
    source_node_ids=None,
):
    """Create a new atomic note in the vault (neurogenesis)."""
    from datetime import datetime

    neurogenesis_dir = neurogenesis_dir or CONFIG['neurogenesis_dir']
    now = datetime.now().isoformat()
    slug = slugify(title)
    note_path = os.path.join(vault_root, neurogenesis_dir, f"{slug}.md")

    # Don't overwrite existing notes
    if os.path.isfile(note_path):
        return None

    # Keep provenance in frontmatter: parser.extract_text() excludes it from
    # embeddings, so generation metadata cannot become a shared retrieval
    # attractor across every neurogenesis note.
    safe_title = _yaml_escape(title)
    safe_query = _yaml_escape(_sanitize_for_note(query, max_len=200))
    safe_sources = _yaml_escape(', '.join(
        _sanitize_for_note(str(s), max_len=120) for s in source_notes[:3]
    ))
    source_ids = _serialize_source_node_ids(source_node_ids)

    content = f"""---
title: {safe_title}
created: {now[:10]}
updated: {now[:10]}
type: concept
tags: [neurogenesis, auto-generated]
sources: []
confidence: low
created_by: bdh-neurogenesis
generation_query: {safe_query}
activated_from: {safe_sources}
activated_from_ids: {source_ids}
---

# {title}

{definition}
"""

    os.makedirs(os.path.dirname(note_path), exist_ok=True)
    with open(note_path, 'w', encoding='utf-8') as f:
        f.write(content)

    note_id = f"{neurogenesis_dir}/{slug}"

    # Update vault index and log
    update_vault_index(vault_root, note_id, title, section=neurogenesis_dir)
    append_to_vault_log(vault_root, f"Neurogenesis: created '{title}' ({note_id}) from query '{query[:80]}'")

    return note_id


def update_vault_index(vault_root, note_id, title, section='concepts'):
    """Add a new note entry to the vault's wiki/index.md under the right section."""
    index_path = os.path.join(vault_root, 'wiki', 'index.md')

    # Read existing index
    content = ""
    if os.path.isfile(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            content = f.read()

    # Build the new entry line
    entry = f"- [[{note_id}|{title}]]"

    # Try to find a section header matching the section name
    section_header = f"## {section}"
    if section_header in content:
        # Insert after the section header, before the next section
        lines = content.split('\n')
        insert_idx = None
        for i, line in enumerate(lines):
            if line.strip() == section_header:
                insert_idx = i + 1
                break
        if insert_idx is not None:
            # Find the right spot (after existing entries in this section)
            while insert_idx < len(lines) and lines[insert_idx].strip().startswith('-'):
                insert_idx += 1
            lines.insert(insert_idx, entry)
            content = '\n'.join(lines)
        else:
            content += f"\n{section_header}\n{entry}\n"
    else:
        # No matching section; append a new section
        if content and not content.endswith('\n'):
            content += '\n'
        content += f"\n{section_header}\n{entry}\n"

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(content)


def append_to_vault_log(vault_root, action_text):
    """Append an action entry to the vault's wiki/log.md."""
    from datetime import datetime

    log_path = os.path.join(vault_root, 'wiki', 'log.md')
    now = datetime.now().isoformat()

    entry = f"- {now} — {action_text}\n"

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Read existing content
    existing = ""
    if os.path.isfile(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            existing = f.read()

    # If file doesn't start with a header, add one
    if not existing.strip():
        existing = "# BDH Graph Harness Log\n\n"
    elif not existing.startswith('#'):
        existing = "# BDH Graph Harness Log\n\n" + existing

    if not existing.endswith('\n'):
        existing += '\n'

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(existing + entry)