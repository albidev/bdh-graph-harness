"""Neurogenesis — extract new concepts, create notes, and update vault index."""

import os
import re
import sys
import json

from bdh_graph_harness.config import CONFIG, retry_with_backoff
import bdh_graph_harness.config as _config
from bdh_graph_harness.neurogenesis.dedupe import is_duplicate


def extract_new_concepts(llm_response, query, active_notes, nodes):
    """Ask the LLM to identify concepts in its response that aren't in the vault."""
    import urllib.request

    # List existing note titles
    existing_titles = [node['title'] for node in nodes.values()]

    system_prompt = (
        "You are a concept extractor for a knowledge graph. "
        "Given an LLM response and a list of existing concept titles in the vault, "
        "identify any NEW concepts introduced in the response that are NOT in the existing list. "
        "For each new concept, provide: 1) a short title, 2) a 1-sentence definition. "
        'Return as JSON array: [{"title": "...", "definition": "..."}]. '
        "If no new concepts, return []."
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
            # Try to find JSON array in the response
            if not content.startswith('['):
                match = re.search(r'\[.*\]', content, re.DOTALL)
                if match:
                    content = match.group(0)
                else:
                    return []
            concepts = json.loads(content)
            if isinstance(concepts, list):
                # Filter out duplicates that already exist
                concepts = [c for c in concepts
                            if isinstance(c, dict)
                            and 'title' in c
                            and not is_duplicate(c['title'], existing_titles)]
                return concepts
            return []

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


def create_note(vault_root, title, definition, source_notes, query):
    """Create a new atomic note in the vault (neurogenesis)."""
    from datetime import datetime

    now = datetime.now().isoformat()
    slug = slugify(title)
    note_path = os.path.join(vault_root, CONFIG['neurogenesis_dir'], f"{slug}.md")

    # Don't overwrite existing notes
    if os.path.isfile(note_path):
        return None

    # Build wikilinks to source notes
    source_links = "\n".join(f"- [[concepts/{slugify(s)}|{s}]]" for s in source_notes[:3])

    content = f"""---
title: {title}
created: {now[:10]}
updated: {now[:10]}
type: concept
tags: [neurogenesis, auto-generated]
sources: []
confidence: low
---

# {title}

{definition}

## Origin
- **Created by:** BDH Graph Harness neurogenesis
- **Query:** {query[:200]}
- **Activated from:** {', '.join(source_notes[:3])}

## Links
{source_links}
"""

    os.makedirs(os.path.dirname(note_path), exist_ok=True)
    with open(note_path, 'w', encoding='utf-8') as f:
        f.write(content)

    note_id = f"{CONFIG['neurogenesis_dir']}/{slug}"

    # Update vault index and log
    update_vault_index(vault_root, note_id, title, section=CONFIG['neurogenesis_dir'])
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