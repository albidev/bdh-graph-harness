"""
BDH Graph Harness — Graph parser module.

Wikilink / frontmatter extraction utilities for Obsidian markdown notes.
"""

import os
import re

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

WIKILINK_RE = re.compile(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]')
FRONTMATTER_RE = re.compile(r'^---\n(.*?)\n---\n', re.DOTALL)

# ---------------------------------------------------------------------------
# Parser functions
# ---------------------------------------------------------------------------

def extract_note_id(filepath, vault_root):
    """Get the note's ID relative to vault root, without .md extension."""
    rel = os.path.relpath(filepath, vault_root)
    return rel[:-3] if rel.endswith('.md') else rel  # strip .md


def find_note_by_id(vault_root, note_id):
    """Find the actual file for a note ID (may have .md or be in a subdir)."""
    candidates = [
        os.path.join(vault_root, note_id + '.md'),
        os.path.join(vault_root, note_id),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Try case-insensitive search
    for root, dirs, files in os.walk(vault_root):
        for f in files:
            if f.endswith('.md'):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, vault_root)[:-3]
                if rel.lower() == note_id.lower():
                    return full
    return None


def parse_frontmatter(content):
    """Extract YAML frontmatter as a dict (simple parser)."""
    match = FRONTMATTER_RE.match(content)
    if not match:
        return {}
    fm = {}
    for line in match.group(1).split('\n'):
        if ':' in line:
            key, _, val = line.partition(':')
            fm[key.strip()] = val.strip()
    return fm


def extract_wikilinks(content):
    """Extract all wikilinks from note content. Returns list of (target, display)."""
    links = []
    for match in WIKILINK_RE.finditer(content):
        target = match.group(1).strip()
        display = match.group(2).strip() if match.group(2) else target
        links.append((target, display))
    return links


def extract_text(content):
    """Extract plain text from markdown (strip frontmatter, wikilinks, markdown)."""
    # Strip frontmatter
    text = FRONTMATTER_RE.sub('', content)
    # Convert wikilinks to display text
    text = WIKILINK_RE.sub(lambda m: m.group(2) or m.group(1), text)
    # Strip markdown formatting
    text = re.sub(r'[#*_`>\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text