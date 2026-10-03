"""
BDH Graph Harness — Graph parser module.

Wikilink / frontmatter extraction utilities for Obsidian markdown notes.
"""

import os
import re
import json

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

WIKILINK_RE = re.compile(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]')
FRONTMATTER_RE = re.compile(r'^---\n(.*?)\n---\n', re.DOTALL)

# Written by neurogenesis/merge.py::assimilate_evidence. Each section holds one
# claim bullet followed by indented audit sub-bullets.
ASSIMILATED_EVIDENCE_HEADING = '## Assimilated Evidence'
_EVIDENCE_AUDIT_LINE_RE = re.compile(
    r'^[ \t]+- (?:source|query|provenance|synthesis_session_id|synthesis_id|'
    r'transcript_sha256|queued_at):[^\n]*(?:\n|$)',
    re.MULTILINE,
)
_EVIDENCE_CLAIM_RE = re.compile(r'^- (?:\*\*\d{4}-\d{2}-\d{2}\*\* — )?(\S[^\n]*)$', re.MULTILINE)

# ---------------------------------------------------------------------------
# Parser functions
# ---------------------------------------------------------------------------

def extract_note_id(filepath, vault_root):
    """Get the note's ID relative to vault root, without .md extension."""
    rel = os.path.relpath(filepath, vault_root)
    return rel[:-3] if rel.endswith('.md') else rel  # strip .md


def find_note_by_id(vault_root, note_id):
    """Find the actual file for a note ID (may have .md or be in a subdir).

    Path traversal protection: normalizes the note_id and rejects any
    path that escapes the vault root.
    """
    # Normalize and check for path traversal
    note_id = os.path.normpath(note_id)
    full_check = os.path.normpath(os.path.join(vault_root, note_id))
    if not full_check.startswith(os.path.normpath(vault_root)):
        return None  # path traversal attempt

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


def parse_json_frontmatter_list(frontmatter, key):
    """Return a sanitized JSON-list frontmatter value as string IDs."""
    value = frontmatter.get(key)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if item and '\n' not in item and '\r' not in item and item not in result:
            result.append(item)
    return result


def extract_wikilinks(content):
    """Extract semantic wikilinks, ignoring Markdown code examples."""
    # Links shown inside fenced or inline code are documentation syntax, not
    # graph edges. Parsing them creates false unresolved targets such as
    # ``[[VALUE]]`` and ``[[COMPLETE]]`` from README templates.
    semantic_content = re.sub(r'(?ms)^\s*(```|~~~).*?^\s*\1\s*$', '', content)
    semantic_content = re.sub(r'`[^`\n]*`', '', semantic_content)
    links = []
    for match in WIKILINK_RE.finditer(semantic_content):
        target = match.group(1).strip()
        display = match.group(2).strip() if match.group(2) else target
        links.append((target, display))
    return links


def strip_evidence_audit(content):
    """Drop the audit sub-bullets of Assimilated Evidence sections.

    They stay in the file for traceability, but session ids, hashes and the
    repeated ``query: Explicit Curate merge target`` line are not knowledge:
    indexed, they become a shared lexical/semantic attractor across every merged
    note. ``create_note`` keeps the same provenance in frontmatter for this reason.
    """
    head, heading, tail = content.partition(ASSIMILATED_EVIDENCE_HEADING)
    if not heading:
        return content
    return head + heading + _EVIDENCE_AUDIT_LINE_RE.sub('', tail)


def _plain(markdown):
    text = WIKILINK_RE.sub(lambda m: m.group(2) or m.group(1), markdown)
    text = re.sub(r'[#*_`>\-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def extract_text(content):
    """Extract plain text from markdown (strip frontmatter, evidence audit lines, wikilinks, markdown)."""
    return _plain(strip_evidence_audit(FRONTMATTER_RE.sub('', content)))


def extract_evidence_claims(content):
    """Plain-text claims appended by merges, one per Assimilated Evidence bullet."""
    sections = re.split(r'^(?=## )', content, flags=re.MULTILINE)
    evidence = '\n'.join(s for s in sections if s.startswith(ASSIMILATED_EVIDENCE_HEADING))
    claims = []
    for match in _EVIDENCE_CLAIM_RE.finditer(_EVIDENCE_AUDIT_LINE_RE.sub('', evidence)):
        claim = _plain(match.group(1))
        if claim and claim not in claims:
            claims.append(claim)
    return claims