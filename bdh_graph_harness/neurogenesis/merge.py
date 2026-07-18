"""Non-destructive assimilation of neurogenesis evidence into canonical notes."""

from __future__ import annotations

import os
import re
import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

from bdh_graph_harness.graph.parser import parse_frontmatter, parse_json_frontmatter_list

MERGE_SIMILARITY_THRESHOLD = 0.82

_CONFLICT_RE = re.compile(
    r"\b(?:not|never|instead|contradict|contrary|supersede|replace|avoid|non|mai|invece|contradd)\b",
    re.IGNORECASE,
)


def looks_conflicting(definition: str) -> bool:
    """Return True for explicit negation/replacement language.

    This is deliberately conservative: a possible conflict must not be silently
    merged. A future LLM-assisted reconciliation step can resolve it explicitly.
    """
    return bool(_CONFLICT_RE.search(definition or ""))


def _update_frontmatter(content: str) -> str:
    """Update the note date without disturbing unknown frontmatter fields."""
    today = date.today().isoformat()
    if re.search(r"^updated:\s*.*$", content, flags=re.MULTILINE):
        return re.sub(r"^updated:\s*.*$", f"updated: {today}", content, count=1, flags=re.MULTILINE)
    if content.startswith("---\n"):
        marker = content.find("\n---", 4)
        if marker >= 0:
            return content[:marker] + f"\nupdated: {today}" + content[marker:]
    return content


def _merge_source_node_ids(content: str, source_node_ids: list[str] | None) -> str:
    """Append canonical provenance IDs to frontmatter without duplicates."""
    incoming = []
    for source_id in source_node_ids or []:
        if isinstance(source_id, str):
            source_id = source_id.strip()
            if source_id and '\n' not in source_id and '\r' not in source_id:
                incoming.append(source_id)
    if not incoming:
        return content
    existing = parse_json_frontmatter_list(
        parse_frontmatter(content), 'activated_from_ids'
    )
    merged = list(existing)
    for source_id in incoming:
        if source_id not in merged:
            merged.append(source_id)
    serialized = json.dumps(merged, ensure_ascii=False)
    replacement = f"activated_from_ids: {serialized}"
    if re.search(r"^activated_from_ids:\s*.*$", content, flags=re.MULTILINE):
        return re.sub(
            r"^activated_from_ids:\s*.*$",
            replacement,
            content,
            count=1,
            flags=re.MULTILINE,
        )
    if content.startswith("---\n"):
        marker = content.find("\n---", 4)
        if marker >= 0:
            return content[:marker] + f"\n{replacement}" + content[marker:]
    return content


def assimilate_evidence(
    vault_root: str | os.PathLike[str],
    node_id: str,
    node: dict[str, Any],
    definition: str,
    *,
    source_notes: list[str] | None = None,
    source_node_ids: list[str] | None = None,
    query: str = "",
) -> dict[str, Any]:
    """Append new evidence to an existing canonical neurogenesis note.

    Returns a structured result. Existing content is never replaced; repeated
    evidence is detected by normalized text and becomes a no-op.
    """
    raw_path = node.get("absolute_path")
    if raw_path:
        note_path = Path(raw_path)
    else:
        relative = node.get("relative_path") or node.get("path")
        if not relative:
            return {"status": "unavailable", "node_id": node_id}
        note_path = Path(vault_root) / str(relative)
    if not note_path.is_file():
        return {"status": "unavailable", "node_id": node_id, "path": str(note_path)}

    existing = note_path.read_text(encoding="utf-8")
    normalized = " ".join((definition or "").split()).casefold()
    if not normalized:
        return {"status": "ignored", "node_id": node_id}
    if normalized in " ".join(existing.split()).casefold():
        return {"status": "already_present", "node_id": node_id, "path": str(note_path)}

    sources = ", ".join(str(item) for item in (source_notes or [])[:3]) or "session recovery"
    query_line = " ".join((query or "").split())[:240]
    section = (
        "\n\n## Assimilated Evidence\n"
        f"- **{date.today().isoformat()}** — {definition.strip()}\n"
        f"  - source: {sources}\n"
        f"  - query: {query_line}\n"
    )
    updated = _merge_source_node_ids(existing.rstrip() + section + "\n", source_node_ids)
    updated = _update_frontmatter(updated)
    if updated == existing:
        return {"status": "already_present", "node_id": node_id, "path": str(note_path)}

    fd, tmp_name = tempfile.mkstemp(prefix=f".{note_path.name}.", dir=str(note_path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(updated)
        os.replace(tmp_name, note_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    return {"status": "merged", "node_id": node_id, "path": str(note_path)}
