"""Read-only synthesis activity aggregation for Curate integrations."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from bdh_graph_harness.neurogenesis.operation_journal import list_operation_records
from bdh_graph_harness.memory.synthesis_audit import read_synthesis_audit


def _read_note_title(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    if text.startswith("---"):
        for line in text.splitlines()[1:]:
            if line.strip() == "---":
                break
            if line.startswith("title:"):
                return line.partition(":")[2].strip().strip('"').strip("'")
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _concept_details(vault_root: Path, concept_ids: list[str], nodes: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    concepts: list[dict[str, Any]] = []
    for concept_id in concept_ids:
        raw_id = str(concept_id or "").strip()
        relative = raw_id.removeprefix("vault:")
        note_path = vault_root / relative
        if note_path.suffix != ".md":
            note_path = note_path.with_suffix(".md")
        node = nodes.get(raw_id) if isinstance(nodes, Mapping) else None
        title = str(node.get("title") or "") if isinstance(node, Mapping) else ""
        title = title or _read_note_title(note_path) or relative.rsplit("/", 1)[-1].removesuffix(".md").replace("-", " ").title()
        concepts.append({
            "id": raw_id,
            "title": title,
            "path": relative,
            "exists": note_path.is_file(),
        })
    return concepts


def build_synthesis_activity(
    vault_root: str | Path,
    *,
    vault_id: str | None = None,
    nodes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build Curate's vault-scoped session synthesis activity payload."""
    root = Path(vault_root).resolve()
    operations = list_operation_records(root)
    latest_operations: dict[str, dict[str, Any]] = {}
    for operation in operations:
        operation_id = str(operation.get("operation_id") or "")
        if operation_id:
            latest_operations[operation_id] = operation
    operations_by_synthesis: dict[str, list[dict[str, Any]]] = {}
    for operation in latest_operations.values():
        synthesis_id = str(operation.get("synthesis_id") or "")
        if synthesis_id:
            operations_by_synthesis.setdefault(synthesis_id, []).append(operation)

    activities: list[dict[str, Any]] = []
    for entry in read_synthesis_audit(str(root)):
        data = asdict(entry)
        synthesis_id = str(data.get("synthesis_id") or "")
        concept_ids = [str(value) for value in data.get("concept_ids", []) if value]
        activities.append({
            **data,
            "concepts": _concept_details(root, concept_ids, nodes),
            "operations": operations_by_synthesis.get(synthesis_id, []),
            "revertible": any(
                operation.get("status") == "applied"
                for operation in operations_by_synthesis.get(synthesis_id, [])
            ),
        })
    activities.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)
    return {
        "vault_id": vault_id or root.name,
        "activities": activities,
        "count": len(activities),
    }
