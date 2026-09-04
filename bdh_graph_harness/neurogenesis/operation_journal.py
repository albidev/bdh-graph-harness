"""Reversible, vault-scoped journal for neurogenesis mutations."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


JOURNAL_NAME = "neurogenesis-operations.jsonl"


def _audit_dir(vault_root: str | os.PathLike[str]) -> Path:
    return Path(vault_root) / ".bdh-audit"


def _journal_path(vault_root: str | os.PathLike[str]) -> Path:
    return _audit_dir(vault_root) / JOURNAL_NAME


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _relative_note_path(vault_root: Path, note_path: Path) -> str:
    root = vault_root.resolve()
    note = note_path.resolve()
    try:
        relative = note.relative_to(root)
    except ValueError as exc:
        raise ValueError("neurogenesis note must be inside its vault") from exc
    return relative.as_posix()


def _append_record(vault_root: Path, record: Mapping[str, Any]) -> None:
    path = _journal_path(vault_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def prepare_operation(
    vault_root: str | os.PathLike[str],
    *,
    synthesis_meta: Mapping[str, Any],
    action: str,
    note_path: str | os.PathLike[str],
    before_content: str | None = None,
) -> dict[str, Any]:
    """Persist the inverse data before a neurogenesis mutation starts."""
    if action not in {"created", "merged"}:
        raise ValueError("operation action must be created or merged")
    root = Path(vault_root).resolve()
    note = Path(note_path).resolve()
    relative_note = _relative_note_path(root, note)
    if action == "merged" and before_content is None:
        raise ValueError("merged operation requires before_content")
    operation_id = str(uuid.uuid4())
    snapshot_path = ""
    if before_content is not None:
        snapshot = _audit_dir(root) / "snapshots" / f"{operation_id}.md"
        _atomic_text(snapshot, before_content)
        snapshot_path = str(snapshot.relative_to(root))
    record = {
        "operation_id": operation_id,
        "synthesis_id": str(synthesis_meta.get("synthesis_id") or ""),
        "session_id": str(synthesis_meta.get("session_id") or ""),
        "transcript_sha256": str(synthesis_meta.get("transcript_sha256") or ""),
        "action": action,
        "status": "prepared",
        "note_path": relative_note,
        "before_hash": _sha256(before_content) if before_content is not None else "",
        "after_hash": "",
        "snapshot_path": snapshot_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _append_record(root, record)
    return record


def complete_operation(
    vault_root: str | os.PathLike[str],
    prepared: Mapping[str, Any],
    *,
    note_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Close a prepared operation after the mutated note is durably written."""
    root = Path(vault_root).resolve()
    note = Path(note_path).resolve()
    relative_note = _relative_note_path(root, note)
    if relative_note != str(prepared.get("note_path") or "") or not note.is_file():
        raise ValueError("prepared operation note does not match the completed mutation")
    record = {
        **dict(prepared),
        "status": "applied",
        "after_hash": _sha256(note.read_text(encoding="utf-8")),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_record(root, record)
    return record


def record_applied_operation(
    vault_root: str | os.PathLike[str],
    *,
    synthesis_meta: Mapping[str, Any],
    action: str,
    note_path: str | os.PathLike[str],
    before_content: str | None = None,
) -> dict[str, Any]:
    """Record a successfully applied create/merge with a reversible snapshot."""
    if action not in {"created", "merged"}:
        raise ValueError("operation action must be created or merged")
    root = Path(vault_root).resolve()
    note = Path(note_path).resolve()
    if not note.is_file():
        raise ValueError(f"neurogenesis note does not exist: {note}")
    if action == "merged" and before_content is None:
        raise ValueError("merged operation requires before_content")
    operation_id = str(uuid.uuid4())
    relative_note = _relative_note_path(root, note)
    snapshot_path = ""
    if before_content is not None:
        snapshot = _audit_dir(root) / "snapshots" / f"{operation_id}.md"
        _atomic_text(snapshot, before_content)
        snapshot_path = str(snapshot.relative_to(root))
    after_content = note.read_text(encoding="utf-8")
    record = {
        "operation_id": operation_id,
        "synthesis_id": str(synthesis_meta.get("synthesis_id") or ""),
        "session_id": str(synthesis_meta.get("session_id") or ""),
        "transcript_sha256": str(synthesis_meta.get("transcript_sha256") or ""),
        "action": action,
        "status": "applied",
        "note_path": relative_note,
        "before_hash": _sha256(before_content) if before_content is not None else "",
        "after_hash": _sha256(after_content),
        "snapshot_path": snapshot_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _append_record(root, record)
    return record


def list_operation_records(vault_root: str | os.PathLike[str]) -> list[dict[str, Any]]:
    path = _journal_path(vault_root)
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("operation_id"):
            records.append(value)
    return records


def _latest_operation(vault_root: Path, operation_id: str) -> dict[str, Any] | None:
    records = [record for record in list_operation_records(vault_root) if record.get("operation_id") == operation_id]
    return records[-1] if records else None


def _remove_index_entry(root: Path, note_path: str) -> bool:
    if not note_path.endswith(".md"):
        return False
    index = root / "wiki" / "index.md"
    if not index.is_file():
        return False
    note_id = note_path[:-3]
    needle = f"[[{note_id}|"
    content = index.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)
    filtered = [line for line in lines if needle not in line]
    if len(filtered) == len(lines):
        return False
    _atomic_text(index, "".join(filtered))
    return True


def revert_operation(
    vault_root: str | os.PathLike[str],
    operation_id: str,
) -> dict[str, Any]:
    """Apply the inverse operation only when the note has not changed since it."""
    root = Path(vault_root).resolve()
    record = _latest_operation(root, operation_id)
    if record is None:
        return {"status": "not_found", "operation_id": operation_id}
    if record.get("status") == "reverted":
        return {**record, "status": "already_reverted"}
    if record.get("status") == "conflict":
        return {**record, "status": "conflict"}
    note = (root / str(record.get("note_path") or "")).resolve()
    try:
        _relative_note_path(root, note)
    except ValueError:
        return {**record, "status": "conflict", "reason": "invalid_note_path"}
    if not note.is_file():
        result = {**record, "status": "conflict", "reason": "note_missing"}
        _append_record(root, result)
        return result
    current_hash = _sha256(note.read_text(encoding="utf-8"))
    if current_hash != record.get("after_hash"):
        result = {**record, "status": "conflict", "reason": "note_changed_after_operation", "current_hash": current_hash}
        _append_record(root, result)
        return result

    result: dict[str, Any] = {**record, "status": "reverted", "reverted_at": datetime.now(timezone.utc).isoformat()}
    if record.get("action") == "created":
        archive = _audit_dir(root) / "reverted" / f"{operation_id}--{note.name}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(note, archive)
        result["archive_path"] = str(archive)
        result["index_entry_removed"] = _remove_index_entry(root, str(record.get("note_path") or ""))
    elif record.get("action") == "merged":
        snapshot_path = (root / str(record.get("snapshot_path") or "")).resolve()
        try:
            _relative_note_path(root, snapshot_path)
        except ValueError:
            return {**record, "status": "conflict", "reason": "invalid_snapshot_path"}
        if not snapshot_path.is_file():
            result = {**record, "status": "conflict", "reason": "snapshot_missing"}
            _append_record(root, result)
            return result
        _atomic_text(note, snapshot_path.read_text(encoding="utf-8"))
    else:
        result = {**record, "status": "conflict", "reason": "unknown_action"}
    _append_record(root, result)
    return result
