"""Semantic sleep helpers: source selection and per-vault idempotent checkpoints."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from bdh_graph_harness.config import CONFIG, logger

DEFAULT_SOURCE_GLOBS = (
    "memory/daily/*.md",
    "memory/learned/*.md",
    "wiki/queries/*.md",
    "wiki/entities/*.md",
    "wiki/lessons/*.md",
)
DEFAULT_EXCLUDE_GLOBS = (
    "wiki/index.md",
    "wiki/log.md",
    "wiki/raw/*",
    "wiki/concepts/*",
    ".bdh-*",
)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _checkpoint_path(vault_root: Path, config: dict[str, Any]) -> Path:
    configured = os.path.expanduser(
        str(config.get("semantic_consolidation_checkpoint", ".bdh-semantic-consolidation.json"))
    )
    path = Path(configured)
    return path if path.is_absolute() else vault_root / path


def compute_content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(vault_root: str | os.PathLike[str], config: dict[str, Any] | None = None) -> dict:
    """Load a checkpoint; a missing/corrupt checkpoint starts a fresh cycle."""
    root = Path(vault_root).expanduser().resolve()
    path = _checkpoint_path(root, config or CONFIG)
    if not path.is_file():
        return {"version": 1, "last_run_at": None, "processed": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("processed", {}), dict):
            raise ValueError("invalid checkpoint shape")
        data.setdefault("version", 1)
        data.setdefault("last_run_at", None)
        return data
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Semantic consolidation checkpoint unreadable at %s: %s", path, exc)
        return {"version": 1, "last_run_at": None, "processed": {}}


def save_checkpoint_atomic(
    vault_root: str | os.PathLike[str],
    checkpoint: dict,
    config: dict[str, Any] | None = None,
) -> Path:
    """Persist a checkpoint with same-directory temp file + atomic replace."""
    root = Path(vault_root).expanduser().resolve()
    path = _checkpoint_path(root, config or CONFIG)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(checkpoint)
    payload["version"] = 1
    payload["last_run_at"] = datetime.now().isoformat()
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return path


def _matches_any(relative_path: str, patterns: list[str] | tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(relative_path, pattern) for pattern in patterns)


def select_candidate_notes(
    vault_root: str | os.PathLike[str],
    config: dict[str, Any] | None = None,
    checkpoint: dict | None = None,
    *,
    max_sources: int | None = None,
) -> list[dict[str, Any]]:
    """Return changed markdown sources in deterministic oldest-first order."""
    cfg = config or CONFIG
    root = Path(vault_root).expanduser().resolve()
    state = checkpoint or load_checkpoint(root, cfg)
    processed = state.get("processed", {})
    includes = tuple(cfg.get("semantic_consolidation_source_globs", DEFAULT_SOURCE_GLOBS))
    excludes = tuple(cfg.get("semantic_consolidation_exclude_globs", DEFAULT_EXCLUDE_GLOBS))
    max_chars = int(cfg.get("semantic_consolidation_max_source_chars", 8000))
    max_age_hours = float(cfg.get("semantic_consolidation_max_age_hours", 48))
    cutoff = datetime.now().timestamp() - max_age_hours * 3600
    limit = max_sources if max_sources is not None else int(
        cfg.get("semantic_consolidation_max_sources", 3)
    )

    candidates: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for pattern in includes:
        for path in root.glob(pattern):
            if path in seen or not path.is_file() or path.suffix.lower() != ".md":
                continue
            seen.add(path)
            relative = _relative(path, root)
            if _matches_any(relative, excludes):
                continue
            try:
                stat = path.stat()
                if stat.st_mtime < cutoff:
                    continue
                content = path.read_text(encoding="utf-8")
                content_hash = compute_content_hash(path)
                previous = processed.get(relative, {})
                if previous.get("sha256") == content_hash:
                    continue
                candidates.append({
                    "path": relative,
                    "absolute_path": str(path),
                    "sha256": content_hash,
                    "content": content[:max_chars],
                    "mtime_ns": path.stat().st_mtime_ns,
                })
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("Unable to inspect semantic source %s: %s", path, exc)

    candidates.sort(key=lambda item: (item["mtime_ns"], item["path"]))
    return candidates[:max(0, limit)]


def mark_processed(checkpoint: dict, source: dict[str, Any], result: dict[str, Any]) -> dict:
    """Return a checkpoint copy with one successfully processed source recorded."""
    updated = dict(checkpoint)
    updated["processed"] = dict(checkpoint.get("processed", {}))
    updated["processed"][source["path"]] = {
        "sha256": source["sha256"],
        "processed_at": datetime.now().isoformat(),
        "new_concepts": result.get("new_concepts", []),
        "hebbian_updates": result.get("hebbian_updates", 0),
    }
    return updated
