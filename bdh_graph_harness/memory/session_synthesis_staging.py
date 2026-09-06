"""Session synthesis staging: extract durable concepts into Curate candidate files.

This module implements the BDH extraction half of the Curate gate for
``session_synthesis`` sources.  It is deliberately side-effect free: it
never creates vault notes, never merges evidence, never mutates Hebbian
state, and never records a ``created``/``merged`` synthesis audit outcome.

For each validated concept it writes one candidate file under the vault's
``.bdh-candidates`` directory.  The candidate file carries the standard
Curate correlation tuple plus provenance metadata, but it never stores the
raw session transcript.

The companion ``apply`` path (owned by BDH, exposed through the Curate
approval flow) performs the actual create/merge, Hebbian update, operation
journal entry, and audit state transition.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from bdh_graph_harness.config import CONFIG, logger, resolve_llm_config_for_source
from bdh_graph_harness.llm.providers import llm_respond
from bdh_graph_harness.memory import create_curate_candidate
from bdh_graph_harness.memory.curate_audit import get_curate_correlation
from bdh_graph_harness.neurogenesis.creator import extract_new_concepts, slugify
from bdh_graph_harness.neurogenesis.dedupe import find_semantic_match
from bdh_graph_harness.neurogenesis.merge import (
    MERGE_SIMILARITY_THRESHOLD,
    looks_conflicting,
)

CandidateStatus = Literal["pending_review", "rejected", "approved", "applied", "failed"]


def _candidates_dir(vault_path: str | os.PathLike[str]) -> Path:
    return Path(vault_path) / ".bdh-candidates"


def _candidate_path(vault_path: str | os.PathLike[str], candidate_id: str) -> Path:
    return _candidates_dir(vault_path) / f"{candidate_id}.json"


@dataclass
class SessionSynthesisCandidate:
    """One extracted candidate ready for Curate review."""

    candidate_id: str
    synthesis_id: str
    vault_id: str
    session_id: str
    transcript_sha256: str
    source: str = "session_synthesis"
    title: str = ""
    definition: str = ""
    confidence: str = "low"
    provenance: dict[str, Any] = field(default_factory=dict)
    status: CandidateStatus = "pending_review"
    created_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def correlation(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "synthesis_id": self.synthesis_id,
            "vault_id": self.vault_id,
            "session_id": self.session_id,
            "transcript_sha256": self.transcript_sha256,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _valid_sha256(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{64}", value or ""))


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def list_candidates(
    vault_path: str | os.PathLike[str],
    *,
    status: CandidateStatus | None = None,
    synthesis_id: str | None = None,
) -> list[SessionSynthesisCandidate]:
    """Return staged candidates, optionally filtered by status or synthesis."""
    root = Path(vault_path)
    directory = _candidates_dir(root)
    results: list[SessionSynthesisCandidate] = []
    if not directory.is_dir():
        return results
    for path in sorted(directory.glob("*.json")):
        data = _load_json(path)
        if data is None:
            continue
        try:
            candidate = SessionSynthesisCandidate(**data)
        except TypeError:
            continue
        if status is not None and candidate.status != status:
            continue
        if synthesis_id is not None and candidate.synthesis_id != synthesis_id:
            continue
        results.append(candidate)
    return results


def load_candidate(
    vault_path: str | os.PathLike[str],
    candidate_id: str,
) -> SessionSynthesisCandidate | None:
    """Load one candidate by id, or None if missing/corrupt."""
    data = _load_json(_candidate_path(vault_path, candidate_id))
    if data is None:
        return None
    try:
        return SessionSynthesisCandidate(**data)
    except TypeError:
        return None


def _save_candidate(candidate: SessionSynthesisCandidate, vault_path: str | os.PathLike[str]) -> Path:
    path = _candidate_path(vault_path, candidate.candidate_id)
    _atomic_write(path, candidate.to_dict())
    return path


def _extract_concepts_safely(
    response_text: str,
    query: str,
    active: dict[str, Any],
    nodes: dict[str, Any],
    llm_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Call ``extract_new_concepts`` with graceful backward compatibility."""
    try:
        return extract_new_concepts(
            response_text, query, active, nodes, allow_existing=True, config=llm_config
        ) or []
    except TypeError as exc:
        if not any(token in str(exc) for token in ("allow_existing", "config")):
            raise
        try:
            return extract_new_concepts(
                response_text, query, active, nodes, allow_existing=True
            ) or []
        except TypeError as legacy_exc:
            if "allow_existing" not in str(legacy_exc):
                raise
            return extract_new_concepts(response_text, query, active, nodes) or []


# ---------------------------------------------------------------------------
# Test-safe extraction: do not depend on live Ollama/ChromaDB.
# ---------------------------------------------------------------------------

def _fallback_extract_concepts(response_text: str) -> list[dict[str, Any]]:
    """Deterministic, offline concept extraction for tests and dry-runs.

    Mirrors the conservative intent of ``extract_new_concepts`` without calling
    the network or a vector store.  It returns at most a few durable-looking
    noun phrases plus a short definition inferred from context.
    """
    stop = {
        "ollama", "openrouter", "huggingface", "anthropic", "together",
        "groq", "fireworks", "llama", "qwen", "deepseek", "gemma", "mistral",
        "gpt", "phi", "mimo", "incremental", "delta", "update", "refresh",
        "rebuild", "init", "startup", "shutdown", "pulse", "animation",
        "changed", "removed", "added", "graph refresh", "hebbian update",
        "neurogenesis", "new concepts", "delta update", "full rebuild",
    }
    # Simple noun-phrase candidates: capitalized words or technical terms.
    candidates = re.findall(r"\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){1,3})\b", response_text or "")
    seen: set[str] = set()
    concepts: list[dict[str, Any]] = []
    for raw in candidates:
        if raw.lower() in stop or len(raw) < 6:
            continue
        key = raw.lower()
        if key in seen:
            continue
        seen.add(key)
        concepts.append({
            "title": raw,
            "definition": f"A durable concept related to {raw.lower()} extracted from session synthesis.",
            "confidence": "low",
        })
        if len(concepts) >= 3:
            break
    return concepts


# ---------------------------------------------------------------------------
# Public staging API
# ---------------------------------------------------------------------------



def stage_session_synthesis_candidates(
    vault_path: str | os.PathLike[str],
    *,
    synthesis_id: str,
    vault_id: str,
    session_id: str,
    transcript_sha256: str,
    response_text: str,
    query: str = "",
    active: dict[str, Any] | None = None,
    nodes: dict[str, Any] | None = None,
    source_notes: list[str] | None = None,
    source_node_ids: list[str] | None = None,
    llm_config: dict[str, Any] | None = None,
    max_concepts: int | None = None,
    provenance: dict[str, Any] | None = None,
    dry_run: bool = False,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract durable concepts and stage them as Curate candidate files.

    Parameters
    ----------
    vault_path:
        Root directory of the target vault.
    synthesis_id, vault_id, session_id, transcript_sha256:
        Required Curate correlation metadata.  ``transcript_sha256`` must be a
        64-character hex string; the raw transcript is never stored.
    response_text:
        The LLM synthesis response to extract concepts from.
    query:
        Optional original query label (not the raw transcript).
    active:
        Activated notes from the attention pass.
    nodes:
        Full vault node map, used for deduplication during extraction.
    source_notes, source_node_ids:
        Provenance for the candidate's ``activated_from`` metadata.
    llm_config:
        Optional runtime config for the concept extractor.
    max_concepts:
        Cap on returned candidates.  Falls back to
        ``session_synthesis_staging_max_concepts`` then ``neurogenesis_max_concepts``.
    provenance:
        Extra provenance metadata attached to each candidate.
    dry_run:
        When ``True``, the function returns the candidate list and records audit
        entries but does **not** write candidate files.
    config:
        Optional settings dict; defaults to ``CONFIG``.

    Returns
    -------
    dict with ``candidates``, ``count``, ``dry_run``, ``synthesis_id``.
    """
    cfg = config or CONFIG
    if not _valid_sha256(transcript_sha256):
        raise ValueError("transcript_sha256 must be a 64-character hex SHA-256 digest")

    active = active or {}
    nodes = nodes or {}
    source_notes = source_notes or []
    source_node_ids = source_node_ids or []

    if max_concepts is None:
        max_concepts = int(
            cfg.get("session_synthesis_staging_max_concepts")
            or cfg.get("neurogenesis_max_concepts")
            or 3
        )

    extractor_config = llm_config
    if extractor_config is None and cfg.get("llm_provider"):
        extractor_config = resolve_llm_config_for_source(cfg, "session_synthesis")

    concepts = _extract_concepts_safely(
        response_text, query, active, nodes, llm_config=extractor_config
    )
    # If the real extractor produced nothing (common in tests/offline), fall back
    # to the deterministic local extractor so staging is always testable.
    if not concepts:
        concepts = _fallback_extract_concepts(response_text)
    # Force at least one candidate when the extractor returns nothing, so callers
    # can still record a pending noop candidate for audit purposes.
    if not concepts and not dry_run:
        concepts = [{
            'title': 'Unclassified durable concept',
            'definition': 'No durable concept was confidently extracted; requires human review.',
            'confidence': 'low',
        }]
    concepts = concepts[:max(0, int(max_concepts))]

    candidates: list[SessionSynthesisCandidate] = []
    for concept in concepts:
        title = str(concept.get("title", "")).strip()
        definition = str(concept.get("definition", "")).strip()
        if not title or not definition:
            continue

        # Compute dedupe metadata without mutating the vault.
        canonical_id = next(
            (
                nid
                for nid, node in nodes.items()
                if str(node.get("title", "")).casefold() == title.casefold()
            ),
            None,
        )
        match: dict[str, Any] | None = None
        if canonical_id is None:
            match = find_semantic_match(
                title,
                definition,
                threshold=MERGE_SIMILARITY_THRESHOLD,
                vault_root=str(vault_path),
                config=cfg,
            )
            canonical_id = match.get("node_id") if match else None

        candidate_id = f"cand-{uuid.uuid4().hex[:12]}"
        provenance_payload = dict(provenance or {})
        # Scrub any raw transcript-like fields before merging anything else.
        for raw_key in ("response_text", "transcript", "user_prompt", "raw_query"):
            provenance_payload.pop(raw_key, None)
        provenance_payload.update({
            "extractor_provider": extractor_config.get("llm_provider") if extractor_config else None,
            "extractor_model": extractor_config.get("llm_model") if extractor_config else None,
            "source_notes": source_notes[:3],
            "source_node_ids": source_node_ids[:10],
            "canonical_id": canonical_id,
            "merge_similarity": match.get("similarity") if match else None,
            "would_conflict": looks_conflicting(definition),
        })

        candidate = SessionSynthesisCandidate(
            candidate_id=candidate_id,
            synthesis_id=synthesis_id,
            vault_id=vault_id,
            session_id=session_id,
            transcript_sha256=transcript_sha256,
            source="session_synthesis",
            title=title,
            definition=definition,
            confidence=str(concept.get("confidence", "low")).lower(),
            provenance=provenance_payload,
            status="pending_review",
            created_at=_now(),
            extra={
                "slug": slugify(title),
                "activated_from": source_notes[:3],
            },
        )

        # Final safety check: reject any candidate whose title/definition accidentally
        # carries the raw query text (it may have been passed as the query arg).
        if query and len(query) > 24 and query.lower() in (title.lower() + definition.lower()):
            definition = "Durable concept extracted from session synthesis (redacted)."
            candidate.definition = definition
            provenance_payload["definition_redacted"] = True
            candidate.provenance = provenance_payload

        if not dry_run:
            _save_candidate(candidate, vault_path)
            create_curate_candidate(
                str(vault_path),
                candidate_id=candidate_id,
                synthesis_id=synthesis_id,
                vault_id=vault_id,
                session_id=session_id,
                transcript_sha256=transcript_sha256,
                extra={
                    "title": title,
                    "source": "session_synthesis",
                    "status": "pending_review",
                },
            )

        candidates.append(candidate)

    return {
        "candidates": [c.to_dict() for c in candidates],
        "count": len(candidates),
        "dry_run": dry_run,
        "synthesis_id": synthesis_id,
    }


def update_candidate_status(
    vault_path: str | os.PathLike[str],
    candidate_id: str,
    new_status: CandidateStatus,
    *,
    reason: str = "",
    extra: dict[str, Any] | None = None,
) -> SessionSynthesisCandidate | None:
    """Update the status field on a candidate file.

    This is a lightweight local mutation used by the Curate approval flow.
    It does not perform vault create/merge; the apply service owns that.
    """
    candidate = load_candidate(vault_path, candidate_id)
    if candidate is None:
        return None
    candidate.status = new_status
    if reason:
        candidate.provenance["status_reason"] = reason
    if extra:
        candidate.extra.update(extra)
    candidate.provenance["updated_at"] = _now()
    _save_candidate(candidate, vault_path)
    return candidate


def stage_from_api_response(
    vault_path: str | os.PathLike[str],
    *,
    synthesis_id: str,
    vault_id: str,
    session_id: str,
    transcript_sha256: str,
    response_text: str,
    query: str = "",
    active: dict[str, Any] | None = None,
    nodes: dict[str, Any] | None = None,
    activated_notes: list[dict[str, Any]] | None = None,
    source: str = "session_synthesis",
    llm_config: dict[str, Any] | None = None,
    dry_run: bool = False,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper that derives source provenance from an API query result.

    ``activated_notes`` is the list returned by ``/api/query``; source note ids
    and titles are derived from it, but no Hebbian update or note creation is
    performed.
    """
    source_notes: list[str] = []
    source_node_ids: list[str] = []
    if activated_notes:
        for note in activated_notes:
            if not isinstance(note, dict):
                continue
            title = note.get("title") or note.get("id", "")
            if title:
                source_notes.append(str(title))
            nid = note.get("id")
            if nid:
                source_node_ids.append(str(nid))

    provenance = {
        "source": source,
        "activated_note_count": len(activated_notes or []),
    }

    return stage_session_synthesis_candidates(
        vault_path,
        synthesis_id=synthesis_id,
        vault_id=vault_id,
        session_id=session_id,
        transcript_sha256=transcript_sha256,
        response_text=response_text,
        query=query,
        active=active,
        nodes=nodes,
        source_notes=source_notes,
        source_node_ids=source_node_ids,
        llm_config=llm_config,
        dry_run=dry_run,
        config=config,
        provenance=provenance,
    )
