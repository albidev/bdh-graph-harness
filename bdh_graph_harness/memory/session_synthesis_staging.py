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


def _normalized_concept_key(title: str, definition: str = "") -> tuple[str, str]:
    """Whitespace-collapsed, casefolded identity used for candidate dedupe."""
    return (
        re.sub(r"\s+", " ", str(title or "")).strip().casefold(),
        re.sub(r"\s+", " ", str(definition or "")).strip().casefold(),
    )


def _title_tokens(title: str) -> frozenset[str]:
    """Significant tokens of a concept title, for near-duplicate detection."""
    return frozenset(
        token for token in re.findall(r"[a-z0-9]+", str(title or "").casefold())
        if len(token) >= 3
    )


def _open_staged_candidates(
    vault_path: str | os.PathLike[str],
    *,
    statuses: tuple[CandidateStatus, ...] = ("pending_review", "approved"),
) -> list[SessionSynthesisCandidate]:
    """Candidates awaiting a decision — the set a new concept must be deduped against.

    ``list_candidates`` reads every staged file; the caller filters by status here so
    the staging loop only pays for the states that can still absorb a duplicate.
    """
    return [
        candidate
        for candidate in list_candidates(vault_path)
        if candidate.status in statuses
    ]


def _find_staged_duplicate(
    title: str,
    definition: str,
    staged: list[SessionSynthesisCandidate],
) -> SessionSynthesisCandidate | None:
    """Return an already-staged candidate for the same concept, or ``None``.

    The vault is not the only place a concept can already exist: the staging queue
    holds concepts extracted from earlier runs of a session that is STILL OPEN, and
    a live session yields a new ``transcript_sha256`` on every run while the concepts
    it implies stay the same. Vault-only dedupe therefore misses them, and one
    conversation produces a fresh copy of the same idea every time the extractor
    runs (observed: 4 "action beats injection" candidates in 87 minutes, all from one
    live session).

    Matching is on the TITLE, deliberately not the definition. The extractor rephrases
    the definition on every run — measured on the real duplicates above, pairwise token
    overlap between two copies of the same concept ran 0.07-0.47 — so requiring
    definition agreement would match nothing, and thresholding on that overlap would
    merge genuinely distinct concepts. Two rules, both high-confidence:

      1. equal normalized title (the common case);
      2. equal significant-token set, which survives word order and punctuation
         ("skill-graph separation" vs "separation skill graph").

    A definition may legitimately be refined, so the newest definition wins is NOT
    applied here: the existing candidate is kept and the duplicate is dropped, leaving
    a human the single row to review. ``definition`` is accepted for callers that want
    to log the near-miss and for future tightening.
    """
    wanted_title = _normalized_concept_key(title)[0]
    if not wanted_title:
        return None
    for candidate in staged:
        if _normalized_concept_key(candidate.title)[0] == wanted_title:
            return candidate

    wanted_tokens = _title_tokens(title)
    # A single-token title is too weak to call a reworded version of another one.
    if len(wanted_tokens) < 2:
        return None
    for candidate in staged:
        if _title_tokens(candidate.title) == wanted_tokens:
            return candidate
    return None


_CANDIDATE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


def _candidate_path(vault_path: str | os.PathLike[str], candidate_id: str) -> Path:
    """Return a candidate path after rejecting traversal/absolute IDs."""
    candidate_id = str(candidate_id)
    if _CANDIDATE_ID_RE.fullmatch(candidate_id) is None:
        raise ValueError("invalid candidate_id")
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
            "definition": f"{raw} was discussed in the session synthesis response.",
            "confidence": "low",
        })
        if len(concepts) >= 3:
            break
    return concepts


# ---------------------------------------------------------------------------
# Public staging API
# ---------------------------------------------------------------------------


_PLACEHOLDER_SYNTHESIS_DEFINITION = re.compile(
    r"^a durable concept related to .+ extracted from session synthesis\.?$",
    re.IGNORECASE,
)


def _notify_curate_gate(candidate_id: str, vault_id: str) -> None:
    """Fire-and-forget: ask the Curate sidecar to classify this candidate in-flow.

    Best-effort by design: if the sidecar is down, the candidate simply stays
    'not classified yet' and the review proceeds as before. Never raises.
    Timeout is 2s per call; the sidecar itself has its own Jev timeout.
    """
    import threading
    import urllib.request

    url = os.environ.get(
        "CURATE_SIDECAR_URL", "http://127.0.0.1:8775"
    ).rstrip("/")
    token = (os.environ.get("MISSION_CONTROL_TOKEN") or
             os.environ.get("API_SERVER_KEY") or "").strip()
    if not token:
        # LaunchAgent env may not carry the token; fall back to the env file
        env_path = Path(os.path.expanduser("~/.hermes/.env"))
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("MISSION_CONTROL_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    break
    body = json.dumps({"id": candidate_id, "vault": vault_id}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    def _post():
        try:
            request = urllib.request.Request(
                f"{url}/api/local/candidates/classify-single",
                data=body, headers=headers, method="POST",
            )
            urllib.request.urlopen(request, timeout=15).read()
        except Exception:  # noqa: BLE001 — gate is advisory, never blocks staging
            pass

    threading.Thread(target=_post, daemon=True).start()


def _filter_session_synthesis_concepts(concepts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop extractor placeholders and repeated concept loops before Curate."""
    accepted: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    filtered = 0
    for concept in concepts:
        title = str(concept.get("title") or "").strip()
        definition = str(concept.get("definition") or "").strip()
        normalized = (re.sub(r"\s+", " ", title).casefold(), re.sub(r"\s+", " ", definition).casefold())
        if not title or not definition or normalized in seen:
            filtered += 1
            continue
        if _PLACEHOLDER_SYNTHESIS_DEFINITION.fullmatch(definition):
            filtered += 1
            continue
        seen.add(normalized)
        accepted.append(concept)
    return accepted, filtered



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
    source: str = "session_synthesis",
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

    # Candidate files are the durable idempotency ledger. A retry after a
    # successful stage must return the original candidates before invoking the
    # extractor or touching any graph state. Reusing a synthesis id with a
    # different correlation tuple is rejected rather than silently conflated.
    if not dry_run:
        existing = list_candidates(vault_path, synthesis_id=synthesis_id)
        if existing:
            if any(
                candidate.vault_id != vault_id
                or candidate.session_id != session_id
                or candidate.transcript_sha256.casefold() != transcript_sha256.casefold()
                for candidate in existing
            ):
                raise ValueError("synthesis_id already exists with different correlation metadata")
            return {
                "candidates": [candidate.to_dict() for candidate in existing],
                "count": len(existing),
                "dry_run": False,
                "synthesis_id": synthesis_id,
                "idempotent": True,
            }

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
        extractor_config = resolve_llm_config_for_source(cfg, source)

    concepts = _extract_concepts_safely(
        response_text, query, active, nodes, llm_config=extractor_config
    )
    # If the real extractor produced nothing (common in tests/offline), fall back
    # to the deterministic local extractor so staging is always testable.
    if not concepts and not bool(cfg.get("session_synthesis_staging_enabled", False)):
        concepts = _fallback_extract_concepts(response_text)
    filtered_count = 0
    concepts, filtered_count = _filter_session_synthesis_concepts(concepts)
    # Production staging is fail-closed: an extractor failure must not become
    # a fake Curate candidate. The offline fallback remains available for the
    # legacy unit-test config where staging is explicitly disabled.
    if not concepts and filtered_count == 0 and not dry_run and not bool(cfg.get("session_synthesis_staging_enabled", False)):
        concepts = [{
            'title': 'Unclassified durable concept',
            'definition': 'No durable concept was confidently extracted; requires human review.',
            'confidence': 'low',
        }]
    concepts = concepts[:max(0, int(max_concepts))]

    # Concepts already awaiting a decision are part of the "does this already
    # exist" answer. Reading them once here keeps the check inside the loop from
    # rescanning the ledger per concept, and lets a skipped concept be reported
    # instead of silently vanishing.
    staged_open = _open_staged_candidates(vault_path) if not dry_run else []

    candidates: list[SessionSynthesisCandidate] = []
    duplicates = 0
    for concept in concepts:
        title = str(concept.get("title", "")).strip()
        definition = str(concept.get("definition", "")).strip()
        if not title or not definition:
            continue

        # The same concept staged by an earlier run of a still-open session must not
        # be staged again: the transcript hash changes every run, so the synthesis-id
        # idempotency above cannot catch it.
        existing_staged = _find_staged_duplicate(title, definition, staged_open)
        if existing_staged is not None:
            duplicates += 1
            logger.debug(
                "session synthesis: concept %r already staged as %s (%s)",
                title, existing_staged.candidate_id, existing_staged.status,
            )
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
            source=source,
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
            _notify_curate_gate(candidate_id, vault_id)
            create_curate_candidate(
                str(vault_path),
                candidate_id=candidate_id,
                synthesis_id=synthesis_id,
                vault_id=vault_id,
                session_id=session_id,
                transcript_sha256=transcript_sha256,
                extra={
                    "title": title,
                    "source": source,
                    "status": "pending_review",
                },
            )

        candidates.append(candidate)

    return {
        "candidates": [c.to_dict() for c in candidates],
        "count": len(candidates),
        "dry_run": dry_run,
        "synthesis_id": synthesis_id,
        "idempotent": False,
        "filtered_count": filtered_count,
        "duplicate_count": duplicates,
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
        source=source,
        dry_run=dry_run,
        config=config,
        provenance=provenance,
    )
