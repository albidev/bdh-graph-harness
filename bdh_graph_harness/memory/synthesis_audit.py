"""Session synthesis audit trail for the BDH graph harness.

Persists a compact, query-free audit record for every synthesis outcome
(created, merged, noop, failed/invalid) without retaining the raw
transcript. The audit log is a JSONL file under the vault's .bdh-audit
directory, one entry per synthesis event.

Each entry carries:
  - session_id: originating Hermes session identifier
  - synthesis_id: unique id for this synthesis event
  - transcript_sha256: SHA-256 of the source transcript (integrity, not content)
  - timestamp: ISO-8601 timestamp of the event
  - source: synthesis source label (e.g. "session_synthesis")
  - vault: vault identifier
  - provider / model: configured LLM runtime for this source
  - hebbian_updates: number of Hebbian synapses updated
  - outcome: one of "created", "merged", "noop", "failed"
  - concept_ids: list of concept note IDs produced or merged into
  - reason: optional human-readable reason for noop/failed outcomes
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Literal

AuditOutcome = Literal["created", "merged", "noop", "failed", "invalid"]


@dataclass
class SynthesisAuditEntry:
    """One synthesis audit record."""

    session_id: str
    synthesis_id: str
    transcript_sha256: str
    timestamp: str
    source: str
    vault: str
    provider: str
    model: str
    hebbian_updates: int
    outcome: AuditOutcome
    concept_ids: list[str] = field(default_factory=list)
    reason: str = ""
    queued_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _audit_dir(vault_path: str) -> str:
    return os.path.join(vault_path, ".bdh-audit")


def _audit_path(vault_path: str) -> str:
    return os.path.join(_audit_dir(vault_path), "synthesis.jsonl")


def record_synthesis_audit(
    vault_path: str,
    *,
    session_id: str,
    synthesis_id: str,
    transcript_sha256: str,
    source: str,
    vault: str,
    provider: str,
    model: str,
    hebbian_updates: int,
    outcome: AuditOutcome,
    concept_ids: list[str] | None = None,
    reason: str = "",
    timestamp: str | None = None,
    queued_at: str = "",
) -> SynthesisAuditEntry:
    """Append one audit entry to the vault's synthesis audit log."""
    entry = SynthesisAuditEntry(
        session_id=session_id,
        synthesis_id=synthesis_id,
        transcript_sha256=transcript_sha256,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        source=source,
        vault=vault,
        provider=provider,
        model=model,
        hebbian_updates=hebbian_updates,
        outcome=outcome,
        concept_ids=concept_ids or [],
        reason=reason,
        queued_at=queued_at,
    )
    path = _audit_path(vault_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
    return entry


def read_synthesis_audit(vault_path: str) -> list[SynthesisAuditEntry]:
    """Read all audit entries from the vault's synthesis audit log."""
    path = _audit_path(vault_path)
    entries: list[SynthesisAuditEntry] = []
    if not os.path.isfile(path):
        return entries
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                entries.append(SynthesisAuditEntry(**data))
            except (TypeError, ValueError):
                continue
    return entries


def compute_transcript_sha256(transcript: str) -> str:
    """Return SHA-256 hex digest of a transcript for integrity auditing."""
    return hashlib.sha256(transcript.encode("utf-8")).hexdigest()
