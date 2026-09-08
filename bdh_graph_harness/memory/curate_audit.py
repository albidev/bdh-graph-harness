"""Curate audit state machine for session_synthesis candidates.

Persists a durable, queryable audit trail for each candidate produced by
session_synthesis.  The state machine links every candidate to its
provenance (synthesis_id, vault_id, session_id, transcript_sha256) and
tracks its lifecycle through Curate review and BDH application.

Audit log: ``.bdh-audit/curate.jsonl`` (one JSON object per state change).

States
------
- pending_review: candidate exists and is waiting for approval
- rejected:      candidate was explicitly rejected, no vault mutation
- created:       candidate applied and created a new vault note
- merged:        candidate applied and merged evidence into an existing note
- noop:          candidate produced no durable vault change
- failed:        candidate application failed after approval/retry
- reverted:      a previous created/merged application was rolled back
- conflict:      revert or apply encountered an unresolvable conflict
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

CurateState = Literal[
    "pending_review",
    "rejected",
    "created",
    "merged",
    "noop",
    "failed",
    "reverted",
    "conflict",
]

# Terminal states: once reached, the candidate audit record must not move again.
_FINAL_STATES: frozenset[str] = frozenset(
    {"rejected", "noop", "failed", "reverted", "conflict"}
)

# Valid transitions.  Anything not listed here is rejected so the log stays
# a faithful, append-only history of the candidate lifecycle.
_VALID_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending_review": frozenset({"rejected", "created", "merged", "noop", "failed", "conflict"}),
    "created": frozenset({"reverted", "conflict"}),
    "merged": frozenset({"reverted", "conflict"}),
}


@dataclass
class CurateAuditEntry:
    """One Curate audit state record."""

    candidate_id: str
    synthesis_id: str
    vault_id: str
    session_id: str
    transcript_sha256: str
    state: CurateState
    timestamp: str
    reason: str = ""
    note_path: str = ""
    operation_id: str = ""
    applied_by: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def correlation(self) -> dict[str, str]:
        """Return the canonical correlation tuple for this candidate."""
        return {
            "candidate_id": self.candidate_id,
            "synthesis_id": self.synthesis_id,
            "vault_id": self.vault_id,
            "session_id": self.session_id,
            "transcript_sha256": self.transcript_sha256,
        }


def _audit_dir(vault_path: str) -> str:
    return os.path.join(vault_path, ".bdh-audit")


def _audit_path(vault_path: str) -> str:
    return os.path.join(_audit_dir(vault_path), "curate.jsonl")


def _append_line(path: str, record: dict[str, Any]) -> None:
    """Append one JSON line and fsync for durability."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_curate_audit(vault_path: str) -> list[CurateAuditEntry]:
    """Read all Curate audit entries from newest to oldest."""
    path = _audit_path(vault_path)
    entries: list[CurateAuditEntry] = []
    if not os.path.isfile(path):
        return entries
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                entries.insert(0, CurateAuditEntry(**data))
            except (TypeError, ValueError):
                continue
    return entries


def latest_curate_state(vault_path: str, candidate_id: str) -> CurateAuditEntry | None:
    """Return the most recent audit entry for ``candidate_id``, or None."""
    for entry in read_curate_audit(vault_path):
        if entry.candidate_id == candidate_id:
            return entry
    return None


def _validate_transition(current_state: str | None, new_state: str) -> str | None:
    """Return an error string if the transition is illegal, else None."""
    if current_state is None:
        if new_state != "pending_review":
            return f"first state must be pending_review, got {new_state}"
        return None
    if current_state == new_state:
        return None  # idempotent repeat
    if current_state in _FINAL_STATES:
        return f"cannot transition from terminal state {current_state} to {new_state}"
    allowed = _VALID_TRANSITIONS.get(current_state, frozenset())
    if new_state not in allowed:
        return f"illegal transition from {current_state} to {new_state}"
    return None


def record_curate_audit(
    vault_path: str,
    *,
    candidate_id: str,
    synthesis_id: str,
    vault_id: str,
    session_id: str,
    transcript_sha256: str,
    state: CurateState,
    timestamp: str | None = None,
    reason: str = "",
    note_path: str = "",
    operation_id: str = "",
    applied_by: str = "",
    extra: dict[str, Any] | None = None,
    validate: bool = True,
) -> CurateAuditEntry:
    """Append a Curate audit entry after validating the state transition.

    Setting ``validate=False`` bypasses the state-machine guard; use only for
    recovery/force paths that know what they are doing.
    """
    current = latest_curate_state(vault_path, candidate_id)
    current_state = current.state if current else None

    if validate:
        error = _validate_transition(current_state, state)
        if error:
            raise ValueError(error)

    # Idempotent repeat: do not add a duplicate log line for the same state.
    if current_state == state:
        return current  # type: ignore[return-value]

    entry = CurateAuditEntry(
        candidate_id=candidate_id,
        synthesis_id=synthesis_id,
        vault_id=vault_id,
        session_id=session_id,
        transcript_sha256=transcript_sha256,
        state=state,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        reason=reason,
        note_path=note_path,
        operation_id=operation_id,
        applied_by=applied_by,
        extra=extra or {},
    )
    _append_line(_audit_path(vault_path), entry.to_dict())
    return entry


def transition_curate_state(
    vault_path: str,
    candidate_id: str,
    new_state: CurateState,
    *,
    reason: str = "",
    note_path: str = "",
    operation_id: str = "",
    applied_by: str = "",
    extra: dict[str, Any] | None = None,
) -> CurateAuditEntry:
    """Move ``candidate_id`` to ``new_state`` if the transition is legal.

    Raises ``ValueError`` for illegal transitions.  Terminal states are
    sticky; repeated calls with the same state are idempotent.
    """
    current = latest_curate_state(vault_path, candidate_id)
    if current is None:
        raise ValueError(f"candidate {candidate_id} not found")
    merged_extra = dict(current.extra) if current.extra else {}
    if extra:
        merged_extra.update(extra)
    return record_curate_audit(
        vault_path,
        candidate_id=candidate_id,
        synthesis_id=current.synthesis_id,
        vault_id=current.vault_id,
        session_id=current.session_id,
        transcript_sha256=current.transcript_sha256,
        state=new_state,
        reason=reason,
        note_path=note_path,
        operation_id=operation_id,
        applied_by=applied_by,
        extra=merged_extra,
    )


def create_curate_candidate(
    vault_path: str,
    *,
    candidate_id: str,
    synthesis_id: str,
    vault_id: str,
    session_id: str,
    transcript_sha256: str,
    extra: dict[str, Any] | None = None,
) -> CurateAuditEntry:
    """Create a new Curate candidate in ``pending_review``.

    Idempotent: if ``candidate_id`` already exists with any state, the
    existing latest entry is returned and no duplicate pending_review line
    is written.
    """
    existing = latest_curate_state(vault_path, candidate_id)
    if existing is not None:
        return existing
    return record_curate_audit(
        vault_path,
        candidate_id=candidate_id,
        synthesis_id=synthesis_id,
        vault_id=vault_id,
        session_id=session_id,
        transcript_sha256=transcript_sha256,
        state="pending_review",
        extra=extra,
    )


def get_curate_correlation(vault_path: str, candidate_id: str) -> dict[str, str] | None:
    """Return the correlation tuple for ``candidate_id`` if it exists."""
    entry = latest_curate_state(vault_path, candidate_id)
    return entry.correlation() if entry else None
