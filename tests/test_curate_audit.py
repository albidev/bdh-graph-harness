"""Tests for the Curate audit state machine and correlation logic."""
import json
import os
from pathlib import Path

import pytest

from bdh_graph_harness.memory import curate_audit as ca


SHA = "a" * 64


class TestCurateAuditBasics:
    def test_create_candidate(self, tmp_path):
        entry = ca.create_curate_candidate(
            str(tmp_path),
            candidate_id="c-1",
            synthesis_id="syn-1",
            vault_id="v-1",
            session_id="sess-1",
            transcript_sha256=SHA,
        )
        assert entry.state == "pending_review"
        assert entry.candidate_id == "c-1"

    def test_create_candidate_writes_jsonl(self, tmp_path):
        ca.create_curate_candidate(
            str(tmp_path),
            candidate_id="c-1",
            synthesis_id="syn-1",
            vault_id="v-1",
            session_id="sess-1",
            transcript_sha256=SHA,
        )
        path = tmp_path / ".bdh-audit" / "curate.jsonl"
        assert path.is_file()
        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert data["candidate_id"] == "c-1"
        assert data["state"] == "pending_review"

    def test_correlation(self, tmp_path):
        entry = ca.create_curate_candidate(
            str(tmp_path),
            candidate_id="c-1",
            synthesis_id="syn-1",
            vault_id="v-1",
            session_id="sess-1",
            transcript_sha256=SHA,
        )
        assert entry.correlation() == {
            "candidate_id": "c-1",
            "synthesis_id": "syn-1",
            "vault_id": "v-1",
            "session_id": "sess-1",
            "transcript_sha256": SHA,
        }

    def test_get_correlation_missing(self, tmp_path):
        assert ca.get_curate_correlation(str(tmp_path), "nope") is None


class TestCurateStateTransitions:
    def test_pending_to_created(self, tmp_path):
        ca.create_curate_candidate(
            str(tmp_path),
            candidate_id="c-1",
            synthesis_id="syn-1",
            vault_id="v-1",
            session_id="sess-1",
            transcript_sha256=SHA,
        )
        entry = ca.transition_curate_state(
            str(tmp_path), "c-1", "created",
            note_path="wiki/concepts/c1.md",
            operation_id="op-1",
        )
        assert entry.state == "created"
        assert entry.note_path == "wiki/concepts/c1.md"
        assert entry.operation_id == "op-1"

    def test_pending_to_merged(self, tmp_path):
        ca.create_curate_candidate(
            str(tmp_path),
            candidate_id="c-1",
            synthesis_id="syn-1",
            vault_id="v-1",
            session_id="sess-1",
            transcript_sha256=SHA,
        )
        entry = ca.transition_curate_state(
            str(tmp_path), "c-1", "merged",
            note_path="wiki/concepts/existing.md",
            operation_id="op-2",
        )
        assert entry.state == "merged"

    def test_pending_to_rejected(self, tmp_path):
        ca.create_curate_candidate(
            str(tmp_path),
            candidate_id="c-1",
            synthesis_id="syn-1",
            vault_id="v-1",
            session_id="sess-1",
            transcript_sha256=SHA,
        )
        entry = ca.transition_curate_state(
            str(tmp_path), "c-1", "rejected",
            reason="low confidence",
        )
        assert entry.state == "rejected"

    def test_pending_to_noop(self, tmp_path):
        ca.create_curate_candidate(
            str(tmp_path),
            candidate_id="c-1",
            synthesis_id="syn-1",
            vault_id="v-1",
            session_id="sess-1",
            transcript_sha256=SHA,
        )
        entry = ca.transition_curate_state(
            str(tmp_path), "c-1", "noop",
            reason="semantic duplicate",
        )
        assert entry.state == "noop"

    def test_created_to_reverted(self, tmp_path):
        ca.create_curate_candidate(
            str(tmp_path),
            candidate_id="c-1",
            synthesis_id="syn-1",
            vault_id="v-1",
            session_id="sess-1",
            transcript_sha256=SHA,
        )
        ca.transition_curate_state(str(tmp_path), "c-1", "created")
        entry = ca.transition_curate_state(str(tmp_path), "c-1", "reverted")
        assert entry.state == "reverted"

    def test_created_to_conflict(self, tmp_path):
        ca.create_curate_candidate(
            str(tmp_path),
            candidate_id="c-1",
            synthesis_id="syn-1",
            vault_id="v-1",
            session_id="sess-1",
            transcript_sha256=SHA,
        )
        ca.transition_curate_state(str(tmp_path), "c-1", "created")
        entry = ca.transition_curate_state(
            str(tmp_path), "c-1", "conflict",
            reason="note changed after operation",
        )
        assert entry.state == "conflict"
        assert entry.reason == "note changed after operation"


class TestInvalidTransitions:
    def test_reject_after_created_is_illegal(self, tmp_path):
        ca.create_curate_candidate(
            str(tmp_path),
            candidate_id="c-1",
            synthesis_id="syn-1",
            vault_id="v-1",
            session_id="sess-1",
            transcript_sha256=SHA,
        )
        ca.transition_curate_state(str(tmp_path), "c-1", "created")
        with pytest.raises(ValueError):
            ca.transition_curate_state(str(tmp_path), "c-1", "rejected")

    def test_create_after_rejected_is_illegal(self, tmp_path):
        ca.create_curate_candidate(
            str(tmp_path),
            candidate_id="c-1",
            synthesis_id="syn-1",
            vault_id="v-1",
            session_id="sess-1",
            transcript_sha256=SHA,
        )
        ca.transition_curate_state(str(tmp_path), "c-1", "rejected")
        with pytest.raises(ValueError):
            ca.transition_curate_state(str(tmp_path), "c-1", "created")

    def test_first_state_must_be_pending_review(self, tmp_path):
        with pytest.raises(ValueError):
            ca.record_curate_audit(
                str(tmp_path),
                candidate_id="c-1",
                synthesis_id="syn-1",
                vault_id="v-1",
                session_id="sess-1",
                transcript_sha256=SHA,
                state="created",
            )

    def test_unknown_transition_rejected(self, tmp_path):
        ca.create_curate_candidate(
            str(tmp_path),
            candidate_id="c-1",
            synthesis_id="syn-1",
            vault_id="v-1",
            session_id="sess-1",
            transcript_sha256=SHA,
        )
        ca.transition_curate_state(str(tmp_path), "c-1", "noop")
        # Terminal noop cannot move anywhere.
        with pytest.raises(ValueError):
            ca.transition_curate_state(str(tmp_path), "c-1", "created")


class TestIdempotency:
    def test_repeat_same_state_is_idempotent(self, tmp_path):
        ca.create_curate_candidate(
            str(tmp_path),
            candidate_id="c-1",
            synthesis_id="syn-1",
            vault_id="v-1",
            session_id="sess-1",
            transcript_sha256=SHA,
        )
        ca.transition_curate_state(str(tmp_path), "c-1", "created")
        before = len(ca.read_curate_audit(str(tmp_path)))
        ca.transition_curate_state(str(tmp_path), "c-1", "created")
        after = len(ca.read_curate_audit(str(tmp_path)))
        assert before == after

    def test_duplicate_create_is_idempotent(self, tmp_path):
        e1 = ca.create_curate_candidate(
            str(tmp_path),
            candidate_id="c-1",
            synthesis_id="syn-1",
            vault_id="v-1",
            session_id="sess-1",
            transcript_sha256=SHA,
        )
        e2 = ca.create_curate_candidate(
            str(tmp_path),
            candidate_id="c-1",
            synthesis_id="syn-1",
            vault_id="v-1",
            session_id="sess-1",
            transcript_sha256=SHA,
        )
        assert e1.timestamp == e2.timestamp
        assert len(ca.read_curate_audit(str(tmp_path))) == 1


class TestReadOrder:
    def test_read_returns_newest_first(self, tmp_path):
        ca.create_curate_candidate(
            str(tmp_path),
            candidate_id="c-1",
            synthesis_id="syn-1",
            vault_id="v-1",
            session_id="sess-1",
            transcript_sha256=SHA,
        )
        ca.create_curate_candidate(
            str(tmp_path),
            candidate_id="c-2",
            synthesis_id="syn-2",
            vault_id="v-1",
            session_id="sess-2",
            transcript_sha256=SHA,
        )
        entries = ca.read_curate_audit(str(tmp_path))
        assert [e.candidate_id for e in entries] == ["c-2", "c-1"]

    def test_latest_ignores_older_entries(self, tmp_path):
        ca.create_curate_candidate(
            str(tmp_path),
            candidate_id="c-1",
            synthesis_id="syn-1",
            vault_id="v-1",
            session_id="sess-1",
            transcript_sha256=SHA,
        )
        ca.transition_curate_state(str(tmp_path), "c-1", "created")
        ca.transition_curate_state(str(tmp_path), "c-1", "reverted")
        latest = ca.latest_curate_state(str(tmp_path), "c-1")
        assert latest is not None
        assert latest.state == "reverted"


class TestFailureRetryConsistency:
    def test_failed_is_terminal(self, tmp_path):
        ca.create_curate_candidate(
            str(tmp_path),
            candidate_id="c-1",
            synthesis_id="syn-1",
            vault_id="v-1",
            session_id="sess-1",
            transcript_sha256=SHA,
        )
        ca.transition_curate_state(str(tmp_path), "c-1", "failed", reason="llm error")
        with pytest.raises(ValueError):
            ca.transition_curate_state(str(tmp_path), "c-1", "created")

    def test_failure_preserves_candidate_record(self, tmp_path):
        ca.create_curate_candidate(
            str(tmp_path),
            candidate_id="c-1",
            synthesis_id="syn-1",
            vault_id="v-1",
            session_id="sess-1",
            transcript_sha256=SHA,
            extra={"title": "Test Concept"},
        )
        ca.transition_curate_state(str(tmp_path), "c-1", "failed", reason="timeout")
        latest = ca.latest_curate_state(str(tmp_path), "c-1")
        assert latest is not None
        assert latest.state == "failed"
        # extra from the original candidate record is preserved
        assert latest.extra == {"title": "Test Concept"}

    def test_create_candidate_after_failure_requires_new_id(self, tmp_path):
        ca.create_curate_candidate(
            str(tmp_path),
            candidate_id="c-1",
            synthesis_id="syn-1",
            vault_id="v-1",
            session_id="sess-1",
            transcript_sha256=SHA,
        )
        ca.transition_curate_state(str(tmp_path), "c-1", "failed", reason="timeout")
        # A retry for the same logical candidate must use a new candidate_id.
        entry = ca.create_curate_candidate(
            str(tmp_path),
            candidate_id="c-1-retry",
            synthesis_id="syn-1",
            vault_id="v-1",
            session_id="sess-1",
            transcript_sha256=SHA,
        )
        assert entry.state == "pending_review"
        assert entry.candidate_id == "c-1-retry"
