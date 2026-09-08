"""Tests for the session synthesis audit trail (TDD: write failing tests first)."""
import json
import os
import pytest

from bdh_graph_harness.memory import synthesis_audit as audit_mod
from bdh_graph_harness.memory.synthesis_audit import (
    SynthesisAuditEntry,
    compute_transcript_sha256,
    read_synthesis_audit,
    record_synthesis_audit,
)


class TestTranscriptSha256:
    def test_sha256_is_deterministic(self):
        h1 = compute_transcript_sha256("hello world")
        h2 = compute_transcript_sha256("hello world")
        assert h1 == h2

    def test_sha256_is_hex_64_chars(self):
        h = compute_transcript_sha256("any content")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_sha256_differs_for_different_content(self):
        h1 = compute_transcript_sha256("content A")
        h2 = compute_transcript_sha256("content B")
        assert h1 != h2


class TestRecordAndRead:
    def test_record_creates_file_and_returns_entry(self, tmp_path):
        entry = record_synthesis_audit(
            str(tmp_path),
            session_id="sess-1",
            synthesis_id="syn-1",
            transcript_sha256="abc123",
            source="session_synthesis",
            vault="test-vault",
            provider="omlx",
            model="qwen3.8-27b-oq4e-mtp",
            hebbian_updates=2,
            outcome="created",
            concept_ids=["wiki/concepts/foo"],
        )
        assert entry.session_id == "sess-1"
        assert entry.outcome == "created"
        assert entry.vault == "test-vault"

    def test_record_appends_multiple_entries(self, tmp_path):
        record_synthesis_audit(
            str(tmp_path),
            session_id="sess-1",
            synthesis_id="syn-1",
            transcript_sha256="a",
            source="session_synthesis",
            vault="v",
            provider="omlx",
            model="m",
            hebbian_updates=1,
            outcome="created",
        )
        record_synthesis_audit(
            str(tmp_path),
            session_id="sess-2",
            synthesis_id="syn-2",
            transcript_sha256="b",
            source="session_synthesis",
            vault="v",
            provider="omlx",
            model="m",
            hebbian_updates=0,
            outcome="merged",
            concept_ids=["wiki/concepts/bar"],
            reason="semantic match",
        )
        entries = read_synthesis_audit(str(tmp_path))
        assert len(entries) == 2
        assert entries[0].outcome == "created"
        assert entries[1].outcome == "merged"
        assert entries[1].reason == "semantic match"

    def test_read_empty_when_no_file(self, tmp_path):
        entries = read_synthesis_audit(str(tmp_path))
        assert entries == []

    def test_read_ignores_malformed_lines(self, tmp_path):
        path = os.path.join(str(tmp_path), ".bdh-audit", "synthesis.jsonl")
        os.makedirs(os.path.dirname(path))
        with open(path, "w") as f:
            f.write('{"valid": "line"}\n')
            f.write("not json\n")
            f.write("\n")
        # Should not crash; returns partial entries
        entries = read_synthesis_audit(str(tmp_path))
        # The malformed JSON and empty lines are skipped
        assert isinstance(entries, list)

    def test_all_outcome_types(self, tmp_path):
        for outcome in ("created", "merged", "noop", "failed"):
            record_synthesis_audit(
                str(tmp_path),
                session_id="s",
                synthesis_id=f"syn-{outcome}",
                transcript_sha256="x",
                source="session_synthesis",
                vault="v",
                provider="omlx",
                model="m",
                hebbian_updates=0,
                outcome=outcome,
                reason="test",
            )
        entries = read_synthesis_audit(str(tmp_path))
        assert len(entries) == 4
        assert {e.outcome for e in entries} == {"created", "merged", "noop", "failed"}
