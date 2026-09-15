"""The synthesis audit must distinguish "extracted" from "extracted nothing".

With Curate staging enabled, ``run_neurogenesis`` creates no notes and therefore
returns ``[]``, so its ``new_concepts_list`` is always empty. The audit used to
derive its outcome from that list alone, which meant:

  a run that extracted 4 concepts (4 candidates queued for review)
  a run that extracted nothing at all

both recorded ``outcome="noop", concept_ids=[]``. They are not the same event,
and reading the log could not tell them apart — which is exactly how a working
synthesis pipeline looks identical to a dead one.

These tests pin the distinction, and that a staging FAILURE is not silently
reported as "nothing to stage".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bdh_graph_harness.memory.synthesis_audit import (  # noqa: E402
    SynthesisAuditEntry,
    read_synthesis_audit,
    record_synthesis_audit,
)


def test_staged_count_is_persisted_and_read_back(tmp_path):
    """A staged run records its candidate count."""
    vault = tmp_path / "vault"
    vault.mkdir()
    record_synthesis_audit(
        str(vault),
        session_id="s1",
        synthesis_id="syn-1",
        transcript_sha256="a" * 64,
        source="room_synthesis",
        vault="client-a",
        provider="omlx",
        model="m",
        hebbian_updates=0,
        outcome="staged",
        staged_count=4,
        staged_duplicate_count=1,
    )
    entries = read_synthesis_audit(str(vault))
    assert len(entries) == 1
    assert entries[0].outcome == "staged"
    assert entries[0].staged_count == 4
    assert entries[0].staged_duplicate_count == 1


def test_staged_and_noop_are_distinguishable(tmp_path):
    """The two runs the old log conflated now read differently."""
    vault = tmp_path / "vault"
    vault.mkdir()

    # a run that extracted nothing
    record_synthesis_audit(
        str(vault),
        session_id="s1", synthesis_id="syn-empty",
        transcript_sha256="a" * 64, source="room_synthesis", vault="client-a",
        provider="omlx", model="m", hebbian_updates=0,
        outcome="noop", reason="extractor produced no concepts",
        staged_count=0,
    )
    # a run that extracted four concepts into Curate
    record_synthesis_audit(
        str(vault),
        session_id="s1", synthesis_id="syn-four",
        transcript_sha256="b" * 64, source="room_synthesis", vault="client-a",
        provider="omlx", model="m", hebbian_updates=0,
        outcome="staged", staged_count=4,
    )

    entries = {e.synthesis_id: e for e in read_synthesis_audit(str(vault))}
    empty, four = entries["syn-empty"], entries["syn-four"]

    assert empty.outcome != four.outcome
    assert empty.staged_count == 0
    assert four.staged_count == 4
    # the old fields are still both empty for the staged run — that is the point
    assert four.concept_ids == []
    assert empty.concept_ids == []


def test_rows_written_before_the_new_fields_still_read(tmp_path):
    """An existing log must stay readable: the fields are additive."""
    vault = tmp_path / "vault"
    (vault / ".bdh-audit").mkdir(parents=True)
    legacy = {
        "session_id": "s1", "synthesis_id": "syn-legacy",
        "transcript_sha256": "c" * 64, "timestamp": "2026-09-15T10:00:00+00:00",
        "source": "session_synthesis", "vault": "client-a",
        "provider": "omlx", "model": "m", "hebbian_updates": 0,
        "outcome": "noop", "concept_ids": [], "reason": "", "queued_at": "",
    }
    (vault / ".bdh-audit" / "synthesis.jsonl").write_text(
        json.dumps(legacy) + "\n", encoding="utf-8"
    )

    entries = read_synthesis_audit(str(vault))
    assert len(entries) == 1
    assert entries[0].synthesis_id == "syn-legacy"
    # absent fields default rather than raising
    assert entries[0].staged_count == 0
    assert entries[0].staged_duplicate_count == 0


def test_staged_outcome_is_a_valid_audit_outcome():
    """`staged` must be part of the outcome vocabulary."""
    entry = SynthesisAuditEntry(
        session_id="s", synthesis_id="i", transcript_sha256="d" * 64,
        timestamp="t", source="room_synthesis", vault="v", provider="p",
        model="m", hebbian_updates=0, outcome="staged", staged_count=2,
    )
    assert entry.outcome == "staged"
    assert entry.to_dict()["staged_count"] == 2
