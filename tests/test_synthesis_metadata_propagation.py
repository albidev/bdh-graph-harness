"""Tests for synthesis metadata propagation into note frontmatter and evidence."""
import os
import pytest

from bdh_graph_harness.neurogenesis.creator import create_note
from bdh_graph_harness.neurogenesis.merge import assimilate_evidence


class TestCreateNoteSynthesisMeta:
    def test_create_note_includes_synthesis_metadata(self, tmp_path):
        """create_note persists synthesis_meta in frontmatter."""
        meta = {
            'session_id': 'sess-abc',
            'synthesis_id': 'syn-xyz',
            'transcript_sha256': 'deadbeef' * 8,
        }
        note_id = create_note(
            str(tmp_path), "My Concept", "A durable concept.",
            ["Source A"], "test query",
            source="session_synthesis",
            synthesis_meta=meta,
        )
        assert note_id is not None
        with open(os.path.join(str(tmp_path), note_id + ".md"), encoding="utf-8") as f:
            content = f.read()
        assert 'synthesis_session_id: "sess-abc"' in content
        assert 'synthesis_id: "syn-xyz"' in content
        assert 'transcript_sha256: "deadbeef' in content

    def test_create_note_without_synthesis_meta_unchanged(self, tmp_path):
        """create_note without synthesis_meta produces unchanged frontmatter."""
        note_id = create_note(
            str(tmp_path), "No Meta", "A concept.",
            ["Source"], "query",
            source="session_synthesis",
        )
        assert note_id is not None
        with open(os.path.join(str(tmp_path), note_id + ".md"), encoding="utf-8") as f:
            content = f.read()
        assert 'synthesis_session_id' not in content
        assert 'synthesis_id' not in content
        assert 'transcript_sha256' not in content

    def test_create_note_with_partial_synthesis_meta(self, tmp_path):
        """create_note with only session_id skips empty fields."""
        meta = {'session_id': 'sess-only'}
        note_id = create_note(
            str(tmp_path), "Partial Meta", "A concept.",
            ["Source"], "query",
            synthesis_meta=meta,
        )
        assert note_id is not None
        with open(os.path.join(str(tmp_path), note_id + ".md"), encoding="utf-8") as f:
            content = f.read()
        assert 'synthesis_session_id: "sess-only"' in content
        assert 'synthesis_id' not in content
        assert 'transcript_sha256' not in content


class TestAssimilateEvidenceSynthesisMeta:
    def test_assimilate_evidence_records_synthesis_metadata(self, tmp_path):
        """assimilate_evidence records synthesis metadata in evidence section."""
        note = tmp_path / "wiki" / "concepts" / "note.md"
        note.parent.mkdir(parents=True)
        note.write_text(
            "---\ntitle: Existing Note\n---\n\nOriginal content.\n",
            encoding="utf-8",
        )
        meta = {
            'session_id': 'sess-merge-1',
            'synthesis_id': 'syn-merge-1',
            'transcript_sha256': 'cafebabe' * 8,
        }
        result = assimilate_evidence(
            str(tmp_path),
            "wiki/concepts/note",
            {"absolute_path": str(note), "title": "Existing Note"},
            "New evidence here.",
            source="session_synthesis",
            synthesis_meta=meta,
        )
        assert result["status"] == "merged"
        content = note.read_text(encoding="utf-8")
        assert 'synthesis_session_id: sess-merge-1' in content
        assert 'synthesis_id: syn-merge-1' in content
        assert 'transcript_sha256: cafebabe' in content

    def test_assimilate_evidence_without_synthesis_meta_unchanged(self, tmp_path):
        """assimilate_evidence without synthesis_meta is unchanged."""
        note = tmp_path / "wiki" / "concepts" / "note.md"
        note.parent.mkdir(parents=True)
        note.write_text(
            "---\ntitle: Existing Note\n---\n\nOriginal content.\n",
            encoding="utf-8",
        )
        result = assimilate_evidence(
            str(tmp_path),
            "wiki/concepts/note",
            {"absolute_path": str(note), "title": "Existing Note"},
            "New evidence here.",
            source="session_synthesis",
        )
        assert result["status"] == "merged"
        content = note.read_text(encoding="utf-8")
        assert 'synthesis_session_id' not in content
        assert 'synthesis_id' not in content
