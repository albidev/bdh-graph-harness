from pathlib import Path

from bdh_graph_harness.neurogenesis.operation_journal import (
    list_operation_records,
    record_applied_operation,
    revert_operation,
)


def _meta():
    return {
        "synthesis_id": "synth-1",
        "session_id": "session-1",
        "transcript_sha256": "a" * 64,
        "queued_at": "1700000000.0",
    }


def test_created_operation_is_archived_on_revert(tmp_path):
    note = tmp_path / "wiki" / "concepts" / "new.md"
    note.parent.mkdir(parents=True)
    note.write_text("created", encoding="utf-8")

    record = record_applied_operation(
        tmp_path,
        synthesis_meta=_meta(),
        action="created",
        note_path=note,
    )
    result = revert_operation(tmp_path, record["operation_id"])

    assert result["status"] == "reverted"
    assert not note.exists()
    assert Path(result["archive_path"]).read_text(encoding="utf-8") == "created"
    assert list_operation_records(tmp_path)[-1]["status"] == "reverted"


def test_created_operation_removes_index_entry_on_revert(tmp_path):
    note = tmp_path / "wiki" / "concepts" / "new.md"
    note.parent.mkdir(parents=True)
    note.write_text("created", encoding="utf-8")
    index = tmp_path / "wiki" / "index.md"
    index.write_text("## concepts\n- [[wiki/concepts/new|New]]\n- [[wiki/concepts/other|Other]]\n", encoding="utf-8")

    record = record_applied_operation(
        tmp_path,
        synthesis_meta=_meta(),
        action="created",
        note_path=note,
    )
    result = revert_operation(tmp_path, record["operation_id"])

    assert result["index_entry_removed"] is True
    assert "wiki/concepts/new" not in index.read_text(encoding="utf-8")
    assert "wiki/concepts/other" in index.read_text(encoding="utf-8")


def test_merged_operation_restores_before_snapshot(tmp_path):
    note = tmp_path / "wiki" / "concepts" / "existing.md"
    note.parent.mkdir(parents=True)
    note.write_text("after", encoding="utf-8")

    record = record_applied_operation(
        tmp_path,
        synthesis_meta=_meta(),
        action="merged",
        note_path=note,
        before_content="before",
    )
    result = revert_operation(tmp_path, record["operation_id"])

    assert result["status"] == "reverted"
    assert note.read_text(encoding="utf-8") == "before"
    assert result["before_hash"] != result["after_hash"]


def test_revert_refuses_note_changed_after_operation(tmp_path):
    note = tmp_path / "wiki" / "concepts" / "existing.md"
    note.parent.mkdir(parents=True)
    note.write_text("after", encoding="utf-8")

    record = record_applied_operation(
        tmp_path,
        synthesis_meta=_meta(),
        action="merged",
        note_path=note,
        before_content="before",
    )
    note.write_text("later edit", encoding="utf-8")

    result = revert_operation(tmp_path, record["operation_id"])

    assert result["status"] == "conflict"
    assert note.read_text(encoding="utf-8") == "later edit"
