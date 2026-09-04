import json
from pathlib import Path

from bdh_graph_harness.neurogenesis.operation_journal import record_applied_operation, revert_operation
from bdh_graph_harness.neurogenesis.synthesis_activity import build_synthesis_activity


def test_created_synthesis_activity_can_be_reverted_end_to_end(tmp_path):
    note = tmp_path / "wiki" / "concepts" / "traceable.md"
    note.parent.mkdir(parents=True)
    note.write_text("---\ntitle: Traceable concept\n---\nBody\n", encoding="utf-8")
    (tmp_path / "wiki" / "index.md").write_text(
        "## concepts\n- [[wiki/concepts/traceable|Traceable concept]]\n",
        encoding="utf-8",
    )
    audit_dir = tmp_path / ".bdh-audit"
    audit_dir.mkdir()
    (audit_dir / "synthesis.jsonl").write_text(
        json.dumps({
            "synthesis_id": "synth-e2e",
            "session_id": "session-e2e",
            "timestamp": "2026-09-04T15:00:00+00:00",
            "transcript_sha256": "a" * 64,
            "source": "session_synthesis",
            "vault": "core",
            "provider": "omlx",
            "model": "test-model",
            "hebbian_updates": 0,
            "outcome": "created",
            "concept_ids": ["wiki/concepts/traceable"],
            "reason": "",
            "queued_at": "",
        }) + "\n",
        encoding="utf-8",
    )
    operation = record_applied_operation(
        tmp_path,
        synthesis_meta={"synthesis_id": "synth-e2e", "session_id": "session-e2e"},
        action="created",
        note_path=note,
    )

    activity = build_synthesis_activity(tmp_path, vault_id="core")
    assert activity["count"] == 1
    assert activity["activities"][0]["operations"][0]["operation_id"] == operation["operation_id"]
    assert activity["activities"][0]["concepts"][0]["exists"] is True

    result = revert_operation(tmp_path, operation["operation_id"])
    assert result["status"] == "reverted"
    assert result["index_entry_removed"] is True

    activity_after = build_synthesis_activity(tmp_path, vault_id="core")
    assert len(activity_after["activities"][0]["operations"]) == 1
    assert activity_after["activities"][0]["operations"][0]["status"] == "reverted"
    assert not note.exists()
