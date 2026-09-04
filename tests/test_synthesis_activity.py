from pathlib import Path

from bdh_graph_harness.neurogenesis.synthesis_activity import build_synthesis_activity


def test_activity_groups_audit_and_operations_by_synthesis(tmp_path):
    audit = tmp_path / ".bdh-audit" / "synthesis.jsonl"
    audit.parent.mkdir(parents=True)
    audit.write_text(
        '{"session_id":"sess-1","synthesis_id":"synth-1","transcript_sha256":"' + 'a' * 64 + '",'
        '"timestamp":"2026-09-04T10:00:00+00:00","source":"session_synthesis","vault":"core",'
        '"provider":"omlx","model":"qwen","hebbian_updates":1,"outcome":"created",'
        '"concept_ids":["wiki/concepts/one"],"reason":"","queued_at":"1"}\n',
        encoding="utf-8",
    )
    note = tmp_path / "wiki" / "concepts" / "one.md"
    note.parent.mkdir(parents=True)
    note.write_text("---\ntitle: One\n---\n\nDefinition.\n", encoding="utf-8")
    op = tmp_path / ".bdh-audit" / "neurogenesis-operations.jsonl"
    op.write_text(
        '{"operation_id":"op-1","synthesis_id":"synth-1","action":"created",'
        '"status":"applied","note_path":"wiki/concepts/one.md","after_hash":"x"}\n',
        encoding="utf-8",
    )

    result = build_synthesis_activity(tmp_path)

    assert result["vault_id"] == tmp_path.name
    assert result["activities"][0]["synthesis_id"] == "synth-1"
    assert result["activities"][0]["concepts"][0]["title"] == "One"
    assert result["activities"][0]["operations"][0]["operation_id"] == "op-1"
