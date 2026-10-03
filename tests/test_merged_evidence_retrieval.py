"""Merged evidence must be retrievable like a new note, and its audit trail must not be indexed.

A merge appends a claim to an existing note instead of creating a node. Two
invariants keep that from costing retrieval:
- audit sub-bullets (session ids, hashes, the repeated merge query) stay in the
  file but never reach the indexed text;
- a claim appended past the note's embedding window still scores its parent note.
"""

from bdh_graph_harness.graph.parser import extract_evidence_claims, extract_text
from bdh_graph_harness.neurogenesis.merge import assimilate_evidence
from bdh_graph_harness.retrieval import evidence as evidence_mod


def _merged_note(tmp_path, body, claims):
    note = tmp_path / "hub.md"
    note.write_text(f"---\ntitle: Hub\n---\n# Hub\n\n{body}\n", encoding="utf-8")
    for index, claim in enumerate(claims):
        assimilate_evidence(
            tmp_path, "hub", {"path": "hub.md"}, claim,
            source_notes=["Some Source Note"], query="Explicit Curate merge target: Hub", source="session_synthesis",
            synthesis_meta={"session_id": f"sess-{index}", "synthesis_id": f"syn-{index}",
                            "transcript_sha256": "ab" * 32},
        )
    return note.read_text(encoding="utf-8")


def test_merge_audit_lines_stay_in_file_but_not_in_index(tmp_path):
    claims = ["Kill by exact PID, never by pattern.", "Launchd restarts hide the original crash."]
    content = _merged_note(tmp_path, "Original definition.", claims)

    assert "transcript_sha256" in content and "syn-1" in content  # traceability kept on disk
    text = extract_text(content)
    for leaked in ("transcript_sha256", "abab", "syn-0", "sess-1", "Explicit Curate merge target", "Some Source Note"):
        assert leaked not in text
    assert "Original definition." in text
    assert extract_evidence_claims(content) == claims
    assert extract_evidence_claims("# Plain note\n- source: not evidence\n") == []


class _Collection:
    """Minimal Chroma stand-in: cosine over 2-d vectors."""

    def __init__(self, name, collection_id=None):
        self.name, self.id, self.rows = name, collection_id, {}

    def count(self):
        return len(self.rows)

    def get(self, include=None, ids=None):
        return {"ids": list(self.rows)}

    def delete(self, ids):
        for row_id in ids:
            self.rows.pop(row_id, None)

    def upsert(self, ids, embeddings, documents, metadatas):
        self.rows[ids[0]] = (embeddings[0], metadatas[0])

    def query(self, query_embeddings, n_results, include):
        q = query_embeddings[0]
        scored = sorted(
            ((1.0 - sum(a * b for a, b in zip(q, emb)), meta) for emb, meta in self.rows.values()),
            key=lambda item: item[0],
        )[:n_results]
        return {"distances": [[d for d, _ in scored]], "metadatas": [[m for _, m in scored]]}


class _Client:
    def __init__(self):
        self.collections = {}

    def get_or_create_collection(self, name, **_kwargs):
        return self.collections.setdefault(name, _Collection(name, collection_id=f"id-{name}"))


def test_claim_past_the_embedding_window_lifts_its_parent_note(tmp_path, monkeypatch):
    claim = "Probabilistic egress failure: retry from a different NAT IP."
    content = _merged_note(tmp_path, "unrelated hub text " * 400, [claim])  # claim lands far past 2000 chars
    assert len(extract_text(content).split(claim)[0]) > 2000

    vectors = {claim: [1.0, 0.0]}
    monkeypatch.setattr(evidence_mod, "get_embeddings", lambda texts: [vectors.get(t, [0.0, 1.0]) for t in texts])
    client = _Client()
    notes = client.get_or_create_collection("notes")
    nodes = {"vault:hub": {"evidence": extract_evidence_claims(content)}, "vault:other": {"evidence": []}}
    evidence_mod.sync_evidence_embeddings(client, notes, "notes", nodes)

    # The hub's own vector matches the query poorly; the merged claim matches it exactly.
    scores = {"vault:hub": 0.2, "vault:other": 0.5}
    lifted = evidence_mod.fold_evidence_scores(notes, [1.0, 0.0], scores, 10, nodes)
    assert lifted == {"vault:hub": 1.0} and scores["vault:hub"] > scores["vault:other"]

    # Removing the claim from the vault removes its vector: no stale lift.
    evidence_mod.sync_evidence_embeddings(client, notes, "notes", {"vault:hub": {"evidence": []}})
    scores = {"vault:hub": 0.2}
    assert evidence_mod.fold_evidence_scores(notes, [1.0, 0.0], scores, 10, nodes) == {}
    assert scores == {"vault:hub": 0.2}
