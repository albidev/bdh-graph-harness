"""
BDH Graph Harness — evidence vectors for merged notes.

A merge (``neurogenesis/merge.py::assimilate_evidence``) appends a claim to an
existing note instead of creating a new node. The note keeps ONE vector built
from the first 2000 chars of its text, so a claim appended to a long hub note is
invisible to vector search, while the same claim filed as a new note would be
top-1 for its own wording. This module gives every merged claim its own vector
in a sibling collection and lifts the parent note's vector score to the best
claim match, so merging never costs retrieval compared to creating a note.

The note graph is untouched: claims are not nodes, they only score their parent.
"""

import hashlib

from bdh_graph_harness.config import CONFIG, logger
from bdh_graph_harness.retrieval.embeddings import get_embeddings

EVIDENCE_COLLECTION_SUFFIX = '__evidence'

# notes collection id -> evidence collection, filled by compute_all_embeddings.
_EVIDENCE_COLLECTIONS = {}


def _key(collection):
    collection_id = getattr(collection, 'id', None)
    return str(collection_id) if collection_id is not None else None


def evidence_collection_for(collection):
    key = _key(collection)
    return _EVIDENCE_COLLECTIONS.get(key) if key else None


def _claim_id(note_id, claim):
    return f"{note_id}#ev-{hashlib.sha256(claim.encode()).hexdigest()[:16]}"


def sync_evidence_embeddings(client, notes_collection, collection_name, nodes, embedding_function=None):
    """Mirror every node's ``evidence`` claims into ``<collection>__evidence``.

    Incremental: ids are content-addressed, so only new claims are embedded and
    claims no longer in the vault are deleted.
    """
    key = _key(notes_collection)
    if not CONFIG.get('evidence_vectors_enabled', True):
        _EVIDENCE_COLLECTIONS.pop(key, None)
        return None

    evidence = client.get_or_create_collection(
        f"{collection_name}{EVIDENCE_COLLECTION_SUFFIX}",
        metadata={'hnsw:space': 'cosine'},
        embedding_function=embedding_function,
    )
    wanted = {}
    for note_id, node in nodes.items():
        for claim in node.get('evidence') or []:
            wanted[_claim_id(note_id, claim)] = (note_id, claim)

    existing = set(evidence.get(include=[])['ids']) if evidence.count() else set()
    stale = existing - set(wanted)
    if stale:
        evidence.delete(ids=list(stale))
    missing = [claim_id for claim_id in wanted if claim_id not in existing]
    if missing:
        embeddings = get_embeddings([wanted[claim_id][1][:2000] for claim_id in missing]) or []
        for claim_id, embedding in zip(missing, embeddings):
            if not embedding:
                continue
            note_id, claim = wanted[claim_id]
            evidence.upsert(
                ids=[claim_id],
                embeddings=[embedding],
                documents=[claim[:500]],
                metadatas=[{'note_id': note_id}],
            )
    if stale or missing:
        print(f"  Evidence vectors: {evidence.count()} claims (+{len(missing)} / -{len(stale)})")
    if key:
        _EVIDENCE_COLLECTIONS[key] = evidence
    return evidence


def fold_evidence_scores(collection, query_embedding, raw_vector_scores, n_results, valid_ids):
    """Raise each parent note's vector score to its best-matching merged claim.

    Mutates ``raw_vector_scores`` and returns ``{note_id: claim_similarity}`` for
    the notes that were lifted.
    """
    evidence = evidence_collection_for(collection)
    if evidence is None:
        return {}
    try:
        count = evidence.count()
        if not count:
            return {}
        results = evidence.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, count),
            include=['metadatas', 'distances'],
        )
    except Exception as exc:  # auxiliary index: a broken sibling must not take retrieval down
        logger.warning(f"Evidence vector query failed, using note vectors only: {exc}")
        return {}

    lifted = {}
    for meta, distance in zip((results.get('metadatas') or [[]])[0], (results.get('distances') or [[]])[0]):
        note_id = (meta or {}).get('note_id')
        if note_id not in valid_ids:
            continue
        similarity = max(0.0, 1.0 - distance)
        if similarity > raw_vector_scores.get(note_id, 0.0):
            raw_vector_scores[note_id] = similarity
            lifted[note_id] = similarity
    return lifted
