"""Duplicate detection and semantic match resolution for neurogenesis."""

import sys


def is_duplicate(title, existing_titles):
    """Check if a concept title already exists (case-insensitive)."""
    title_lower = title.lower().strip()
    return any(existing.lower().strip() == title_lower for existing in existing_titles)


def find_semantic_match(title, definition, threshold=0.65, *, vault_root=None, config=None):
    """Return the nearest existing node above ``threshold`` or ``None``.

    The result includes the Chroma node ID so callers can assimilate evidence
    into the canonical note instead of merely discarding the candidate.
    """
    try:
        from bdh_graph_harness.retrieval.embeddings import get_embeddings
        from bdh_graph_harness.config import CONFIG as _GLOBAL_CONFIG
        import chromadb
        import os

        cfg = config or _GLOBAL_CONFIG
        query_text = f"{title}. {definition}" if definition else title
        embs = get_embeddings([query_text[:2000]])
        if not embs or not embs[0]:
            return None

        _vault_root = vault_root or cfg.get('vault_path', '')
        raw_cp = cfg.get('chroma_path', '.bdh-chroma')
        chroma_path = raw_cp if os.path.isabs(raw_cp) else os.path.join(_vault_root, raw_cp)
        client = chromadb.PersistentClient(path=chroma_path)
        from bdh_graph_harness.retrieval.chroma_store import _get_ollama_embedding_function
        collection = client.get_or_create_collection(
            cfg.get('chroma_collection', 'notes'),
            metadata={'hnsw:space': 'cosine'},
            embedding_function=_get_ollama_embedding_function(),
        )
        if collection.count() == 0:
            return None

        results = collection.query(
            query_embeddings=[embs[0]],
            n_results=3,
            include=['documents', 'distances', 'metadatas'],
        )
        distances = results.get('distances', [[]])[0]
        if not distances:
            return None
        similarity = 1.0 - distances[0]
        if similarity < threshold:
            return None

        ids = results.get('ids', [[]])[0]
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        document = documents[0] if documents else ''
        metadata = metadatas[0] if metadatas else {}
        print(
            f"  🔁 Semantic match: '{title}' matches existing "
            f"(similarity={similarity:.3f}, match='{document[:60]}')",
            file=sys.stderr,
        )
        return {
            'node_id': ids[0] if ids else None,
            'title': metadata.get('title') or document[:120],
            'similarity': similarity,
            'document': document,
        }
    except Exception as exc:
        # Dedup/merge must fail open: an embedding outage must not lose a concept.
        print(f"  ⚠ Semantic match error: {exc}", file=sys.stderr)
        return None


def is_semantic_duplicate(title, definition, threshold=0.65, *, vault_root=None, config=None):
    """Backward-compatible boolean wrapper around :func:`find_semantic_match`."""
    return find_semantic_match(
        title,
        definition,
        threshold,
        vault_root=vault_root,
        config=config,
    ) is not None
