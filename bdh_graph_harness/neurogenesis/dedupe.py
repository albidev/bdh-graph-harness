"""Duplicate detection for neurogenesis — avoid recreating existing concepts."""

import sys


def is_duplicate(title, existing_titles):
    """Check if a concept title already exists (case-insensitive).

    Args:
        title: The candidate concept title.
        existing_titles: Iterable of existing note titles.

    Returns:
        True if the title matches an existing title (case-insensitive), False otherwise.
    """
    title_lower = title.lower().strip()
    for existing in existing_titles:
        if existing.lower().strip() == title_lower:
            return True
    return False


def is_semantic_duplicate(title, definition, threshold=0.65, *, vault_root=None, config=None):
    """Check if a concept is semantically similar to an existing note via embeddings.

    Uses the ChromaDB collection to query the nearest neighbor by cosine similarity.
    If the best match exceeds the threshold, the concept is considered a duplicate.

    This catches variants like 'sleepcycle-consolidation' vs 'sleep-cycle-consolidation',
    or 'hebbian-update' vs 'hebbian-updates' — cases where exact string match fails
    but the semantic meaning is identical.

    Args:
        title: The candidate concept title.
        definition: The candidate concept definition (combined with title for the query).
        threshold: Cosine similarity threshold above which we consider it a duplicate.
        vault_root: Vault path for ChromaDB location.  Defaults to ``CONFIG['vault_path']``.
        config: Per-vault settings dict.  Falls back to global ``CONFIG``.

    Returns:
        True if a semantically similar note exists, False otherwise.
    """
    try:
        from bdh_graph_harness.retrieval.embeddings import get_embeddings, cosine_similarity
        from bdh_graph_harness.config import CONFIG as _GLOBAL_CONFIG
        import chromadb
        import os

        cfg = config or _GLOBAL_CONFIG

        # Build query text from title + definition
        query_text = f"{title}. {definition}" if definition else title

        # Get embedding for the candidate concept
        embs = get_embeddings([query_text[:2000]])
        if not embs or not embs[0]:
            return False  # Can't check — don't block creation

        query_emb = embs[0]

        # Query ChromaDB for nearest neighbors
        _vault_root = vault_root or cfg.get('vault_path', '')
        raw_cp = cfg.get('chroma_path', '.bdh-chroma')
        if os.path.isabs(raw_cp):
            chroma_path = raw_cp
        else:
            chroma_path = os.path.join(_vault_root, raw_cp)
        client = chromadb.PersistentClient(path=chroma_path)
        from bdh_graph_harness.retrieval.chroma_store import _get_ollama_embedding_function
        collection = client.get_or_create_collection(
            cfg.get('chroma_collection', 'notes'),
            metadata={'hnsw:space': 'cosine'},
            embedding_function=_get_ollama_embedding_function(),
        )

        if collection.count() == 0:
            return False  # Empty collection — nothing to duplicate

        results = collection.query(
            query_embeddings=[query_emb],
            n_results=3,
            include=['documents', 'distances'],
        )

        # ChromaDB returns distances (1 - cosine_similarity for cosine space)
        # but actually with 'cosine' space, distance = 1 - similarity
        # So similarity = 1 - distance
        if results.get('distances') and results['distances'][0]:
            distances = results['distances'][0]
            # ChromaDB cosine distance = 1 - cosine_similarity
            best_similarity = 1.0 - distances[0]
            if best_similarity >= threshold:
                best_match = results.get('documents', [['']])[0][0] if results.get('documents') else ''
                print(f"  🔁 Semantic dup: '{title}' matches existing (similarity={best_similarity:.3f}, match='{best_match[:60]}')", file=sys.stderr)
                return True

        return False

    except Exception as e:
        # If anything goes wrong (ChromaDB not available, embeddings fail, etc.),
        # don't block creation — fail open.
        print(f"  ⚠ Semantic dedup error: {e}", file=sys.stderr)
        return False
