"""
BDH Graph Harness — ChromaDB store module.

Computes and stores embeddings in ChromaDB with incremental refresh.
"""

import os
import hashlib

from bdh_graph_harness.config import CONFIG
from bdh_graph_harness.retrieval.embeddings import get_embeddings


def _get_ollama_embedding_function():
    """Create an OllamaEmbeddingFunction configured from CONFIG.
    
    This allows ChromaDB to use nomic-embed-text-v2-moe (768d) for query_texts
    auto-embedding, instead of the default all-MiniLM-L6-v2 (384d) which
    causes dimension mismatch errors.
    
    Returns None if the ollama python package is not installed (graceful fallback).
    """
    try:
        from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
        return OllamaEmbeddingFunction(
            url=CONFIG.get('ollama_url', 'http://127.0.0.1:11434'),
            model_name=CONFIG.get('embedding_model', 'nomic-embed-text-v2-moe'),
        )
    except (ImportError, ValueError):
        return None


def compute_all_embeddings(
    nodes,
    vault_root,
    force_refresh=False,
    *,
    chroma_path=None,
    collection_name=None,
    config=None,
):
    """Compute and store embeddings in ChromaDB, with incremental refresh.

    Uses ChromaDB PersistentClient — all vectors stored under *chroma_path*
    (defaults to ``.bdh-chroma/`` inside the vault).  Content hash stored as
    metadata to detect note changes.

    Parameters
    ----------
    nodes:
        Graph nodes dict ``{note_id: {...}}``.
    vault_root:
        Path to the vault (used as base for a relative *chroma_path*).
    force_refresh:
        Re-embed all notes even if the content hash is unchanged.
    chroma_path:
        Explicit Chroma persistence directory.  If omitted, resolved from
        *config* (``chroma_path`` key) relative to *vault_root*.
    collection_name:
        Explicit collection name.  If omitted, falls back to *config* or
        the global ``CONFIG``.
    config:
        Per-vault settings dict.  Merged with global ``CONFIG`` as fallback.
    """
    import chromadb

    cfg = config or CONFIG

    if chroma_path is None:
        raw_cp = cfg.get('chroma_path', '.bdh-chroma')
        if os.path.isabs(raw_cp):
            chroma_path = raw_cp
        else:
            chroma_path = os.path.join(vault_root, raw_cp)

    resolved_collection = collection_name or cfg.get('chroma_collection', 'notes')

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        resolved_collection,
        metadata={'hnsw:space': 'cosine'},
        embedding_function=_get_ollama_embedding_function(),
    )

    # Compute content hashes for all notes
    current_hashes = {}
    for note_id, node in nodes.items():
        text = node['text']
        current_hashes[note_id] = hashlib.sha256(text.encode()).hexdigest()[:16]

    # Get existing hashes from ChromaDB metadata
    existing = {}
    if collection.count() > 0:
        all_data = collection.get(include=['metadatas'])
        for i, mid in enumerate(all_data['ids']):
            meta = all_data['metadatas'][i] if all_data['metadatas'] else {}
            existing[mid] = meta

    # Find notes that need embedding
    to_compute = []
    for note_id, node in nodes.items():
        if force_refresh:
            to_compute.append(note_id)
        elif note_id not in existing:
            to_compute.append(note_id)
        elif existing.get(note_id, {}).get('content_hash') != current_hashes.get(note_id):
            print(f"  🔄 Note changed: {note_id}")
            to_compute.append(note_id)

    # Remove deleted notes from ChromaDB
    deleted = set(existing.keys()) - set(nodes.keys())
    if deleted:
        collection.delete(ids=list(deleted))
        print(f"  🗑️ Removed {len(deleted)} deleted notes from ChromaDB")

    if to_compute:
        print(f"Computing embeddings for {len(to_compute)} notes...")
        texts = [nodes[nid]['text'][:2000] for nid in to_compute]
        embs = get_embeddings(texts)

        # Verify we got the right number of embeddings (guard against Ollama
        # returning fewer than requested, which would misalign IDs↔embeddings)
        if embs is None:
            print(f"  ⚠ No embeddings returned from Ollama, skipping")
            return collection
        if len(embs) < len(to_compute):
            print(f"  ⚠ Ollama returned {len(embs)} embeddings for {len(to_compute)} notes "
                  f"— truncating to avoid misalignment")
            to_compute = to_compute[:len(embs)]

        # Upsert into ChromaDB
        for nid, emb in zip(to_compute, embs):
            if not emb:
                continue
            collection.upsert(
                ids=[nid],
                embeddings=[emb],
                documents=[nodes[nid]['text'][:500]],
                metadatas=[{
                    'content_hash': current_hashes[nid],
                    'title': nodes[nid]['title'],
                    'tags': nodes[nid]['tags'],
                    'type': nodes[nid].get('type', 'concept'),
                    'source_id': nodes[nid].get('source_id', 'vault'),
                    'source_type': nodes[nid].get('source_type', 'vault'),
                    'relative_path': nodes[nid].get('relative_path', ''),
                    'writable': str(nodes[nid].get('writable', True)).lower(),
                }],
            )
        print(f"  ChromaDB: {collection.count()} notes stored")
    else:
        print(f"Using ChromaDB cache ({collection.count()} notes)")

    return collection