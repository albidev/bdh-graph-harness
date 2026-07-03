"""
BDH Graph Harness — ChromaDB store module.

Computes and stores embeddings in ChromaDB with incremental refresh.
"""

import os
import hashlib

from bdh_graph_harness.config import CONFIG
from bdh_graph_harness.retrieval.embeddings import get_embeddings


def compute_all_embeddings(nodes, vault_root, force_refresh=False):
    """Compute and store embeddings in ChromaDB, with incremental refresh.

    Uses ChromaDB PersistentClient — all vectors stored in .bdh-chroma/ inside the vault.
    Content hash stored as metadata to detect note changes.
    """
    import chromadb

    chroma_path = os.path.join(vault_root, CONFIG['chroma_path'])
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        CONFIG['chroma_collection'],
        metadata={'hnsw:space': 'cosine'},
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
                }],
            )
        print(f"  ChromaDB: {collection.count()} notes stored")
    else:
        print(f"Using ChromaDB cache ({collection.count()} notes)")

    return collection