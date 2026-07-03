"""
BDH Graph Harness — Embedding module.

Ollama-based text embeddings and cosine similarity.
"""

import sys
import json

from bdh_graph_harness.config import CONFIG, OLLAMA_EMBED_URL, logger, retry_with_backoff


def get_embeddings(texts, batch_size=32):
    """Get embeddings from Ollama using the batch-capable /api/embed endpoint."""
    import urllib.request
    import time as _time

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        def _embed_batch():
            data = json.dumps({
                "model": CONFIG['embedding_model'],
                "input": batch,
            }).encode()
            req = urllib.request.Request(OLLAMA_EMBED_URL, data=data,
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                return result.get('embeddings', [])

        try:
            batch_embs = retry_with_backoff(_embed_batch)
            all_embeddings.extend(batch_embs)
            print(f"  ... {min(i+batch_size, len(texts))}/{len(texts)} embedded")
        except Exception as e:
            print(f"  ⚠ Batch error at {i}: {e}", file=sys.stderr)
            # Fallback: embed one by one with delay
            _time.sleep(1)
            for text in batch:
                single_data = json.dumps({
                    "model": CONFIG['embedding_model'],
                    "prompt": text[:2000],
                }).encode()
                single_req = urllib.request.Request(
                    CONFIG['ollama_url'].rstrip('/') + '/api/embeddings',
                    data=single_data,
                    headers={'Content-Type': 'application/json'},
                )
                try:
                    with urllib.request.urlopen(single_req, timeout=60) as resp:
                        result = json.loads(resp.read())
                        all_embeddings.append(result.get('embedding', []))
                except Exception:
                    all_embeddings.append([])
                _time.sleep(0.1)

    return all_embeddings


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)