#!/usr/bin/env python3
"""
BDH Graph Harness
=================
Replicates BDH properties using an Obsidian vault graph as neuron substrate.

- Notes = neurons (monosemantic, atomic concepts)
- Wikilinks = synapses (weighted connections)
- Hebbian update = reinforce links between co-activated notes
- Working memory = active note subset + weights, persisted between sessions
- Attention = embedding similarity (seed) + graph traversal (k-hop expansion)
- ChromaDB vector store for KNN search
- LLM integration: grounded responses with citations
- Neurogenesis: auto-create new notes from novel concepts

Usage:
    python3 harness.py --config bdh-config.yaml <query>
    python3 harness.py --vault ~/Documents/Hermes --interactive
    python3 harness.py --stats
    python3 harness.py --hebbian-show
    python3 harness.py --refresh-embeddings
    python3 harness.py --serve              # start API server
"""

import os
import re
import sys
import json
import time
import math
import fcntl
import hashlib
import pathlib
import logging
import argparse
import statistics
from collections import defaultdict, deque
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STATE_FILE = ".bdh-state.json"
LOCK_FILE = ".bdh-state.lock"
DEFAULT_CONFIG_PATHS = [
    "bdh-config.yaml",
    "~/.bdh-config.yaml",
]

# Defaults (overridden by config file)
CONFIG = {
    'vault_path': os.path.expanduser('~/Documents/Hermes'),
    'ollama_url': 'http://127.0.0.1:11434',
    # Embedding (Ollama)
    'embedding_model': 'nomic-embed-text-v2-moe',
    # LLM provider (ollama | openrouter)
    'llm_provider': 'ollama',
    'llm_model': 'gemma4:12b-mlx',
    'openrouter_url': 'https://openrouter.ai/api/v1/chat/completions',
    'openrouter_key': '',  # set in config or env
    'llm_temperature': 0.3,
    'llm_max_ctx': 4096,
    'llm_timeout': 300,
    'chroma_path': '.bdh-chroma',
    'chroma_collection': 'notes',
    'seed_count': 5,
    'max_hop': 2,
    'active_threshold': 0.25,
    'hub_dampening': True,
    'hub_degree_threshold': 15,
    'max_neighbors_per_hop': 10,
    'alpha': 0.7,
    'beta': 0.3,
    'decay': 0.95,
    'neurogenesis_dir': 'concepts',
    'neurogenesis_enabled': True,
    'api_host': '127.0.0.1',
    'api_port': 8642,
    'python_exec': sys.executable,
    # Hybrid search (Phase 3.1)
    'hybrid_search': True,
    'hybrid_alpha': 0.7,   # weight for vector similarity
    'hybrid_beta': 0.3,    # weight for BM25 keyword score
    'bm25_k1': 1.5,
    'bm25_b': 0.75,
    # Adaptive threshold (Phase 3.3)
    'adaptive_threshold': True,
    'threshold_floor': 0.15,
    # Online plasticity (Phase 3.2)
    'online_plasticity': True,
    'stream_enabled': True,
}

# Derived config (set after loading)
OLLAMA_EMBED_URL = None
OLLAMA_LLM_URL = None

# Logging
logger = logging.getLogger('bdh')


def load_config(config_path=None):
    """Load configuration from YAML file and merge with defaults.

    Tries the given path, then DEFAULT_CONFIG_PATHS. Sets global
    OLLAMA_EMBED_URL and OLLAMA_LLM_URL derived from ollama_url.
    Returns the merged config dict.
    """
    global CONFIG, OLLAMA_EMBED_URL, OLLAMA_LLM_URL

    import yaml  # PyYAML

    merged = dict(CONFIG)  # start with defaults

    paths_to_try = []
    if config_path:
        paths_to_try.append(os.path.expanduser(config_path))
    paths_to_try.extend(os.path.expanduser(p) for p in DEFAULT_CONFIG_PATHS)

    loaded = False
    for p in paths_to_try:
        if os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f) or {}
            merged.update(file_config)
            logger.info(f"Loaded config from {p}")
            loaded = True
            break

    if not loaded:
        logger.warning("No config file found; using defaults")

    # Expand vault path
    merged['vault_path'] = os.path.expanduser(merged['vault_path'])

    # Expand ${ENV_VAR} in config values (e.g. openrouter_key: ${OPENROUTER_API_KEY})
    for key, val in merged.items():
        if isinstance(val, str) and val.startswith('${') and val.endswith('}'):
            env_var = val[2:-1]
            merged[key] = os.environ.get(env_var, '')

    # Derived URLs — embeddings always from Ollama
    OLLAMA_EMBED_URL = merged['ollama_url'].rstrip('/') + '/api/embed'
    OLLAMA_LLM_URL = merged['ollama_url'].rstrip('/') + '/api/chat'

    # LLM endpoint depends on provider
    if merged.get('llm_provider') == 'openrouter':
        OLLAMA_LLM_URL = merged.get('openrouter_url', 'https://openrouter.ai/api/v1/chat/completions')
        key = merged.get('openrouter_key', '')
        if not key:
            logger.warning("OpenRouter provider selected but no API key found!")
        logger.info(f"LLM provider: OpenRouter ({merged.get('llm_model')})")
    else:
        logger.info(f"LLM provider: Ollama ({merged.get('llm_model')})")

    CONFIG = merged
    return merged


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def retry_with_backoff(fn, max_attempts=3, delay=2):
    """Call fn() with exponential backoff. Returns result or raises last exception."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            wait = delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Graph Extraction
# ---------------------------------------------------------------------------

WIKILINK_RE = re.compile(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]')
FRONTMATTER_RE = re.compile(r'^---\n(.*?)\n---\n', re.DOTALL)

def extract_note_id(filepath, vault_root):
    """Get the note's ID relative to vault root, without .md extension."""
    rel = os.path.relpath(filepath, vault_root)
    return rel[:-3] if rel.endswith('.md') else rel  # strip .md

def find_note_by_id(vault_root, note_id):
    """Find the actual file for a note ID (may have .md or be in a subdir)."""
    candidates = [
        os.path.join(vault_root, note_id + '.md'),
        os.path.join(vault_root, note_id),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Try case-insensitive search
    for root, dirs, files in os.walk(vault_root):
        for f in files:
            if f.endswith('.md'):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, vault_root)[:-3]
                if rel.lower() == note_id.lower():
                    return full
    return None

def parse_frontmatter(content):
    """Extract YAML frontmatter as a dict (simple parser)."""
    match = FRONTMATTER_RE.match(content)
    if not match:
        return {}
    fm = {}
    for line in match.group(1).split('\n'):
        if ':' in line:
            key, _, val = line.partition(':')
            fm[key.strip()] = val.strip()
    return fm

def extract_wikilinks(content):
    """Extract all wikilinks from note content. Returns list of (target, display)."""
    links = []
    for match in WIKILINK_RE.finditer(content):
        target = match.group(1).strip()
        display = match.group(2).strip() if match.group(2) else target
        links.append((target, display))
    return links

def extract_text(content):
    """Extract plain text from markdown (strip frontmatter, wikilinks, markdown)."""
    # Strip frontmatter
    text = FRONTMATTER_RE.sub('', content)
    # Convert wikilinks to display text
    text = WIKILINK_RE.sub(lambda m: m.group(2) or m.group(1), text)
    # Strip markdown formatting
    text = re.sub(r'[#*_`>\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

GRAPH_CACHE_FILE = ".bdh-graph-cache.json"

def build_graph(vault_root, use_cache=True):
    """Build the note graph: nodes with text, edges from wikilinks.
    
    With use_cache=True, loads from .bdh-graph-cache.json and only re-reads
    files whose mtime has changed since last cache. Falls back to full rebuild
    if cache is missing, corrupt, or file count has changed drastically.
    """
    cache_path = os.path.join(vault_root, GRAPH_CACHE_FILE)
    
    # Try loading cached graph
    cached = None
    if use_cache and os.path.isfile(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Graph cache corrupt, full rebuild: {e}")
            cached = None
    
    if cached:
        return _incremental_graph_update(vault_root, cached, cache_path)
    else:
        nodes, edges = _full_graph_build(vault_root)
        _save_graph_cache(vault_root, nodes, edges, cache_path)
        return nodes, edges


def _full_graph_build(vault_root):
    """Full graph build by walking the entire vault."""
    nodes = {}  # note_id -> {text, title, tags, path, mtime}
    edges = defaultdict(list)  # note_id -> [(target_id, ...), ...]
    
    for root, dirs, files in os.walk(vault_root):
        # Skip hidden dirs, .obsidian, raw/
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '.obsidian']
        for f in files:
            if not f.endswith('.md'):
                continue
            filepath = os.path.join(root, f)
            note_id = extract_note_id(filepath, vault_root)
            
            mtime = os.path.getmtime(filepath)
            with open(filepath, 'r', encoding='utf-8') as fh:
                content = fh.read()
            
            fm = parse_frontmatter(content)
            text = extract_text(content)
            links = extract_wikilinks(content)
            
            nodes[note_id] = {
                'id': note_id,
                'title': fm.get('title', os.path.basename(f)[:-3]),
                'tags': fm.get('tags', ''),
                'text': text,
                'path': filepath,
                'mtime': mtime,
            }
            
            for target, display in links:
                edges[note_id].append({
                    'target': target,
                    'display': display,
                })
    
    return nodes, dict(edges)


def _incremental_graph_update(vault_root, cached, cache_path):
    """Update cached graph by only re-reading changed files.
    
    Compares mtimes: if a file's mtime changed, re-read it. If a file
    disappeared, remove it. If new files appeared, add them. Also rebuild
    edges for any changed node (since wikilinks may have changed).
    """
    cached_nodes = cached.get('nodes', {})
    cached_edges = cached.get('edges', {})
    cached_mtimes = {nid: n.get('mtime', 0) for nid, n in cached_nodes.items()}
    
    # Walk vault to find current files
    current_files = {}  # note_id -> (filepath, mtime)
    for root, dirs, files in os.walk(vault_root):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '.obsidian']
        for f in files:
            if not f.endswith('.md'):
                continue
            filepath = os.path.join(root, f)
            note_id = extract_note_id(filepath, vault_root)
            mtime = os.path.getmtime(filepath)
            current_files[note_id] = (filepath, mtime)
    
    # Determine what changed
    current_ids = set(current_files.keys())
    cached_ids = set(cached_nodes.keys())
    
    new_notes = current_ids - cached_ids
    deleted_notes = cached_ids - current_ids
    potentially_changed = current_ids & cached_ids
    
    changed_notes = set()
    for nid in potentially_changed:
        if current_files[nid][1] != cached_mtimes.get(nid, 0):
            changed_notes.add(nid)
    
    total_changes = len(new_notes) + len(deleted_notes) + len(changed_notes)
    
    if total_changes == 0:
        # Nothing changed — return cached graph as-is
        logger.info(f"Graph cache hit (0 changes, {len(cached_nodes)} nodes)")
        print(f"   ✓ Graph cache hit — 0 changes, {len(cached_nodes)} neurons")
        return cached_nodes, cached_edges
    
    # If >50% of notes changed, do a full rebuild (more efficient)
    if total_changes > len(current_ids) * 0.5:
        logger.info(f"Too many changes ({total_changes}/{len(current_ids)}), full rebuild")
        print(f"   🔄 {total_changes} changes (>50%), full rebuild")
        nodes, edges = _full_graph_build(vault_root)
        _save_graph_cache(vault_root, nodes, edges, cache_path)
        return nodes, edges
    
    # Incremental update
    nodes = dict(cached_nodes)
    edges = dict(cached_edges)
    
    # Remove deleted notes
    for nid in deleted_notes:
        nodes.pop(nid, None)
        edges.pop(nid, None)
        # Also remove edges pointing to deleted notes
        for src, links in edges.items():
            edges[src] = [l for l in links if _resolve_target(l['target'], nodes) != nid
                         and l['target'] != nid]
    
    # Re-read changed and new notes
    to_read = new_notes | changed_notes
    for nid in to_read:
        filepath, mtime = current_files[nid]
        with open(filepath, 'r', encoding='utf-8') as fh:
            content = fh.read()
        
        fm = parse_frontmatter(content)
        text = extract_text(content)
        links = extract_wikilinks(content)
        
        nodes[nid] = {
            'id': nid,
            'title': fm.get('title', os.path.basename(filepath)[:-3]),
            'tags': fm.get('tags', ''),
            'text': text,
            'path': filepath,
            'mtime': mtime,
        }
        
        edges[nid] = [{'target': t, 'display': d} for t, d in links]
    
    print(f"   🔄 Incremental update: {len(new_notes)} new, {len(changed_notes)} changed, {len(deleted_notes)} deleted")
    logger.info(f"Graph incremental: +{len(new_notes)} ~{len(changed_notes)} -{len(deleted_notes)}")
    
    _save_graph_cache(vault_root, nodes, edges, cache_path)
    return nodes, edges


def _save_graph_cache(vault_root, nodes, edges, cache_path):
    """Save graph + mtimes to cache file for incremental rebuilds."""
    cache = {
        'nodes': nodes,
        'edges': edges,
        'cached_at': datetime.now().isoformat(),
        'vault_path': vault_root,
    }
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f)
    except IOError as e:
        logger.warning(f"Failed to save graph cache: {e}")

# ---------------------------------------------------------------------------
# Embedding (Ollama)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# BM25 Index for Hybrid Search (Phase 3.1)
# ---------------------------------------------------------------------------

class BM25Index:
    """In-memory BM25 index over note texts.

    Built once from the graph nodes, supports keyword scoring
    for hybrid search (vector + keyword combination).
    """

    def __init__(self, nodes, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = {}       # note_id -> token list
        self.doc_len = {}    # note_id -> int
        self.avg_dl = 0.0
        self.tf = {}         # note_id -> {term: freq}
        self.df = defaultdict(int)  # term -> doc count
        self.idf = {}        # term -> idf value
        self.N = 0
        self._build(nodes)

    @staticmethod
    def _tokenize(text):
        """Simple tokenizer: lowercase, split on non-alphanumeric."""
        return re.findall(r'[a-z0-9]+', text.lower())

    def _build(self, nodes):
        """Build the BM25 index from graph nodes."""
        self.N = len(nodes)
        if self.N == 0:
            return

        total_len = 0
        for note_id, node in nodes.items():
            text = node.get('text', '')
            tokens = self._tokenize(text)
            self.docs[note_id] = tokens
            self.doc_len[note_id] = len(tokens)
            total_len += len(tokens)

            # Term frequencies
            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            self.tf[note_id] = dict(tf)

            # Document frequencies
            for term in tf:
                self.df[term] += 1

        self.avg_dl = total_len / self.N if self.N > 0 else 0.0

        # Compute IDF (BM25+ variant: idf = ln(1 + (N - df + 0.5) / (df + 0.5)))
        for term, df in self.df.items():
            self.idf[term] = float(math.log(1.0 + (self.N - df + 0.5) / (df + 0.5)))

        logger.info(f"BM25 index built: {self.N} docs, {len(self.df)} unique terms, avg_dl={self.avg_dl:.1f}")

    def score(self, query, note_id):
        """Compute BM25 score for a single note against a query."""
        if note_id not in self.docs or self.N == 0:
            return 0.0

        query_terms = self._tokenize(query)
        if not query_terms:
            return 0.0

        dl = self.doc_len[note_id]
        score = 0.0
        tf_map = self.tf.get(note_id, {})

        for term in query_terms:
            if term not in self.idf:
                continue
            f = tf_map.get(term, 0)
            if f == 0:
                continue
            idf = self.idf[term]
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1.0))
            score += idf * f * (self.k1 + 1) / denom

        # Normalize to [0, 1] range (approximate)
        return min(score / 10.0, 1.0)

    def search(self, query, note_ids=None, top_k=None):
        """Score all (or subset of) notes against query.

        Returns list of (note_id, score) sorted descending.
        """
        results = []
        ids = note_ids if note_ids is not None else self.docs.keys()
        for nid in ids:
            s = self.score(query, nid)
            if s > 0:
                results.append((nid, s))
        results.sort(key=lambda x: -x[1])
        if top_k:
            results = results[:top_k]
        return results


# ---------------------------------------------------------------------------
# Adaptive Threshold (Phase 3.3)
# ---------------------------------------------------------------------------

def compute_adaptive_threshold(scores, floor=0.15):
    """Compute a dynamic threshold from score distribution.

    Uses max(percentile_75, mean + 1*std, floor) to adaptively
    select only genuinely relevant notes per query.

    Args:
        scores: list of float scores from attention
        floor: minimum threshold (never go below this)
    Returns:
        float threshold value
    """
    if not scores or len(scores) < 3:
        return floor

    import statistics
    sorted_scores = sorted(scores)
    n = len(sorted_scores)

    # Percentile 75 (Q3)
    q75_idx = int(n * 0.75)
    q75 = sorted_scores[min(q75_idx, n - 1)]

    # Mean + 1 std
    mean = statistics.mean(scores)
    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    mean_plus_std = mean + stdev

    threshold = max(q75, mean_plus_std, floor)
    logger.info(f"Adaptive threshold: Q75={q75:.3f}, mean+std={mean_plus_std:.3f}, floor={floor} → {threshold:.3f}")
    return threshold

# ---------------------------------------------------------------------------
# Attention: Embedding seed + Graph traversal
# ---------------------------------------------------------------------------

def attention(query, nodes, edges, collection, k=None, max_hop=None, bm25_index=None):
    """
    BDH-style attention: find relevant notes via ChromaDB KNN search (seed)
    + graph traversal (k-hop expansion). Returns active note set with scores.
    
    Uses ChromaDB for vector similarity (HNSW index, cosine space).
    Phase 3.1: Hybrid search — combines vector similarity with BM25 keyword score.
    Phase 3.3: Adaptive threshold — dynamic threshold from score distribution.
    
    Improvements over v1:
    - Hub dampening: high-degree nodes get activation scaled by 1/log(degree)
    - Neighbor cap: only top-k neighbors per hop (by embedding similarity to query)
    - Hybrid: α * vector_sim + β * BM25_score (if bm25_index provided)
    - Adaptive threshold: max(Q75, mean+1std, floor) instead of fixed 0.25
    """
    if k is None:
        k = CONFIG['seed_count']
    if max_hop is None:
        max_hop = CONFIG['max_hop']
    
    # Precompute degree for each node
    degree = defaultdict(int)
    for src, links in edges.items():
        degree[src] += len(links)
        for link in links:
            target = _resolve_target(link['target'], nodes)
            if target:
                degree[target] += 1
    
    # Step 1: Embedding seed — ChromaDB KNN search
    query_emb = get_embeddings([query])[0]
    if not query_emb:
        return {}
    
    # ChromaDB returns distances (lower = more similar), convert to similarity
    overfetch = min(k * 5, collection.count()) if collection.count() > 0 else k * 5
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=overfetch,
        include=['metadatas', 'distances'],
    )
    
    # Collect raw vector scores
    raw_vector_scores = {}
    if results['ids'] and results['ids'][0]:
        for i, note_id in enumerate(results['ids'][0]):
            dist = results['distances'][0][i]
            sim = max(0.0, 1.0 - dist)
            raw_vector_scores[note_id] = sim

    # Hybrid search: combine vector + BM25
    hybrid_enabled = CONFIG.get('hybrid_search', False) and bm25_index is not None
    alpha = CONFIG.get('hybrid_alpha', 0.7)
    beta = CONFIG.get('hybrid_beta', 0.3)
    
    if hybrid_enabled:
        # Get BM25 scores for the same candidate set
        candidate_ids = list(raw_vector_scores.keys())
        bm25_scores = {}
        for nid in candidate_ids:
            bm25_scores[nid] = bm25_index.score(query, nid)
        
        # Combine: α * vec + β * bm25 (both in [0,1])
        scores = {}
        for nid in candidate_ids:
            vec_s = raw_vector_scores[nid]
            bm_s = bm25_scores.get(nid, 0.0)
            combined = alpha * vec_s + beta * bm_s
            if combined > CONFIG['active_threshold']:
                # Hub dampening
                if CONFIG['hub_dampening'] and degree.get(nid, 0) > CONFIG['hub_degree_threshold']:
                    dampen = 1.0 / (1.0 + 0.15 * (degree[nid] - CONFIG['hub_degree_threshold']))
                    combined *= dampen
                scores[nid] = combined
    else:
        scores = {}
        for note_id, sim in raw_vector_scores.items():
            if sim > CONFIG['active_threshold']:
                if CONFIG['hub_dampening'] and degree.get(note_id, 0) > CONFIG['hub_degree_threshold']:
                    dampen = 1.0 / (1.0 + 0.15 * (degree[note_id] - CONFIG['hub_degree_threshold']))
                    sim *= dampen
                scores[note_id] = sim
    
    # Adaptive threshold (Phase 3.3)
    if CONFIG.get('adaptive_threshold', False) and len(scores) >= 5:
        threshold = compute_adaptive_threshold(
            list(scores.values()),
            floor=CONFIG.get('threshold_floor', 0.15),
        )
        scores = {nid: s for nid, s in scores.items() if s >= threshold}
    
    # Top-k seeds
    seeds = sorted(scores.items(), key=lambda x: -x[1])[:k]
    
    # Step 2: Graph traversal — expand from seeds via wikilinks
    active = dict(seeds)
    queue = deque()
    for note_id, score in seeds:
        queue.append((note_id, score, 0))
    
    while queue:
        current_id, score, hop = queue.popleft()
        if hop >= max_hop:
            continue
        
        # Get neighbors — use ChromaDB to rank them by similarity to query
        neighbors = []
        for edge in edges.get(current_id, []):
            target = edge['target']
            target_id = _resolve_target(target, nodes)
            if target_id is None:
                continue
            # Get similarity from ChromaDB if available, else use 0.1
            n_sim = scores.get(target_id, 0.1)
            neighbors.append((target_id, n_sim))
        
        # Cap: only top max_neighbors_per_hop by similarity
        neighbors.sort(key=lambda x: -x[1])
        neighbors = neighbors[:CONFIG['max_neighbors_per_hop']]
        
        for target_id, n_sim in neighbors:
            # Decay score by hop distance
            new_score = score * (0.5 ** (hop + 1))
            
            # Hub dampening for the target
            if CONFIG['hub_dampening'] and degree.get(target_id, 0) > CONFIG['hub_degree_threshold']:
                dampen = 1.0 / (1.0 + 0.15 * (degree[target_id] - CONFIG['hub_degree_threshold']))
                new_score *= dampen
            
            if target_id in active:
                active[target_id] = max(active[target_id], new_score)
            elif new_score > CONFIG['active_threshold']:
                active[target_id] = new_score
                queue.append((target_id, new_score, hop + 1))
    
    return active

def _resolve_target(target, nodes):
    """Resolve a wikilink target to an actual note ID in the graph."""
    # Direct match
    if target in nodes:
        return target
    # Try with wiki/ prefix or without
    for prefix in ['', 'wiki/', 'concepts/', 'entities/', 'comparisons/', 'queries/']:
        candidate = f"{prefix}{target}" if prefix else target
        if candidate in nodes:
            return candidate
    # Try matching by basename
    basename = os.path.basename(target)
    for note_id in nodes:
        if os.path.basename(note_id) == basename:
            return note_id
    return None

# ---------------------------------------------------------------------------
# Hebbian Update
# ---------------------------------------------------------------------------

def load_state(vault_root):
    """Load persisted BDH state (synaptic weights, co-activation history).
    Uses fcntl.flock for concurrency safety.
    """
    state_path = os.path.join(vault_root, STATE_FILE)
    lock_path = os.path.join(vault_root, LOCK_FILE)
    
    with open(lock_path, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            if os.path.isfile(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
            else:
                state = {
                    'synapses': {},  # "note_a|note_b" -> {weight, frequency, last_coactivated}
                    'created': datetime.now().isoformat(),
                    'updated': datetime.now().isoformat(),
                    'queries': 0,
                }
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)
    
    return state

def save_state(vault_root, state):
    """Persist BDH state. Uses fcntl.flock for concurrency safety."""
    state['updated'] = datetime.now().isoformat()
    state_path = os.path.join(vault_root, STATE_FILE)
    lock_path = os.path.join(vault_root, LOCK_FILE)
    
    with open(lock_path, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)

def hebbian_update(active_notes, state):
    """
    Hebbian update: reinforce links between co-activated notes.
    'Neurons that fire together, wire together.'
    """
    note_ids = sorted(active_notes.keys())
    now = datetime.now().isoformat()
    
    for i, a in enumerate(note_ids):
        for b in note_ids[i+1:]:
            key = f"{a}|{b}"
            if key not in state['synapses']:
                state['synapses'][key] = {
                    'weight': 0.0,
                    'frequency': 0,
                    'last_coactivated': None,
                    'created': now,
                }
            
            syn = state['synapses'][key]
            syn['frequency'] += 1
            syn['last_coactivated'] = now
            
            # Weight = alpha * normalized_freq + beta * recency
            # Recency: 1.0 if just activated, decays over time
            syn['weight'] = CONFIG['alpha'] * min(syn['frequency'] / 10.0, 1.0) + CONFIG['beta'] * 1.0
    
    # Decay unused synapses
    for key, syn in list(state['synapses'].items()):
        if key.split('|')[0] not in active_notes and key.split('|')[1] not in active_notes:
            syn['weight'] *= CONFIG['decay']
            if syn['weight'] < 0.01:
                del state['synapses'][key]
    
    state['queries'] += 1
    return state

# ---------------------------------------------------------------------------
# LLM Integration
# ---------------------------------------------------------------------------

def _build_llm_payload(query, active_notes, nodes, stream=False):
    """Build request payload + headers for the configured LLM provider."""
    context = format_context(active_notes, nodes)
    
    system_prompt = (
        "You are a knowledge assistant grounded in the user's Obsidian vault. "
        "Answer the user's question using ONLY the provided note context. "
        "If the context doesn't contain enough information, say so explicitly. "
        "Cite notes by name when you use information from them, e.g. '[from: Baby Dragon Hatchling]'. "
        "Keep responses concise and factual. Do not invent information not present in the context."
    )
    
    user_prompt = f"""## Activated Notes Context

{context}

## Question
{query}
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    provider = CONFIG.get('llm_provider', 'ollama')
    
    if provider == 'openrouter':
        # OpenAI-compatible format
        payload = {
            "model": CONFIG['llm_model'],
            "messages": messages,
            "stream": stream,
            "temperature": CONFIG['llm_temperature'],
            "max_tokens": CONFIG['llm_max_ctx'],
        }
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {CONFIG.get('openrouter_key', '')}",
            'HTTP-Referer': 'https://github.com/bdh-graph-harness',
            'X-Title': 'BDH Graph Harness',
        }
    else:
        # Ollama format
        payload = {
            "model": CONFIG['llm_model'],
            "messages": messages,
            "stream": stream,
            "options": {"temperature": CONFIG['llm_temperature'], "num_ctx": CONFIG['llm_max_ctx']},
        }
        headers = {'Content-Type': 'application/json'}
    
    return json.dumps(payload).encode(), headers


def _parse_llm_response(result, provider='ollama'):
    """Parse LLM response from either provider format."""
    if provider == 'openrouter':
        # OpenAI format: choices[0].message.content
        choices = result.get('choices', [])
        if choices:
            return choices[0].get('message', {}).get('content', '[no response]')
        return '[no response]'
    else:
        # Ollama format: message.content
        return result.get('message', {}).get('content', '[no response]')


def _parse_llm_stream_token(obj, provider='ollama'):
    """Parse a single streaming chunk from either provider."""
    if provider == 'openrouter':
        # OpenAI SSE: choices[0].delta.content
        choices = obj.get('choices', [])
        if choices:
            delta = choices[0].get('delta', {})
            content = delta.get('content', '')
            return content if content else None
        return None
    else:
        # Ollama: message.content
        if obj.get('done', False):
            return None
        return obj.get('message', {}).get('content', '')


def llm_respond(query, active_notes, nodes):
    """Send query + activated note context to LLM, get grounded response."""
    import urllib.request
    
    data, headers = _build_llm_payload(query, active_notes, nodes, stream=False)
    provider = CONFIG.get('llm_provider', 'ollama')
    
    def _llm_call():
        req = urllib.request.Request(OLLAMA_LLM_URL, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=CONFIG.get('llm_timeout', 300)) as resp:
            result = json.loads(resp.read())
            return _parse_llm_response(result, provider)
    
    try:
        raw = retry_with_backoff(_llm_call)
        # Sanitize: strip <pad> tokens and whitespace-only responses
        raw = re.sub(r'<pad>', '', raw).strip()
        return raw if raw else '[no response from LLM]'
    except Exception as e:
        return f"[LLM error: {e}]"


def llm_stream(query, active_notes, nodes):
    """Stream LLM response token-by-token.

    Supports both Ollama (NDJSON stream) and OpenRouter (SSE format).
    Yields token strings as they arrive from the LLM.

    Phase 3.2: Online plasticity — the caller can use the streamed tokens
    to update Hebbian state progressively as the LLM generates.
    """
    import urllib.request
    
    data, headers = _build_llm_payload(query, active_notes, nodes, stream=True)
    provider = CONFIG.get('llm_provider', 'ollama')
    
    req = urllib.request.Request(OLLAMA_LLM_URL, data=data, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=CONFIG.get('llm_timeout', 300)) as resp:
            buffer = b''
            for chunk in iter(lambda: resp.read(1), b''):
                buffer += chunk
                if buffer.endswith(b'\n'):
                    line = buffer.strip()
                    buffer = b''
                    if not line:
                        continue
                    
                    if provider == 'openrouter':
                        # OpenRouter SSE: lines start with "data: "
                        if line.startswith(b'data: '):
                            line = line[6:]
                        if line == b'[DONE]':
                            break
                        try:
                            obj = json.loads(line)
                            token = _parse_llm_stream_token(obj, provider)
                            if token and token != '<pad>':
                                yield token
                        except json.JSONDecodeError:
                            continue
                    else:
                        # Ollama NDJSON: one JSON object per line
                        try:
                            obj = json.loads(line)
                            if obj.get('done', False):
                                break
                            token = obj.get('message', {}).get('content', '')
                            if token and token != '<pad>':
                                yield token
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        yield f"[LLM stream error: {e}]"

# ---------------------------------------------------------------------------
# Neurogenesis
# ---------------------------------------------------------------------------

def extract_new_concepts(llm_response, query, active_notes, nodes):
    """Ask the LLM to identify concepts in its response that aren't in the vault."""
    import urllib.request
    
    # List existing note titles
    existing_titles = [node['title'] for node in nodes.values()]
    
    system_prompt = (
        "You are a concept extractor for a knowledge graph. "
        "Given an LLM response and a list of existing concept titles in the vault, "
        "identify any NEW concepts introduced in the response that are NOT in the existing list. "
        "For each new concept, provide: 1) a short title, 2) a 1-sentence definition. "
        "Return as JSON array: [{\"title\": \"...\", \"definition\": \"...\"}]. "
        "If no new concepts, return []."
    )
    
    user_prompt = f"""## Existing concept titles in vault
{json.dumps(existing_titles)}

## LLM response
{llm_response}

## Original query
{query}
"""
    
    data = json.dumps({
        "model": CONFIG['llm_model'],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        **({"format": "json", "options": {"temperature": 0.1, "num_ctx": CONFIG['llm_max_ctx']}}
           if CONFIG.get('llm_provider', 'ollama') == 'ollama'
           else {"temperature": 0.1, "max_tokens": CONFIG['llm_max_ctx'],
                 "response_format": {"type": "json_object"}}),
    }).encode()
    
    provider = CONFIG.get('llm_provider', 'ollama')
    headers = {'Content-Type': 'application/json'}
    if provider == 'openrouter':
        headers['Authorization'] = f"Bearer {CONFIG.get('openrouter_key', '')}"
        headers['HTTP-Referer'] = 'https://github.com/bdh-graph-harness'
        headers['X-Title'] = 'BDH Graph Harness'
    
    def _extract_call():
        req = urllib.request.Request(OLLAMA_LLM_URL, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            if provider == 'openrouter':
                choices = result.get('choices', [])
                content = choices[0].get('message', {}).get('content', '[]') if choices else '[]'
            else:
                content = result.get('message', {}).get('content', '[]')
            # Handle LLM returning text with embedded JSON
            content = content.strip()
            # Aggressively strip markdown code fences and prefixes
            import re as _re
            # Remove any ```json or ``` markers
            content = _re.sub(r'```(?:json)?', '', content)
            # Remove leading "json" prefix that gemma4 sometimes adds
            content = _re.sub(r'^json\s*', '', content)
            content = content.strip()
            # Try to find JSON array in the response
            if not content.startswith('['):
                match = _re.search(r'\[.*\]', content, _re.DOTALL)
                if match:
                    content = match.group(0)
                else:
                    return []
            concepts = json.loads(content)
            if isinstance(concepts, list):
                return concepts
            return []
    
    try:
        return retry_with_backoff(_extract_call)
    except Exception as e:
        print(f"  ⚠ Neurogenesis extraction error: {e}", file=sys.stderr)
        return []


def slugify(title):
    """Convert a title to a filename-safe slug."""
    import re
    slug = title.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[\s_]+', '-', slug)
    slug = re.sub(r'-+', '-', slug)
    return slug.strip('-')


def create_note(vault_root, title, definition, source_notes, query):
    """Create a new atomic note in the vault (neurogenesis)."""
    now = datetime.now().isoformat()
    slug = slugify(title)
    note_path = os.path.join(vault_root, CONFIG['neurogenesis_dir'], f"{slug}.md")
    
    # Don't overwrite existing notes
    if os.path.isfile(note_path):
        return None
    
    # Build wikilinks to source notes
    source_links = "\n".join(f"- [[concepts/{slugify(s)}|{s}]]" for s in source_notes[:3])
    
    content = f"""---
title: {title}
created: {now[:10]}
updated: {now[:10]}
type: concept
tags: [neurogenesis, auto-generated]
sources: []
confidence: low
---

# {title}

{definition}

## Origin
- **Created by:** BDH Graph Harness neurogenesis
- **Query:** {query[:200]}
- **Activated from:** {', '.join(source_notes[:3])}

## Links
{source_links}
"""
    
    os.makedirs(os.path.dirname(note_path), exist_ok=True)
    with open(note_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    note_id = f"{CONFIG['neurogenesis_dir']}/{slug}"
    
    # Update vault index and log
    update_vault_index(vault_root, note_id, title, section=CONFIG['neurogenesis_dir'])
    append_to_vault_log(vault_root, f"Neurogenesis: created '{title}' ({note_id}) from query '{query[:80]}'")
    
    return note_id


def update_vault_index(vault_root, note_id, title, section='concepts'):
    """Add a new note entry to the vault's wiki/index.md under the right section."""
    index_path = os.path.join(vault_root, 'wiki', 'index.md')
    
    # Read existing index
    content = ""
    if os.path.isfile(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            content = f.read()
    
    # Build the new entry line
    entry = f"- [[{note_id}|{title}]]"
    
    # Try to find a section header matching the section name
    section_header = f"## {section}"
    if section_header in content:
        # Insert after the section header, before the next section
        lines = content.split('\n')
        insert_idx = None
        for i, line in enumerate(lines):
            if line.strip() == section_header:
                insert_idx = i + 1
                break
        if insert_idx is not None:
            # Find the right spot (after existing entries in this section)
            while insert_idx < len(lines) and lines[insert_idx].strip().startswith('-'):
                insert_idx += 1
            lines.insert(insert_idx, entry)
            content = '\n'.join(lines)
        else:
            content += f"\n{section_header}\n{entry}\n"
    else:
        # No matching section; append a new section
        if content and not content.endswith('\n'):
            content += '\n'
        content += f"\n{section_header}\n{entry}\n"
    
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(content)


def append_to_vault_log(vault_root, action_text):
    """Append an action entry to the vault's wiki/log.md."""
    log_path = os.path.join(vault_root, 'wiki', 'log.md')
    now = datetime.now().isoformat()
    
    entry = f"- {now} — {action_text}\n"
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Read existing content
    existing = ""
    if os.path.isfile(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            existing = f.read()
    
    # If file doesn't start with a header, add one
    if not existing.strip():
        existing = "# BDH Graph Harness Log\n\n"
    elif not existing.startswith('#'):
        existing = "# BDH Graph Harness Log\n\n" + existing
    
    if not existing.endswith('\n'):
        existing += '\n'
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(existing + entry)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

def format_context(active_notes, nodes):
    """Format active notes as context for an LLM."""
    sorted_notes = sorted(active_notes.items(), key=lambda x: -x[1])
    parts = []
    for note_id, score in sorted_notes:
        node = nodes.get(note_id)
        if not node:
            continue
        parts.append(f"### {node['title']} (activation: {score:.3f})\n{node['text'][:300]}\n")
    return "\n---\n".join(parts)

def show_stats(nodes, edges, state):
    """Print graph statistics."""
    print(f"\n📊 BDH Graph Stats")
    print(f"  Neurons (notes): {len(nodes)}")
    print(f"  Synapses (links): {sum(len(e) for e in edges.values())}")
    print(f"  Avg degree: {sum(len(e) for e in edges.values()) / max(len(edges), 1):.1f}")
    
    # Degree distribution
    degrees = [len(e) for e in edges.values()]
    if degrees:
        degrees.sort(reverse=True)
        print(f"  Max degree: {degrees[0]}")
        print(f"  Top-5 hubs:")
        sorted_by_degree = sorted(edges.items(), key=lambda x: -len(x[1]))
        for note_id, links in sorted_by_degree[:5]:
            print(f"    {note_id} ({len(links)} links)")
    
    print(f"\n  Hebbian synapses: {len(state['synapses'])}")
    print(f"  Queries processed: {state['queries']}")
    
    # Show top Hebbian connections
    if state['synapses']:
        sorted_syn = sorted(state['synapses'].items(), key=lambda x: -x[1]['weight'])
        print(f"\n  Top Hebbian connections:")
        for key, syn in sorted_syn[:5]:
            a, b = key.split('|')
            print(f"    {a} ↔ {b} (w={syn['weight']:.3f}, freq={syn['frequency']})")

def show_hebbian(state):
    """Show all Hebbian synaptic connections."""
    if not state['synapses']:
        print("\n🔌 No Hebbian synapses yet. Run some queries first.")
        return
    
    print(f"\n🔌 Hebbian Synaptic State ({len(state['synapses'])} connections)")
    sorted_syn = sorted(state['synapses'].items(), key=lambda x: -x[1]['weight'])
    for key, syn in sorted_syn:
        a, b = key.split('|')
        print(f"  {a} ↔ {b}")
        print(f"    weight: {syn['weight']:.3f} | freq: {syn['frequency']} | last: {syn['last_coactivated']}")

def interactive_mode(vault_root, nodes, edges, collection, state, bm25_index=None):
    """Interactive REPL for querying the vault."""
    print(f"\n🐉 BDH Graph Harness — Interactive Mode")
    print(f"   Vault: {vault_root}")
    print(f"   Neurons: {len(nodes)} | Synapses: {sum(len(e) for e in edges.values())}")
    print(f"   Type 'exit' to quit, 'stats' for graph stats, 'hebbian' for synaptic state\n")
    
    while True:
        try:
            query = input("❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        
        if not query:
            continue
        if query.lower() == 'exit':
            break
        if query.lower() == 'stats':
            show_stats(nodes, edges, state)
            continue
        if query.lower() == 'hebbian':
            show_hebbian(state)
            continue
        
        # Attention (with hybrid search)
        active = attention(query, nodes, edges, collection, bm25_index=bm25_index)
        
        if not active:
            print("  No notes activated above threshold.\n")
            continue
        
        # Online plasticity: Hebbian update right after attention (Phase 3.2)
        if CONFIG.get('online_plasticity', True):
            state = hebbian_update(active, state)
            save_state(vault_root, state)
        
        # Show results
        sorted_active = sorted(active.items(), key=lambda x: -x[1])
        print(f"\n  🧠 Activated {len(active)} neurons:")
        for note_id, score in sorted_active:
            node = nodes.get(note_id)
            title = node['title'] if node else note_id
            print(f"    [{score:.3f}] {title}")
        
        # LLM response
        print(f"\n  🤖 LLM response:")
        print("  " + "-" * 60)
        response = llm_respond(query, active, nodes)
        for line in response.split('\n'):
            print(f"  {line}")
        print("  " + "-" * 60)
        
        # Neurogenesis
        active_titles = [nodes[nid]['title'] for nid in active if nid in nodes]
        new_concepts = extract_new_concepts(response, query, active, nodes)
        
        if new_concepts:
            print(f"\n  🧬 Neurogenesis: {len(new_concepts)} new concept(s) detected")
            for concept in new_concepts:
                title = concept.get('title', '').strip()
                definition = concept.get('definition', '').strip()
                if not title or not definition:
                    continue
                new_note_id = create_note(vault_root, title, definition, active_titles, query)
                if new_note_id:
                    print(f"    ✨ Created: {new_note_id} — {title}")
                else:
                    print(f"    ⊘ Skipped (already exists): {title}")
        else:
            print(f"\n  🧬 Neurogenesis: no new concepts")
        
        # Hebbian update
        state = hebbian_update(active, state)
        save_state(vault_root, state)
        print(f"  🔌 Hebbian update: {len(state['synapses'])} synapses\n")


# ---------------------------------------------------------------------------
# Visualization HTML page
# ---------------------------------------------------------------------------

def _viz_html(neuron_count, synapse_count, hebbian_count):
    """Return a self-contained HTML page with vis.js graph visualization."""
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🐉 BDH Graph Harness</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0d1117;
    color: #c9d1d9;
    font-family: system-ui, -apple-system, sans-serif;
    overflow: hidden;
    height: 100vh;
  }}
  #titlebar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 20px;
    background: #161b22;
    border-bottom: 1px solid #30363d;
    height: 52px;
  }}
  #titlebar h1 {{
    font-size: 18px;
    font-weight: 600;
    color: #f0883e;
  }}
  #titlebar .stats {{
    font-size: 13px;
    color: #8b949e;
  }}
  #titlebar .stats span {{ color: #58a6ff; font-weight: 600; }}
  #main {{
    display: flex;
    height: calc(100vh - 52px);
  }}
  #graph-container {{
    width: 70%;
    height: 100%;
    background: #0d1117;
  }}
  #side-panel {{
    width: 30%;
    height: 100%;
    background: #161b22;
    border-left: 1px solid #30363d;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}
  #query-section {{
    padding: 16px;
    border-bottom: 1px solid #30363d;
  }}
  #query-input {{
    width: 100%;
    padding: 10px 12px;
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    color: #c9d1d9;
    font-size: 14px;
    font-family: inherit;
    outline: none;
    transition: border-color 0.2s;
  }}
  #query-input:focus {{ border-color: #58a6ff; }}
  #query-btn {{
    width: 100%;
    margin-top: 10px;
    padding: 10px;
    background: #238636;
    border: none;
    border-radius: 6px;
    color: #fff;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
  }}
  #query-btn:hover {{ background: #2ea043; }}
  #query-btn:disabled {{ background: #21262d; color: #6e7681; cursor: not-allowed; }}
  #response-section {{
    padding: 16px;
    border-bottom: 1px solid #30363d;
    max-height: 200px;
    overflow-y: auto;
  }}
  #response-section h3, #activated-section h3, #stats-section h3 {{
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #8b949e;
    margin-bottom: 10px;
  }}
  #response-text {{
    font-size: 13px;
    line-height: 1.5;
    color: #c9d1d9;
  }}
  #activated-section {{
    padding: 16px;
    flex: 1;
    overflow-y: auto;
    border-bottom: 1px solid #30363d;
  }}
  #activated-list {{ list-style: none; }}
  #activated-list li {{
    padding: 8px 10px;
    margin-bottom: 6px;
    background: #0d1117;
    border-radius: 6px;
    border-left: 3px solid #f0883e;
    font-size: 13px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: all 0.3s;
  }}
  #activated-list li.seed {{ border-left-color: #58a6ff; }}
  #activated-list li .score {{
    color: #8b949e;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }}
  #activated-list .empty {{
    color: #6e7681;
    font-style: italic;
    padding: 8px 0;
  }}
  #stats-section {{
    padding: 16px;
    font-size: 13px;
  }}
  #stats-section .stat-row {{
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    color: #8b949e;
  }}
  #stats-section .stat-row .val {{ color: #58a6ff; font-weight: 600; }}
  #status-indicator {{
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-left: 8px;
    background: #6e7681;
    transition: background 0.3s;
  }}
  #status-indicator.connected {{ background: #2ea043; }}
  #status-indicator.active {{ background: #f0883e; }}

  /* ======== MOBILE RESPONSIVE ======== */
  @media (max-width: 768px) {{
    #titlebar {{
      padding: 8px 12px;
      height: 44px;
      flex-direction: row;
      flex-wrap: wrap;
    }}
    #titlebar h1 {{ font-size: 15px; }}
    #titlebar .stats {{ font-size: 11px; }}
    #main {{
      flex-direction: column;
      height: calc(100vh - 44px - 36px);  /* minus tabbar */
    }}
    #graph-container {{
      width: 100%;
      height: 100%;
      flex: 1;
      display: block;
    }}
    #side-panel {{
      width: 100%;
      height: 100%;
      flex: 1;
      border-left: none;
      border-top: 1px solid #30363d;
      display: none;  /* hidden by default, toggle via tab */
    }}
    /* Mobile tab bar */
    #mobile-tabs {{
      display: flex;
      height: 36px;
      background: #161b22;
      border-bottom: 1px solid #30363d;
    }}
    #mobile-tabs .tab {{
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 13px;
      font-weight: 600;
      color: #6e7681;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: all 0.2s;
      user-select: none;
      -webkit-user-select: none;
    }}
    #mobile-tabs .tab.active {{
      color: #f0883e;
      border-bottom-color: #f0883e;
    }}
    /* When graph tab active on mobile */
    body.graph-tab #graph-container {{ display: block; flex: 1; }}
    body.graph-tab #side-panel {{ display: none; }}
    /* When panel tab active on mobile */
    body.panel-tab #graph-container {{ display: none; }}
    body.panel-tab #side-panel {{ display: flex; flex-direction: column; flex: 1; overflow-y: auto; }}
    body.panel-tab #side-panel #activated-section {{ flex: 0 1 auto; max-height: 250px; }}
    body.panel-tab #side-panel #response-section {{ max-height: none; flex: 0 1 auto; }}
    /* Larger touch targets */
    #query-input {{ font-size: 16px; padding: 12px 14px; }}
    #query-btn {{ padding: 14px; font-size: 15px; }}
    #activated-list li {{ padding: 12px 10px; font-size: 14px; }}
  }}
  /* Hide mobile tabs on desktop */
  @media (min-width: 769px) {{
    #mobile-tabs {{ display: none; }}
  }}
</style>
</head>
<body class="graph-tab">
<div id="titlebar">
  <h1>🐉 BDH Graph Harness</h1>
  <div class="stats">
    Neurons: <span id="stat-neurons">{neuron_count}</span>
    | Synapses: <span id="stat-synapses">{synapse_count}</span>
    | Hebbian: <span id="stat-hebbian">{hebbian_count}</span>
    <span id="status-indicator"></span>
  </div>
</div>
<div id="mobile-tabs">
  <div class="tab active" onclick="switchTab('graph-tab')">🗺️ Graph</div>
  <div class="tab" onclick="switchTab('panel-tab')">📋 Panel</div>
</div>
<div id="main">
  <div id="graph-container"></div>
  <div id="side-panel">
    <div id="query-section">
      <input type="text" id="query-input" placeholder="Enter a query..." autocomplete="off">
      <button id="query-btn" onclick="sendQuery()">Query</button>
    </div>
    <div id="response-section">
      <h3>Response</h3>
      <div id="response-text">—</div>
    </div>
    <div id="activated-section">
      <h3>Activated Notes</h3>
      <ul id="activated-list"><div class="empty">No activations yet</div></ul>
    </div>
    <div id="stats-section">
      <h3>Stats</h3>
      <div class="stat-row"><span>Queries processed</span><span class="val" id="stat-queries">0</span></div>
      <div class="stat-row"><span>Last query</span><span class="val" id="stat-last">—</span></div>
    </div>
  </div>
</div>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<script>
const COLORS = {{
  inactive: '#6e7681',
  activated: '#f0883e',
  seed: '#58a6ff',
  edgeWikilink: '#30363d',
  edgeHebbian: '#58a6ff',
  edgeHebbianPulse: '#39d353',  // green pulse for strengthened synapses
  bg: '#0d1117',
}};
let network = null;
let nodesDS = null;
let edgesDS = null;
let hebbianMap = {{}};  // "a|b" -> weight
let allNodeIds = [];
let allEdgeIds = [];

function initNetwork(graphData) {{
  allNodeIds = graphData.nodes.map(n => n.id);
  const visNodes = graphData.nodes.map(n => ({{
    id: n.id,
    label: n.title,
    title: n.title + (n.tags ? '\\\\n[' + (Array.isArray(n.tags) ? n.tags.join(', ') : n.tags) + ']' : ''),
    group: 'inactive',
  }}));

  const edgeSet = new Set();
  const visEdges = [];
  graphData.edges.forEach(e => {{
    const eid = e.source + '→' + e.target;
    if (edgeSet.has(eid)) return;
    edgeSet.add(eid);
    visEdges.push({{
      id: eid,
      from: e.source,
      to: e.target,
      color: {{ color: COLORS.edgeWikilink, highlight: COLORS.edgeWikilink }},
      width: 0.5,
    }});
  }});

  // Add hebbian edges
  hebbianMap = {{}};
  graphData.hebbian.forEach(h => {{
    const key = h.note_a + '|' + h.note_b;
    hebbianMap[key] = h.weight;
    const eid = 'hebb_' + h.note_a + '→' + h.note_b;
    if (edgeSet.has(eid)) return;
    edgeSet.add(eid);
    const w = Math.min(1 + h.weight * 3, 5);
    visEdges.push({{
      id: eid,
      from: h.note_a,
      to: h.note_b,
      color: {{ color: COLORS.edgeHebbian, highlight: COLORS.edgeHebbian }},
      width: w,
      dashes: false,
    }});
  }});

  allEdgeIds = Array.from(edgeSet);

  nodesDS = new vis.DataSet(visNodes);
  edgesDS = new vis.DataSet(visEdges);

  const container = document.getElementById('graph-container');
  const data = {{ nodes: nodesDS, edges: edgesDS }};
  const options = {{
    nodes: {{
      shape: 'dot',
      size: 8,
      font: {{ color: '#c9d1d9', size: 12, face: 'system-ui' }},
      borderWidth: 1,
      borderWidthSelected: 2,
    }},
    edges: {{
      smooth: {{ type: 'continuous', roundness: 0.15 }},
      selectionWidth: 0,
    }},
    groups: {{
      inactive: {{ color: {{ background: COLORS.inactive, border: COLORS.inactive }} }},
      activated: {{ color: {{ background: COLORS.activated, border: COLORS.activated }} }},
      seed: {{ color: {{ background: COLORS.seed, border: COLORS.seed }} }},
    }},
    physics: {{
      solver: 'barnesHut',
      barnesHut: {{ gravitationalConstant: -3000, springLength: 120, springConstant: 0.04 }},
      stabilization: {{ iterations: 100, updateInterval: 25 }},
    }},
    interaction: {{ hover: true, tooltipDelay: 150 }},
  }};

  if (network) network.destroy();
  network = new vis.Network(container, data, options);

  document.getElementById('stat-neurons').textContent = graphData.stats.neurons;
  document.getElementById('stat-synapses').textContent = graphData.stats.synapses;
  document.getElementById('stat-hebbian').textContent = graphData.stats.hebbian_synapses;
}}

function animateNodeColor(nodeId, newGroup) {{
  if (!nodesDS) return;
  nodesDS.update({{ id: nodeId, group: newGroup }});
}}

function handleActivation(event) {{
  const activated = event.activated_notes || [];
  const activatedIds = new Set(activated.map(n => n.id));
  const seedId = activated.length > 0 ? activated[0].id : null;

  // Dim all nodes first
  allNodeIds.forEach(id => {{
    if (nodesDS.get(id)) {{
      nodesDS.update({{ id, group: 'inactive', opacity: activatedIds.size > 0 ? 0.3 : 1.0 }});
    }}
  }});

  // Light up activated nodes with staggered timing for cascade effect
  activated.forEach((note, i) => {{
    setTimeout(() => {{
      const group = note.id === seedId ? 'seed' : 'activated';
      if (nodesDS.get(note.id)) {{
        nodesDS.update({{ id: note.id, group: group, opacity: 1.0 }});
      }}
    }}, i * 80);
  }});

  // Highlight edges between activated nodes + pulse Hebbian synapses
  allEdgeIds.forEach(eid => {{
    const edge = edgesDS.get(eid);
    if (!edge) return;
    const bothActive = activatedIds.has(edge.from) && activatedIds.has(edge.to);
    if (bothActive) {{
      edgesDS.update({{ id: eid, opacity: 1.0 }});
    }} else {{
      edgesDS.update({{ id: eid, opacity: 0.15 }});
    }}
  }});

  // Pulse Hebbian synapses that were strengthened in this query
  const hebbianUpdates = event.hebbian_updates || [];
  hebbianUpdates.forEach((h, idx) => {{
    const parts = h.pair.split('|');
    if (parts.length !== 2) return;
    const [a, b] = parts;
    // Update hebbianMap with new weight
    hebbianMap[h.pair] = h.weight;

    // Find the Hebbian edge in vis (could be a→b or b→a)
    const eid1 = 'hebb_' + a + '→' + b;
    const eid2 = 'hebb_' + b + '→' + a;
    const eid = edgesDS.get(eid1) ? eid1 : (edgesDS.get(eid2) ? eid2 : null);

    // Stagger pulses so they cascade instead of all firing at once
    const delay = idx * 120;
    const newWidth = Math.min(1 + h.weight * 3, 5);

    if (eid) {{
      // Existing edge: multi-step fade from pulse → settle
      const pulseSteps = [
        {{ w: newWidth + 5, c: COLORS.edgeHebbianPulse, t: 0 }},
        {{ w: newWidth + 3, c: COLORS.edgeHebbianPulse, t: 600 }},
        {{ w: newWidth + 1.5, c: COLORS.edgeHebbian,    t: 1400 }},
        {{ w: newWidth,      c: COLORS.edgeHebbian,    t: 2500 }},
      ];
      pulseSteps.forEach(s => {{
        setTimeout(() => {{
          if (edgesDS.get(eid)) {{
            edgesDS.update({{
              id: eid,
              color: {{ color: s.c, highlight: s.c }},
              width: s.w,
              opacity: 1.0,
            }});
          }}
        }}, delay + s.t);
      }});
    }} else if (activatedIds.has(a) || activatedIds.has(b)) {{
      // New Hebbian synapse — create edge if both nodes exist
      if (nodesDS.get(a) && nodesDS.get(b)) {{
        const newEid = 'hebb_' + a + '→' + b;
        if (!edgesDS.get(newEid)) {{
          const pulseStepsNew = [
            {{ w: newWidth + 5,   c: COLORS.edgeHebbianPulse, t: 0 }},
            {{ w: newWidth + 3,   c: COLORS.edgeHebbianPulse, t: 600 }},
            {{ w: newWidth + 1.5, c: COLORS.edgeHebbian,    t: 1400 }},
            {{ w: newWidth,       c: COLORS.edgeHebbian,    t: 2500 }},
          ];
          pulseStepsNew.forEach(s => {{
            setTimeout(() => {{
              if (edgesDS.get(newEid)) {{
                edgesDS.update({{
                  id: newEid,
                  color: {{ color: s.c, highlight: s.c }},
                  width: s.w,
                  opacity: 1.0,
                }});
              }} else if (s.t === 0) {{
                // First step — create the edge
                edgesDS.add({{
                  id: newEid,
                  from: a,
                  to: b,
                  color: {{ color: s.c, highlight: s.c }},
                  width: s.w,
                  dashes: false,
                  opacity: 1.0,
                }});
                allEdgeIds.push(newEid);
              }}
            }}, delay + s.t);
          }});
        }}
      }}
    }}
  }});

  // Reset non-Hebbian edge opacity after 4s
  setTimeout(() => {{
    allEdgeIds.forEach(eid => {{
      const edge = edgesDS.get(eid);
      if (!edge) return;
      // Don't reset Hebbian edges — they keep their weight-based width
      if (!eid.startsWith('hebb_')) {{
        edgesDS.update({{ id: eid, opacity: 1.0 }});
      }}
    }});
  }}, 4000);

  // Update side panel
  const listEl = document.getElementById('activated-list');
  listEl.innerHTML = '';
  if (activated.length === 0) {{
    listEl.innerHTML = '<div class="empty">No notes activated</div>';
  }} else {{
    activated.forEach(note => {{
      const li = document.createElement('li');
      if (note.id === seedId) li.className = 'seed';
      li.innerHTML = '<span>' + escapeHtml(note.title) + '</span><span class="score">' + note.score.toFixed(4) + '</span>';
      listEl.appendChild(li);
    }});
  }}

  // Update stats
  document.getElementById('stat-last').textContent = event.query || '—';
  if (hebbianUpdates.length > 0) {{
    const hebbEl = document.getElementById('stat-hebbian');
    if (hebbEl) {{
      // Flash the hebbian count
      hebbEl.textContent = hebbianUpdates.length;
      hebbEl.style.color = COLORS.edgeHebbianPulse;
      setTimeout(() => {{ hebbEl.style.color = ''; }}, 1000);
    }}
  }}

  // Show indicator pulse
  const ind = document.getElementById('status-indicator');
  ind.classList.add('active');
  setTimeout(() => ind.classList.remove('active'), 1000);
}}

function escapeHtml(s) {{
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}}

function sendQuery() {{
  const input = document.getElementById('query-input');
  const btn = document.getElementById('query-btn');
  const query = input.value.trim();
  if (!query) return;
  btn.disabled = true;
  btn.textContent = 'Processing...';
  fetch('/api/query', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ query }}),
  }}).then(r => r.json()).then(data => {{
    if (data.error) {{
      document.getElementById('response-text').textContent = 'Error: ' + data.error;
    }} else {{
      if (data.response) {{
        document.getElementById('response-text').textContent = data.response;
      }}
      // Update activation panel from HTTP response (works even if WS is down)
      if (data.activated_notes) {{
        handleActivation({{ type: 'activation', query: query, activated_notes: data.activated_notes, hebbian_updates: data.hebbian_updates || [] }});
      }}
    }}
    btn.disabled = false;
    btn.textContent = 'Query';
  }}).catch(err => {{
    btn.disabled = false;
    btn.textContent = 'Query';
    document.getElementById('response-text').textContent = 'Request failed: ' + err;
  }});
}}

// WebSocket connection
function connectWS() {{
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(proto + '//' + location.host + '/ws');
  const ind = document.getElementById('status-indicator');

  ws.onopen = () => {{ ind.classList.add('connected'); }};

  ws.onmessage = (msg) => {{
    try {{
      const event = JSON.parse(msg.data);
      if (event.type === 'graph') {{
        initNetwork(event);
      }} else if (event.type === 'activation') {{
        handleActivation(event);
      }}
    }} catch (e) {{ console.error('Parse error:', e); }}
  }};

  ws.onclose = () => {{
    ind.classList.remove('connected');
    setTimeout(connectWS, 2000);  // auto-reconnect
  }};

  ws.onerror = () => {{ ws.close(); }};
}}

// Mobile tab switching
function switchTab(tabClass) {{
  document.body.className = tabClass;
  document.querySelectorAll('#mobile-tabs .tab').forEach(t => t.classList.remove('active'));
  const labels = {{ 'graph-tab': 0, 'panel-tab': 1 }};
  const tabs = document.querySelectorAll('#mobile-tabs .tab');
  if (tabs[labels[tabClass]]) tabs[labels[tabClass]].classList.add('active');
  // Force vis.js to redraw when switching back to graph
  if (tabClass === 'graph-tab' && network) setTimeout(() => network.redraw(), 50);
}}

// Enter key sends query
document.getElementById('query-input').addEventListener('keydown', (e) => {{
  if (e.key === 'Enter') sendQuery();
}});

// Init
connectWS();
</script>
</body>
</html>'''


# ---------------------------------------------------------------------------
# API Server (aiohttp)
# ---------------------------------------------------------------------------

def start_api_server(config, nodes, edges, collection, state):
    """Start an aiohttp web API server for the BDH graph harness."""
    from aiohttp import web

    # Monkeypatch tcp_keepalive to avoid OSError [Errno 22] on macOS/Tailscale
    # This is a known aiohttp bug on macOS where setsockopt(SO_KEEPALIVE) fails
    try:
        import aiohttp.tcp_helpers as _tcp_helpers
        _orig = _tcp_helpers.tcp_keepalive
        def _safe_keepalive(transport):
            try:
                _orig(transport)
            except OSError:
                pass  # Ignore on macOS/Tailscale
        _tcp_helpers.tcp_keepalive = _safe_keepalive
    except Exception:
        pass

    # Build BM25 index for hybrid search (Phase 3.1)
    bm25_idx = None
    if config.get('hybrid_search', False):
        print("📊 Building BM25 index for hybrid search...")
        bm25_idx = BM25Index(
            nodes,
            k1=config.get('bm25_k1', 1.5),
            b=config.get('bm25_b', 0.75),
        )
        print(f"   ✓ BM25: {bm25_idx.N} docs, {len(bm25_idx.df)} terms")

    # Shared mutable state container
    app_state = {
        'nodes': nodes,
        'edges': edges,
        'collection': collection,
        'state': state,
        'config': config,
        'bm25_index': bm25_idx,
    }

    # Global set of connected WebSocket clients
    ws_clients = set()

    async def broadcast_activation(event):
        """Broadcast an activation event to all connected WebSocket clients."""
        msg = json.dumps(event)
        dead = []
        for ws in ws_clients:
            try:
                await ws.send_str(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            ws_clients.discard(ws)

    async def websocket_handler(request):
        """Handle WebSocket connections for real-time graph visualization."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        ws_clients.add(ws)

        # Send full graph on connect
        n = app_state['nodes']
        e = app_state['edges']
        s = app_state['state']

        node_list = []
        for note_id, node in n.items():
            node_list.append({
                'id': note_id,
                'title': node['title'],
                'tags': node.get('tags', []),
            })

        edge_list = []
        for src, links in e.items():
            for link in links:
                target_id = _resolve_target(link['target'], n)
                if target_id:
                    edge_list.append({
                        'source': src,
                        'target': target_id,
                        'display': link.get('display', ''),
                    })

        hebbian_list = []
        for key, syn in s['synapses'].items():
            a, b = key.split('|')
            hebbian_list.append({
                'note_a': a,
                'note_b': b,
                'weight': syn['weight'],
                'frequency': syn.get('frequency', 0),
            })

        init_msg = {
            'type': 'graph',
            'nodes': node_list,
            'edges': edge_list,
            'hebbian': hebbian_list,
            'stats': {
                'neurons': len(n),
                'synapses': sum(len(links) for links in e.values()),
                'hebbian_synapses': len(s['synapses']),
            },
        }
        await ws.send_str(json.dumps(init_msg))

        # Listen for messages (ping/pong keepalive)
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.ERROR:
                    break
        finally:
            ws_clients.discard(ws)

        return ws

    async def index(request):
        """Serve the vis.js visualization page."""
        n = app_state['nodes']
        e = app_state['edges']
        s = app_state['state']
        html = _viz_html(len(n), sum(len(links) for links in e.values()), len(s['synapses']))
        return web.Response(text=html, content_type='text/html')
    
    async def api_stats(request):
        """Return graph stats + Hebbian summary as JSON."""
        s = app_state['state']
        n = app_state['nodes']
        e = app_state['edges']
        stats = {
            'neurons': len(n),
            'synapses': sum(len(links) for links in e.values()),
            'avg_degree': sum(len(links) for links in e.values()) / max(len(e), 1),
            'hebbian_synapses': len(s['synapses']),
            'queries_processed': s.get('queries', 0),
            'top_hebbian': [],
        }
        if s['synapses']:
            sorted_syn = sorted(s['synapses'].items(), key=lambda x: -x[1]['weight'])[:10]
            stats['top_hebbian'] = [
                {'pair': key, 'weight': syn['weight'], 'frequency': syn['frequency']}
                for key, syn in sorted_syn
            ]
        return web.json_response(stats)
    
    async def api_graph(request):
        """Return nodes and edges as JSON for visualization."""
        n = app_state['nodes']
        e = app_state['edges']
        
        node_list = []
        for note_id, node in n.items():
            node_list.append({
                'id': note_id,
                'title': node['title'],
                'tags': node['tags'],
            })
        
        edge_list = []
        for src, links in e.items():
            for link in links:
                target_id = _resolve_target(link['target'], n)
                if target_id:
                    edge_list.append({
                        'source': src,
                        'target': target_id,
                        'display': link['display'],
                    })
        
        return web.json_response({'nodes': node_list, 'edges': edge_list})
    
    async def api_hebbian(request):
        """Return Hebbian synaptic state as JSON."""
        s = app_state['state']
        synapses = []
        for key, syn in sorted(s['synapses'].items(), key=lambda x: -x[1]['weight']):
            a, b = key.split('|')
            synapses.append({
                'note_a': a,
                'note_b': b,
                'weight': syn['weight'],
                'frequency': syn['frequency'],
                'last_coactivated': syn['last_coactivated'],
            })
        return web.json_response({
            'total': len(s['synapses']),
            'queries': s.get('queries', 0),
            'synapses': synapses,
        })
    
    async def api_query(request):
        """Accept {"query": "..."} and run the full BDH pipeline.
        
        Phase 3.2: Online plasticity — Hebbian update happens right after
        attention (not after LLM), so synaptic state reflects what was
        activated, not what the LLM happened to use.
        """
        try:
            data = await request.json()
        except Exception:
            return web.json_response({'error': 'Invalid JSON body'}, status=400)
        
        query = data.get('query', '').strip()
        if not query:
            return web.json_response({'error': 'Missing "query" field'}, status=400)
        
        n = app_state['nodes']
        e = app_state['edges']
        coll = app_state['collection']
        bm25 = app_state.get('bm25_index')
        
        # Attention (with hybrid search if enabled)
        active = attention(query, n, e, coll, bm25_index=bm25)
        
        activated_notes = []
        if active:
            for note_id, score in sorted(active.items(), key=lambda x: -x[1]):
                node = n.get(note_id)
                activated_notes.append({
                    'id': note_id,
                    'title': node['title'] if node else note_id,
                    'score': round(score, 4),
                })
        
        # Online plasticity: Hebbian update immediately after attention
        if CONFIG.get('online_plasticity', True):
            app_state['state'] = hebbian_update(active, app_state['state'])
            save_state(config['vault_path'], app_state['state'])
        
        # LLM response
        response_text = llm_respond(query, active, n)
        
        # Neurogenesis
        new_concepts_list = []
        if CONFIG.get('neurogenesis_enabled', True):
            active_titles = [n[nid]['title'] for nid in active if nid in n]
            new_concepts = extract_new_concepts(response_text, query, active, n)
            for concept in new_concepts:
                title = concept.get('title', '').strip()
                definition = concept.get('definition', '').strip()
                if not title or not definition:
                    continue
                vault_root = config['vault_path']
                new_note_id = create_note(vault_root, title, definition, active_titles, query)
                if new_note_id:
                    new_concepts_list.append({'id': new_note_id, 'title': title})
        
        # Collect hebbian synapses for broadcast
        hebbian_updates = []
        for key, syn in app_state['state']['synapses'].items():
            hebbian_updates.append({
                'pair': key,
                'weight': syn['weight'],
                'frequency': syn.get('frequency', 0),
            })

        # Broadcast activation event to WebSocket clients
        activation_event = {
            'type': 'activation',
            'query': query,
            'activated_notes': activated_notes,
            'hebbian_updates': hebbian_updates,
            'new_concepts': new_concepts_list,
            'timestamp': datetime.now().isoformat(),
        }
        await broadcast_activation(activation_event)

        return web.json_response({
            'response': response_text,
            'activated_notes': activated_notes,
            'new_concepts': new_concepts_list,
            'hebbian_synapses': len(app_state['state']['synapses']),
            'hebbian_updates': hebbian_updates,
        })
    
    async def api_stream(request):
        """Streaming query endpoint using Server-Sent Events.
        
        Phase 3.2: Online plasticity with streaming.
        Streams tokens as they arrive from the LLM, with Hebbian update
        performed right after attention (before streaming starts).
        
        SSE format: data: {json}\\n\\n
        """
        try:
            data = await request.json()
        except Exception:
            return web.json_response({'error': 'Invalid JSON body'}, status=400)
        
        query = data.get('query', '').strip()
        if not query:
            return web.json_response({'error': 'Missing "query" field'}, status=400)
        
        n = app_state['nodes']
        e = app_state['edges']
        coll = app_state['collection']
        bm25 = app_state.get('bm25_index')
        
        # Attention with hybrid search
        active = attention(query, n, e, coll, bm25_index=bm25)
        
        activated_notes = []
        if active:
            for note_id, score in sorted(active.items(), key=lambda x: -x[1]):
                node = n.get(note_id)
                activated_notes.append({
                    'id': note_id,
                    'title': node['title'] if node else note_id,
                    'score': round(score, 4),
                })
        
        # Online plasticity: Hebbian update right after attention
        if CONFIG.get('online_plasticity', True):
            app_state['state'] = hebbian_update(active, app_state['state'])
            save_state(config['vault_path'], app_state['state'])
        
        # Broadcast activation to WebSocket clients
        activation_event = {
            'type': 'activation',
            'query': query,
            'activated_notes': activated_notes,
            'hebbian_updates': [
                {'pair': k, 'weight': s['weight'], 'frequency': s.get('frequency', 0)}
                for k, s in app_state['state']['synapses'].items()
            ],
            'timestamp': datetime.now().isoformat(),
        }
        await broadcast_activation(activation_event)
        
        # Stream LLM response via SSE
        resp = web.StreamResponse(
            status=200,
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            },
        )
        await resp.prepare(request)
        
        # Send activation info first
        init_data = json.dumps({
            'type': 'activation',
            'activated_notes': activated_notes,
            'hebbian_synapses': len(app_state['state']['synapses']),
        })
        await resp.write(f"data: {init_data}\n\n".encode())
        
        # Stream tokens
        full_response = []
        try:
            for token in llm_stream(query, active, n):
                full_response.append(token)
                token_data = json.dumps({'type': 'token', 'content': token})
                await resp.write(f"data: {token_data}\n\n".encode())
        except Exception as e:
            logger.warning(f"Stream interrupted: {e}")
        
        # Neurogenesis on the full response
        new_concepts_list = []
        if CONFIG.get('neurogenesis_enabled', True):
            response_text = ''.join(full_response)
            active_titles = [n[nid]['title'] for nid in active if nid in n]
            new_concepts = extract_new_concepts(response_text, query, active, n)
            for concept in new_concepts:
                title = concept.get('title', '').strip()
                definition = concept.get('definition', '').strip()
                if not title or not definition:
                    continue
                vault_root = config['vault_path']
                new_note_id = create_note(vault_root, title, definition, active_titles, query)
                if new_note_id:
                    new_concepts_list.append({'id': new_note_id, 'title': title})
        
        # Send final event with new concepts
        done_data = json.dumps({
            'type': 'done',
            'new_concepts': new_concepts_list,
            'hebbian_synapses': len(app_state['state']['synapses']),
        })
        await resp.write(f"data: {done_data}\n\n".encode())
        await resp.write(b'data: [DONE]\n\n')
        
        return resp
    
    async def api_refresh(request):
        """Force refresh all embeddings."""
        n = app_state['nodes']
        vault_root = config['vault_path']
        coll = compute_all_embeddings(n, vault_root, force_refresh=True)
        app_state['collection'] = coll
        return web.json_response({'status': 'ok', 'embeddings': coll.count()})
    
    app = web.Application()
    app.router.add_get('/', index)
    app.router.add_get('/ws', websocket_handler)
    app.router.add_get('/api/stats', api_stats)
    app.router.add_get('/api/graph', api_graph)
    app.router.add_get('/api/hebbian', api_hebbian)
    app.router.add_post('/api/query', api_query)
    app.router.add_post('/api/stream', api_stream)
    app.router.add_post('/api/refresh', api_refresh)
    
    host = config['api_host']
    port = config['api_port']
    print(f"🌐 BDH Graph Harness API server starting on http://{host}:{port}")
    web.run_app(app, host=host, port=port)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='BDH Graph Harness — neural vault knowledge engine',
    )
    parser.add_argument('query', nargs='*', help='Query text (positional)')
    parser.add_argument('--config', default=None, help='Path to config YAML file')
    parser.add_argument('--vault', default=None, help='Vault root path (overrides config)')
    parser.add_argument('--serve', action='store_true', help='Start the API server')
    parser.add_argument('--stats', action='store_true', help='Show graph statistics')
    parser.add_argument('--hebbian-show', action='store_true', help='Show Hebbian synaptic state')
    parser.add_argument('--refresh-embeddings', action='store_true',
                        help='Force refresh all embeddings in ChromaDB')
    parser.add_argument('--interactive', action='store_true', help='Interactive REPL mode')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force full graph rebuild (skip cache)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine vault root
    vault_root = args.vault or config['vault_path']
    vault_root = os.path.expanduser(vault_root)
    if not os.path.isdir(vault_root):
        print(f"Error: vault path '{vault_root}' not found")
        sys.exit(1)
    
    print(f"🐉 BDH Graph Harness — Building graph from vault...")
    print(f"   Vault: {vault_root}")
    
    nodes, edges = build_graph(vault_root, use_cache=not args.no_cache)
    print(f"   ✓ {len(nodes)} neurons, {sum(len(e) for e in edges.values())} synapses")
    
    # Compute embeddings
    collection = compute_all_embeddings(nodes, vault_root)
    
    # Build BM25 index for hybrid search (Phase 3.1)
    # Skip if --serve: the server builds its own index
    bm25_idx = None
    if config.get('hybrid_search', False) and not args.serve:
        print("📊 Building BM25 index...")
        bm25_idx = BM25Index(nodes, k1=config.get('bm25_k1', 1.5), b=config.get('bm25_b', 0.75))
        print(f"   ✓ BM25: {bm25_idx.N} docs, {len(bm25_idx.df)} terms")
    
    # Load state
    state = load_state(vault_root)
    
    # --- Mode dispatch ---
    
    if args.serve:
        start_api_server(config, nodes, edges, collection, state)
        return
    
    if args.stats:
        show_stats(nodes, edges, state)
        return
    
    if args.hebbian_show:
        show_hebbian(state)
        return
    
    if args.refresh_embeddings:
        print("🔄 Force-refreshing all embeddings...")
        collection = compute_all_embeddings(nodes, vault_root, force_refresh=True)
        print(f"  ✓ Refreshed {collection.count()} embeddings")
        return
    
    if args.interactive:
        interactive_mode(vault_root, nodes, edges, collection, state, bm25_index=bm25_idx)
        return
    
    # Single query mode
    query = ' '.join(args.query).strip()
    if not query:
        print("No query provided. Use --interactive for REPL mode or --serve for API server.")
        return
    
    print(f"\n🔍 Query: '{query}'")
    
    active = attention(query, nodes, edges, collection, bm25_index=bm25_idx)
    
    if not active:
        print("  No notes activated above threshold.")
        return
    
    sorted_active = sorted(active.items(), key=lambda x: -x[1])
    print(f"\n  🧠 Activated {len(active)} neurons:")
    for note_id, score in sorted_active:
        node = nodes.get(note_id)
        title = node['title'] if node else note_id
        print(f"    [{score:.3f}] {title}")
    
    # Online plasticity: Hebbian update right after attention (Phase 3.2)
    if config.get('online_plasticity', True):
        state = hebbian_update(active, state)
        save_state(vault_root, state)
        print(f"  🔌 Hebbian update (online): {len(state['synapses'])} synapses")
    
    # LLM response
    print(f"\n  🤖 LLM response:")
    print("  " + "-" * 60)
    response = llm_respond(query, active, nodes)
    # Indent response
    for line in response.split('\n'):
        print(f"  {line}")
    print("  " + "-" * 60)
    
    # Neurogenesis: extract new concepts from LLM response
    active_titles = [nodes[nid]['title'] for nid in active if nid in nodes]
    new_concepts = extract_new_concepts(response, query, active, nodes)
    
    if new_concepts:
        print(f"\n  🧬 Neurogenesis: {len(new_concepts)} new concept(s) detected")
        for concept in new_concepts:
            title = concept.get('title', '').strip()
            definition = concept.get('definition', '').strip()
            if not title or not definition:
                continue
            new_note_id = create_note(vault_root, title, definition, active_titles, query)
            if new_note_id:
                print(f"    ✨ Created: {new_note_id} — {title}")
            else:
                print(f"    ⊘ Skipped (already exists): {title}")
    else:
        print(f"\n  🧬 Neurogenesis: no new concepts")

if __name__ == '__main__':
    main()