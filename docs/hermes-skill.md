# Hermes Skill: BDH Graph Harness

This file is a Hermes Agent skill definition. Copy it to `~/.hermes/skills/research/bdh-graph-harness/SKILL.md` to let your Hermes agent query the harness via natural language.

---

## When This Skill Activates

Use this skill when the user:
- Asks about BDH, "graph memory", "self-organizing vault", "Hebbian retrieval"
- Wants to query the knowledge graph or check graph stats
- Asks about Hebbian synapses, activated notes, or neurogenesis
- Wants an alternative to RAG that learns from its own usage

## Architecture

```
[Query]
   ↓
[Attention: ChromaDB KNN seed + BM25 hybrid scoring + k-hop graph traversal + hub dampening + adaptive threshold]
   ↓
[Active Notes subset] → [Hebbian Update (online plasticity — before LLM)]
   ↓                          ↓
   ↓                    [LLM: grounded response with citations]
   ↓                          ↓
   ↓                    [Neurogenesis: extract new concepts → create notes]
   ↓                          ↓
[Persist state] ← ← ← [New notes join the graph for future queries]
```

The key difference from RAG: **the graph learns from its own activity**. Every query strengthens connections between co-activated notes (before the LLM responds — online plasticity). Every LLM response can create new notes. The next query operates on a modified graph.

### Hybrid Search (BM25 + Vector)

A `BM25Index` builds an in-memory keyword index over all note texts at startup. During attention, ChromaDB returns vector similarity scores, BM25 returns keyword scores, and they combine: `final = α * vec_sim + β * bm25_score` (default α=0.7, β=0.3). This catches exact-keyword matches that embedding similarity misses.

### Online Plasticity

Hebbian update runs *immediately after attention*, not after the LLM responds. Synaptic state reflects what was *activated*, not what the LLM happened to use. The graph's connection weights update before the LLM even starts generating.

### Adaptive Threshold

Instead of a fixed threshold, the system computes a dynamic threshold from the score distribution: `threshold = max(Q75, mean + 1*std, floor)` where `floor` defaults to 0.15. Requires ≥5 candidates to activate.

## Key Parameters

```python
SEED_COUNT = 5              # top-k ChromaDB KNN results as seeds
MAX_HOP = 2                 # graph traversal depth
MAX_NEIGHBORS_PER_HOP = 10  # cap traversal breadth
HUB_DAMPENING = True        # scale activation by 1/(1+0.15*(deg-15))
ALPHA = 0.7                 # Hebbian frequency weight
BETA = 0.3                  # Hebbian recency weight
DECAY = 0.95                # decay factor for unused synapses
HYBRID_ALPHA = 0.7          # vector similarity weight
HYBRID_BETA = 0.3           # BM25 keyword score weight
ADAPTIVE_THRESHOLD = True  # dynamic threshold from score distribution
THRESHOLD_FLOOR = 0.15      # minimum threshold
```

## Running the Harness

```bash
# Start API server
python -m bdh_graph_harness --serve

# Single query (CLI)
python -m bdh_graph_harness "query text"

# Interactive REPL
python -m bdh_graph_harness --interactive

# Stats
python -m bdh_graph_harness --stats

# Hebbian state
python -m bdh_graph_harness --hebbian-show

# Force graph rebuild (skip cache)
python -m bdh_graph_harness --stats --no-cache

# Full embedding refresh (only when changing embedding model)
python -m bdh_graph_harness --refresh-embeddings
```

## API Server — Query from Chat

The harness has an HTTP API server (default `127.0.0.1:8643`). This is the preferred way to query from a Hermes chat session.

### Endpoints

| Method | Endpoint | Body / Notes | Returns |
|--------|----------|--------------|---------|
| POST | `/api/query` | `{"query": "..."}` | `{response, activated_notes, new_concepts, hebbian_synapses, hebbian_updates}` |
| POST | `/api/stream` | `{"query": "..."}` | SSE stream: activation → tokens → done |
| GET | `/api/stats` | — | graph stats + hebbian summary |
| GET | `/api/graph` | — | `{nodes, edges}` JSON |
| GET | `/api/hebbian` | — | synaptic state |
| POST | `/api/refresh` | — | force refresh embeddings |

### Query workflow

1. **Check server is up:**
   ```bash
   curl -sS --max-time 3 http://127.0.0.1:8643/api/stats
   ```

2. **Query the graph:**
   ```bash
   curl -sS --max-time 30 http://127.0.0.1:8643/api/query \
     -H 'Content-Type: application/json' \
     -d '{"query": "USER_QUESTION_HERE"}'
   ```

3. **Parse the response:**
   - `response` — the synthesized natural-language answer (show this to the user)
   - `activated_notes` — list of vault notes activated during retrieval (present as citations)
   - `new_concepts` — new concepts extracted from the query
   - `hebbian_synapses` — synaptic strengthening that occurred

4. **If server is down**, start it:
   ```bash
   cd /path/to/bdh-graph-harness && python -m bdh_graph_harness --config bdh-config.yaml --serve
   ```

### Formatting the answer

1. The `response` text as the main answer.
2. A **Sources / Activated notes** section listing the activated notes with similarity scores.
3. (Optional) a note on new concepts or Hebbian changes if relevant.

Don't dump raw JSON at the user.

## Hebbian Learning Rule

**When:** immediately after attention (online plasticity) — before the LLM generates a response.

For all pairs of co-activated notes:
- Increment frequency
- Set weight = `ALPHA * min(freq/10, 1.0) + BETA * 1.0`
- Update `last_coactivated` timestamp
- Decay unused synapses by `DECAY` factor; prune below 0.01

The Hebbian layer discovers relationships NOT present as wikilinks. Co-activated notes create new semantic connections the vault didn't have.

## LLM Provider Configuration

The harness supports two LLM providers, controlled by `llm_provider` in `bdh-config.yaml`. **Embeddings always stay on Ollama** — only the LLM inference moves.

### Ollama (local, default)

```yaml
llm_provider: ollama
llm_model: gemma4:12b-mlx
```

- Uses Ollama `/api/chat` endpoint
- Payload: `{"model": "...", "messages": [...], "options": {"temperature": 0.3, "num_ctx": 4096}}`

### OpenRouter (cloud, OpenAI-compatible)

```yaml
llm_provider: openrouter
llm_model: openrouter/free
openrouter_url: https://openrouter.ai/api/v1/chat/completions
openrouter_key: ${OPENROUTER_API_KEY}  # env var expansion
```

- Uses OpenAI-compatible `/v1/chat/completions` endpoint
- Headers: `Authorization: Bearer <key>`
- Config env var expansion: `${OPENROUTER_API_KEY}` is expanded at config load time

Start the server with the API key in the environment:

```bash
export OPENROUTER_API_KEY=your-key-here
cd /path/to/bdh-graph-harness && python -m bdh_graph_harness --config bdh-config.yaml --serve
```

## Neurogenesis

After each LLM response, a second LLM call extracts concepts NOT already in the vault:
- For each new concept: create atomic note in `concepts/` with frontmatter (`tags: [neurogenesis, auto-generated]`, `confidence: low`)
- Include 1-sentence definition from LLM
- Auto-link to seed notes (notes activated in this query)
- At next harness run, incremental embedding refresh detects and embeds the new notes automatically

## Incremental Graph Rebuild

The harness caches the full graph (nodes + edges + per-file mtimes) in `.bdh-graph-cache.json` inside the vault. On each startup:

1. Load cache, compare current file mtimes vs cached mtimes
2. If 0 changes → instant cache hit
3. If <50% changed → re-read only changed/new/deleted notes, patch graph in-memory
4. If >50% changed → full rebuild

Override with `--no-cache` to force full rebuild.

## Visualization
## Visualization
The web UI at `:8643` shows:
- **Nodes** colored by activation state or by Obsidian tags (toggle)
- **Wikilink edges** in dark gray (thin)
- **Hebbian synapses** in blue, width proportional to synaptic weight
- **Edge tooltips** — hover synapses to see weight, type, connected notes
- **Tag legend overlay** — bottom-left, shows tag→color mapping
- **Orphan nodes toggle** — hidden by default, show isolated nodes on demand
- **Hebbian pulse animation:** during a query, strengthened synapses flash green (`#39d353`), thicken, then settle back — staggered cascade (120ms per edge, 4-step fade over 2.5s)
- **Live neurogenesis** — new edges appear in real-time as concepts are created
- **WebSocket auto-reconnect** with status indicator
- **Dark theme**, mobile responsive (iPhone safe area, touch dismiss, tab-based layout ≤768px)

## Critical Techniques

### Ollama Embedding — Use /api/embed, NOT /api/embeddings

The legacy `/api/embeddings` endpoint accepts single `prompt` and fails with HTTP 500 under rapid calls. The newer `/api/embed` accepts `input` array and handles batches reliably.

### ChromaDB on macOS

macOS Python sqlite3 lacks `enable_load_extension` at compile time. Use ChromaDB PersistentClient — no server, stores in a directory, works out of the box.

### Content Hash for Incremental Refresh

SHA256 of each note's full text stored as ChromaDB metadata field `content_hash`. On each run, compare hashes — only re-embed changed/new notes.

### Hub Dampening

High-degree nodes (like `index.md` with 35 links) activate during almost every query. Scale activation by `1/(1+0.15*(deg-15))` for nodes with degree > 15.

## Pitfalls

- **Ollama model loading**: first embedding call after model load may timeout. Warm up with 2-3 dummy calls before batch embedding.
- **Gemma4 JSON format**: even with `format: "json"`, output may include markdown fences. Parse defensively.
- **Threshold sensitivity**: 0.15 admits too much noise, 0.25 is clean. Tune per vault density.
- **Port conflicts**: `OSError: [Errno 48] address already in use` if a previous instance is still running. Kill with `lsof -ti:8643 | xargs kill`.
- **aiohttp `tcp_keepalive` crashes on macOS/Tailscale**: monkeypatch `aiohttp.tcp_helpers.tcp_keepalive` to swallow `OSError` on `utun` interfaces.
- **`from module import GLOBAL` captures the value at import time**: use `import bdh_graph_harness.config as _config` and access `_config.CONFIG` at call time, not import time.
- **Frontmatter `tags` field — string or array**: handle both in JS: `Array.isArray(n.tags) ? n.tags.join(', ') : n.tags`.

## Scaling Notes

- **42 notes**: linear scan JSON was fine (~0.5ms per query)
- **344 notes**: ChromaDB KNN is instant
- **5,000+ notes**: ChromaDB essential
- **ChromaDB storage**: ~30KB per note (768 dim × 4 bytes + metadata)
- **k-hop explosion**: `MAX_NEIGHBORS_PER_HOP=10` prevents combinatorial explosion. At 10K notes, consider reducing to 5.