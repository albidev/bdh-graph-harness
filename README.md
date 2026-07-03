<p align="center">
  <img src="docs/cover.png" alt="BDH Graph Harness" width="100%">
</p>

# BDH Graph Harness

Graph-based retrieval system for Obsidian vaults with Hebbian synaptic plasticity.

## What it does

Turns an Obsidian vault into a living knowledge graph where:
- **Notes → neurons** — each note embedded with `nomic-embed-text-v2-moe` (Ollama, local)
- **Wikilinks → synapses** — graph edges from `[[wikilinks]]`
- **Hebbian learning** — co-activated notes strengthen their synaptic weight over time (frequency + recency + activation correlation)
- **Hybrid retrieval** — vector similarity (α=0.7) + BM25 lexical search (β=0.3)
- **Adaptive thresholding** — `max(Q75, mean+1std, 0.15)` to filter noise dynamically
- **Neurogenesis** — LLM extracts new concepts from queries and creates notes in the vault
- **LLM responses** — OpenRouter (`openrouter/free`) or local Ollama, with citations back to source notes
- **Real-time visualization** — vis.js network showing nodes activating, edges pulsing as Hebbian weights update during queries

## Architecture

```
Obsidian Vault → Embed (Ollama) → ChromaDB + Graph
                                    ↓
Query → Hybrid Search (vector + BM25) → Attention Spread (max_hop=2)
                                    ↓
Hebbian Update (co-activation strengthening) → LLM Response (OpenRouter)
                                    ↓
WebSocket → vis.js visualization (nodes light up, synapses pulse)
```

## Package structure

```
bdh_graph_harness/
├── __main__.py              # CLI entry point (--serve, --query, --refresh)
├── config.py                # Config loading, env var expansion, retry logic
├── graph/
│   ├── parser.py            # Frontmatter + wikilink parsing
│   ├── builder.py           # Graph construction + incremental cache
│   └── cache.py             # Graph cache serialization
├── retrieval/
│   ├── embeddings.py        # Ollama embedding client
│   ├── chroma_store.py      # ChromaDB vector store
│   ├── bm25.py              # BM25 lexical index
│   ├── hybrid.py            # Vector + BM25 fusion
│   └── attention.py         # Seed selection + k-hop spread + adaptive threshold
├── memory/
│   ├── hebbian.py           # Synaptic weight update + decay
│   └── state_store.py       # Persistent state (file-locked)
├── llm/
│   ├── providers.py         # LLM factory + payload builder
│   ├── ollama.py            # Ollama backend
│   ├── openrouter.py        # OpenRouter backend (OpenAI-compatible)
│   └── prompt.py            # System prompt + context formatting
├── neurogenesis/
│   ├── creator.py           # Concept extraction + note creation
│   └── dedupe.py            # Fuzzy duplicate detection
├── api/
│   ├── server.py            # aiohttp app setup + WebSocket
│   ├── routes.py            # REST endpoints
│   └── ws.py                # WebSocket handlers
└── visualization/
    └── templates/index.html # vis.js real-time graph UI
```

`harness.py` is a compatibility shim that re-exports from the package — tests use `import harness`.

## Setup

1. **Ollama** running locally with `nomic-embed-text-v2-moe` pulled
2. **OpenRouter API key** in `OPENROUTER_API_KEY` env var (or switch `llm_provider: ollama` in config)
3. **Python 3.11+** with dependencies:

```bash
pip install -r requirements.txt
```

4. **Configure** the vault path:

```bash
cp bdh-config.yaml bdh-config.local.yaml
# Edit vault_path to point at your Obsidian vault
```

## Usage

```bash
# Start server
python -m bdh_graph_harness --serve

# Single query (CLI)
python -m bdh_graph_harness --query "come funziona l'apprendimento Hebbian?"

# Force graph rebuild
python -m bdh_graph_harness --refresh

# Open visualization
open http://localhost:8643
```

### Running as a service (macOS)

```bash
# Install launchd service
cp ai.bdh.graph-harness.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/ai.bdh.graph-harness.plist
```

The service auto-restarts on crash (`KeepAlive: true`). Logs at `~/.hermes/logs/bdh-server.log`. The `start-server.sh` wrapper exports `OPENROUTER_API_KEY` from `~/.hermes/.env` before launching.

## Config

See `bdh-config.yaml` for all parameters. Key ones:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed_count` | 5 | Top-k embedding matches as seed nodes |
| `max_hop` | 2 | Graph traversal depth from seeds |
| `active_threshold` | 0.25 | Min activation score (overridden by adaptive) |
| `alpha` | 0.7 | Frequency weight in Hebbian |
| `beta` | 0.3 | Recency weight in Hebbian |
| `decay` | 0.95 | Per-session decay for unused synapses |
| `hybrid_alpha` | 0.7 | Vector search weight |
| `hybrid_beta` | 0.3 | BM25 search weight |
| `llm_provider` | `openrouter` | `openrouter` or `ollama` |
| `llm_model` | `openrouter/free` | Auto-selects available free models |
| `api_port` | 8643 | Server port |

## Tests

```bash
pytest tests/ -v
```

108 tests covering graph building, attention spread, adaptive threshold, BM25, hybrid search, Hebbian updates, LLM providers (Ollama + OpenRouter), neurogenesis, and API endpoints.

## Visualization

The web UI at `:8643` shows:
- **Nodes** colored by type (note, concept, seed, activated)
- **Wikilink edges** in blue
- **Hebbian synapses** in cyan, width proportional to synaptic weight
- **Real-time activation cascade** — nodes light up in sequence, edges pulse green (`#39d353`) when Hebbian weights strengthen during a query
- **Dark theme** with staggered animation (120ms cascade, 4-step fade over 2.5s)

## License

MIT