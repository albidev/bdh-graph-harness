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
- **Vector retrieval** — semantic search via embeddings (default). Optional BM25 hybrid mode for multilingual vaults (disabled by default — see `benchmarks/BM25_ANALYSIS.md`)
- **Adaptive thresholding** — `max(Q75, mean+1std, 0.15)` to filter noise dynamically
- **Neurogenesis** — LLM extracts new concepts from queries and creates notes in the vault
- **Node quality scoring** — composite score (strong edges + mean weight + frequency) auto-prunes dormant nodes from visualization; re-activates on strong re-encounter
- **Sleep-cycle consolidation** — periodic synaptic downscaling (×0.9), structural pruning below weight floor, and stale dormant node removal — mirrors biological sleep consolidation
- **Server-side file watcher** — mtime-based polling detects vault changes from any source (Obsidian, LLM, scripts) and triggers incremental graph updates
- **LLM responses** — OpenRouter (`openrouter/free`) or local Ollama, with citations back to source notes
- **Real-time visualization** — vis.js network showing nodes activating, edges pulsing as Hebbian weights update during queries

For the theory behind these choices — why Hebbian plasticity, why Obsidian, why not just RAG — see [`docs/philosophy.md`](docs/philosophy.md).

## Architecture

```
Obsidian Vault → Embed (Ollama) → ChromaDB + Graph
                                    ↓
Query → Vector Search → Attention Spread (max_hop=2)
                                    ↓
Hebbian Update (co-activation strengthening) → LLM Response (OpenRouter)
                                    ↓
WebSocket → vis.js visualization (nodes light up, synapses pulse)

Sleep Cycle (periodic):
  Synaptic Downscaling (×0.9) → Prune (< floor) → Quality Re-eval → Stale Removal
```

## Package structure

```
bdh_graph_harness/
├── __main__.py              # CLI entry point (--serve, --mcp, --query, --refresh)
├── config.py                # Config loading, env var expansion, retry logic
├── mcp_server.py            # MCP server (FastMCP, stdio + HTTP transport)
├── graph/
│   ├── parser.py            # Frontmatter + wikilink parsing
│   ├── builder.py           # Graph construction + incremental cache
│   └── cache.py             # Graph cache serialization
├── retrieval/
│   ├── embeddings.py        # Ollama embedding client
│   ├── chroma_store.py      # ChromaDB vector store
│   ├── bm25.py              # BM25 lexical index (optional, disabled by default)
│   ├── hybrid.py            # Vector + BM25 fusion (optional, disabled by default)
│   └── attention.py         # Seed selection + k-hop spread + adaptive threshold
├── memory/
│   ├── hebbian.py           # Synaptic weight update + decay
│   ├── quality.py           # Node quality scoring + dormant pruning
│   ├── consolidation.py     # Sleep-cycle: downscaling + pruning + stale removal
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
│   ├── ws.py                # WebSocket handlers
│   └── watcher.py           # Server-side vault file watcher (mtime polling)
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
| `hybrid_search` | `false` | Enable BM25 hybrid mode (disabled by default for Italian vaults) |
| `hybrid_alpha` | 0.7 | Vector search weight (only when `hybrid_search: true`) |
| `hybrid_beta` | 0.3 | BM25 search weight (only when `hybrid_search: true`) |
| `llm_provider` | `openrouter` | `openrouter` or `ollama` |
| `llm_model` | `openrouter/free` | Auto-selects available free models |
| `api_port` | 8643 | Server port |
| `quality_threshold` | 0.25 | Quality score below this → node marked dormant |
| `quality_reactivation_score` | 0.50 | Activation score to re-awaken a dormant node |
| `quality_prune_interval` | 50 | Re-evaluate node quality every N queries |
| `graph_ignore` | `[]` | fnmatch patterns to exclude nodes from the graph (e.g. `[".bdh-*"]`) |
| `consolidation_downscale_factor` | 0.90 | Global weight multiplier per sleep cycle |
| `consolidation_prune_weight_floor` | 0.02 | Delete synapses below this weight after downscaling |
| `consolidation_dormant_persist_cycles` | 3 | Remove nodes dormant for N+ consolidation cycles |
| `consolidation_prune_dormant_nodes` | `true` | Delete stale dormant nodes (not just hide) |

## Tests

```bash
pytest tests/ -v
```

180 tests covering graph building, attention spread, adaptive threshold, BM25, hybrid search (optional), Hebbian updates, LLM providers (Ollama + OpenRouter), neurogenesis, consolidation (downscaling, pruning, stale removal), and API endpoints.

## Visualization

The web UI at `:8643` shows a real-time vis.js graph with:
- **Nodes** colored by activation state or by Obsidian tags (toggle)
- **Wikilink edges** + **Hebbian synapses** with hover tooltips (weight, type, connected notes)
- **Live neurogenesis** — new edges appear in real-time as concepts are created
- **Orphan nodes toggle**, **tag legend overlay**, **dark theme**, **mobile responsive** (iPhone safe area, touch dismiss)
- **Node quality** — dormant nodes dimmed (gray, 30% opacity) with 💤 tooltip; stats bar shows dormant count

See [`docs/visualization.md`](docs/visualization.md) for full details on controls, tooltips, and mobile support.

## MCP Server

The harness includes a [Model Context Protocol](https://modelcontextprotocol.io) server that exposes the Hebbian graph as tools to any MCP-compatible client (Claude Desktop, Cursor, Windsurf, Continue).

```bash
# stdio mode (Claude Desktop, Cursor)
python -m bdh_graph_harness --mcp

# HTTP mode (web clients)
python -m bdh_graph_harness --mcp --mcp-transport http --mcp-port 8644
```

**Tools:** `query` (grounded Q&A with citations), `stats` (graph overview), `hebbian` (learned synapses), `graph` (full network), `refresh` (rebuild embeddings).

The MCP server imports the package directly — no dependency on the HTTP API server. Both can run independently or simultaneously.

See [`docs/mcp-server.md`](docs/mcp-server.md) for client configuration (Claude Desktop, Cursor, etc.).

## Hermes Agent integration

The harness ships with a [Hermes Agent](https://hermes-agent.nousresearch.com) skill that lets you query the graph from chat. The skill definition is in [`docs/hermes-skill.md`](docs/hermes-skill.md) — copy it to `~/.hermes/skills/research/bdh-graph-harness/SKILL.md` to activate it.

Once installed, your Hermes agent can:
- Query the graph via natural language ("bdh query: how does Hebbian learning work?")
- Show graph stats and Hebbian synaptic state
- Start/stop the API server
- Present answers with source citations

## Obsidian Sync Plugin

Auto-sync vault changes to the BDH server via an Obsidian plugin — no manual refresh needed.

```
Obsidian edit → Plugin detects → Debounce 1s → POST /api/node-update
    → Server diffs graph → WebSocket broadcast → Viz updates in real-time
```

**Setup:**
1. Build the plugin: `cd plugins/obsidian && npm install && npm run build`
2. Copy `manifest.json` + `main.js` to your vault's `.obsidian/plugins/bdh-graph-harness-sync/`
3. Enable "BDH Graph Harness Sync" in Obsidian Settings → Community Plugins

**Features:**
- Debounced updates (1s, configurable) — no server spam
- Status bar indicator (○ idle, ◎ syncing, ● ok, ✗ error)
- Ignores non-`.md` files and `.obsidian/` directory
- Configurable server URL, debounce delay, enable/disable

**Pulse animation:** updated nodes pulse orange for 2.5s then restore to original appearance via remove + re-add (vis.js cannot reset explicit color overrides via `update()` — the remove+re-add trick restores group defaults cleanly).

## Sleep-Cycle Consolidation

Periodic graph maintenance that mirrors biological sleep consolidation. Run it manually or schedule it (e.g. nightly via cron):

```bash
# Trigger a consolidation cycle
curl -X POST http://localhost:8643/api/consolidate

# Dry run (see what would change without committing)
curl -X POST http://localhost:8643/api/consolidate -H "Content-Type: application/json" -d '{"dry_run": true}'

# View config and cycle count
curl http://localhost:8643/api/consolidation-stats
```

**Cycle steps:**
1. **Synaptic downscaling** — multiply all Hebbian weights by `consolidation_downscale_factor` (default 0.90). Prevents runaway strengthening.
2. **Structural pruning** — delete synapses with weight below `consolidation_prune_weight_floor` (default 0.02) after downscaling.
3. **Quality re-evaluation** — recalculate node quality scores and update dormant state.
4. **Stale removal** — delete nodes dormant for more than `consolidation_dormant_persist_cycles` (default 3) consecutive cycles, if `consolidation_prune_dormant_nodes` is true.

No tokens consumed — pure algorithmic operation on the local graph state. Safe to run while the server is serving queries (file-locked state access).

## License

MIT