# BDH Graph Harness

A graph-based retrieval system for Obsidian vaults with Hebbian synaptic plasticity.

## What it does

Turns an Obsidian vault into a living knowledge graph where:
- **Notes become neurons** — each note gets embedded with `nomic-embed-text-v2-moe` (Ollama, local)
- **Wikilinks become synapses** — graph edges based on `[[wikilinks]]`
- **Hebbian learning** — co-activated notes strengthen their synaptic weight over time (frequency + recency + activation correlation)
- **Hybrid retrieval** — vector similarity (α=0.7) + BM25 lexical search (β=0.3)
- **Adaptive thresholding** — `max(Q75, mean+1std, 0.15)` to filter noise dynamically
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

## Setup

1. **Ollama** running locally with `nomic-embed-text-v2-moe` pulled
2. **OpenRouter API key** in `OPENROUTER_API_KEY` env var (or switch `llm_provider: ollama`)
3. **Python 3.11+** with dependencies: `aiohttp`, `aiohttp-cors`, `chromadb`, `numpy`, `pyyaml`, `websockets`

```bash
# Configure
cp bdh-config.yaml bdh-config.local.yaml
# Edit vault_path to point at your Obsidian vault

# Run
python3 harness.py --serve

# Open
open http://localhost:8643
```

## Config

See `bdh-config.yaml` for all parameters. Key ones:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed_count` | 5 | Top-k embedding matches as seed nodes |
| `max_hop` | 2 | Graph traversal depth from seeds |
| `active_threshold` | 0.25 | Min activation score (overridden by adaptive) |
| `alpha` | 0.7 | Frequency weight in Hebbian |
| `beta` | 0.3 | Recency weight in Hebbian |
| `hybrid_alpha` | 0.7 | Vector search weight |
| `hybrid_beta` | 0.3 | BM25 search weight |

## Running as a service (macOS)

```bash
# Install launchd service
cp ai.bdh.graph-harness.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/ai.bdh.graph-harness.plist
```

Logs at `~/.hermes/logs/bdh-server.log`.

## Tests

```bash
pytest tests/ -v
```

108 tests covering BM25, adaptive threshold, LLM providers (Ollama + OpenRouter), streaming, and Hebbian updates.

## Visualization

The web UI at `:8643` shows:
- **Nodes** colored by type (note, concept, seed, activated)
- **Wikilink edges** in blue
- **Hebbian synapses** in cyan, width proportional to synaptic weight
- **Real-time activation cascade** — nodes light up in sequence, edges pulse green when Hebbian weights strengthen during a query

## License

MIT