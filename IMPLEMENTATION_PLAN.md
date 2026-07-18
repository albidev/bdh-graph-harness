# Roadmap

## Completed

### Phase 1 — Core
- [x] YAML config with CLI overrides (`--config`, `--vault`, `--model`)
- [x] HTTP API server (aiohttp): `/query`, `/stats`, `/graph`, `/hebbian`, `/refresh`
- [x] Error handling: retry with backoff, graceful degradation, ChromaDB lock handling
- [x] Neurogenesis: auto-create notes, vault index updates, fuzzy duplicate detection
- [x] Concurrent-safe state persistence with file locks

### Phase 2 — Usability
- [x] Real-time visualization (WebSocket + vis.js): activation cascade, Hebbian pulse
- [x] Incremental graph rebuild with mtime-based cache invalidation
- [x] Test suite (108 tests): graph, attention, Hebbian, neurogenesis, API, BM25, adaptive threshold, LLM providers
- [x] launchd service for macOS with auto-restart

### Phase 3 — Retrieval quality
- [x] Hybrid search: vector similarity (α=0.7) + BM25 (β=0.3)
- [x] Adaptive threshold: `max(Q75, mean+1std, 0.15)` with ≥5 candidate floor
- [x] Online plasticity: Hebbian update after attention, not after LLM response
- [x] Streaming: token-by-token LLM response with continuous Hebbian update
- [x] LLM provider abstraction: Ollama + OpenRouter (OpenAI-compatible)

### Refactor
- [x] Modular package structure (2600-line monolith → 29-file package)
- [x] Dynamic config resolution (no stale imports)

### MCP Server
- [x] Model Context Protocol server (FastMCP, stdio + HTTP transport)
- [x] 5 tools: query, stats, hebbian, graph, refresh
- [x] Compatible with Claude Desktop, Cursor, Windsurf, Continue

## Future

### Multi-vault
### Data Quality Hardening
- [x] Structural self-loop filtering (federated + legacy builders)
- [x] External source derived-artifact exclusions (test-results, coverage, .next, cache, reports)
- [x] Stale-weak Hebbian retention policy (age + frequency gated pruning)
- [x] Neurogenesis source provenance (`activated_from_ids` frontmatter + `neurogenesis_source` generated edges)
- [x] Hebbian tail observability metrics (`hebbian_strong/weak/stale_weak_synapses`)
- [x] Read-only graph quality audit script (`scripts/audit_graph_quality.py`)

### 3D Visualization
- [x] Migration from 2D force-graph to 3D force-graph (3d-force-graph + Three.js 0.180.x)
- [x] Camera-preserving structural updates in 3D space
- [x] Per-link materials for highlight glow on every edge type
- [x] LOD for weak Hebbian edges, opacity, width, and label visibility
- [x] `query_response` WebSocket event for plugin-launched query visibility
- [x] Mobile tab-based layout (Graph / Controls / Inspector)

### Progressive context
- [ ] LLM receives progressive context that updates during generation

### Scale
- [ ] Benchmark: incremental vs full rebuild on 1000+ notes
- [ ] Embedding cache warming on startup
### Multi-vault
- [x] Multiple ChromaDB collections (one per vault)
- [x] Config: `vaults: [path1, path2, ...]`
- [ ] Bridge links across vaults