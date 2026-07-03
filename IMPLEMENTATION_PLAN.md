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
- [ ] Multiple ChromaDB collections (one per vault)
- [ ] Bridge links across vaults
- [ ] Config: `vaults: [path1, path2, ...]`

### Progressive context
- [ ] LLM receives progressive context that updates during generation

### Scale
- [ ] Benchmark: incremental vs full rebuild on 1000+ notes
- [ ] Embedding cache warming on startup
- [ ] Graceful handling of vault files removed/renamed