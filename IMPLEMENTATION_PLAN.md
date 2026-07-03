# BDH Graph Harness — Implementation Plan

> From PoC to production. Ordered by priority: must-have → should-have → nice-to-have.

## Phase 1 — Must-have (senza questi non è utilizzabile)

### 1.1 Config file (`bdh-config.yaml`)
- [x] Estrarre tutti gli hardcoded values in un YAML
- [x] Fields: vault_path, llm_model, embedding_model, ollama_url, chroma_path, thresholds, hebbian params
- [x] Override via CLI flags (`--config`, `--vault`, `--model`)
- [x] Default values se config manca

### 1.2 HTTP API server (aiohttp)
- [x] `POST /query` — query → attention → LLM → neurogenesis → Hebbian → response JSON
- [x] `GET /stats` — graph stats + Hebbian state
- [x] `GET /graph` — nodes + edges JSON per visualization
- [x] `GET /hebbian` — synaptic state JSON
- [x] `POST /refresh` — force refresh embeddings
- [x] Bind to 127.0.0.1 only (security)
- [x] Graceful shutdown

### 1.3 Error handling serio
- [x] Retry con backoff su Ollama 500/timeout (embedding + LLM)
- [x] Graceful degradation se neurogenesis fallisce (non bloccare query)
- [x] ChromaDB lock handling (retry su ConcurrentUpdateError)
- [x] File lock su `.bdh-state.json` (fcntl o filelock)
- [x] Structured logging (non solo print)

### 1.4 Neurogenesis → vault integration
- [x] Auto-update `wiki/index.md` quando neurogenesis crea una nota
- [x] Auto-append a `wiki/log.md`
- [x] Validazione: non creare note duplicate (title match fuzzy)
- [x] Tags corretti nel frontmatter

### 1.5 Hebbian state concurrent-safe
- [x] File lock su `.bdh-state.json` con `fcntl.flock`
- [x] Read-modify-write atomico
- [x] Oppure: migrare Hebbian state in SQLite (stesso ChromaDB o file separato)

## Phase 2 — Should-have (per uso regolare)

### 2.1 Skill Hermes
- [x] `skill_manage(action='create', name='bdh-graph-harness')`
- [x] SKILL.md con: when to use, prerequisites, how to run, quick reference
- [x] Il skill chiama l'API server locale
- [x] Esempio: "bdh query: come funziona la memoria di Hermes?"

### 2.2 Realtime visualization (WebSocket + vis.js)
- [x] Server WebSocket sull'API server (`/ws`)
- [x] HTML page con vis.js force-directed graph
- [x] Nodi: grigio (inattivo) → arancione (attivato) → verde (seed)
- [x] Archi: grigio sottile (wikilink) → blu spesso (Hebbian, spessore = weight)
- [x] Push realtime su query: attivazioni animate
- [x] Timeline slider per replay query precedenti
- [x] Dark theme (awwwards level per albi)

### 2.3 Tests
- [x] `test_graph.py` — build_graph su vault mock, verifica nodes/edges
- [x] `test_attention.py` — attention con mock embeddings, verifica seed + k-hop
- [x] `test_hebbian.py` — update + decay + persist
- [x] `test_neurogenesis.py` — extract + create_note + no overwrite
- [x] `test_api.py` — endpoint happy path con TestClient
- [x] Run con: `python -m pytest tests/ -v`

### 2.4 Incremental graph rebuild
- [x] Cache del grafo in `.bdh-graph-cache.json` (nodes + edges + file mtimes)
- [x] Invalidation: se mtime di un file cambia, rebuild solo quel node
- [x] Se numero di file cambia, full rebuild
- [x] Benchmark: 344 note full rebuild vs incremental

## Phase 3 — Nice-to-have (per scale futuro)

### 3.1 Hybrid search (vector + keyword)
- [x] BM25 o TF-IDF su tutti i note texts
- [x] Combine: `final_score = α * vector_sim + β * keyword_score`
- [x] ChromaDB `where_document` per pre-filter
- [x] Config: `hybrid_alpha`, `hybrid_beta`

### 3.2 Online plasticity
- [x] Hebbian update dopo ogni nota recuperata (non dopo LLM response)
- [ ] LLM riceve context progressivo che si aggiorna durante generation
- [x] Streaming: LLM response token-by-token con update continuo

### 3.3 Adaptive threshold
- [x] Calcolare score distribution per query
- [x] Threshold = percentile(Q75) o mean + 1std
- [x] Min threshold floor: 0.15 (non scendere sotto)
- [x] Log threshold adattivo per debugging

### 3.4 Multi-vault
- [ ] Multiple ChromaDB collections (una per vault)
- [ ] Bridge links: wikilinks che attraversano vault
- [ ] Config: `vaults: [path1, path2, ...]`
- [ ] Query attraversa vault via bridge links

## Development order

1. **1.1 Config** → ✅ done
2. **1.2 API server** → ✅ done
3. **1.3 Error handling** → ✅ done
4. **1.5 Concurrent-safe** → ✅ done
5. **1.4 Neurogenesis vault integration** → ✅ done
6. **2.1 Skill** → ✅ done
7. **2.4 Incremental graph** → ✅ done
8. **2.2 Visualization** → ✅ done
9. **2.3 Tests** → ✅ done
10. **3.1-3.4** → post-MVP