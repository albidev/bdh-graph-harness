# Multiple ChromaDB Collections per Vault — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Support multiple Obsidian vaults in one BDH Graph Harness server, with one isolated ChromaDB collection per vault and explicit vault selection on API/CLI/MCP calls.

**Architecture:** Introduce a `VaultContext` abstraction and a `VaultRegistry` that owns per-vault graph, state, Chroma collection, BM25 index, watcher, and locks. Keep the default single-vault config backward-compatible, then add a `vaults:` config list for multi-vault mode. Route handlers should resolve the target vault once at request entry and operate only on that context — no global `CONFIG['vault_path']` leaks.

**Tech Stack:** Python 3.10+, aiohttp, ChromaDB PersistentClient, pytest/pytest-asyncio, existing BDH graph/retrieval/memory modules.

---

## Current State / Problem

Today the project is effectively single-vault:

- `bdh_graph_harness/config.py` exposes one `vault_path`, one `chroma_path`, one `chroma_collection`.
- `bdh_graph_harness/retrieval/chroma_store.py::compute_all_embeddings(nodes, vault_root, ...)` opens Chroma at `os.path.join(vault_root, CONFIG['chroma_path'])` and always uses `CONFIG['chroma_collection']`.
- `bdh_graph_harness/api/server.py::start_api_server(...)` stores one global `app_state` with `nodes`, `edges`, `collection`, `state`, `bm25_index`, and one watcher.
- `bdh_graph_harness/api/routes.py` persists state/neurogenesis through `config['vault_path']`, so request handlers cannot safely switch vaults.
- `bdh_graph_harness/neurogenesis/dedupe.py` and `bdh_graph_harness/memory/phantom.py` reopen Chroma using global `CONFIG`, which will query the wrong collection unless vault context is explicit.
- Current Chroma default collection naming is inconsistent in fallbacks: config says `notes`, dedupe fallback says `bdh_notes`, phantom fallback says `bdh_embeddings`. That’s a footgun with a hat.

The result: adding a second vault by changing config or CLI flags risks cross-vault embedding reuse, wrong semantic dedupe, wrong phantom links, and state writes to the wrong `.bdh-state.json`.

---

## Non-Goals

- Do **not** merge vault graphs in this issue.
- Do **not** implement cross-vault retrieval/ranking yet.
- Do **not** add auth/permissions per vault yet.
- Do **not** migrate existing `.bdh-state.json` format unless necessary.
- Do **not** change note IDs globally; keep IDs vault-local and disambiguate at API boundaries with `vault_id`.

Cross-vault search can be a follow-up once per-vault isolation is boring and correct.

---

## Proposed Config Schema

Backward-compatible single-vault config remains valid:

```yaml
vault_path: ~/Documents/Hermes
chroma_path: .bdh-chroma
chroma_collection: notes
```

New multi-vault config:

```yaml
# Optional default used when request does not include vault_id.
default_vault: hermes

# Shared Chroma DB directory. If omitted, each vault keeps its own .bdh-chroma.
# Recommended for multi-vault: central DB with one collection per vault.
chroma_path: ~/.bdh/chroma

vaults:
  - id: hermes
    name: Hermes
    path: ~/Documents/Hermes
    chroma_collection: vault_hermes_notes
    neurogenesis_dir: wiki/concepts
    graph_ignore:
      - wiki/index
      - wiki/log
      - wiki/raw/*

  - id: research
    name: Research Vault
    path: ~/Documents/Research
    chroma_collection: vault_research_notes
    neurogenesis_dir: concepts
```

Rules:

1. `id` is required, unique, stable, URL-safe: `^[a-zA-Z0-9_-]+$`.
2. `path` is required and must exist at startup unless `lazy_load_vaults: true` is later added.
3. `chroma_collection` defaults to `vault_{safe_id}_notes`.
4. Existing single-vault configs are normalized internally into one vault with `id: default` unless `default_vault` is set.
5. Vault-specific config overrides inherit global defaults.
6. `vault_path` remains accepted but should be treated as deprecated once `vaults:` exists.

---

## API Contract

Every route that reads/writes vault state must accept a vault selector.

### Query endpoints

`POST /api/query`

```json
{
  "vault_id": "hermes",
  "query": "...",
  "source": "assistant_response",
  "user_prompt": "optional"
}
```

If `vault_id` is omitted:
- Use `default_vault`.
- If more than one vault exists and `default_vault` is missing, return `400` with `available_vaults`.

Same behavior for:
- `POST /api/stream`
- `POST /api/refresh`
- `POST /api/refresh-graph`
- `POST /api/node-update`
- `POST /api/consolidate`

### Read endpoints

Support query param selector:

- `GET /api/stats?vault_id=hermes`
- `GET /api/graph?vault_id=hermes`
- `GET /api/hebbian?vault_id=hermes`
- `GET /api/quality?vault_id=hermes`
- `GET /api/consolidation-stats?vault_id=hermes`

### New endpoint

`GET /api/vaults`

```json
{
  "default_vault": "hermes",
  "vaults": [
    {
      "id": "hermes",
      "name": "Hermes",
      "path": "~/Documents/Hermes",
      "neurons": 174,
      "synapses": 312,
      "embeddings": 174,
      "queries_processed": 51
    }
  ]
}
```

### WebSocket events

Include `vault_id` on every event:

```json
{
  "type": "activation",
  "vault_id": "hermes",
  "query": "...",
  "activated_notes": []
}
```

Clients can ignore other vaults initially. Later, UI can add a vault picker.

---

## Internal Design

Create `bdh_graph_harness/vaults.py`:

```python
from dataclasses import dataclass, field
import asyncio
from pathlib import Path
from typing import Any

@dataclass
class VaultConfig:
    id: str
    name: str
    path: str
    chroma_path: str
    chroma_collection: str
    settings: dict[str, Any]

@dataclass
class VaultContext:
    config: VaultConfig
    nodes: dict
    edges: dict
    collection: Any
    state: dict
    bm25_index: Any | None = None
    state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    watcher: Any | None = None

class VaultRegistry:
    def __init__(self, global_config: dict): ...
    def list(self) -> list[VaultContext]: ...
    def get(self, vault_id: str | None = None) -> VaultContext: ...
    def default_id(self) -> str: ...
```

Important: `VaultContext.config.settings` should be a merged config dict where `vault_path`, `chroma_path`, and `chroma_collection` are already vault-specific. That lets legacy functions keep receiving a normal config dict during migration.

---

## Phase 1 — Config Normalization and VaultContext

### Task 1: Add config normalization tests

**Objective:** Lock down backward compatibility and new multi-vault parsing before touching runtime code.

**Files:**
- Create: `tests/test_vaults.py`
- Modify later: `bdh_graph_harness/vaults.py`

**Tests:**

```python
def test_single_vault_config_normalizes_to_default_context(tmp_path):
    cfg = {
        'vault_path': str(tmp_path / 'Hermes'),
        'chroma_path': '.bdh-chroma',
        'chroma_collection': 'notes',
    }
    (tmp_path / 'Hermes').mkdir()

    vaults = normalize_vault_configs(cfg)

    assert len(vaults) == 1
    assert vaults[0].id == 'default'
    assert vaults[0].path == str(tmp_path / 'Hermes')
    assert vaults[0].chroma_collection == 'notes'


def test_multi_vault_config_assigns_default_collection_names(tmp_path):
    (tmp_path / 'A').mkdir()
    (tmp_path / 'B').mkdir()
    cfg = {
        'default_vault': 'alpha',
        'chroma_path': str(tmp_path / 'chroma'),
        'vaults': [
            {'id': 'alpha', 'name': 'Alpha', 'path': str(tmp_path / 'A')},
            {'id': 'beta', 'name': 'Beta', 'path': str(tmp_path / 'B')},
        ],
    }

    vaults = normalize_vault_configs(cfg)

    assert [v.id for v in vaults] == ['alpha', 'beta']
    assert vaults[0].chroma_collection == 'vault_alpha_notes'
    assert vaults[1].chroma_collection == 'vault_beta_notes'
```

**Run:**

```bash
pytest tests/test_vaults.py -q
```

Expected first: fail because `bdh_graph_harness.vaults` does not exist.

### Task 2: Implement `VaultConfig` and `normalize_vault_configs`

**Objective:** Parse both config shapes into a validated list of vault configs.

**Files:**
- Create: `bdh_graph_harness/vaults.py`

**Implementation notes:**

- Validate duplicate IDs.
- Validate ID regex.
- Expand `~` in paths.
- For each vault, build `settings = dict(global_config)` then apply vault overrides.
- Set `settings['vault_path'] = path` and `settings['chroma_collection'] = collection`.

**Run:**

```bash
pytest tests/test_vaults.py -q
```

Expected: pass.

### Task 3: Add Chroma collection name helper

**Objective:** Centralize safe collection naming and kill inconsistent fallbacks.

**Files:**
- Modify: `bdh_graph_harness/vaults.py`
- Modify later: `dedupe.py`, `phantom.py`, `chroma_store.py`

**Rules:**

```python
def default_collection_name(vault_id: str) -> str:
    safe = re.sub(r'[^a-zA-Z0-9_-]+', '_', vault_id).strip('_')
    return f'vault_{safe}_notes'
```

**Tests:**

```python
def test_default_collection_name_is_stable():
    assert default_collection_name('my-vault') == 'vault_my-vault_notes'
    assert default_collection_name('Research Vault') == 'vault_Research_Vault_notes'
```

---

## Phase 2 — Retrieval: Explicit Chroma Collection per Vault

### Task 4: Extend `compute_all_embeddings` signature

**Objective:** Make Chroma location/collection explicit without breaking callers.

**Files:**
- Modify: `bdh_graph_harness/retrieval/chroma_store.py`
- Test: `tests/test_vaults.py` or new `tests/test_chroma_store.py`

**Target signature:**

```python
def compute_all_embeddings(
    nodes,
    vault_root,
    force_refresh=False,
    *,
    chroma_path=None,
    collection_name=None,
    config=None,
):
    cfg = config or CONFIG
    resolved_chroma_path = chroma_path or cfg.get('chroma_path', '.bdh-chroma')
    if not os.path.isabs(resolved_chroma_path):
        resolved_chroma_path = os.path.join(vault_root, resolved_chroma_path)
    resolved_collection = collection_name or cfg.get('chroma_collection', 'notes')
```

**Acceptance:**

- Existing callers still work.
- Two vaults using the same central `chroma_path` but different `collection_name` do not share IDs/counts.

### Task 5: Add test for Chroma isolation

**Objective:** Prove one DB directory can host two isolated vault collections.

**Test shape:**

```python
def test_chroma_collections_are_isolated_per_vault(tmp_path, monkeypatch):
    # monkeypatch get_embeddings to deterministic vectors
    # create nodes_a = {'a': ...}, nodes_b = {'b': ...}
    # same chroma_path, collection_name='vault_a_notes' vs 'vault_b_notes'
    # assert collection_a.count() == 1
    # assert collection_b.count() == 1
    # assert collection_a.get()['ids'] == ['a']
    # assert collection_b.get()['ids'] == ['b']
```

**Run:**

```bash
pytest tests/test_vaults.py tests/test_chroma_store.py -q
```

---

## Phase 3 — Server Registry and Route Resolution

### Task 6: Introduce `VaultRegistry.load_all()`

**Objective:** Build graph/state/embeddings/BM25 once per vault at server startup.

**Files:**
- Modify: `bdh_graph_harness/vaults.py`
- Modify: `bdh_graph_harness/api/server.py`

**Behavior:**

For each `VaultConfig`:
1. `build_graph(vault.path, use_cache=True)`
2. `compute_all_embeddings(nodes, vault.path, chroma_path=vault.chroma_path, collection_name=vault.chroma_collection, config=vault.settings)`
3. `load_state(vault.path)`
4. Build `BM25Index` if enabled for that vault.
5. Store as `VaultContext`.

### Task 7: Change `start_api_server` app_state shape

**Objective:** Replace single-vault app state with registry while preserving helper access.

**Current:**

```python
app_state = {
    'nodes': nodes,
    'edges': edges,
    'collection': collection,
    'state': state,
    'config': config,
}
```

**Target:**

```python
app_state = {
    'registry': registry,
    'config': config,
}
```

During migration, route wrappers can pass a per-request context dict:

```python
def context_to_state(ctx: VaultContext) -> dict:
    return {
        'vault_id': ctx.config.id,
        'nodes': ctx.nodes,
        'edges': ctx.edges,
        'collection': ctx.collection,
        'state': ctx.state,
        'config': ctx.config.settings,
        'bm25_index': ctx.bm25_index,
        'state_lock': ctx.state_lock,
    }
```

This avoids rewriting every route in one huge, bug-shaped commit.

### Task 8: Add vault resolver for requests

**Objective:** Resolve `vault_id` consistently for JSON bodies and query params.

**Files:**
- Modify: `bdh_graph_harness/api/routes.py`

**Helper:**

```python
async def resolve_vault_state(request, app_state):
    registry = app_state['registry']
    vault_id = request.query.get('vault_id')
    if request.method in {'POST', 'PUT', 'PATCH'}:
        # Avoid consuming request JSON twice: parse once in endpoint or use request['json_body'] middleware.
        ...
    ctx = registry.get(vault_id)
    return context_to_state(ctx)
```

Practical approach: in each POST handler, parse JSON first, pull `vault_id`, then call `registry.get(vault_id)`. Do not make a middleware that consumes JSON bodies and creates a new class of nonsense.

### Task 9: Add `/api/vaults`

**Objective:** Make multi-vault state visible and debuggable.

**Files:**
- Modify: `bdh_graph_harness/api/routes.py`
- Test: `tests/test_api.py`

**Acceptance:**

- `GET /api/vaults` returns all configured vaults and default vault.
- Does not leak hidden config values or API keys.

---

## Phase 4 — Update API Handlers and Watchers

### Task 10: Update read routes to use selected vault

**Objective:** `stats`, `graph`, `hebbian`, `quality`, `consolidation-stats` operate on selected context.

**Files:**
- Modify: `bdh_graph_harness/api/routes.py`
- Tests: `tests/test_api.py`

**Tests:**

- Build two fake `VaultContext` objects with different node counts.
- `GET /api/stats?vault_id=a` returns A count.
- `GET /api/stats?vault_id=b` returns B count.
- Missing `vault_id` returns default.
- Unknown `vault_id` returns 404/400 with `available_vaults`.

### Task 11: Update query/stream routes

**Objective:** `POST /api/query` and `POST /api/stream` use the selected vault for retrieval, LLM context, Hebbian updates, neurogenesis, and WS event metadata.

**Files:**
- Modify: `bdh_graph_harness/api/routes.py`

**Acceptance:**

- Query against vault A cannot activate notes from vault B.
- `save_state` receives vault A path for vault A query.
- `create_note` writes to vault A neurogenesis dir.
- WS activation event includes `vault_id`.

### Task 12: Update refresh/update/consolidation routes

**Objective:** All state-mutating maintenance routes operate on selected vault only.

**Files:**
- Modify: `bdh_graph_harness/api/routes.py`

**Routes:**

- `POST /api/refresh`
- `POST /api/refresh-graph`
- `POST /api/node-update`
- `POST /api/consolidate`

**Acceptance:**

- Refresh vault A does not change vault B `collection.count()`.
- Consolidate vault A does not mutate vault B state.

### Task 13: Start one watcher per vault

**Objective:** File watching remains automatic in multi-vault mode.

**Files:**
- Modify: `bdh_graph_harness/api/server.py`

**Implementation notes:**

- Loop through registry contexts.
- Each watcher callback closes over its own `VaultContext`.
- Broadcast includes `vault_id`.
- On cleanup, stop all watchers.

**Pitfall:** Python closure capture in loops. Use a factory:

```python
def make_trigger_node_update(ctx):
    async def trigger_node_update():
        ...
    return trigger_node_update
```

Do not close over the loop variable unless you enjoy debugging ghosts.

---

## Phase 5 — Fix Global CONFIG Leaks

### Task 14: Refactor semantic dedupe to accept vault config

**Objective:** `is_semantic_duplicate` queries the selected vault collection, not global config.

**Files:**
- Modify: `bdh_graph_harness/neurogenesis/dedupe.py`
- Modify callers in `bdh_graph_harness/neurogenesis/creator.py` if needed

**Target signature:**

```python
def is_semantic_duplicate(title, definition, threshold=0.65, *, vault_root=None, config=None):
```

Use:
- `vault_root or cfg['vault_path']`
- `cfg.get('chroma_path', '.bdh-chroma')`
- `cfg.get('chroma_collection', 'notes')`

### Task 15: Refactor phantom links to accept collection/config explicitly

**Objective:** Consolidation creates phantom links from the selected vault collection only.

**Files:**
- Modify: `bdh_graph_harness/memory/phantom.py`
- Modify: `bdh_graph_harness/memory/consolidation.py` if it calls `find_phantom_links`

**Target:**

```python
def find_phantom_links(nodes, edges, vault_root, *, config=None, collection=None):
```

If `collection` is passed, use it directly. Otherwise open Chroma from the explicit config.

### Task 16: Audit `CONFIG['vault_path']` and `config['vault_path']` usages

**Objective:** Remove unsafe global vault assumptions in runtime paths.

**Command:**

```bash
rg "CONFIG\[['\"]vault_path|config\[['\"]vault_path|chroma_collection|chroma_path" bdh_graph_harness tests -n
```

**Acceptance:**

- Global `CONFIG['vault_path']` only remains in CLI/default fallback code, not API request paths.
- Chroma collection fallback is consistently `notes` or explicit `vault_{id}_notes`; no `bdh_notes` / `bdh_embeddings` drift.

---

## Phase 6 — CLI and MCP

### Task 17: Add CLI `--vault-id`

**Objective:** CLI can choose a configured vault without passing raw paths.

**Files:**
- Modify: `bdh_graph_harness/__main__.py`

**Behavior:**

- `--vault /path` still works and overrides config for one-off single-vault mode.
- `--vault-id hermes` selects from `vaults:`.
- If neither is passed, use `default_vault`.

**Commands:**

```bash
python -m bdh_graph_harness --vault-id hermes --stats
python -m bdh_graph_harness --vault-id research "query text"
```

### Task 18: Add MCP vault selector

**Objective:** MCP fallback tools can query a specific vault.

**Files:**
- Modify: `bdh_graph_harness/mcp_server.py`

**Behavior:**

- Add optional `vault_id` parameter to query/stats/refresh tools.
- Fallback cache becomes per-vault, not one `_fallback` dict.

Suggested structure:

```python
_fallback_by_vault: dict[str, dict] = {}
```

---

## Phase 7 — Documentation and Migration

### Task 19: Update config docs and sample config

**Files:**
- Modify: `bdh-config.yaml`
- Modify: `README.md`
- Modify: `docs/mcp-server.md` if MCP selector is added

**Docs must include:**

- Single-vault config still works.
- Multi-vault example.
- `vault_id` API usage examples.
- Chroma isolation explanation.
- Migration note: existing single-vault collection remains `notes`; multi-vault defaults to `vault_{id}_notes` unless overridden.

### Task 20: Add migration sanity command

**Objective:** Help users verify which collection has which IDs/counts.

**Option A:** CLI subcommand:

```bash
python -m bdh_graph_harness --list-vaults
```

**Output:**

```text
id       path                         collection            neurons embeddings
hermes   ~/Documents/Hermes vault_hermes_notes    174     174
```

**Option B:** Keep it API-only via `/api/vaults`. Prefer A if cheap.

---

## Verification Matrix

Run these before marking done:

```bash
pytest tests/test_vaults.py -q
pytest tests/test_chroma_store.py -q
pytest tests/test_api.py -q
pytest tests/test_neurogenesis.py -q
pytest tests/test_consolidation.py -q
```

Manual smoke test with two tiny temp vaults:

```bash
mkdir -p /tmp/bdh-vault-a /tmp/bdh-vault-b /tmp/bdh-chroma
cat > /tmp/bdh-vault-a/a.md <<'EOF'
---
title: Alpha Vault Note
tags: test
---
Alpha-only content.
EOF
cat > /tmp/bdh-vault-b/b.md <<'EOF'
---
title: Beta Vault Note
tags: test
---
Beta-only content.
EOF
```

Create `/tmp/bdh-multivault.yaml`:

```yaml
default_vault: a
chroma_path: /tmp/bdh-chroma
vaults:
  - id: a
    name: A
    path: /tmp/bdh-vault-a
  - id: b
    name: B
    path: /tmp/bdh-vault-b
ollama_url: http://127.0.0.1:11434
embedding_model: nomic-embed-text-v2-moe
llm_provider: ollama
llm_model: gemma4:12b-mlx
hybrid_search: false
neurogenesis_enabled: false
api_host: 127.0.0.1
api_port: 8650
```

Start server:

```bash
python -m bdh_graph_harness --config /tmp/bdh-multivault.yaml --serve
```

Verify:

```bash
curl -s http://127.0.0.1:8650/api/vaults | jq .
curl -s 'http://127.0.0.1:8650/api/stats?vault_id=a' | jq .neurons
curl -s 'http://127.0.0.1:8650/api/stats?vault_id=b' | jq .neurons
curl -s -X POST http://127.0.0.1:8650/api/query \
  -H 'Content-Type: application/json' \
  -d '{"vault_id":"a","query":"Alpha"}' | jq '.activated_notes'
curl -s -X POST http://127.0.0.1:8650/api/query \
  -H 'Content-Type: application/json' \
  -d '{"vault_id":"b","query":"Beta"}' | jq '.activated_notes'
```

Expected:

- `/api/vaults` lists both vaults.
- Stats for `a` and `b` are independent.
- Querying `a` activates only `a` note IDs.
- Querying `b` activates only `b` note IDs.
- WebSocket activation payloads include `vault_id`.

---

## Risks / Gotchas

### 🚨 Risk 1: Global `CONFIG` silently points to the wrong vault

**Problem:** Many modules import `CONFIG` directly. In multi-vault mode, a mutable global cannot represent request-local vault state.

**Solution:** Pass explicit `config`/`VaultContext` into retrieval, dedupe, phantom links, and route helpers. Leave global `CONFIG` only for defaults and CLI legacy mode.

### 🚨 Risk 2: Same note IDs across vaults collide

**Problem:** `wiki/index` or `concepts/foo` can exist in two vaults. If collections or API events are shared without `vault_id`, clients cannot distinguish them.

**Solution:** Keep Chroma one collection per vault. Add `vault_id` to API/WS payloads. Do not prefix internal note IDs unless implementing cross-vault search later.

### 🚨 Risk 3: Existing single-vault users lose embeddings

**Problem:** Automatically renaming the default collection from `notes` to `vault_default_notes` would make the first run look like an empty cache and re-embed everything.

**Solution:** In normalized single-vault legacy mode, preserve configured `chroma_collection` exactly (`notes`). Only auto-generate `vault_{id}_notes` for `vaults:` entries without explicit collection.

### 🚨 Risk 4: Watcher callbacks all update the last vault

**Problem:** Async closures inside loops can capture the final loop variable.

**Solution:** Use a callback factory per `VaultContext` and add a test or log that includes `vault_id`.

### 🚨 Risk 5: Semantic dedupe / phantom links reopen the wrong collection

**Problem:** Those modules independently open Chroma using global config and inconsistent fallback collection names.

**Solution:** Refactor them to accept explicit `config` or `collection`; add regression tests that vault A cannot see vault B embeddings.

---

## Acceptance Criteria

- [ ] Existing single-vault config works without changes.
- [ ] Multi-vault config with two vaults starts one server successfully.
- [ ] Each vault has its own ChromaDB collection.
- [ ] Same note ID in two vaults does not collide.
- [ ] API read/query/refresh/consolidate routes can target `vault_id`.
- [ ] `/api/vaults` lists vaults with counts and no secrets.
- [ ] WebSocket events include `vault_id`.
- [ ] Neurogenesis creates notes in the selected vault only.
- [ ] Semantic dedupe checks the selected vault only.
- [ ] Phantom links use the selected vault only.
- [ ] CLI supports `--vault-id`.
- [ ] MCP fallback supports optional `vault_id`.
- [ ] Tests cover isolation, default selection, unknown vault errors, and legacy config compatibility.

---

## Suggested Labels

`enhancement`, `retrieval`

## Suggested Title

`Support multiple vaults with one ChromaDB collection per vault`
