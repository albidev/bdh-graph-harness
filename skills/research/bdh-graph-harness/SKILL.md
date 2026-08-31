---
name: bdh-graph-harness
description: Query BDH graph state from Hermes Agent.
version: 0.1.0
author: Alberto Sigismondi (albidev), Davide Davin (imbundle), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [BDH, graph memory, Hebbian retrieval, Obsidian]
    related_skills: []
---

# BDH Graph Harness Skill

Use this skill to operate the BDH Graph Harness from Hermes Agent. BDH is an experimental graph-based memory system for Markdown/Obsidian vaults; it combines structural links, vector retrieval, optional BM25 search, graph attention, Hebbian learning, and controlled neurogenesis. This skill covers the BDH integration only: the `bdh-hermes-bridge` plugin is a separate component and is not installed by this file.

## When to Use

Use this skill when the user:

- asks about BDH, graph memory, a self-organizing vault, or Hebbian retrieval;
- wants a grounded answer from the knowledge graph;
- asks for current graph statistics or Hebbian state;
- asks which notes were activated, what concepts were generated, or how retrieval works;
- wants to inspect or operate a local BDH API server.

Do not use it for generic RAG advice, unrelated Obsidian operations, or Hermes core/plugin development. For bridge installation and hook configuration, use the [bdh-hermes-bridge documentation](https://github.com/albidev/bdh-hermes-bridge).

## Prerequisites

- Hermes Agent with the `bdh` bridge plugin installed and enabled when using `bdh_query` or `bdh_stats`.
- A running BDH API server for bridge tools and HTTP fallbacks.
- A valid BDH configuration containing the vault path and the effective API host/port.
- Ollama available for embeddings; the configured LLM provider may be local Ollama, oMLX, Ollama Cloud, OpenRouter, or another OpenAI-compatible endpoint.

The bridge URL is deployment-specific. Read `bdh-config.yaml`, the bridge configuration, or the service definition and use that effective URL; never assume that a port in an example is universal. Keep API credentials in the environment, not in committed YAML.

The bridge plugin and this skill have different jobs:

- **Skill:** instructions that teach Hermes when and how to use BDH.
- **`bdh-hermes-bridge` plugin:** registers the Hermes-native `bdh_query` and `bdh_stats` tools and can connect automatic read/write hooks.
- **BDH server:** owns the graph, embeddings, Hebbian state, and HTTP API.

Installing this skill does not install or enable the plugin, start the server, or change graph state.

## How to Run

### Install the skill

Install the canonical file directly from this repository; no copy from `docs/` is required:

```bash
hermes skills install https://raw.githubusercontent.com/albidev/bdh-graph-harness/main/skills/research/bdh-graph-harness/SKILL.md
```

Verify discovery:

```bash
hermes skills list
```

The repository also contains the source file at `skills/research/bdh-graph-harness/SKILL.md`. A checked-out copy can be installed with the same `hermes skills install` command using a reachable raw URL for the selected branch or commit.

### Query from Hermes

Prefer the bridge tools when they are available:

```text
bdh_query(query="How does Hebbian retrieval work?", vault_id="research")
bdh_stats(vault_id="research")
```

`vault_id` is optional. If omitted, BDH selects its configured default vault. Do not invent a vault ID; discover configured vaults through BDH or its configuration first.

### Run the server or use the API fallback

From a checkout of this repository:

```bash
python -m bdh_graph_harness --config bdh-config.yaml --serve
```

For a direct HTTP fallback, substitute the actual configured base URL:

```bash
curl -sS --max-time 5 <BDH_BASE_URL>/api/stats
curl -sS --max-time 30 <BDH_BASE_URL>/api/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"How does Hebbian retrieval work?"}'
```

The API server may also expose `/api/vaults` for configured vault discovery. When a `vault_id` is required, pass it explicitly as a query parameter or request field.

## Quick Reference

| Operation | Hermes/API entry point | State effect |
|---|---|---|
| Grounded graph query | `bdh_query(query, vault_id?)` / `POST /api/query` | **Mutates:** Hebbian learning; may create neurogenesis notes |
| Graph statistics | `bdh_stats(vault_id?)` / `GET /api/stats` | **Read-only** |
| Full graph inspection | `GET /api/graph` | Read-only |
| Hebbian inspection | `GET /api/hebbian` | Read-only |
| Source/configured-vault discovery | `GET /api/vaults` | Read-only |
| Embedding refresh | `POST /api/refresh` or `--refresh` | **Mutates:** embeddings/cache |
| Consolidation | `POST /api/consolidate` | **Mutates:** synapses and possibly dormant nodes |
| Dry-run consolidation | `POST /api/consolidate` with `{"dry_run":true}` | Read-only preview |
| Source scan | `python -m bdh_graph_harness --scan-sources` | Read-only |

### Typical configuration values

These are examples, not universal defaults:

```yaml
llm_provider: ollama
llm_model: <configured-model>
api_host: <configured-host>
api_port: <configured-port>
seed_count: 5
max_hop: 2
hybrid_search: false
hebbian_dynamic_edges_enabled: true
```

The effective per-vault LLM can override the global provider, model, endpoint, timeout, and privacy mode. Embeddings remain on Ollama even when completion inference uses another provider.

## Procedure

1. **Establish the integration path.** Check whether the `bdh` toolset exposes `bdh_query` and `bdh_stats`. If it does not, report that the bridge plugin is missing or disabled; do not pretend that installing this skill provides those tools.
2. **Discover the target vault and endpoint.** Read the active BDH configuration or call the configured vault-discovery endpoint. Record the effective base URL and optional `vault_id`; do not replace them with an example port.
3. **Check connectivity read-only.** Call `bdh_stats` or `GET /api/stats`. Confirm a structured response containing graph metrics, or report the exact connection/error response and stop before attempting a mutating operation.
4. **Choose the least invasive operation.** Use `bdh_stats`, `/api/graph`, `/api/hebbian`, or `--scan-sources` for inspection. Use `bdh_query` only when the user wants a grounded graph answer or explicitly asks for retrieval.
5. **Run a grounded query when requested.** Pass the user's question as the query. If the response includes `response`, present it as the main answer; cite `activated_notes` with note names and scores. Mention `new_concepts` or Hebbian changes only when relevant.
6. **Keep mutations explicit.** Warn before refresh, consolidation, neurogenesis-related writes, or any operation that changes the vault or persistent graph state. A normal `bdh_query` is not read-only even when the user only wants information.
7. **Handle failures without hiding them.** Distinguish an unavailable bridge tool, an unreachable server, an invalid vault ID, an LLM/embedding failure, and a valid empty result. Do not retry a timed-out `POST /api/query` blindly: the endpoint may already have applied Hebbian updates or neurogenesis.
8. **Verify the requested result.** For a read-only operation, ensure the returned JSON is structurally valid and matches the requested vault. For a mutation, report the response and re-read the relevant stats or resource when the operation exposes a stable read-back path.

### Architecture reference

```text
Markdown/Obsidian vaults + optional read-only sources
                    ↓
       watcher → graph + ChromaDB + optional BM25
                    ↓
       query → vector/BM25 seeds → k-hop attention
                    ↓
       activated notes → Hebbian update → grounded LLM response
                    ↓                         ↓
       persistent state ← neurogenesis notes and links
                    ↓
       sleep-cycle downscaling, pruning, and quality checks
```

The important distinction from a conventional RAG wrapper is that retrieval is stateful: co-activated notes can strengthen learned edges, and a response can produce a validated concept note for later retrieval. Declared wikilinks remain separate from learned Hebbian relations and phantom links.

### Retrieval and learning behavior

- ChromaDB provides vector seeds; optional BM25 catches exact terms and combines with vector scores.
- Attention expands through bounded graph hops, with adaptive thresholds and optional hub dampening to prevent high-degree index notes from dominating.
- Online Hebbian plasticity runs after attention and before LLM generation.
- Unused synapses decay; consolidation can downscale weights, prune stale weak traces, re-evaluate node quality, and remove persistently dormant nodes according to configuration.
- Dynamic learned-only edges are traversed only when enabled and above their configured thresholds.

### Neurogenesis and ingestion

After an eligible LLM response, BDH can extract genuinely new concepts, validate them against blocklists and semantic duplicates, write atomic notes under the configured concepts directory, and link them to activated source notes. The source-aware watcher then detects new or changed Markdown and updates graph/index state incrementally. Treat those operations as writes, not as a side effect to hide from the user.

### Operational commands

```bash
# Read-only source scan: no embeddings, LLM, ChromaDB, or writes
python -m bdh_graph_harness --config bdh-config.yaml --scan-sources

# Read-only CLI statistics
python -m bdh_graph_harness --config bdh-config.yaml --stats

# Read-only Hebbian inspection
python -m bdh_graph_harness --config bdh-config.yaml --hebbian-show

# Force a graph rebuild or embedding refresh only when explicitly needed
python -m bdh_graph_harness --config bdh-config.yaml --refresh
python -m bdh_graph_harness --config bdh-config.yaml --refresh-embeddings
```

Use the configured command-line options for the installed version; experimental APIs, storage formats, and defaults can change.

## Pitfalls

- **Skill versus plugin:** installing the skill does not register `bdh_query` or `bdh_stats`. The bridge plugin must be installed and enabled separately.
- **Deployment-specific ports:** `8643`, `8644`, `8083`, and other ports may occur in local examples, but none is a universal endpoint. Resolve the active configuration first.
- **Read-only versus query:** `bdh_stats` is read-only; `bdh_query` performs retrieval and normal Hebbian learning and may trigger neurogenesis. `POST /api/query` has the same non-read-only nature.
- **Timeout retries:** query POSTs are not safely idempotent. A client timeout does not prove that BDH did nothing; avoid blind retries that could duplicate learning or concept creation.
- **Provider split:** embeddings use Ollama; completion inference can use another configured provider. A working completion provider does not prove that embeddings are available.
- **Cold starts:** the first embedding request after loading a model can be slow. Warm the embedding model or use the configured timeout rather than treating one slow first request as a permanent outage.
- **Neurogenesis output:** provider responses can wrap JSON in Markdown fences or return an object instead of an array. BDH parses defensively, but malformed provider output can still produce no concepts.
- **Threshold tuning:** low activation thresholds admit noise; the right value depends on vault density. Do not change thresholds merely to make one query return more notes.
- **macOS/Tailscale sockets:** aiohttp keep-alive setup can fail on some `utun` interfaces; use the repository's configured compatibility handling instead of bypassing the API.
- **Local state safety:** refresh, consolidation, and neurogenesis can change persistent files. Back up important state before destructive maintenance and prefer dry-run consolidation when available.

## Verification

Run these checks after installing or changing the skill:

```bash
# 1. Install from the canonical repository path
hermes skills install https://raw.githubusercontent.com/albidev/bdh-graph-harness/main/skills/research/bdh-graph-harness/SKILL.md

# 2. Confirm discovery and metadata
hermes skills list

# 3. Validate frontmatter and required sections with the local Hermes validator
python -c "
from pathlib import Path
from tools.skill_manager_tool import _validate_frontmatter
p = Path('skills/research/bdh-graph-harness/SKILL.md')
content = p.read_text(encoding='utf-8')
error = _validate_frontmatter(content, new_skill=True)
assert error is None, error
print('SKILL.md frontmatter: OK')
"

# 4. Check the running graph without mutating it
# Use bdh_stats in Hermes, or:
curl -sS --max-time 5 <BDH_BASE_URL>/api/stats
```

A valid verification run must show the skill in `hermes skills list`, print `SKILL.md frontmatter: OK`, and return a successful structured stats response from the configured BDH endpoint. If the bridge or server is unavailable, record the exact error and mark connectivity as unverified; do not substitute a cached or invented result.

For a full repository change, also run:

```bash
python -m pytest -q
```

The canonical source is this file. `docs/hermes-skill.md` is retained only as a migration pointer and must not contain a second copy of these instructions.
