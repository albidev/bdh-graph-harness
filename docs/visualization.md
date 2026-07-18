# Visualization — BDH Graph Harness

The real-time web visualization at `:8643` renders the knowledge graph using
[3d-force-graph](https://github.com/vasturiano/3d-force-graph) (WebGL/Three.js) with live
activation feedback during queries.

## Features

### Node rendering

The renderer uses [3d-force-graph](https://github.com/vasturiano/3d-force-graph) 1.80.0 with a pinned Three.js 0.180.x ES module for custom geometries. Nodes are Three.js objects with label sprites, selection rings, and per-node materials.

- **Nodes** are colored by activation state: inactive (gray), seed (blue), activated (orange); neurogenesis nodes use aqua (`#00E5FF`)
- **Tag-based coloring** — toggle the "Tags" button to color nodes by their Obsidian frontmatter tags. Each tag gets a unique color from a consistent palette
- **Source filtering** — the Source selector shows all nodes, only primary vault nodes, or only external source nodes. With tag coloring disabled, vault nodes use blue and external nodes use orange.
- **Orphan nodes hidden by default** — nodes with zero connections are hidden to reduce clutter. Toggle "Orphans" to show them. The node counter updates to reflect visible vs total
- **Shape-aware hit area** — 3D raycasting handles node picking; diamond shape for neurogenesis nodes

### Edge rendering

Edge families are rendered as 3D Three.js line objects with per-link materials:

- **Wikilink edges** — dark gray, thin lines representing `[[wikilinks]]` from the vault
- **Hebbian synapses** — green edges whose width is proportional to synaptic weight
- **Phantom links** — blue dashed edges for semantic similarity connections
- **Counterpart edges** — reciprocal vault ↔ external project-anchor edges
- **Project-context edges** — generated star edges connecting external project notes to their root README
- **Neurogenesis source edges** — generated edges from `activated_from_ids` frontmatter connecting newborn notes to their validated source nodes (weight 0.35, type `neurogenesis_source`)
- **Z-order by importance** — wikilinks (bottom) → phantom (middle) → hebbian (top)
- **Edge tooltips** — hover over any synapse to see: weight, type (hebbian/wikilink/phantom), and the connected note titles
- **Edge visibility filtering** — Hebbian threshold slider hides edges below weight; phantom and direct-only toggles

### Hebbian pulse animation

During a query, strengthened synapses get animated particles:
- Staggered cascade: 120ms delay per edge
- Particle color transitions from green (#39d353) through blue (#58a6ff) to settle
- Particle count proportional to weight gain (1–6 particles)
- Timeout scales dynamically with number of edges: `max(5000, (N-1) × 120 + 3000)`ms

### Hover highlighting

- **Node hover** — highlights the 1-hop subgraph: connected nodes and edges glow, everything else dims (opacity 0.38)
- **Edge hover** — highlights only the hovered edge (color #d2a8ff, width 2.8, 2 particles), does NOT dim the graph
- Rendering uses idle pause: the canvas pauses when the force engine and meaningful animation settle, and resumes on interaction

### Activation explainability

Each activated note now carries both its final activation score and its role in the retrieval graph:

- **`seed`** — selected directly by vector/Hybrid retrieval (`hop: 0`)
- **`graph_neighbor`** — reached through a wikilink expansion, with `hop` and `parent_id`
- `vector_score`, `bm25_score`, and `hybrid_score` preserve retrieval evidence
- `hebbian_boost` shows the contextual memory contribution
- `final_score` is the score used for activation and visualization

The side panel labels seeds and graph neighbors separately. This avoids presenting a directly retrieved note and a hop-2 contextual neighbor as if they had the same evidence.

### Viewport management

- **Camera fit** — graph camera fits on first load after `getGraphBbox()` exposes real rendered bounds; fog density and node world scale adapt to the fitted camera distance
- **Camera-preserving updates** — `setGraphDataPreservingView` saves and restores camera position, target, orientation, and all three spatial dimensions across graph rebuilds
- **Safe structural updates** — empty or malformed node datasets are ignored instead of clearing the live graph; activation events received before graph initialization are ignored safely
- **Node drag** — nodes can be repositioned by dragging in 3D space
- **LOD** — camera-distance level of detail applies to weak Hebbian edges, opacity, width, and label visibility

## Controls

| Control | Description |
|---------|-------------|
| **Tags** toggle | Switch between activation-based and tag-based node coloring |
| **Source** selector | Show all, vault-only, or external-only nodes |
| **Orphans** toggle | Show/hide nodes with no connections |
| **Direct-only** toggle | Show only wikilink edges |
| **Counterpart** toggle | Show/hide reciprocal vault ↔ external project-anchor edges |
| **Phantom** toggle | Show/hide semantic similarity edges |
| **Hebbian threshold** slider | Minimum weight for Hebbian edges to show (0–1, default 0.3) |
| **Neurogenesis source** toggle | Show/hide generated `neurogenesis_source` edges |
| **Spacing** slider | Network spacing — distance between nodes |
| **Edge length** slider | Edge length multiplier |
| **Query bar** | Type a query and press Enter to run retrieval + Hebbian update + LLM response |
| **Stats counters** | Live neuron, synapse, and Hebbian counts — update from server truth after each query |

Slider values are persisted in `localStorage` and restored on page refresh.

## Tooltips

- **Node tooltip** — shows on hover: note title, source type/id, relative path, open-file link, tags, degree (connection count), and a text preview of the note content. Dismiss on tap (mobile)
- **Edge tooltip** — shows on hover: synaptic weight, edge type, connected note titles

## State management

Activation state (opacity, color, visibility) is managed via external Maps rather than mutating force-graph's internal objects:

- `nodeActivationOpacity` — per-node opacity override during query
- `nodeActivationColor` — per-node color override during Hebbian birth animation
- `linkActivationVisible` — per-link visibility override during query
- `linkVisibilityState` — per-link visibility from edge filters (threshold, phantom, direct-only)
- `linkParticleState` / `linkParticleColorState` / `linkParticleCountState` — Hebbian pulse particle configuration

This approach avoids the `TypeError: Attempted to assign to readonly property` crash that occurred when spreading force-graph's live objects.

## Mobile support

The visualization is fully responsive:
- **Tab-based layout** at ≤768px — controls and graph on separate tabs
- **iPhone safe area** — respects `env(safe-area-inset-top/bottom)` for notch/Dynamic Island
- **Touch-first graph interactions** — one-finger drag pans the graph, pinch zooms, and taps select nodes without relying on hover events
- **Touch dismiss** — tap anywhere outside an open tooltip to dismiss it
- **Tag legend hidden** on mobile to save space (tags visible in node tooltips)
- **Compact mobile layout** — reduced chrome and controls preserve graph area on narrow screens

The mobile interaction model deliberately avoids treating touch as a synthetic mouse. This prevents accidental node selection while panning and keeps the graph usable on phones and tablets.

## Neurogenesis provenance

Generated concept notes store provenance in YAML frontmatter rather than in a visible `## Origin` body section:

- `created_by` identifies the generator;
- `generation_query` stores the sanitized triggering query;
- `activated_from` stores the human-readable source note titles;
- `activated_from_ids` stores the canonical node IDs (JSON list) used for generated edge materialization.

The parser excludes frontmatter from note embeddings, so generation metadata remains available for auditing without becoming a retrieval attractor shared by every generated note.

When `neurogenesis_source_edges_enabled` is `true` (default), the graph builder materializes reciprocal generated edges of type `neurogenesis_source` for each valid ID in `activated_from_ids`. Missing IDs are reported as unresolved provenance and never trigger basename fallback.

## WebSocket

The visualization connects to the server via WebSocket for real-time updates:
- **Initial payload** — on connect, sends full graph (nodes with id/title/tags/path/text, edges, hebbian synapses, stats) to populate the visualization
- Auto-reconnect with status indicator
- Receives activation cascades, Hebbian updates, neurogenesis events, and `query_response` events as they happen
- Structural events (`graph_refresh`, `node_update`) rebuild the dataset through `setGraphDataPreservingView` so the viewport is preserved
- Empty structural updates are ignored defensively; this prevents a transient empty payload during a refresh/reconnect from clearing the force-graph
- Activation events are ignored safely until the initial graph payload has created the force-graph instance
- Falls back to polling if WebSocket is unavailable

## Visualization update safety

The browser treats graph updates as two separate classes:

- **Activation updates** change opacity, color, particles, and pulse state through external Maps. They must not replace `graphData()`.
- **Structural updates** add/remove nodes or links and are the only updates allowed to call `setGraphDataPreservingView()`.

The client rejects empty structural payloads and safely ignores activation events received before graph initialization. This protects the current graph during transient WebSocket reconnects or vault-refresh races. The original intermittent disappearance report is tracked in [issue #12](https://github.com/albidev/bdh-graph-harness/issues/12).

## Dark theme

The entire UI uses a dark theme with:
- Dark background (`#0d1117`)
- High-contrast node/edge colors for readability
- Smooth animations with CSS transitions

## Architecture

```
Server (aiohttp)
  ├── REST endpoints (query, stats, graph, hebbian, refresh)
  └── WebSocket (/ws) — real-time events
        ↓
Browser (3d-force-graph / WebGL / Three.js)
  ├── 3d-force-graph instance for nodes/edges
  ├── Three.js custom objects (geometries, sprites, materials)
  ├── External Maps for activation/filter state
  ├── WebSocket listener → live updates
  ├── Tooltip system (custom HTML overlays)
  └── Tag coloring engine (frontmatter → color map)
```

See [`docs/visualization-3d-migration.md`](visualization-3d-migration.md) for the full 3D migration architecture, lifecycle, risks, and validation.
