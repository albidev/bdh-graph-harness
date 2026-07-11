# Visualization — BDH Graph Harness

The real-time web visualization at `:8643` renders the knowledge graph using
[force-graph](https://github.com/vasturiano/force-graph) (WebGL) with live
activation feedback during queries.

## Features

### Node rendering

- **Nodes** are colored by activation state: inactive (gray), seed (blue), activated (orange)
- **Tag-based coloring** — toggle the "Tags" button to color nodes by their Obsidian frontmatter tags. Each tag gets a unique color from a consistent palette
- **Orphan nodes hidden by default** — nodes with zero connections are hidden to reduce clutter. Toggle "Orphans" to show them. The node counter updates to reflect visible vs total
- **Shape-aware hit area** — node pointer area includes +6px padding for easier selection in dense regions

### Edge rendering

- **Wikilink edges** — dark gray, thin lines representing `[[wikilinks]]` from the vault
- **Hebbian synapses** — green edges whose width is proportional to synaptic weight
- **Phantom links** — blue dashed edges for semantic similarity connections
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
- Rendering uses `autoPauseRedraw(false)` to stay active even after simulation cools

### Viewport management

- **zoomToFit** — graph is centered on first load
- **Viewport-preserving updates** — `setGraphDataPreservingView` saves and restores camera position (center + zoom) across graph rebuilds
- **Node drag** — nodes can be repositioned by dragging
- **Manual collision force** — a lightweight loop (every 50ms) pushes overlapping nodes apart by 15% when border distance < 8px

## Controls

| Control | Description |
|---------|-------------|
| **Tags** toggle | Switch between activation-based and tag-based node coloring |
| **Orphans** toggle | Show/hide nodes with no connections |
| **Direct-only** toggle | Show only wikilink edges |
| **Phantom** toggle | Show/hide semantic similarity edges |
| **Hebbian threshold** slider | Minimum weight for Hebbian edges to show (0–1, default 0.3) |
| **Spacing** slider | Network spacing — distance between nodes |
| **Edge length** slider | Edge length multiplier |
| **Query bar** | Type a query and press Enter to run retrieval + Hebbian update + LLM response |
| **Stats counters** | Live neuron, synapse, and Hebbian counts — update from server truth after each query |

Slider values are persisted in `localStorage` and restored on page refresh.

## Tooltips

- **Node tooltip** — shows on hover: note title, tags, vault path, degree (connection count), and a text preview of the note content. Dismiss on tap (mobile)
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
- `activated_from` stores the source notes used for generation.

The parser excludes frontmatter from note embeddings, so generation metadata remains available for auditing without becoming a retrieval attractor shared by every generated note.

## WebSocket

The visualization connects to the server via WebSocket for real-time updates:
- **Initial payload** — on connect, sends full graph (nodes with id/title/tags/path/text, edges, hebbian synapses, stats) to populate the visualization
- Auto-reconnect with status indicator
- Receives activation cascades, Hebbian updates, and neurogenesis events as they happen
- Falls back to polling if WebSocket is unavailable

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
Browser (force-graph / WebGL)
  ├── force-graph instance for nodes/edges
  ├── External Maps for activation/filter state
  ├── WebSocket listener → live updates
  ├── Tooltip system (custom HTML overlays)
  └── Tag coloring engine (frontmatter → color map)
```
