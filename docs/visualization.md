# Visualization — BDH Graph Harness

The real-time web visualization at `:8643` renders the knowledge graph with live activation feedback during queries.

## Features

### Node rendering

- **Nodes** are colored by activation state: inactive (gray), seed (blue), activated (orange)
- **Tag-based coloring** — toggle the "Tags" button to color nodes by their Obsidian frontmatter tags. Each tag gets a unique color from a consistent palette
- **Orphan nodes hidden by default** — nodes with zero connections are hidden to reduce clutter. Toggle "Orphans" to show them. The node counter updates to reflect visible vs total

### Edge rendering

- **Wikilink edges** — dark gray, thin lines representing `[[wikilinks]]` from the vault
- **Hebbian synapses** — blue/cyan edges whose width is proportional to synaptic weight
- **Edge tooltips** — hover over any synapse to see: weight, type (hebbian/wikilink), and the connected note titles
- **Live neurogenesis edges** — when a query creates new concept notes and links them to seed notes, the new edges appear on the graph in real-time without requiring a page refresh

### Hebbian pulse animation

During a query, strengthened synapses flash green (`#39d353`), thicken, then settle back:
- Staggered cascade: 120ms delay per edge
- 4-step fade animation over 2.5 seconds
- Edge width temporarily increases proportional to weight gain

### Tag legend

When tag-based coloring is active, a floating legend appears in the bottom-left corner of the graph canvas showing the tag→color mapping. The legend is hidden on mobile (tags are shown in the node tooltip instead).

## Controls

| Control | Description |
|---------|-------------|
| **Tags** toggle | Switch between activation-based and tag-based node coloring |
| **Orphans** toggle | Show/hide nodes with no connections |
| **Query bar** | Type a query and press Enter to run retrieval + Hebbian update + LLM response |
| **Stats counters** | Live neuron, synapse, and Hebbian counts — update from server truth after each query |

## Tooltips

- **Node tooltip** — shows on hover: note title, tags, vault path, degree (connection count), and a text preview of the note content. Dismiss on tap (mobile)
- **Edge tooltip** — shows on hover: synaptic weight, edge type, connected note titles

## Mobile support

The visualization is fully responsive:
- **Tab-based layout** at ≤768px — controls and graph on separate tabs
- **iPhone safe area** — respects `env(safe-area-inset-top/bottom)` for notch/Dynamic Island
- **Touch dismiss** — tap anywhere to dismiss open tooltips
- **Tag legend hidden** on mobile to save space (tags visible in node tooltips)

## WebSocket

The visualization connects to the server via WebSocket for real-time updates:
- **Initial payload** — on connect, sends full graph (nodes with id/title/tags/path/text, edges, hebbian synapses, stats) to populate the visualization
- Auto-reconnect with status indicator
- Receives activation cascades, Hebbian updates, and neurogenesis events as they happen
- Falls back to polling if WebSocket is unavailable

## Dark theme

The entire UI uses a dark theme with:
- Dark background (`#0d1117`)
- Subtle grid pattern on the graph canvas
- High-contrast node/edge colors for readability
- Smooth animations with CSS transitions

## Architecture

```
Server (aiohttp) 
  ├── REST endpoints (query, stats, graph, hebbian, refresh)
  └── WebSocket (/ws) — real-time events
        ↓
Browser (vis.js)
  ├── vis.DataSet for nodes/edges
  ├── WebSocket listener → live updates
  ├── Tooltip system (custom HTML overlays)
  └── Tag coloring engine (frontmatter → color map)
```
