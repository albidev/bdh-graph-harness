# 3D Visualization Migration — Architecture Assessment

Status: implemented and live-verified on `feature/3d-knowledge-graph-viz`

## Decision

Use [`3d-force-graph` 1.80.0](https://github.com/vasturiano/3d-force-graph) with its native Three.js/WebGL renderer and `d3-force-3d` layout. Load a pinned Three.js 0.180.x ES module only for custom node geometries, label sprites, rings, and dashed semantic links.

This is the smallest reliable migration path. The library preserves the current force-graph data/accessor model, D3 force controls, node dragging, directional particles, hover/click callbacks, and incremental `graphData()` updates. A lower-level Three.js + `d3-force-3d` renderer would provide more batching control, but would require rebuilding picking, camera controls, drag behavior, link particles, force lifecycle, and resize handling. That trade is not justified at the current 353-node / ~2,700-link live scale.

## Implemented graph lifecycle

1. `ui-controls.js` restores persisted controls and vault scope, then opens `/ws`.
2. The initial `graph` WebSocket event triggers a vault-scoped `GET /api/graph`; polling uses the same rich snapshot when WebSocket is unavailable.
3. `initNetwork()` builds node/tag/degree/neighbor maps, transforms every edge family into 3D force-graph objects, creates the renderer once, configures forces, applies filters, and updates stats.
4. Initial mounting deliberately does **not** call `d3ReheatSimulation()` synchronously after `graphData()`. The library creates its internal layout on a deferred update; starting `tickFrame` earlier crashes with an undefined layout.
5. The first camera fit waits until `getGraphBbox()` exposes real rendered bounds. Fog density and node world scale then adapt to the fitted camera distance and viewport height, keeping overview nodes above a two-CSS-pixel radius even on a narrow phone viewport.
6. Local source filtering and structural WebSocket events update graph data while preserving camera position, target, orientation, all three spatial dimensions, and render-object identity.

## Current activation lifecycle

1. `sendQuery()` protects against concurrent requests by disabling the Query button and uses a 120-second `AbortController` timeout.
2. WebSocket activation is the primary rendering owner. The HTTP response updates response text and becomes the activation fallback only when WebSocket is unavailable.
3. `handleActivation()` increments `activationGeneration`; stale timers check the generation before changing state.
4. Node opacity/color, link visibility/color/width, and particle state live in external Maps. Activation does not replace graph data.
5. Missing persisted Hebbian links are collected and submitted in one structural update. Neurogenesis nodes and links are likewise submitted once, then animated through state Maps.
6. Structural refreshes preserve the 3D camera and live `x/y/z`, velocity, and fixed-position values.

## Existing controls and handlers

- Vault selector and vault-scoped REST/WebSocket URLs.
- Source filter: All / Vault / External.
- Orphans, Direct-only, tag colors, phantom visibility.
- Edge family toggles: wikilink, counterpart, project context, project reference, Hebbian, phantom.
- Hebbian threshold, spacing, and edge-length sliders.
- Search, zoom slider/buttons, Fit.
- Query, Reset, activated-note focus, tag legend collapse.
- Side-panel collapse/resize and explicit Graph / Controls / Inspector mobile tabs.
- Node/link hover, node click, background click, drag, touch detail sheet, and touch dismissal.

## Current transport contracts

- `graph`: `sequence`, `vault_id`, `nodes`, `edges`, `hebbian`, `stats`.
- `activation`: `sequence`, `vault_id`, `query`, `activated_notes`, `hebbian_updates`, graph counters, routing metadata, timestamp.
- `neurogenesis`: ordered follow-up containing `activated_notes`, `new_concepts`, counters, and routing metadata.
- `graph_refresh`: `sequence`, `vault_id`, counts, `new_concepts`, `changed_nodes`, `deleted_nodes` where available, and `added_node_data` with node-local edges.
- `node_update`: ordered changed/deleted node delta.
- `ping`: heartbeat.

The WebSocket bootstrap is intentionally thinner than `GET /api/graph`: it currently omits phantom-link records and per-node dormant/quality fields. The migration will use the vault-scoped REST graph snapshot to normalize initial load and polling fallback, while WebSocket remains the real-time event channel. No backend contract change is required.

## Tooltip and mobile baseline

Node tooltips already expose title, source identity, path/open action, tags, degree, neighbors, and preview. Edge tooltips expose type-specific metrics and project metadata. Activated-note tooltips expose retrieval provenance. Mobile uses a safe-area-aware bottom sheet, tap selection, explicit dismissal, and Graph / Controls / Inspector tabs. The Inspector clears the persisted desktop inline width before display, so it always occupies the full mobile viewport.

## Migration risks and mitigations

1. **Canvas APIs do not map to 3D.** `nodeCanvasObject`, pointer-area paint, line dash, `centerAt`, and scalar `zoom()` must become Three.js objects, raycasting, custom dashed lines, and camera-position/target operations.
2. **Camera loss on rebuild.** Persist position, control target, up vector, and focus state; preserve `z/vz/fz` alongside existing coordinates.
3. **Material churn at live scale.** Reuse geometries, textures, and a bounded material cache; synchronize state only when activation, hover, focus, filters, or LOD change.
4. **Label/edge soup.** Cap labels to semantic/interaction candidates and apply camera-distance LOD to weak Hebbian edges, opacity, width, and detail.
5. **Renderer idle cost.** Resume for interaction/animation, then pause after the force engine and meaningful animation settle; never add an independent perpetual animation loop.
6. **Camera fighting the user.** Only explicit search, selection, activated-note focus, Fit, and Reset move the camera. Structural and activation updates do not auto-fit.
7. **Mobile control loss.** Replace the current two-tab layout with explicit Graph / Controls / Inspector tabs; preserve 44px targets and safe areas.
8. **WebGL/library failure.** Show a non-blank fallback with live stats, query UI, and retry guidance; use REST polling when WebSocket is unavailable.
9. **Duplicate Three.js runtimes.** Keep the graph renderer on the pinned `3d-force-graph` bundle and use matching Three.js 0.180.x objects only through public `nodeThreeObject`, `linkThreeObject`, and scene APIs.

## Validation

- Full suite: 286 automated tests pass.
- Focused visualization suite: 25 tests pass.
- Desktop Chromium live path: 354 nodes and 1,046 rendered links, non-zero WebGL draw calls/triangles, no captured runtime errors.
- Mobile Chromium emulation at 390×844: Inspector is 390px wide, scrollable, unclipped, and free of runtime errors.
- Server listens on `0.0.0.0:8643`; HTML, static assets, REST graph/stats, and vault-scoped WebSocket bootstrap were verified through the host's Tailscale address.
