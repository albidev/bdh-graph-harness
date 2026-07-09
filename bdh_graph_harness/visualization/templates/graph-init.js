// ============================================================================
// Graph initialization
// ============================================================================
function initNetwork(graphData) {
  // Store full node list for orphan toggle + build fast lookup map
  allGraphNodes = graphData.nodes;
  nodeDataMap = {};
  graphData.nodes.forEach(n => { nodeDataMap[n.id] = n; });

  // Build tag color map from unique tags
  const tagSet = new Set();
  graphData.nodes.forEach(n => {
    const tags = n.tags || '';
    if (Array.isArray(tags)) {
      tags.forEach(t => tagSet.add(t.replace(/^[\[\]]+|[\[\]]+$/g, '').trim()));
    } else if (typeof tags === 'string' && tags.trim()) {
      tags.replace(/^[\[\]]+|[\[\]]+$/g, '').split(',').forEach(t => {
        const clean = t.trim();
        if (clean) tagSet.add(clean);
      });
    }
  });
  tagColorMap = {};
  let ci = 0;
  const seenTags = new Set();
  tagSet.forEach(tag => {
    const t = (tag || '').trim();
    if (t && !seenTags.has(t.toLowerCase())) {
      seenTags.add(t.toLowerCase());
      tagColorMap[t] = TAG_COLORS[ci++ % TAG_COLORS.length];
    }
  });

  // Build edge set and compute degree/neighbor maps
  const edgeSet = new Set();
  const connectedNodes = new Set();
  degreeMap = {};
  neighborMap = {};

  graphData.edges.forEach(e => {
    const eid = e.source + '→' + e.target;
    if (edgeSet.has(eid)) return;
    edgeSet.add(eid);
    connectedNodes.add(e.source);
    connectedNodes.add(e.target);
    degreeMap[e.source] = (degreeMap[e.source] || 0) + 1;
    degreeMap[e.target] = (degreeMap[e.target] || 0) + 1;
    if (!neighborMap[e.source]) neighborMap[e.source] = [];
    if (!neighborMap[e.target]) neighborMap[e.target] = [];
    const srcNode = graphData.nodes.find(n => n.id === e.source);
    const tgtNode = graphData.nodes.find(n => n.id === e.target);
    if (tgtNode) neighborMap[e.source].push(tgtNode.title || e.target);
    if (srcNode) neighborMap[e.target].push(srcNode.title || e.source);
  });

  // Add hebbian edges to degree count
  hebbianMap = {};
  graphData.hebbian.forEach(h => {
    const key = h.note_a + '|' + h.note_b;
    hebbianMap[key] = h.weight;
    connectedNodes.add(h.note_a);
    connectedNodes.add(h.note_b);
    degreeMap[h.note_a] = (degreeMap[h.note_a] || 0) + 1;
    degreeMap[h.note_b] = (degreeMap[h.note_b] || 0) + 1;
  });

  // Phantom edges degree count
  (graphData.phantom || []).forEach(p => {
    connectedNodes.add(p.source);
    connectedNodes.add(p.target);
    degreeMap[p.source] = (degreeMap[p.source] || 0) + 1;
    degreeMap[p.target] = (degreeMap[p.target] || 0) + 1;
  });

  // Identify orphans
  const connectedSet = new Set(connectedNodes);
  orphanNodeIds = graphData.nodes.filter(n => !connectedSet.has(n.id)).map(n => n.id);

  // Size nodes by degree
  const maxDeg = Math.max(1, ...Object.values(degreeMap));
  const HUB_THRESHOLD = maxDeg * 0.4;

  // Build force-graph nodes
  fgNodes = [];
  const activeNodeIds = showOrphans
    ? graphData.nodes.map(n => n.id)
    : graphData.nodes.filter(n => connectedSet.has(n.id)).map(n => n.id);

  activeNodeIds.forEach(id => {
    const n = graphData.nodes.find(x => x.id === id);
    if (!n) return;
    const deg = degreeMap[id] || 0;
    const isDormant = n.dormant === true;
    const qualityScore = n.quality_score || 0;
    const tags = (n.tags || '').toLowerCase();

    // Determine shape
    let shape = 'circle';
    if (isDormant) shape = 'triangleDown';
    else if (tags.includes('neurogenesis')) shape = 'diamond';
    else if (deg >= HUB_THRESHOLD) shape = 'hexagon';

    // Determine color
    let color = COLORS.inactive;
    if (isDormant) {
      color = COLORS.dormant;
    } else if (showTagColors && tagColorMap) {
      const rawTags = n.tags || '';
      let primaryTag = '';
      if (Array.isArray(rawTags) && rawTags.length > 0) {
        primaryTag = rawTags[0].replace(/^[\[\]]+|[\[\]]+$/g, '').trim();
      } else if (typeof rawTags === 'string' && rawTags.trim()) {
        primaryTag = rawTags.replace(/^[\[\]]+|[\[\]]+$/g, '').split(',')[0].trim();
      }
      if (primaryTag && tagColorMap[primaryTag]) {
        color = tagColorMap[primaryTag];
      }
    }
    nodeTagColorMap[n.id] = color;

    // val = node size (area), scaled by degree
    const MIN_VAL = 4, MAX_VAL = 40;
    const val = MIN_VAL + (deg / maxDeg) * (MAX_VAL - MIN_VAL);

    fgNodes.push({
      id: n.id,
      name: n.title || n.id,
      color: color,
      val: val,
      // Random initial positions to prevent all-at-origin collapse
      x: (Math.random() - 0.5) * 1000,
      y: (Math.random() - 0.5) * 1000,
      _opacity: isDormant ? 0.3 : 1.0,
      _shape: shape,
      _dormant: isDormant,
      _qualityScore: qualityScore,
      _tags: n.tags || '',
      _title: n.title || n.id,
      _path: n.path || '',
      _text: n.text || '',
    });
  });

  // Build force-graph links
  fgLinks = [];
  edgeInfoMap = {};
  const nodeIdSet = new Set(activeNodeIds);

  // === Z-ORDER: links drawn in array order, last = on top ===
  // 1. Wikilinks (bottom — thin, least important)
  graphData.edges.forEach(e => {
    if (!nodeIdSet.has(e.source) || !nodeIdSet.has(e.target)) return;
    const eid = e.source + '→' + e.target;
    const srcTitle = (graphData.nodes.find(n => n.id === e.source) || {}).title || e.source;
    const tgtTitle = (graphData.nodes.find(n => n.id === e.target) || {}).title || e.target;
    edgeInfoMap[eid] = { source_title: srcTitle, target_title: tgtTitle, type: 'wikilink' };
    fgLinks.push({
      source: e.source,
      target: e.target,
      color: COLORS.edgeWikilink,
      width: 0.5,
      type: 'wikilink',
      particles: 0,
      _id: eid,
      _visible: true,
    });
  });

  // 2. Phantom links (middle — dashed, medium importance)
  (graphData.phantom || []).forEach(p => {
    if (!nodeIdSet.has(p.source) || !nodeIdSet.has(p.target)) return;
    const eid = 'phantom_' + p.source + '→' + p.target;
    const srcTitle = (graphData.nodes.find(n => n.id === p.source) || {}).title || p.source;
    const tgtTitle = (graphData.nodes.find(n => n.id === p.target) || {}).title || p.target;
    edgeInfoMap[eid] = { source_title: srcTitle, target_title: tgtTitle, type: 'phantom', similarity: p.similarity };
    fgLinks.push({
      source: p.source,
      target: p.target,
      color: COLORS.edgePhantom,
      width: 1.5,
      type: 'phantom',
      _id: eid,
      _dashes: true,
      _visible: true,
    });
  });

  // 3. Hebbian synapses (top — thickest, most important)
  // Filter: only render hebbian edges above a weight threshold to avoid
  // rendering 3000+ thin edges that turn the graph into an unreadable blob.
  // The threshold is configurable via the HEBBIAN_MIN_RENDER_WEIGHT constant.
  const HEBBIAN_MIN_RENDER_WEIGHT = 0.15;
  let hebbianRendered = 0;
  let hebbianSkipped = 0;
  graphData.hebbian.forEach(h => {
    if (!nodeIdSet.has(h.note_a) || !nodeIdSet.has(h.note_b)) return;
    if (h.weight < HEBBIAN_MIN_RENDER_WEIGHT) {
      hebbianSkipped++;
      return;
    }
    const eid = 'hebb_' + h.note_a + '→' + h.note_b;
    const srcTitle = (graphData.nodes.find(n => n.id === h.note_a) || {}).title || h.note_a;
    const tgtTitle = (graphData.nodes.find(n => n.id === h.note_b) || {}).title || h.note_b;
    edgeInfoMap[eid] = { source_title: srcTitle, target_title: tgtTitle, type: 'hebbian', weight: h.weight, frequency: h.frequency };
    const w = Math.min(1 + h.weight * 3, 5);
    fgLinks.push({
      source: h.note_a,
      target: h.note_b,
      color: weightColor(h.weight),
      width: w,
      type: 'hebbian',
      weight: h.weight,
      frequency: h.frequency,
      particles: 0,
      _id: eid,
      _visible: true,
    });
    hebbianRendered++;
  });
  if (hebbianSkipped > 0) {
    console.log(`[BDH] Hebbian edges: ${hebbianRendered} rendered, ${hebbianSkipped} skipped (weight < ${HEBBIAN_MIN_RENDER_WEIGHT})`);
  }

  // Create or recreate force-graph instance
  const container = document.getElementById('graph-container');
  if (graph) {
    graph._destructor();
    graph = null;
  }

  graph = ForceGraph()(container)
    .graphData({ nodes: fgNodes, links: fgLinks })
    .backgroundColor(COLORS.bg)
    .nodeRelSize(20)
    .nodeId('id')
    .nodeVal(node => node.val)
    .nodeLabel(() => null)
    .nodeColor(node => node.color)
    .nodeCanvasObject(drawNode)
    .nodeCanvasObjectMode(() => 'replace')
    .nodePointerAreaPaint((node, color, ctx, globalScale) => {
      // Generous hit area — larger than the visual node so hovering is forgiving
      const r = Math.sqrt((node.val || 4) * 20) + 6 / globalScale;
      const shape = node._shape || 'circle';
      ctx.fillStyle = color;
      if (shape === 'diamond') {
        ctx.beginPath();
        ctx.moveTo(node.x, node.y - r);
        ctx.lineTo(node.x + r, node.y);
        ctx.lineTo(node.x, node.y + r);
        ctx.lineTo(node.x - r, node.y);
        ctx.closePath();
        ctx.fill();
      } else if (shape === 'triangleDown') {
        ctx.beginPath();
        ctx.moveTo(node.x, node.y + r);
        ctx.lineTo(node.x + r, node.y - r * 0.7);
        ctx.lineTo(node.x - r, node.y - r * 0.7);
        ctx.closePath();
        ctx.fill();
      } else if (shape === 'hexagon') {
        ctx.beginPath();
        for (let i = 0; i < 6; i++) {
          const angle = (Math.PI / 3) * i;
          const px = node.x + r * Math.cos(angle);
          const py = node.y + r * Math.sin(angle);
          if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.fill();
      } else {
        ctx.beginPath();
        ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
        ctx.fill();
      }
    })
    .nodeVisibility(node => !node._hidden)
    .linkSource('source')
    .linkTarget('target')
    .linkVisibility(link => {
      if (linkActivationVisible.get(linkKey(link)) === false) return false;
      const vis = linkVisibilityState.get(linkKey(link));
      return vis !== false; // undefined = visible (default)
    })
    .linkColor(hoverAwareLinkColor)
    .linkWidth(hoverAwareLinkWidth)
    .linkCurvature(0.15)
    .linkLineDash(link => link._dashes ? [5, 5] : null)
    .linkDirectionalParticles(hoverAwareParticles)
    .linkDirectionalParticleSpeed(0.006)
    .linkDirectionalParticleWidth(2)
    .linkDirectionalParticleColor(link => {
      const key = linkKey(link);
      return linkParticleColorState.get(key) || link.particleColor || link.color;
    })
    .warmupTicks(100)
    .cooldownTime(30000)
    .autoPauseRedraw(false)
    .enableNodeDrag(true)
    .onNodeHover((node, prevNode) => {
      if (node) {
        clearHoverEdge(false);
        setHoverHighlight([node.id]);
        const evt = lastMouseEvent || { clientX: 0, clientY: 0 };
        showTooltip(node, evt);
      } else {
        scheduleClearHoverHighlight();
      }
    })
    .onLinkHover((link, prevLink) => {
      if (link) {
        // Edge hover highlights only the hovered edge. Full graph highlighting stays
        // tied to nodes; edges are too easy to brush accidentally.
        setHoverEdge(link._id);
        const evt = lastMouseEvent || { clientX: 0, clientY: 0 };
        showEdgeTooltip(link, evt);
      } else {
        clearHoverEdge();
        hideTooltip();
      }
    })
    .onZoom(() => {
      syncZoomUI(true);
    });

  // Physics — tuned for readable graph layout with 500+ nodes
  graph.d3Force('charge').strength(-800);
  graph.d3Force('link').distance(80);
  graph.d3Force('center').strength(0.05);
  // Add collision force via d3-force to prevent node overlap
  // This is more efficient than the manual O(N²) loop below
  if (graph.d3Force('collision')) {
    graph.d3Force('collision').radius(node => Math.sqrt((node.val || 4) * 20) + 4);
  }
  // Manual collision force — grid-based spatial hash for O(N) performance.
  // With 584 nodes, the old O(N²) loop did 170k comparisons per tick.
  // This grid approach bins nodes into cells and only checks neighbors.
  if (!window.__collisionForceInstalled) {
    window.__collisionForceInstalled = true;
    (function installCollisionForce() {
      const MIN_GAP = 8; // minimum pixel gap between node edges
      let lastTick = 0;
      function tick() {
        if (!graph) return;
        const now = performance.now();
        if (now - lastTick < 50) { requestAnimationFrame(tick); return; }
        lastTick = now;
        const data = graph.graphData();
        if (!data || !data.nodes) { requestAnimationFrame(tick); return; }
        const nodes = data.nodes;
        const N = nodes.length;

        // Compute bounding box and cell size
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (let i = 0; i < N; i++) {
          const n = nodes[i];
          if (n.x == null || n.y == null) continue;
          if (n.x < minX) minX = n.x;
          if (n.x > maxX) maxX = n.x;
          if (n.y < minY) minY = n.y;
          if (n.y > maxY) maxY = n.y;
        }
        if (minX === Infinity) { requestAnimationFrame(tick); return; }

        // Cell size = max node radius + MIN_GAP (ensures neighbors are in adjacent cells)
        const CELL_SIZE = 80; // ~sqrt(20*5) + MIN_GAP + slack
        const cols = Math.max(1, Math.ceil((maxX - minX) / CELL_SIZE) + 2);
        const rows = Math.max(1, Math.ceil((maxY - minY) / CELL_SIZE) + 2);

        // Build spatial hash grid
        const grid = new Array(cols * rows);
        for (let i = 0; i < N; i++) {
          const n = nodes[i];
          if (n.x == null || n.y == null) continue;
          const cx = Math.floor((n.x - minX) / CELL_SIZE) + 1;
          const cy = Math.floor((n.y - minY) / CELL_SIZE) + 1;
          const idx = cy * cols + cx;
          if (!grid[idx]) grid[idx] = [];
          grid[idx].push(i);
        }

        // Check collisions only within 3x3 neighborhood of each cell
        for (let gy = 1; gy < rows - 1; gy++) {
          for (let gx = 1; gx < cols - 1; gx++) {
            const cellIdx = gy * cols + gx;
            const cell = grid[cellIdx];
            if (!cell) continue;
            // Check against same cell + 8 neighbors
            for (let dy = -1; dy <= 1; dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                const nIdx = (gy + dy) * cols + (gx + dx);
                const nCell = grid[nIdx];
                if (!nCell) continue;
                for (const i of cell) {
                  const a = nodes[i];
                  const ra = Math.sqrt((a.val || 4) * 20);
                  for (const j of nCell) {
                    if (j <= i) continue; // avoid double-checking pairs
                    const b = nodes[j];
                    const rb = Math.sqrt((b.val || 4) * 20);
                    const ddx = b.x - a.x;
                    const ddy = b.y - a.y;
                    const dist = Math.sqrt(ddx * ddx + ddy * ddy);
                    const minDist = ra + rb + MIN_GAP;
                    if (dist < minDist && dist > 0.01) {
                      const push = (minDist - dist) * 0.15;
                      const nx = ddx / dist;
                      const ny = ddy / dist;
                      a.x -= nx * push;
                      a.y -= ny * push;
                      b.x += nx * push;
                      b.y += ny * push;
                    }
                  }
                }
              }
            }
          }
        }
        requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    })();
  }
  applyRestoredGraphControls();
  // Let simulation settle before rendering
  graph.d3ReheatSimulation();

  // Track mouse position globally for tooltip positioning. Install once only —
  // initNetwork can run after WS graph_refresh/full reloads.
  if (!mouseTrackingInstalled) {
    document.addEventListener('mousemove', (e) => {
      lastMouseEvent = e;
      if (tooltipEl && tooltipEl.style.display === 'block') {
        positionTooltip(e);
      }
    });
    mouseTrackingInstalled = true;
  }

  // onZoom callback handles slider sync (no polling needed)

  // Update stats
  document.getElementById('stat-neurons').textContent = graphData.stats.neurons;
  document.getElementById('stat-synapses').textContent = graphData.stats.synapses;
  document.getElementById('stat-hebbian').textContent = graphData.stats.hebbian_synapses;

  // NOTE: applyEdgeFilters() is NOT called here — it calls setGraphDataPreservingView
  // which calls graph.graphData() again, breaking all interactions (drag/zoom/click).
  // Instead, initEdgeVisibility() populates the visibility Map without rebuilding the graph.
  // This applies saved hebbian threshold/phantom/directOnly from localStorage.
  initEdgeVisibility();

  // Center viewport on graph — must be last
  graph.zoomToFit(400);

  // Show dormant count if > 0
  const dormantEl = document.getElementById('stat-dormant');
  const dormantCountEl = document.getElementById('stat-dormant-count');
  const dormantCount = graphData.stats.dormant_neurons || 0;
  if (dormantCount > 0) {
    dormantEl.style.display = '';
    dormantCountEl.textContent = dormantCount;
  } else {
    dormantEl.style.display = 'none';
  }

  // Show phantom link count if > 0
  const phantomEl = document.getElementById('stat-phantom');
  const phantomCountEl = document.getElementById('stat-phantom-count');
  const phantomCount = graphData.stats.phantom_links || 0;
  if (phantomCount > 0) {
    phantomEl.style.display = '';
    phantomCountEl.textContent = phantomCount;
  } else {
    phantomEl.style.display = 'none';
  }
}

// Repaint mutated node/link style without resetting the force-graph viewport.
// Calling graph.graphData(currentData) for every query highlight makes force-graph
// recalculate bounds/transform and the graph appears to jump out of the window.
function requestGraphRedraw() {
  if (!graph) return;
  if (typeof graph.autoPauseRedraw === 'function') graph.autoPauseRedraw(false);
  if (typeof graph.resumeAnimation === 'function') graph.resumeAnimation();
}

// Use only for structural changes (added/removed nodes or links). Preserve the
// user's current pan/zoom so live updates don't throw the graph off-screen.
function setGraphDataPreservingView(data, opts = {}) {
  if (!graph) return;
  const center = typeof graph.centerAt === 'function' ? graph.centerAt() : null;
  const zoom = typeof graph.zoom === 'function' ? graph.zoom() : null;
  // Build a fresh graph. NEVER spread force-graph internal objects — their
  // properties may have non-writable descriptors that crash in Safari/WebKit.
  const safeData = {
    nodes: (data.nodes || []).map(node => ({
      id: node.id, name: node.name, color: node.color, val: node.val,
      _opacity: node._opacity, _shape: node._shape, _dormant: node._dormant,
      _tags: node._tags, _title: node._title, _path: node._path, _text: node._text,
      _hidden: node._hidden,
    })),
    links: (data.links || []).map(link => {
      const s = typeof link.source === 'object' ? link.source.id : link.source;
      const t = typeof link.target === 'object' ? link.target.id : link.target;
      return {
        source: s, target: t,
        color: link.color, width: link.width, type: link.type,
        particles: link.particles, _id: link._id, _visible: link._visible,
        _dashes: link._dashes, weight: link.weight, frequency: link.frequency,
        particleColor: link.particleColor,
      };
    }),
  };
  // Z-ORDER: sort links so important ones render on top (last in array = drawn last)
  safeData.links.sort((a, b) => {
    const order = { wikilink: 0, phantom: 1, hebbian: 2 };
    return (order[a.type] || 0) - (order[b.type] || 0);
  });
  graph.graphData(safeData);
  if (opts.reheat && typeof graph.d3ReheatSimulation === 'function') graph.d3ReheatSimulation();
  requestAnimationFrame(() => {
    if (center && typeof graph.centerAt === 'function') graph.centerAt(center.x, center.y, 0);
    if (zoom != null && typeof graph.zoom === 'function') graph.zoom(zoom, 0);
    if (typeof syncZoomUI === 'function') syncZoomUI();
  });
  return safeData;
}
