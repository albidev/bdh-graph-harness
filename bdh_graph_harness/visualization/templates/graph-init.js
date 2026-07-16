// ============================================================================
// Graph initialization
// ============================================================================
function initNetwork(graphData) {
  // Keep the server payload so source filters can rebuild the view without
  // losing hidden nodes or cross-source edges.
  sourceGraphData = JSON.parse(JSON.stringify(graphData));
  // Store full node list for orphan toggle + build fast lookup map
  allGraphNodes = graphData.nodes;
  const sourceVisibleIds = new Set(
    graphData.nodes
      .filter(nodeMatchesSourceFilter)
      .map(node => node.id)
  );
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
  const counterpartSet = new Set();
  const connectedNodes = new Set();
  degreeMap = {};
  neighborMap = {};

  graphData.edges.forEach(e => {
    if (!sourceVisibleIds.has(e.source) || !sourceVisibleIds.has(e.target)) return;
    const type = e.type || 'wikilink';
    if (type === 'counterpart') {
      const counterpartKey = [e.source, e.target].sort().join('↔');
      if (counterpartSet.has(counterpartKey)) return;
      counterpartSet.add(counterpartKey);
    }
    const eid = e.source + '→' + e.target;
    if (edgeSet.has(eid)) return;
    edgeSet.add(eid);
    connectedNodes.add(e.source);
    connectedNodes.add(e.target);
    degreeMap[e.source] = (degreeMap[e.source] || 0) + 1;
    degreeMap[e.target] = (degreeMap[e.target] || 0) + 1;
    if (!neighborMap[e.source]) neighborMap[e.source] = [];
    if (!neighborMap[e.target]) neighborMap[e.target] = [];
    const srcNode = nodeDataMap[e.source];
    const tgtNode = nodeDataMap[e.target];
    if (tgtNode) neighborMap[e.source].push(tgtNode.title || e.target);
    if (srcNode) neighborMap[e.target].push(srcNode.title || e.source);
  });

  // Add hebbian edges to degree count
  hebbianMap = {};
  graphData.hebbian.forEach(h => {
    if (!sourceVisibleIds.has(h.note_a) || !sourceVisibleIds.has(h.note_b)) return;
    const key = h.note_a + '|' + h.note_b;
    hebbianMap[key] = h.weight;
    connectedNodes.add(h.note_a);
    connectedNodes.add(h.note_b);
    degreeMap[h.note_a] = (degreeMap[h.note_a] || 0) + 1;
    degreeMap[h.note_b] = (degreeMap[h.note_b] || 0) + 1;
  });

  // Phantom edges degree count
  (graphData.phantom || []).forEach(p => {
    if (!sourceVisibleIds.has(p.source) || !sourceVisibleIds.has(p.target)) return;
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
    ? graphData.nodes.filter(n => sourceVisibleIds.has(n.id)).map(n => n.id)
    : graphData.nodes.filter(n => sourceVisibleIds.has(n.id) && connectedSet.has(n.id)).map(n => n.id);

  // Synaptic aura score: nodes attached to strong rendered Hebbian edges glow a
  // little more, so the visual density follows learned plasticity instead of
  // looking like isolated colored pins on top of the links.
  const activeNodeSet = new Set(activeNodeIds);
  const synapticGlowRaw = {};
  graphData.hebbian.forEach(h => {
    if (!activeNodeSet.has(h.note_a) || !activeNodeSet.has(h.note_b)) return;
    if (h.weight < HEBBIAN_MIN_RENDER_WEIGHT) return;
    synapticGlowRaw[h.note_a] = (synapticGlowRaw[h.note_a] || 0) + h.weight;
    synapticGlowRaw[h.note_b] = (synapticGlowRaw[h.note_b] || 0) + h.weight;
  });
  const maxSynapticGlow = Math.max(1, ...Object.values(synapticGlowRaw));

  activeNodeIds.forEach(id => {
    const n = nodeDataMap[id];
    if (!n) return;
    const deg = degreeMap[id] || 0;
    const isDormant = n.dormant === true;
    const qualityScore = n.quality_score || 0;
    const tags = Array.isArray(n.tags)
      ? n.tags.join(',').toLowerCase()
      : String(n.tags || '').toLowerCase();
    const isNeurogenesis = tags.includes('neurogenesis');

    // Determine shape
    let shape = 'circle';
    if (isDormant) shape = 'triangleDown';
    else if (tags.includes('neurogenesis')) shape = 'diamond';
    else if (deg >= HUB_THRESHOLD) shape = 'hexagon';

    // Determine color
    let color = COLORS.inactive;
    if (isDormant) {
      color = COLORS.dormant;
    } else if (isNeurogenesis) {
      // Neurogenesis is an identity, not a tag category: keep it ultraviolet.
      color = COLORS.neurogenesis;
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
    } else {
      color = sourceColor(n);
    }
    nodeTagColorMap[n.id] = color;
    const mass = computeNodeMass(n, deg, maxDeg);

    // val = node size (area), scaled by degree
    const MIN_VAL = 4, MAX_VAL = 40;
    const val = MIN_VAL + (deg / maxDeg) * (MAX_VAL - MIN_VAL);

    fgNodes.push({
      id: n.id,
      name: n.display_label || n.title || n.id,
      color: color,
      val: val,
      _mass: mass,
      _synapticGlow: Math.min(1, (synapticGlowRaw[n.id] || 0) / maxSynapticGlow),
      // Random initial positions to prevent all-at-origin collapse
      x: (Math.random() - 0.5) * 1000,
      y: (Math.random() - 0.5) * 1000,
      _opacity: isDormant ? 0.3 : 1.0,
      _shape: shape,
      _dormant: isDormant,
      _qualityScore: qualityScore,
      _tags: n.tags || '',
      _title: n.title || n.id,
      _displayLabel: n.display_label || n.title || n.id,
      _path: n.path || '',
      _text: n.text || '',
    });
  });

  // One-time tag/community layout seeding. This is deliberately NOT a perpetual
  // force: an endless centroid pull keeps shrinking each color/tag cluster after
  // d3 cools down. Dragging a node reheats d3, so it briefly expands, then the
  // centroid pull collapses it again. Seed clusters once, then let charge/link/
  // collision own the physics.
  const tagCounts = {};
  fgNodes.forEach(n => {
    const tags = n._tags || '';
    let primaryTag = '';
    if (Array.isArray(tags) && tags.length > 0) {
      primaryTag = tags[0].replace(/^[\[\]]+|[\[\]]+$/g, '').trim();
    } else if (typeof tags === 'string' && tags.trim()) {
      primaryTag = tags.replace(/^[\[\]]+|[\[\]]+$/g, '').split(',')[0].trim();
    }
    if (!primaryTag) primaryTag = '_unsorted';
    n._cluster = primaryTag;
    tagCounts[primaryTag] = (tagCounts[primaryTag] || 0) + 1;
  });

  const tagList = Object.keys(tagCounts).sort((a, b) => tagCounts[b] - tagCounts[a]);
  const clusterCenters = {};
  const clusterRadius = 900 + tagList.length * 45;
  tagList.forEach((tag, i) => {
    const angle = (i / Math.max(1, tagList.length)) * 2 * Math.PI;
    clusterCenters[tag] = { x: Math.cos(angle) * clusterRadius, y: Math.sin(angle) * clusterRadius };
  });

  const clusterSeen = {};
  fgNodes.forEach(n => {
    const center = clusterCenters[n._cluster] || { x: 0, y: 0 };
    const idx = clusterSeen[n._cluster] || 0;
    clusterSeen[n._cluster] = idx + 1;
    const count = Math.max(1, tagCounts[n._cluster] || 1);
    const angle = idx * Math.PI * (3 - Math.sqrt(5));
    const radius = 30 + Math.sqrt(idx / count) * Math.max(160, Math.sqrt(count) * 42);
    n.x = center.x + Math.cos(angle) * radius + (Math.random() - 0.5) * 35;
    n.y = center.y + Math.sin(angle) * radius + (Math.random() - 0.5) * 35;
  });

  // Build force-graph links
  fgLinks = [];
  edgeInfoMap = {};
  const nodeIdSet = new Set(activeNodeIds);

  // === Z-ORDER: links drawn in array order, last = on top ===
  // 1. Structural/direct links (bottom — wikilinks thin, counterparts highlighted)
  graphData.edges.forEach(e => {
    if (!nodeIdSet.has(e.source) || !nodeIdSet.has(e.target)) return;
    const type = e.type || 'wikilink';
    if (type === 'counterpart') {
      const counterpartKey = [e.source, e.target].sort().join('↔');
      if (counterpartSet.has(counterpartKey + ':rendered')) return;
      counterpartSet.add(counterpartKey + ':rendered');
    }
    const eid = e.source + '→' + e.target;
    const srcNode = nodeDataMap[e.source] || {};
    const tgtNode = nodeDataMap[e.target] || {};
    const srcTitle = srcNode.display_label || srcNode.title || e.source;
    const tgtTitle = tgtNode.display_label || tgtNode.title || e.target;
    edgeInfoMap[eid] = {
      source_title: srcTitle,
      target_title: tgtTitle,
      type: type,
      relation: e.relation,
      group_id: e.group_id,
    };
    const counterpart = type === 'counterpart';
    const projectContext = type === 'project_context';
    fgLinks.push({
      source: e.source,
      target: e.target,
      color: counterpart ? COLORS.edgeCounterpart : (projectContext ? COLORS.edgeProjectContext : COLORS.edgeWikilink),
      width: counterpart ? 2.2 : (projectContext ? 1.2 : 0.5),
      type: type,
      relation: e.relation,
      group_id: e.group_id,
      particles: 0,
      _id: eid,
      _dashes: counterpart || projectContext,
      _visible: true,
    });
  });

  // 2. Phantom links (middle — dashed, medium importance)
  (graphData.phantom || []).forEach(p => {
    if (!nodeIdSet.has(p.source) || !nodeIdSet.has(p.target)) return;
    const eid = 'phantom_' + p.source + '→' + p.target;
    const srcTitle = (nodeDataMap[p.source] || {}).title || p.source;
    const tgtTitle = (nodeDataMap[p.target] || {}).title || p.target;
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
  let hebbianRendered = 0;
  let hebbianSkipped = 0;
  graphData.hebbian.forEach(h => {
    if (!nodeIdSet.has(h.note_a) || !nodeIdSet.has(h.note_b)) return;
    if (h.weight < HEBBIAN_MIN_RENDER_WEIGHT) {
      hebbianSkipped++;
      return;
    }
    const eid = 'hebb_' + h.note_a + '→' + h.note_b;
    const srcTitle = (nodeDataMap[h.note_a] || {}).title || h.note_a;
    const tgtTitle = (nodeDataMap[h.note_b] || {}).title || h.note_b;
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
      const r = nodeRadius(node.val || 4) + 6 / globalScale;
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
    .linkDirectionalParticleSpeed(particleSpeed)
    .linkDirectionalParticleWidth(hoverAwareParticleWidth)
    .linkDirectionalParticleColor(link => {
      const key = linkKey(link);
      return linkParticleColorState.get(key) || link.particleColor || link.color;
    })
    .linkDirectionalParticleCanvasObject(drawNeuralParticle)
    .warmupTicks(100)
    .cooldownTime(30000)
    .enableNodeDrag(true)
    .onNodeHover((node, prevNode) => {
      if (node) {
        clearHoverEdge(false);
        setHoverHighlight([node.id]);
        // Hover cards make touch navigation miserable. Mobile opens details on tap.
        if (!isMobile()) {
          const evt = lastMouseEvent || { clientX: 0, clientY: 0 };
          showTooltip(node, evt);
        }
      } else {
        scheduleClearHoverHighlight();
        if (!isMobile()) hideTooltip();
      }
    })
    .onNodeClick((node) => {
      if (!node) return;
      clearHoverEdge(false);
      setHoverHighlight([node.id]);
      const graphArea = document.getElementById('graph-area').getBoundingClientRect();
      showTooltip(node, { clientX: graphArea.left + graphArea.width / 2, clientY: graphArea.top + graphArea.height / 2 });
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
    .onBackgroundClick(() => {
      clearHoverEdge();
      clearHoverHighlight();
      hideTooltip();
    })
    .onZoom(() => {
      syncZoomUI(true);
    });

  // Physics — tuned for readable graph layout with 500+ nodes
  graph.d3Force('charge').strength(node => massAwareChargeStrength(node, -800));
  graph.d3Force('link').distance(link => massAwareLinkDistance(link, 80));
  graph.d3Force('center').strength(0.05);
  // Add collision force via d3-force to prevent node overlap
  // This is more efficient than the manual O(N²) loop below
  if (graph.d3Force('collision')) {
    graph.d3Force('collision').radius(node => nodeRadius(node.val || 4) + 4);
  }

  // Use force-graph's built-in d3 collision force only. A second perpetual
  // requestAnimationFrame collision loop used to fight d3 over node.x/y and
  // kept the page busy forever, especially during graphData() rebuilds.
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

  // Update stats using the currently rendered source-filtered graph.
  const renderedWikilinks = fgLinks.filter(link => link.type === 'wikilink').length;
  const renderedHebbian = fgLinks.filter(link => link.type === 'hebbian').length;
  document.getElementById('stat-neurons').textContent = fgNodes.length;
  document.getElementById('stat-synapses').textContent = renderedWikilinks;
  document.getElementById('stat-hebbian').textContent = renderedHebbian;

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
let redrawRestoreTimer = null;

function requestGraphRedraw() {
  if (!graph) return;
  const hoverActive = (typeof isHoverActive === 'function' && isHoverActive()) || !!hoverEdgeId;
  if (redrawRestoreTimer) {
    clearTimeout(redrawRestoreTimer);
    redrawRestoreTimer = null;
  }
  if (typeof graph.autoPauseRedraw === 'function') graph.autoPauseRedraw(false);
  if (typeof graph.resumeAnimation === 'function') graph.resumeAnimation();
  if (hoverActive) return;

  // One-shot style updates get a short repaint window, then return to idle.
  redrawRestoreTimer = setTimeout(() => {
    redrawRestoreTimer = null;
    if (!graph) return;
    const stillHovering = (typeof isHoverActive === 'function' && isHoverActive()) || !!hoverEdgeId;
    if (!stillHovering && typeof graph.autoPauseRedraw === 'function') {
      graph.autoPauseRedraw(true);
    }
  }, 120);
}

// Use only for structural changes (added/removed nodes or links). Preserve the
// user's current pan/zoom so live updates don't throw the graph off-screen.
function setGraphDataPreservingView(data, opts = {}) {
  if (!graph) return;
  if (!data || !Array.isArray(data.nodes) || data.nodes.length === 0) {
    console.warn('Ignoring empty graph structural update');
    return;
  }
  const center = typeof graph.centerAt === 'function' ? graph.centerAt() : null;
  const zoom = typeof graph.zoom === 'function' ? graph.zoom() : null;
  const liveData = graph.graphData ? graph.graphData() : { nodes: [] };
  const liveNodeById = new Map((liveData.nodes || []).map(n => [n.id, n]));
  const finiteOrUndefined = value => Number.isFinite(value) ? value : undefined;
  // Build a fresh graph. NEVER spread force-graph internal objects — their
  // properties may have non-writable descriptors that crash in Safari/WebKit.
  // Preserve live coordinates across structural updates; otherwise any query that
  // adds a Hebbian/neurogenesis edge makes d3 reinitialize nodes into phyllotaxis
  // packing — the “perfect turtle shell” collapse.
  const safeData = {
    nodes: (data.nodes || []).map(node => {
      const liveNode = liveNodeById.get(node.id) || {};
      return {
      id: node.id, name: node.name, color: node.color, val: node.val,
      x: finiteOrUndefined(node.x) ?? finiteOrUndefined(liveNode.x),
      y: finiteOrUndefined(node.y) ?? finiteOrUndefined(liveNode.y),
      vx: finiteOrUndefined(node.vx) ?? finiteOrUndefined(liveNode.vx),
      vy: finiteOrUndefined(node.vy) ?? finiteOrUndefined(liveNode.vy),
      fx: finiteOrUndefined(node.fx) ?? finiteOrUndefined(liveNode.fx),
      fy: finiteOrUndefined(node.fy) ?? finiteOrUndefined(liveNode.fy),
      _opacity: node._opacity, _shape: node._shape, _dormant: node._dormant,
      _mass: node._mass, _synapticGlow: node._synapticGlow,
      _tags: node._tags, _title: node._title, _path: node._path, _text: node._text,
      _hidden: node._hidden,
    };
    }),
    links: (data.links || []).map(link => {
      const s = typeof link.source === 'object' ? link.source.id : link.source;
      const t = typeof link.target === 'object' ? link.target.id : link.target;
      const type = link.type || 'wikilink';
      const fallbackColor = type === 'wikilink' ? COLORS.edgeWikilink :
        (type === 'phantom' ? COLORS.edgePhantom : COLORS.edgeHebbianMid);
      return {
        source: s, target: t,
        color: link.color || fallbackColor,
        width: Number.isFinite(link.width) ? link.width : (type === 'wikilink' ? 0.5 : 1),
        type,
        particles: link.particles || 0, _id: link._id || (s + '→' + t),
        _visible: link._visible !== false,
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
  if (typeof initEdgeVisibility === 'function') initEdgeVisibility();
  if (opts.reheat && typeof graph.d3ReheatSimulation === 'function') graph.d3ReheatSimulation();
  requestAnimationFrame(() => {
    if (center && typeof graph.centerAt === 'function') graph.centerAt(center.x, center.y, 0);
    if (zoom != null && typeof graph.zoom === 'function') graph.zoom(zoom, 0);
    if (typeof syncZoomUI === 'function') syncZoomUI();
  });
  return safeData;
}
