// ============================================================================
// BDH Graph Harness — 3D graph construction, force lifecycle and render control
// ============================================================================

function hashNodeId(id) {
  const value = String(id || '');
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function deterministicUnit(id, salt) {
  let state = (hashNodeId(id) ^ Math.imul(salt + 1, 2654435761)) >>> 0;
  state ^= state << 13;
  state ^= state >>> 17;
  state ^= state << 5;
  return (state >>> 0) / 4294967295;
}

function clusterCenter3D(index, count, radius) {
  const safeCount = Math.max(1, count);
  const y = 1 - 2 * ((index + 0.5) / safeCount);
  const radial = Math.sqrt(Math.max(0, 1 - y * y));
  const theta = Math.PI * (3 - Math.sqrt(5)) * index;
  return {
    x: Math.cos(theta) * radial * radius,
    y: y * radius,
    z: Math.sin(theta) * radial * radius,
  };
}

function seededNodePosition(nodeId, center, index, count) {
  const theta = deterministicUnit(nodeId, 1) * Math.PI * 2;
  const phi = Math.acos(1 - 2 * deterministicUnit(nodeId, 2));
  const radius = 35 + Math.cbrt((index + 1) / Math.max(1, count)) * Math.max(150, Math.cbrt(count) * 75);
  return {
    x: center.x + Math.sin(phi) * Math.cos(theta) * radius,
    y: center.y + Math.cos(phi) * radius,
    z: center.z + Math.sin(phi) * Math.sin(theta) * radius,
  };
}

function graphNodeSnapshot(node, liveNode = {}) {
  return window.BDH3DUtils.cloneNodeForStructuralUpdate(node, liveNode);
}

function graphLinkSnapshot(link) {
  const source = linkEndpointId(link.source);
  const target = linkEndpointId(link.target);
  const type = link.type || 'wikilink';
  const fallbackColor = type === 'wikilink' ? COLORS.edgeWikilink
    : type === 'phantom' ? COLORS.edgePhantom
      : type === 'counterpart' ? COLORS.edgeCounterpart
        : type === 'project_context' ? COLORS.edgeProjectContext
          : type === 'project_reference' ? COLORS.edgeProjectReference
            : type === 'neurogenesis' ? COLORS.edgeNeurogenesis
              : weightColor(link.weight || 0);
  return {
    source,
    target,
    color: link.color || fallbackColor,
    width: Number.isFinite(link.width) ? link.width : 0,
    type,
    relation: link.relation,
    group_id: link.group_id,
    particles: link.particles || 0,
    _id: link._id || `${type}_${source}→${target}`,
    _visible: link._visible !== false,
    _dashes: Boolean(link._dashes),
    weight: link.weight,
    frequency: link.frequency,
    similarity: link.similarity,
    particleColor: link.particleColor,
  };
}

function addNeighbor(source, target) {
  if (!source || !target) return;
  if (!neighborMap[source]) neighborMap[source] = [];
  const targetNode = nodeDataMap[target];
  const title = targetNode ? (targetNode.display_label || targetNode.title || target) : target;
  if (!neighborMap[source].includes(title)) neighborMap[source].push(title);
}

let initialFitPending = false;

function initNetwork(graphData, options = {}) {
  if (!graphData || !Array.isArray(graphData.nodes)) {
    console.warn('[BDH 3D] Ignoring malformed graph snapshot');
    return;
  }

  const firstGraph = !graph;
  const preserveView = options.preserveView !== false && !firstGraph;
  const graphNodes = graphData.nodes || [];
  const graphEdges = graphData.edges || [];
  const graphHebbian = graphData.hebbian || [];
  const graphPhantom = graphData.phantom || [];
  const graphStats = graphData.stats || {};

  sourceGraphData = JSON.parse(JSON.stringify(graphData));
  allGraphNodes = graphNodes;
  nodeDataMap = {};
  graphNodes.forEach(node => { nodeDataMap[node.id] = node; });

  const sourceVisibleIds = new Set(graphNodes.filter(nodeMatchesSourceFilter).map(node => node.id));

  const tagSet = new Set();
  graphNodes.forEach(node => normalizeTags(node.tags || '').forEach(tag => tagSet.add(tag)));
  tagColorMap = {};
  [...tagSet]
    .sort((first, second) => first.localeCompare(second))
    .forEach((tag, index) => { tagColorMap[tag] = TAG_COLORS[index % TAG_COLORS.length]; });

  const connectedNodes = new Set();
  degreeMap = {};
  neighborMap = {};
  const incrementDegree = (source, target) => {
    if (!sourceVisibleIds.has(source) || !sourceVisibleIds.has(target)) return;
    connectedNodes.add(source);
    connectedNodes.add(target);
    degreeMap[source] = (degreeMap[source] || 0) + 1;
    degreeMap[target] = (degreeMap[target] || 0) + 1;
    addNeighbor(source, target);
    addNeighbor(target, source);
  };

  const structuralSeen = new Set();
  graphEdges.forEach(edge => {
    const type = edge.type || 'wikilink';
    const key = `${type}:${edge.source}→${edge.target}`;
    if (structuralSeen.has(key)) return;
    structuralSeen.add(key);
    incrementDegree(edge.source, edge.target);
  });

  hebbianMap = {};
  graphHebbian.forEach(synapse => {
    const key = synapse.note_a + '|' + synapse.note_b;
    hebbianMap[key] = synapse.weight;
    incrementDegree(synapse.note_a, synapse.note_b);
  });
  graphPhantom.forEach(link => incrementDegree(link.source, link.target));

  orphanNodeIds = graphNodes.filter(node => !connectedNodes.has(node.id)).map(node => node.id);
  const maxDegree = Math.max(1, ...Object.values(degreeMap));
  const hubThreshold = Math.max(5, maxDegree * 0.4);
  // When there are no edges, every node is an orphan — show them all regardless of the orphan toggle.
  const showAllOrphans = showOrphans || graphEdges.length === 0;
  const activeNodeIds = graphNodes
    .filter(node => sourceVisibleIds.has(node.id) && (showAllOrphans || connectedNodes.has(node.id)))
    .map(node => node.id);
  const activeNodeSet = new Set(activeNodeIds);

  const synapticGlowRaw = {};
  graphHebbian.forEach(synapse => {
    if (!activeNodeSet.has(synapse.note_a) || !activeNodeSet.has(synapse.note_b)) return;
    if (synapse.weight < HEBBIAN_MIN_RENDER_WEIGHT) return;
    synapticGlowRaw[synapse.note_a] = (synapticGlowRaw[synapse.note_a] || 0) + synapse.weight;
    synapticGlowRaw[synapse.note_b] = (synapticGlowRaw[synapse.note_b] || 0) + synapse.weight;
  });
  const maxSynapticGlow = Math.max(1, ...Object.values(synapticGlowRaw));

  const liveNodeById = new Map();
  if (graph && typeof graph.graphData === 'function') {
    (graph.graphData().nodes || []).forEach(node => liveNodeById.set(node.id, node));
  }

  fgNodes = [];
  activeNodeIds.forEach(id => {
    const data = nodeDataMap[id];
    if (!data) return;
    const degree = degreeMap[id] || 0;
    const tags = normalizeTags(data.tags || '');
    const isNeurogenesis = isNeurogenesisNode({ _tags: data.tags });
    const isDormant = data.dormant === true;
    let shape = 'circle';
    if (isNeurogenesis) shape = 'diamond';
    else if (isDormant) shape = 'triangleDown';
    else if (degree >= hubThreshold) shape = 'hexagon';

    let color = COLORS.inactive;
    if (isNeurogenesis) color = COLORS.neurogenesis;
    else if (isDormant) color = COLORS.dormant;
    else if (showTagColors && tags.length && tagColorMap[tags[0]]) color = tagColorMap[tags[0]];
    else if (!showTagColors) color = sourceColor(data);
    nodeTagColorMap[id] = color;

    const value = 4 + (degree / maxDegree) * 36;
    const live = liveNodeById.get(id) || {};
    fgNodes.push(graphNodeSnapshot({
      id,
      name: data.display_label || data.title || id,
      color,
      val: value,
      _mass: computeNodeMass(data, degree, maxDegree),
      _synapticGlow: Math.min(1, (synapticGlowRaw[id] || 0) / maxSynapticGlow),
      _opacity: isDormant ? 0.84 : 1,
      _shape: shape,
      _dormant: isDormant,
      _qualityScore: data.quality_score || 0,
      _tags: data.tags || '',
      _title: data.title || id,
      _displayLabel: data.display_label || data.title || id,
      _path: data.path || '',
      _text: data.text || '',
    }, live));
  });

  const tagCounts = {};
  fgNodes.forEach(node => {
    const tags = normalizeTags(node._tags || '');
    node._cluster = tags[0] || '_unsorted';
    tagCounts[node._cluster] = (tagCounts[node._cluster] || 0) + 1;
  });
  const tagList = Object.keys(tagCounts).sort((first, second) => tagCounts[second] - tagCounts[first]);
  const clusterRadius = 720 + tagList.length * 34;
  const clusterCenters = {};
  tagList.forEach((tag, index) => {
    clusterCenters[tag] = clusterCenter3D(index, tagList.length, clusterRadius);
  });
  const clusterSeen = {};
  fgNodes.forEach(node => {
    if (Number.isFinite(node.x) && Number.isFinite(node.y) && Number.isFinite(node.z)) return;
    const index = clusterSeen[node._cluster] || 0;
    clusterSeen[node._cluster] = index + 1;
    const position = seededNodePosition(
      node.id,
      clusterCenters[node._cluster] || { x: 0, y: 0, z: 0 },
      index,
      tagCounts[node._cluster] || 1,
    );
    node.x = position.x;
    node.y = position.y;
    node.z = position.z;
  });

  fgLinks = [];
  edgeInfoMap = {};
  const nodeIdSet = new Set(activeNodeIds);
  const renderedStructural = new Set();

  graphEdges.forEach(edge => {
    if (!nodeIdSet.has(edge.source) || !nodeIdSet.has(edge.target)) return;
    const type = edge.type || 'wikilink';
    const pair = [edge.source, edge.target].sort().join('↔');
    const dedupeKey = `${type}:${pair}`;
    if (renderedStructural.has(dedupeKey)) return;
    renderedStructural.add(dedupeKey);
    const id = type === 'wikilink' ? edge.source + '→' + edge.target : `${type}_${edge.source}→${edge.target}`;
    const sourceNode = nodeDataMap[edge.source] || {};
    const targetNode = nodeDataMap[edge.target] || {};
    edgeInfoMap[id] = {
      source_title: sourceNode.display_label || sourceNode.title || edge.source,
      target_title: targetNode.display_label || targetNode.title || edge.target,
      type,
      relation: edge.relation,
      group_id: edge.group_id,
    };
    const color = type === 'counterpart' ? COLORS.edgeCounterpart
      : type === 'project_context' ? COLORS.edgeProjectContext
        : type === 'project_reference' ? COLORS.edgeProjectReference
          : COLORS.edgeWikilink;
    fgLinks.push({
      source: edge.source,
      target: edge.target,
      color,
      width: type === 'counterpart' ? 2.2 : (type === 'project_context' || type === 'project_reference') ? 1.2 : 0.5,
      type,
      relation: edge.relation,
      group_id: edge.group_id,
      _id: id,
      _dashes: type === 'counterpart' || type === 'project_context' || type === 'project_reference',
      _visible: true,
    });
  });

  graphPhantom.forEach(link => {
    if (!nodeIdSet.has(link.source) || !nodeIdSet.has(link.target)) return;
    const id = `phantom_${link.source}→${link.target}`;
    edgeInfoMap[id] = {
      source_title: (nodeDataMap[link.source] || {}).display_label || (nodeDataMap[link.source] || {}).title || link.source,
      target_title: (nodeDataMap[link.target] || {}).display_label || (nodeDataMap[link.target] || {}).title || link.target,
      type: 'phantom',
      similarity: link.similarity,
    };
    fgLinks.push({
      source: link.source,
      target: link.target,
      color: COLORS.edgePhantom,
      width: 1,
      type: 'phantom',
      similarity: link.similarity,
      _id: id,
      _dashes: true,
      _visible: true,
    });
  });

  let hebbianRendered = 0;
  let hebbianSkipped = 0;
  graphHebbian.forEach(synapse => {
    if (!nodeIdSet.has(synapse.note_a) || !nodeIdSet.has(synapse.note_b)) return;
    if (synapse.weight < HEBBIAN_MIN_RENDER_WEIGHT) {
      hebbianSkipped += 1;
      return;
    }
    const id = `hebb_${synapse.note_a}→${synapse.note_b}`;
    edgeInfoMap[id] = {
      source_title: (nodeDataMap[synapse.note_a] || {}).display_label || (nodeDataMap[synapse.note_a] || {}).title || synapse.note_a,
      target_title: (nodeDataMap[synapse.note_b] || {}).display_label || (nodeDataMap[synapse.note_b] || {}).title || synapse.note_b,
      type: 'hebbian',
      weight: synapse.weight,
      frequency: synapse.frequency,
    };
    fgLinks.push({
      source: synapse.note_a,
      target: synapse.note_b,
      color: weightColor(synapse.weight),
      width: Math.min(1 + synapse.weight * 3, 5),
      type: 'hebbian',
      weight: synapse.weight,
      frequency: synapse.frequency,
      _id: id,
      _visible: true,
    });
    hebbianRendered += 1;
  });

  fgLinks.sort((first, second) => {
    const order = { wikilink: 0, phantom: 1, project_context: 2, counterpart: 3, project_reference: 3, hebbian: 4, neurogenesis: 5 };
    return (order[first.type] || 0) - (order[second.type] || 0);
  });

  if (hebbianSkipped > 0) {
    console.info(`[BDH 3D] Hebbian edges: ${hebbianRendered} rendered, ${hebbianSkipped} below ${HEBBIAN_MIN_RENDER_WEIGHT}`);
  }

  updateGraphStats({
    nodes: fgNodes.length,
    structural: graphEdges.length,
    wikilinks: graphEdges.filter(edge => (edge.type || 'wikilink') === 'wikilink').length,
    hebbian: graphHebbian.length,
    dormant: graphStats.dormant_neurons || graphNodes.filter(node => node.dormant).length,
    phantom: graphStats.phantom_links || graphPhantom.length,
  });

  if (!supportsWebGL() || typeof window.ForceGraph3D !== 'function' || !window.THREE) {
    webglUnavailable = true;
    showWebGLFallback('WebGL or the 3D rendering library is unavailable.');
    return;
  }

  const data = {
    nodes: fgNodes.map(node => graphNodeSnapshot(node, liveNodeById.get(node.id) || {})),
    links: fgLinks.map(graphLinkSnapshot),
  };

  if (firstGraph) {
    createGraphInstance();
    graphLayoutActive = true;
    initialFitPending = true;
    graph.graphData(data);
    initEdgeVisibility();
    configureForces();
    resizeGraphToContainer();
    syncThreeVisualState();
    // graphData() creates and heats the internal force layout on its deferred
    // update. Reheating synchronously here starts tickFrame before layout exists.
  } else if (preserveView) {
    setGraphDataPreservingView(data, { reheat: options.reheat !== false });
  } else {
    graphLayoutActive = true;
    initialFitPending = true;
    graph.graphData(data);
    initEdgeVisibility();
    configureForces();
    syncThreeVisualState();
  }

  if (showTagColors) toggleTagColors(true, false);
  updateSceneModeUI();
}

function createGraphInstance() {
  const container = document.getElementById('graph-container');
  const constrained = isConstrainedDevice();
  graph = new ForceGraph3D(container, {
    controlType: 'trackball',
    rendererConfig: {
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    },
  })
    .numDimensions(3)
    .backgroundColor(COLORS.bg)
    .showNavInfo(false)
    .nodeId('id')
    .nodeLabel(() => null)
    .nodeVisibility(isCoreNodeVisible)
    .nodeThreeObject(createNodeThreeObject)
    .nodeThreeObjectExtend(false)
    .linkSource('source')
    .linkTarget('target')
    .linkVisibility(effectiveLinkVisibility)
    .linkMaterial(linkMaterial)
    .linkWidth(linkDisplayWidth)
    .linkResolution(currentLodLevel === 'overview' ? 8 : 12)
    .linkHoverPrecision(constrained ? 16 : 24)
    .linkCurvature(organicLinkCurvature)
    .linkCurveRotation(organicLinkRotation)
    .linkDirectionalParticles(hoverAwareParticles)
    .linkDirectionalParticleSpeed(particleSpeed)
    .linkDirectionalParticleWidth(hoverAwareParticleWidth)
    .linkDirectionalParticleColor(particleColor)
    .linkDirectionalParticleThreeObject(directionalParticleObject)
    .linkDirectionalParticleResolution(constrained ? 8 : 10)
    .warmupTicks(constrained ? 45 : 90)
    .cooldownTime(constrained ? 8000 : 15000)
    .enablePointerInteraction(true)
    .enableNavigationControls(true)
    .enableNodeDrag(true)
    .onNodeHover((node) => {
      if (node) {
        clearHoverEdge(false);
        setHoverHighlight([node.id]);
        if (!isMobile()) showTooltip(node, lastMouseEvent);
      } else {
        scheduleClearHoverHighlight();
      }
    })
    .onNodeClick((node, event) => {
      if (!node) return;
      clearHoverEdge(false);
      setHoverHighlight([node.id]);
      selectGraphNode(node);
      focusGraphNode(node, { kind: 'node' });
      showTooltip(node, event || lastMouseEvent);
    })
    .onNodeDrag(() => markGraphActive(1200))
    .onNodeDragEnd((node) => {
      selectGraphNode(node);
      markGraphActive(900);
    })
    .onLinkHover((link) => {
      if (link) {
        setHoverEdge(link._id);
        showEdgeTooltip(link, lastMouseEvent);
      } else {
        clearHoverEdge();
        if (!isMobile()) hideTooltip();
      }
    })
    .onLinkClick((link, event) => {
      if (!link) return;
      selectGraphLink(link);
      setHoverEdge(link._id);
      showEdgeTooltip(link, event || lastMouseEvent);
    })
    .onBackgroundClick(() => {
      clearHoverEdge();
      clearHoverHighlight();
      hideTooltip();
    })
    .onEngineTick(() => {
      if (!initialFitPending) return;
      initialFitPending = false;
      // onEngineTick fires before Three.js objects receive their positions.
      // Wait until the renderer exposes real graph bounds before fitting.
      scheduleInitialCameraFit();
    })
    .onEngineStop(() => {
      graphLayoutActive = false;
      hideGraphLoader();
      scheduleGraphIdlePause(1000);
    });

  const renderer = graph.renderer();
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, constrained ? 1.75 : 2));
  renderer.setClearColor(COLORS.bg, 1);
  graph.scene().background = new window.THREE.Color(COLORS.bg);
  graph.scene().fog = new window.THREE.FogExp2(COLORS.bg, fogDensity);
  ensureNeuralField();
  scheduleBloomInstall();

  const controls = graph.controls();
  controls.rotateSpeed = 1.15;
  controls.zoomSpeed = 1.05;
  controls.panSpeed = 0.72;
  controls.dynamicDampingFactor = 0.16;
  controls.addEventListener('change', onCameraChanged);

  installGraphInteractionWakeup();
  installGraphResizeObserver();
  installMouseTracking();
  hideWebGLFallback();
}

function configureForces() {
  if (!graph) return;
  const spacing = spacingValue / 100;
  const chargeStrength = Math.round(-380 - spacing * 920);
  const linkDistance = 60 + edgeLengthMultiplier * 2.45;
  graph.d3Force('charge').strength(node => massAwareChargeStrength(node, chargeStrength));
  graph.d3Force('link').distance(link => massAwareLinkDistance(link, linkDistance));
  graph.d3Force('center').strength(0.035);
  // Collision force prevents nodes from overlapping — keeps clusters readable.
  // Uses a spatial grid for O(N) performance instead of O(N²) pairwise checks.
  if (!graph.d3Force('collision')) {
    graph.d3Force('collision', (function() {
      let nodes;
      const cellSize = 12; // grid cell size in world units
      const force = function(alpha) {
        if (!nodes || nodes.length === 0) return;
        const radiusMul = 1.8;
        // Build spatial grid
        const grid = new Map();
        for (let i = 0; i < nodes.length; i++) {
          const node = nodes[i];
          if (!node || node.x == null) continue;
          const r = nodeRadius(node.val || 4) * nodeWorldScale * radiusMul;
          node._collideRadius = r;
          const cx = Math.floor(node.x / cellSize);
          const cy = Math.floor(node.y / cellSize);
          const cz = Math.floor(node.z / cellSize);
          const key = cx + ',' + cy + ',' + cz;
          if (!grid.has(key)) grid.set(key, []);
          grid.get(key).push(node);
        }
        // Check only neighboring cells (3×3×3 = 27 cells max per node)
        const checked = new Set();
        for (let i = 0; i < nodes.length; i++) {
          const a = nodes[i];
          if (!a || a.x == null) continue;
          const cx = Math.floor(a.x / cellSize);
          const cy = Math.floor(a.y / cellSize);
          const cz = Math.floor(a.z / cellSize);
          for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
              for (let dz = -1; dz <= 1; dz++) {
                const key = (cx + dx) + ',' + (cy + dy) + ',' + (cz + dz);
                const cell = grid.get(key);
                if (!cell) continue;
                for (let k = 0; k < cell.length; k++) {
                  const b = cell[k];
                  if (b === a) continue;
                  const pairKey = a._id < b._id ? a._id + ':' + b._id : b._id + ':' + a._id;
                  if (checked.has(pairKey)) continue;
                  checked.add(pairKey);
                  let ddx = b.x - a.x, ddy = b.y - a.y, ddz = b.z - a.z;
                  let dist = Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
                  const minDist = (a._collideRadius || 1) + (b._collideRadius || 1);
                  if (dist < minDist && dist > 0) {
                    const push = (minDist - dist) * alpha * 0.5;
                    const nx = ddx / dist, ny = ddy / dist, nz = ddz / dist;
                    a.x -= nx * push; a.y -= ny * push; a.z -= nz * push;
                    b.x += nx * push; b.y += ny * push; b.z += nz * push;
                  }
                }
              }
            }
          }
        }
      };
      force.initialize = function(n) { nodes = n; };
      return force;
    })());
  }
}

function reheatGraphLayout() {
  if (!graph) return;
  graphLayoutActive = true;
  graph.d3ReheatSimulation();
  resumeGraphRendering();
}

function captureCameraState() {
  if (!graph) return null;
  const position = graph.cameraPosition();
  const controls = graph.controls();
  const target = controls && controls.target ? controls.target : { x: 0, y: 0, z: 0 };
  const up = graph.camera().up;
  return {
    position: { x: position.x, y: position.y, z: position.z },
    target: { x: target.x, y: target.y, z: target.z },
    up: { x: up.x, y: up.y, z: up.z },
    viewScale: currentViewScale,
  };
}

function restoreCameraState(state, duration = 0) {
  if (!graph || !state) return;
  if (state.up && graph.camera().up) graph.camera().up.set(state.up.x, state.up.y, state.up.z);
  graph.cameraPosition(state.position, state.target, duration);
  if (graph.controls() && graph.controls().target) {
    graph.controls().target.set(state.target.x, state.target.y, state.target.z);
    graph.controls().update();
  }
  currentViewScale = state.viewScale || currentViewScale;
  syncZoomUI(false);
}

function setGraphDataPreservingView(data, options = {}) {
  if (!graph) return null;
  if (!data || !Array.isArray(data.nodes) || data.nodes.length === 0) {
    console.warn('[BDH 3D] Ignoring empty structural update');
    return null;
  }
  const cameraState = captureCameraState();
  const liveData = graph.graphData();
  const liveNodeById = new Map((liveData.nodes || []).map(node => [node.id, node]));
  const incomingIds = new Set(data.nodes.map(node => node.id));
  (liveData.nodes || []).forEach(node => {
    if (!incomingIds.has(node.id)) disposeNodeVisual(node);
  });

  const safeData = {
    nodes: data.nodes.map(node => graphNodeSnapshot(node, liveNodeById.get(node.id) || {})),
    links: (data.links || []).map(graphLinkSnapshot),
  };
  safeData.links.sort((first, second) => {
    const order = { wikilink: 0, phantom: 1, project_context: 2, counterpart: 3, project_reference: 3, hebbian: 4, neurogenesis: 5 };
    return (order[first.type] || 0) - (order[second.type] || 0);
  });

  graph.graphData(safeData);
  initEdgeVisibility();
  configureForces();
  syncThreeVisualState();
  if (options.reheat) reheatGraphLayout();
  requestAnimationFrame(() => {
    restoreCameraState(cameraState, 0);
    markGraphActive(options.reheat ? 1800 : 600);
  });
  return safeData;
}

function requestGraphRedraw() {
  if (!graph) return;
  // Keep post-processing inside the same visual budget as the selected
  // neighborhood. A dense hub gets a tighter/weaker bloom automatically.
  if (typeof bloomPass !== 'undefined' && bloomPass) {
    const intensity = typeof activeHighlightIntensity === 'function' ? activeHighlightIntensity() : 1;
    bloomPass.strength = 0.52 + intensity * 0.35;
    bloomPass.radius = 0.42 + intensity * 0.16;
  }
  resumeGraphRendering();
  syncThreeVisualState();
  markGraphActive(520);
}

let graphActiveUntil = 0;
function markGraphActive(duration = 700) {
  if (!graph) return;
  resumeGraphRendering();
  graphActiveUntil = Math.max(graphActiveUntil, performance.now() + Math.max(100, duration));
  scheduleGraphIdlePause();
}

function resumeGraphRendering() {
  if (!graph || !graphRenderPaused) return;
  // resumeAnimation() can synchronously emit a controls change. Clear the guard
  // first so onCameraChanged() cannot recurse through markGraphActive().
  graphRenderPaused = false;
  graph.resumeAnimation();
}

function hasAmbientParticleFlow() {
  if (!graph || typeof isAmbientFlowLink !== 'function') return false;
  return graph.graphData().links.some(isAmbientFlowLink);
}

function scheduleGraphIdlePause(delay = 0) {
  if (!graph) return;
  if (graphIdleTimer) clearTimeout(graphIdleTimer);
  const remaining = Math.max(delay, graphActiveUntil - performance.now(), 220);
  graphIdleTimer = setTimeout(() => {
    graphIdleTimer = null;
    if (!graph || graphLayoutActive || performance.now() < graphActiveUntil || queryParticleMode || isHoverActive() || hasAmbientParticleFlow()) {
      scheduleGraphIdlePause(300);
      return;
    }
    graph.pauseAnimation();
    graphRenderPaused = true;
  }, remaining);
}

function installGraphInteractionWakeup() {
  const area = document.getElementById('graph-area');
  if (!area || area.dataset.wakeupInstalled === 'true') return;
  area.dataset.wakeupInstalled = 'true';
  const dismissHoverUI = () => {
    clearHoverEdge();
    clearHoverHighlight();
    hideTooltip();
  };
  ['pointermove', 'pointerdown', 'wheel', 'touchstart'].forEach(eventName => {
    area.addEventListener(eventName, () => markGraphActive(1200), { capture: true, passive: true });
  });
  area.addEventListener('pointerleave', dismissHoverUI, { capture: true, passive: true });
  const canvas = area.querySelector('canvas');
  if (canvas) {
    canvas.addEventListener('pointerleave', dismissHoverUI, { passive: true });
    canvas.addEventListener('mouseleave', dismissHoverUI, { passive: true });
  }
}

function installMouseTracking() {
  if (mouseTrackingInstalled) return;
  document.addEventListener('mousemove', event => {
    lastMouseEvent = event;
    if (tooltipEl && !tooltipEl.hidden && !isMobile()) positionTooltip(event);
  }, { passive: true });
  mouseTrackingInstalled = true;
}

let graphResizeObserver = null;
function installGraphResizeObserver() {
  const area = document.getElementById('graph-area');
  if (!area || graphResizeObserver) return;
  graphResizeObserver = new ResizeObserver(() => resizeGraphToContainer());
  graphResizeObserver.observe(area);
}

function resizeGraphToContainer() {
  if (!graph) return;
  const container = document.getElementById('graph-container');
  if (!container) return;
  const width = Math.max(1, container.clientWidth);
  const height = Math.max(1, container.clientHeight);
  graph.width(width).height(height);
  if (fitCameraDistance) updateNodeWorldScale(fitCameraDistance);
  markGraphActive(350);
}

function onCameraChanged() {
  if (!graph) return;
  updateCameraScaleFromPosition();
  scheduleLabelUpdate();
  updateSceneModeUI();
  markGraphActive(800);
}

function updateCameraScaleFromPosition() {
  if (!graph || !fitCameraDistance) return;
  const camera = graph.cameraPosition();
  const target = graph.controls().target || { x: 0, y: 0, z: 0 };
  const distance = Math.max(1, Math.hypot(camera.x - target.x, camera.y - target.y, camera.z - target.z));
  currentViewScale = Math.max(0.2, Math.min(4, fitCameraDistance / distance));
  const nextLod = currentViewScale < 0.9 ? 'overview' : currentViewScale < 1.7 ? 'balanced' : 'detail';
  if (nextLod !== currentLodLevel) {
    currentLodLevel = nextLod;
    syncThreeVisualState();
  }
  syncZoomUI(true);
}

function scheduleInitialCameraFit(attempt = 0) {
  if (!graph) return;
  requestAnimationFrame(() => {
    if (!graph) return;
    const bounds = graph.getGraphBbox();
    const spans = ['x', 'y', 'z'].map(axis => {
      const range = bounds && bounds[axis];
      return Array.isArray(range) && range.length === 2 && range.every(Number.isFinite)
        ? Math.abs(range[1] - range[0])
        : 0;
    });
    if (Math.max(...spans) > 10) {
      initialCameraFit();
      return;
    }
    if (attempt < 40) {
      setTimeout(() => scheduleInitialCameraFit(attempt + 1), 50);
    } else {
      console.warn('[BDH 3D] Camera fit skipped: graph bounds never became ready');
    }
  });
}

function updateSceneFog(distance) {
  if (!graph || !graph.scene().fog) return;
  // Respect user-configured fogDensity; only scale down for large graphs to keep visibility.
  const safeDistance = Number.isFinite(distance) && distance > 0 ? distance : 1000;
  const maxDensity = 0.45 / safeDistance;
  graph.scene().fog.density = Math.min(fogDensity, maxDensity);
}

function updateNodeWorldScale(distance, render = true) {
  if (!graph) return false;
  const container = document.getElementById('graph-container');
  const nextScale = BDH3DUtils.nodeScaleForFitDistance(
    distance,
    container ? container.clientHeight : window.innerHeight,
    graph.camera().fov,
  );
  if (Math.abs(nextScale - nodeWorldScale) < 0.01) return false;
  nodeWorldScale = nextScale;
  if (render) syncThreeVisualState();
  return true;
}

function initialCameraFit() {
  if (!graph || !graph.graphData().nodes.length) return;
  graph.zoomToFit(0, 72);
  setTimeout(() => {
    if (!graph) return;
    // First fit the full topology, then refit the visible neural core. This
    // keeps overview composition dense instead of centering on orphan tails.
    currentLodLevel = 'overview';
    syncThreeVisualState();
    graph.zoomToFit(220, 108);
    setTimeout(() => finalizeInitialCameraFit(), 240);
  }, 80);
}

function finalizeInitialCameraFit() {
    if (!graph) return;
    const camera = graph.cameraPosition();
    const target = graph.controls().target || { x: 0, y: 0, z: 0 };
    fitCameraDistance = Math.max(1, Math.hypot(camera.x - target.x, camera.y - target.y, camera.z - target.z));
    restoredZoom = null;
    updateSceneFog(fitCameraDistance);
    updateNodeWorldScale(fitCameraDistance, false);
    currentViewScale = 1;
    currentLodLevel = 'overview';
    syncThreeVisualState();
    if (restoredZoom != null && typeof zoomTo === 'function') zoomTo(restoredZoom, false);
    else syncZoomUI(false);
    updateSceneModeUI();
    scheduleLabelUpdate();
    markGraphActive(1200);
}

function updateGraphStats(values) {
  const setText = (id, value) => {
    const element = document.getElementById(id);
    if (element && value != null) element.textContent = value;
  };
  setText('stat-neurons', values.nodes);
  setText('stat-synapses', values.structural);
  setText('stat-wikilinks', values.wikilinks);
  setText('stat-hebbian', values.hebbian);
  setText('stat-dormant-count', values.dormant);
  setText('stat-phantom-count', values.phantom);
  setText('view-stats', `${values.visibleNodes ?? values.nodes} nodes · ${values.visibleEdges ?? values.structural} edges visible`);
  const dormant = document.getElementById('stat-dormant');
  const phantom = document.getElementById('stat-phantom');
  if (dormant) dormant.hidden = !(values.dormant > 0);
  if (phantom) phantom.hidden = !(values.phantom > 0);
}

function updateSceneModeUI() {
  const cameraMode = document.getElementById('camera-mode');
  const lodState = document.getElementById('lod-state');
  if (cameraMode) cameraMode.textContent = queryLensActive ? 'Retrieval lens' : focusMode ? 'Focus mode' : 'Free exploration';
  if (lodState) lodState.textContent = queryLensActive
    ? `${retrievalVisibleNodeIds ? retrievalVisibleNodeIds.size : 0} nodes · query context`
    : `${currentLodLevel[0].toUpperCase()}${currentLodLevel.slice(1)} LOD`;
  if (graph) {
    const data = graph.graphData();
    const visibleNodes = queryLensActive && retrievalVisibleNodeIds
      ? retrievalVisibleNodeIds.size
      : data.nodes.filter(isCoreNodeVisible).length;
    const visibleEdges = data.links.filter(link => effectiveLinkVisibility(link)).length;
    const viewStats = document.getElementById('view-stats');
    if (viewStats) viewStats.textContent = `${visibleNodes} nodes · ${visibleEdges} edges visible`;
    updateGraphMinimap();
  }
}

function updateGraphMinimap() {
  const canvas = document.getElementById('graph-minimap-canvas');
  if (!canvas || !graph) return;
  const context = canvas.getContext('2d');
  if (!context) return;
  const data = graph.graphData();
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = 'rgba(7,10,15,0.82)';
  context.fillRect(0, 0, width, height);

  const nodes = data.nodes.filter(node => isCoreNodeVisible(node) && Number.isFinite(node.x) && Number.isFinite(node.z));
  if (!nodes.length) return;
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  nodes.forEach(node => {
    minX = Math.min(minX, node.x); maxX = Math.max(maxX, node.x);
    minZ = Math.min(minZ, node.z); maxZ = Math.max(maxZ, node.z);
  });
  const spanX = Math.max(1, maxX - minX);
  const spanZ = Math.max(1, maxZ - minZ);
  const padding = 10;
  const project = node => ({
    x: padding + ((node.x - minX) / spanX) * (width - padding * 2),
    y: padding + ((node.z - minZ) / spanZ) * (height - padding * 2),
  });
  const nodeById = new Map(nodes.map(node => [node.id, node]));
  context.lineWidth = 0.55;
  data.links.forEach(link => {
    if (!effectiveLinkVisibility(link)) return;
    const source = nodeById.get(linkEndpointId(link.source));
    const target = nodeById.get(linkEndpointId(link.target));
    if (!source || !target) return;
    const a = project(source), b = project(target);
    context.strokeStyle = link.type === 'hebbian' ? 'rgba(168,121,255,0.52)' : link.type === 'phantom' ? 'rgba(31,111,235,0.34)' : 'rgba(143,168,194,0.24)';
    context.beginPath();
    context.moveTo(a.x, a.y);
    context.lineTo(b.x, b.y);
    context.stroke();
  });
  nodes.forEach(node => {
    const point = project(node);
    const highlight = isHighlightedNode(node) || activatedNotesById.has(node.id);
    context.fillStyle = highlight ? '#f0d2ff' : node.color || '#6e7681';
    context.globalAlpha = highlight ? 1 : 0.72;
    context.beginPath();
    context.arc(point.x, point.y, highlight ? 2.4 : 1.45, 0, Math.PI * 2);
    context.fill();
  });
  context.globalAlpha = 1;
  const mode = document.getElementById('minimap-mode');
  if (mode) mode.textContent = queryLensActive ? 'query' : 'full';
}

function supportsWebGL() {
  try {
    const canvas = document.createElement('canvas');
    return Boolean(window.WebGLRenderingContext && (canvas.getContext('webgl2') || canvas.getContext('webgl')));
  } catch (error) {
    return false;
  }
}

function isConstrainedDevice() {
  const memory = Number(navigator.deviceMemory || 8);
  const cores = Number(navigator.hardwareConcurrency || 8);
  return isMobile() || memory <= 4 || cores <= 4;
}

function showWebGLFallback(message) {
  const fallback = document.getElementById('webgl-fallback');
  if (!fallback) return;
  fallback.hidden = false;
  const paragraph = fallback.querySelector('p');
  if (paragraph && message) paragraph.textContent = message + ' Graph data and query tools remain available.';
  setConnectionStatus('degraded', 'Data only');
}

function hideWebGLFallback() {
  const fallback = document.getElementById('webgl-fallback');
  if (fallback) fallback.hidden = true;
}

function hideGraphLoader() {
  const loader = document.getElementById('graph-loader');
  if (loader) {
    loader.classList.add('loader-fade');
    setTimeout(() => { loader.hidden = true; }, 400);
  }
  // Restore panel/controls to their saved state once the graph is ready.
  const panelCollapsed = localStorage.getItem('bdh-panel-collapsed-v2') === 'true';
  const controlsCollapsed = localStorage.getItem('bdh-controls-collapsed-v1') === 'true';
  document.body.classList.toggle('panel-collapsed', panelCollapsed);
  document.body.classList.toggle('controls-collapsed', controlsCollapsed);
  // Restore panel width
  const sidePanel = document.getElementById('side-panel');
  if (sidePanel) {
    const savedWidth = Math.max(22, Math.min(48, parseFloat(localStorage.getItem('bdh-panel-width-v2')) || 38));
    sidePanel.style.width = savedWidth + '%';
    requestAnimationFrame(() => { if (typeof resizeGraphToContainer === 'function') resizeGraphToContainer(); });
  }
}

function retryVisualization() {
  window.location.reload();
}
