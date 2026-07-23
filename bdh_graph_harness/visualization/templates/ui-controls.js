// ============================================================================
// BDH Graph Harness — controls, camera model, responsive layout and startup
// ============================================================================

function switchTab(tabClass) {
  document.body.classList.remove('graph-tab', 'controls-tab', 'panel-tab');
  document.body.classList.add(tabClass);
  document.querySelectorAll('#mobile-tabs .tab').forEach(tab => {
    const active = tab.dataset.tab === tabClass;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-current', active ? 'page' : 'false');
  });
  syncPanelWidthForViewport();
  if (tabClass === 'graph-tab') requestAnimationFrame(() => resizeGraphToContainer());
  hideTooltip();
}

function setConnectionStatus(state, label) {
  const indicator = document.getElementById('status-indicator');
  const text = document.getElementById('status-text');
  if (indicator) indicator.className = state || '';
  if (text) text.textContent = label || 'Connecting';
}

function setResponseState(label) {
  const state = document.getElementById('response-state');
  if (state) {
    state.textContent = label;
    state.dataset.state = String(label || '').toLowerCase().replace(/\s+/g, '-');
  }
}

// ============================================================================
// Source, orphan and appearance controls
// ============================================================================
function syncSourceFilterUI() {
  const control = document.getElementById('source-filter');
  if (!control) return;
  control.querySelectorAll('button[data-value]').forEach(button => {
    const active = button.dataset.value === sourceFilter;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
}

function setSourceFilter(value, persist = true) {
  sourceFilter = ['all', 'vault', 'external'].includes(value) ? value : 'all';
  syncSourceFilterUI();
  if (persist) saveControlValue(STORAGE_KEYS.sourceFilter, sourceFilter);
  if (sourceGraphData) initNetwork(sourceGraphData, { preserveView: true, reheat: true });
}

function toggleOrphans(show) {
  showOrphans = Boolean(show);
  saveControlValue(STORAGE_KEYS.orphanVisibility, showOrphans);
  const toggle = document.getElementById('orphan-toggle');
  if (toggle) toggle.checked = showOrphans;
  if (sourceGraphData) initNetwork(sourceGraphData, { preserveView: true, reheat: true });
}

function toggleTagColors(enabled, render = true) {
  showTagColors = Boolean(enabled);
  const legend = document.getElementById('tag-legend');
  const toggle = document.getElementById('tag-toggle');
  if (toggle) toggle.checked = showTagColors;

  if (legend) {
    if (showTagColors && Object.keys(tagColorMap).length) {
      const items = Object.entries(tagColorMap)
        .sort((first, second) => first[0].localeCompare(second[0]))
        .map(([tag, color]) => `<span class="tag-item"><i style="background:${color}"></i>${escapeHtml(tag)}</span>`);
      legend.innerHTML = '<button class="legend-collapse" onclick="toggleLegendCollapse()" title="Collapse legend" aria-label="Collapse tag legend">◀</button>' + items.join('');
      legend.classList.remove('hidden');
      legend.classList.toggle('collapsed', localStorage.getItem('bdh-legend-collapsed') === 'true');
    } else {
      legend.classList.add('hidden');
    }
  }

  if (!graph || !render) return;
  graph.graphData().nodes.forEach(node => {
    const data = nodeDataMap[node.id] || {};
    const tags = normalizeTags(data.tags || node._tags || '');
    const isNeurogenesis = isNeurogenesisNode(node);
    if (isNeurogenesis) node.color = COLORS.neurogenesis;
    else if (node._dormant) node.color = COLORS.dormant;
    else if (showTagColors && tags.length && tagColorMap[tags[0]]) node.color = tagColorMap[tags[0]];
    else node.color = sourceColor(data);
    nodeTagColorMap[node.id] = node.color;
  });
  requestGraphRedraw();
}

function toggleLegendCollapse() {
  const legend = document.getElementById('tag-legend');
  if (!legend) return;
  legend.classList.toggle('collapsed');
  localStorage.setItem('bdh-legend-collapsed', String(legend.classList.contains('collapsed')));
}

document.getElementById('tag-legend')?.addEventListener('click', function expandCollapsedLegend(event) {
  if (this.classList.contains('collapsed') && event.target === this) toggleLegendCollapse();
});

// ============================================================================
// Edge visibility and layout
// ============================================================================
const edgeTypeVisible = {
  wikilink: true,
  counterpart: true,
  project_context: true,
  project_reference: true,
  hebbian: true,
  phantom: true,
  neurogenesis: true,
};

function restoreGraphControlState() {
  hebbianThreshold = clampNumber(loadControlValue(STORAGE_KEYS.hebbianThreshold, hebbianThreshold), 0, 1, hebbianThreshold);
  spacingValue = clampNumber(loadControlValue(STORAGE_KEYS.spacing, spacingValue), 0, 100, spacingValue);
  edgeLengthMultiplier = clampNumber(loadControlValue(STORAGE_KEYS.edgeLength, edgeLengthMultiplier), 0, 100, edgeLengthMultiplier);
  const savedSourceFilter = loadControlValue(STORAGE_KEYS.sourceFilter, sourceFilter);
  if (['all', 'vault', 'external'].includes(savedSourceFilter)) sourceFilter = savedSourceFilter;
  restoredZoom = clampNumber(loadControlValue(STORAGE_KEYS.zoom, ''), 0.2, 4, null);
  showOrphans = String(loadControlValue(STORAGE_KEYS.orphanVisibility, showOrphans)) === 'true';
  showPhantom = String(loadControlValue(STORAGE_KEYS.phantomVisibility, showPhantom)) === 'true';

  // Neural atmosphere controls (7/8/9/10)
  edgeFadeStrength = clampNumber(loadControlValue(STORAGE_KEYS.edgeFade, edgeFadeStrength), 0, 0.5, 0.05);
  const storedFogDensity = Number(loadControlValue(STORAGE_KEYS.fogDensity, 38));
  fogDensity = clampNumber(storedFogDensity, 0, 100, 38) / 100000;
  particleFlowIntensity = clampNumber(loadControlValue(STORAGE_KEYS.particleFlow, particleFlowIntensity), 0, 1, 0.5);
  edgeCurvatureBase = clampNumber(loadControlValue(STORAGE_KEYS.edgeCurvature, edgeCurvatureBase), 0, 1, 0.25);

  const thresholdSlider = document.getElementById('hebbian-threshold');
  const thresholdValue = document.getElementById('threshold-val');
  const spacingSlider = document.getElementById('spacing-slider');
  const spacingOutput = document.getElementById('spacing-val');
  const edgeSlider = document.getElementById('edge-length-slider');
  const edgeOutput = document.getElementById('el-val');
  if (thresholdSlider) thresholdSlider.value = hebbianThreshold;
  if (thresholdValue) thresholdValue.textContent = Number(hebbianThreshold).toFixed(2);
  if (spacingSlider) spacingSlider.value = spacingValue;
  if (spacingOutput) spacingOutput.textContent = spacingValue;
  if (edgeSlider) edgeSlider.value = edgeLengthMultiplier;
  if (edgeOutput) edgeOutput.textContent = edgeLengthMultiplier;

  // Restore atmosphere slider positions
  const fadeSlider = document.getElementById('edge-fade-slider');
  if (fadeSlider) fadeSlider.value = edgeFadeStrength;
  const fogSlider = document.getElementById('fog-slider');
  if (fogSlider) fogSlider.value = Math.round(fogDensity * 100000);
  const fogOutput = document.getElementById('fog-val');
  if (fogOutput) fogOutput.textContent = Math.round(fogDensity * 100000);
  const flowSlider = document.getElementById('particle-flow-slider');
  if (flowSlider) flowSlider.value = Math.round(particleFlowIntensity * 100);
  const curvSlider = document.getElementById('curvature-slider');
  if (curvSlider) curvSlider.value = Math.round(edgeCurvatureBase * 100);

  const orphanToggle = document.getElementById('orphan-toggle');
  if (orphanToggle) orphanToggle.checked = showOrphans;
  const phantomToggle = document.getElementById('phantom-toggle');
  if (phantomToggle) phantomToggle.checked = showPhantom;

  syncSourceFilterUI();
}

function toggleDirectOnly(enabled) {
  directOnly = Boolean(enabled);
  const threshold = document.getElementById('hebbian-threshold');
  if (threshold) threshold.disabled = directOnly;
  applyEdgeFilters();
}

function updateHebbianThreshold(value, persist = true) {
  hebbianThreshold = clampNumber(value, 0, 1, 0.15);
  const slider = document.getElementById('hebbian-threshold');
  const output = document.getElementById('threshold-val');
  if (slider) slider.value = hebbianThreshold;
  if (output) output.textContent = Number(hebbianThreshold).toFixed(2);
  if (persist) saveControlValue(STORAGE_KEYS.hebbianThreshold, hebbianThreshold);
  applyEdgeFiltersDebounced();
}

let edgeFilterTimer = null;
function applyEdgeFiltersDebounced() {
  if (edgeFilterTimer) clearTimeout(edgeFilterTimer);
  edgeFilterTimer = setTimeout(() => {
    edgeFilterTimer = null;
    applyEdgeFilters();
  }, 35);
}

function edgeVisibleFromControls(link) {
  const info = edgeInfoMap[link._id] || {};
  const type = info.type || link.type || 'wikilink';
  if (directOnly) {
    return ['wikilink', 'counterpart', 'project_context', 'project_reference'].includes(type);
  }
  if (edgeTypeVisible[type] === false) return false;
  if (type === 'hebbian') return (info.weight ?? link.weight ?? 0) >= hebbianThreshold;
  if (type === 'phantom') return showPhantom;
  return true;
}

function initEdgeVisibility() {
  if (!graph) return;
  linkVisibilityState.clear();
  graph.graphData().links.forEach(link => linkVisibilityState.set(linkKey(link), edgeVisibleFromControls(link)));
}

function applyEdgeFilters() {
  if (!graph) return;
  initEdgeVisibility();
  requestGraphRedraw();
}

function togglePhantom(enabled) {
  showPhantom = Boolean(enabled);
  saveControlValue(STORAGE_KEYS.phantomVisibility, showPhantom);
  const toggle = document.getElementById('phantom-toggle');
  if (toggle) toggle.checked = showPhantom;
  applyEdgeFilters();
}

function toggleEdgeType(type, button) {
  edgeTypeVisible[type] = !edgeTypeVisible[type];
  if (button) {
    button.classList.toggle('active', edgeTypeVisible[type]);
    button.setAttribute('aria-pressed', String(edgeTypeVisible[type]));
  }
  applyEdgeFilters();
}

// ============================================================================
// Retrieval lens — query-first without destroying the full neural topology
// ============================================================================
const QUERY_LENS_DEFAULTS = {
  maxNodes: 140,
  maxSeeds: 12,
  hopLimit: 1,
};

function setRetrievalLensUI(active) {
  document.body.classList.toggle('retrieval-lens-active', Boolean(active));
  updateSceneModeUI();
}

function applyRetrievalLens(notes = [], query = '') {
  if (!graph) return;
  const ranked = [...notes]
    .filter(note => note && note.id)
    .sort((first, second) => Number(second.final_score ?? second.score ?? 0) - Number(first.final_score ?? first.score ?? 0));
  if (!ranked.length) {
    restoreFullGraphView({ fit: false });
    return;
  }

  const seedIds = ranked.slice(0, QUERY_LENS_DEFAULTS.maxSeeds).map(note => note.id);
  const visible = new Set(seedIds);
  const data = graph.graphData();
  let frontier = new Set(seedIds);
  for (let hop = 0; hop < QUERY_LENS_DEFAULTS.hopLimit; hop += 1) {
    const next = new Set();
    data.links.forEach(link => {
      const source = linkEndpointId(link.source);
      const target = linkEndpointId(link.target);
      if (link.type === 'phantom') return;
      if (frontier.has(source) && !visible.has(target)) next.add(target);
      if (frontier.has(target) && !visible.has(source)) next.add(source);
    });
    for (const id of next) {
      if (visible.size >= QUERY_LENS_DEFAULTS.maxNodes) break;
      visible.add(id);
    }
    frontier = next;
    if (!frontier.size || visible.size >= QUERY_LENS_DEFAULTS.maxNodes) break;
  }

  queryLensActive = true;
  retrievalVisibleNodeIds = visible;
  retrievalQuery = query;
  currentLodLevel = 'balanced';
  setRetrievalLensUI(true);
  initEdgeVisibility();
  requestGraphRedraw();
  window.setTimeout(() => {
    if (queryLensActive) fitToScreen();
  }, reducedMotion.matches ? 0 : 90);
}

function restoreFullGraphView(options = {}) {
  queryLensActive = false;
  retrievalVisibleNodeIds = null;
  retrievalQuery = '';
  setRetrievalLensUI(false);
  initEdgeVisibility();
  requestGraphRedraw();
  if (options.fit !== false && graph) fitToScreen();
}

const VIEW_PRESETS = {
  clean: { orphans: false, phantom: false, threshold: 0.42, fade: 0.08, fog: 24, particles: 0.28, ambient: false },
  evidence: { orphans: false, phantom: false, threshold: 0.42, fade: 0.12, fog: 18, particles: 0.42, ambient: false },
  'full-neural': { orphans: true, phantom: true, threshold: 0.15, fade: 0.05, fog: 25, particles: 0.7, ambient: true },
  debug: { orphans: true, phantom: true, threshold: 0, fade: 0, fog: 0, particles: 0, ambient: false },
};

function applyVisualizationPreset(name) {
  const preset = VIEW_PRESETS[name] || VIEW_PRESETS.clean;
  showOrphans = preset.orphans;
  showPhantom = preset.phantom;
  directOnly = false;
  saveControlValue(STORAGE_KEYS.orphanVisibility, showOrphans);
  saveControlValue(STORAGE_KEYS.phantomVisibility, showPhantom);
  updateHebbianThreshold(preset.threshold);
  updateEdgeFade(preset.fade);
  updateFogDensity(preset.fog, false);
  updateParticleFlow(preset.particles);
  toggleAmbientMotion(preset.ambient);
  const orphanToggle = document.getElementById('orphan-toggle');
  if (orphanToggle) orphanToggle.checked = showOrphans;
  const phantomToggle = document.getElementById('phantom-toggle');
  if (phantomToggle) phantomToggle.checked = showPhantom;
  const directToggle = document.getElementById('direct-toggle');
  if (directToggle) directToggle.checked = false;
  if (sourceGraphData) initNetwork(sourceGraphData, { preserveView: true, reheat: true });
  applyEdgeFilters();
  // Re-apply fog density AFTER initNetwork, since initNetwork's camera fit recalculates fog.
  updateFogDensity(preset.fog);
  if (name === 'evidence' && typeof lastRetrievalNotes !== 'undefined' && lastRetrievalNotes.length) {
    applyRetrievalLens(lastRetrievalNotes, lastRetrievalQuery);
  } else if (name !== 'evidence' && queryLensActive) {
    restoreFullGraphView({ fit: false });
  }
  const select = document.getElementById('view-preset');
  if (select) select.value = VIEW_PRESETS[name] ? name : 'clean';
  return { name: VIEW_PRESETS[name] ? name : 'clean', ...preset };
}

function setEdgeLengthMultiplier(value, persist = true) {
  edgeLengthMultiplier = clampNumber(value, 0, 100, 10);
  const slider = document.getElementById('edge-length-slider');
  const output = document.getElementById('el-val');
  if (slider) slider.value = edgeLengthMultiplier;
  if (output) output.textContent = edgeLengthMultiplier;
  if (persist) saveControlValue(STORAGE_KEYS.edgeLength, edgeLengthMultiplier);
  if (graph) {
    const distance = 42 + edgeLengthMultiplier * 2.45;
    graph.d3Force('link').distance(link => massAwareLinkDistance(link, distance));
    reheatGraphLayout();
    markGraphActive(1800);
  }
}

function updateSpacing(value, persist = true) {
  spacingValue = clampNumber(value, 0, 100, 50);
  const slider = document.getElementById('spacing-slider');
  const output = document.getElementById('spacing-val');
  if (slider) slider.value = spacingValue;
  if (output) output.textContent = spacingValue;
  if (persist) saveControlValue(STORAGE_KEYS.spacing, spacingValue);
  if (graph) {
    const chargeStrength = Math.round(-280 - (spacingValue / 100) * 820);
    graph.d3Force('charge').strength(node => massAwareChargeStrength(node, chargeStrength));
    reheatGraphLayout();
    markGraphActive(1800);
  }
}

// ============================================================================
// Query, reset and input state
// ============================================================================
const queryInput = document.getElementById('query-input');
const queryClear = document.getElementById('query-clear');

function updateClearBtn() {
  if (queryClear && queryInput) queryClear.classList.toggle('visible', queryInput.value.length > 0);
}

function clearQuery() {
  if (!queryInput) return;
  queryInput.value = '';
  updateClearBtn();
  queryInput.focus();
}

queryInput?.addEventListener('input', updateClearBtn);
queryInput?.addEventListener('keydown', event => {
  if (event.key === 'Enter') sendQuery();
});

function resetQuery() {
  if (typeof invalidateActivationAnimations === 'function') invalidateActivationAnimations();
  clearQuery();
  document.getElementById('response-text').textContent = '—';
  document.getElementById('activated-list').innerHTML = '<li class="empty-state">No activations yet.</li>';
  document.getElementById('activation-count').textContent = '0';
  const concepts = document.getElementById('new-concepts-section');
  const conceptsList = document.getElementById('new-concepts-list');
  if (concepts) concepts.hidden = true;
  if (conceptsList) conceptsList.replaceChildren();
  activatedNotesById.clear();
  clearActivationState();
  endQueryParticles();
  restoreFullGraphView({ fit: false });
  setResponseState('Ready');
  if (focusMode) exitFocusMode();
  applyEdgeFilters();
}

// ============================================================================
// Explicit 3D camera model
// ============================================================================
const ZOOM_MIN = 0.2;
const ZOOM_MAX = 4;
const zoomSlider = document.getElementById('zoom-slider');
const zoomLabel = document.getElementById('zoom-value');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

function resetCameraFitBaseline() {
  fitCameraDistance = null;
  restoredZoom = null;
  currentViewScale = 1;
  if (typeof syncZoomUI === 'function') syncZoomUI(false);
}


function syncZoomUI(persist = false) {
  const scale = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, currentViewScale || 1));
  if (zoomSlider) zoomSlider.value = scale;
  if (zoomLabel) zoomLabel.textContent = Math.round(scale * 100) + '%';
  if (persist) saveControlValue(STORAGE_KEYS.zoom, scale);
}

function cameraTarget() {
  const controls = graph && graph.controls ? graph.controls() : null;
  const target = controls && controls.target ? controls.target : { x: 0, y: 0, z: 0 };
  return { x: target.x, y: target.y, z: target.z };
}

function cameraDistance() {
  if (!graph) return 1;
  const camera = graph.cameraPosition();
  const target = cameraTarget();
  return Math.max(1, Math.hypot(camera.x - target.x, camera.y - target.y, camera.z - target.z));
}

function zoomTo(value, persist = true) {
  if (!graph) return;
  const scale = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, Number(value)));
  const camera = graph.cameraPosition();
  const target = cameraTarget();
  if (!fitCameraDistance) fitCameraDistance = cameraDistance();
  const desiredDistance = fitCameraDistance / scale;
  const dx = camera.x - target.x;
  const dy = camera.y - target.y;
  const dz = camera.z - target.z;
  const length = Math.hypot(dx, dy, dz) || 1;
  const next = {
    x: target.x + (dx / length) * desiredDistance,
    y: target.y + (dy / length) * desiredDistance,
    z: target.z + (dz / length) * desiredDistance,
  };
  graph.cameraPosition(next, target, reducedMotion.matches ? 0 : 160);
  currentViewScale = scale;
  currentLodLevel = scale < 0.9 ? 'overview' : scale < 1.7 ? 'balanced' : 'detail';
  syncZoomUI(persist);
  updateSceneModeUI();
  requestGraphRedraw();
}

function zoomStep(delta) {
  zoomTo((currentViewScale || 1) + delta, true);
}

function clearFocusStateWithoutCameraRestore() {
  focusMode = null;
  focusedNodeId = null;
  focusHighlight = null;
  updateFocusUI();
  requestGraphRedraw();
}

function fitToScreen() {
  if (!graph) return;
  clearFocusStateWithoutCameraRestore();
  resetCameraFitBaseline();
  graph.zoomToFit(reducedMotion.matches ? 0 : 520, 72);
  setTimeout(() => {
    if (!graph) return;
    fitCameraDistance = cameraDistance();
    updateSceneFog(fitCameraDistance);
    updateNodeWorldScale(fitCameraDistance);
    currentViewScale = 1;
    currentLodLevel = 'balanced';
    restoredZoom = null;
    syncZoomUI(true);
    updateSceneModeUI();
    requestGraphRedraw();
  }, reducedMotion.matches ? 0 : 560);
}

function resetCameraOrientation() {
  if (!graph) return;
  const target = cameraTarget();
  const distance = cameraDistance();
  graph.camera().up.set(0, 1, 0);
  graph.cameraPosition(
    { x: target.x, y: target.y, z: target.z + distance },
    target,
    reducedMotion.matches ? 0 : 480,
  );
  markGraphActive(700);
}

function focusGraphNode(node, options = {}) {
  if (!graph || !node) return;
  const previousCamera = focusMode ? focusMode.previousCamera : captureCameraState();
  focusMode = {
    nodeId: node.id,
    kind: options.kind || 'node',
    previousCamera,
  };
  focusedNodeId = node.id;
  if (options.pathIds && options.pathIds.length) setPathHighlight(options.pathIds);
  else setNeighborhoodFocus(node.id);

  // Fit the entire highlighted subgraph in view instead of zooming aggressively on the node.
  const highlight = activeHighlight();
  const data = graph.graphData();
  const focusNodeIds = highlight && highlight.nodeIds && highlight.nodeIds.size
    ? [...highlight.nodeIds]
    : [node.id];
  const focusNodes = data.nodes.filter(n => focusNodeIds.includes(n.id));
  if (focusNodes.length === 0) return;

  let min = { x: Infinity, y: Infinity, z: Infinity };
  let max = { x: -Infinity, y: -Infinity, z: -Infinity };
  focusNodes.forEach(n => {
    min.x = Math.min(min.x, n.x || 0); max.x = Math.max(max.x, n.x || 0);
    min.y = Math.min(min.y, n.y || 0); max.y = Math.max(max.y, n.y || 0);
    min.z = Math.min(min.z, n.z || 0); max.z = Math.max(max.z, n.z || 0);
  });
  const center = {
    x: (min.x + max.x) / 2,
    y: (min.y + max.y) / 2,
    z: (min.z + max.z) / 2,
  };
  const span = Math.max(
    max.x - min.x, max.y - min.y, max.z - min.z, 30
  );
  const currentCamera = graph.cameraPosition();
  const dir = {
    x: (currentCamera.x || 0) - (node.x || 0),
    y: (currentCamera.y || 0) - (node.y || 0),
    z: (currentCamera.z || 0) - (node.z || 0),
  };
  const dirLen = Math.hypot(dir.x, dir.y, dir.z) || 1;
  const focusDistance = Math.max(140, span * 2.4);
  const nextPosition = {
    x: center.x + (dir.x / dirLen) * focusDistance,
    y: center.y + (dir.y / dirLen) * focusDistance,
    z: center.z + (dir.z / dirLen) * focusDistance,
  };
  graph.cameraPosition(nextPosition, center, reducedMotion.matches ? 0 : 620);
  selectGraphNode(node);
  updateFocusUI();
  scheduleLabelUpdate();
  markGraphActive(900);
}

function exitFocusMode() {
  if (!focusMode) return;
  const previousCamera = focusMode.previousCamera;
  focusMode = null;
  focusedNodeId = null;
  focusHighlight = null;
  if (previousCamera) restoreCameraState(previousCamera, reducedMotion.matches ? 0 : 520);
  updateFocusUI();
  scheduleLabelUpdate();
  requestGraphRedraw();
}

function updateFocusUI() {
  const hud = document.getElementById('focus-hud');
  const title = document.getElementById('focus-title');
  const back = document.getElementById('focus-back-btn');
  if (hud) hud.hidden = !focusMode;
  if (back) back.hidden = !focusMode;
  if (title && focusMode) {
    const data = nodeDataMap[focusMode.nodeId] || {};
    title.textContent = data.display_label || data.title || focusMode.nodeId;
  }
  updateSceneModeUI();
}

function searchNode(query) {
  if (!query || !graph) return;
  const needle = query.trim().toLowerCase();
  if (!needle) return;
  const nodes = graph.graphData().nodes;
  const ranked = nodes
    .map(node => {
      const name = String(node.name || '').toLowerCase();
      const score = name === needle ? 0 : name.startsWith(needle) ? 1 : name.includes(needle) ? 2 : 99;
      return { node, score };
    })
    .filter(item => item.score < 99)
    .sort((first, second) => first.score - second.score || first.node.name.length - second.node.name.length);
  if (!ranked.length) {
    const input = document.getElementById('search-input');
    if (input) {
      input.classList.add('not-found');
      setTimeout(() => input.classList.remove('not-found'), 900);
    }
    return;
  }
  const node = ranked[0].node;
  selectGraphNode(node);
  focusGraphNode(node, { kind: 'search' });
  showTooltip(node, lastMouseEvent || { clientX: 200, clientY: 120 });
}

// ============================================================================
// Inspector and control dock layout
// ============================================================================
const PANEL_WIDTH_KEY = 'bdh-panel-width-v2';
const PANEL_COLLAPSED_KEY = 'bdh-panel-collapsed-v2';
const CONTROLS_COLLAPSED_KEY = 'bdh-controls-collapsed-v1';
const sidePanel = document.getElementById('side-panel');
const panelResize = document.getElementById('panel-resize');
const collapseButton = document.getElementById('collapse-btn');
let isResizing = false;

function syncPanelWidthForViewport() {
  if (!sidePanel) return;
  if (isMobile()) {
    // Desktop width is persisted as an inline percentage, which otherwise wins
    // over the mobile width rule and crushes Inspector to roughly one third.
    sidePanel.style.removeProperty('width');
    return;
  }
  const savedWidth = clampNumber(localStorage.getItem(PANEL_WIDTH_KEY), 22, 48, 31);
  sidePanel.style.width = savedWidth + '%';
}

function restorePanelState() {
  syncPanelWidthForViewport();
  const collapsed = localStorage.getItem(PANEL_COLLAPSED_KEY) === 'true';
  document.body.classList.toggle('panel-collapsed', collapsed);
  const controlsCollapsed = localStorage.getItem(CONTROLS_COLLAPSED_KEY) === 'true';
  document.body.classList.toggle('controls-collapsed', controlsCollapsed);
  syncPanelToggleUI();
}

function syncPanelToggleUI() {
  const panelCollapsed = document.body.classList.contains('panel-collapsed');
  const controlsCollapsed = document.body.classList.contains('controls-collapsed');
  const inspectorButtons = [document.getElementById('collapse-btn'), document.getElementById('expand-panel')].filter(Boolean);
  const controlsButtons = [document.getElementById('controls-collapse'), document.getElementById('expand-controls')].filter(Boolean);

  inspectorButtons.forEach(button => {
    const expanded = !panelCollapsed;
    const label = expanded ? 'Hide inspector' : 'Show inspector';
    const icon = button.querySelector('.panel-toggle-icon');
    const text = button.querySelector('.panel-toggle-label');
    if (icon) icon.textContent = expanded ? '▸' : '◂';
    if (text) text.textContent = 'Inspector';
    button.title = label;
    button.setAttribute('aria-label', label);
    button.setAttribute('aria-expanded', String(expanded));
  });

  controlsButtons.forEach(button => {
    const expanded = !controlsCollapsed;
    const label = expanded ? 'Hide controls' : 'Show controls';
    const icon = button.querySelector('.panel-toggle-icon');
    const text = button.querySelector('.panel-toggle-label');
    if (icon) icon.textContent = expanded ? '‹' : '›';
    if (text) text.textContent = 'Controls';
    button.title = label;
    button.setAttribute('aria-label', label);
    button.setAttribute('aria-expanded', String(expanded));
  });
}

panelResize?.addEventListener('mousedown', event => {
  event.preventDefault();
  isResizing = true;
  panelResize.classList.add('dragging');
  document.body.classList.add('panel-resizing');
});

document.addEventListener('mousemove', event => {
  if (!isResizing || !sidePanel) return;
  const main = document.getElementById('main').getBoundingClientRect();
  const percentage = Math.max(22, Math.min(48, ((main.right - event.clientX) / main.width) * 100));
  sidePanel.style.width = percentage + '%';
});

document.addEventListener('mouseup', () => {
  if (!isResizing) return;
  isResizing = false;
  panelResize.classList.remove('dragging');
  document.body.classList.remove('panel-resizing');
  if (sidePanel) localStorage.setItem(PANEL_WIDTH_KEY, String(parseFloat(sidePanel.style.width)));
  requestAnimationFrame(() => resizeGraphToContainer());
});

function togglePanel() {
  document.body.classList.toggle('panel-collapsed');
  const collapsed = document.body.classList.contains('panel-collapsed');
  localStorage.setItem(PANEL_COLLAPSED_KEY, String(collapsed));
  syncPanelToggleUI();
  requestAnimationFrame(() => resizeGraphToContainer());
}

function toggleControlsDock() {
  document.body.classList.toggle('controls-collapsed');
  const collapsed = document.body.classList.contains('controls-collapsed');
  localStorage.setItem(CONTROLS_COLLAPSED_KEY, String(collapsed));
  syncPanelToggleUI();
  requestAnimationFrame(() => resizeGraphToContainer());
}

window.addEventListener('resize', () => {
  syncPanelWidthForViewport();
  requestAnimationFrame(() => resizeGraphToContainer());
});
window.addEventListener('keydown', event => {
  if ((event.metaKey || event.ctrlKey) && event.key === '\\') {
    event.preventDefault();
    if (event.shiftKey) toggleControlsDock();
    else togglePanel();
    return;
  }
  if (event.key === 'Escape') {
    if (focusMode) exitFocusMode();
    else hideTooltip();
  }
});

// ============================================================================
// Startup — connect transport independently from optional renderer dependencies
// ============================================================================
const RENDERER_READY_TIMEOUT_MS = 5000;

function waitForRendererReady(timeoutMs = RENDERER_READY_TIMEOUT_MS) {
  const rendererPromise = window.BDH3DReady;
  if (!rendererPromise || typeof rendererPromise.then !== 'function') return Promise.resolve(false);
  return Promise.race([
    Promise.resolve(rendererPromise).then(() => true).catch(() => false),
    new Promise(resolve => setTimeout(() => resolve(false), timeoutMs)),
  ]);
}

function recoverWhenRendererArrives() {
  const rendererPromise = window.BDH3DReady;
  if (!rendererPromise || typeof rendererPromise.then !== 'function') return;
  Promise.resolve(rendererPromise).then(() => {
    if (typeof window.ForceGraph3D !== 'function' || !window.THREE) return;
    webglUnavailable = false;
    hideWebGLFallback();
    fetchGraphSnapshot({ reason: 'renderer-late', preserveView: false, force: true });
  }).catch(error => {
    console.warn('[BDH 3D] Renderer dependency failed:', error);
  });
}

async function startVisualization() {
  restoreGraphControlState();
  // Force panels closed while the graph loader is visible — restored in hideGraphLoader().
  document.body.classList.add('panel-collapsed', 'controls-collapsed');
  setConnectionStatus('', 'Connecting');
  recoverWhenRendererArrives();

  // Safety net: if hideGraphLoader() is never called (CDN timeout, engine stall,
  // WS failure), restore panels after 12s so the UI is never stuck collapsed.
  setTimeout(() => {
    if (document.getElementById('graph-loader') && !document.getElementById('graph-loader').hidden) {
      console.warn('[BDH 3D] Safety timeout — forcing panel restore');
      if (typeof hideGraphLoader === 'function') hideGraphLoader();
    }
  }, 12000);

  try {
    if (typeof loadVaultSelector === 'function') await loadVaultSelector();
  } catch (error) {
    console.warn('[BDH 3D] Vault selector bootstrap failed:', error);
  }

  // Never hold the live transport hostage to a CDN/WebGL dependency.
  connectWS();

  const rendererReady = await waitForRendererReady();
  if (!rendererReady) {
    webglUnavailable = true;
    showWebGLFallback('The 3D renderer dependency is unavailable or still loading.');
    return;
  }

  hideWebGLFallback();
  fetchGraphSnapshot({ reason: 'renderer-ready', preserveView: false, force: true });
}

startVisualization();
