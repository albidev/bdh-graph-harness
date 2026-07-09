// ============================================================================
// Mobile tab switching
// ============================================================================
function switchTab(tabClass) {
  document.body.className = tabClass;
  document.querySelectorAll('#mobile-tabs .tab').forEach(t => t.classList.remove('active'));
  const labels = { 'graph-tab': 0, 'panel-tab': 1 };
  const tabs = document.querySelectorAll('#mobile-tabs .tab');
  if (tabs[labels[tabClass]]) tabs[labels[tabClass]].classList.add('active');
  // force-graph auto-resizes, no redraw needed
}

// ============================================================================
// Orphan nodes toggle
// ============================================================================
function toggleOrphans(show) {
  showOrphans = show;
  if (!graph) return;
  const currentData = graph.graphData();
  const currentIds = new Set(currentData.nodes.map(n => n.id));

  // Build fresh copies — never mutate live force-graph data
  const freshNodes = currentData.nodes.map(n => ({
    id: n.id, name: n.name, color: n.color, val: n.val,
    _opacity: n._opacity, _shape: n._shape, _dormant: n._dormant,
    _mass: n._mass,
    _tags: n._tags, _title: n._title, _path: n._path, _text: n._text,
    _hidden: n._hidden,
  }));
  const freshLinks = currentData.links.map(l => ({
    source: linkEndpointId(l.source), target: linkEndpointId(l.target),
    color: l.color, width: l.width, type: l.type,
    particles: l.particles, _id: l._id, _visible: l._visible,
    _dashes: l._dashes, weight: l.weight, frequency: l.frequency,
    particleColor: l.particleColor,
  }));

  if (show) {
    // Add orphan nodes
    orphanNodeIds.forEach(nid => {
      if (!currentIds.has(nid)) {
        const n = allGraphNodes.find(x => x.id === nid);
        if (n) {
          freshNodes.push({
            id: n.id,
            name: n.title,
            color: '#1c2128',
            val: 6,
            _mass: computeNodeMass(n, 0, Math.max(1, ...Object.values(degreeMap))),
            _opacity: 0.5,
            _shape: 'circle',
            _dormant: false,
            _tags: n.tags || '',
            _title: n.title,
            _path: n.path || '',
            _text: n.text || '',
          });
          nodeTagColorMap[n.id] = '#1c2128';
        }
      }
    });
  } else {
    // Remove orphan nodes
    const orphanSet = new Set(orphanNodeIds);
    const filtered = freshNodes.filter(n => !orphanSet.has(n.id));
    freshNodes.length = 0;
    freshNodes.push(...filtered);
  }

  setGraphDataPreservingView({ nodes: freshNodes, links: freshLinks }, { reheat: true });

  // Update neuron count
  const el = document.getElementById('stat-neurons');
  if (el) el.textContent = show ? allGraphNodes.length : (allGraphNodes.length - orphanNodeIds.length);
}

// ============================================================================
// Tag-based node coloring toggle
// ============================================================================
function toggleTagColors(enabled) {
  showTagColors = enabled;
  const legend = document.getElementById('tag-legend');

  if (legend) {
    if (enabled && Object.keys(tagColorMap).length > 0) {
      const seen = new Set();
      const items = [];
      Object.entries(tagColorMap)
        .sort((a, b) => a[0].localeCompare(b[0]))
        .forEach(([tag, color]) => {
          const key = tag.trim().toLowerCase();
          if (!seen.has(key)) {
            seen.add(key);
            items.push('<span class="tag-item"><span class="tag-dot" style="background:' + color + '"></span>' + escapeHtml(tag) + '</span>');
          }
        });
      const collapseBtn = '<button class="legend-collapse" onclick="toggleLegendCollapse()" title="Collapse legend">◀</button>';
      legend.innerHTML = collapseBtn + items.join('');
      legend.classList.remove('hidden');
      if (localStorage.getItem('bdh-legend-collapsed') === 'true') {
        legend.classList.add('collapsed');
      }
    } else {
      legend.classList.add('hidden');
    }
  }

  if (!graph) return;
  const currentData = graph.graphData();
  currentData.nodes.forEach(node => {
    const n = allGraphNodes.find(x => x.id === node.id);
    if (!n || node._dormant) return;
    if (enabled && tagColorMap) {
      const rawTags = n.tags || '';
      let primaryTag = '';
      if (Array.isArray(rawTags) && rawTags.length > 0) {
        primaryTag = rawTags[0].replace(/^[\[\]]+|[\[\]]+$/g, '').trim();
      } else if (typeof rawTags === 'string' && rawTags.trim()) {
        primaryTag = rawTags.replace(/^[\[\]]+|[\[\]]+$/g, '').split(',')[0].trim();
      }
      if (primaryTag && tagColorMap[primaryTag]) {
        node.color = tagColorMap[primaryTag];
        nodeTagColorMap[node.id] = tagColorMap[primaryTag];
      } else {
        node.color = COLORS.inactive;
        nodeTagColorMap[node.id] = COLORS.inactive;
      }
    } else {
      node.color = COLORS.inactive;
      nodeTagColorMap[node.id] = COLORS.inactive;
    }
  });
  requestGraphRedraw();
}

// ============================================================================
// Edge filters — direct-only, Hebbian threshold, phantom toggle
// ============================================================================
function restoreGraphControlState() {
  hebbianThreshold = clampNumber(loadControlValue(STORAGE_KEYS.hebbianThreshold, hebbianThreshold), 0, 1, hebbianThreshold);
  spacingValue = clampNumber(loadControlValue(STORAGE_KEYS.spacing, spacingValue), 0, 100, spacingValue);
  edgeLengthMultiplier = clampNumber(loadControlValue(STORAGE_KEYS.edgeLength, edgeLengthMultiplier), 0, 100, edgeLengthMultiplier);
  restoredZoom = clampNumber(loadControlValue(STORAGE_KEYS.zoom, ''), ZOOM_MIN, ZOOM_MAX, null);

  const thresholdSlider = document.getElementById('hebbian-threshold');
  const thresholdVal = document.getElementById('threshold-val');
  const spacingSlider = document.getElementById('spacing-slider');
  const edgeLengthSlider = document.getElementById('edge-length-slider');
  const edgeLengthVal = document.getElementById('el-val');
  if (thresholdSlider) thresholdSlider.value = hebbianThreshold;
  if (thresholdVal) thresholdVal.textContent = hebbianThreshold;
  if (spacingSlider) spacingSlider.value = spacingValue;
  if (edgeLengthSlider) edgeLengthSlider.value = edgeLengthMultiplier;
  if (edgeLengthVal) edgeLengthVal.textContent = edgeLengthMultiplier;
}

function applyRestoredGraphControls() {
  if (!graph) return;
  setEdgeLengthMultiplier(edgeLengthMultiplier, false);
  updateSpacing(spacingValue, false);
  // Saved zoom alone is not enough to restore a view: force-graph layouts are
  // re-simulated on every page load, so the previous camera center can point at
  // empty space. First fit the freshly-laid-out graph, then optionally restore
  // the user's zoom level while keeping the newly-correct center.
  requestAnimationFrame(() => {
    if (!graph) return;
    graph.zoomToFit(0, 60);
    if (restoredZoom != null) {
      requestAnimationFrame(() => {
        if (!graph) return;
        graph.zoom(restoredZoom);
        syncZoomUI(false);
      });
    } else {
      syncZoomUI(false);
    }
  });
}

function toggleDirectOnly(on) {
  directOnly = on;
  const thresholdCtrl = document.getElementById('threshold-control');
  if (thresholdCtrl) thresholdCtrl.style.display = on ? 'none' : 'flex';
  applyEdgeFilters();
}

function updateHebbianThreshold(val, persist = true) {
  hebbianThreshold = clampNumber(val, 0, 1, 0.3);
  const slider = document.getElementById('hebbian-threshold');
  if (slider) slider.value = hebbianThreshold;
  document.getElementById('threshold-val').textContent = hebbianThreshold;
  if (persist) saveControlValue(STORAGE_KEYS.hebbianThreshold, hebbianThreshold);
  applyEdgeFiltersDebounced();
}

let _applyEdgeFiltersTimer = null;
function applyEdgeFiltersDebounced() {
  clearTimeout(_applyEdgeFiltersTimer);
  _applyEdgeFiltersTimer = setTimeout(applyEdgeFilters, 30);
}

function applyEdgeFilters() {
  if (!graph) return;
  // Update visibility Map — no graph rebuild needed
  const currentData = graph.graphData();
  let visibleCount = 0;
  linkVisibilityState.clear();
  currentData.links.forEach(l => {
    const info = edgeInfoMap[l._id] || {};
    let visible = true;
    if (directOnly) {
      visible = info.type === 'wikilink';
    } else {
      // Check edge type toggle first
      if (!edgeTypeVisible[info.type]) {
        visible = false;
      } else if (info.type === 'hebbian') {
        visible = (info.weight || 0) >= hebbianThreshold;
      } else if (info.type === 'phantom') {
        visible = showPhantom;
      }
    }
    if (visible) visibleCount++;
    linkVisibilityState.set(linkKey(l), visible);
  });

  // Update visible synapse count
  const synEl = document.getElementById('stat-synapses');
  if (synEl) synEl.textContent = visibleCount;

  // Force force-graph to re-evaluate linkVisibility by passing fresh data
  setGraphDataPreservingView({ nodes: currentData.nodes, links: currentData.links });
}

// Populate linkVisibilityState Map WITHOUT triggering a graph rebuild.
// Used during init to apply saved filter state without breaking interactions.
function initEdgeVisibility() {
  if (!graph) return;
  const currentData = graph.graphData();
  linkVisibilityState.clear();
  currentData.links.forEach(l => {
    const info = edgeInfoMap[l._id] || {};
    let visible = true;
    if (directOnly) {
      visible = info.type === 'wikilink';
    } else {
      if (!edgeTypeVisible[info.type]) {
        visible = false;
      } else if (info.type === 'hebbian') {
        visible = (info.weight || 0) >= hebbianThreshold;
      } else if (info.type === 'phantom') {
        visible = showPhantom;
      }
    }
    linkVisibilityState.set(linkKey(l), visible);
  });
}

function togglePhantom(on) {
  showPhantom = on;
  applyEdgeFilters();
}

// ============================================================================
// Edge length multiplier — d3Force link distance
// ============================================================================
function setEdgeLengthMultiplier(m, persist = true) {
  edgeLengthMultiplier = clampNumber(m, 0, 100, 10);
  const slider = document.getElementById('edge-length-slider');
  const valEl = document.getElementById('el-val');
  if (slider) slider.value = edgeLengthMultiplier;
  if (valEl) valEl.textContent = edgeLengthMultiplier;
  if (persist) saveControlValue(STORAGE_KEYS.edgeLength, edgeLengthMultiplier);

  if (graph) {
    // Map 0-100 to link distance 40-300, then adjust per edge type/mass.
    const distance = 40 + edgeLengthMultiplier * 2.6;
    graph.d3Force('link').distance(link => massAwareLinkDistance(link, distance));
    // Re-heat simulation so nodes react to new distance
    graph.d3ReheatSimulation();
  }

  // Scale node sizes in-place (no graphData reset — avoids jump/pan)
  if (graph) {
    const currentData = graph.graphData();
    const maxDeg = Math.max(1, ...Object.values(degreeMap));
    const nodeScale = 0.5 + (edgeLengthMultiplier / 50);
    const MIN_VAL = 4, MAX_VAL = 40;
    currentData.nodes.forEach(node => {
      const deg = degreeMap[node.id] || 0;
      // Modify node directly — it's a reference, next render frame picks it up
      node.val = Math.max(2, (MIN_VAL + (deg / maxDeg) * (MAX_VAL - MIN_VAL)) * nodeScale);
    });
    // Do NOT call graph.graphData() — that would reset positions and cause a jump
  }

  console.log('Edge length: ' + edgeLengthMultiplier + ', node scale: ' + (0.5 + edgeLengthMultiplier / 50).toFixed(2) + 'x');
}

// ============================================================================
// Network spacing — d3Force charge strength
// ============================================================================
function updateSpacing(val, persist = true) {
  spacingValue = clampNumber(val, 0, 100, 50);
  const slider = document.getElementById('spacing-slider');
  if (slider) slider.value = spacingValue;
  if (persist) saveControlValue(STORAGE_KEYS.spacing, spacingValue);

  const t = spacingValue / 100; // 0..1
  // Map: 0=dense (charge -300) → 1=sparse (charge -1200), then scale by node mass
  const chargeStrength = Math.round(-300 - t * 900);
  if (graph) {
    graph.d3Force('charge').strength(node => massAwareChargeStrength(node, chargeStrength));
    graph.d3ReheatSimulation();
  }
}

// ============================================================================
// Legend collapse toggle
// ============================================================================
function toggleLegendCollapse() {
  const legend = document.getElementById('tag-legend');
  if (!legend) return;
  legend.classList.toggle('collapsed');
  localStorage.setItem('bdh-legend-collapsed', legend.classList.contains('collapsed'));
}

document.getElementById('tag-legend').addEventListener('click', function(e) {
  if (this.classList.contains('collapsed') && e.target === this) {
    toggleLegendCollapse();
  }
});

// ============================================================================
// Query input
// ============================================================================
document.getElementById('query-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendQuery();
});

const qInput = document.getElementById('query-input');
const qClear = document.getElementById('query-clear');
function updateClearBtn() { qClear.classList.toggle('visible', qInput.value.length > 0); }
qInput.addEventListener('input', updateClearBtn);
function clearQuery() { qInput.value = ''; updateClearBtn(); qInput.focus(); }

// ============================================================================
// Reset — clears results and restores graph to pre-query state
// ============================================================================
function resetQuery() {
  qInput.value = '';
  updateClearBtn();
  document.getElementById('response-text').textContent = '—';
  document.getElementById('activated-list').innerHTML = '<div class="empty">No activations yet</div>';

  if (graph) {
    clearActivationState();
    endQueryParticles();
    requestGraphRedraw();
    applyEdgeFilters();
  }
}

// ============================================================================
// Zoom controls
// ============================================================================
const ZOOM_MIN = 0.1;
const ZOOM_MAX = 4;
const zoomSlider = document.getElementById('zoom-slider');
const zoomLabel = document.getElementById('zoom-value');

function syncZoomUI(persist = false) {
  if (!graph) return;
  const s = graph.zoom();
  zoomSlider.value = s;
  zoomLabel.textContent = Math.round(s * 100) + '%';
  if (persist) saveControlValue(STORAGE_KEYS.zoom, s);
}

function zoomTo(val, persist = true) {
  if (!graph) return;
  const s = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, parseFloat(val)));
  graph.zoom(s);
  syncZoomUI(persist);
}

function zoomStep(delta) {
  if (!graph) return;
  zoomTo(graph.zoom() + delta, true);
}

function fitToScreen() {
  if (!graph) return;
  graph.zoomToFit(300, 60);
  setTimeout(() => syncZoomUI(true), 350);
}

// Poll zoom value to keep slider in sync (force-graph has no zoom event)
function startZoomPoll() {
  if (zoomPollTimer) clearInterval(zoomPollTimer);
  zoomPollTimer = setInterval(() => {
    if (graph) {
      const s = graph.zoom();
      if (Math.abs(parseFloat(zoomSlider.value) - s) > 0.01) {
        zoomSlider.value = s;
        zoomLabel.textContent = Math.round(s * 100) + '%';
      }
    }
  }, 200);
}

// ============================================================================
// Panel resize & collapse
// ============================================================================
const PANEL_WIDTH_KEY = 'bdh-panel-width';
const PANEL_COLLAPSED_KEY = 'bdh-panel-collapsed';
const sidePanel = document.getElementById('side-panel');
const panelResize = document.getElementById('panel-resize');
const collapseBtn = document.getElementById('collapse-btn');
let isResizing = false;

function savePanelWidth() {
  if (sidePanel.style.width) localStorage.setItem(PANEL_WIDTH_KEY, sidePanel.style.width);
}

function restorePanelWidth() {
  const saved = localStorage.getItem(PANEL_WIDTH_KEY);
  if (saved) {
    const pct = Math.max(15, Math.min(60, parseFloat(saved)));
    sidePanel.style.width = pct + '%';
  }
}

function restorePanelState() {
  if (localStorage.getItem(PANEL_COLLAPSED_KEY) === 'true') {
    document.body.classList.add('panel-collapsed');
    if (collapseBtn) collapseBtn.textContent = '▸';
  }
}

panelResize.addEventListener('mousedown', (e) => {
  e.preventDefault();
  isResizing = true;
  panelResize.classList.add('dragging');
  document.body.style.cursor = 'col-resize';
  document.body.style.userSelect = 'none';
});

document.addEventListener('mousemove', (e) => {
  if (!isResizing) return;
  const mainRect = document.getElementById('main').getBoundingClientRect();
  const handleX = e.clientX - mainRect.left;
  const totalW = mainRect.width;
  let panelPct = ((totalW - handleX) / totalW) * 100;
  panelPct = Math.max(15, Math.min(60, panelPct));
  sidePanel.style.width = panelPct + '%';
});

document.addEventListener('mouseup', () => {
  if (!isResizing) return;
  isResizing = false;
  panelResize.classList.remove('dragging');
  document.body.style.cursor = '';
  document.body.style.userSelect = '';
  savePanelWidth();
});

function togglePanel() {
  document.body.classList.toggle('panel-collapsed');
  const collapsed = document.body.classList.contains('panel-collapsed');
  if (collapseBtn) collapseBtn.textContent = collapsed ? '◂' : '▸';
  localStorage.setItem(PANEL_COLLAPSED_KEY, collapsed);
}

// ============================================================================
// Window resize — force-graph auto-resizes to container
// ============================================================================
window.addEventListener('resize', () => {
  // force-graph handles resize automatically via its container
});

// ============================================================================
// Init
// ============================================================================
restoreGraphControlState();
restorePanelWidth();
restorePanelState();
connectWS();

// ============================================================================
// Search + focus — find node by name, center viewport and zoom in
// ============================================================================
function searchNode(query) {
  if (!query || !graph) return;
  const q = query.trim().toLowerCase();
  if (!q) return;

  const data = graph.graphData();
  // Find first node whose name contains the query
  const match = data.nodes.find(n => (n.name || '').toLowerCase().includes(q));
  if (!match) {
    // Flash the search input red briefly
    const el = document.getElementById('search-input');
    if (el) { el.style.borderColor = '#f97583'; setTimeout(() => { el.style.borderColor = ''; }, 800); }
    return;
  }

  // Center on the node and zoom in
  graph.centerAt(match.x, match.y, 600);
  graph.zoom(2.5, 600);

  // Highlight the found node
  setHoverHighlight([match.id]);
  showTooltip(match, lastMouseEvent || { clientX: 200, clientY: 200 });
}

// ============================================================================
// Edge type filtering — toggle visibility by edge type (wikilink/hebbian/phantom)
// ============================================================================
const edgeTypeVisible = { wikilink: true, hebbian: true, phantom: true };

function toggleEdgeType(type, btn) {
  edgeTypeVisible[type] = !edgeTypeVisible[type];
  if (btn) btn.classList.toggle('active', edgeTypeVisible[type]);
  applyEdgeFilters();
}

// ============================================================================
// Stats dashboard — sync bottom-left mini-dashboard with top-bar stats
// ============================================================================
function updateStatsDashboard() {
  const neurons = document.getElementById('stat-neurons')?.textContent || '—';
  const synapses = document.getElementById('stat-synapses')?.textContent || '—';
  const hebbian = document.getElementById('stat-hebbian')?.textContent || '—';
  const queries = document.getElementById('stat-queries')?.textContent || '—';
  const dashN = document.getElementById('dash-neurons');
  const dashS = document.getElementById('dash-synapses');
  const dashH = document.getElementById('dash-hebbian');
  const dashQ = document.getElementById('dash-queries');
  if (dashN) dashN.textContent = neurons;
  if (dashS) dashS.textContent = synapses;
  if (dashH) dashH.textContent = hebbian;
  if (dashQ) dashQ.textContent = queries;
}
// Periodic sync — dashboard mirrors the top-bar stats
setInterval(updateStatsDashboard, 1000);
