// ============================================================================
// BDH Graph Harness — force-graph visualization
// Replaces vis.js with WebGL-powered force-graph for 60 FPS on large vaults
// ============================================================================

const COLORS = {
  inactive: '#6e7681',
  activated: '#f0883e',
  seed: '#58a6ff',
  neurogenesis: '#3fb950',  // green — new born nodes
  dormant: '#30363d',       // dark gray — dormant/low-quality nodes
  edgeWikilink: '#484f58',
  edgeHebbianLow: '#3b2066',
  edgeHebbianMid: '#8957e5',
  edgeHebbianHigh: '#d2a8ff',
  edgeHebbianPulse: '#d2a8ff',
  edgeNeurogenesis: '#3fb950',  // green dashed edges for new connections
  edgePhantom: '#1f6feb',       // blue dashed edges for phantom links
  bg: '#0d1117',
};

// Logarithmic node radius — compresses high-degree hubs so they don't dominate.
// val=4 → r≈5.6, val=20 → r≈9.3, val=40 → r≈10.7 (vs old sqrt: 4→8.9, 20→20, 40→28.3)
function nodeRadius(val) {
  return Math.log2((val || 4) + 1) * 4 + 2;
}

// Semantic/structural mass used by the physics layer. d3-force itself has no
// real per-node mass, so we emulate it consistently: heavier nodes repel more,
// drift less toward tag centroids, and move less when collision resolution runs.
function computeNodeMass(node, degree, maxDegree) {
  const degNorm = maxDegree > 0 ? Math.max(0, Math.min(1, degree / maxDegree)) : 0;
  let mass = degree <= 0 ? 0.45 : 0.7 + Math.sqrt(degNorm) * 2.4;
  const tags = (node && node.tags ? String(node.tags) : '').toLowerCase();
  if (node && node.dormant) mass *= 0.65;
  if (tags.includes('neurogenesis')) mass *= 1.15;
  return Math.max(0.35, Math.min(3.4, mass));
}

function massAwareChargeStrength(node, baseStrength) {
  const mass = node && node._mass ? node._mass : 1;
  return baseStrength * (0.65 + Math.sqrt(mass) * 0.45);
}

function endpointMass(endpoint) {
  return endpoint && typeof endpoint === 'object' && endpoint._mass ? endpoint._mass : 1;
}

function massAwareLinkDistance(link, baseDistance) {
  const avgMass = (endpointMass(link.source) + endpointMass(link.target)) / 2;
  let typeScale = 1.0;
  if (link.type === 'hebbian') {
    // Strong Hebbian links should be closer, but weak rendered Hebbian links
    // should not act like rubber bands that collapse the whole graph.
    const weight = typeof link.weight === 'number' ? link.weight : 0.3;
    typeScale = 1.35 - Math.min(0.55, weight * 0.8);
  } else if (link.type === 'phantom') {
    typeScale = 1.35;
  } else if (link.type === 'wikilink') {
    typeScale = 1.05;
  }
  const massSpread = 1 + Math.max(0, avgMass - 1) * 0.12;
  return baseDistance * typeScale * massSpread;
}

// Convert hex color to rgba with alpha (for edge opacity by type)
function withAlpha(hex, alpha) {
  if (!hex || hex[0] !== '#') return hex;
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// Edge opacity by type — reduces visual noise from 3500+ Hebbian edges
const EDGE_OPACITY = {
  wikilink: 0.45,
  hebbian: 0.22,
  phantom: 0.35,
  neurogenesis: 0.7,
};
const HEBBIAN_MIN_RENDER_WEIGHT = 0.15;

// Hebbian edge color by weight (dim → bright green)
function weightColor(weight) {
  if (weight < 0.3) return COLORS.edgeHebbianLow;
  if (weight < 0.6) return COLORS.edgeHebbianMid;
  return COLORS.edgeHebbianHigh;
}

// Tag-based color palette (distinct, readable on dark bg)
const TAG_COLORS = [
  '#f97583', // red (entities)
  '#79c0ff', // blue (concepts)
  '#7ee787', // green (lessons)
  '#d2a8ff', // purple (comparisons)
  '#ffa657', // orange (queries)
  '#ff7b72', // coral
  '#56d4dd', // cyan
  '#e3b341', // yellow
  '#db61a2', // pink
  '#a5d6ff', // light blue
  '#b392f0', // lavender
  '#85e89d', // mint
];

const STORAGE_KEYS = {
  hebbianThreshold: 'bdh-graph-hebbian-threshold-v2',
  spacing: 'bdh-graph-spacing',
  edgeLength: 'bdh-graph-edge-length',
  zoom: 'bdh-graph-zoom',
};

function clampNumber(value, min, max, fallback) {
  if (value == null || value === '') return fallback;
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

function saveControlValue(key, value) {
  try { localStorage.setItem(key, String(value)); }
  catch(e) { /* Storage can be blocked; controls still work for this session. */ }
}

function loadControlValue(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value == null ? fallback : value;
  } catch(e) {
    return fallback;
  }
}

// Console-tunable neural particle bloom. Default mode is subtle ambient flow:
//   BDHParticles.preset('insane')   // manual temporary fireworks
//   BDHParticles.preset('subtle')   // return to always-on calm synapses
//   BDHParticles.get()
const PARTICLE_CONFIG_KEY = 'bdh-particle-bloom-config';
const PARTICLE_PRESETS = {
  subtle: {
    enabled: true, ambient: true,
    activeBloom: 18, activeCore: 3.6, activeHalo: 13, activeParticles: 6,
    ambientBloom: 6, ambientCore: 1.4, ambientHalo: 6, ambientAlpha: 0.28,
    ambientParticles: 1, ambientThreshold: 0.78, speed: 0.01,
    activeAlpha: 0.95, ambientColor: '#d2a8ff', activeColor: '#f0d2ff',
  },
  loud: {
    enabled: true, ambient: true,
    activeBloom: 38, activeCore: 5.8, activeHalo: 24, activeParticles: 11,
    ambientBloom: 12, ambientCore: 2.2, ambientHalo: 10, ambientAlpha: 0.5,
    ambientParticles: 1, ambientThreshold: 0.68, speed: 0.016,
    activeAlpha: 0.95, ambientColor: '#d2a8ff', activeColor: '#f0d2ff',
  },
  insane: {
    enabled: true, ambient: true,
    activeBloom: 58, activeCore: 7, activeHalo: 32, activeParticles: 16,
    ambientBloom: 18, ambientCore: 2.8, ambientHalo: 14, ambientAlpha: 0.65,
    ambientParticles: 1, ambientThreshold: 0.6, speed: 0.02,
    activeAlpha: 0.98, ambientColor: '#d2a8ff', activeColor: '#f0d2ff',
  },
  off: { enabled: false, ambient: false },
  ambientOff: { enabled: true, ambient: false },
};
const DEFAULT_PARTICLE_CONFIG = { ...PARTICLE_PRESETS.subtle };
let particleConfig = { ...DEFAULT_PARTICLE_CONFIG };
let queryParticleMode = false;
let queryParticleTimer = null;

function loadParticleConfig() {
  // The app default is intentionally subtle. Old localStorage values (especially
  // a manually saved `insane`) must not make the graph rave forever after reload.
  particleConfig = { ...DEFAULT_PARTICLE_CONFIG };
}

function saveParticleConfig() {
  try { localStorage.setItem(PARTICLE_CONFIG_KEY, JSON.stringify(particleConfig)); }
  catch(e) { /* Ignore storage failures; console controls still work in-session. */ }
}

function applyParticlePreset(name, { persist = true } = {}) {
  const preset = PARTICLE_PRESETS[name];
  if (!preset) return { error: 'Unknown preset', presets: Object.keys(PARTICLE_PRESETS) };
  particleConfig = { ...particleConfig, ...preset };
  if (persist) saveParticleConfig();
  requestGraphRedraw();
  return { ...particleConfig, preset: name };
}

function beginQueryParticles() {
  queryParticleMode = true;
  if (queryParticleTimer) clearTimeout(queryParticleTimer);
  applyParticlePreset('insane', { persist: false });
}

function endQueryParticles(delayMs = 0) {
  if (queryParticleTimer) clearTimeout(queryParticleTimer);
  queryParticleTimer = setTimeout(() => {
    queryParticleMode = false;
    applyParticlePreset('subtle', { persist: false });
    queryParticleTimer = null;
  }, Math.max(0, delayMs));
}

function isActiveFlowLink(link) {
  return linkParticlesState.has(linkKey(link));
}

function isAmbientFlowLink(link) {
  return particleConfig.enabled && particleConfig.ambient && link.type === 'hebbian' && (link.weight || 0) >= particleConfig.ambientThreshold;
}

function particleSpeed(link) {
  return isActiveFlowLink(link) ? particleConfig.speed * 1.35 : particleConfig.speed;
}

function drawNeuralParticle(x, y, link, ctx, globalScale) {
  if (!particleConfig.enabled) return;
  const active = isActiveFlowLink(link);
  const core = active ? particleConfig.activeCore : particleConfig.ambientCore;
  const halo = active ? particleConfig.activeHalo : particleConfig.ambientHalo;
  const bloom = active ? particleConfig.activeBloom : particleConfig.ambientBloom;
  const alpha = active ? particleConfig.activeAlpha : particleConfig.ambientAlpha;
  const color = active
    ? (linkParticleColorState.get(linkKey(link)) || particleConfig.activeColor)
    : (particleConfig.ambientColor || link.color || COLORS.edgeHebbianHigh);
  const scaleDamp = Math.max(0.6, Math.min(1.4, 1 / Math.sqrt(globalScale || 1)));
  const coreR = Math.max(0.8, core * scaleDamp);
  const haloR = Math.max(coreR + 1, halo * scaleDamp);

  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  ctx.shadowColor = color;
  ctx.shadowBlur = bloom * scaleDamp;

  const grad = ctx.createRadialGradient(x, y, 0, x, y, haloR);
  grad.addColorStop(0, withAlpha(color, alpha));
  grad.addColorStop(0.35, withAlpha(color, alpha * 0.38));
  grad.addColorStop(1, withAlpha(color, 0));
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(x, y, haloR, 0, 2 * Math.PI);
  ctx.fill();

  ctx.shadowBlur = bloom * 0.45 * scaleDamp;
  ctx.fillStyle = withAlpha('#ffffff', active ? 0.92 : 0.55);
  ctx.beginPath();
  ctx.arc(x, y, coreR, 0, 2 * Math.PI);
  ctx.fill();

  ctx.fillStyle = withAlpha(color, active ? 0.95 : 0.65);
  ctx.beginPath();
  ctx.arc(x, y, coreR * 1.55, 0, 2 * Math.PI);
  ctx.fill();
  ctx.restore();
}

function installParticleConsoleControls() {
  loadParticleConfig();
  window.BDHParticles = {
    get() { return { ...particleConfig }; },
    set(next = {}) {
      particleConfig = { ...particleConfig, ...next };
      saveParticleConfig();
      requestGraphRedraw();
      return { ...particleConfig };
    },
    reset() {
      particleConfig = { ...DEFAULT_PARTICLE_CONFIG };
      saveParticleConfig();
      requestGraphRedraw();
      return { ...particleConfig, preset: 'subtle' };
    },
    preset(name) {
      return applyParticlePreset(name, { persist: true });
    },
    beginQuery() { beginQueryParticles(); return { ...particleConfig, mode: 'query' }; },
    endQuery() { endQueryParticles(); return { ...particleConfig, mode: 'subtle' }; },
  };
}
installParticleConsoleControls();

// ============================================================================
// Global state
// ============================================================================
let graph = null;             // force-graph instance
let allGraphNodes = [];       // full node list from server
let fgNodes = [];             // force-graph node objects
let fgLinks = [];             // force-graph link objects
let hebbianMap = {};          // "a|b" -> weight
let orphanNodeIds = [];       // nodes with no connections (hidden by default)
let tagColorMap = {};         // tag -> color for tag-based node coloring
let showTagColors = true;     // toggle for tag-based coloring (on by default)
let directOnly = false;       // toggle for showing only direct wikilink edges
let hebbianThreshold = 0.15;   // minimum weight for Hebbian edges to show
let showPhantom = true;       // toggle for phantom (semantic similarity) edges
let degreeMap = {};           // node_id -> degree (computed from edges)
let neighborMap = {};         // node_id -> [connected node titles]
let neurogenesisNodes = {};   // node_id -> node data (preserved across graph refresh)
let nodeDataMap = {};         // node_id -> full node data (fast tooltip lookup)
let edgeInfoMap = {};         // edge_id -> {source_title, target_title, type, weight?, frequency?}
let nodeTagColorMap = {};     // node_id -> color string (preserves tag color across dim/restore)
let totalConcepts = 0;        // cumulative neurogenesis counter
let edgeLengthMultiplier = 10; // default: 10px per degree unit
let spacingValue = 50;        // default: balanced graph spacing
let restoredZoom = null;      // saved zoom slider value, applied after graph init
let showOrphans = true;
let zoomPollTimer = null;
let lastMouseEvent = { clientX: 0, clientY: 0 }; // tracked for tooltip positioning
let mouseTrackingInstalled = false;
let hoverHighlight = null;       // { nodeIds:Set, linkIds:Set, hoverNodeId?, hoverLinkId? }
let hoverHighlightKey = null;
let hoverClearFrame = null;
let hoverEdgeId = null;           // edge-only hover cue; does not dim/highlight the graph

// ============================================================================
// Custom HTML tooltip
// ============================================================================
let tooltipEl = null;

function ensureTooltip() {
  if (tooltipEl) return;
  tooltipEl = document.createElement('div');
  tooltipEl.id = 'custom-tooltip';
  tooltipEl.style.cssText = 'display:none;position:fixed;z-index:9999;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 14px;pointer-events:none;max-width:300px;font-size:12px;line-height:1.5;color:#c9d1d9;box-shadow:0 4px 12px rgba(0,0,0,0.4)';
  document.body.appendChild(tooltipEl);
}

function showTooltip(node, evt) {
  ensureTooltip();
  const n = node ? nodeDataMap[node.id] : null;
  if (!n) { hideTooltip(); return; }
  const mobile = isMobile();

  const title = escapeHtml(n.title || n.id);
  const path = n._path || n.path || '';
  const shortPath = path.replace(/.*\/Documents\/Hermes\//, '').replace(/.*\/wiki\//, 'wiki/');
  const text = (n._text || n.text || '').replace(/\n/g, ' ').substring(0, mobile ? 80 : 150);
  const deg = degreeMap[n.id] || 0;

  // Tags
  const rawTags = n._tags || n.tags || '';
  let tagList = [];
  if (Array.isArray(rawTags)) {
    tagList = rawTags.map(t => t.replace(/^[\[\]]+|[\[\]]+$/g, '').trim()).filter(Boolean);
  } else if (typeof rawTags === 'string' && rawTags.trim()) {
    tagList = rawTags.replace(/^[\[\]]+|[\[\]]+$/g, '').split(',').map(t => t.trim()).filter(Boolean);
  }
  const tagHtml = tagList.map(t => {
    const c = tagColorMap[t] || '#6e7681';
    return '<span style="display:inline-flex;align-items:center;gap:3px;margin:1px 6px 1px 0;font-size:11px"><span style="width:7px;height:7px;border-radius:50%;background:' + c + ';display:inline-block"></span>' + escapeHtml(t) + '</span>';
  }).join('');

  // Connected nodes (first 6)
  const neighbors = neighborMap[n.id] || [];
  const neighborLimit = mobile ? 2 : 6;
  const neighborHtml = neighbors.slice(0, mobile ? 2 : 6).map(nb => '<span style="color:#58a6ff">' + escapeHtml(nb) + '</span>').join(', ') + (neighbors.length > neighborLimit ? ' <span style="color:#6e7681">+' + (neighbors.length - neighborLimit) + ' more</span>' : '');

  let html = mobile ? '<button class="mobile-sheet-close" onclick="dismissMobileSheet()" aria-label="Close node details">×</button>' : '';
  html += '<div style="font-weight:600;color:#f0883e;margin-bottom:4px;font-size:13px">' + title + '</div>';
  if (tagHtml) html += '<div style="margin-bottom:6px">' + tagHtml + '</div>';
  if (shortPath) html += '<div style="color:#8b949e;font-size:11px;margin-bottom:3px">📄 ' + escapeHtml(shortPath) + '</div>';
  html += '<div style="color:#8b949e;font-size:11px;margin-bottom:3px">🔗 ' + deg + ' connection' + (deg !== 1 ? 's' : '') + '</div>';
  if (neighborHtml) html += '<div style="color:#6e7681;font-size:11px;margin-bottom:4px;border-top:1px solid #30363d;padding-top:4px">→ ' + neighborHtml + '</div>';
  if (text) html += '<div style="color:#6e7681;font-size:11px;border-top:1px solid #30363d;padding-top:4px">' + escapeHtml(text) + '…</div>';

  tooltipEl.innerHTML = html;
  tooltipEl.classList.toggle('mobile-tooltip', isMobile());
  tooltipEl.style.display = 'block';
  positionTooltip(evt);
}

function showEdgeTooltip(link, evt) {
  ensureTooltip();
  const info = edgeInfoMap[link._id];
  if (!info) { hideTooltip(); return; }

  const isHebbian = info.type === 'hebbian';
  const icon = isHebbian ? '⚡' : (info.type === 'phantom' ? '👻' : '🔗');
  const typeLabel = isHebbian ? 'Hebbian Synapse' : (info.type === 'phantom' ? 'Phantom Link' : 'Wikilink');

  let html = '<div style="max-width:280px">';
  html += '<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">';
  html += '<span style="font-size:14px">' + icon + '</span>';
  html += '<span style="font-weight:600;color:' + (isHebbian ? weightColor(info.weight || 0) : '#8b949e') + '">' + typeLabel + '</span>';
  html += '</div>';
  html += '<div style="margin-bottom:4px">';
  html += '<span style="color:#58a6ff">' + escapeHtml(info.source_title) + '</span>';
  html += ' <span style="color:#6e7681">→</span> ';
  html += '<span style="color:#58a6ff">' + escapeHtml(info.target_title) + '</span>';
  html += '</div>';

  if (isHebbian) {
    html += '<div style="display:flex;gap:16px;font-size:11px;color:#8b949e">';
    html += '<span>Weight: <b style="color:' + weightColor(info.weight) + '">' + info.weight.toFixed(3) + '</b></span>';
    html += '<span>Freq: <b>' + info.frequency + '</b></span>';
    html += '</div>';
  } else if (info.type === 'phantom') {
    html += '<div style="font-size:11px;color:#8b949e">Similarity: <b>' + (info.similarity || 0).toFixed(2) + '</b></div>';
  }

  html += '</div>';

  tooltipEl.innerHTML = html;
  tooltipEl.style.display = 'block';
  positionTooltip(evt);
}

function isMobile() {
  return window.matchMedia('(max-width: 768px)').matches;
}

function positionTooltip(evt) {
  if (!tooltipEl) return;
  if (isMobile()) {
    const bottom = 'calc(env(safe-area-inset-bottom) + 12px)';
    tooltipEl.style.left = '10px';
    tooltipEl.style.right = '10px';
    tooltipEl.style.bottom = bottom;
    tooltipEl.style.top = 'auto';
    tooltipEl.style.maxWidth = 'calc(100vw - 20px)';
    tooltipEl.style.width = 'auto';
  } else {
    const pad = 16;
    let x = (evt.clientX || evt.pageX || 0) + pad;
    let y = (evt.clientY || evt.pageY || 0) + pad;
    tooltipEl.style.width = '';
    tooltipEl.style.right = '';
    tooltipEl.style.bottom = '';
    tooltipEl.classList.remove('mobile-tooltip');
    const rect = tooltipEl.getBoundingClientRect();
    if (x + rect.width > window.innerWidth) x = (evt.clientX || evt.pageX || 0) - rect.width - pad;
    if (y + rect.height > window.innerHeight) y = (evt.clientY || evt.pageY || 0) - rect.height - pad;
    tooltipEl.style.left = x + 'px';
    tooltipEl.style.top = y + 'px';
  }
}

function envInset() {
  try { return parseInt(getComputedStyle(document.documentElement).getPropertyValue('env(safe-area-inset-bottom)')) || 0; }
  catch(e) { return 0; }
}

function hideTooltip() {
  if (tooltipEl) tooltipEl.style.display = 'none';
}

// Exposed for the mobile sheet close button. Kept separate from hover cleanup so
// a deliberate dismiss does not disturb the selected graph context.
function dismissMobileSheet() {
  hideTooltip();
}

function linkEndpointId(endpoint) {
  return typeof endpoint === 'object' ? endpoint.id : endpoint;
}

// Stable key for a link: "sourceId→targetId:type" sorted alphabetically.
function linkKey(link) {
  const s = linkEndpointId(link.source);
  const t = linkEndpointId(link.target);
  return s < t ? s + '→' + t + ':' + (link.type || 'syn') : t + '→' + s + ':' + (link.type || 'syn');
}

function isHoverActive() {
  return !!(hoverHighlight && (hoverHighlight.nodeIds.size || hoverHighlight.linkIds.size));
}

function isHoverNode(node) {
  return isHoverActive() && hoverHighlight.nodeIds.has(node.id);
}

function isHoverLink(link) {
  return isHoverActive() && hoverHighlight.linkIds.has(link._id);
}

function isHoverEdge(link) {
  return hoverEdgeId && link && link._id === hoverEdgeId;
}

function hoverKey(seedNodeIds = [], seedLinkId = null) {
  return seedNodeIds.slice().sort().join('|') + '::' + (seedLinkId || '');
}

function setHoverHighlight(seedNodeIds = [], seedLinkId = null) {
  if (!graph) return;
  cancelScheduledHoverClear();
  clearHoverEdge(false);
  const nextKey = hoverKey(seedNodeIds, seedLinkId);
  if (hoverHighlightKey === nextKey) return;
  const data = graph.graphData();
  const seedSet = new Set(seedNodeIds.filter(Boolean));
  const nodeIds = new Set(seedSet);
  const linkIds = new Set(seedLinkId ? [seedLinkId] : []);

  // DEBUG: log traversal for hovered node
  const _debugLinks = { total: 0, phantomSkip: 0, invisSkip: 0, actSkip: 0, visSkip: 0, matched: 0 };
  _debugLinks.total = data.links.length;

  data.links.forEach(link => {
    if (link._visible === false) { _debugLinks.invisSkip++; return; }
    if (linkActivationVisible.get(linkKey(link)) === false) { _debugLinks.actSkip++; return; }
    if (link.type === 'phantom') { _debugLinks.phantomSkip++; return; }
    // Skip links hidden by hebbian threshold slider — they're not visually connected
    const linkVis = linkVisibilityState.get(linkKey(link));
    if (linkVis === false) { _debugLinks.visSkip++; return; }
    const sourceId = linkEndpointId(link.source);
    const targetId = linkEndpointId(link.target);
    if (seedSet.has(sourceId) || seedSet.has(targetId) || link._id === seedLinkId) {
      _debugLinks.matched++;
      linkIds.add(link._id);
      nodeIds.add(sourceId);
      nodeIds.add(targetId);
    }
  });
  // Expose for console debugging
  window._hoverDebug = { seed: seedNodeIds[0], nodeIds: [...nodeIds], links: _debugLinks, totalNodes: data.nodes.length };

  hoverHighlight = {
    nodeIds,
    linkIds,
    hoverNodeId: seedNodeIds.length === 1 ? seedNodeIds[0] : null,
    hoverLinkId: seedLinkId,
  };
  hoverHighlightKey = nextKey;
  requestGraphRedraw();
}

function cancelScheduledHoverClear() {
  if (hoverClearFrame == null) return;
  cancelAnimationFrame(hoverClearFrame);
  hoverClearFrame = null;
}

function scheduleClearHoverHighlight() {
  cancelScheduledHoverClear();
  hoverClearFrame = requestAnimationFrame(() => {
    hoverClearFrame = null;
    clearHoverHighlight();
    hideTooltip();
  });
}

function clearHoverHighlight() {
  cancelScheduledHoverClear();
  if (!hoverHighlight) return;
  hoverHighlight = null;
  hoverHighlightKey = null;
  requestGraphRedraw();
}

function setHoverEdge(linkId) {
  if (!graph || !linkId) return;
  cancelScheduledHoverClear();
  clearHoverHighlight();
  if (hoverEdgeId === linkId) return;
  hoverEdgeId = linkId;
  requestGraphRedraw();
}

function clearHoverEdge(redraw = true) {
  if (!hoverEdgeId) return;
  hoverEdgeId = null;
  if (redraw) requestGraphRedraw();
}

function hoverAwareLinkColor(link) {
  const key = linkKey(link);
  const actVisible = linkActivationVisible.get(key);
  if (actVisible === false) return 'rgba(0,0,0,0)';
  const mapColor = linkActivationColor.get(key);
  if (mapColor) return mapColor;
  if (isHoverEdge(link)) return COLORS.edgeHebbianPulse;
  if (!isHoverActive()) {
    // Apply type-based opacity to reduce visual noise
    // LOD: at low zoom, further reduce opacity
    const zoom = graph ? graph.zoom() : 1;
    const lodFactor = zoom < 0.5 ? 0.5 : 1.0;
    const opacity = (EDGE_OPACITY[link.type] || 0.5) * lodFactor;
    return withAlpha(link.color, opacity);
  }
  return isHoverLink(link) ? COLORS.edgeHebbianPulse : 'rgba(139,148,158,0.1)';
}

function hoverAwareLinkWidth(link) {
  const key = linkKey(link);
  const actVisible = linkActivationVisible.get(key);
  if (actVisible === false) return 0;
  const mapWidth = linkWidthState.get(key);
  const baseWidth = mapWidth != null ? mapWidth : (link.width || 1);
  if (isHoverEdge(link)) return Math.max(baseWidth * 2.4, 2.8);
  if (!isHoverActive()) return baseWidth;
  return isHoverLink(link) ? Math.max(baseWidth * 2.2, 2.5) : Math.max(baseWidth * 0.85, 0.75);
}

function hoverAwareParticles(link) {
  const key = linkKey(link);
  if (link._visible === false) return 0;
  if (linkActivationVisible.get(key) === false) return 0;
  if (linkVisibilityState.get(key) === false) return 0;
  const mapParticles = linkParticlesState.get(key) || 0;
  if (mapParticles) return particleConfig.enabled ? particleConfig.activeParticles : 0;
  if (isHoverEdge(link)) return Math.max(link.particles || 0, 4);
  if (isHoverLink(link) && hoverHighlight && link._id === hoverHighlight.hoverLinkId) return Math.max(link.particles || 0, 4);
  // Ambient flow: always animate only strong Hebbian edges. Animating all 3k+
  // Hebbian links turns the graph into Times Square and eats frames; a single
  // particle on high-weight links gives a living synapse feel without soup.
  if (isAmbientFlowLink(link)) return particleConfig.ambientParticles;
  return link.particles || 0;
}

function hoverAwareParticleWidth(link) {
  const key = linkKey(link);
  if (linkActivationVisible.get(key) === false) return 0;
  if (linkParticlesState.get(key)) return particleConfig.activeCore;
  if (isHoverEdge(link) || (isHoverLink(link) && hoverHighlight && link._id === hoverHighlight.hoverLinkId)) return 4;
  if (isAmbientFlowLink(link)) return particleConfig.ambientCore;
  return 2;
}

// ============================================================================
// Node canvas rendering — custom shapes, neural bloom, 💤 text
// ============================================================================
function hashNodeId(id) {
  const s = String(id || '');
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function nodeGlowBoost(node, hoverMatch, isDirectHover) {
  const active = nodeActivationColor.has(node.id);
  const synGlow = Math.max(0, Math.min(1, node._synapticGlow || 0));
  let boost = 0.85 + synGlow * 1.1;
  if (node._dormant) boost *= 0.35;
  if (node._shape === 'hexagon') boost *= 1.28;
  if (node._shape === 'diamond') boost *= 1.55;
  if (active) boost *= 2.75;
  if (hoverMatch) boost *= 1.8;
  if (isDirectHover) boost *= 1.45;
  return Math.max(0.15, Math.min(5.0, boost));
}

function drawNodeGlow(node, ctx, globalScale, color, opacity, radius, boost) {
  const x = node.x;
  const y = node.y;
  const t = performance.now() * 0.001;
  const phase = (hashNodeId(node.id) % 628) / 100;
  const pulse = 0.9 + Math.sin(t * 0.72 + phase) * 0.1;
  const scaleDamp = Math.max(0.72, Math.min(1.28, 1 / Math.sqrt(globalScale || 1)));
  const glowR = radius * (2.7 + boost * 0.72) * pulse * scaleDamp;
  const alpha = Math.min(0.55, (0.045 + boost * 0.045) * opacity);

  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  const grad = ctx.createRadialGradient(x, y, 0, x, y, glowR);
  grad.addColorStop(0, withAlpha(color, alpha * 1.45));
  grad.addColorStop(0.33, withAlpha(color, alpha * 0.52));
  grad.addColorStop(0.72, withAlpha(color, alpha * 0.16));
  grad.addColorStop(1, withAlpha(color, 0));
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(x, y, glowR, 0, 2 * Math.PI);
  ctx.fill();
  ctx.restore();
}

function nodeBodyGradient(ctx, x, y, r, color, opacity) {
  const grad = ctx.createRadialGradient(x - r * 0.32, y - r * 0.42, 0, x, y, r * 1.25);
  grad.addColorStop(0, withAlpha('#ffffff', Math.min(0.5, 0.3 * opacity)));
  grad.addColorStop(0.24, withAlpha(color, Math.min(1, 0.96 * opacity)));
  grad.addColorStop(0.7, withAlpha(color, Math.min(0.84, 0.72 * opacity)));
  grad.addColorStop(1, withAlpha(color, Math.min(0.5, 0.42 * opacity)));
  return grad;
}

function pathNodeShape(ctx, shape, x, y, r) {
  ctx.beginPath();
  if (shape === 'diamond') {
    ctx.moveTo(x, y - r);
    ctx.lineTo(x + r, y);
    ctx.lineTo(x, y + r);
    ctx.lineTo(x - r, y);
  } else if (shape === 'triangleDown') {
    ctx.moveTo(x, y + r);
    ctx.lineTo(x + r, y - r * 0.7);
    ctx.lineTo(x - r, y - r * 0.7);
  } else if (shape === 'hexagon') {
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 3) * i;
      const px = x + r * Math.cos(angle);
      const py = y + r * Math.sin(angle);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
  } else {
    ctx.arc(x, y, r, 0, 2 * Math.PI);
  }
  ctx.closePath();
}

function drawNode(node, ctx, globalScale) {
  const val = node.val || 4;
  const actColor = nodeActivationColor.get(node.id);
  const color = actColor || node.color || COLORS.inactive;
  const actOpacity = nodeActivationOpacity.get(node.id);
  const baseOpacity = actOpacity != null ? actOpacity : 1.0;
  const hoverActive = isHoverActive();
  const hoverMatch = isHoverNode(node);
  const isDirectHover = hoverMatch && hoverHighlight && node.id === hoverHighlight.hoverNodeId;
  // Ensure baseOpacity is a valid number — Math.min(undefined, 0.38) === NaN
  // which makes ctx.globalAlpha default to 1.0, so non-query nodes appear
  // full-bright during hover and look "highlighted" when they shouldn't be.
  const safeBase = (typeof baseOpacity === 'number' && !isNaN(baseOpacity)) ? baseOpacity : 1.0;
  const opacity = hoverActive ? (hoverMatch ? Math.max(safeBase, 0.95) : Math.min(safeBase, 0.38)) : safeBase;
  const x = node.x;
  const y = node.y;
  const shape = node._shape || 'circle';
  // Logarithmic radius — hubs stay readable without dominating
  const r = nodeRadius(val);
  const boost = nodeGlowBoost(node, hoverMatch, isDirectHover);

  // Ambient aura first: nodes become condensations of light inside the synaptic web.
  drawNodeGlow(node, ctx, globalScale, color, opacity, r, boost);
  if (actColor || isDirectHover) {
    drawNodeGlow(node, ctx, globalScale, actColor || COLORS.edgeHebbianPulse, opacity, r * 1.18, boost * 1.15);
  }

  // LOD: at low zoom, simplify body to dots but keep a tiny aura so the graph
  // still feels alive instead of turning into dust.
  const lod = globalScale < 0.5;
  if (lod) {
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.globalAlpha = opacity;
    ctx.beginPath();
    ctx.arc(x, y, Math.max(1.6, r * 0.72), 0, 2 * Math.PI);
    ctx.fillStyle = withAlpha(color, node._dormant ? 0.48 : 0.82);
    ctx.fill();
    ctx.restore();
    return;
  }

  ctx.save();
  ctx.globalAlpha = opacity;
  pathNodeShape(ctx, shape, x, y, r);
  ctx.fillStyle = nodeBodyGradient(ctx, x, y, r, color, opacity);
  ctx.fill();

  // Thin inner rim, not a hard cartoon border. Keeps shapes readable without
  // breaking the bloom aesthetic.
  ctx.globalAlpha = Math.min(0.9, opacity * (hoverMatch ? 0.75 : 0.38));
  ctx.strokeStyle = withAlpha('#ffffff', hoverMatch ? 0.55 : 0.22);
  ctx.lineWidth = Math.max(0.7, 1.1 / globalScale);
  ctx.stroke();

  if (hoverMatch) {
    pathNodeShape(ctx, shape, x, y, r + (2.4 / globalScale));
    ctx.strokeStyle = COLORS.edgeHebbianPulse;
    ctx.lineWidth = 2.4 / globalScale;
    ctx.globalAlpha = 0.82;
    ctx.stroke();
  }

  // Draw 💤 text above dormant nodes
  if (shape === 'triangleDown' && globalScale >= 1.5) {
    ctx.font = (10 / globalScale) + 'px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillStyle = '#6e7681';
    ctx.globalAlpha = opacity * 0.85;
    ctx.fillText('\u{1F4A4}', x, y - r - 2);
  }

  // Highlight ring around the directly hovered node only
  if (isDirectHover) {
    ctx.beginPath();
    ctx.arc(x, y, r + (5 / globalScale), 0, 2 * Math.PI);
    ctx.strokeStyle = COLORS.edgeHebbianPulse;
    ctx.lineWidth = 1.5 / globalScale;
    ctx.globalAlpha = 0.9;
    ctx.stroke();
  }
  ctx.restore();

  // Node label (visible when zoomed in)
  if (globalScale >= 1.2 && node.name) {
    ctx.save();
    ctx.font = (10 / globalScale) + 'px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillStyle = '#c9d1d9';
    ctx.globalAlpha = opacity * 0.8;
    const label = node.name.length > 25 ? node.name.substring(0, 22) + '\u2026' : node.name;
    ctx.fillText(label, x, y + r + 2);
    ctx.restore();
  }
}


// ============================================================================
// Activation state — stored in maps, never mutates live force-graph objects
// Defined here (graph-core.js) so they're available to graph-init.js and
// activation.js which load after this file.
// ============================================================================
const nodeActivationOpacity = new Map(); // nodeId → opacity
const nodeActivationColor = new Map();   // nodeId → color string
const linkActivationVisible = new Map(); // linkKey → bool
const linkActivationColor = new Map();   // linkKey → color string
const linkParticlesState = new Map();    // linkKey → number
const linkParticleColorState = new Map();// linkKey → color string
const linkWidthState = new Map();        // linkKey → number
const linkVisibilityState = new Map();  // linkKey → bool (threshold/direct filter)

function clearActivationState() {
  nodeActivationOpacity.clear();
  nodeActivationColor.clear();
  linkActivationVisible.clear();
  linkActivationColor.clear();
  linkParticlesState.clear();
  linkParticleColorState.clear();
  linkWidthState.clear();
}

// ============================================================================
// Utilities
// ============================================================================
function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}
