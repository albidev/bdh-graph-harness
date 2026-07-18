// ============================================================================
// BDH Graph Harness — shared 3D graph state, rendering, tooltips and inspectors
// ============================================================================

const COLORS = {
  inactive: '#6e7681',
  activated: '#f0883e',
  seed: '#58a6ff',
  selected: '#f0f6fc',
  neurogenesis: '#00e5ff',
  dormant: '#718096',
  edgeWikilink: '#8fa8c2',
  edgeHebbianLow: '#7047b7',
  edgeHebbianMid: '#a879ff',
  edgeHebbianHigh: '#e4c5ff',
  edgeHebbianPulse: '#f0d2ff',
  edgeNeurogenesis: '#67f3ff',
  edgePhantom: '#1f6feb',
  edgeCounterpart: '#56d4dd',
  edgeProjectContext: '#2f81f7',
  edgeProjectReference: '#f2cc60',
  sourceVault: '#58a6ff',
  sourceExternal: '#f0883e',
  bg: '#070a0f',
  surface: '#161b22',
};

const EDGE_OPACITY = {
  wikilink: 0.72,
  hebbian: 0.46,
  phantom: 0.46,
  counterpart: 0.64,
  project_context: 0.52,
  project_reference: 0.62,
  neurogenesis: 0.86,
};

const HEBBIAN_MIN_RENDER_WEIGHT = 0.15;
const HEBBIAN_OVERVIEW_WEIGHT = 0.42;
const MAX_DYNAMIC_LABELS = 22;

function nodeRadius(val) {
  return Math.log2((val || 4) + 1) * 1.35 + 1.5;
}

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
  let typeScale = 1;
  if (link.type === 'hebbian') {
    const weight = typeof link.weight === 'number' ? link.weight : 0.3;
    typeScale = 1.35 - Math.min(0.55, weight * 0.8);
  } else if (link.type === 'phantom') {
    typeScale = 1.35;
  } else if (link.type === 'wikilink') {
    typeScale = 1.05;
  } else if (link.type === 'counterpart') {
    typeScale = 1.15;
  }
  const massSpread = 1 + Math.max(0, avgMass - 1) * 0.12;
  return baseDistance * typeScale * massSpread;
}

function withAlpha(hex, alpha) {
  if (!hex || hex[0] !== '#') return hex;
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function weightColor(weight) {
  if (weight < 0.3) return COLORS.edgeHebbianLow;
  if (weight < 0.6) return COLORS.edgeHebbianMid;
  return COLORS.edgeHebbianHigh;
}

function sourceColor(node) {
  return node && node.source_type === 'external' ? COLORS.sourceExternal : COLORS.sourceVault;
}

function isNeurogenesisNode(node) {
  if (!node) return false;
  if (node._shape === 'diamond') return true;
  const tags = normalizeTags(node._tags || node.tags || '');
  return tags.some(tag => tag.toLowerCase() === 'neurogenesis');
}

function nodeMatchesSourceFilter(node) {
  if (sourceFilter === 'all') return true;
  return (node && (node.source_type || 'vault')) === sourceFilter;
}

const TAG_COLORS = [
  '#f97583', '#79c0ff', '#7ee787', '#d2a8ff', '#ffa657', '#ff7b72',
  '#56d4dd', '#e3b341', '#db61a2', '#a5d6ff', '#b392f0', '#85e89d',
];

const STORAGE_KEYS = {
  hebbianThreshold: 'bdh-graph-hebbian-threshold-v3',
  spacing: 'bdh-graph-spacing-v2',
  edgeLength: 'bdh-graph-edge-length-v2',
  sourceFilter: 'bdh-graph-source-filter',
  zoom: 'bdh-graph-camera-scale-v1',
};

function clampNumber(value, min, max, fallback) {
  if (value == null || value === '') return fallback;
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.max(min, Math.min(max, number));
}

function saveControlValue(key, value) {
  try { localStorage.setItem(key, String(value)); }
  catch (error) { /* Storage is optional. */ }
}

function loadControlValue(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value == null ? fallback : value;
  } catch (error) {
    return fallback;
  }
}

// ============================================================================
// Particle configuration — 3D particles are native low-resolution spheres
// ============================================================================
const PARTICLE_PRESETS = {
  subtle: {
    enabled: true,
    ambient: true,
    activeParticles: 9,
    activeWidth: 3.4,
    ambientParticles: 2,
    ambientWidth: 1.6,
    ambientThreshold: 0.28,
    speed: 0.011,
    activeColor: '#f0d2ff',
    ambientColor: '#a371f7',
  },
  loud: {
    enabled: true,
    ambient: true,
    activeParticles: 16,
    activeWidth: 5.2,
    ambientParticles: 3,
    ambientWidth: 2.4,
    ambientThreshold: 0.24,
    speed: 0.016,
    activeColor: '#f0d2ff',
    ambientColor: '#d2a8ff',
  },
  off: { enabled: false, ambient: false },
  ambientOff: { enabled: true, ambient: false },
};

const DEFAULT_PARTICLE_CONFIG = { ...PARTICLE_PRESETS.subtle };
let particleConfig = { ...DEFAULT_PARTICLE_CONFIG };
let queryParticleMode = false;
let queryParticleTimer = null;

function applyParticlePreset(name) {
  const preset = PARTICLE_PRESETS[name];
  if (!preset) return { error: 'Unknown preset', presets: Object.keys(PARTICLE_PRESETS) };
  particleConfig = { ...particleConfig, ...preset };
  requestGraphRedraw();
  return { ...particleConfig, preset: name };
}

function beginQueryParticles() {
  queryParticleMode = true;
  if (queryParticleTimer) clearTimeout(queryParticleTimer);
  particleConfig = { ...particleConfig, ...PARTICLE_PRESETS.loud };
  if (typeof markGraphActive === 'function') markGraphActive(5000);
  requestGraphRedraw();
}

function endQueryParticles(delayMs = 0) {
  if (queryParticleTimer) clearTimeout(queryParticleTimer);
  queryParticleTimer = setTimeout(() => {
    queryParticleMode = false;
    particleConfig = { ...DEFAULT_PARTICLE_CONFIG, ambient: particleConfig.ambient };
    queryParticleTimer = null;
    requestGraphRedraw();
  }, Math.max(0, delayMs));
}

function toggleAmbientMotion(enabled) {
  particleConfig.ambient = Boolean(enabled);
  requestGraphRedraw();
}

window.BDHParticles = {
  get: () => ({ ...particleConfig }),
  set(next = {}) {
    particleConfig = { ...particleConfig, ...next };
    requestGraphRedraw();
    return { ...particleConfig };
  },
  preset: applyParticlePreset,
  reset() {
    particleConfig = { ...DEFAULT_PARTICLE_CONFIG };
    requestGraphRedraw();
    return { ...particleConfig };
  },
};

// ============================================================================
// Global graph and interaction state
// ============================================================================
let graph = null;
let allGraphNodes = [];
let fgNodes = [];
let fgLinks = [];
let hebbianMap = {};
let orphanNodeIds = [];
let tagColorMap = {};
let showTagColors = true;
let directOnly = false;
let hebbianThreshold = 0.15;
let showPhantom = true;
let degreeMap = {};
let neighborMap = {};
let neurogenesisNodes = {};
let nodeDataMap = {};
let edgeInfoMap = {};
let nodeTagColorMap = {};
let totalConcepts = 0;
let edgeLengthMultiplier = 10;
let spacingValue = 50;
let restoredZoom = null;
let showOrphans = true;
let sourceFilter = 'all';
let sourceGraphData = null;
let lastMouseEvent = { clientX: 0, clientY: 0 };
let mouseTrackingInstalled = false;
let hoverHighlight = null;
let hoverHighlightKey = null;
let hoverClearFrame = null;
let hoverEdgeId = null;
let focusHighlight = null;
let selectedNodeId = null;
let selectedLinkId = null;
let focusedNodeId = null;
let currentViewScale = 1;
let nodeWorldScale = 1;
let currentLodLevel = 'overview';
let fitCameraDistance = null;
let focusMode = null;
let graphRenderPaused = false;
let graphLayoutActive = false;
let graphIdleTimer = null;
let webglUnavailable = false;

const activatedNotesById = new Map();

const nodeActivationOpacity = new Map();
const nodeActivationColor = new Map();
const nodeBirthScaleState = new Map();
const linkActivationVisible = new Map();
const linkActivationColor = new Map();
const linkParticlesState = new Map();
const linkParticleColorState = new Map();
const linkWidthState = new Map();
const linkVisibilityState = new Map();

function clearActivationState() {
  nodeActivationOpacity.clear();
  nodeActivationColor.clear();
  nodeBirthScaleState.clear();
  linkActivationVisible.clear();
  linkActivationColor.clear();
  linkParticlesState.clear();
  linkParticleColorState.clear();
  linkWidthState.clear();
  scheduleLabelUpdate();
}

function linkEndpointId(endpoint) {
  return typeof endpoint === 'object' && endpoint ? endpoint.id : endpoint;
}

function linkKey(link) {
  const source = linkEndpointId(link.source);
  const target = linkEndpointId(link.target);
  return source < target
    ? `${source}→${target}:${link.type || 'syn'}`
    : `${target}→${source}:${link.type || 'syn'}`;
}

function isAmbientFlowLink(link) {
  return particleConfig.enabled
    && particleConfig.ambient
    && link.type === 'hebbian'
    && (link.weight || 0) >= particleConfig.ambientThreshold;
}

function isActiveFlowLink(link) {
  return linkParticlesState.has(linkKey(link));
}

function particleSpeed(link) {
  return isActiveFlowLink(link) ? particleConfig.speed * 1.4 : particleConfig.speed;
}

function isHoverActive() {
  return Boolean(hoverHighlight && (hoverHighlight.nodeIds.size || hoverHighlight.linkIds.size));
}

function activeHighlight() {
  if (isHoverActive()) return hoverHighlight;
  return focusHighlight;
}

function isHighlightedNode(node) {
  const highlight = activeHighlight();
  return Boolean(highlight && highlight.nodeIds.has(node.id));
}

function isHighlightedLink(link) {
  const highlight = activeHighlight();
  return Boolean(highlight && highlight.linkIds.has(link._id));
}

function isDirectHighlightedNode(node) {
  const highlight = activeHighlight();
  return Boolean(highlight && highlight.hoverNodeId === node.id);
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

  data.links.forEach(link => {
    if (!effectiveLinkVisibility(link, { ignoreHighlight: true })) return;
    if (link.type === 'phantom') return;
    const source = linkEndpointId(link.source);
    const target = linkEndpointId(link.target);
    if (seedSet.has(source) || seedSet.has(target) || link._id === seedLinkId) {
      linkIds.add(link._id);
      nodeIds.add(source);
      nodeIds.add(target);
    }
  });

  hoverHighlight = {
    nodeIds,
    linkIds,
    hoverNodeId: seedNodeIds.length === 1 ? seedNodeIds[0] : null,
    hoverLinkId: seedLinkId,
  };
  hoverHighlightKey = nextKey;
  scheduleLabelUpdate();
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
  scheduleLabelUpdate();
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

function setPathHighlight(pathIds = []) {
  if (!graph || !pathIds.length) return;
  const pathSet = new Set(pathIds);
  const pathPairs = new Set();
  for (let index = 0; index < pathIds.length - 1; index += 1) {
    const first = pathIds[index];
    const second = pathIds[index + 1];
    pathPairs.add(first < second ? first + '→' + second : second + '→' + first);
  }
  const linkIds = new Set();
  graph.graphData().links.forEach(link => {
    const source = linkEndpointId(link.source);
    const target = linkEndpointId(link.target);
    const pair = source < target ? source + '→' + target : target + '→' + source;
    if (pathPairs.has(pair)) linkIds.add(link._id);
  });
  focusHighlight = { nodeIds: pathSet, linkIds, hoverNodeId: pathIds[0], hoverLinkId: null };
  scheduleLabelUpdate();
  requestGraphRedraw();
}

function setNeighborhoodFocus(nodeId) {
  if (!graph || !nodeId) return;
  const nodeIds = new Set([nodeId]);
  const linkIds = new Set();
  graph.graphData().links.forEach(link => {
    if (!effectiveLinkVisibility(link, { ignoreHighlight: true })) return;
    const source = linkEndpointId(link.source);
    const target = linkEndpointId(link.target);
    if (source === nodeId || target === nodeId) {
      nodeIds.add(source);
      nodeIds.add(target);
      linkIds.add(link._id);
    }
  });
  focusHighlight = { nodeIds, linkIds, hoverNodeId: nodeId, hoverLinkId: null };
  scheduleLabelUpdate();
  requestGraphRedraw();
}

// ============================================================================
// Link LOD and visual state
// ============================================================================
function isCoreNodeVisible(node) {
  if (!node || node._hidden) return false;
  if (currentLodLevel !== 'overview') return true;
  if (selectedNodeId === node.id || focusedNodeId === node.id || activatedNotesById.has(node.id)) return true;
  if (isNeurogenesisNode(node)) return true;
  return (degreeMap[node.id] || 0) >= 2;
}

function effectiveLinkVisibility(link, options = {}) {
  if (!link || link._visible === false) return false;
  if (linkActivationVisible.get(linkKey(link)) === false) return false;
  if (linkVisibilityState.get(linkKey(link)) === false) return false;

  const highlighted = !options.ignoreHighlight && isHighlightedLink(link);
  if (highlighted) return true;
  if (currentLodLevel === 'overview') {
    const source = nodeDataMap[linkEndpointId(link.source)] || {};
    const target = nodeDataMap[linkEndpointId(link.target)] || {};
    if (!isCoreNodeVisible(source) || !isCoreNodeVisible(target)) return false;
  }
  if (currentLodLevel === 'overview' && link.type === 'hebbian') {
    return (link.weight || 0) >= Math.max(hebbianThreshold, HEBBIAN_OVERVIEW_WEIGHT);
  }
  return true;
}

function linkDisplayColor(link) {
  const key = linkKey(link);
  if (linkActivationColor.has(key)) return linkActivationColor.get(key);
  if (hoverEdgeId === link._id || isHighlightedLink(link)) return COLORS.edgeHebbianPulse;
  return link.color || COLORS.edgeWikilink;
}

function linkDisplayOpacity(link) {
  if (!effectiveLinkVisibility(link)) return 0;
  if (hoverEdgeId === link._id || isHighlightedLink(link)) return 0.92;
  const highlight = activeHighlight();
  if (highlight) return 0.055;
  let opacity = EDGE_OPACITY[link.type] || 0.28;
  if (currentLodLevel === 'overview') opacity *= link.type === 'wikilink' ? 0.78 : 0.82;
  if (link.type === 'hebbian') opacity *= 0.7 + Math.min(0.8, link.weight || 0);
  return Math.max(0.025, Math.min(0.9, opacity));
}

const EDGE_WIDTH_SCALE = 2;
const HIGHLIGHT_WIDTH_SCALE = 4;

function linkDisplayWidth(link) {
  const key = linkKey(link);
  const stateWidth = linkWidthState.get(key);
  let width;
  if (stateWidth != null) width = Math.min(3.2, Math.max(0, stateWidth * 0.42));
  else if (hoverEdgeId === link._id || isHighlightedLink(link)) width = Math.max(1.6, Math.min(4.4, (link.width || 1) * 1.1));
  else if (currentLodLevel === 'overview') {
    if (link.type === 'wikilink') width = 2.00;
    else if (link.type === 'phantom') width = 1.65;
    else if (link.type === 'hebbian') width = 2.45;
    else if (link.type === 'project_context') width = 2.20;
    else if (link.type === 'counterpart' || link.type === 'project_reference' || link.type === 'neurogenesis') width = 3.10;
    else width = 1.80;
  } else if (link.type === 'wikilink') width = 2.35;
  else if (link.type === 'phantom') width = 1.95;
  else if (link.type === 'hebbian') width = (link.weight || 0) >= 0.7 ? 3.20 : 2.40;
  else if (link.type === 'project_context') width = 2.75;
  else if (link.type === 'counterpart' || link.type === 'project_reference' || link.type === 'neurogenesis') width = 3.80;
  else width = 2.10;
  const scale = (hoverEdgeId === link._id || isHighlightedLink(link)) ? HIGHLIGHT_WIDTH_SCALE : EDGE_WIDTH_SCALE;
  return width * scale;
}

function stableLinkHash(link) {
  const value = String(link && (link._id || link.id || link.source + ':' + link.target) || 'synapse');
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function organicLinkCurvature(link) {
  if (link && link._dashes) return 0;
  const hash = stableLinkHash(link);
  return 0.14 + ((hash % 100) / 100) * 0.18;
}

function organicLinkRotation(link) {
  if (link && link._dashes) return 0;
  return (stableLinkHash(link) % 628) / 100;
}

function hoverAwareParticles(link) {
  if (!effectiveLinkVisibility(link)) return 0;
  const activeCount = linkParticlesState.get(linkKey(link));
  if (activeCount) return particleConfig.enabled ? particleConfig.activeParticles : 0;
  if (hoverEdgeId === link._id || isHighlightedLink(link)) return particleConfig.enabled ? 2 : 0;
  if (isAmbientFlowLink(link)) return particleConfig.ambientParticles;
  return 0;
}

function hoverAwareParticleWidth(link) {
  if (!effectiveLinkVisibility(link)) return 0;
  if (linkParticlesState.has(linkKey(link))) return particleConfig.activeWidth;
  if (hoverEdgeId === link._id || isHighlightedLink(link)) return 1.8;
  if (isAmbientFlowLink(link)) return particleConfig.ambientWidth;
  return 0.5;
}

function particleColor(link) {
  return linkParticleColorState.get(linkKey(link))
    || (isActiveFlowLink(link) ? particleConfig.activeColor : particleConfig.ambientColor)
    || linkDisplayColor(link);
}

function directionalParticleObject(link) {
  const T = window.THREE;
  if (!T) return null;
  const color = particleColor(link);
  const particle = new T.Mesh(
    new T.SphereGeometry(1.9, 10, 8),
    new T.MeshBasicMaterial({ color, transparent: true, opacity: 0.98 }),
  );
  const halo = new T.Sprite(ringMaterial(color, 0.58));
  halo.scale.setScalar(8.5);
  halo.renderOrder = 4;
  particle.add(halo);
  particle.renderOrder = 4;
  return particle;
}

// ============================================================================
// Reused Three.js resources and selective labels
// ============================================================================
const threeResources = {
  initialized: false,
  geometries: {},
  nodeMaterials: new Map(),
  linkMaterials: new Map(),
  ringMaterials: new Map(),
  ringTexture: null,
};

let bloomPass = null;
let smaaPass = null;
let bloomInstallPromise = null;
let neuralField = null;

function ensureNeuralField() {
  if (neuralField || !graph || !window.THREE) return;
  const T = window.THREE;
  const group = new T.Group();
  const count = isConstrainedDevice() ? 150 : 300;
  const makeLayer = (color, offset) => {
    const positions = new Float32Array(count * 3);
    for (let index = 0; index < count; index += 1) {
      const radius = 260 * Math.cbrt(Math.random());
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const slot = index * 3;
      positions[slot] = Math.sin(phi) * Math.cos(theta) * radius + offset;
      positions[slot + 1] = Math.sin(phi) * Math.sin(theta) * radius + offset * 0.35;
      positions[slot + 2] = Math.cos(phi) * radius - offset;
    }
    const geometry = new T.BufferGeometry();
    geometry.setAttribute('position', new T.BufferAttribute(positions, 3));
    const material = new T.PointsMaterial({
      color,
      size: 1.7,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.24,
      depthWrite: false,
      blending: T.AdditiveBlending,
    });
    return new T.Points(geometry, material);
  };
  group.add(makeLayer('#35d9ff', -26));
  group.add(makeLayer('#a879ff', 26));
  group.renderOrder = -1;
  graph.scene().add(group);
  neuralField = group;
}

function installBloomPass() {
  if (!graph || bloomPass || !window.THREE || typeof window.UnrealBloomPass !== 'function') return false;
  const composer = typeof graph.postProcessingComposer === 'function'
    ? graph.postProcessingComposer()
    : null;
  if (!composer) return false;
  const container = document.getElementById('graph-container');
  const width = Math.max(1, container ? container.clientWidth : window.innerWidth);
  const height = Math.max(1, container ? container.clientHeight : window.innerHeight);
  bloomPass = new window.UnrealBloomPass(
    new window.THREE.Vector2(width, height),
    0.95,
    0.68,
    0.52,
  );
  bloomPass.threshold = 0.62;
  bloomPass.strength = 1.15;
  bloomPass.radius = 0.82;
  composer.addPass(bloomPass);
  if (typeof window.SMAAPass === 'function') {
    smaaPass = new window.SMAAPass(width, height);
    composer.addPass(smaaPass);
  }
  return true;
}

function scheduleBloomInstall() {
  if (bloomPass || bloomInstallPromise || !window.BDHBloomReady) return;
  bloomInstallPromise = Promise.resolve(window.BDHBloomReady)
    .then(() => {
      bloomInstallPromise = null;
      if (installBloomPass()) requestGraphRedraw();
    })
    .catch(error => {
      bloomInstallPromise = null;
      console.warn('[BDH 3D] Bloom unavailable; continuing without post-processing:', error);
    });
}

function ensureThreeResources() {
  if (threeResources.initialized) return threeResources;
  if (!window.THREE) throw new Error('Three.js module is not ready');
  const T = window.THREE;
  threeResources.geometries = {
    sphere: new T.SphereGeometry(1, 20, 14),
    hub: new T.IcosahedronGeometry(1, 2),
    diamond: new T.OctahedronGeometry(1, 0),
    dormant: new T.ConeGeometry(1, 1.5, 4, 1),
  };
  threeResources.geometries.dormant.rotateX(Math.PI);
  threeResources.ringTexture = createRingTexture();
  threeResources.initialized = true;
  return threeResources;
}

function createRingTexture() {
  const T = window.THREE;
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const context = canvas.getContext('2d');
  const gradient = context.createRadialGradient(64, 64, 30, 64, 64, 62);
  gradient.addColorStop(0, 'rgba(255,255,255,0)');
  gradient.addColorStop(0.58, 'rgba(255,255,255,0)');
  gradient.addColorStop(0.72, 'rgba(255,255,255,0.95)');
  gradient.addColorStop(0.82, 'rgba(255,255,255,0.22)');
  gradient.addColorStop(1, 'rgba(255,255,255,0)');
  context.fillStyle = gradient;
  context.fillRect(0, 0, 128, 128);
  const texture = new T.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

function nodeMaterial(color, opacity, emphasis, dormant = false) {
  const resources = ensureThreeResources();
  const opacityBucket = Math.round(Math.max(0.05, Math.min(1, opacity)) * 20) / 20;
  const emphasisBucket = Math.round(Math.max(0, Math.min(1, emphasis)) * 4) / 4;
  const key = `${color}|${opacityBucket}|${emphasisBucket}|${dormant ? 'dormant' : 'active'}`;
  if (!resources.nodeMaterials.has(key)) {
    const baseEmissive = dormant ? 0.24 : 0.12;
    const material = new window.THREE.MeshLambertMaterial({
      color,
      emissive: color,
      emissiveIntensity: baseEmissive + emphasisBucket * 0.78,
      transparent: opacityBucket < 1,
      opacity: opacityBucket,
      depthWrite: opacityBucket >= 0.55,
    });
    resources.nodeMaterials.set(key, material);
  }
  return resources.nodeMaterials.get(key);
}

function ringMaterial(color, opacity) {
  const resources = ensureThreeResources();
  const opacityBucket = Math.round(Math.max(0.05, Math.min(1, opacity)) * 20) / 20;
  const key = `${color}|${opacityBucket}`;
  if (!resources.ringMaterials.has(key)) {
    resources.ringMaterials.set(key, new window.THREE.SpriteMaterial({
      map: resources.ringTexture,
      color,
      transparent: true,
      opacity: opacityBucket,
      depthWrite: false,
      depthTest: true,
      blending: window.THREE.AdditiveBlending,
    }));
  }
  return resources.ringMaterials.get(key);
}

function geometryForNode(node) {
  const geometries = ensureThreeResources().geometries;
  if (node._shape === 'diamond') return geometries.diamond;
  if (node._shape === 'hexagon') return geometries.hub;
  if (node._shape === 'triangleDown') return geometries.dormant;
  return geometries.sphere;
}

function createNodeThreeObject(node) {
  const T = window.THREE;
  ensureThreeResources();
  const group = new T.Group();
  const body = new T.Mesh(geometryForNode(node), nodeMaterial(node.color || COLORS.inactive, 1, 0, node._dormant));
  const aura = new T.Sprite(ringMaterial(node.color || COLORS.inactive, 0.2));
  aura.visible = false;
  aura.renderOrder = 2;
  group.add(body);
  group.add(aura);
  group.userData.nodeId = node.id;
  node._threeObject = group;
  node._bodyObject = body;
  node._auraObject = aura;
  updateNodeThreeObject(node);
  return group;
}

function updateNodeThreeObject(node) {
  const group = node && node._threeObject;
  if (!group || !node._bodyObject || !node._auraObject) return;
  const body = node._bodyObject;
  const aura = node._auraObject;
  const activationColor = nodeActivationColor.get(node.id);
  const neurogenesis = isNeurogenesisNode(node);
  const baseColor = activationColor || (neurogenesis ? COLORS.neurogenesis : (node.color || COLORS.inactive));
  const activationOpacity = nodeActivationOpacity.get(node.id);
  let opacity = activationOpacity != null ? activationOpacity : (node._opacity != null ? node._opacity : 1);
  const highlighted = isHighlightedNode(node);
  const direct = isDirectHighlightedNode(node);
  const highlight = activeHighlight();
  if (highlight) opacity = highlighted ? Math.max(opacity, 0.96) : Math.min(opacity, 0.22);
  if (node._dormant) opacity *= 0.82;

  const selected = selectedNodeId === node.id;
  const focused = focusedNodeId === node.id;
  const synapticGlow = Number(node._synapticGlow || 0);
  const emphasis = activationColor ? 1 : Math.max(
    synapticGlow * 0.72,
    selected || focused || direct ? 0.85 : (highlighted ? 0.45 : 0),
  );
  body.geometry = geometryForNode(node);
  body.material = nodeMaterial(baseColor, opacity, emphasis, node._dormant);

  const birthScale = nodeBirthScaleState.get(node.id) || 1;
  const radius = nodeRadius(node.val || 4) * nodeWorldScale * birthScale * (selected || focused ? 1.08 : 1);
  body.scale.setScalar(radius);

  let ringColor = baseColor;
  let ringOpacity = 0;
  if (activationColor) ringOpacity = 0.92;
  else if (selected || focused || direct) {
    ringColor = selected ? COLORS.selected : baseColor;
    ringOpacity = 0.78;
  } else if (neurogenesis) {
    ringColor = COLORS.neurogenesis;
    ringOpacity = 0.46;
  } else if (synapticGlow >= 0.58) {
    ringColor = baseColor;
    ringOpacity = 0.16 + synapticGlow * 0.16;
  } else if (highlighted) {
    ringOpacity = 0.42;
  }
  aura.visible = ringOpacity > 0;
  if (aura.visible) {
    aura.material = ringMaterial(ringColor, ringOpacity * opacity);
    aura.scale.setScalar(radius * (activationColor ? 3.25 : 2.65));
  }
  group.visible = !node._hidden;
}

function roundedRect(context, x, y, width, height, radius) {
  const r = Math.min(radius, height / 2, width / 2);
  context.beginPath();
  context.moveTo(x + r, y);
  context.arcTo(x + width, y, x + width, y + height, r);
  context.arcTo(x + width, y + height, x, y + height, r);
  context.arcTo(x, y + height, x, y, r);
  context.arcTo(x, y, x + width, y, r);
  context.closePath();
}

function ensureNodeLabelSprite(node) {
  if (node._labelObject) return node._labelObject;
  if (!node._threeObject || !window.THREE) return null;
  const label = node.name || node._title || node.id;
  const display = label.length > 34 ? label.slice(0, 31) + '…' : label;
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 96;
  const context = canvas.getContext('2d');
  context.font = '600 28px system-ui, -apple-system, sans-serif';
  const measured = Math.min(460, Math.ceil(context.measureText(display).width + 42));
  const x = (512 - measured) / 2;
  roundedRect(context, x, 14, measured, 68, 18);
  context.fillStyle = 'rgba(13,17,23,0.88)';
  context.fill();
  context.strokeStyle = 'rgba(139,148,158,0.38)';
  context.lineWidth = 2;
  context.stroke();
  context.fillStyle = '#f0f6fc';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(display, 256, 49, 450);

  const texture = new window.THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  const material = new window.THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
    depthTest: false,
  });
  const sprite = new window.THREE.Sprite(material);
  const width = Math.max(34, measured * 0.11);
  sprite.scale.set(width, 10.5, 1);
  sprite.position.set(0, nodeRadius(node.val || 4) * nodeWorldScale + 9, 0);
  sprite.renderOrder = 4;
  sprite.visible = false;
  node._threeObject.add(sprite);
  node._labelObject = sprite;
  node._labelTexture = texture;
  return sprite;
}

function disposeNodeVisual(node) {
  if (!node) return;
  if (node._labelTexture) node._labelTexture.dispose();
  if (node._labelObject && node._labelObject.material) node._labelObject.material.dispose();
  node._labelObject = null;
  node._labelTexture = null;
  node._threeObject = null;
  node._bodyObject = null;
  node._auraObject = null;
}

let labelUpdateTimer = null;
function scheduleLabelUpdate() {
  if (labelUpdateTimer) clearTimeout(labelUpdateTimer);
  labelUpdateTimer = setTimeout(() => {
    labelUpdateTimer = null;
    updateVisibleLabels();
  }, 70);
}

function updateVisibleLabels() {
  if (!graph) return;
  const data = graph.graphData();
  const wanted = new Set();
  if (selectedNodeId) wanted.add(selectedNodeId);
  if (focusedNodeId) wanted.add(focusedNodeId);
  const highlight = activeHighlight();
  if (highlight) highlight.nodeIds.forEach(id => wanted.add(id));
  nodeActivationColor.forEach((value, id) => wanted.add(id));

  if (currentViewScale >= 1.25 && graph.controls) {
    const target = graph.controls().target || { x: 0, y: 0, z: 0 };
    data.nodes
      .filter(node => Number.isFinite(node.x) && Number.isFinite(node.y) && Number.isFinite(node.z))
      .sort((first, second) => {
        const firstDistance = (first.x - target.x) ** 2 + (first.y - target.y) ** 2 + (first.z - target.z) ** 2;
        const secondDistance = (second.x - target.x) ** 2 + (second.y - target.y) ** 2 + (second.z - target.z) ** 2;
        return firstDistance - secondDistance || (degreeMap[second.id] || 0) - (degreeMap[first.id] || 0);
      })
      .slice(0, 14)
      .forEach(node => wanted.add(node.id));
  }

  const capped = new Set([...wanted].slice(0, MAX_DYNAMIC_LABELS));
  data.nodes.forEach(node => {
    if (capped.has(node.id)) {
      const label = ensureNodeLabelSprite(node);
      if (label) label.visible = true;
    } else if (node._labelObject) {
      node._labelObject.visible = false;
    }
  });
}

const perLinkMaterials = new Map();

function linkMaterial(link) {
  const id = link._id || linkKey(link);
  if (!perLinkMaterials.has(id)) {
    const width = linkDisplayWidth(link);
    const kind = width > 0 ? 'mesh' : 'line';
    const material = kind === 'mesh'
      ? new window.THREE.MeshBasicMaterial({
          color: linkDisplayColor(link),
          transparent: true,
          opacity: linkDisplayOpacity(link),
          depthWrite: false,
          blending: window.THREE.NormalBlending,
        })
      : new window.THREE.LineBasicMaterial({
          color: linkDisplayColor(link),
          transparent: true,
          opacity: linkDisplayOpacity(link),
          depthWrite: false,
          blending: window.THREE.NormalBlending,
        });
    perLinkMaterials.set(id, material);
  }
  const mat = perLinkMaterials.get(id);
  mat.color.set(linkDisplayColor(link));
  mat.opacity = linkDisplayOpacity(link);
  return mat;
}

function createDashedLinkObject(link) {
  if (!link._dashes) return null;
  const highlighted = hoverEdgeId === link._id || isHighlightedLink(link);
  const radius = highlighted ? 5.4 : Math.max(1.24, Math.min(2.4, linkDisplayWidth(link) * 0.38));
  const geometry = new window.THREE.CylinderGeometry(1, 1, 1, 12, 1, false);
  const material = new window.THREE.MeshBasicMaterial({
    color: linkDisplayColor(link),
    transparent: true,
    opacity: linkDisplayOpacity(link),
    depthWrite: false,
    blending: window.THREE.NormalBlending,
  });
  const filament = new window.THREE.Mesh(geometry, material);
  filament.scale.set(radius, 1, radius);
  filament.frustumCulled = false;
  link._threeLinkObject = filament;
  return filament;
}

function updateDashedLinkPosition(object, coordinates, link) {
  if (!link || !link._dashes || !object || !coordinates) return false;
  const start = new window.THREE.Vector3(coordinates.start.x, coordinates.start.y, coordinates.start.z);
  const end = new window.THREE.Vector3(coordinates.end.x, coordinates.end.y, coordinates.end.z);
  const direction = end.clone().sub(start);
  const length = Math.max(0.001, direction.length());
  const highlighted = hoverEdgeId === link._id || isHighlightedLink(link);
  const radius = highlighted ? 5.4 : Math.max(1.24, Math.min(2.4, linkDisplayWidth(link) * 0.38));
  object.position.copy(start.clone().add(end).multiplyScalar(0.5));
  object.scale.set(radius, length, radius);
  object.quaternion.setFromUnitVectors(new window.THREE.Vector3(0, 1, 0), direction.normalize());
  return true;
}

function syncDashedLinkVisual(link) {
  const object = link && link._threeLinkObject;
  if (!object) return;
  object.visible = effectiveLinkVisibility(link);
  object.material.color.set(linkDisplayColor(link));
  object.material.opacity = linkDisplayOpacity(link);
  const highlighted = hoverEdgeId === link._id || isHighlightedLink(link);
  const radius = highlighted ? 5.4 : Math.max(1.24, Math.min(2.4, linkDisplayWidth(link) * 0.38));
  object.scale.x = radius;
  object.scale.z = radius;
  object.material.needsUpdate = true;
}

function syncRegularLinkVisual(link) {
  if (!link || link._dashes || !link.__lineObj) return;
  const object = link.__lineObj;
  const material = linkMaterial(link);
  if (object.material !== material) object.material = material;
  object.visible = effectiveLinkVisibility(link);
  object.material.color.set(linkDisplayColor(link));
  object.material.opacity = linkDisplayOpacity(link);
  object.material.needsUpdate = true;
  if (object.geometry && typeof object.geometry.dispose === 'function') {
    const currentWidth = linkDisplayWidth(link);
    if (object.geometry.parameters && object.geometry.parameters.radius !== currentWidth) {
      object.geometry.dispose();
      object.geometry = new window.THREE.CylinderGeometry(currentWidth, currentWidth, 1, 12, 1, false);
    }
  }
}

function syncThreeVisualState() {
  if (!graph || !window.THREE) return;
  const data = graph.graphData();
  graph
    .nodeVisibility(isCoreNodeVisible)
    .linkVisibility(effectiveLinkVisibility)
    .linkMaterial(linkMaterial)
    .linkWidth(linkDisplayWidth)
    .linkDirectionalParticles(hoverAwareParticles)
    .linkDirectionalParticleSpeed(particleSpeed)
    .linkDirectionalParticleWidth(hoverAwareParticleWidth)
    .linkDirectionalParticleColor(particleColor);
  graph.refresh();
  // Apply per-link visual state AFTER refresh so it survives object recreation.
  data.nodes.forEach(updateNodeThreeObject);
  data.links.forEach(syncDashedLinkVisual);
  data.links.forEach(syncRegularLinkVisual);
  scheduleLabelUpdate();
}

// ============================================================================
// Tooltips
// ============================================================================
let tooltipEl = null;

function ensureTooltip() {
  if (tooltipEl) return;
  tooltipEl = document.createElement('div');
  tooltipEl.id = 'custom-tooltip';
  tooltipEl.className = 'graph-tooltip';
  tooltipEl.hidden = true;
  document.body.appendChild(tooltipEl);
}

function isMobile() {
  return window.matchMedia('(max-width: 768px)').matches;
}

function positionTooltip(event) {
  if (!tooltipEl || tooltipEl.hidden) return;
  if (isMobile()) {
    tooltipEl.style.left = '12px';
    tooltipEl.style.right = '12px';
    tooltipEl.style.bottom = 'calc(env(safe-area-inset-bottom) + 12px)';
    tooltipEl.style.top = 'auto';
    tooltipEl.style.width = 'auto';
    return;
  }
  const source = event || lastMouseEvent || { clientX: 0, clientY: 0 };
  const pad = 16;
  let x = (source.clientX || source.pageX || 0) + pad;
  let y = (source.clientY || source.pageY || 0) + pad;
  tooltipEl.style.right = '';
  tooltipEl.style.bottom = '';
  tooltipEl.style.width = '';
  const rect = tooltipEl.getBoundingClientRect();
  if (x + rect.width > window.innerWidth) x -= rect.width + pad * 2;
  if (y + rect.height > window.innerHeight) y -= rect.height + pad * 2;
  tooltipEl.style.left = Math.max(8, x) + 'px';
  tooltipEl.style.top = Math.max(8, y) + 'px';
}

function showTooltip(node, event) {
  if (graphLayoutActive) return;
  ensureTooltip();
  const data = node ? nodeDataMap[node.id] : null;
  if (!data) { hideTooltip(); return; }
  const mobile = isMobile();
  const title = escapeHtml(data.display_label || data.title || data.id);
  const path = data.absolute_path || data._path || data.path || '';
  const shortPath = data.relative_path || data.path || '';
  const sourceType = data.source_type || 'vault';
  const sourceId = data.source_id || sourceType;
  const rawTags = data._tags || data.tags || '';
  const tags = normalizeTags(rawTags);
  const tagHtml = tags.map(tag => {
    const color = tagColorMap[tag] || COLORS.inactive;
    return `<span class="tooltip-tag"><i style="background:${color}"></i>${escapeHtml(tag)}</span>`;
  }).join('');
  const preview = (data._text || data.text || '').replace(/\s+/g, ' ').trim().slice(0, mobile ? 90 : 180);
  const degree = degreeMap[data.id] || 0;
  const neighbors = neighborMap[data.id] || [];
  const neighborLimit = mobile ? 2 : 6;
  const neighborsHtml = neighbors.slice(0, neighborLimit).map(value => `<span>${escapeHtml(value)}</span>`).join(', ');
  const openUrl = sourceType === 'vault'
    ? 'obsidian://open?path=' + encodeURIComponent(path)
    : 'file://' + path;

  let html = mobile ? '<button class="mobile-sheet-close" onclick="dismissMobileSheet()" aria-label="Close node details">×</button>' : '';
  html += `<div class="tooltip-kicker">${escapeHtml(sourceType)} · ${escapeHtml(sourceId)}</div>`;
  html += `<strong class="tooltip-title">${title}</strong>`;
  if (tagHtml) html += `<div class="tooltip-tags">${tagHtml}</div>`;
  if (shortPath) html += `<div class="tooltip-path">${escapeHtml(shortPath)}</div>`;
  html += `<div class="tooltip-metric">${degree} connection${degree === 1 ? '' : 's'}</div>`;
  if (neighborsHtml) html += `<div class="tooltip-neighbors">${neighborsHtml}${neighbors.length > neighborLimit ? ` <em>+${neighbors.length - neighborLimit}</em>` : ''}</div>`;
  if (preview) html += `<p>${escapeHtml(preview)}${preview.length >= (mobile ? 90 : 180) ? '…' : ''}</p>`;
  if (path) html += `<a href="${escapeHtml(openUrl)}" target="_blank" rel="noopener">Open file ↗</a>`;

  tooltipEl.innerHTML = html;
  tooltipEl.classList.toggle('mobile-tooltip', mobile);
  tooltipEl.hidden = false;
  tooltipEl.style.display = 'block';
  positionTooltip(event);
}

function edgeTypeLabel(type) {
  return {
    hebbian: 'Hebbian synapse',
    phantom: 'Phantom link',
    counterpart: 'Project counterpart',
    project_context: 'Project context',
    project_reference: 'Project reference',
    neurogenesis: 'Neurogenesis link',
    wikilink: 'Wikilink',
  }[type] || type || 'Wikilink';
}

function showEdgeTooltip(link, event) {
  if (graphLayoutActive) return;
  ensureTooltip();
  const info = link ? edgeInfoMap[link._id] : null;
  if (!info) { hideTooltip(); return; }
  let html = `<div class="tooltip-kicker">${escapeHtml(edgeTypeLabel(info.type))}</div>`;
  html += `<strong class="tooltip-title">${escapeHtml(info.source_title)} <span>→</span> ${escapeHtml(info.target_title)}</strong>`;
  if (info.type === 'hebbian') {
    html += `<div class="tooltip-metrics"><span>Weight <b>${Number(info.weight || 0).toFixed(3)}</b></span><span>Frequency <b>${Number(info.frequency || 0)}</b></span></div>`;
  } else if (info.type === 'phantom') {
    html += `<div class="tooltip-metric">Similarity <b>${Number(info.similarity || 0).toFixed(3)}</b></div>`;
  }
  if (info.relation) html += `<div class="tooltip-metric">Relation <b>${escapeHtml(info.relation)}</b></div>`;
  if (info.group_id) html += `<div class="tooltip-metric">Project <b>${escapeHtml(info.group_id)}</b></div>`;
  tooltipEl.innerHTML = html;
  tooltipEl.classList.remove('mobile-tooltip');
  tooltipEl.hidden = false;
  tooltipEl.style.display = 'block';
  positionTooltip(event);
}

function showActivatedTooltip(note, event) {
  if (graphLayoutActive) return;
  ensureTooltip();
  if (!note) return;
  const role = note.role === 'seed' ? 'Seed' : 'Graph neighbor';
  let html = `<div class="tooltip-kicker">${role}</div><strong class="tooltip-title">${escapeHtml(note.title || note.id)}</strong>`;
  html += '<div class="score-grid">';
  html += `<span>Final score</span><b>${Number(note.final_score ?? note.score ?? 0).toFixed(4)}</b>`;
  html += `<span>Hybrid</span><b>${Number(note.hybrid_score || 0).toFixed(4)}</b>`;
  html += `<span>Vector</span><b>${Number(note.vector_score || 0).toFixed(4)}</b>`;
  html += `<span>BM25</span><b>${Number(note.bm25_score || 0).toFixed(4)}</b>`;
  html += `<span>Hebbian boost</span><b>${Number(note.hebbian_boost || 0).toFixed(4)}</b>`;
  html += `<span>Hop</span><b>${note.hop ?? 0}</b>`;
  html += '</div>';
  if (note.parent_id) html += `<div class="tooltip-path">From ${escapeHtml(note.parent_id)}</div>`;
  tooltipEl.innerHTML = html;
  tooltipEl.hidden = false;
  tooltipEl.style.display = 'block';
  positionTooltip(event);
}

function hideTooltip() {
  if (!tooltipEl) return;
  tooltipEl.hidden = true;
  tooltipEl.style.display = 'none';
}

function dismissMobileSheet() {
  hideTooltip();
}

// ============================================================================
// Selection inspector
// ============================================================================
function normalizeTags(rawTags) {
  if (Array.isArray(rawTags)) {
    return rawTags.map(tag => String(tag).replace(/^[\[\]]+|[\[\]]+$/g, '').trim()).filter(Boolean);
  }
  if (typeof rawTags === 'string' && rawTags.trim()) {
    return rawTags.replace(/^[\[\]]+|[\[\]]+$/g, '').split(',').map(tag => tag.trim()).filter(Boolean);
  }
  return [];
}

function renderNodeInspector(node) {
  const content = document.getElementById('selection-content');
  const clearButton = document.getElementById('clear-selection-btn');
  if (!content || !node) return;
  const data = nodeDataMap[node.id] || {};
  const path = data.absolute_path || data.path || '';
  const tags = normalizeTags(data.tags || node._tags);
  content.className = 'selection-detail';
  content.innerHTML = `
    <div class="selection-type">Node · ${escapeHtml(data.source_type || 'vault')}</div>
    <h3>${escapeHtml(data.display_label || data.title || node.name || node.id)}</h3>
    <div class="selection-metrics">
      <span><small>Degree</small><b>${degreeMap[node.id] || 0}</b></span>
      <span><small>Shape</small><b>${escapeHtml(node._shape || 'sphere')}</b></span>
      <span><small>Source</small><b>${escapeHtml(data.source_id || 'vault')}</b></span>
    </div>
    ${tags.length ? `<div class="selection-tags">${tags.map(tag => `<span>${escapeHtml(tag)}</span>`).join('')}</div>` : ''}
    ${data.relative_path || data.path ? `<p class="selection-path">${escapeHtml(data.relative_path || data.path)}</p>` : ''}
    <div class="selection-actions"><button type="button" id="selection-focus-action">Focus neighborhood</button>${path ? `<a href="${escapeHtml((data.source_type || 'vault') === 'vault' ? 'obsidian://open?path=' + encodeURIComponent(path) : 'file://' + path)}" target="_blank" rel="noopener">Open file</a>` : ''}</div>
  `;
  const focusButton = document.getElementById('selection-focus-action');
  if (focusButton) focusButton.addEventListener('click', () => focusGraphNode(node, { kind: 'node' }));
  if (clearButton) clearButton.hidden = false;
}

function renderEdgeInspector(link) {
  const content = document.getElementById('selection-content');
  const clearButton = document.getElementById('clear-selection-btn');
  const info = link ? edgeInfoMap[link._id] : null;
  if (!content || !info) return;
  content.className = 'selection-detail';
  content.innerHTML = `
    <div class="selection-type">${escapeHtml(edgeTypeLabel(info.type))}</div>
    <h3>${escapeHtml(info.source_title)} <span class="arrow">→</span> ${escapeHtml(info.target_title)}</h3>
    <div class="selection-metrics">
      ${info.weight != null ? `<span><small>Weight</small><b>${Number(info.weight).toFixed(3)}</b></span>` : ''}
      ${info.frequency != null ? `<span><small>Frequency</small><b>${Number(info.frequency)}</b></span>` : ''}
      ${info.similarity != null ? `<span><small>Similarity</small><b>${Number(info.similarity).toFixed(3)}</b></span>` : ''}
    </div>
    ${info.relation ? `<p class="selection-path">Relation: ${escapeHtml(info.relation)}</p>` : ''}
    ${info.group_id ? `<p class="selection-path">Project: ${escapeHtml(info.group_id)}</p>` : ''}
  `;
  if (clearButton) clearButton.hidden = false;
}

function selectGraphNode(node) {
  if (!node) return;
  selectedNodeId = node.id;
  selectedLinkId = null;
  renderNodeInspector(node);
  requestGraphRedraw();
}

function selectGraphLink(link) {
  if (!link) return;
  selectedLinkId = link._id;
  selectedNodeId = null;
  renderEdgeInspector(link);
  requestGraphRedraw();
}

function clearSelection(options = {}) {
  selectedNodeId = null;
  selectedLinkId = null;
  const content = document.getElementById('selection-content');
  const clearButton = document.getElementById('clear-selection-btn');
  if (content) {
    content.className = 'empty-state';
    content.textContent = 'Select a node or edge to inspect it.';
  }
  if (clearButton) clearButton.hidden = true;
  if (!options.keepFocus && focusMode && typeof exitFocusMode === 'function') exitFocusMode();
  requestGraphRedraw();
}

function focusActivatedNote(note) {
  if (!graph || !note) return;
  const target = graph.graphData().nodes.find(node => node.id === note.id);
  if (!target) return;
  const path = [note.id];
  let current = note;
  const seen = new Set(path);
  while (current && current.parent_id && !seen.has(current.parent_id)) {
    path.push(current.parent_id);
    seen.add(current.parent_id);
    current = activatedNotesById.get(current.parent_id);
  }
  setPathHighlight(path);
  selectGraphNode(target);
  focusGraphNode(target, { kind: note.role === 'seed' ? 'seed' : 'activation-path', pathIds: path });
  showActivatedTooltip(note, lastMouseEvent || { clientX: 200, clientY: 200 });
}

function escapeHtml(value) {
  const element = document.createElement('div');
  element.textContent = value == null ? '' : String(value);
  return element.innerHTML;
}
