// ============================================================================
// BDH Graph Harness — query transport, WebSocket events and polling fallback
// ============================================================================
let activeWebSocket = null;
let lastEventSequence = null;
let graphRefreshGeneration = 0;
let graphFetchGeneration = 0;
let graphFetchInFlight = null;
let graphPollTimer = null;
let reconnectTimer = null;
let resetCameraOnNextGraph = false;
let lastRetrievalNotes = [];
let lastRetrievalQuery = '';

const GRAPH_POLL_INTERVAL_MS = 12000;

function closeActiveWebSocket() {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  if (!activeWebSocket) return;
  const socket = activeWebSocket;
  activeWebSocket = null;
  socket.onclose = null;
  socket.close();
}

function sendQuery() {
  const input = document.getElementById('query-input');
  const button = document.getElementById('query-btn');
  const query = input.value.trim();
  if (!query || button.disabled) return;

  button.disabled = true;
  button.textContent = 'Processing…';
  beginQueryParticles();
  setResponseState('Running');
  document.getElementById('response-text').textContent = 'Tracing activation through the graph…';

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 120000);
  fetch('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, vault_id: getActiveVaultId() }),
    signal: controller.signal,
  })
    .then(response => {
      clearTimeout(timeoutId);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then(data => {
      if (data.error) {
        document.getElementById('response-text').textContent = 'Error: ' + data.error;
        setResponseState('Error');
        endQueryParticles();
      } else {
        if (data.response) document.getElementById('response-text').textContent = data.response;
        renderRetrievalDiagnostics(data);
        setResponseState('Complete');
        const wsReady = activeWebSocket && activeWebSocket.readyState === WebSocket.OPEN;
        if (data.activated_notes && !wsReady) {
          handleActivation({
            type: 'activation',
            query,
            activated_notes: data.activated_notes,
            hebbian_updates: data.hebbian_updates || [],
            new_concepts: data.new_concepts || [],
            hebbian_synapses: data.hebbian_synapses,
            queries_processed: data.queries_processed,
            neuron_count: data.neuron_count,
            synapse_count: data.synapse_count,
          });
        }
      }
    })
    .catch(error => {
      clearTimeout(timeoutId);
      endQueryParticles();
      setResponseState('Error');
      document.getElementById('response-text').textContent = error.name === 'AbortError'
        ? 'Request timed out after two minutes.'
        : 'Request failed: ' + error.message;
    })
    .finally(() => {
      button.disabled = false;
      button.textContent = 'Query';
    });
}

function renderRemoteQueryResponse(event) {
  const queryInput = document.getElementById('query-input');
  const responseText = document.getElementById('response-text');
  const lastQuery = document.getElementById('stat-last');
  if (queryInput) {
    queryInput.value = event.query || '';
    queryInput.scrollTop = 0;
    if (typeof updateClearBtn === 'function') updateClearBtn();
  }
  if (responseText) {
    responseText.textContent = event.response || 'No response returned for this query.';
    responseText.scrollTop = 0;
  }
  if (lastQuery) lastQuery.textContent = event.query || '—';
  setResponseState(event.error ? 'Error' : 'Complete');
  renderRetrievalDiagnostics(event);
}

function renderRetrievalDiagnostics(payload = {}) {
  const hasNotesPayload = Object.prototype.hasOwnProperty.call(payload, 'activated_notes');
  const notes = hasNotesPayload
    ? (Array.isArray(payload.activated_notes) ? payload.activated_notes : [])
    : lastRetrievalNotes;
  const query = payload.query || lastRetrievalQuery || '';
  lastRetrievalNotes = notes;
  lastRetrievalQuery = query;
  const diagnostics = document.getElementById('retrieval-diagnostics');
  const status = document.getElementById('retrieval-status');
  const found = document.getElementById('retrieval-found');
  const confidence = document.getElementById('retrieval-confidence');
  const missing = document.getElementById('retrieval-missing');
  const focusButton = document.getElementById('retrieval-focus-btn');
  if (!diagnostics || !status || !found || !confidence || !missing) return;

  diagnostics.hidden = false;
  const topScore = notes.reduce((best, note) => Math.max(best, Number(note.final_score ?? note.score ?? 0)), 0);
  const state = !notes.length ? 'no-evidence' : topScore >= 0.65 ? 'direct-evidence' : topScore >= 0.45 ? 'weak-evidence' : 'insufficient-evidence';
  const labels = {
    'no-evidence': 'No direct evidence',
    'direct-evidence': 'Direct evidence found',
    'weak-evidence': 'Weak contextual evidence',
    'insufficient-evidence': 'Evidence below confidence threshold',
  };
  status.textContent = labels[state];
  status.dataset.state = state;
  found.textContent = notes.length ? `${notes.length} notes · top ${topScore.toFixed(3)}` : '0 notes activated';
  confidence.textContent = notes.length ? `${Math.round(topScore * 100)}% retrieval score` : '—';

  const missingItems = [];
  const normalizedQuery = query.toLowerCase();
  if (!notes.length || topScore < 0.65) missingItems.push('a note with a direct supporting snippet');
  if (normalizedQuery.includes('commit')) missingItems.push('exact commit SHA or commit status');
  if (normalizedQuery.includes('push')) missingItems.push('remote push result or branch state');
  if (normalizedQuery.includes('branch')) missingItems.push('branch name and checkout state');
  missing.textContent = missingItems.length ? missingItems.join(' · ') : 'No obvious evidence gap in the retrieved context.';
  if (focusButton) focusButton.disabled = !notes.length;
}

function focusRetrievalEvidence() {
  if (!lastRetrievalNotes.length) return;
  applyRetrievalLens(lastRetrievalNotes, lastRetrievalQuery);
  const first = lastRetrievalNotes[0];
  if (first && typeof focusActivatedNote === 'function') focusActivatedNote(first);
}

async function fetchGraphSnapshot(options = {}) {
  if (graphFetchInFlight && !options.force) return graphFetchInFlight;
  const generation = ++graphFetchGeneration;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 10000);
  const url = vaultApiUrl('/api/graph');
  console.log('[BDH 3D] fetchGraphSnapshot:', options.reason || 'manual', 'vault=', typeof activeVaultId !== 'undefined' ? activeVaultId : null, 'url=', url);
  const task = fetch(url, { signal: controller.signal, cache: 'no-store' })
    .then(response => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then(data => {
      if (generation !== graphFetchGeneration) { console.warn('[BDH 3D] graph snapshot stale (generation mismatch)'); return null; }
      if (!data || !Array.isArray(data.nodes) || data.nodes.length === 0) {
        console.warn('[BDH 3D] graph snapshot empty or malformed:', data);
        return null;
      }
      console.log('[BDH 3D] graph snapshot received:', data.nodes.length, 'nodes,', data.edges ? data.edges.length : 0, 'edges');
      if (generation !== graphFetchGeneration) return null;
      const preserveView = options.preserveView !== false && !resetCameraOnNextGraph && Boolean(graph);
      initNetwork(data, {
        preserveView,
        reheat: options.reheat === true,
      });
      const graphArea = document.getElementById('graph-area');
      if (graphArea) graphArea.classList.remove('vault-loading');
      // Safety net: ensure panels are restored even if onEngineStop never fires.
      if (typeof hideGraphLoader === 'function') hideGraphLoader();
      setVaultSelectorStatus('');
      resetCameraOnNextGraph = false;
      if (options.reason === 'polling') setConnectionStatus('degraded', 'Polling');
      return data;
    })
    .catch(error => {
      if (error.name !== 'AbortError') console.warn(`[BDH 3D] Graph snapshot failed (${options.reason || 'manual'}):`, error.message);
      if (!graph) showWebGLFallback('Graph data could not be loaded.');
      return null;
    })
    .finally(() => {
      clearTimeout(timeoutId);
      if (graphFetchInFlight === task) graphFetchInFlight = null;
    });
  graphFetchInFlight = task;
  return task;
}

function startPollingFallback() {
  if (graphPollTimer) return;
  fetchGraphSnapshot({ reason: 'polling', preserveView: Boolean(graph) });
  graphPollTimer = setInterval(() => {
    fetchGraphSnapshot({ reason: 'polling', preserveView: true });
  }, GRAPH_POLL_INTERVAL_MS);
}

function stopPollingFallback() {
  if (!graphPollTimer) return;
  clearInterval(graphPollTimer);
  graphPollTimer = null;
}

function structuralNodeCopy(node) {
  return graphNodeSnapshot(node, node);
}

function structuralLinkCopy(link) {
  return graphLinkSnapshot(link);
}

function edgeIdForPayload(edge) {
  const type = edge.type || 'wikilink';
  if (type === 'hebbian') return `hebb_${edge.source || edge.note_a}→${edge.target || edge.note_b}`;
  if (type === 'phantom') return `phantom_${edge.source}→${edge.target}`;
  return type === 'wikilink' ? `${edge.source}→${edge.target}` : `${type}_${edge.source}→${edge.target}`;
}

function addedNodePosition(nodeData, nodes) {
  const edges = nodeData.edges || [];
  const byId = new Map(nodes.map(node => [node.id, node]));
  let anchor = null;
  for (const edge of edges) {
    const otherId = edge.source === nodeData.id ? edge.target : edge.source;
    if (byId.has(otherId)) {
      anchor = byId.get(otherId);
      break;
    }
  }
  const base = anchor && Number.isFinite(anchor.x)
    ? anchor
    : { x: 0, y: 0, z: 0 };
  const theta = deterministicUnit(nodeData.id, 7) * Math.PI * 2;
  const phi = Math.acos(1 - 2 * deterministicUnit(nodeData.id, 8));
  const radius = 80;
  return {
    x: base.x + Math.sin(phi) * Math.cos(theta) * radius,
    y: base.y + Math.cos(phi) * radius,
    z: base.z + Math.sin(phi) * Math.sin(theta) * radius,
  };
}

function mergeEventIntoSourceSnapshot(event) {
  if (!sourceGraphData) return;
  const deleted = new Set(event.deleted_nodes || []);
  const changedById = new Map((event.changed_nodes || []).map(node => [node.id, node]));
  sourceGraphData.nodes = (sourceGraphData.nodes || [])
    .filter(node => !deleted.has(node.id))
    .map(node => changedById.has(node.id) ? { ...node, ...changedById.get(node.id) } : node);

  const knownNodes = new Set(sourceGraphData.nodes.map(node => node.id));
  (event.added_node_data || []).forEach(node => {
    if (!knownNodes.has(node.id)) {
      sourceGraphData.nodes.push({ ...node });
      knownNodes.add(node.id);
    }
  });

  sourceGraphData.edges = (sourceGraphData.edges || []).filter(edge => !deleted.has(edge.source) && !deleted.has(edge.target));
  const edgeKeys = new Set(sourceGraphData.edges.map(edge => edgeIdForPayload(edge)));
  (event.added_node_data || []).forEach(node => {
    (node.edges || []).forEach(edge => {
      const id = edgeIdForPayload(edge);
      if (!edgeKeys.has(id)) {
        sourceGraphData.edges.push({ ...edge });
        edgeKeys.add(id);
      }
    });
  });

  sourceGraphData.hebbian = (sourceGraphData.hebbian || []).filter(link => !deleted.has(link.note_a) && !deleted.has(link.note_b));
  const hebbianKeys = new Set(sourceGraphData.hebbian.map(link => [link.note_a, link.note_b].sort().join('|')));
  (event.added_node_data || []).forEach(node => {
    (node.hebbian || []).forEach(link => {
      const key = [link.note_a, link.note_b].sort().join('|');
      if (!hebbianKeys.has(key)) {
        sourceGraphData.hebbian.push({ ...link });
        hebbianKeys.add(key);
      }
    });
  });
}

function applyGraphRefresh(event) {
  if (!graph) {
    fetchGraphSnapshot({ reason: 'refresh-without-graph', preserveView: false, force: true });
    return;
  }
  const changedNodes = event.changed_nodes || [];
  const newConcepts = event.new_concepts || [];
  const deletedNodes = event.deleted_nodes || [];
  const addedNodeData = event.added_node_data || [];
  const refreshGeneration = ++graphRefreshGeneration;
  const current = graph.graphData();
  const existingIds = new Set(current.nodes.map(node => node.id));
  const deletedSet = new Set(deletedNodes);
  const newConceptsForAnimation = newConcepts.filter(concept =>
    !existingIds.has(concept.id) && addedNodeData.some(node => node.id === concept.id)
  );

  const nodes = current.nodes.map(structuralNodeCopy).filter(node => !deletedSet.has(node.id));
  const links = current.links.map(structuralLinkCopy).filter(link => {
    if (deletedSet.has(link.source) || deletedSet.has(link.target)) {
      delete edgeInfoMap[link._id];
      return false;
    }
    return true;
  });
  const finalNodeIds = new Set(nodes.map(node => node.id));
  const linkIds = new Set(links.map(link => link._id));

  deletedNodes.forEach(id => {
    delete nodeDataMap[id];
    delete nodeTagColorMap[id];
    delete neurogenesisNodes[id];
  });

  addedNodeData.forEach(data => {
    if (!finalNodeIds.has(data.id) && nodeMatchesSourceFilter(data) && (showOrphans || (data.edges || []).length)) {
      const tags = normalizeTags(data.tags || '');
      const neurogenesis = tags.some(tag => tag.toLowerCase() === 'neurogenesis');
      const position = addedNodePosition(data, nodes);
      const color = neurogenesis ? COLORS.neurogenesis
        : showTagColors && tags.length && tagColorMap[tags[0]] ? tagColorMap[tags[0]]
          : sourceColor(data);
      nodeDataMap[data.id] = data;
      nodeTagColorMap[data.id] = color;
      nodes.push({
        id: data.id,
        name: data.display_label || data.title || data.id.split('/').pop(),
        color,
        val: neurogenesis ? 18 : 8,
        x: position.x,
        y: position.y,
        z: position.z,
        _mass: neurogenesis ? 2.1 : 1,
        _synapticGlow: neurogenesis ? 0.85 : 0,
        _opacity: data.dormant ? 0.42 : 1,
        _shape: neurogenesis ? 'diamond' : data.dormant ? 'triangleDown' : 'circle',
        _dormant: Boolean(data.dormant),
        _tags: data.tags || '',
        _title: data.title || data.id,
        _displayLabel: data.display_label || data.title || data.id,
        _path: data.path || '',
        _text: data.text || '',
      });
      finalNodeIds.add(data.id);
      if (neurogenesis) neurogenesisNodes[data.id] = { title: data.title || data.id };
    } else if (!nodeDataMap[data.id]) {
      nodeDataMap[data.id] = data;
    }

    (data.edges || []).forEach(edge => {
      if (!finalNodeIds.has(edge.source) || !finalNodeIds.has(edge.target)) return;
      const type = edge.type || 'wikilink';
      const id = edgeIdForPayload(edge);
      if (linkIds.has(id)) return;
      edgeInfoMap[id] = {
        source_title: (nodeDataMap[edge.source] || {}).display_label || (nodeDataMap[edge.source] || {}).title || edge.source,
        target_title: (nodeDataMap[edge.target] || {}).display_label || (nodeDataMap[edge.target] || {}).title || edge.target,
        type,
        relation: edge.relation,
        group_id: edge.group_id,
      };
      links.push(graphLinkSnapshot({
        source: edge.source,
        target: edge.target,
        type,
        relation: edge.relation,
        group_id: edge.group_id,
        color: type === 'counterpart' ? COLORS.edgeCounterpart
          : type === 'project_context' ? COLORS.edgeProjectContext
            : type === 'project_reference' ? COLORS.edgeProjectReference
              : COLORS.edgeWikilink,
        width: type === 'wikilink' ? 0.5 : 1.2,
        _id: id,
        _dashes: ['counterpart', 'project_context', 'project_reference'].includes(type),
      }));
      linkIds.add(id);
    });

    (data.hebbian || []).forEach(synapse => {
      if (!finalNodeIds.has(synapse.note_a) || !finalNodeIds.has(synapse.note_b)) return;
      const id = `hebb_${synapse.note_a}→${synapse.note_b}`;
      if (linkIds.has(id)) return;
      edgeInfoMap[id] = {
        source_title: (nodeDataMap[synapse.note_a] || {}).display_label || (nodeDataMap[synapse.note_a] || {}).title || synapse.note_a,
        target_title: (nodeDataMap[synapse.note_b] || {}).display_label || (nodeDataMap[synapse.note_b] || {}).title || synapse.note_b,
        type: 'hebbian',
        weight: synapse.weight,
        frequency: synapse.frequency,
      };
      links.push(graphLinkSnapshot({
        source: synapse.note_a,
        target: synapse.note_b,
        type: 'hebbian',
        weight: synapse.weight,
        frequency: synapse.frequency,
        color: weightColor(synapse.weight),
        width: Math.min(1 + synapse.weight * 3, 5),
        _id: id,
      }));
      linkIds.add(id);
    });
  });

  changedNodes.forEach(change => {
    const existing = nodes.find(node => node.id === change.id);
    if (existing) {
      existing.name = change.display_label || change.title || existing.name;
      existing._title = change.title || existing._title;
      existing._displayLabel = change.display_label || change.title || existing._displayLabel;
    }
    if (nodeDataMap[change.id]) Object.assign(nodeDataMap[change.id], change);
  });

  mergeEventIntoSourceSnapshot(event);
  setGraphDataPreservingView({ nodes, links }, { reheat: true });
  if (event.neurons != null) document.getElementById('stat-neurons').textContent = event.neurons;
  if (event.synapses != null) document.getElementById('stat-synapses').textContent = event.synapses;
  document.getElementById('stat-wikilinks').textContent = links.filter(link => link.type === 'wikilink').length;

  newConceptsForAnimation.forEach((concept, index) => {
    nodeBirthScaleState.set(concept.id, 1.8);
    nodeActivationColor.set(concept.id, COLORS.neurogenesis);
    setTimeout(() => {
      if (refreshGeneration !== graphRefreshGeneration) return;
      nodeBirthScaleState.set(concept.id, 1.25);
      requestGraphRedraw();
    }, index * 120 + 280);
    setTimeout(() => {
      if (refreshGeneration !== graphRefreshGeneration) return;
      nodeBirthScaleState.delete(concept.id);
      nodeActivationColor.delete(concept.id);
      requestGraphRedraw();
    }, index * 120 + 1050);
  });

  changedNodes.forEach((node, index) => {
    setTimeout(() => {
      if (refreshGeneration !== graphRefreshGeneration) return;
      nodeActivationColor.set(node.id, COLORS.activated);
      nodeActivationOpacity.set(node.id, 1);
      requestGraphRedraw();
    }, index * 180 + 500);
    setTimeout(() => {
      if (refreshGeneration !== graphRefreshGeneration) return;
      nodeActivationColor.delete(node.id);
      nodeActivationOpacity.delete(node.id);
      requestGraphRedraw();
    }, index * 180 + 1900);
  });
}

function applyNodeUpdate(event) {
  if (!graph) return;
  const generation = ++graphRefreshGeneration;
  const changedNodes = event.changed_nodes || [];
  const deletedSet = new Set(event.deleted_nodes || []);
  const current = graph.graphData();
  const nodes = current.nodes.map(structuralNodeCopy).filter(node => !deletedSet.has(node.id));
  const links = current.links.map(structuralLinkCopy).filter(link => !deletedSet.has(link.source) && !deletedSet.has(link.target));

  changedNodes.forEach(change => {
    const node = nodes.find(item => item.id === change.id);
    if (node) {
      node.name = change.display_label || change.title || node.name;
      node._title = change.title || node._title;
      node._displayLabel = change.display_label || change.title || node._displayLabel;
    }
    if (nodeDataMap[change.id]) Object.assign(nodeDataMap[change.id], change);
  });
  deletedSet.forEach(id => {
    delete nodeDataMap[id];
    delete nodeTagColorMap[id];
    delete neurogenesisNodes[id];
  });
  mergeEventIntoSourceSnapshot(event);
  if (changedNodes.length || deletedSet.size) {
    setGraphDataPreservingView({ nodes, links }, { reheat: deletedSet.size > 0 });
  }

  changedNodes.forEach((change, index) => {
    setTimeout(() => {
      if (generation !== graphRefreshGeneration) return;
      nodeActivationColor.set(change.id, COLORS.activated);
      nodeActivationOpacity.set(change.id, 1);
      requestGraphRedraw();
    }, index * 180);
    setTimeout(() => {
      if (generation !== graphRefreshGeneration) return;
      nodeActivationColor.delete(change.id);
      nodeActivationOpacity.delete(change.id);
      requestGraphRedraw();
    }, index * 180 + 1600);
  });
}

function connectWS() {
  closeActiveWebSocket();
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(protocol + '//' + location.host + vaultApiUrl('/ws'));
  activeWebSocket = socket;
  setConnectionStatus('', 'Connecting');

  socket.onopen = () => {
    if (activeWebSocket !== socket) return;
    lastEventSequence = null;
    stopPollingFallback();
    setConnectionStatus('connected', 'Live');
    setVaultSelectorStatus('');
    // Proactively fetch the graph snapshot — the server may not send a 'graph' event on connect.
    if (typeof fetchGraphSnapshot === 'function') {
      fetchGraphSnapshot({ reason: 'websocket-connect', preserveView: Boolean(graph && sourceGraphData), force: true });
    }
  };

  socket.onmessage = message => {
    try {
      const event = JSON.parse(message.data);
      if (!isActiveVaultEvent(event)) return;
      if (Number.isFinite(event.sequence)) {
        if (lastEventSequence !== null && event.sequence <= lastEventSequence) return;
        lastEventSequence = event.sequence;
      }

      if (event.type === 'graph') {
        // The onopen handler already fetches on connect; skip duplicate bootstrap.
      } else if (event.type === 'activation' || event.type === 'neurogenesis') {
        handleActivation(event);
      } else if (event.type === 'query_response') {
        renderRemoteQueryResponse(event);
      } else if (event.type === 'graph_refresh') {
        applyGraphRefresh(event);
      } else if (event.type === 'node_update') {
        applyNodeUpdate(event);
      }
    } catch (error) {
      console.error('[BDH 3D] WebSocket message error:', error);
    }
  };

  socket.onclose = () => {
    if (activeWebSocket !== socket) return;
    activeWebSocket = null;
    setConnectionStatus('degraded', 'Polling');
    startPollingFallback();
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connectWS();
    }, 2000);
  };

  socket.onerror = () => {
    setConnectionStatus('degraded', 'Reconnecting');
    socket.close();
  };
}
