// ============================================================================
// Query — HTTP POST to /api/query
// ============================================================================
function sendQuery() {
  const input = document.getElementById('query-input');
  const btn = document.getElementById('query-btn');
  const query = input.value.trim();
  if (!query) return;
  btn.disabled = true;
  btn.textContent = 'Processing...';
  document.getElementById('response-text').textContent = 'Thinking...';

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 120000);

  fetch('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
    signal: controller.signal,
  }).then(r => {
    clearTimeout(timeoutId);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }).then(data => {
    if (data.error) {
      document.getElementById('response-text').textContent = 'Error: ' + data.error;
    } else {
      if (data.response) {
        document.getElementById('response-text').textContent = data.response;
      }
      if (data.activated_notes) {
        handleActivation({
          type: 'activation',
          query: query,
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
    btn.disabled = false;
    btn.textContent = 'Query';
  }).catch(err => {
    clearTimeout(timeoutId);
    btn.disabled = false;
    btn.textContent = 'Query';
    if (err.name === 'AbortError') {
      document.getElementById('response-text').textContent = 'Request timed out (2 min). The LLM might be slow — try again.';
    } else {
      document.getElementById('response-text').textContent = 'Request failed: ' + err.message;
    }
  });
}

// ============================================================================
// WebSocket connection
// ============================================================================
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(proto + '//' + location.host + '/ws');
  const ind = document.getElementById('status-indicator');

  ws.onopen = () => { ind.classList.add('connected'); };

  ws.onmessage = (msg) => {
    try {
      const event = JSON.parse(msg.data);
      if (event.type === 'graph') {
        initNetwork(event);
        if (showTagColors) toggleTagColors(true);
      } else if (event.type === 'activation') {
        handleActivation(event);
      } else if (event.type === 'graph_refresh') {
        // Delta update: build a fresh graph and submit all at once.
        // NEVER mutate live graph.graphData() objects — force-graph marks
        // internal properties as non-writable and mutations crash in Safari.
        console.log('Graph refresh (delta):', event.message);
        const changedNodes = event.changed_nodes || [];
        const newConcepts = event.new_concepts || [];
        const deletedNodes = event.deleted_nodes || [];
        const addedNodeData = event.added_node_data || [];

        if (!graph) return;
        const currentData = graph.graphData();
        const deletedSet = new Set(deletedNodes);

        // Build a fresh, independent copy of the current graph
        const freshNodes = currentData.nodes.map(n => ({
          id: n.id, name: n.name, color: n.color, val: n.val,
          _opacity: n._opacity, _shape: n._shape, _dormant: n._dormant,
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

        // 1. Remove deleted nodes and their links from fresh copies
        const keepNodes = freshNodes.filter(n => !deletedSet.has(n.id));
        const keepLinks = freshLinks.filter(l => {
          if (deletedSet.has(l.source) || deletedSet.has(l.target)) {
            delete edgeInfoMap[l._id];
            return false;
          }
          return true;
        });
        deletedNodes.forEach(nid => {
          delete nodeDataMap[nid];
          delete nodeTagColorMap[nid];
          delete neurogenesisNodes[nid];
        });

        // 2. Add new nodes with their edges
        addedNodeData.forEach(nd => {
          if (!nodeDataMap[nd.id]) {
            const isNeurogenesis = (nd.tags || '').toLowerCase().includes('neurogenesis');
            const color = isNeurogenesis ? COLORS.neurogenesis : COLORS.inactive;
            nodeDataMap[nd.id] = nd;
            nodeTagColorMap[nd.id] = color;
            keepNodes.push({
              id: nd.id,
              name: nd.title || nd.id.split('/').pop(),
              color: color,
              val: 10,
              _opacity: 1.0,
              _shape: isNeurogenesis ? 'diamond' : 'circle',
              _dormant: false,
              _tags: nd.tags || '',
              _title: nd.title || nd.id,
              _path: nd.path || '',
              _text: nd.text || '',
            });
            if (isNeurogenesis) neurogenesisNodes[nd.id] = { title: nd.title };
          }
          // Add edges from this new node
          (nd.edges || []).forEach(e => {
            const eid = e.source + '→' + e.target;
            const srcTitle = (nodeDataMap[e.source] || {}).title || e.source;
            const tgtTitle = (nodeDataMap[e.target] || {}).title || e.target;
            edgeInfoMap[eid] = { source_title: srcTitle, target_title: tgtTitle, type: 'wikilink' };
            keepLinks.push({
              source: e.source, target: e.target,
              color: COLORS.edgeWikilink, width: 0.5, type: 'wikilink',
              particles: 0, _id: eid, _visible: true,
            });
          });
          // Add hebbian edges from this new node
          (nd.hebbian || []).forEach(h => {
            const eid = 'hebb_' + h.note_a + '→' + h.note_b;
            const srcTitle = (nodeDataMap[h.note_a] || {}).title || h.note_a;
            const tgtTitle = (nodeDataMap[h.note_b] || {}).title || h.note_b;
            edgeInfoMap[eid] = { source_title: srcTitle, target_title: tgtTitle, type: 'hebbian', weight: h.weight, frequency: h.frequency };
            keepLinks.push({
              source: h.note_a, target: h.note_b,
              color: weightColor(h.weight), width: Math.min(1 + h.weight * 3, 5),
              type: 'hebbian', weight: h.weight, frequency: h.frequency,
              particles: 0, _id: eid, _visible: true,
            });
          });
        });

        // 3. Update changed node labels
        changedNodes.forEach(n => {
          const existing = keepNodes.find(x => x.id === n.id);
          if (existing) {
            existing.name = n.title;
            existing._title = n.title;
          }
          if (nodeDataMap[n.id]) nodeDataMap[n.id].title = n.title;
        });

        // 4. Submit the fresh, complete graph
        setGraphDataPreservingView({ nodes: keepNodes, links: keepLinks }, { reheat: true });

        // 5. Update stats
        if (event.neurons != null) document.getElementById('stat-neurons').textContent = event.neurons;
        if (event.synapses != null) document.getElementById('stat-synapses').textContent = event.synapses;

        // 6. Animate new concept birth (do NOT clobber existing activation state)
        if (newConcepts.length > 0 && isHoverActive()) {
          // Only pulse new concepts if user is actively hovering
        } else if (newConcepts.length > 0) {
          // Subtle birth pulse without wiping query activation
          newConcepts.forEach((nc, i) => {
            setTimeout(() => {
              nodeActivationOpacity.set(nc.id, 1.0);
              nodeActivationColor.set(nc.id, COLORS.neurogenesis);
              requestGraphRedraw();
            }, i * 150);
            setTimeout(() => {
              // Restore to dim if query is active, or full opacity if not
              const hasQuery = nodeActivationOpacity.size > 0 && [...nodeActivationOpacity.values()].some(o => o < 1);
              nodeActivationOpacity.set(nc.id, hasQuery ? 0.3 : 1.0);
              nodeActivationColor.delete(nc.id);
              requestGraphRedraw();
            }, i * 150 + 1200);
          });
        }

        // 7. Pulse changed nodes — use activation Maps
        if (changedNodes.length > 0) {
          changedNodes.forEach((n, i) => {
            setTimeout(() => {
              nodeActivationColor.set(n.id, COLORS.activated);
              nodeActivationOpacity.set(n.id, 1.0);
              requestGraphRedraw();
              setTimeout(() => {
                nodeActivationColor.delete(n.id);
                const hasQuery = nodeActivationOpacity.size > 0 && [...nodeActivationOpacity.values()].some(o => o < 1);
                nodeActivationOpacity.set(n.id, hasQuery ? 0.3 : 1.0);
                requestGraphRedraw();
              }, 2500);
            }, i * 300 + 1500);
          });
        }

      } else if (event.type === 'node_update') {
        // Lightweight update — build fresh graph, no mutation of live data
        console.log('Node update:', event.message);
        const changedNodes = event.changed_nodes || [];
        const deletedNodes = event.deleted_nodes || [];

        if (!graph) return;
        const currentData = graph.graphData();
        const deletedSet = new Set(deletedNodes);

        // Build fresh copies
        const freshNodes = currentData.nodes.map(n => ({
          id: n.id, name: n.name, color: n.color, val: n.val,
          _opacity: n._opacity, _shape: n._shape, _dormant: n._dormant,
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

        // Filter deleted nodes
        const keepNodes = freshNodes.filter(n => !deletedSet.has(n.id));
        const keepLinks = freshLinks.filter(l => {
          return !deletedSet.has(l.source) && !deletedSet.has(l.target);
        });

        if (deletedNodes.length > 0) {
          setGraphDataPreservingView({ nodes: keepNodes, links: keepLinks }, { reheat: true });
        }

        // Update changed nodes with pulse via activation Maps
        changedNodes.forEach((n, i) => {
          setTimeout(() => {
            // Update name in fresh data (will be reflected on next graph refresh)
            const existing = keepNodes.find(x => x.id === n.id);
            if (existing) {
              existing.name = n.title;
              existing._title = n.title;
            }
            if (nodeDataMap[n.id]) nodeDataMap[n.id].title = n.title;

            // Pulse via Maps
            nodeActivationColor.set(n.id, COLORS.activated);
            nodeActivationOpacity.set(n.id, 1.0);
            requestGraphRedraw();
            setTimeout(() => {
              nodeActivationColor.delete(n.id);
              const hasQuery = nodeActivationOpacity.size > 0 && [...nodeActivationOpacity.values()].some(o => o < 1);
              nodeActivationOpacity.set(n.id, hasQuery ? 0.3 : 1.0);
              requestGraphRedraw();
            }, 2500);
          }, i * 300);
        });
      }
    } catch (e) { console.error('Parse error:', e); }
  };

  ws.onclose = () => {
    ind.classList.remove('connected');
    setTimeout(connectWS, 2000);  // auto-reconnect
  };

  ws.onerror = () => { ws.close(); };
}

