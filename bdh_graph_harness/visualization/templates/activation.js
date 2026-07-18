
// Animation generation invalidates callbacks from older queries.
let activationGeneration = 0;

function invalidateActivationAnimations() {
  activationGeneration += 1;
}

function handleActivation(event) {
  const isNeurogenesisUpdate = event.type === 'neurogenesis';
  const generation = isNeurogenesisUpdate
    ? activationGeneration
    : ++activationGeneration;
  const isCurrent = () => generation === activationGeneration;
  const activated = isNeurogenesisUpdate
    ? Array.from(activatedNotesById.values())
    : (event.activated_notes || []);
  const newConcepts = event.new_concepts || [];
  if (!isNeurogenesisUpdate) {
    activatedNotesById.clear();
    activated.forEach(note => activatedNotesById.set(note.id, note));
  }
  const activatedIds = new Set(activated.map(n => n.id));
  const seedId = activated.length > 0 ? activated[0].id : null;
  const hasActivation = activatedIds.size > 0;

  if (!graph || typeof graph.graphData !== 'function') {
    console.warn('Activation received before graph initialization; skipping visual update');
    return;
  }

  const currentData = graph.graphData();

  // Write activation state to maps (never touch live node objects)
  currentData.nodes.forEach(node => {
    if (hasActivation) {
      nodeActivationOpacity.set(node.id, activatedIds.has(node.id) ? 1.0 : 0.55);
      if (node.id === seedId) {
        nodeActivationColor.set(node.id, COLORS.seed);
      } else if (activatedIds.has(node.id)) {
        nodeActivationColor.set(node.id, COLORS.activated);
      } else {
        nodeActivationColor.set(node.id, nodeTagColorMap[node.id] || COLORS.inactive);
      }
    } else {
      nodeActivationOpacity.delete(node.id);
      nodeActivationColor.delete(node.id);
    }
  });

  // Write link activation state to maps. Keep the background graph visible:
  // hiding every inactive link makes the query view feel like the graph vanished.
  // Active links get brighter particles; inactive links simply fall back to their
  // normal low-opacity rendering.
  currentData.links.forEach(link => {
    const key = linkKey(link);
    const sourceId = linkEndpointId(link.source);
    const targetId = linkEndpointId(link.target);
    const bothActive = activatedIds.has(sourceId) && activatedIds.has(targetId);
    if (hasActivation) {
      // Live particle animation on active edges — visible immediately, not just at settle
      if (bothActive && link.type !== 'wikilink') {
        linkActivationVisible.set(key, true);
        linkParticlesState.set(key, 8);
        linkParticleColorState.set(key, COLORS.edgeHebbianPulse);
        linkActivationColor.set(key, COLORS.edgeHebbianPulse);
      } else {
        linkActivationVisible.delete(key);
        linkParticlesState.delete(key);
        linkParticleColorState.delete(key);
        linkActivationColor.delete(key);
      }
    } else {
      linkActivationVisible.delete(key);
      linkParticlesState.delete(key);
      linkParticleColorState.delete(key);
      linkActivationColor.delete(key);
    }
  });

  requestGraphRedraw();

  // Pulse Hebbian synapses that were strengthened in this query
  const hebbianUpdates = event.hebbian_updates || [];
  const pendingHebbianLinks = [];

  hebbianUpdates.forEach((h, idx) => {
    const parts = h.pair.split('|');
    if (parts.length !== 2) return;
    const [a, b] = parts;
    hebbianMap[h.pair] = h.weight;

    // Find the Hebbian link in current graph data
    const targetLink = currentData.links.find(l => {
      const ls = linkEndpointId(l.source);
      const lt = linkEndpointId(l.target);
      return (ls === a && lt === b) || (ls === b && lt === a);
    });

    const delay = idx * 120;
    const newWidth = Math.min(1 + h.weight * 3, 5);

    // A new Hebbian edge is persisted by the server and will arrive in the
    // following graph_refresh/node_update. Do not rebuild force-graph here:
    // rebuilding once per update resets drag/pick handlers and can strand the
    // canvas after an otherwise normal query.
    if (!targetLink && (activatedIds.has(a) || activatedIds.has(b))) {
      const eid = 'hebb_' + a + '→' + b;
      const srcTitle = (nodeDataMap[a] || {}).display_label || (nodeDataMap[a] || {}).title || a;
      const tgtTitle = (nodeDataMap[b] || {}).display_label || (nodeDataMap[b] || {}).title || b;
      edgeInfoMap[eid] = {
        source_title: srcTitle,
        target_title: tgtTitle,
        type: 'hebbian', weight: h.weight, frequency: h.frequency,
      };
      pendingHebbianLinks.push({
        source: a,
        target: b,
        color: weightColor(h.weight),
        width: newWidth,
        type: 'hebbian',
        weight: h.weight,
        frequency: h.frequency,
        particles: 0,
        _id: eid,
        _visible: true,
      });
      return;
    }

    if (!targetLink) return;
    const key = linkKey(targetLink);

    // Pulse: use Maps for particles/width/color
    setTimeout(() => {
      if (!isCurrent()) return;
      linkParticlesState.set(key, 8);
      linkParticleColorState.set(key, COLORS.edgeHebbianPulse);
      linkActivationColor.set(key, COLORS.edgeHebbianPulse);
      linkWidthState.set(key, newWidth + 6);
      linkActivationVisible.set(key, true);
      requestGraphRedraw();
    }, delay);

    setTimeout(() => {
      if (!isCurrent()) return;
      linkWidthState.set(key, newWidth + 4);
      requestGraphRedraw();
    }, delay + 600);

    setTimeout(() => {
      if (!isCurrent()) return;
      linkWidthState.set(key, newWidth + 1.5);
      linkActivationColor.set(key, weightColor(h.weight));
      requestGraphRedraw();
    }, delay + 1400);

    setTimeout(() => {
      if (!isCurrent()) return;
      linkWidthState.delete(key);
      linkParticlesState.delete(key);
      linkParticleColorState.delete(key);
      linkActivationColor.delete(key);
      // No applyEdgeFilters here — would rebuild graph and cause flash
    }, delay + 2500);
  });

  // The server persists new Hebbian pairs immediately, but no graph_refresh is
  // emitted for a normal query. Add all missing pairs in one structural update
  // so they become visible without resetting the graph once per synapse.
  if (pendingHebbianLinks.length > 0) {
    const existingKeys = new Set(currentData.links.map(linkKey));
    const newLinks = pendingHebbianLinks.filter(link => !existingKeys.has(linkKey(link)));
    if (newLinks.length > 0) {
      setGraphDataPreservingView(
        { nodes: currentData.nodes, links: currentData.links.concat(newLinks) },
        { reheat: true },
      );
    }
  }

  // Reset activation state after all Hebbian settle animations finish
  const activationTimeoutMs = Math.max(5000, (hebbianUpdates.length - 1) * 120 + 3000);
  setTimeout(() => {
    if (!isCurrent()) return;
    clearActivationState();
    endQueryParticles();
    requestGraphRedraw();
  }, activationTimeoutMs);

  // Update side panel
  const listEl = document.getElementById('activated-list');
  const countEl = document.getElementById('activation-count');
  if (countEl) countEl.textContent = String(activated.length);
  listEl.innerHTML = '';
  if (activated.length === 0) {
    listEl.innerHTML = '<div class="empty">No notes activated</div>';
  } else {
    activated.forEach(note => {
      const li = document.createElement('li');
      const role = note.role || (note.id === seedId ? 'seed' : 'graph_neighbor');
      const roleLabel = role === 'seed' ? 'seed' : 'hop ' + (note.hop ?? 1);
      li.className = role;
      const score = Number(note.final_score ?? note.score ?? 0).toFixed(4);
      const hybrid = Number(note.hybrid_score || 0).toFixed(3);
      const label = note.display_label || note.title || note.id;
      const path = note.relative_path || note.path || '';
      const source = note.source_id || note.source_type || 'vault';
      li.innerHTML = '<span class="activation-copy"><strong class="activation-role">' + escapeHtml(roleLabel) + '</strong> ' +
        '<strong class="activation-title">' + escapeHtml(label) + '</strong>' +
        (path ? '<small class="activation-path">' + escapeHtml(path) + '</small>' : '') +
        '<small class="activation-source">' + escapeHtml(source) + '</small></span>' +
        '<span class="score" title="final ' + score + ' · hybrid ' + hybrid + '">' + score + '</span>';
      li.addEventListener('mouseenter', (evt) => showActivatedTooltip(note, evt));
      li.addEventListener('mousemove', (evt) => positionTooltip(evt));
      li.addEventListener('mouseleave', () => hideTooltip());
      li.addEventListener('click', () => focusActivatedNote(note));
      listEl.appendChild(li);
    });
  }

  const conceptsSection = document.getElementById('new-concepts-section');
  const conceptsList = document.getElementById('new-concepts-list');
  if (conceptsSection && conceptsList) {
    conceptsList.innerHTML = '';
    conceptsSection.hidden = newConcepts.length === 0;
    newConcepts.forEach(concept => {
      const li = document.createElement('li');
      li.className = 'new-concept';
      li.innerHTML = '<span><strong>✦</strong> ' + escapeHtml(concept.title || concept.id) + '</span>' +
        '<span class="concept-source">neurogenesis</span>';
      li.addEventListener('click', () => focusActivatedNote({
        id: concept.id, title: concept.title || concept.id, role: 'seed', hop: 0,
      }));
      conceptsList.appendChild(li);
    });
  }

  if (typeof renderRetrievalDiagnostics === 'function') {
    renderRetrievalDiagnostics({ ...event, query: event.query || lastRetrievalQuery, activated_notes: activated });
  }
  // Query lens is opt-in — do not auto-activate it on every query.
  // The graph stays fully visible with activated nodes highlighted by opacity + bloom.
  // Users can still click "Retrieval lens" to filter manually.
  // if (typeof applyRetrievalLens === 'function') applyRetrievalLens(activated, event.query || lastRetrievalQuery);

  // Add new concept nodes (neurogenesis) with birth animation
  // Build fresh graph with new nodes — never mutate live data.
  if (newConcepts.length > 0) {
    // Clone current graph
    const freshNodes = currentData.nodes.map(n => ({
      id: n.id, name: n.name, color: n.color, val: n.val,
      _opacity: n._opacity, _shape: n._shape, _dormant: n._dormant,
      _mass: n._mass, _synapticGlow: n._synapticGlow,
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

    newConcepts.forEach((nc, ncIdx) => {
      if (nc.id && !nodeDataMap[nc.id]) {
        const ncNode = { id: nc.id, title: nc.title || nc.id, tags: 'neurogenesis', text: nc.definition || '', path: '' };
        nodeDataMap[nc.id] = ncNode;
        nodeTagColorMap[nc.id] = COLORS.neurogenesis;
        neurogenesisNodes[nc.id] = { title: nc.title || nc.id };
        totalConcepts++;

        // Add node and edges to fresh graph (not live data)
        const birthDelay = ncIdx * 200;
        const newNode = {
          id: nc.id,
          name: nc.title || nc.id.split('/').pop(),
          color: COLORS.neurogenesis,
          val: 20,
          _mass: 2.2,
          _synapticGlow: 0.85,
          _opacity: 1.0,
          _shape: 'diamond',
          _dormant: false,
          _tags: 'neurogenesis',
          _title: nc.title || nc.id,
          _path: '',
          _text: nc.definition || '',
        };
        nodeBirthScaleState.set(nc.id, 2.0);
        freshNodes.push(newNode);

        // Add edges to source notes
        const sources = nc.source_notes || [];
        sources.forEach(srcTitle => {
          const srcNode = allGraphNodes.find(n => n.title === srcTitle);
          if (srcNode) {
            const eid = nc.id + '→' + srcNode.id;
            edgeInfoMap[eid] = { source_title: nc.title || nc.id, target_title: srcTitle, type: 'neurogenesis' };
            freshLinks.push({
              source: nc.id,
              target: srcNode.id,
              color: COLORS.edgeNeurogenesis,
              width: 2,
              type: 'neurogenesis',
              particles: 0,
              _id: eid,
              _visible: true,
              _dashes: true,
            });
          }
        });

        // Animate the birth scale through the renderer Maps. Do not rebuild
        // graphData for every pulse: that restarts d3 and causes visible jumps.
        setTimeout(() => {
          if (!isCurrent()) return;
          nodeBirthScaleState.set(nc.id, 1.5);
          requestGraphRedraw();
        }, birthDelay + 300);

        setTimeout(() => {
          if (!isCurrent()) return;
          nodeBirthScaleState.delete(nc.id);
          requestGraphRedraw();
        }, birthDelay + 600);

        const cEl = document.getElementById('stat-concepts');
        if (cEl) {
          cEl.textContent = totalConcepts;
          cEl.style.color = COLORS.neurogenesis;
          setTimeout(() => {
            if (isCurrent()) cEl.style.color = '';
          }, 1500);
        }
      }
    });

    // Submit all new nodes/links once, then animate only through Maps.
    setGraphDataPreservingView({ nodes: freshNodes, links: freshLinks }, { reheat: true });

    // Do not auto-fit here: a delayed camera animation fights user navigation
    // and any subsequent activation/graph refresh.
  }

  // Update stats
  document.getElementById('stat-last').textContent = event.query || '—';

  const nEl = document.getElementById('stat-neurons');
  if (nEl) {
    if (event.neuron_count != null) nEl.textContent = event.neuron_count;
    else if (newConcepts.length > 0) nEl.textContent = parseInt(nEl.textContent) + newConcepts.length;
  }

  const sEl = document.getElementById('stat-synapses');
  if (sEl && event.synapse_count != null) sEl.textContent = event.synapse_count;

  if (event.queries_processed != null) {
    const qEl = document.getElementById('stat-queries');
    if (qEl) qEl.textContent = event.queries_processed;
  }

  if (event.hebbian_synapses != null) {
    const hebbEl = document.getElementById('stat-hebbian');
    if (hebbEl) {
      const prev = parseInt(hebbEl.textContent) || 0;
      hebbEl.textContent = event.hebbian_synapses;
      if (event.hebbian_synapses > prev) {
        hebbEl.style.color = COLORS.edgeHebbianPulse;
        setTimeout(() => { hebbEl.style.color = ''; }, 1000);
      }
    }
  }

  if (event.dormant_count != null) {
    const dormantEl = document.getElementById('stat-dormant');
    const dormantCountEl = document.getElementById('stat-dormant-count');
    if (dormantEl && dormantCountEl) {
      dormantCountEl.textContent = event.dormant_count;
      dormantEl.style.display = event.dormant_count > 0 ? '' : 'none';
    }
  }

  // Show indicator pulse
  const ind = document.getElementById('status-indicator');
  if (ind) {
    ind.classList.add('active');
    setTimeout(() => ind.classList.remove('active'), 1000);
  }
}

