from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "bdh_graph_harness/visualization/templates/graph-3d-utils.js"


def run_node(source: str) -> None:
    result = subprocess.run(
        ["node", "-e", source],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_structural_node_clone_preserves_all_three_dimensions():
    assert UTILS.exists(), "3D graph utility module is missing"

    run_node(
        f"""
        const utils = require('{UTILS.as_posix()}');
        const live = {{ x: 10, y: 20, z: 30, vx: 1, vy: 2, vz: 3, fx: 4, fy: 5, fz: 6 }};
        const incoming = {{ id: 'node-1', name: 'Node', color: '#fff', val: 8, z: 90, _shape: 'diamond' }};
        const cloned = utils.cloneNodeForStructuralUpdate(incoming, live);
        const expected = {{ x: 10, y: 20, z: 90, vx: 1, vy: 2, vz: 3, fx: 4, fy: 5, fz: 6 }};
        for (const [key, value] of Object.entries(expected)) {{
          if (cloned[key] !== value) throw new Error(`${{key}}: expected ${{value}}, got ${{cloned[key]}}`);
        }}
        if (cloned._shape !== 'diamond') throw new Error('semantic shape was not preserved');
        """
    )


def test_focus_camera_keeps_node_centered_at_a_bounded_distance():
    assert UTILS.exists(), "3D graph utility module is missing"

    run_node(
        f"""
        const utils = require('{UTILS.as_posix()}');
        const node = {{ x: 10, y: 20, z: 30 }};
        const camera = {{ x: 310, y: 420, z: 530 }};
        const position = utils.cameraPositionForFocus(node, camera, 120);
        const distance = Math.hypot(position.x - node.x, position.y - node.y, position.z - node.z);
        if (Math.abs(distance - 120) > 1e-9) throw new Error(`focus distance ${{distance}}`);

        const fallback = utils.cameraPositionForFocus(node, node, 90);
        if (fallback.x !== 10 || fallback.y !== 20 || fallback.z !== 120) {{
          throw new Error(`invalid zero-vector fallback: ${{JSON.stringify(fallback)}}`);
        }}
        """
    )


def test_fog_density_adapts_to_large_graph_fit_distance():
    """A fixed fog density makes large fitted graphs indistinguishable from the background."""
    assert UTILS.exists(), "3D graph utility module is missing"

    run_node(
        f"""
        const utils = require('{UTILS.as_posix()}');
        const nearby = utils.fogDensityForFitDistance(1000, false);
        const distant = utils.fogDensityForFitDistance(10000, false);
        if (nearby !== 0.00038) throw new Error(`unexpected nearby density: ${{nearby}}`);
        if (!(distant > 0 && distant <= 0.00005)) throw new Error(`distant graph remains fogged out: ${{distant}}`);
        if (!(distant < nearby)) throw new Error('fog density did not adapt to graph extent');
        """
    )


def test_node_scale_keeps_fitted_mobile_nodes_above_two_css_pixels():
    """A full mobile camera fit must not shrink nodes below a visible screen-space radius."""
    assert UTILS.exists(), "3D graph utility module is missing"

    run_node(
        f"""
        const utils = require('{UTILS.as_posix()}');
        const nearby = utils.nodeScaleForFitDistance(1000, 800, 50);
        const mobile = utils.nodeScaleForFitDistance(13130, 546, 50);
        if (nearby !== 1) throw new Error(`nearby graph should keep native scale: ${{nearby}}`);
        if (!(mobile >= 8 && mobile <= 20)) throw new Error(`mobile fit remains sub-pixel: ${{mobile}}`);
        """
    )


def test_node_scale_keeps_fitted_nodes_above_four_css_pixels():
    """Fit must preserve a readable node radius on both desktop and mobile."""
    assert UTILS.exists(), "3D graph utility module is missing"

    run_node(
        f"""
        const utils = require('{UTILS.as_posix()}');
        function screenRadius(distance, height, fov) {{
          const scale = utils.nodeScaleForFitDistance(distance, height, fov);
          const worldUnitsPerPixel = 2 * distance * Math.tan(fov * Math.PI / 360) / height;
          return 5.5 * scale / worldUnitsPerPixel;
        }}
        const desktop = screenRadius(10757, 1164, 50);
        const mobile = screenRadius(13130, 546, 50);
        if (desktop < 4) throw new Error(`desktop radius is only ${{desktop.toFixed(2)}}px`);
        if (mobile < 4) throw new Error(`mobile radius is only ${{mobile.toFixed(2)}}px`);
        """
    )


def test_dormant_nodes_remain_visible_instead_of_being_dimmed_three_times():
    """Most real graphs are dormant-heavy, so dormant styling must remain legible."""
    graph_core = (ROOT / "bdh_graph_harness/visualization/templates/graph-core.js").read_text()
    graph_init = (ROOT / "bdh_graph_harness/visualization/templates/graph-init.js").read_text()

    assert "dormant: '#718096'" in graph_core
    assert "_opacity: isDormant ? 0.84 : 1" in graph_init
    assert "if (node._dormant) opacity *= 0.82" in graph_core
    assert "const baseEmissive = dormant ? 0.24 : 0.12;" in graph_core


def test_neurogenesis_identity_overrides_dormant_visual_classification():
    """Dormant neurogenesis notes must remain cyan diamonds, not gray dormant triangles."""
    graph_core = (ROOT / "bdh_graph_harness/visualization/templates/graph-core.js").read_text()
    graph_init = (ROOT / "bdh_graph_harness/visualization/templates/graph-init.js").read_text()
    controls = (ROOT / "bdh_graph_harness/visualization/templates/ui-controls.js").read_text()

    assert "function isNeurogenesisNode(node)" in graph_core
    assert "if (isNeurogenesis) shape = 'diamond';" in graph_init
    assert "if (isNeurogenesis) color = COLORS.neurogenesis;" in graph_init
    assert "if (isNeurogenesis) node.color = COLORS.neurogenesis;" in controls
    assert "const baseColor = activationColor || (neurogenesis ? COLORS.neurogenesis : (node.color || COLORS.inactive));" in graph_core


def test_overview_edges_keep_a_visible_dark_mode_baseline():
    """Fit/overview must not blend the structural graph into the scene background."""
    graph_core = (ROOT / "bdh_graph_harness/visualization/templates/graph-core.js").read_text()

    assert "edgeWikilink: '#6b7787'" in graph_core
    assert "wikilink: 0.52" in graph_core
    assert "hebbian: 0.32" in graph_core
    assert "phantom: 0.46" in graph_core
    assert "link.type === 'wikilink' ? 0.78 : 0.82" in graph_core


def test_page_bootstraps_the_pinned_3d_renderer_and_real_three_dimensional_layout():
    html = (ROOT / "bdh_graph_harness/visualization/templates/index.html").read_text()
    graph_init = (ROOT / "bdh_graph_harness/visualization/templates/graph-init.js").read_text()

    assert "3d-force-graph@1.80.0" in html
    assert "three@0.180.0/build/three.module.min.js" in html
    assert "graph-3d-utils.js" in html
    assert "force-graph/dist/force-graph.min.js" not in html
    assert "ForceGraph3D" in graph_init
    assert ".numDimensions(3)" in graph_init
    assert ".nodeThreeObject(createNodeThreeObject)" in graph_init
    assert "BDH3DUtils.fogDensityForFitDistance" in graph_init
    assert "BDH3DUtils.nodeScaleForFitDistance" in graph_init
    assert "nodeWorldScale" in (ROOT / "bdh_graph_harness/visualization/templates/graph-core.js").read_text()
    assert ".enableNavigationControls(true)" in graph_init
    assert ".enableNodeDrag(true)" in graph_init
    assert ".nodeCanvasObject(" not in graph_init


def test_neural_cosmetics_keep_particles_and_install_bloom_on_native_composer():
    html = (ROOT / "bdh_graph_harness/visualization/templates/index.html").read_text()
    core = (ROOT / "bdh_graph_harness/visualization/templates/graph-core.js").read_text()
    graph_init = (ROOT / "bdh_graph_harness/visualization/templates/graph-init.js").read_text()

    assert "UnrealBloomPass.js" in html
    assert "window.BDHBloomReady" in html
    assert "postProcessingComposer" in core
    assert "function installBloomPass()" in core
    assert "composer.addPass(bloomPass)" in core
    assert "if (isAmbientFlowLink(link)) return particleConfig.ambientParticles;" in core
    assert "scheduleBloomInstall();" in graph_init
    assert "ambientThreshold: 0.62" in core
    assert "ambientWidth: 1.6" in core
    assert "function hasAmbientParticleFlow()" in graph_init
    assert "|| hasAmbientParticleFlow()" in graph_init
    assert "function directionalParticleObject(link)" in core
    assert "const particle = new T.Mesh(" in core
    assert "particle.add(halo);" in core
    assert "new T.Group()" not in core[core.index("function directionalParticleObject(link)"):core.index("// ============================================================================", core.index("function directionalParticleObject(link)"))]
    assert ".linkDirectionalParticleThreeObject(directionalParticleObject)" in graph_init
    assert "renderer.setClearColor(COLORS.bg, 1)" in graph_init
    assert "graph.scene().background = new window.THREE.Color(COLORS.bg)" in graph_init


def test_pointer_exit_clears_tooltips_and_desktop_rendering_uses_antialiasing_budget():
    core = (ROOT / "bdh_graph_harness/visualization/templates/graph-core.js").read_text()
    graph_init = (ROOT / "bdh_graph_harness/visualization/templates/graph-init.js").read_text()

    assert "sphere: new T.SphereGeometry(1, 20, 14)" in core
    assert "hub: new T.IcosahedronGeometry(1, 2)" in core
    assert ".linkResolution(constrained ? 8 : 12)" in graph_init
    assert ".linkHoverPrecision(constrained ? 10 : 16)" in graph_init
    assert ".linkDirectionalParticleResolution(constrained ? 6 : 8)" in graph_init
    assert "Math.min(window.devicePixelRatio || 1, constrained ? 1.75 : 2)" in graph_init
    assert "antialias: true" in graph_init
    assert "area.addEventListener('pointerleave'" in graph_init
    assert "clearHoverHighlight();\n    hideTooltip();" in graph_init
    assert "canvas.addEventListener('pointerleave', dismissHoverUI" in graph_init
    assert "tooltipEl.style.display = 'none';" in core
    assert "tooltipEl.style.display = 'block';" in core


def test_camera_model_supports_focus_restore_fit_and_orientation_reset():
    graph_init = (ROOT / "bdh_graph_harness/visualization/templates/graph-init.js").read_text()
    controls = (ROOT / "bdh_graph_harness/visualization/templates/ui-controls.js").read_text()
    html = (ROOT / "bdh_graph_harness/visualization/templates/index.html").read_text()

    assert "function captureCameraState()" in graph_init
    assert "target: { x: target.x, y: target.y, z: target.z }" in graph_init
    assert "function restoreCameraState(" in graph_init
    assert "function focusGraphNode(" in controls
    assert "function exitFocusMode(" in controls
    assert "function resetCameraOrientation(" in controls
    assert "graph.cameraPosition(" in controls
    assert "BDH3DUtils.cameraPositionForFocus" in controls
    assert 'id="focus-back-btn"' in html
    assert ".centerAt(" not in controls


def test_initial_graph_mount_does_not_reheat_before_force_layout_exists():
    """3d-force-graph builds its internal layout asynchronously after graphData()."""
    graph_init = (ROOT / "bdh_graph_harness/visualization/templates/graph-init.js").read_text()
    initial_mount = graph_init.split("if (firstGraph) {", 1)[1].split("} else if (preserveView)", 1)[0]

    assert "graph.graphData(data);" in initial_mount
    assert "graphLayoutActive = true;" in initial_mount
    assert "initialFitPending = true;" in initial_mount
    assert "reheatGraphLayout();" not in initial_mount
    assert "setTimeout(() => initialCameraFit()" not in initial_mount
    assert ".onEngineTick(() =>" in graph_init
    assert "scheduleInitialCameraFit();" in graph_init
    assert "graph.getGraphBbox()" in graph_init


def test_initial_fit_enters_balanced_lod_instead_of_staying_artificially_dim():
    graph_init = (ROOT / "bdh_graph_harness/visualization/templates/graph-init.js").read_text()
    initial_fit = graph_init.split("function initialCameraFit() {", 1)[1].split("\n}", 1)[0]

    assert "currentLodLevel = 'balanced';" in initial_fit
    assert initial_fit.index("currentLodLevel = 'balanced';") < initial_fit.index("syncThreeVisualState();")


def test_websocket_connects_before_renderer_promise_can_block_startup():
    """A stalled CDN import must not leave the live connection stuck on Connecting."""
    controls = (ROOT / "bdh_graph_harness/visualization/templates/ui-controls.js").read_text()
    startup = controls.split("async function startVisualization() {", 1)[1].split("\n}\n\nstartVisualization();", 1)[0]

    assert "connectWS();" in startup
    assert "function waitForRendererReady(" in controls
    assert startup.index("connectWS();") < startup.index("await waitForRendererReady(")
    assert "fetchGraphSnapshot({ reason: 'renderer-ready'" in startup


def test_websocket_bootstrap_uses_rich_rest_snapshot_and_polling_fallback():
    websocket = (ROOT / "bdh_graph_harness/visualization/templates/websocket.js").read_text()

    assert "function fetchGraphSnapshot(" in websocket
    assert "vaultApiUrl('/api/graph')" in websocket
    assert "function startPollingFallback(" in websocket
    assert "function stopPollingFallback(" in websocket
    assert "graphPollTimer = setInterval" in websocket
    assert "stopPollingFallback();" in websocket
    assert "startPollingFallback();" in websocket
    assert "fetchGraphSnapshot({ reason: 'websocket-bootstrap'" in websocket


def test_remote_query_response_populates_query_and_response_panel():
    html = (ROOT / "bdh_graph_harness/visualization/templates/index.html").read_text()
    websocket = (ROOT / "bdh_graph_harness/visualization/templates/websocket.js").read_text()

    assert '<textarea' in html
    assert 'id="query-input"' in html
    assert 'rows="5"' in html
    assert "event.type === 'query_response'" in websocket
    assert "function renderRemoteQueryResponse(event)" in websocket
    assert "queryInput.value = event.query || ''" in websocket
    assert "responseText.textContent = event.response" in websocket


def test_activation_updates_visual_state_without_replacing_graph_data():
    activation = (ROOT / "bdh_graph_harness/visualization/templates/activation.js").read_text()

    assert "nodeActivationOpacity.set" in activation
    assert "nodeActivationColor.set" in activation
    assert "linkParticlesState.set" in activation
    assert "setGraphDataPreservingView" in activation
    assert "graph.graphData(currentData)" not in activation
    assert "setGraphDataPreservingView(updated" not in activation


def test_ui_groups_controls_around_a_graph_first_scene_and_mobile_tabs():
    html = (ROOT / "bdh_graph_harness/visualization/templates/index.html").read_text()
    styles = (ROOT / "bdh_graph_harness/visualization/templates/styles.css").read_text()

    for label in ["View", "Sources", "Edges", "Layout", "Appearance"]:
        assert f">{label}<" in html
    assert 'id="control-dock"' in html
    assert 'id="selection-section"' in html
    assert 'id="focus-hud"' in html
    assert 'data-tab="controls-tab"' in html
    assert styles.count("@media (max-width: 768px)") == 1
    assert "body.controls-tab #control-dock" in styles
    assert "min-height: 44px" in styles
    assert "--scene-bg: #070a0f" in styles
    assert "#graph-area" in styles
    assert "prefers-reduced-motion: reduce" in styles
