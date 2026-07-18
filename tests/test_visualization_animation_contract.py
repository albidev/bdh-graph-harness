from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "bdh_graph_harness" / "visualization" / "templates"


def read_template(name: str) -> str:
    return (TEMPLATES / name).read_text()


def test_query_has_one_activation_owner():
    websocket = read_template("websocket.js")

    # The WebSocket event is the single owner of graph activation rendering.
    # The HTTP response may update text/stats, but must not replay the same event.
    assert "handleActivation(data)" not in websocket
    assert "data.activated_notes && !wsReady" in websocket
    assert "event.type === 'activation' || event.type === 'neurogenesis'" in websocket


def test_query_cannot_overlap_while_button_is_disabled():
    websocket = read_template("websocket.js")

    assert "if (!query || button.disabled) return" in websocket


def test_renderer_keeps_idle_pause_and_has_no_second_collision_loop():
    graph_init = read_template("graph-init.js")
    graph_core = read_template("graph-core.js")

    assert "let graphRenderPaused = false" in graph_core
    assert "graph.pauseAnimation()" in graph_init
    assert "graph.resumeAnimation()" in graph_init
    assert ".onEngineStop(() => {" in graph_init
    assert "graphLayoutActive = false" in graph_init
    assert "setInterval(" not in graph_init
    assert "window.__collisionForceInstalled" not in graph_init


def test_renderer_clears_paused_state_before_synchronous_resume():
    """resumeAnimation may emit a camera change synchronously; the guard must already be open."""
    graph_init = read_template("graph-init.js")
    resume_body = graph_init.split("function resumeGraphRendering() {", 1)[1].split("\n}", 1)[0]

    assert resume_body.index("graphRenderPaused = false") < resume_body.index("graph.resumeAnimation()")


def test_initial_tag_legend_waits_for_force_layout_instead_of_starting_a_broken_tick():
    graph_init = read_template("graph-init.js")
    initial_mount = graph_init.split("if (firstGraph) {", 1)[1].split("} else if (preserveView)", 1)[0]

    assert "graphLayoutActive = true" in initial_mount
    assert "initialFitPending = true" in initial_mount
    assert "reheatGraphLayout();" not in initial_mount
    assert "toggleTagColors(true, false)" in graph_init


def test_activation_invalidates_stale_animation_callbacks():
    activation = read_template("activation.js")

    assert "let activationGeneration = 0" in activation
    assert "++activationGeneration" in activation
    assert "const isCurrent = () => generation === activationGeneration" in activation
    assert "if (!isCurrent()) return" in activation


def test_lod_preserves_semantic_node_shapes():
    core = read_template("graph-core.js")

    assert "function geometryForNode(node)" in core
    assert "node._shape === 'diamond'" in core
    assert "node._shape === 'hexagon'" in core
    assert "node._shape === 'triangleDown'" in core
    assert "currentLodLevel === 'overview'" in core
    assert "HEBBIAN_OVERVIEW_WEIGHT" in core


def test_neurogenesis_does_not_rebuild_graph_for_each_birth_pulse():
    activation = read_template("activation.js")

    assert "setGraphDataPreservingView(updated, { reheat: true })" not in activation
