"""Regression checks for the compact mobile graph visualization."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STYLES = ROOT / "bdh_graph_harness/visualization/templates/styles.css"
CORE = ROOT / "bdh_graph_harness/visualization/templates/graph-core.js"


def test_mobile_layout_uses_one_viewport_grid_without_magic_height_splits():
    """Mobile must reserve real space for header/tabs instead of fixed 65vh graph slices."""
    styles = STYLES.read_text()

    assert styles.count("@media (max-width: 768px)") == 1
    assert "grid-template-rows: auto 48px minmax(0, 1fr)" in styles
    assert "height: 65vh" not in styles
    assert "body.controls-tab #control-dock" in styles
    assert "body.panel-tab #side-panel" in styles


def test_mobile_inspector_clears_persisted_desktop_inline_width():
    """The Inspector tab must fill the viewport instead of inheriting a 31% desktop width."""
    controls = (ROOT / "bdh_graph_harness/visualization/templates/ui-controls.js").read_text()

    assert "function syncPanelWidthForViewport()" in controls
    assert "sidePanel.style.removeProperty('width')" in controls
    assert "syncPanelWidthForViewport();" in controls


def test_mobile_tooltip_is_compact_and_clamped_above_browser_chrome():
    """Touch tooltips should be a short bottom sheet, not a screen-blocking card."""
    core = CORE.read_text()

    assert "const mobile = isMobile();" in core
    assert "const neighborLimit = mobile ? 2 : 6" in core
    assert "tooltipEl.classList.toggle('mobile-tooltip', mobile)" in core
    assert "calc(env(safe-area-inset-bottom) + 12px)" in core


def test_mobile_uses_a_dismissible_node_sheet_instead_of_a_hover_card():
    """A tapped node needs an intentional, shallow detail surface with a close action."""
    styles = STYLES.read_text()
    core = CORE.read_text()

    assert ".graph-tooltip.mobile-tooltip {" in styles
    assert "pointer-events: auto" in styles
    assert ".mobile-sheet-close" in styles
    assert "dismissMobileSheet" in core
    assert "onNodeClick((node, event) =>" in (ROOT / "bdh_graph_harness/visualization/templates/graph-init.js").read_text()


def test_mobile_graph_keeps_controls_compact_and_reserves_canvas_space():
    """The graph is the product on mobile, not a desktop header squeezed into a phone."""
    styles = STYLES.read_text()

    assert "grid-template-rows: auto 48px minmax(0, 1fr)" in styles
    assert "grid-template-columns: repeat(3, 1fr)" in styles
    assert "#mobile-tabs .tab {" in styles
    assert "min-height: 44px" in styles


def test_mobile_mode_and_cached_assets_share_the_mobile_breakpoint_and_revision():
    """Mobile-only interaction must match the CSS breakpoint and bypass stale browser assets."""
    core = CORE.read_text()
    html = (ROOT / "bdh_graph_harness/visualization/templates/index.html").read_text()

    assert "return window.matchMedia('(max-width: 768px)').matches;" in core
    assert "?v=3d-v16" in html
