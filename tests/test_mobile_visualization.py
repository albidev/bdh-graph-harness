"""Regression checks for the compact mobile graph visualization."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STYLES = ROOT / "bdh_graph_harness/visualization/templates/styles.css"
CORE = ROOT / "bdh_graph_harness/visualization/templates/graph-core.js"


def test_mobile_layout_uses_one_viewport_grid_without_magic_height_splits():
    """Mobile must reserve real space for header/tabs instead of fixed 65vh graph slices."""
    styles = STYLES.read_text()

    assert styles.count("@media (max-width: 768px)") == 1
    assert "grid-template-rows: auto 36px minmax(0, 1fr)" in styles
    assert "height: 65vh" not in styles
    assert "#stats-dashboard { display: none; }" in styles
    assert "#titlebar h1 { display: none; }" in styles


def test_mobile_tooltip_is_compact_and_clamped_above_browser_chrome():
    """Touch tooltips should be a short bottom sheet, not a screen-blocking card."""
    core = CORE.read_text()

    assert "const mobile = isMobile();" in core
    assert "neighbors.slice(0, mobile ? 2 : 6)" in core
    assert "tooltipEl.classList.toggle('mobile-tooltip', isMobile())" in core
    assert "bottom = 'calc(env(safe-area-inset-bottom) + 12px)'" in core
