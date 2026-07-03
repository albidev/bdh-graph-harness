"""Visualization module — renders the self-contained vis.js graph page."""

import os

__all__ = ["render_viz_html", "get_template_path"]


def get_template_path() -> str:
    """Return the absolute path to the viz HTML template."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "templates", "index.html")


def render_viz_html(neuron_count: int, synapse_count: int, hebbian_count: int) -> str:
    """Load the HTML template and substitute graph statistics.

    Replaces the legacy ``_viz_html()`` f-string from harness.py.

    Uses ``str.replace`` rather than ``str.format`` so the HTML template can
    contain regular single braces in CSS/JS without needing to double-escape
    them.  Only the three placeholders ``{neuron_count}``,
    ``{synapse_count}``, and ``{hebbian_count}`` are substituted.
    """
    with open(get_template_path(), "r", encoding="utf-8") as f:
        template = f.read()
    return (
        template
        .replace("{neuron_count}", str(neuron_count))
        .replace("{synapse_count}", str(synapse_count))
        .replace("{hebbian_count}", str(hebbian_count))
    )