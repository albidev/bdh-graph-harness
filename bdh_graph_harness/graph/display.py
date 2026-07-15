"""Human-readable node labels for the federated graph.

Graph identity and semantic titles stay untouched.  This module only derives
an optional contextual label for UI surfaces such as the force graph tooltip.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Mapping


_CONTEXT_KEYS = ("project", "project_name", "project_group")
_ACRONYMS = {
    "ai", "api", "bdh", "cd", "ci", "cpu", "gpu", "glm", "http",
    "https", "llm", "mcp", "mlx", "pr", "tui", "ui",
}


def _pretty_context(value: object) -> str:
    """Turn a project slug into a compact human-readable label."""
    text = str(value or "").strip().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return " ".join(
        part.upper() if part.casefold() in _ACRONYMS else part.capitalize()
        for part in text.split()
    ) if text else ""


def infer_context_label(
    *,
    source_type: str = "vault",
    source_id: str = "vault",
    relative_path: str = "",
    project_group: str | None = None,
    frontmatter: Mapping[str, object] | None = None,
) -> str | None:
    """Infer a project context without changing graph identity.

    Explicit metadata wins.  External sources fall back to their first path
    segment (or source id for a single-repository source).  Vault context is
    inferred only from an explicit project field/group or ``projects/<name>``
    paths; generic wiki concepts remain unscoped.
    """
    metadata = frontmatter or {}
    for key in _CONTEXT_KEYS:
        value = project_group if key == "project_group" else metadata.get(key)
        label = _pretty_context(value)
        if label:
            return label

    path = str(relative_path or "").replace("\\", "/").strip("/")
    parts = PurePosixPath(path).parts if path else ()

    if source_type == "external":
        candidate = parts[0] if len(parts) > 1 else source_id
        label = _pretty_context(candidate)
        return label or None

    if len(parts) >= 3 and parts[0].lower() == "projects":
        label = _pretty_context(parts[1])
        return label or None

    return None


def compose_display_label(title: object, context_label: str | None = None) -> str:
    """Return ``context — title`` while preserving the original title."""
    title_text = str(title or "").strip()
    context = str(context_label or "").strip()
    if not title_text:
        title_text = "Untitled"
    if not context or context.casefold() == title_text.casefold():
        return title_text
    return f"{context} — {title_text}"


def add_display_label(
    node: dict,
    *,
    frontmatter: Mapping[str, object] | None = None,
) -> dict:
    """Add UI metadata to a node in place and return it."""
    context = infer_context_label(
        source_type=node.get("source_type", "vault"),
        source_id=node.get("source_id", "vault"),
        relative_path=node.get("relative_path", node.get("id", "")),
        project_group=node.get("project_group"),
        frontmatter=frontmatter,
    )
    if context:
        node["context_label"] = context
    else:
        node.pop("context_label", None)
    node["display_label"] = compose_display_label(node.get("title"), context)
    return node


__all__ = [
    "add_display_label",
    "compose_display_label",
    "infer_context_label",
]
