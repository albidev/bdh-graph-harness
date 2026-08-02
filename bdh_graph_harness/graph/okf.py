"""Small OKF v0.2 compatibility helpers.

The OKF format deliberately stays a Markdown/YAML convention.  This module
contains only the syntax-level adapter; BDH runtime state (Hebbian synapses,
embeddings, telemetry, and consolidation state) remains outside OKF documents.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

import yaml

from bdh_graph_harness.graph.parser import FRONTMATTER_RE


_RESERVED_FILENAMES = {"index.md", "log.md"}

# Markdown links, excluding image links.  Code is removed before applying this
# regex so examples in documentation do not become graph edges.
_MARKDOWN_LINK_RE = re.compile(
    r'(?<!!)\[([^\]\n]+)\]\(\s*(<[^>\n]+>|[^)\s]+)'
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*[)]"
)


def parse_okf_frontmatter(content: str) -> dict:
    """Parse a YAML frontmatter block using OKF's typed representation.

    Missing frontmatter and non-mapping YAML are treated as an empty mapping so
    a mixed legacy/OKF corpus can still be scanned.  A bare ``verified``
    mapping is normalized to the list form required by OKF consumers.
    """
    match = FRONTMATTER_RE.match(content)
    if not match:
        return {}

    parsed = yaml.safe_load(match.group(1))
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        return {}

    if isinstance(parsed.get("verified"), dict):
        parsed["verified"] = [parsed["verified"]]
    return _json_safe(parsed)


def _json_safe(value: Any) -> Any:
    """Keep YAML metadata directly serializable by BDH's JSON API."""
    if isinstance(value, datetime):
        normalized = value
        if normalized.tzinfo is not None:
            normalized = normalized.astimezone(timezone.utc)
        result = normalized.isoformat()
        return result.replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _without_code(content: str) -> str:
    """Remove fenced and inline code before link extraction."""
    content = re.sub(
        r"(?ms)^\s*(```|~~~).*?^\s*\1\s*$",
        "",
        content,
    )
    return re.sub(r"`[^`\n]*`", "", content)


def _is_local_target(target: str) -> bool:
    """Return whether a Markdown target can identify a bundle-local concept."""
    if not target or target.startswith(("#", "//")):
        return False
    parsed = urlparse(target)
    # Relative paths and root-relative paths have no URI scheme.  This filters
    # http(s), mailto, tel, data, and other external URI forms.
    return not parsed.scheme


def extract_markdown_links(content: str) -> list[tuple[str, str]]:
    """Extract local standard Markdown links as ``(target, display)`` pairs."""
    links: list[tuple[str, str]] = []
    for match in _MARKDOWN_LINK_RE.finditer(_without_code(content)):
        display = match.group(1).strip()
        target = match.group(2).strip()
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        if _is_local_target(target):
            links.append((target, display))
    return links


def is_reserved_filename(relative_path: str) -> bool:
    """Return whether a path names an OKF reserved file at any depth."""
    filename = PurePosixPath(relative_path.replace("\\", "/")).name.lower()
    return filename in _RESERVED_FILENAMES


__all__ = [
    "extract_markdown_links",
    "is_reserved_filename",
    "parse_okf_frontmatter",
]
