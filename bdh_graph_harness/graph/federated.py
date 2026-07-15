"""Source-aware Markdown graph construction.

The legacy :func:`build_graph` remains the compatibility path.  This module
provides the deliberately small federated path used when ``external_sources``
is configured.  It emits the same ``nodes``/``edges`` dictionaries consumed by
attention, Hebbian learning, ChromaDB, and the existing API.
"""

from __future__ import annotations

import fnmatch
import os
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Iterable

from bdh_graph_harness.graph.builder import build_graph
from bdh_graph_harness.graph.display import add_display_label
from bdh_graph_harness.graph.parser import extract_text, extract_wikilinks, parse_frontmatter
from bdh_graph_harness.graph.sources import (
    CounterpartSpec,
    Document,
    DocumentSource,
    counterpart_specs_from_config,
    sources_from_config,
)


def _without_md(path: str) -> str:
    path = path.replace(os.sep, "/").lstrip("/")
    return path[:-3] if path.lower().endswith(".md") else path


def _with_md(path: str) -> str:
    path = path.replace(os.sep, "/").lstrip("/")
    return path if path.lower().endswith(".md") else f"{path}.md"


def _canonical_for(source_type: str, source_id: str, relative_path: str) -> str:
    relative = _with_md(relative_path)
    if source_type == "vault":
        return f"vault:{relative}"
    return f"external:{source_id}/{relative}"


def _is_ignored(document: Document, ignore_patterns: Iterable[str]) -> bool:
    """Apply legacy graph-ignore patterns without leaking source namespaces."""
    relative_without_ext = _without_md(document.relative_path)
    candidates = (
        relative_without_ext,
        document.relative_path,
        document.id,
        f"{document.source_id}/{relative_without_ext}",
    )
    return any(
        fnmatch.fnmatch(candidate, pattern)
        for pattern in ignore_patterns
        for candidate in candidates
    )


def _source_key(document: Document, relative_path: str) -> tuple[str, str, str]:
    return document.source_type, document.source_id, _with_md(relative_path)


def _normalise_explicit_target(target: str) -> tuple[str, str, str] | None:
    """Parse ``vault:...`` or ``external:source/...`` link targets."""
    target = target.strip().replace("\\", "/")
    if target.startswith("vault:"):
        relative = target[len("vault:"):]
        return "vault", "vault", _with_md(relative)
    if target.startswith("external:"):
        remainder = target[len("external:"):].lstrip("/")
        source_id, separator, relative = remainder.partition("/")
        if not separator or not source_id or not relative:
            return None
        return "external", source_id, _with_md(relative)
    return None


def _resolve_target(
    document: Document,
    target: str,
    by_source_path: dict[tuple[str, str, str], str],
) -> str | None:
    """Resolve one wikilink against explicit or same-source candidates."""
    # Obsidian supports heading/block suffixes; the graph points at the note.
    target = target.strip().split("#", 1)[0].split("^", 1)[0].strip()
    if not target:
        return None

    explicit = _normalise_explicit_target(target)
    if explicit:
        return by_source_path.get(explicit)

    target = target.lstrip("/")
    parent_path = PurePosixPath(document.relative_path).parent
    candidates = [_with_md(target)]
    ancestor = parent_path
    while str(ancestor) != ".":
        candidates.append(_with_md(str(ancestor / target)))
        ancestor = ancestor.parent
    # Existing vault notes commonly omit the ``wiki/`` namespace. Preserve
    # those explicit structural aliases for the primary vault only; external
    # sources stay strict and never fall back to global/basename matching.
    if document.source_type == "vault":
        candidates.extend(
            _with_md(f"{prefix}{target}")
            for prefix in ("wiki/", "concepts/", "entities/", "comparisons/", "queries/")
        )
    for candidate in candidates:
        if not candidate:
            continue
        resolved = by_source_path.get(
            _source_key(document, candidate)
        )
        if resolved:
            return resolved
    return None


def build_federated_graph(
    sources: Iterable[DocumentSource],
    *,
    graph_ignore: Iterable[str] | None = None,
    counterparts: Iterable[CounterpartSpec] | None = None,
) -> tuple[dict, dict, list[dict]]:
    """Build a federated graph from multiple Markdown sources.

    The function performs a manual deterministic scan.  It intentionally does
    not write cache files to any source; the MVP can add a vault-owned cache
    once the source contract is stable.
    """
    ignore_patterns = tuple(graph_ignore or ())
    documents: list[Document] = []
    for source in sources:
        documents.extend(
            document
            for document in source.scan()
            if not _is_ignored(document, ignore_patterns)
        )

    documents.sort(key=lambda document: document.id)
    nodes: dict[str, dict] = {}
    by_source_path: dict[tuple[str, str, str], str] = {}
    raw_links: dict[str, list[tuple[str, str]]] = {}

    for document in documents:
        frontmatter = parse_frontmatter(document.content)
        node_id = document.id
        nodes[node_id] = {
            "id": node_id,
            "title": frontmatter.get(
                "title",
                os.path.basename(document.relative_path)[:-3],
            ),
            "tags": frontmatter.get("tags", ""),
            "text": extract_text(document.content),
            "path": document.absolute_path,
            "absolute_path": document.absolute_path,
            "relative_path": document.relative_path,
            "source_id": document.source_id,
            "source_type": document.source_type,
            "writable": document.writable,
            "mtime": os.path.getmtime(document.absolute_path),
        }
        by_source_path[
            _source_key(document, document.relative_path)
        ] = node_id
        raw_links[node_id] = extract_wikilinks(document.content)

    edges: dict[str, list[dict]] = defaultdict(list)
    unresolved: list[dict] = []
    documents_by_id = {document.id: document for document in documents}

    for source_id, links in raw_links.items():
        document = documents_by_id[source_id]
        for target, display in links:
            target_id = _resolve_target(document, target, by_source_path)
            if target_id is None:
                unresolved.append({
                    "source": source_id,
                    "target": target,
                    "display": display,
                    "source_path": document.relative_path,
                })
                continue
            edges[source_id].append({
                "target": target_id,
                "display": display,
                "type": "wikilink",
                "weight": 1.0,
                "explicit": bool(_normalise_explicit_target(target)),
            })

    # Counterparts are explicit reciprocal structural links between the two
    # anchor documents that represent the same project. They are not wikilinks
    # and not Hebbian edges; they exist to bridge the two provenance domains.
    for spec in counterparts or ():
        vault_id = by_source_path.get(("vault", "vault", _with_md(spec.vault_path)))
        external_id = by_source_path.get(("external", spec.source_id, _with_md(spec.external_path)))
        if not vault_id or not external_id:
            missing = []
            if not vault_id:
                missing.append(f"vault:{_with_md(spec.vault_path)}")
            if not external_id:
                missing.append(f"external:{spec.source_id}/{_with_md(spec.external_path)}")
            raise ValueError(
                f"Counterpart anchor(s) not found for {spec.group_id!r}: {', '.join(missing)}"
            )
        nodes[vault_id]["project_group"] = spec.group_id
        nodes[external_id]["project_group"] = spec.group_id
        for source_id, target_id in ((vault_id, external_id), (external_id, vault_id)):
            if any(
                edge.get("target") == target_id and edge.get("type") == "counterpart"
                for edge in edges[source_id]
            ):
                continue
            edges[source_id].append({
                "target": target_id,
                "type": "counterpart",
                "relation": "same_project",
                "group_id": spec.group_id,
                "weight": 1.0,
                "explicit": False,
                "generated": True,
                "traversable": True,
            })

    # Derive UI labels after counterpart assignment so explicit project_group
    # context is available to both vault and external anchor documents.
    for document in documents:
        add_display_label(
            nodes[document.id],
            frontmatter=parse_frontmatter(document.content),
        )

    return nodes, dict(edges), unresolved


def migrate_legacy_state_ids(state: dict, nodes: dict) -> dict:
    """Migrate legacy vault-relative IDs into canonical federated vault IDs.

    Enabling the federated path changes ``wiki/foo`` into
    ``vault:wiki/foo.md``.  Existing Hebbian state must follow the notes or it
    becomes a pile of disconnected historical synapses.
    """
    if not nodes or not any(node_id.startswith("vault:") for node_id in nodes):
        return state

    def canonical(note_id: str) -> str:
        if note_id in nodes:
            return note_id
        candidate = f"vault:{_with_md(note_id)}"
        return candidate if candidate in nodes else note_id

    migrated_synapses: dict = {}
    for key, synapse in state.get("synapses", {}).items():
        parts = key.split("|", 1)
        if len(parts) != 2:
            migrated_synapses[key] = synapse
            continue
        mapped_key = "|".join(sorted((canonical(parts[0]), canonical(parts[1]))))
        previous = migrated_synapses.get(mapped_key)
        if previous is None or synapse.get("weight", 0.0) > previous.get("weight", 0.0):
            migrated_synapses[mapped_key] = synapse
    state["synapses"] = migrated_synapses

    quality = state.get("node_quality", {})
    if quality:
        migrated_quality = {}
        for note_id, value in quality.items():
            migrated_quality[canonical(note_id)] = value
        state["node_quality"] = migrated_quality

    if "dormant_nodes" in state:
        state["dormant_nodes"] = sorted({canonical(note_id) for note_id in state["dormant_nodes"]})
    return state


def build_configured_graph(config: dict, *, use_cache: bool = True) -> tuple[dict, dict, list[dict]]:
    """Build either the legacy graph or the configured federated graph."""
    if config.get("external_sources"):
        return build_federated_graph(
            sources_from_config(config),
            graph_ignore=config.get("graph_ignore", []),
            counterparts=counterpart_specs_from_config(config),
        )
    nodes, edges = build_graph(
        config["vault_path"],
        use_cache=use_cache,
        graph_ignore=config.get("graph_ignore"),
    )
    return nodes, edges, []


__all__ = [
    "build_configured_graph",
    "build_federated_graph",
]
