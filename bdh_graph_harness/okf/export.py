"""OKF v0.2 bundle validation and export for BDH.

The exporter is intentionally a document-layer adapter. It exports static
concept metadata and structural links, while excluding BDH runtime state such
as Hebbian weights, phantom edges, embeddings, activation history, and local
filesystem paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_type
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import yaml

from bdh_graph_harness.graph.okf import (
    extract_markdown_links,
    is_reserved_filename,
)
from bdh_graph_harness.graph.parser import FRONTMATTER_RE


_OKF_VERSION = "0.2"
_RESERVED_INDEX = "index.md"
_RESERVED_LOG = "log.md"
_DYNAMIC_EDGE_TYPES = {
    "hebbian",
    "phantom",
    "learned",
    "learned_only",
    "semantic",
    "attention",
    "neurogenesis_source",
}
_DATE_HEADING_RE = re.compile(r"^\s*#{2,6}\s+(.+?)\s*$", re.MULTILINE)
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Only these source-level fields are copied. Concept-level fields are selected
# explicitly in _metadata_for_node below.
_SAFE_SOURCE_FIELDS = {
    "id",
    "resource",
    "title",
    "author",
    "usage_count",
    "last_modified",
    "usage_window",
}


@dataclass(frozen=True)
class OKFIssue:
    """One deterministic validator diagnostic."""

    path: str
    code: str
    message: str
    severity: str = "error"


@dataclass
class OKFValidationResult:
    """Validation result for an OKF bundle."""

    root: Path
    errors: list[OKFIssue] = field(default_factory=list)
    warnings: list[OKFIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass
class OKFExportResult:
    """Files and non-fatal redactions produced by an export."""

    destination: Path
    files: list[str]
    warnings: list[str] = field(default_factory=list)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _frontmatter(content: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return parsed frontmatter and a parse/missing error code."""
    match = FRONTMATTER_RE.match(content)
    if not match:
        return None, "missing_frontmatter"
    try:
        parsed = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None, "invalid_frontmatter_yaml"
    if not isinstance(parsed, dict):
        return None, "frontmatter_not_mapping"
    return parsed, None


def _body(content: str) -> str:
    match = FRONTMATTER_RE.match(content)
    return content[match.end() :] if match else content


def _issue(
    issues: list[OKFIssue],
    path: str,
    code: str,
    message: str,
    *,
    severity: str = "error",
) -> None:
    issues.append(OKFIssue(path=path, code=code, message=message, severity=severity))


def _validate_concept(path: str, content: str, errors: list[OKFIssue]) -> None:
    metadata, error = _frontmatter(content)
    if error:
        _issue(errors, path, error, "Concept documents require a YAML frontmatter block.")
        return
    assert metadata is not None

    concept_type = metadata.get("type")
    if not isinstance(concept_type, str) or not concept_type.strip():
        _issue(
            errors,
            path,
            "missing_frontmatter_type",
            "Every non-reserved OKF document requires a non-empty type field.",
        )

    sources = metadata.get("sources")
    if sources is not None:
        if isinstance(sources, Mapping):
            sources = [sources]
        if not isinstance(sources, list):
            _issue(errors, path, "sources_not_list", "sources must be a YAML list.")
        else:
            for index, source in enumerate(sources):
                if not isinstance(source, Mapping):
                    _issue(
                        errors,
                        path,
                        "source_not_mapping",
                        f"sources[{index}] must be a YAML mapping.",
                    )
                    continue
                resource = source.get("resource")
                if not isinstance(resource, str) or not resource.strip():
                    _issue(
                        errors,
                        path,
                        "source_missing_resource",
                        f"sources[{index}] requires a non-empty resource.",
                    )

    generated = metadata.get("generated")
    if generated is not None:
        if not isinstance(generated, Mapping):
            _issue(errors, path, "generated_not_mapping", "generated must be a YAML mapping.")
        elif not isinstance(generated.get("by"), str) or not str(generated["by"]).strip():
            _issue(errors, path, "generated_missing_actor", "generated.by must be non-empty.")

    verified = metadata.get("verified")
    if isinstance(verified, Mapping):
        verified = [verified]
    if verified is not None:
        if not isinstance(verified, list):
            _issue(errors, path, "verified_not_list", "verified must be a list or mapping.")
        else:
            for index, event in enumerate(verified):
                if not isinstance(event, Mapping):
                    _issue(
                        errors,
                        path,
                        "verified_event_not_mapping",
                        f"verified[{index}] must be a YAML mapping.",
                    )
                    continue
                if not isinstance(event.get("by"), str) or not str(event["by"]).strip():
                    _issue(
                        errors,
                        path,
                        "verified_missing_actor",
                        f"verified[{index}].by must be non-empty.",
                    )

    stale_after = metadata.get("stale_after")
    if stale_after is not None:
        value = _scalar_string(stale_after)
        if not value or not _valid_iso_date(value):
            _issue(
                errors,
                path,
                "invalid_stale_after",
                "stale_after must be an ISO YYYY-MM-DD date.",
            )


def _validate_index(path: str, content: str, root_index: bool, errors: list[OKFIssue]) -> None:
    metadata, error = _frontmatter(content)
    if error == "missing_frontmatter":
        metadata = None
    elif error:
        _issue(errors, path, error, "index.md frontmatter is not valid YAML.")
        return

    if metadata is not None:
        if not root_index or set(metadata) - {"okf_version"}:
            _issue(
                errors,
                path,
                "reserved_frontmatter",
                "index.md cannot carry concept frontmatter; only root okf_version is allowed.",
            )
        if root_index and "okf_version" in metadata:
            version = _scalar_string(metadata["okf_version"])
            if version != _OKF_VERSION:
                _issue(
                    errors,
                    path,
                    "unsupported_okf_version",
                    f"Expected OKF version {_OKF_VERSION}, got {version!r}.",
                )

    if not extract_markdown_links(_body(content)):
        _issue(
            errors,
            path,
            "index_missing_links",
            "An index.md file should contain at least one Markdown directory entry.",
        )


def _validate_log(path: str, content: str, errors: list[OKFIssue]) -> None:
    if FRONTMATTER_RE.match(content):
        _issue(errors, path, "reserved_frontmatter", "log.md must not contain frontmatter.")

    dates: list[date_type] = []
    for match in _DATE_HEADING_RE.finditer(content):
        heading = match.group(1).strip()
        if not _valid_iso_date(heading):
            _issue(
                errors,
                path,
                "log_invalid_date_heading",
                f"Log heading {heading!r} is not YYYY-MM-DD.",
            )
            continue
        dates.append(date_type.fromisoformat(heading))

    if not dates:
        _issue(
            errors,
            path,
            "log_missing_date_heading",
            "log.md requires at least one ISO date heading.",
        )
        return

    if any(previous < current for previous, current in zip(dates, dates[1:])):
        _issue(
            errors,
            path,
            "log_dates_not_descending",
            "log.md date sections must be ordered newest first.",
        )


def validate_okf_bundle(bundle_root: str | os.PathLike[str]) -> OKFValidationResult:
    """Validate the hard OKF v0.2 conformance requirements.

    Broken concept links and unknown type values are intentionally tolerated;
    OKF treats both as valid consumer-side situations.
    """
    root = Path(bundle_root).expanduser().resolve()
    result = OKFValidationResult(root=root)
    if not root.exists() or not root.is_dir():
        _issue(result.errors, ".", "bundle_missing", f"Bundle directory does not exist: {root}")
        return result

    markdown_files = sorted(
        path
        for path in root.rglob("*.md")
        if not any(part.startswith(".") for part in path.relative_to(root).parts)
    )
    for path in markdown_files:
        relative = _relative(path, root)
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            _issue(result.errors, relative, "invalid_utf8", "OKF documents must be UTF-8 Markdown.")
            continue

        if is_reserved_filename(relative):
            filename = PurePosixPath(relative).name.lower()
            if filename == _RESERVED_INDEX:
                _validate_index(relative, content, relative == _RESERVED_INDEX, result.errors)
            else:
                _validate_log(relative, content, result.errors)
        else:
            _validate_concept(relative, content, result.errors)

    return result


def _scalar_string(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (date_type, datetime)):
        return value.isoformat()
    return None


def _valid_iso_date(value: str | None) -> bool:
    if not value or not _ISO_DATE_RE.fullmatch(value):
        return False
    try:
        date_type.fromisoformat(value)
    except ValueError:
        return False
    return True


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        normalized = value.astimezone(timezone.utc) if value.tzinfo else value
        return normalized.isoformat().replace("+00:00", "Z")
    if isinstance(value, date_type):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _node_path(node_id: str) -> str:
    original = str(node_id)
    value = original.replace("\\", "/")
    if value.startswith(("/", "~")) or re.match(r"^[A-Za-z]:/", value):
        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:16]
        return f"concepts/redacted-{digest}.md"
    if ":" in value and not re.match(r"^[A-Za-z]:/", value):
        prefix, remainder = value.split(":", 1)
        if remainder.startswith(("/", "~")):
            digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:16]
            return f"concepts/redacted-{digest}.md"
        value = remainder if prefix == "vault" else f"{prefix}/{remainder}"
    value = value.lstrip("/")
    if value.lower().endswith(".md"):
        value = value[:-3]
    normalized = posixpath.normpath(value)
    if normalized in {"", ".", ".."} or normalized.startswith("../"):
        digest = hashlib.sha256(str(node_id).encode("utf-8")).hexdigest()[:16]
        normalized = f"concepts/redacted-{digest}"
    return f"{normalized}.md"


def _normalise_tags(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        values = value
    elif isinstance(value, str):
        parsed: Any = None
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError:
            parsed = None
        if isinstance(parsed, list):
            values = parsed
        else:
            values = [part.strip() for part in value.split(",")]
    else:
        return None
    tags = [str(tag).strip() for tag in values if str(tag).strip()]
    return tags or None


def _safe_actor(value: Any) -> str | None:
    actor = _scalar_string(value)
    if not actor or "\n" in actor or "\r" in actor:
        return None
    return actor


def _redacted_resource(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"redacted://local/{digest}"


def _safe_resource(value: Any, known_paths: set[str], warnings: list[str]) -> str | None:
    raw = _scalar_string(value)
    if not raw:
        return None
    raw = raw.replace("\\", "/")
    parsed = urlparse(raw)
    if parsed.scheme:
        if parsed.scheme.lower() in {"http", "https"}:
            return raw
        warnings.append(f"Redacted non-web resource URI: {parsed.scheme}:...")
        return _redacted_resource(raw)

    if raw.startswith("/"):
        candidate = posixpath.normpath(raw.lstrip("/"))
        if candidate in known_paths:
            return f"/{candidate}"
        warnings.append(f"Redacted local filesystem resource: {raw}")
        return _redacted_resource(raw)
    if raw.startswith("~") or re.match(r"^[A-Za-z]:/", raw):
        warnings.append(f"Redacted local filesystem resource: {raw}")
        return _redacted_resource(raw)

    candidate = posixpath.normpath(raw)
    if candidate in {".", ".."} or candidate.startswith("../"):
        warnings.append(f"Redacted path-traversal resource: {raw}")
        return _redacted_resource(raw)
    if candidate in known_paths:
        return candidate
    return candidate


def _safe_event(value: Any) -> dict[str, str] | None:
    if not isinstance(value, Mapping):
        return None
    actor = _safe_actor(value.get("by"))
    at = _scalar_string(value.get("at"))
    if not actor or not at:
        return None
    return {"by": actor, "at": at}


def _safe_generated(value: Any, generated_at: str) -> dict[str, str]:
    event = _safe_event(value)
    if event is None:
        return {"by": "process:bdh-okf-export", "at": generated_at}
    return event


def _safe_verified(value: Any) -> list[dict[str, str]] | None:
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, list):
        return None
    events = [event for item in value if (event := _safe_event(item)) is not None]
    return events or None


def _safe_usage_window(value: Any) -> dict[str, str] | None:
    if not isinstance(value, Mapping):
        return None
    start = _scalar_string(value.get("from"))
    end = _scalar_string(value.get("to"))
    if not start or not end:
        return None
    return {"from": start, "to": end}


def _safe_sources(
    value: Any,
    known_paths: set[str],
    warnings: list[str],
) -> list[dict[str, Any]] | None:
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, list):
        return None

    sources: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            warnings.append("Dropped non-mapping provenance source.")
            continue
        resource = _safe_resource(item.get("resource"), known_paths, warnings)
        if not resource:
            warnings.append("Dropped provenance source without a resource.")
            continue
        source: dict[str, Any] = {"resource": resource}
        for key in _SAFE_SOURCE_FIELDS - {"resource", "usage_window"}:
            if key not in item:
                continue
            if key == "usage_count":
                if isinstance(item[key], (int, float)) and not isinstance(item[key], bool):
                    source[key] = item[key]
            else:
                scalar = _scalar_string(item[key])
                if scalar:
                    source[key] = scalar
        window = _safe_usage_window(item.get("usage_window"))
        if window:
            source["usage_window"] = window
        sources.append(source)
    return sources or None


def _metadata_for_node(
    node: Mapping[str, Any],
    known_paths: set[str],
    generated_at: str,
    warnings: list[str],
) -> dict[str, Any]:
    raw_okf = node.get("okf")
    okf = dict(raw_okf) if isinstance(raw_okf, Mapping) else {}
    metadata: dict[str, Any] = {}

    concept_type = _scalar_string(okf.get("type")) or _scalar_string(node.get("type")) or "Concept"
    metadata["type"] = concept_type

    for key in ("title", "description"):
        value = _scalar_string(okf.get(key)) or _scalar_string(node.get(key))
        if value:
            metadata[key] = value

    resource = _safe_resource(okf.get("resource"), known_paths, warnings)
    if resource:
        metadata["resource"] = resource

    tags = _normalise_tags(okf.get("tags", node.get("tags")))
    if tags:
        metadata["tags"] = tags

    metadata["generated"] = _safe_generated(okf.get("generated"), generated_at)

    verified = _safe_verified(okf.get("verified"))
    if verified:
        metadata["verified"] = verified

    sources = _safe_sources(okf.get("sources"), known_paths, warnings)
    if sources:
        metadata["sources"] = sources

    usage_window = _safe_usage_window(okf.get("usage_window"))
    if usage_window:
        metadata["usage_window"] = usage_window

    status = _scalar_string(okf.get("status"))
    if status:
        metadata["status"] = status

    stale_after = _scalar_string(okf.get("stale_after"))
    if stale_after:
        metadata["stale_after"] = stale_after

    return _json_safe(metadata)


def _edge_parts(edge: Any) -> tuple[str | None, str | None, str | None]:
    if isinstance(edge, Mapping):
        target = _scalar_string(edge.get("target"))
        display = _scalar_string(edge.get("display"))
        edge_type = _scalar_string(edge.get("type"))
        return target, display, edge_type
    if isinstance(edge, (tuple, list)) and edge:
        target = _scalar_string(edge[0])
        display = _scalar_string(edge[1]) if len(edge) > 1 else None
        return target, display, None
    return None, None, None


def _is_dynamic_edge(edge: Any) -> bool:
    if not isinstance(edge, Mapping):
        return False
    edge_type = _scalar_string(edge.get("type"))
    relation = _scalar_string(edge.get("relation"))
    return (edge_type or "").lower() in _DYNAMIC_EDGE_TYPES or (relation or "").lower() in _DYNAMIC_EDGE_TYPES


def _render_body(
    node_id: str,
    node: Mapping[str, Any],
    edges: Mapping[str, Any],
    id_to_path: Mapping[str, str],
) -> str:
    body = _scalar_string(node.get("text")) or _scalar_string(node.get("body")) or ""
    body = body.strip()
    related: list[str] = []
    seen_targets: set[str] = set()
    current_path = id_to_path[node_id]
    current_parent = posixpath.dirname(current_path)

    for raw_edge in edges.get(node_id, []) if isinstance(edges, Mapping) else []:
        if _is_dynamic_edge(raw_edge):
            continue
        target, display, _ = _edge_parts(raw_edge)
        if not target or target not in id_to_path or target in seen_targets:
            continue
        seen_targets.add(target)
        target_path = id_to_path[target]
        link = posixpath.relpath(target_path, current_parent or ".")
        label = display or Path(target_path).stem.replace("-", " ").title()
        related.append(f"- [{label}]({link})")

    if related:
        if body:
            body += "\n\n"
        body += "## Related concepts\n\n" + "\n".join(related)
    return body.rstrip() + "\n"


def _render_index(
    id_to_path: Mapping[str, str],
    nodes: Mapping[str, Mapping[str, Any]],
) -> str:
    lines = ["---", 'okf_version: "0.2"', "---", "# Knowledge Bundle", "", "## Concepts", ""]
    for node_id in sorted(id_to_path, key=lambda item: id_to_path[item]):
        path = id_to_path[node_id]
        node = nodes[node_id]
        title = _scalar_string(node.get("title")) or Path(path).stem.replace("-", " ").title()
        raw_okf = node.get("okf")
        okf: Mapping[str, Any] = dict(raw_okf) if isinstance(raw_okf, Mapping) else {}
        description = _scalar_string(okf.get("description")) or _scalar_string(node.get("description"))
        suffix = f" - {description}" if description else ""
        lines.append(f"* [{title}]({path}){suffix}")
    return "\n".join(lines).rstrip() + "\n"


def _render_log(
    log_entries: Sequence[Mapping[str, Any]] | None,
    generated_at: str,
) -> str:
    entries = list(log_entries or [{"date": generated_at[:10], "kind": "Export", "message": "Created OKF bundle."}])
    normalised: list[tuple[str, str, str]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("log_entries must contain mappings")
        entry_date = _scalar_string(entry.get("date"))
        if not _valid_iso_date(entry_date):
            raise ValueError(f"Invalid OKF log entry date: {entry_date!r}")
        assert entry_date is not None
        kind = _scalar_string(entry.get("kind")) or "Update"
        message = _scalar_string(entry.get("message")) or "Updated bundle."
        normalised.append((entry_date, kind, message))

    lines = ["# Directory Update Log", ""]
    for entry_date, kind, message in sorted(normalised, key=lambda item: item[0], reverse=True):
        lines.extend([f"## {entry_date}", f"* **{kind}**: {message}", ""])
    return "\n".join(lines).rstrip() + "\n"


def export_okf_bundle(
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Mapping[str, Any],
    destination: str | os.PathLike[str],
    *,
    generated_at: str | datetime | None = None,
    log_entries: Sequence[Mapping[str, Any]] | None = None,
) -> OKFExportResult:
    """Export static BDH graph knowledge as a sanitized OKF bundle.

    Only concept documents and structural links are exported. The destination
    is created if needed; existing generated files are overwritten, but the
    exporter never recursively deletes unrelated user files.
    """
    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    generated_value = _scalar_string(generated_at) if generated_at is not None else None
    if not generated_value:
        generated_value = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    id_to_path: dict[str, str] = {}
    path_to_id: dict[str, str] = {}
    for node_id in sorted(nodes):
        if is_reserved_filename(f"{node_id}.md"):
            continue
        relative = _node_path(node_id)
        previous = path_to_id.get(relative)
        if previous is not None and previous != node_id:
            raise ValueError(f"Node path collision: {previous!r} and {node_id!r} -> {relative}")
        id_to_path[node_id] = relative
        path_to_id[relative] = node_id

    known_paths = set(id_to_path.values())
    files: list[str] = []
    for node_id in sorted(id_to_path, key=lambda item: id_to_path[item]):
        node = nodes[node_id]
        relative = id_to_path[node_id]
        metadata = _metadata_for_node(node, known_paths, generated_value, warnings)
        content = "---\n" + yaml.safe_dump(
            metadata,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ) + "---\n\n"
        content += _render_body(node_id, node, edges, id_to_path)
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        files.append(relative)

    (root / _RESERVED_INDEX).write_text(_render_index(id_to_path, nodes), encoding="utf-8")
    files.append(_RESERVED_INDEX)
    (root / _RESERVED_LOG).write_text(_render_log(log_entries, generated_value), encoding="utf-8")
    files.append(_RESERVED_LOG)

    return OKFExportResult(destination=root, files=files, warnings=warnings)


__all__ = [
    "OKFIssue",
    "OKFExportResult",
    "OKFValidationResult",
    "export_okf_bundle",
    "validate_okf_bundle",
]
