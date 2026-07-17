"""Markdown document sources for the federated BDH graph.

A source owns a filesystem root and turns Markdown files into normalized
:class:`Document` objects.  Sources are read-only by default; the graph
builder never writes caches or generated files into an external source.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
DEFAULT_MARKDOWN_INCLUDE = ("**/*.md",)
DEFAULT_EXTERNAL_EXCLUDES = (
    "**/.git/**",
    "**/.hg/**",
    "**/.svn/**",
    "**/node_modules/**",
    "**/.venv/**",
    "**/venv/**",
    "**/__pycache__/**",
    "**/build/**",
    "**/dist/**",
    "**/target/**",
    # Graphify outputs are derived artifacts, not source knowledge.
    "**/graphify-out/**",
)


@dataclass(frozen=True)
class Document:
    """A Markdown document with stable source-aware identity."""

    source_id: str
    source_type: str
    relative_path: str
    absolute_path: str
    content: str
    writable: bool

    @property
    def id(self) -> str:
        """Return the canonical graph ID for this document."""
        relative = self.relative_path.replace(os.sep, "/")
        if self.source_type == "vault":
            return f"vault:{relative}"
        return f"external:{self.source_id}/{relative}"


@dataclass(frozen=True)
class CounterpartSpec:
    """Explicit pair of anchor notes and their semantic relationship."""

    source_id: str
    group_id: str
    vault_path: str
    external_path: str
    relation: str = "same_project"


class DocumentSource(Protocol):
    """Minimal source contract consumed by the federated graph builder."""

    source_id: str
    source_type: str
    root_path: str
    writable: bool

    def iter_paths(self) -> Iterable[tuple[str, str]]:
        ...

    def scan(self) -> Iterable[Document]:
        ...


def validate_source_id(source_id: str) -> str:
    """Validate and return a source ID safe for canonical node IDs."""
    if not source_id or not _SOURCE_ID_RE.match(source_id):
        raise ValueError(
            f"Invalid source id {source_id!r}; expected [A-Za-z0-9_-]+"
        )
    return source_id


def _globstar_match(path_parts: tuple[str, ...], pattern_parts: tuple[str, ...]) -> bool:
    """Match path segments with ``**`` meaning zero or more directories."""
    if not pattern_parts:
        return not path_parts
    if pattern_parts[0] == "**":
        return (
            _globstar_match(path_parts, pattern_parts[1:])
            or bool(path_parts) and _globstar_match(path_parts[1:], pattern_parts)
        )
    if not path_parts or not fnmatch.fnmatchcase(path_parts[0], pattern_parts[0]):
        return False
    return _globstar_match(path_parts[1:], pattern_parts[1:])


def _matches(relative_path: str, pattern: str) -> bool:
    """Match POSIX relative paths with a real zero-or-more ``**`` globstar."""
    relative_path = relative_path.replace(os.sep, "/").strip("/")
    pattern = pattern.replace(os.sep, "/").strip("/")
    if _globstar_match(tuple(relative_path.split("/")), tuple(pattern.split("/"))):
        return True
    try:
        return Path(relative_path).match(pattern)
    except ValueError:
        return False


def _matches_any(relative_path: str, patterns: Iterable[str]) -> bool:
    return any(_matches(relative_path, pattern) for pattern in patterns)


class _MarkdownSource:
    """Shared scanner implementation for vault and external sources."""

    source_type = "external"

    def __init__(
        self,
        root_path: str,
        *,
        source_id: str,
        writable: bool,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
    ) -> None:
        self.source_id = validate_source_id(source_id)
        self.root_path = os.path.abspath(os.path.expanduser(root_path))
        self.writable = bool(writable)
        self.include = tuple(include or DEFAULT_MARKDOWN_INCLUDE)
        self.exclude = tuple(exclude or ())

    def iter_paths(self) -> Iterable[tuple[str, str]]:
        """Yield matching ``(relative_path, absolute_path)`` pairs."""
        if not os.path.isdir(self.root_path):
            raise FileNotFoundError(
                f"{self.source_type.title()} source path not found: {self.root_path}"
            )

        discovered: list[tuple[str, str]] = []
        for root, dirs, files in os.walk(self.root_path):
            # Hidden directories are operational data in practice (.git,
            # .obsidian, caches). They are never part of the Markdown corpus.
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for filename in files:
                if filename.startswith(".") or not filename.lower().endswith(".md"):
                    continue
                absolute = os.path.join(root, filename)
                relative = os.path.relpath(absolute, self.root_path).replace(os.sep, "/")
                if not _matches_any(relative, self.include):
                    continue
                if _matches_any(relative, self.exclude):
                    continue
                discovered.append((relative, absolute))
        yield from sorted(discovered)

    def scan(self) -> Iterable[Document]:
        """Yield matching Markdown documents in deterministic path order."""
        for relative, absolute in self.iter_paths():
            with open(absolute, "r", encoding="utf-8") as handle:
                content = handle.read()
            yield Document(
                source_id=self.source_id,
                source_type=self.source_type,
                relative_path=relative,
                absolute_path=absolute,
                content=content,
                writable=self.writable,
            )


class VaultMarkdownSource(_MarkdownSource):
    """The writable primary vault source."""

    source_type = "vault"

    def __init__(
        self,
        root_path: str,
        *,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
    ) -> None:
        super().__init__(
            root_path,
            source_id="vault",
            writable=True,
            include=include,
            exclude=exclude,
        )


class ExternalMarkdownSource(_MarkdownSource):
    """A configurable, normally read-only external Markdown source."""

    source_type = "external"

    def __init__(
        self,
        root_path: str,
        *,
        source_id: str,
        writable: bool = False,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
    ) -> None:
        super().__init__(
            root_path,
            source_id=source_id,
            writable=writable,
            include=include,
            exclude=tuple(DEFAULT_EXTERNAL_EXCLUDES) + tuple(exclude or ()),
        )


def sources_from_config(config: dict) -> list[DocumentSource]:
    """Create the configured vault + external sources for one VaultContext."""
    sources: list[DocumentSource] = [
        VaultMarkdownSource(
            config["vault_path"],
            include=config.get("vault_include"),
            exclude=config.get("vault_exclude"),
        )
    ]
    seen_ids = {"vault"}

    for entry in config.get("external_sources", []) or []:
        if not isinstance(entry, dict):
            raise ValueError(f"external_sources entries must be mappings: {entry!r}")
        source_id = entry.get("id")
        if not source_id:
            raise ValueError(f"External source entry missing 'id': {entry!r}")
        source_id = validate_source_id(source_id)
        if source_id in seen_ids:
            raise ValueError(f"Duplicate document source id: {source_id!r}")
        seen_ids.add(source_id)
        path = entry.get("path")
        if not path:
            raise ValueError(f"External source {source_id!r} missing 'path'")
        sources.append(
            ExternalMarkdownSource(
                path,
                source_id=source_id,
                writable=bool(entry.get("writable", False)),
                include=entry.get("include") or entry.get("include_globs"),
                exclude=entry.get("exclude") or entry.get("exclude_globs"),
            )
        )
    return sources


def counterpart_specs_from_config(config: dict) -> list[CounterpartSpec]:
    """Read explicit vault/external anchor pairs from external source config."""
    specs: list[CounterpartSpec] = []
    for entry in config.get("external_sources", []) or []:
        if not isinstance(entry, dict):
            continue
        source_id = validate_source_id(entry.get("id", ""))
        raw_specs = entry.get("counterparts")
        if raw_specs is None and entry.get("counterpart") is not None:
            raw_specs = [entry["counterpart"]]
        for raw in raw_specs or []:
            if not isinstance(raw, dict):
                raise ValueError(f"Counterpart entries must be mappings: {raw!r}")
            group_id = raw.get("group_id") or entry.get("project_group")
            vault_path = raw.get("vault_path")
            external_path = raw.get("external_path")
            if not group_id or not vault_path or not external_path:
                raise ValueError(
                    "Counterpart requires 'group_id', 'vault_path', and 'external_path'"
                )
            relation = raw.get("relation", "same_project")
            if relation not in {"same_project", "references_project"}:
                raise ValueError(
                    f"Unsupported counterpart relation {relation!r}; "
                    "expected 'same_project' or 'references_project'"
                )
            specs.append(CounterpartSpec(
                source_id=source_id,
                group_id=validate_source_id(group_id),
                vault_path=_with_posix_path(vault_path),
                external_path=_with_posix_path(external_path),
                relation=relation,
            ))
    return specs


def _with_posix_path(path: str) -> str:
    """Normalize a configured relative path without requiring a graph module."""
    path = str(path).replace("\\", "/").lstrip("/")
    return path if path.lower().endswith(".md") else f"{path}.md"


__all__ = [
    "Document",
    "DocumentSource",
    "CounterpartSpec",
    "ExternalMarkdownSource",
    "VaultMarkdownSource",
    "sources_from_config",
    "counterpart_specs_from_config",
    "validate_source_id",
]
