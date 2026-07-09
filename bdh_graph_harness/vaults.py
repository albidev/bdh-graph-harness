"""Vault abstraction — multi-vault support for BDH Graph Harness.

Provides ``VaultConfig``, ``VaultContext``, ``VaultRegistry``,
``normalize_vault_configs``, ``default_collection_name``, and ``context_to_state``.

Architecture
------------
- ``VaultConfig`` holds the static, configuration-derived description of a vault
  (id, paths, collection name, merged settings dict).
- ``VaultContext`` is the runtime state of a vault: graph nodes/edges, ChromaDB
  collection, Hebbian state, BM25 index, async lock, and optional file watcher.
- ``VaultRegistry`` owns all configured vaults and resolves the right context for
  each API request.  It supports both the legacy single-vault path and the new
  multi-vault ``vaults:`` config list.

Config backward compatibility
------------------------------
An existing single-vault config (``vault_path:``, ``chroma_collection:``, etc.)
is normalised internally to a single ``VaultConfig`` with ``id='default'`` and the
*exact* collection name from the config (preserving the cached ChromaDB collection).
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "VaultConfig",
    "VaultContext",
    "VaultRegistry",
    "normalize_vault_configs",
    "default_collection_name",
    "context_to_state",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Vault ID validation: must be URL-safe alphanumeric with hyphens/underscores
_ID_RE = re.compile(r'^[a-zA-Z0-9_-]+$')


def default_collection_name(vault_id: str) -> str:
    """Return the default ChromaDB collection name for a vault ID.

    Replaces sequences of characters outside ``[a-zA-Z0-9_-]`` with ``_``,
    strips leading/trailing underscores, and returns ``vault_{safe}_notes``.

    Examples::

        default_collection_name('my-vault')       # -> 'vault_my-vault_notes'
        default_collection_name('Research Vault')  # -> 'vault_Research_Vault_notes'
    """
    safe = re.sub(r'[^a-zA-Z0-9_-]+', '_', vault_id).strip('_')
    return f'vault_{safe}_notes'


def _resolve_chroma_path(vault_path: str, global_cp: str, vault_cp: str | None = None) -> str:
    """Resolve the Chroma persistence directory for a vault.

    Priority: vault-specific ``chroma_path`` > global ``chroma_path`` > ``.bdh-chroma``.
    Relative paths are joined with *vault_path*; absolute/``~``-prefixed paths are kept.
    """
    cp = vault_cp or global_cp or '.bdh-chroma'
    cp = os.path.expanduser(cp)
    if not os.path.isabs(cp):
        cp = os.path.join(vault_path, cp)
    return cp


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VaultConfig:
    """Static, config-derived description of a single vault."""

    id: str
    name: str
    path: str
    chroma_path: str
    chroma_collection: str
    settings: dict  # merged config dict with vault-specific values applied


@dataclass
class VaultContext:
    """Runtime state for a single vault."""

    config: VaultConfig
    nodes: dict
    edges: dict
    collection: Any
    state: dict
    bm25_index: Any | None = None
    state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    watcher: Any | None = None


# ---------------------------------------------------------------------------
# Config normalisation
# ---------------------------------------------------------------------------

def normalize_vault_configs(cfg: dict) -> list[VaultConfig]:
    """Parse a config dict into a validated list of :class:`VaultConfig` objects.

    Handles both single-vault (legacy ``vault_path:`` key) and multi-vault
    (``vaults:`` list) configs.

    Single-vault rules
    ------------------
    - Creates one ``VaultConfig`` with ``id='default'``.
    - Preserves ``chroma_collection`` exactly (Risk 3: avoid cache invalidation).
    - ``chroma_path`` resolves relative to ``vault_path``.

    Multi-vault rules
    -----------------
    - ``id`` is required, unique, and must match ``^[a-zA-Z0-9_-]+$``.
    - ``path`` is required.
    - ``chroma_collection`` defaults to ``vault_{id}_notes`` unless specified.
    - A shared ``chroma_path`` at the top level is used for all vaults unless
      overridden per-vault.

    Parameters
    ----------
    cfg:
        The merged config dict (from ``load_config`` or test fixtures).

    Returns
    -------
    list[VaultConfig]
        One entry per vault, in declaration order.

    Raises
    ------
    ValueError
        On duplicate IDs, invalid ID format, or missing required fields.
    """
    if 'vaults' not in cfg:
        return _normalise_single_vault(cfg)
    return _normalise_multi_vault(cfg)


def _normalise_single_vault(cfg: dict) -> list[VaultConfig]:
    vault_path = os.path.expanduser(cfg.get('vault_path', ''))
    chroma_path = _resolve_chroma_path(vault_path, cfg.get('chroma_path', '.bdh-chroma'))
    chroma_collection = cfg.get('chroma_collection', 'notes')

    settings = dict(cfg)
    settings['vault_path'] = vault_path
    settings['chroma_path'] = chroma_path
    settings['chroma_collection'] = chroma_collection

    return [VaultConfig(
        id='default',
        name=cfg.get('name', 'Default Vault'),
        path=vault_path,
        chroma_path=chroma_path,
        chroma_collection=chroma_collection,
        settings=settings,
    )]


def _normalise_multi_vault(cfg: dict) -> list[VaultConfig]:
    global_cp = cfg.get('chroma_path', '.bdh-chroma')
    vault_entries = cfg.get('vaults', [])
    seen_ids: set[str] = set()
    result: list[VaultConfig] = []

    for entry in vault_entries:
        vid = entry.get('id', '')
        if not vid:
            raise ValueError(f"Vault entry missing 'id': {entry}")
        if not _ID_RE.match(vid):
            raise ValueError(
                f"Vault id '{vid}' is invalid — must match ^[a-zA-Z0-9_-]+$"
            )
        if vid in seen_ids:
            raise ValueError(f"Duplicate vault id: '{vid}'")
        seen_ids.add(vid)

        vault_path = os.path.expanduser(entry.get('path', ''))
        if not vault_path:
            raise ValueError(f"Vault '{vid}' is missing a 'path'")

        resolved_cp = _resolve_chroma_path(vault_path, global_cp, entry.get('chroma_path'))
        chroma_collection = entry.get('chroma_collection') or default_collection_name(vid)

        # Build per-vault settings: global config + vault overrides
        settings = dict(cfg)
        settings.pop('vaults', None)  # exclude the vaults list from per-vault settings
        settings['vault_path'] = vault_path
        settings['chroma_path'] = resolved_cp
        settings['chroma_collection'] = chroma_collection
        # Per-vault overrides for vault-specific keys
        for key in ('neurogenesis_dir', 'graph_ignore', 'neurogenesis_enabled'):
            if key in entry:
                settings[key] = entry[key]

        result.append(VaultConfig(
            id=vid,
            name=entry.get('name', vid),
            path=vault_path,
            chroma_path=resolved_cp,
            chroma_collection=chroma_collection,
            settings=settings,
        ))

    return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class VaultRegistry:
    """Registry of all configured vaults and their runtime :class:`VaultContext` objects.

    Usage (server startup)::

        registry = VaultRegistry(config)
        registry.load_all()   # multi-vault: build graph/embeddings/state for each vault

        # OR single-vault (legacy): pre-built data passed in
        registry.register_context('default', ctx)
    """

    def __init__(self, global_config: dict):
        self._global_config = global_config
        self._vault_configs: list[VaultConfig] = normalize_vault_configs(global_config)
        self._contexts: dict[str, VaultContext] = {}
        self._default_id: str = (
            global_config.get('default_vault')
            or self._vault_configs[0].id
        )

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Build graph, embeddings, state, and BM25 index for every vault.

        Called at server startup in multi-vault mode.  For single-vault legacy
        mode, prefer :meth:`register_context` with pre-built data.
        """
        from bdh_graph_harness.graph.builder import build_graph
        from bdh_graph_harness.retrieval.chroma_store import compute_all_embeddings
        from bdh_graph_harness.retrieval.bm25 import BM25Index
        from bdh_graph_harness.memory.state_store import load_state

        for vc in self._vault_configs:
            print(f"   Loading vault '{vc.id}' from {vc.path} …")
            nodes, edges = build_graph(vc.path, use_cache=True)
            collection = compute_all_embeddings(
                nodes, vc.path,
                chroma_path=vc.chroma_path,
                collection_name=vc.chroma_collection,
                config=vc.settings,
            )
            state = load_state(vc.path)

            bm25_idx = None
            if vc.settings.get('hybrid_search', False):
                bm25_idx = BM25Index(
                    nodes,
                    k1=vc.settings.get('bm25_k1', 1.5),
                    b=vc.settings.get('bm25_b', 0.75),
                )

            self._contexts[vc.id] = VaultContext(
                config=vc,
                nodes=nodes,
                edges=edges,
                collection=collection,
                state=state,
                bm25_index=bm25_idx,
            )

    def register_context(self, vault_id: str, ctx: VaultContext) -> None:
        """Register a pre-built :class:`VaultContext` (single-vault legacy path)."""
        self._contexts[vault_id] = ctx

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list(self) -> list[VaultContext]:
        """Return all registered vault contexts in declaration order."""
        return [self._contexts[vc.id] for vc in self._vault_configs if vc.id in self._contexts]

    def vault_configs(self) -> list[VaultConfig]:
        """Return all vault configs."""
        return list(self._vault_configs)

    def get(self, vault_id: str | None = None) -> VaultContext:
        """Resolve a :class:`VaultContext` by ID.

        Parameters
        ----------
        vault_id:
            The requested vault.  ``None`` resolves to the default vault.

        Raises
        ------
        KeyError
            If *vault_id* is unknown (yields a 404/400-worthy response for callers).
        """
        target = vault_id if vault_id is not None else self._default_id
        if target not in self._contexts:
            available = list(self._contexts.keys())
            raise KeyError(f"Unknown vault '{target}'. Available: {available}")
        return self._contexts[target]

    def default_id(self) -> str:
        """Return the default vault ID."""
        return self._default_id

    def available_ids(self) -> list[str]:
        """Return IDs of all configured vaults."""
        return [vc.id for vc in self._vault_configs]


# ---------------------------------------------------------------------------
# State bridge
# ---------------------------------------------------------------------------

def context_to_state(ctx: VaultContext) -> dict:
    """Convert a :class:`VaultContext` to a legacy ``app_state``-compatible dict.

    Lets existing route handlers and helper functions work without changes
    during the multi-vault migration — they receive a dict that looks like
    the old single-vault ``app_state``, but is scoped to one vault.

    Keys match the original ``app_state`` shape::

        {
            'vault_id': str,
            'nodes': dict,
            'edges': dict,
            'collection': ChromaDB collection,
            'state': dict,
            'config': dict,          # vault-specific merged settings
            'bm25_index': BM25Index | None,
            'state_lock': asyncio.Lock,
        }
    """
    return {
        'vault_id': ctx.config.id,
        'nodes': ctx.nodes,
        'edges': ctx.edges,
        'collection': ctx.collection,
        'state': ctx.state,
        'config': ctx.config.settings,
        'bm25_index': ctx.bm25_index,
        'state_lock': ctx.state_lock,
    }
