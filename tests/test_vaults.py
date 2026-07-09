"""Tests for the multi-vault VaultConfig/VaultRegistry system.

Covers:
- normalize_vault_configs for both single-vault (legacy) and multi-vault configs
- default_collection_name helper
- Chroma collection isolation between vaults
- VaultRegistry.get() resolution and error handling
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# normalize_vault_configs
# ---------------------------------------------------------------------------

def test_single_vault_config_normalizes_to_default_context(tmp_path):
    from bdh_graph_harness.vaults import normalize_vault_configs

    (tmp_path / "Hermes").mkdir()
    cfg = {
        "vault_path": str(tmp_path / "Hermes"),
        "chroma_path": ".bdh-chroma",
        "chroma_collection": "notes",
    }

    vaults = normalize_vault_configs(cfg)

    assert len(vaults) == 1
    vault = vaults[0]
    assert vault.id == "default"
    assert vault.path == str(tmp_path / "Hermes")
    assert vault.chroma_collection == "notes"
    # relative chroma_path is resolved relative to vault path
    assert vault.chroma_path == str(tmp_path / "Hermes" / ".bdh-chroma")


def test_multi_vault_config_assigns_default_collection_names(tmp_path):
    from bdh_graph_harness.vaults import normalize_vault_configs

    (tmp_path / "A").mkdir()
    (tmp_path / "B").mkdir()
    cfg = {
        "default_vault": "alpha",
        "chroma_path": str(tmp_path / "chroma"),
        "vaults": [
            {"id": "alpha", "name": "Alpha", "path": str(tmp_path / "A")},
            {"id": "beta", "name": "Beta", "path": str(tmp_path / "B")},
        ],
    }

    vaults = normalize_vault_configs(cfg)

    assert [v.id for v in vaults] == ["alpha", "beta"]
    assert vaults[0].chroma_collection == "vault_alpha_notes"
    assert vaults[1].chroma_collection == "vault_beta_notes"


def test_multi_vault_explicit_collection_overrides_default(tmp_path):
    from bdh_graph_harness.vaults import normalize_vault_configs

    (tmp_path / "A").mkdir()
    cfg = {
        "chroma_path": str(tmp_path / "chroma"),
        "vaults": [
            {
                "id": "alpha",
                "name": "Alpha",
                "path": str(tmp_path / "A"),
                "chroma_collection": "my_custom_collection",
            }
        ],
    }

    vaults = normalize_vault_configs(cfg)

    assert vaults[0].chroma_collection == "my_custom_collection"


def test_duplicate_vault_ids_raise_error(tmp_path):
    from bdh_graph_harness.vaults import normalize_vault_configs

    (tmp_path / "A").mkdir()
    (tmp_path / "B").mkdir()
    cfg = {
        "vaults": [
            {"id": "dup", "name": "First", "path": str(tmp_path / "A")},
            {"id": "dup", "name": "Second", "path": str(tmp_path / "B")},
        ],
    }

    with pytest.raises((ValueError, KeyError)):
        normalize_vault_configs(cfg)


def test_invalid_vault_id_raises_error(tmp_path):
    from bdh_graph_harness.vaults import normalize_vault_configs

    (tmp_path / "A").mkdir()
    cfg = {
        "vaults": [
            {"id": "has spaces!", "name": "Bad", "path": str(tmp_path / "A")},
        ],
    }

    with pytest.raises((ValueError, KeyError)):
        normalize_vault_configs(cfg)


def test_vault_settings_contains_vault_path(tmp_path):
    """VaultConfig.settings should include vault_path for legacy module compat."""
    from bdh_graph_harness.vaults import normalize_vault_configs

    (tmp_path / "V").mkdir()
    cfg = {
        "vault_path": str(tmp_path / "V"),
        "chroma_path": ".bdh-chroma",
        "chroma_collection": "notes",
    }

    vaults = normalize_vault_configs(cfg)
    assert vaults[0].settings["vault_path"] == str(tmp_path / "V")


# ---------------------------------------------------------------------------
# default_collection_name
# ---------------------------------------------------------------------------

def test_default_collection_name_is_stable():
    from bdh_graph_harness.vaults import default_collection_name

    assert default_collection_name("my-vault") == "vault_my-vault_notes"
    assert default_collection_name("Research Vault") == "vault_Research_Vault_notes"
    assert default_collection_name("simple") == "vault_simple_notes"
    assert default_collection_name("abc_123") == "vault_abc_123_notes"


# ---------------------------------------------------------------------------
# VaultRegistry
# ---------------------------------------------------------------------------

def _make_ctx(vault_id: str, neurons: int = 3):
    """Create a minimal VaultContext for testing."""
    from bdh_graph_harness.vaults import VaultConfig, VaultContext
    import asyncio

    vc = VaultConfig(
        id=vault_id,
        name=vault_id.capitalize(),
        path=f"/tmp/vault_{vault_id}",
        chroma_path=f"/tmp/chroma",
        chroma_collection=f"vault_{vault_id}_notes",
        settings={
            "vault_path": f"/tmp/vault_{vault_id}",
            "chroma_collection": f"vault_{vault_id}_notes",
        },
    )
    nodes = {f"{vault_id}/note{i}": {"title": f"Note {i}"} for i in range(neurons)}
    return VaultContext(
        config=vc,
        nodes=nodes,
        edges={},
        collection=MagicMock(),
        state={"synapses": {}, "queries": 0},
        state_lock=asyncio.Lock(),
    )


def test_registry_get_by_id():
    from bdh_graph_harness.vaults import VaultRegistry

    reg = VaultRegistry.__new__(VaultRegistry)
    reg._vault_configs = []
    reg._contexts = {}
    reg._default_id = None

    ctx_a = _make_ctx("alpha")
    ctx_b = _make_ctx("beta")
    reg._contexts["alpha"] = ctx_a
    reg._contexts["beta"] = ctx_b
    reg._default_id = "alpha"

    assert reg.get("alpha") is ctx_a
    assert reg.get("beta") is ctx_b


def test_registry_get_default_when_id_omitted():
    from bdh_graph_harness.vaults import VaultRegistry

    reg = VaultRegistry.__new__(VaultRegistry)
    reg._vault_configs = []
    reg._contexts = {}
    reg._default_id = "alpha"

    ctx_a = _make_ctx("alpha")
    reg._contexts["alpha"] = ctx_a

    assert reg.get(None) is ctx_a
    assert reg.get() is ctx_a


def test_registry_get_unknown_id_raises():
    from bdh_graph_harness.vaults import VaultRegistry

    reg = VaultRegistry.__new__(VaultRegistry)
    reg._vault_configs = []
    reg._contexts = {"alpha": _make_ctx("alpha")}
    reg._default_id = "alpha"

    with pytest.raises(KeyError):
        reg.get("nonexistent")


def test_registry_list_returns_all():
    from bdh_graph_harness.vaults import VaultRegistry, VaultConfig

    reg = VaultRegistry.__new__(VaultRegistry)
    ctx_a = _make_ctx("alpha")
    ctx_b = _make_ctx("beta")
    # _vault_configs drives ordering in list()
    reg._vault_configs = [ctx_a.config, ctx_b.config]
    reg._contexts = {"alpha": ctx_a, "beta": ctx_b}
    reg._default_id = "alpha"

    vaults = reg.list()
    assert len(vaults) == 2
    ids = {v.config.id for v in vaults}
    assert ids == {"alpha", "beta"}


def test_registry_register_context():
    """register_context should add the vault to the registry."""
    from bdh_graph_harness.vaults import VaultRegistry

    reg = VaultRegistry.__new__(VaultRegistry)
    reg._vault_configs = []
    reg._contexts = {}
    reg._default_id = None

    ctx = _make_ctx("myv")
    reg.register_context("myv", ctx)

    assert "myv" in reg._contexts
    assert reg._contexts["myv"] is ctx


# ---------------------------------------------------------------------------
# Chroma collection isolation (using real ChromaDB in tmp_path)
# ---------------------------------------------------------------------------

def test_chroma_collections_are_isolated_per_vault(tmp_path, monkeypatch):
    """Two vaults sharing a chroma_path must not share embeddings."""
    import chromadb
    from bdh_graph_harness.retrieval.chroma_store import compute_all_embeddings

    # Monkeypatch get_embeddings to return deterministic unit vectors
    def fake_get_embeddings(texts, config=None):
        return [[float(i + 1)] * 384 for i in range(len(texts))]

    monkeypatch.setattr(
        "bdh_graph_harness.retrieval.chroma_store.get_embeddings",
        fake_get_embeddings,
    )

    chroma_dir = str(tmp_path / "shared_chroma")

    # Vault A has one note
    nodes_a = {"note_a": {"title": "Alpha Note", "text": "Alpha only content", "tags": "test"}}
    vault_a_root = str(tmp_path / "vault_a")
    (tmp_path / "vault_a").mkdir()

    # Vault B has one note with different ID
    nodes_b = {"note_b": {"title": "Beta Note", "text": "Beta only content", "tags": "test"}}
    vault_b_root = str(tmp_path / "vault_b")
    (tmp_path / "vault_b").mkdir()

    collection_a = compute_all_embeddings(
        nodes_a, vault_a_root,
        chroma_path=chroma_dir,
        collection_name="vault_a_notes",
    )
    collection_b = compute_all_embeddings(
        nodes_b, vault_b_root,
        chroma_path=chroma_dir,
        collection_name="vault_b_notes",
    )

    # Each collection has exactly one embedding — no cross-contamination
    assert collection_a.count() == 1
    assert collection_b.count() == 1

    ids_a = collection_a.get()["ids"]
    ids_b = collection_b.get()["ids"]

    assert ids_a == ["note_a"]
    assert ids_b == ["note_b"]

    # Verify independence: note_b is not in collection_a and vice-versa
    assert "note_b" not in ids_a
    assert "note_a" not in ids_b
