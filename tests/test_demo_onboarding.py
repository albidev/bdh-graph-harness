"""Public demo onboarding invariants, independent of private operator configuration."""
from pathlib import Path
import shutil
from urllib.parse import urlparse

import pytest

from bdh_graph_harness.config import CONFIG, load_config, resolve_llm_config_for_source
from bdh_graph_harness.__main__ import show_source_scan

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def demo(tmp_path, monkeypatch):
    config_path = ROOT / 'examples' / 'demo.yaml'
    assert config_path.is_file(), 'The public demo configuration must be shipped'
    vault = tmp_path / 'vault'
    shutil.copytree(ROOT / 'examples' / 'demo-vault', vault)
    monkeypatch.setenv('BDH_DEMO_VAULT', str(vault))
    monkeypatch.setenv('BDH_DEMO_CHROMA', str(tmp_path / 'chroma'))
    original = dict(CONFIG)
    try:
        yield load_config(str(config_path)), vault
    finally:
        CONFIG.clear()
        CONFIG.update(original)


def test_demo_configuration_has_explicit_local_storage_and_privacy_gate(demo):
    config, vault = demo
    assert Path(config['vault_path']) == vault
    assert Path(config['chroma_path']).parent == vault.parent
    assert config['neurogenesis_enabled'] is False
    assert config['session_synthesis_staging_enabled'] is True
    effective = resolve_llm_config_for_source(config, 'session_synthesis')
    assert effective.get('llm_local_only') is True
    assert urlparse(effective['llm_endpoint']).hostname in {'localhost', '127.0.0.1', '::1'}
    cloud = dict(config)
    cloud['llm'] = dict(config.get('llm', {}), provider='ollama-cloud', base_url='https://example.invalid/v1')
    with pytest.raises(ValueError, match='local_only'):
        resolve_llm_config_for_source(cloud, 'session_synthesis')


def test_demo_source_scan_resolves_links_without_persistent_writes(demo, monkeypatch):
    config, vault = demo
    before = {str(p.relative_to(vault)): p.read_bytes() for p in vault.rglob('*') if p.is_file()}
    import socket

    def forbid_network(*args, **kwargs):
        raise AssertionError('Source scanning must not access a network provider')

    monkeypatch.setattr(socket.socket, 'connect', forbid_network)
    result = show_source_scan(config)
    assert result['nodes']
    assert any(result['edges'].values()), 'The demo must exercise structural links'
    assert result['unresolved'] == []
    after = {str(p.relative_to(vault)): p.read_bytes() for p in vault.rglob('*') if p.is_file()}
    assert after == before
    assert not Path(config['chroma_path']).exists()
