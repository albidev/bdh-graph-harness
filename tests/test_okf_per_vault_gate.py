"""The OKF retrieval policy follows the settings of the vault being queried."""
import pytest

import bdh_graph_harness.api.routes as bdh_routes
import bdh_graph_harness.retrieval.attention as attention_mod
from bdh_graph_harness.config import config_overlay
from bdh_graph_harness.retrieval.okf_policy import is_okf_retrieval_policy_enabled

from tests.test_multi_query_api import _capture_app, mock_app_mq_setup  # noqa: F401
from tests.test_traversal_quota import BASE, graph  # noqa: F401

ON = {'okf_mode': 'read', 'okf_retrieval_policy_enabled': True}
OFF = {'okf_mode': 'off', 'okf_retrieval_policy_enabled': False}


def test_vault_settings_win_over_the_global_gate():
    with config_overlay(OFF):
        assert is_okf_retrieval_policy_enabled() is False
        assert is_okf_retrieval_policy_enabled(ON) is True
    with config_overlay(ON):
        assert is_okf_retrieval_policy_enabled() is True
        assert is_okf_retrieval_policy_enabled(OFF) is False
        assert is_okf_retrieval_policy_enabled({'okf_mode': 'read', 'okf_retrieval_policy_enabled': False}) is False


def _routing(graph, vault_settings):  # noqa: F811
    nodes, edges, collection = graph
    routing = {}
    with config_overlay({**BASE, **OFF}):
        attention_mod.attention(
            'query', nodes, edges, collection, k=2,
            routing_meta=routing, vault_settings=vault_settings,
        )
    return routing


def test_attention_applies_the_policy_of_the_queried_vault(graph):  # noqa: F811
    assert _routing(graph, None)['okf_policy']['enabled'] is False
    assert _routing(graph, ON)['okf_policy']['enabled'] is True
    assert _routing(graph, OFF)['okf_policy']['enabled'] is False


@pytest.mark.asyncio
async def test_api_query_hands_the_vault_settings_to_attention(mock_app_mq_setup, monkeypatch):  # noqa: F811
    from aiohttp.test_utils import TestClient, TestServer

    nodes, edges, collection, state, config, _ = mock_app_mq_setup
    config.update(ON)
    seen = {}

    def _attention(query, *args, routing_meta=None, vault_settings=None, **kwargs):
        seen['vault_settings'] = vault_settings
        if routing_meta is not None:
            routing_meta.update({'vector_top_score': 0.9, 'activation_details': []})
        return {}

    monkeypatch.setattr(bdh_routes, 'attention', _attention)
    app = _capture_app(monkeypatch, config, nodes, edges, collection, state)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.post('/api/query', json={'query': 'anything', 'learn': False, 'respond': False})
        assert resp.status == 200
    finally:
        await client.close()
    assert seen['vault_settings']['okf_mode'] == 'read'
    assert seen['vault_settings']['okf_retrieval_policy_enabled'] is True
