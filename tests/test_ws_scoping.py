from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from bdh_graph_harness.api.routes import setup_routes
from bdh_graph_harness.api.ws import broadcast_activation


class FakeWebSocket:
    def __init__(self, vault_id):
        self._bdh_vault_id = vault_id
        self.messages = []

    async def send_str(self, message):
        self.messages.append(message)


@pytest.mark.asyncio
async def test_broadcast_activation_scopes_events_to_vault():
    core = FakeWebSocket("core")
    episodic = FakeWebSocket("episodic")

    await broadcast_activation(
        {"type": "activation", "vault_id": "core"},
        {core, episodic},
    )

    assert len(core.messages) == 1
    assert episodic.messages == []


@pytest.mark.asyncio
async def test_broadcast_activation_keeps_legacy_clients_compatible():
    legacy = FakeWebSocket(None)
    core = FakeWebSocket("core")

    await broadcast_activation(
        {"type": "activation", "vault_id": "core"},
        {legacy, core},
    )

    assert len(legacy.messages) == 1
    assert len(core.messages) == 1


class _Registry:
    """Minimal multi-vault registry for the WebSocket routing contract."""

    def __init__(self):
        self._contexts = {
            "core": SimpleNamespace(
                config=SimpleNamespace(id="core"),
                nodes={},
                edges={},
                state={"synapses": {}},
                event_sequence=0,
            ),
            "episodic": SimpleNamespace(
                config=SimpleNamespace(id="episodic"),
                nodes={},
                edges={},
                state={"synapses": {}},
                event_sequence=0,
            ),
        }

    def get(self, vault_id=None):
        target = vault_id or "core"
        if target not in self._contexts:
            raise KeyError(target)
        return self._contexts[target]

    def available_ids(self):
        return list(self._contexts)


@pytest.mark.asyncio
async def test_websocket_rejects_unknown_vault_instead_of_falling_back_to_default():
    app = web.Application()
    setup_routes(app, {"registry": _Registry()}, set())
    client = TestClient(TestServer(app))
    await client.start_server()

    try:
        response = await client.get("/ws?vault_id=unknown")

        assert response.status == 400
        assert await response.json() == {
            "error": "Unknown vault 'unknown'",
            "available_vaults": ["core", "episodic"],
        }
    finally:
        await client.close()
