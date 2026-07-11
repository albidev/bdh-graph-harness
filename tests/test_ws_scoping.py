import pytest

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
