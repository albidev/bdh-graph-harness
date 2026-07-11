import asyncio

import pytest

from bdh_graph_harness.api import routes
from tests.test_multivault_api_regression import make_context


@pytest.mark.asyncio
async def test_queries_for_one_vault_are_serialized(monkeypatch):
    ctx = make_context("serialized")
    active_count = 0
    max_active = 0

    async def fake_unlocked(query, ctx, ws_clients, source=None, learn=True):
        nonlocal active_count, max_active
        active_count += 1
        max_active = max(max_active, active_count)
        await asyncio.sleep(0.01)
        active_count -= 1
        return {}, [], [], {}

    monkeypatch.setattr(routes, "_run_attention_and_plasticity_unlocked", fake_unlocked)

    await asyncio.gather(
        routes.run_attention_and_plasticity("one", ctx, set()),
        routes.run_attention_and_plasticity("two", ctx, set()),
    )

    assert max_active == 1
