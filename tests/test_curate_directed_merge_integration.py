"""Real HTTP: MC manifest loader → Curate adapter → BDH → isolated note/audit.

Requires sibling mc-curate-plugin and hermes-mission-control checkouts, overridable
with CURATE_PLUGIN_REPO and CURATE_MC_ROOT. No service restart or production state.
"""
import asyncio
import importlib.util
import os
from pathlib import Path
from urllib.parse import parse_qs

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tests.test_directed_merge import merge_env, snapshot
from bdh_graph_harness.memory import curate_audit as audit
from bdh_graph_harness.memory import session_synthesis_staging as staging

PROJECTS = Path(__file__).resolve().parents[2]
PLUGIN = Path(os.environ.get('CURATE_PLUGIN_REPO', PROJECTS / 'mc-curate-plugin'))
HOST = Path(os.environ.get('CURATE_MC_ROOT', PROJECTS / 'hermes-mission-control'))
pytestmark = [pytest.mark.asyncio, pytest.mark.skipif(not (PLUGIN / 'manifest.json').exists() or not (HOST / 'server/plugins/loader.py').exists(), reason='requires sibling Curate and MC checkouts')]


async def test_real_plugin_loader_http_merge_preserves_scoping_confirmation_and_reversible_audit(merge_env, monkeypatch, tmp_path):
    env = merge_env
    home = tmp_path / 'hermes-home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('VB_VAULT', str(env['root']))
    monkeypatch.setenv('BDH_API_URL', str(env['client'].make_url('/')).rstrip('/'))
    monkeypatch.syspath_prepend(str(HOST / 'server'))
    monkeypatch.syspath_prepend(str(PLUGIN))
    spec = importlib.util.spec_from_file_location('isolated_mc_loader', HOST / 'server/plugins/loader.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    external = tmp_path / 'mc-plugins'
    external.mkdir()
    (external / 'curate').symlink_to(PLUGIN, target_is_directory=True)
    loader = module.PluginLoader(internal_dir=tmp_path / 'empty-internal', external_dir=external)
    assert loader.load_plugin('curate')
    manifest = loader.get_manifest('curate')
    assert loader.get_plugin_dir('curate').resolve() == PLUGIN.resolve()

    async def dispatch(request):
        path = request.path.removeprefix('/api/local') or '/'
        definition = next(e for e in manifest['endpoints'] if e['path'] == path and e['method'] == request.method)
        if definition.get('authRequired') and request.headers.get('Authorization') != 'Bearer isolated-test-token':
            return web.json_response({'error': 'Unauthorized'}, status=401)
        endpoint = loader.resolve(request.method, path)
        assert endpoint is not None
        body = await request.json() if request.method == 'POST' else {}
        try:
            result = await asyncio.to_thread(endpoint.handler_fn, body, parse_qs(request.query_string), None)
        except Exception as exc:
            if not hasattr(exc, 'status_code'):
                raise
            return web.json_response({'error': str(exc), 'code': getattr(exc, 'code', '')}, status=getattr(exc, 'status_code'))
        return web.json_response(result)

    app = web.Application()
    app.router.add_route('*', '/api/local{tail:.*}', dispatch)
    client = TestClient(TestServer(app))
    await client.start_server()
    headers = {'Authorization': 'Bearer isolated-test-token'}
    identity = {'vault': 'core', 'candidate_id': env['candidate'].candidate_id}
    prefix = '/api/local'
    before = snapshot(env)
    try:
        response = await client.get(prefix + '/synthesis/merge-targets', params=identity)
        assert response.status == 401 and snapshot(env) == before
        response = await client.get(prefix + '/synthesis/merge-targets', params=identity, headers=headers)
        assert response.status == 200, await response.text()
        targets = await response.json()
        assert targets['suggested_target_node_id'] == env['target_id']
        response = await client.post(prefix + '/synthesis/merge-preview', json={**identity, 'target_node_id': env['target_id']}, headers=headers)
        assert response.status == 200, await response.text()
        preview = await response.json()
        assert snapshot(env) == before
        payload = {**identity, 'target_node_id': env['target_id'], **{k: preview[k] for k in ('candidate_revision', 'target_revision')}}
        response = await client.post(prefix + '/synthesis/merge', json={**payload, 'confirmed': False}, headers=headers)
        assert response.status == 400 and snapshot(env) == before
        response = await client.post(prefix + '/synthesis/merge', json={**payload, 'target_revision': 'b' * 64, 'confirmed': True}, headers=headers)
        assert response.status == 409 and snapshot(env) == before
        response = await client.post(prefix + '/candidates/approve', json={'id': identity['candidate_id'], 'vault': 'core'}, headers=headers)
        assert response.status == 409 and snapshot(env) == before
        response = await client.post(prefix + '/synthesis/merge', json={**payload, 'confirmed': True, 'session_id': 'untrusted', 'synthesis_id': 'untrusted'}, headers=headers)
        assert response.status == 200, await response.text()
        result = await response.json()
        assert result['status'] == 'merged' and result['operation_id']
        assert env['candidate'].definition in env['target'].read_text()
        assert len(list((env['root'] / 'wiki').rglob('*.md'))) == 1
        staged = staging.load_candidate(env['root'], identity['candidate_id'])
        assert staged is not None and staged.status == 'applied'
        latest = audit.latest_curate_state(str(env['root']), identity['candidate_id'])
        assert latest is not None and latest.state == 'merged' and latest.applied_by == 'human'
        after = snapshot(env)
        response = await client.post(prefix + '/synthesis/merge', json={**payload, 'confirmed': True}, headers=headers)
        assert response.status == 200 and (await response.json())['idempotent'] is True
        assert snapshot(env) == after
        response = await client.post(prefix + '/synthesis/revert', json={'operation_id': result['operation_id'], 'vault': 'core'}, headers=headers)
        assert response.status == 200, await response.text()
        assert (await response.json())['status'] == 'reverted'
        assert env['target'].read_bytes() == before['wiki/concepts/curate-decision-flow.md']
    finally:
        await client.close()
