"""Behavioral directed-merge tests: real staging, vault, HTTP routes and journal."""
import asyncio
import json
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from bdh_graph_harness.api.routes import setup_routes
from bdh_graph_harness.memory import curate_audit as audit
from bdh_graph_harness.memory import session_synthesis_staging as staging
from bdh_graph_harness.neurogenesis.operation_journal import list_operation_records, revert_operation
from bdh_graph_harness.vaults import VaultContext, VaultRegistry


@pytest_asyncio.fixture
async def merge_env(tmp_path):
    roots = {vid: tmp_path / vid for vid in ('core', 'other')}
    registry = VaultRegistry({'vaults': [{'id': v, 'path': str(p)} for v, p in roots.items()]})
    for vc in registry.vault_configs():
        root = Path(vc.path)
        path = root / 'wiki/concepts/curate-decision-flow.md'
        path.parent.mkdir(parents=True)
        path.write_text('---\ntitle: Curate Decision Flow\n---\n# Curate Decision Flow\nOriginal criteria.\n')
        node = {'id': 'vault:wiki/concepts/curate-decision-flow.md', 'title': 'Curate Decision Flow',
                'relative_path': 'wiki/concepts/curate-decision-flow.md', 'absolute_path': str(path),
                'path': str(path), 'source_type': 'vault', 'writable': True}
        ctx = VaultContext(vc, {node['id']: node}, {}, None, {})
        registry.register_context(vc.id, ctx)
    root = roots['core']
    candidate = staging.SessionSynthesisCandidate(
        'cand-directed', 'syn-directed', 'core', 'sess-directed', 'a' * 64,
        title='Merge vs Reject Heuristics', definition='Merge adds specific operational criteria.',
        provenance={'source_notes': ['Source'], 'source_node_ids': ['vault:source.md'], 'query': 'Review'},
        extra={'curator_merge_target': 'curate-decision-flow.md'})
    staging._save_candidate(candidate, root)
    audit.create_curate_candidate(str(root), **candidate.correlation())
    app = web.Application()
    setup_routes(app, {'registry': registry, 'config': {}}, set())
    client = TestClient(TestServer(app))
    await client.start_server()
    yield {'client': client, 'registry': registry, 'root': root, 'candidate': candidate,
           'target_id': node['id'], 'target': root / node['relative_path'], 'roots': roots}
    await client.close()


def request_body(env, **overrides):
    c = env['candidate']
    body = {k: getattr(c, k) for k in ('candidate_id', 'synthesis_id', 'session_id', 'source', 'vault_id')}
    return {**body, 'target_node_id': env['target_id'], 'confirmed': True, **overrides}


async def preview(env, **overrides):
    resp = await env['client'].post('/api/synthesis/merge-preview', json=request_body(env, **overrides))
    assert resp.status == 200, await resp.text()
    return await resp.json()


def snapshot(env):
    return {str(p.relative_to(env['root'])): p.read_bytes() for p in env['root'].rglob('*') if p.is_file()}


@pytest.mark.asyncio
async def test_pending_candidate_merges_into_existing_note_and_records_reversible_provenance(merge_env):
    env = merge_env
    before = env['target'].read_text()
    listing = await env['client'].get('/api/synthesis/merge-targets?vault_id=core&candidate_id=cand-directed')
    assert listing.status == 200, await listing.text()
    data = await listing.json()
    assert data['suggested_target_node_id'] == env['target_id']
    assert data['total_count'] == len(data['targets']) == 1
    assert data['targets'][0]['note_path'] == 'wiki/concepts/curate-decision-flow.md'
    assert env['target'].read_text() == before
    p = await preview(env)
    resp = await env['client'].post('/api/synthesis/merge', json=request_body(env, **{k: p[k] for k in ('candidate_revision', 'target_revision')}))
    assert resp.status == 200, await resp.text()
    result = await resp.json()
    assert result['status'] == 'merged' and result['applied'] and result['operation_id']
    assert env['candidate'].definition in env['target'].read_text()
    assert 'synthesis_id: syn-directed' in env['target'].read_text()
    assert list((env['root'] / 'wiki').rglob('*.md')) == [env['target']]
    assert staging.load_candidate(env['root'], 'cand-directed').status == 'applied'
    last = audit.latest_curate_state(str(env['root']), 'cand-directed')
    assert last.state == 'merged' and last.extra['target_node_id'] == env['target_id']
    assert last.applied_by == 'human'
    records = list_operation_records(str(env['root']))
    assert records[-1]['status'] == 'applied'
    reverted = revert_operation(str(env['root']), result['operation_id'])
    assert reverted['status'] == 'reverted'
    assert env['target'].read_text() == before


@pytest.mark.asyncio
@pytest.mark.parametrize('override,status', [
    ({'confirmed': False}, 400), ({'confirmed': 'true'}, 400),
    ({'candidate_id': '../bad'}, 400), ({'vault_id': 'other'}, 404),
    ({'synthesis_id': 'tampered'}, 409), ({'session_id': 'tampered'}, 409),
    ({'source': 'interactive'}, 409), ({'target_node_id': 'missing'}, 404),
    ({'target_revision': 'b' * 64}, 409), ({'candidate_revision': 'b' * 64}, 409),
])
async def test_invalid_merge_is_side_effect_free(merge_env, override, status):
    env = merge_env
    p = await preview(env)
    before = snapshot(env)
    body = request_body(env, **{k: p[k] for k in ('candidate_revision', 'target_revision')})
    resp = await env['client'].post('/api/synthesis/merge', json={**body, **override})
    assert resp.status == status, await resp.text()
    assert snapshot(env) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['external', 'readonly', 'escaped', 'symlink', 'missing'])
async def test_invalid_targets_are_unlisted_and_cannot_be_previewed(merge_env, kind):
    env = merge_env
    ctx = env['registry'].get('core')
    node = dict(ctx.nodes[env['target_id']])
    if kind == 'external':
        node['source_type'] = 'external'
    elif kind == 'readonly':
        node['writable'] = False
    elif kind == 'escaped':
        node['absolute_path'] = str(env['roots']['other'] / 'wiki/concepts/curate-decision-flow.md')
    elif kind == 'symlink':
        link = env['root'] / 'link.md'
        link.symlink_to(env['target'])
        node['absolute_path'] = str(link)
    else:
        node['absolute_path'] = str(env['root'] / 'missing.md')
    ctx.nodes['invalid'] = node
    before = snapshot(env)
    resp = await env['client'].post('/api/synthesis/merge-preview', json=request_body(env, target_node_id='invalid'))
    assert resp.status in {400, 403, 404}
    listing = await env['client'].get('/api/synthesis/merge-targets?vault_id=core&candidate_id=cand-directed')
    assert 'invalid' not in {item['node_id'] for item in (await listing.json())['targets']}
    assert snapshot(env) == before


@pytest.mark.asyncio
async def test_retry_and_concurrent_requests_write_once_and_bind_target(merge_env):
    env = merge_env
    p = await preview(env)
    body = request_body(env, **{k: p[k] for k in ('candidate_revision', 'target_revision')})
    responses = await asyncio.gather(*(env['client'].post('/api/synthesis/merge', json=body) for _ in range(2)))
    results = [await r.json() for r in responses]
    assert [r.status for r in responses] == [200, 200], results
    assert {r['idempotent'] for r in results} == {False, True}
    assert len({r['operation_id'] for r in results}) == 1
    before = snapshot(env)
    resp = await env['client'].post('/api/synthesis/merge', json={**body, 'target_node_id': 'different'})
    assert resp.status == 409
    assert snapshot(env) == before


@pytest.mark.asyncio
async def test_already_present_is_noop_not_false_merge(merge_env):
    env = merge_env
    env['target'].write_text(env['target'].read_text() + '\n' + env['candidate'].definition)
    p = await preview(env)
    before = env['target'].read_bytes()
    resp = await env['client'].post('/api/synthesis/merge', json=request_body(env, **{k: p[k] for k in ('candidate_revision', 'target_revision')}))
    result = await resp.json()
    assert resp.status == 200 and result['status'] == 'noop' and not result['applied']
    assert not result['operation_id'] and not list_operation_records(str(env['root']))
    assert env['target'].read_bytes() == before


@pytest.mark.asyncio
async def test_audit_failure_rolls_back_real_note_instead_of_leaving_untracked_merge(merge_env, monkeypatch):
    env = merge_env
    p = await preview(env)
    before = env['target'].read_bytes()
    def fail(*a, **kw):
        raise OSError('injected audit failure')
    monkeypatch.setattr(audit, 'transition_curate_state', fail)
    resp = await env['client'].post('/api/synthesis/merge', json=request_body(env, **{k: p[k] for k in ('candidate_revision', 'target_revision')}))
    assert resp.status == 500
    assert env['target'].read_bytes() == before
    assert staging.load_candidate(env['root'], 'cand-directed').status == 'pending_review'


@pytest.mark.asyncio
async def test_preapproved_candidate_still_requires_human_directed_confirmation(merge_env):
    env = merge_env
    candidate = env['candidate']
    candidate.status = 'pre_approved'
    staging._save_candidate(candidate, env['root'])
    p = await preview(env)
    before = snapshot(env)
    response = await env['client'].post('/api/synthesis/merge', json=request_body(env, **{k: p[k] for k in ('candidate_revision', 'target_revision')}, confirmed=False))
    assert response.status == 400 and snapshot(env) == before
    response = await env['client'].post('/api/synthesis/merge', json=request_body(env, **{k: p[k] for k in ('candidate_revision', 'target_revision')}))
    assert response.status == 200, await response.text()


@pytest.mark.asyncio
@pytest.mark.parametrize('hint', ['curate-decision-flow.md', 'missing-note.md'])
async def test_auto_apply_never_ignores_a_directed_target_hint(merge_env, hint):
    env = merge_env
    candidate = env['candidate']
    candidate.status = 'approved'
    candidate.extra['curator_merge_target'] = hint
    staging._save_candidate(candidate, env['root'])
    before = snapshot(env)
    response = await env['client'].post('/api/synthesis/apply', json=request_body(env))
    assert response.status == 409 and snapshot(env) == before


@pytest.mark.asyncio
async def test_target_search_is_bounded_and_ambiguous_hint_is_not_auto_selected(merge_env):
    env = merge_env
    ctx = env['registry'].get('core')
    assert ctx is not None
    original = ctx.nodes[env['target_id']]
    for index in range(55):
        ctx.nodes[f'alias-{index}'] = {**original, 'title': f'Searchable concept {index}'}
    before = snapshot(env)
    response = await env['client'].get('/api/synthesis/merge-targets?vault_id=core&candidate_id=cand-directed')
    assert response.status == 200
    result = await response.json()
    assert len(result['targets']) == 50 and result['total_count'] == 56 and result['has_more']
    assert result['suggested_target_node_id'] is None and 'ambiguous' in result['suggested_target_error']
    response = await env['client'].get('/api/synthesis/merge-targets', params={'vault_id': 'core', 'candidate_id': 'cand-directed', 'q': 'concept 54'})
    result = await response.json()
    assert result['total_count'] == len(result['targets']) == 1 and not result['has_more']
    assert result['targets'][0]['node_id'] == 'alias-54'
    assert snapshot(env) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('changed', ['candidate', 'target'])
async def test_real_content_changes_invalidate_reviewed_preview(merge_env, changed):
    env = merge_env
    p = await preview(env)
    if changed == 'target':
        env['target'].write_text(env['target'].read_text() + '\nConcurrent edit.\n')
    else:
        env['candidate'].definition += ' New information not yet reviewed.'
        staging._save_candidate(env['candidate'], env['root'])
    before = snapshot(env)
    response = await env['client'].post('/api/synthesis/merge', json=request_body(env, **{k: p[k] for k in ('candidate_revision', 'target_revision')}))
    assert response.status == 409 and snapshot(env) == before
