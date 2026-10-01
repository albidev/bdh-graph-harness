"""Human-confirmed, vault-local merges; automatic apply never chooses this target."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from aiohttp import web

from bdh_graph_harness.memory import curate_audit as audit
from bdh_graph_harness.memory import session_synthesis_staging as staging
from bdh_graph_harness.neurogenesis.merge import assimilate_evidence, looks_conflicting


class MergeError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(message)


def _string(data, key, *, pattern=None):
    value = data.get(key)
    if not isinstance(value, str) or not value or value != value.strip():
        raise MergeError(400, f'Invalid or missing {key}')
    if pattern and not re.fullmatch(pattern, value):
        raise MergeError(400, f'Invalid {key}')
    return value


def _context(app_state, data):
    from bdh_graph_harness.api.routes import _resolve_vault_ctx
    vault = _string(data, 'vault_id', pattern=r'[A-Za-z0-9_-]+')
    ctx, error = _resolve_vault_ctx(app_state, vault)
    if error is not None:
        raise MergeError(error.status, 'Unknown vault')
    if ctx.config.settings.get('writable') is False:
        raise MergeError(403, 'Vault is read-only')
    return ctx


def _candidate(ctx, data):
    cid = _string(data, 'candidate_id', pattern=r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}')
    candidate = staging.load_candidate(ctx.config.path, cid)
    if candidate is None:
        raise MergeError(404, 'Candidate not found in requested vault')
    if candidate.vault_id != ctx.config.id or candidate.source != 'session_synthesis':
        raise MergeError(409, 'Candidate is not a session synthesis owned by this vault')
    return candidate


def _target(ctx, node_id):
    node = ctx.nodes.get(node_id)
    if not node:
        raise MergeError(404, 'Target note not found in requested vault')
    if node_id.startswith('external:') or node.get('source_type', 'vault') != 'vault' or node.get('writable') is False:
        raise MergeError(403, 'Only writable vault-local notes may receive evidence')
    root = Path(ctx.config.path).resolve()
    raw = node.get('absolute_path') or node.get('relative_path') or node.get('path')
    if not isinstance(raw, str) or not raw or '://' in raw or '\\' in raw or '..' in Path(raw).parts:
        raise MergeError(400, 'Invalid target path')
    path = Path(raw)
    path = path if path.is_absolute() else root / path
    try:
        path.relative_to(root)
        # Refuse symlink targets (including inside-vault aliases) to avoid replacing
        # the link rather than updating its destination, or a swapped parent link.
        cursor = path
        while cursor != root:
            if cursor.is_symlink():
                raise MergeError(403, 'Symlink targets are not supported')
            cursor = cursor.parent
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(root).as_posix()
    except (ValueError, OSError) as exc:
        raise MergeError(403, 'Target must exist inside the candidate vault') from exc
    if path.suffix != '.md' or not path.is_file() or path.name in {'index.md', 'log.md'}:
        raise MergeError(400, 'Target must be an existing graph Markdown note')
    import os
    if not os.access(path, os.W_OK):
        raise MergeError(403, 'Target note is read-only')
    return {'node_id': node_id, 'title': str(node.get('title') or node_id), 'note_path': relative}, path


def _revision(candidate):
    # Includes correlation, provenance and curator intent; excludes lifecycle-only
    # fields so an approval/retry does not manufacture a different concept.
    payload = candidate.to_dict()
    payload.pop('status', None)
    payload['provenance'] = {k: v for k, v in candidate.provenance.items() if k not in {'updated_at', 'status_reason'}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _hash(content):
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def _preview(ctx, candidate, node_id):
    target, path = _target(ctx, node_id)
    content = path.read_text(encoding='utf-8')
    return {'vault_id': ctx.config.id,
            'candidate': {'candidate_id': candidate.candidate_id, 'title': candidate.title, 'definition': candidate.definition},
            'target': {**target, 'content': content},
            'candidate_revision': _revision(candidate), 'target_revision': _hash(content)}


def _available(ctx):
    items = []
    for node_id in ctx.nodes:
        try:
            item, _ = _target(ctx, node_id)
        except MergeError:
            continue
        items.append(item)
    return sorted(items, key=lambda item: (item['title'].casefold(), item['note_path'], item['node_id']))


def _suggested(candidate, items):
    hint = candidate.extra.get('curator_merge_target')
    if not hint:
        return None, None
    if not isinstance(hint, str):
        return None, 'Suggested target has an invalid format; choose a note explicitly.'
    # Do not normalize a malformed hint or use fuzzy similarity to choose a file.
    for key in ('title', 'note_path', 'basename', 'slug'):
        matches = [i for i in items if {
            'title': i['title'], 'note_path': i['note_path'],
            'basename': Path(i['note_path']).name, 'slug': Path(i['note_path']).stem,
        }[key] == hint]
        if len(matches) == 1:
            return matches[0]['node_id'], None
        if len(matches) > 1:
            return None, f'Suggested target "{hint}" is ambiguous; choose a note explicitly.'
    return None, f'Suggested target "{hint}" is missing or not writable in this vault.'


def _result(candidate, entry, *, idempotent):
    return {'candidate_id': candidate.candidate_id, 'synthesis_id': candidate.synthesis_id,
            'vault_id': candidate.vault_id, 'status': entry.state, 'note_path': entry.note_path,
            'operation_id': entry.operation_id, 'idempotent': idempotent, 'applied': entry.state == 'merged'}


def _merge(ctx, candidate, data):
    for key in ('synthesis_id', 'session_id', 'source'):
        if _string(data, key) != getattr(candidate, key):
            raise MergeError(409, f'Candidate {key} does not match request')
    if data.get('confirmed') is not True:
        raise MergeError(400, 'Explicit confirmed:true is required')
    node_id = _string(data, 'target_node_id')
    revisions = {k: _string(data, k, pattern=r'[0-9a-f]{64}') for k in ('candidate_revision', 'target_revision')}
    entry = audit.latest_curate_state(ctx.config.path, candidate.candidate_id)
    if entry is None or entry.correlation() != candidate.correlation():
        raise MergeError(409, 'Candidate audit provenance is missing or inconsistent')
    # Durable audit binds retries to precisely the reviewed destination/revisions.
    if entry.state in {'merged', 'noop'}:
        if not entry.extra.get('directed_merge') or any(entry.extra.get(k) != v for k, v in {**revisions, 'target_node_id': node_id}.items()):
            raise MergeError(409, 'Candidate was already applied with a different decision')
        _target(ctx, node_id)
        if candidate.status != 'applied':
            staging.update_candidate_status(ctx.config.path, candidate.candidate_id, 'applied')
        return _result(candidate, entry, idempotent=True)
    if entry.state != 'pending_review' or candidate.status not in {'pending_review', 'approved', 'pre_approved'}:
        raise MergeError(409, 'Candidate is no longer pending review')
    p = _preview(ctx, candidate, node_id)
    if any(p[k] != v for k, v in revisions.items()):
        raise MergeError(409, 'Candidate or target changed; reload and review the preview before confirming')
    if not candidate.definition.strip():
        raise MergeError(409, 'Candidate definition is empty')
    if candidate.provenance.get('would_conflict') or looks_conflicting(candidate.definition):
        raise MergeError(409, 'Possible conflicting evidence needs reconciliation before merge')
    _, path = _target(ctx, node_id)
    # Passing only the validated path prevents a stale alternate absolute_path in
    # graph metadata from overriding the human-reviewed destination.
    merged = assimilate_evidence(
        ctx.config.path, node_id, {'absolute_path': str(path)}, candidate.definition,
        source_notes=candidate.provenance.get('source_notes'),
        source_node_ids=candidate.provenance.get('source_node_ids'),
        query=str(candidate.provenance.get('query') or ''), source=candidate.source,
        synthesis_meta=candidate.correlation())
    outcome = merged.get('status')
    if outcome not in {'merged', 'already_present'}:
        raise MergeError(409, f'Merge produced no supported outcome ({outcome}); candidate remains pending')
    try:
        entry = audit.transition_curate_state(
            ctx.config.path, candidate.candidate_id, 'merged' if outcome == 'merged' else 'noop',
            reason='Human-confirmed directed merge' if outcome == 'merged' else 'Evidence already present in selected note',
            note_path=p['target']['note_path'], operation_id=merged.get('operation_id', ''), applied_by='human',
            extra={**revisions, 'target_node_id': node_id, 'directed_merge': True, 'confirmed': True})
    except Exception as exc:
        # A failed audit cannot leave an untracked vault mutation behind.
        from bdh_graph_harness.neurogenesis.operation_journal import revert_operation
        if merged.get('operation_id'):
            rollback = revert_operation(ctx.config.path, merged['operation_id'])
            if rollback.get('status') != 'reverted':
                raise MergeError(500, 'Audit failed and rollback needs manual recovery; do not retry blindly') from exc
        raise MergeError(500, 'Audit failed; note mutation rolled back') from exc
    staging.update_candidate_status(ctx.config.path, candidate.candidate_id, 'applied')
    return _result(candidate, entry, idempotent=False)


async def _body(request):
    try:
        data = await request.json()
    except (ValueError, TypeError) as exc:
        raise MergeError(400, 'Invalid JSON body') from exc
    if not isinstance(data, dict):
        raise MergeError(400, 'Expected a JSON object')
    return data


async def api_synthesis_merge_targets(request, app_state):
    try:
        data = request.query
        ctx = _context(app_state, data)
        candidate = _candidate(ctx, data)
        items = _available(ctx)
        suggested, error = _suggested(candidate, items)
        q = data.get('q', '').casefold()
        filtered = [i for i in items if q in i['title'].casefold() or q in i['note_path'].casefold()]
        # Keep the suggested note visible on the initial page without silently
        # selecting the first arbitrary search hit.
        if not q and suggested:
            filtered.sort(key=lambda i: i['node_id'] != suggested)
        return web.json_response({'vault_id': ctx.config.id, 'targets': filtered[:50],
                                  'suggested_target_node_id': suggested, 'suggested_target_error': error,
                                  'total_count': len(filtered), 'has_more': len(filtered) > 50})
    except MergeError as exc:
        return web.json_response({'error': str(exc)}, status=exc.status)


async def api_synthesis_merge_preview(request, app_state):
    try:
        data = await _body(request)
        ctx = _context(app_state, data)
        candidate = _candidate(ctx, data)
        if candidate.status not in {'pending_review', 'approved', 'pre_approved'}:
            raise MergeError(409, 'Candidate is no longer pending review')
        return web.json_response(_preview(ctx, candidate, _string(data, 'target_node_id')))
    except MergeError as exc:
        return web.json_response({'error': str(exc)}, status=exc.status)


async def api_synthesis_merge(request, app_state):
    try:
        data = await _body(request)
        ctx = _context(app_state, data)
        # Covers the full read/validate/write interval, not just the final write.
        async with ctx.runtime_lock:
            return web.json_response(_merge(ctx, _candidate(ctx, data), data))
    except MergeError as exc:
        return web.json_response({'error': str(exc)}, status=exc.status)
