"""Revision-bound advice records. Only the separate human merge request can write knowledge."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import tempfile

from aiohttp import web

from bdh_graph_harness.api.directed_merge import MergeError, _body, _candidate, _context, _preview, _string
from bdh_graph_harness.memory import curate_audit as audit

MAX_INPUT_CHARS = 60_000


def _assessment(value, preview):
    keys = {'classification', 'reason', 'candidate_quote', 'target_quote', 'provider', 'model'}
    if not isinstance(value, dict) or set(value) != keys:
        raise MergeError(400, 'Assessment requires classification, reason, exact quotes, provider and model')
    if value['classification'] not in ('compatible', 'conflicting', 'uncertain'):
        raise MergeError(400, 'Invalid assessment classification')
    for key, limit in (('reason',1000),('candidate_quote',1200),('target_quote',1200),('provider',200),('model',200)):
        if not isinstance(value[key], str) or not value[key].strip() or len(value[key]) > limit:
            raise MergeError(400, f'Invalid or oversized assessment {key}')
    for key, text in (('candidate_quote',preview['candidate']['definition']),('target_quote',preview['target']['content'])):
        if value[key] not in text:
            raise MergeError(400, f'{key} is not an exact quote from the reviewed text')
    return dict(value)


def _binding(candidate, preview):
    return {**candidate.correlation(), 'target_node_id': preview['target']['node_id'],
            'candidate_revision': preview['candidate_revision'], 'target_revision': preview['target_revision']}


def _record_path(ctx, rid):
    return Path(ctx.config.path) / '.bdh-audit' / 'merge-reconciliations' / (rid + '.json')


def register_assessment(ctx, candidate, data):
    for key in ('synthesis_id','session_id','source'):
        if _string(data,key) != getattr(candidate,key):
            raise MergeError(409, f'Candidate {key} does not match request')
    entry = audit.latest_curate_state(ctx.config.path,candidate.candidate_id)
    if entry is None or entry.correlation() != candidate.correlation() or entry.state != 'pending_review':
        raise MergeError(409, 'Candidate audit is not pending review or has inconsistent provenance')
    if candidate.status not in {'pending_review','approved','pre_approved'}:
        raise MergeError(409, 'Candidate is no longer pending review')
    preview = _preview(ctx,candidate,_string(data,'target_node_id'))
    for key in ('candidate_revision','target_revision'):
        if _string(data,key,pattern=r'[0-9a-f]{64}') != preview[key]:
            raise MergeError(409, 'Candidate or target changed; reload before analysis')
    if not preview['conflict']['required']:
        raise MergeError(409, 'This preview does not require reconciliation')
    if len(candidate.definition) + len(preview['target']['content']) > MAX_INPUT_CHARS:
        raise MergeError(413, 'Full texts exceed the comparison limit; nothing was truncated')
    opinion = _assessment(data.get('assessment'),preview)
    rid = 'rec-' + secrets.token_hex(16)
    record = {**_binding(candidate,preview), 'reconciliation_id':rid, 'assessment':opinion,
              'created_at':datetime.now(timezone.utc).isoformat()}
    path = _record_path(ctx,rid)
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            json.dump(record,tmp,ensure_ascii=False)
            tmp.flush(); os.fsync(tmp.fileno())
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)
    return record


def require_human_reconciliation(ctx,candidate,preview,data):
    if data.get('conflict_confirmed') is not True:
        raise MergeError(409,'Possible conflicting evidence needs a compatible assessment and separate human confirmation')
    rid = _string(data,'reconciliation_id',pattern=r'rec-[0-9a-f]{32}')
    try:
        record = json.loads(_record_path(ctx,rid).read_text(encoding='utf-8'))
    except (OSError,ValueError) as exc:
        raise MergeError(409,'Reconciliation record is missing or unreadable; analyze again') from exc
    if not isinstance(record,dict) or record.get('reconciliation_id') != rid or any(
        record.get(key) != value for key,value in _binding(candidate,preview).items()
    ):
        raise MergeError(409,'Reconciliation belongs to different texts, target, candidate or vault')
    opinion = _assessment(record.get('assessment'),preview)
    if opinion['classification'] != 'compatible':
        raise MergeError(409,'Conflicting or uncertain evidence cannot be merged')
    return {'reconciliation_id':rid,'conflict_confirmed':True}


async def api_synthesis_merge_reconciliation(request,app_state):
    try:
        data = await _body(request)
        ctx = _context(app_state,data)
        async with ctx.runtime_lock:
            return web.json_response(register_assessment(ctx,_candidate(ctx,data),data))
    except MergeError as exc:
        return web.json_response({'error':str(exc)},status=exc.status)
