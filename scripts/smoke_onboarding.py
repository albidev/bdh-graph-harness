#!/usr/bin/env python3
"""Verify onboarding against an explicitly selected local BDH server, without learning.

Startup/index caches and observability are outside this check. It checks query counters,
Markdown notes, Hebbian state, staged candidates and audit artifacts in the supplied
vault. Do not edit the vault or run other clients during the check.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlencode, urlparse
from urllib.request import ProxyHandler, Request, build_opener


def learning_snapshot(vault):
    """Fingerprint learning artifacts without copying private note text into output."""
    result = {}
    for path in vault.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(vault)
        if path.suffix == '.md' or relative.as_posix() == '.bdh-state.json' or relative.parts[0] in {'.bdh-candidates', '.bdh-audit'}:
            result[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def run_smoke(base_url, vault, query, *, vault_id=None, timeout=30):
    """Run one read-only query. No retry, even when a response is ambiguous."""
    parsed = urlparse(base_url)
    if parsed.scheme != 'http' or parsed.hostname not in {'localhost', '127.0.0.1', '::1'} or parsed.username is not None or parsed.password is not None or parsed.path not in {'', '/'} or parsed.query or parsed.fragment:
        raise ValueError('Supply an explicit http loopback base URL without credentials or an API path')
    if not 0 < timeout <= 120:
        raise ValueError('timeout must be greater than zero and at most 120 seconds')
    vault = Path(vault).resolve()
    if not vault.is_dir():
        raise ValueError('The supplied vault directory does not exist')
    if not isinstance(query, str) or not query.strip():
        raise ValueError('A non-empty query is required')
    base_url = base_url.rstrip('/')

    def request(path, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        req = Request(base_url + path, data=body, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}, method='POST' if body is not None else 'GET')
        # Local verification must not inherit an ambient corporate/cloud proxy.
        with build_opener(ProxyHandler({})).open(req, timeout=timeout) as response:
            data = json.loads(response.read())
        if not isinstance(data, dict) or data.get('error'):
            raise RuntimeError('BDH did not return a successful structured response')
        return data

    stats_path = '/api/stats' + ('?' + urlencode({'vault_id': vault_id}) if vault_id is not None else '')
    before_files = learning_snapshot(vault)
    before = request(stats_path)
    if not isinstance(before.get('queries_processed'), int) or before.get('neurons', 0) <= 0:
        raise RuntimeError('Stats must expose a populated graph and queries_processed counter')
    payload = {'query': query, 'learn': False, 'respond': False}
    if vault_id is not None:
        payload['vault_id'] = vault_id
    response = request('/api/query', payload)
    after = request(stats_path)
    if response.get('hebbian_updates') or response.get('new_concepts') or response.get('response'):
        raise RuntimeError('Read-only query unexpectedly returned learning or a generated response')
    if not isinstance(response.get('activated_notes'), list):
        raise RuntimeError('Query response must contain activated_notes')
    if after.get('queries_processed') != before['queries_processed'] or learning_snapshot(vault) != before_files:
        raise RuntimeError('Learning artifacts or query counter changed during the read-only check')
    return {'read_only_verified': True, 'neurons': before['neurons'], 'activated_notes': len(response['activated_notes']), 'queries_processed': after['queries_processed'], 'learning_artifacts_checked': len(before_files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='Explicit local demo server, e.g. http://127.0.0.1:18643')
    parser.add_argument('--vault', required=True, type=Path, help='Local vault served by that server; use the copied demo vault')
    parser.add_argument('--query', default='How does retrieval use embeddings and wikilinks?')
    parser.add_argument('--vault-id', default=None)
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args()
    try:
        result = run_smoke(args.url, args.vault, args.query, vault_id=args.vault_id, timeout=args.timeout)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'Onboarding check failed: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
