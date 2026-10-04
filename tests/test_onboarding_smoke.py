"""Transport contract for the opt-in, read-only onboarding smoke command."""
import importlib.util
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / 'scripts' / 'smoke_onboarding.py'
    assert path.is_file(), 'An executable onboarding smoke command must be shipped'
    spec = importlib.util.spec_from_file_location('smoke_onboarding', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_uses_read_only_payload_and_checks_learning_artifacts(tmp_path):
    module = _module()
    vault = tmp_path / 'vault'
    vault.mkdir()
    note = vault / 'Retrieval.md'
    note.write_text('# Retrieval\nPublic test fixture.\n')
    seen = []
    state = {'mutate': False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _reply(self, data):
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            seen.append(('GET', self.path, None))
            self._reply({'neurons': 1, 'queries_processed': 0})

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(('POST', self.path, data))
            if state['mutate']:
                note.write_text('# Retrieval\nUnexpected write.\n')
            self._reply({'activated_notes': [], 'response': '', 'hebbian_updates': [], 'new_concepts': []})

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        url = f'http://127.0.0.1:{server.server_port}'
        result = module.run_smoke(url, vault, 'retrieval', vault_id='demo')
        assert result['read_only_verified'] is True
        payload = next(body for method, path, body in seen if method == 'POST')
        assert payload == {'query': 'retrieval', 'learn': False, 'respond': False, 'vault_id': 'demo'}
        state['mutate'] = True
        with pytest.raises(RuntimeError, match='changed'):
            module.run_smoke(url, vault, 'retrieval', vault_id='demo')
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_smoke_rejects_non_loopback_targets_before_opening_a_socket(tmp_path):
    module = _module()
    for target in ('https://example.com', 'http://127.0.0.1.evil.invalid:18643', 'http://user:pass@127.0.0.1:18643', 'http://127.0.0.1:18643/api/query'):
        with pytest.raises(ValueError, match='loopback'):
            module.run_smoke(target, tmp_path, 'retrieval')
