import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "bdh-nightly-sleep.sh"


class _FakeBDHHandler(BaseHTTPRequestHandler):
    graph_reads = 0

    def _json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/graph"):
            type(self).graph_reads += 1
            self._json({"nodes": [{"id": "node-a", "title": "Node A"}]})
            return
        self.send_error(404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if self.path == "/api/semantic-consolidate":
            self._json(
                {
                    "sources_processed": 1,
                    "sources_discovered": 1,
                    "new_concepts": [],
                    "hebbian_updates": 0,
                    "failed_sources": [],
                }
            )
            return
        if self.path == "/api/refresh-graph":
            self._json({})
            return
        if self.path == "/api/consolidate":
            self._json(
                {
                    "aborted": True,
                    "abort_reason": "candidate ratio exceeded safety gate",
                    "candidate_synapses": 93,
                    "candidate_prune_ratio": 0.7381,
                    "planned_prune_ratio": 0.1429,
                    "cycles": 14,
                }
            )
            return
        self.send_error(404)

    def log_message(self, format, *args):
        del format, args
        return


def test_nightly_sleep_reports_structural_abort_without_claiming_completion():
    _FakeBDHHandler.graph_reads = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeBDHHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = os.environ.copy()
        env["BDH_SERVER"] = f"http://127.0.0.1:{server.server_port}"
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert result.returncode == 0, result.stderr
    assert _FakeBDHHandler.graph_reads == 2
    assert "BDH structural pruning bloccato" in result.stdout
    assert "candidate ratio exceeded safety gate" in result.stdout
    assert "cycle #14 complete" not in result.stdout
