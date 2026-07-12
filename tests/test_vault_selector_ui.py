"""Contract tests for the multi-vault visualization selector."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_SCRIPT = ROOT / "bdh_graph_harness/visualization/templates/vault-selector.js"


def _run_selector_contract(source: str) -> None:
    """Execute the browser-independent selector helpers in Node."""
    result = subprocess.run(
        ["node", "-e", source],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_vault_selector_builds_scoped_api_urls_and_filters_foreign_events():
    """A selected vault scopes requests and ignores another vault's live events."""
    assert SELECTOR_SCRIPT.exists(), "multi-vault selector script is missing"

    script_path = SELECTOR_SCRIPT.as_posix()
    _run_selector_contract(
        f"""
        const fs = require('fs');
        const vm = require('vm');
        const context = {{ window: {{ addEventListener() {{}} }}, console }};
        vm.createContext(context);
        vm.runInContext(fs.readFileSync('{script_path}', 'utf8'), context);

        context.setActiveVaultId('research');
        if (context.vaultApiUrl('/api/graph') !== '/api/graph?vault_id=research') process.exit(1);
        if (context.vaultApiUrl('/api/graph?detail=full') !== '/api/graph?detail=full&vault_id=research') process.exit(2);
        if (!context.isActiveVaultEvent({{ vault_id: 'research' }})) process.exit(3);
        if (context.isActiveVaultEvent({{ vault_id: 'hermes' }})) process.exit(4);
        if (!context.isActiveVaultEvent({{ type: 'ping' }})) process.exit(5);
        """
    )


def test_vault_selector_template_wires_the_control_and_scoped_transport():
    """The page loads the selector before the WebSocket/query transport modules."""
    html = (ROOT / "bdh_graph_harness/visualization/templates/index.html").read_text()
    websocket = (ROOT / "bdh_graph_harness/visualization/templates/websocket.js").read_text()
    styles = (ROOT / "bdh_graph_harness/visualization/templates/styles.css").read_text()
    controls = (ROOT / "bdh_graph_harness/visualization/templates/ui-controls.js").read_text()
    activation = (ROOT / "bdh_graph_harness/visualization/templates/activation.js").read_text()

    assert 'id="vault-selector"' in html
    assert '#vault-control[hidden]' in styles
    assert html.index('vault-selector.js') < html.index('websocket.js')
    assert 'vault_id: getActiveVaultId()' in websocket
    assert "vaultApiUrl('/ws')" in websocket
    assert 'isActiveVaultEvent(event)' in websocket
    assert 'let lastEventSequence = null;' in websocket
    assert 'lastEventSequence !== null' in websocket
    assert 'const type = info.type || l.type || \'wikilink\';' in controls
    assert 'edgeTypeVisible[type] === false' in controls
    assert 'following graph_refresh/node_update' in activation
    assert 'Do not rebuild force-graph here' in activation
