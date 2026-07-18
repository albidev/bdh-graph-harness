#!/usr/bin/env python3
"""E2E test: vault switch produces visible nodes, even for vaults with 0 edges.

Requires the server running on 0.0.0.0:8643.
Run: /tmp/bdh-viz-check-venv311/bin/python tests/e2e_vault_switch.py
"""
import sys
import time
import json
from playwright.sync_api import sync_playwright

BASE = "http://100.84.148.17:8643"
CACHE = "3d-v59"

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        console_logs = []
        page.on("console", lambda msg: console_logs.append(f"[{msg.type}] {msg.text}"))

        page_errors = []
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        print("1. Loading page with default vault (core)...")
        page.goto(f"{BASE}/?cache={CACHE}", wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(8000)  # let Three.js CDN + warmup settle

        stats_core = page.text_content("#stat-neurons")
        edges_core = page.text_content("#stat-synapses")
        print(f"   Core vault: {stats_core} nodes, {edges_core} edges")

        # Check if the graph canvas exists and has content
        canvas = page.query_selector("#graph-container canvas")
        canvas_exists = canvas is not None
        print(f"   Canvas exists: {canvas_exists}")

        if not canvas_exists:
            print("   FAIL: No canvas found — 3D renderer did not boot")
            browser.close()
            return False

        print("2. Switching to episodic vault (0 edges)...")
        selector = page.query_selector("#vault-selector")
        if not selector:
            print("   FAIL: vault selector not found")
            browser.close()
            return False

        selector.select_option(label="Hermes Episodic")
        page.wait_for_timeout(5000)  # wait for fetch + initNetwork

        stats_episodic = page.text_content("#stat-neurons")
        edges_episodic = page.text_content("#stat-synapses")
        print(f"   Episodic vault: {stats_episodic} nodes, {edges_episodic} edges")

        # Check console logs for our diagnostic messages
        fetch_logs = [l for l in console_logs if "fetchGraphSnapshot" in l]
        print(f"   Fetch logs: {len(fetch_logs)}")
        for l in fetch_logs[-3:]:
            print(f"     {l}")

        snapshot_logs = [l for l in console_logs if "graph snapshot received" in l]
        for l in snapshot_logs[-2:]:
            print(f"     {l}")

        # Check the HUD text — it shows "N nodes · M edges visible"
        hud = page.text_content("#scene-hud") or ""
        print(f"   HUD: {hud.strip()[:120]}")

        # Verify the graph actually rendered nodes
        # We check via JS evaluation if fgNodes is populated
        js_result = page.evaluate("""() => {
            try {
                return JSON.stringify({
                    fgNodes: typeof fgNodes !== 'undefined' ? fgNodes.length : 'undef',
                    fgLinks: typeof fgLinks !== 'undefined' ? fgLinks.length : 'undef',
                    graphData: (typeof graph !== 'undefined' && graph && graph.graphData) ?
                        graph.graphData().nodes.length : 'no-graph',
                    sourceGraphData: (typeof sourceGraphData !== 'undefined' && sourceGraphData) ?
                        sourceGraphData.nodes.length : 'null',
                    activeVaultId: typeof activeVaultId !== 'undefined' ? activeVaultId : 'undef'
                });
            } catch(e) { return 'ERROR: ' + e.message; }
        }""")
        print(f"   JS state: {js_result}")

        result = json.loads(js_result) if js_result.startswith("{") else {}

        # Assertions
        passed = True

        if result.get("activeVaultId") != "episodic":
            print("   FAIL: activeVaultId is not 'episodic'")
            passed = False

        if result.get("sourceGraphData") != 151:
            print(f"   FAIL: sourceGraphData has {result.get('sourceGraphData')} nodes, expected 151")
            passed = False

        if result.get("fgNodes") != 151:
            print(f"   FAIL: fgNodes has {result.get('fgNodes')} nodes, expected 151")
            passed = False

        if result.get("graphData") != 151:
            print(f"   FAIL: graph.graphData() has {result.get('graphData')} nodes, expected 151")
            passed = False

        if page_errors:
            print(f"   Page errors: {page_errors}")
            passed = False

        if passed:
            print("\n✅ E2E PASS: Vault switch to episodic shows 151 nodes with 0 edges")
        else:
            print("\n❌ E2E FAIL")

        browser.close()
        return passed


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)