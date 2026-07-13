#!/usr/bin/env bash
# BDH Graph Harness — semantic + structural nightly sleep.
# Runtime wrapper: ~/.hermes/scripts/bdh-consolidate.sh

set -euo pipefail

SERVER="${BDH_SERVER:-http://localhost:8643}"
VAULT_ID="${BDH_VAULT_ID:-core}"
MAX_SOURCES="${BDH_MAX_SOURCES:-}"
SEMANTIC_ENDPOINT="${SERVER}/api/semantic-consolidate"
REFRESH_ENDPOINT="${SERVER}/api/refresh-graph"
STRUCTURAL_ENDPOINT="${SERVER}/api/consolidate"

TMPDIR_BDH=$(mktemp -d)
trap 'rm -rf "$TMPDIR_BDH"' EXIT

if [[ -n "$MAX_SOURCES" ]]; then
    semantic_payload=$(printf '{"vault_id":"%s","max_sources":%s}' "$VAULT_ID" "$MAX_SOURCES")
else
    semantic_payload=$(printf '{"vault_id":"%s"}' "$VAULT_ID")
fi
structural_payload=$(printf '{"vault_id":"%s"}' "$VAULT_ID")

curl -sf "${SERVER}/api/graph?vault_id=${VAULT_ID}" >"$TMPDIR_BDH/before.json" || {
    echo "❌ BDH nightly sleep failed — unable to read graph before semantic sleep"
    exit 1
}

curl -sf -X POST "$SEMANTIC_ENDPOINT" \
    -H 'Content-Type: application/json' \
    -d "$semantic_payload" >"$TMPDIR_BDH/semantic.json" || {
    echo "❌ BDH semantic consolidation failed — server unreachable or endpoint error"
    exit 1
}

curl -sf -X POST "$REFRESH_ENDPOINT" \
    -H 'Content-Type: application/json' \
    -d "$structural_payload" >/dev/null || {
    echo "❌ BDH semantic consolidation completed, but graph refresh failed"
    exit 1
}

curl -sf -X POST "$STRUCTURAL_ENDPOINT" \
    -H 'Content-Type: application/json' \
    -d "$structural_payload" >"$TMPDIR_BDH/structural.json" || {
    echo "❌ BDH structural consolidation failed — server unreachable or endpoint error"
    exit 1
}

curl -sf "${SERVER}/api/graph?vault_id=${VAULT_ID}" >"$TMPDIR_BDH/after.json" || {
    echo "❌ BDH nightly sleep completed, but unable to read graph afterwards"
    exit 1
}

python3 - "$TMPDIR_BDH/before.json" "$TMPDIR_BDH/after.json" "$TMPDIR_BDH/structural.json" "$TMPDIR_BDH/semantic.json" <<'PY'
import json
import sys

before_path, after_path, structural_path, semantic_path = sys.argv[1:]
with open(before_path, encoding="utf-8") as f:
    before = {node["id"]: node for node in json.load(f).get("nodes", [])}
with open(after_path, encoding="utf-8") as f:
    after = {node["id"]: node for node in json.load(f).get("nodes", [])}
with open(structural_path, encoding="utf-8") as f:
    structural = json.load(f)
with open(semantic_path, encoding="utf-8") as f:
    semantic = json.load(f)

def format_nodes(nodes):
    if not nodes:
        return "nessuno"
    return ", ".join(
        f'{node.get("title") or node["id"]} [{node["id"]}]'
        for node in sorted(nodes, key=lambda item: (item.get("title", ""), item["id"]))
    )

added = [after[node_id] for node_id in sorted(set(after) - set(before))]
removed = [before[node_id] for node_id in sorted(set(before) - set(after))]

print('🌙 BDH semantic sleep')
print(f'fonti processate: {semantic.get("sources_processed", "?")}/{semantic.get("sources_discovered", "?")}')
print(f'concetti nuovi: {len(semantic.get("new_concepts", []))}')
print(f'Hebbian updates: {semantic.get("hebbian_updates", "?")}')
print(f'fonti fallite: {len(semantic.get("failed_sources", []))}')
print(f'🧹 BDH structural consolidation cycle #{structural.get("cycles", "?")} complete')
print(f'synapses: {structural.get("synapses_before", "?")} (pruned: {structural.get("synapses_pruned", "?")})')
print(f'nodi aggiunti: {format_nodes(added)}')
print(f'nodi rimossi: {format_nodes(removed)}')
PY
