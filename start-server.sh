#!/bin/bash
# BDH Graph Harness server launcher
# Expects OPENROUTER_API_KEY in env or in a .env file next to this script
if [ -z "$OPENROUTER_API_KEY" ]; then
  # Try .env in the script directory, then common locations
  for envfile in "$(dirname "$0")/.env" "$HOME/.env"; do
    if [ -f "$envfile" ]; then
      export OPENROUTER_API_KEY=$(grep OPENROUTER_API_KEY "$envfile" 2>/dev/null | grep -v '^#' | cut -d'=' -f2-)
      [ -n "$OPENROUTER_API_KEY" ] && break
    fi
  done
fi
cd "$(dirname "$0")"
# Use local config if it exists (not committed), otherwise the public one
CONFIG="bdh-config.yaml"
[ -f "bdh-config.local.yaml" ] && CONFIG="bdh-config.local.yaml"
exec python3 -m bdh_graph_harness --config "$CONFIG" --serve