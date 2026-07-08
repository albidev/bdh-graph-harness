#!/bin/bash
# BDH Graph Harness server launcher
# Expects OPENCODE_ZEN_API_KEY (preferred) or OPENROUTER_API_KEY in env or .env file
for varname in OPENCODE_ZEN_API_KEY OPENROUTER_API_KEY OLLAMA_API_KEY; do
  if [ -z "${!varname}" ]; then
    for envfile in "$(dirname "$0")/.env" "$HOME/.hermes/.env" "$HOME/.env"; do
      if [ -f "$envfile" ]; then
        val=$(grep "^${varname}=" "$envfile" 2>/dev/null | grep -v '^#' | head -1 | cut -d'=' -f2-)
        if [ -n "$val" ]; then
          export "$varname=$val"
          break
        fi
      fi
    done
  fi
done
cd "$(dirname "$0")"
# Use local config if it exists (not committed), otherwise the public one
CONFIG="bdh-config.yaml"
[ -f "bdh-config.local.yaml" ] && CONFIG="bdh-config.local.yaml"
# Prefer venv python if available, otherwise system python3
PYTHON="python3"
[ -x "$HOME/.hermes/hermes-agent/venv/bin/python" ] && PYTHON="$HOME/.hermes/hermes-agent/venv/bin/python"
exec "$PYTHON" -m bdh_graph_harness --config "$CONFIG" --serve