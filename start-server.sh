#!/bin/bash
# BDH Graph Harness server launcher
# Expects OLLAMA_API_KEY for Ollama Cloud, OPENCODE_ZEN_API_KEY for OpenCode,
# or OPENROUTER_API_KEY for the legacy OpenRouter provider.
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
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"
# Use local config if it exists (not committed), otherwise the public one.
CONFIG="$SCRIPT_DIR/bdh-config.yaml"
[ -f "$SCRIPT_DIR/bdh-config.local.yaml" ] && CONFIG="$SCRIPT_DIR/bdh-config.local.yaml"
# Use the project venv first so the checked-out package and dependencies match.
PYTHON="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON" ] && [ -x "$HOME/.hermes/hermes-agent/venv/bin/python" ]; then
  PYTHON="$HOME/.hermes/hermes-agent/venv/bin/python"
fi
[ -x "$PYTHON" ] || PYTHON="python3"
exec "$PYTHON" -m bdh_graph_harness --config "$CONFIG" --serve
