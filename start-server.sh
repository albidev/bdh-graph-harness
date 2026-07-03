#!/bin/bash
# BDH Graph Harness server launcher — exports OpenRouter key and starts the server
# Expects OPENROUTER_API_KEY in ~/.hermes/.env or already in environment
if [ -z "$OPENROUTER_API_KEY" ]; then
  export OPENROUTER_API_KEY=$(grep OPENROUTER_API_KEY ~/.hermes/.env 2>/dev/null | grep -v '^#' | cut -d'=' -f2-)
fi
cd "$(dirname "$0")"
exec /Users/albi/.hermes/hermes-agent/venv/bin/python -m bdh_graph_harness --config bdh-config.yaml --serve