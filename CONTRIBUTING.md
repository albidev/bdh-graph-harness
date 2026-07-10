# Contributing

## Development setup

```bash
git clone https://github.com/albidev/bdh-graph-harness.git
cd bdh-graph-harness
pip install -r requirements-dev.txt
python -m pytest -q
```

## Architecture

The codebase is a Python package (`bdh_graph_harness/`) with 8 subpackages:

- `graph/` — vault parsing and graph construction (with incremental cache)
- `retrieval/` — embeddings, ChromaDB, BM25 (optional), attention spread
- `memory/` — Hebbian synaptic plasticity and persistent state
- `llm/` — provider abstraction (Ollama, OpenRouter), prompt building
- `neurogenesis/` — concept extraction and note creation in the vault
- `api/` — aiohttp REST + WebSocket server, scoped through `VaultRegistry`
- `vaults.py` — `VaultConfig`, `VaultContext`, and per-vault runtime isolation
- `mcp_server.py` — MCP thin client with an in-process fallback
- `visualization/` — force-graph WebGL real-time graph UI (see `docs/visualization.md`)
- `config.py` — YAML config loading with env var expansion

`harness.py` is a compatibility shim re-exporting from the package (tests use `import harness`).

## Running tests

```bash
# Full suite
python -m pytest -q

# Required before opening a PR that changes application code
python -m pytest -q --cov=bdh_graph_harness --cov-branch --cov-report=term-missing
```

The current `develop` baseline is 214 passing tests and 50% package branch coverage. Do not lower it. The project is working toward 100% by adding behavioral tests, not by omitting application modules from collection.

For commands, scope, and the regression-test policy, see [`docs/testing.md`](docs/testing.md).

## Adding a new LLM provider

1. Create `bdh_graph_harness/llm/<provider>.py` with a payload builder and stream token parser
2. Add the provider to the dispatch in `bdh_graph_harness/llm/providers.py`
3. Add config fields in `bdh-config.yaml`
4. Add tests in `tests/test_llm_providers.py`

## Code style

- Functional style, no classes unless state is genuinely needed
- Functions take explicit arguments, no hidden global state (use `_config.*` for runtime config access)
- Docstrings on public functions
- Tests use `pytest` with `monkeypatch` for mocking

## Pull requests

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/ -v`)
5. Open a PR with a clear description