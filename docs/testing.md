# Testing and Coverage

The project uses `pytest` for unit, regression, API, and integration-style tests.
External systems (Ollama, ChromaDB services, LLM providers, and network calls) are mocked unless a test explicitly creates an isolated local ChromaDB collection in `tmp_path`.

## Setup

```bash
pip install -r requirements-dev.txt
```

## Commands

```bash
# Full test suite
python -m pytest -q

# Focus a module while iterating
python -m pytest -q tests/test_multivault_api_regression.py

# Statement + branch coverage for the package
python -m pytest -q \
  --cov=bdh_graph_harness \
  --cov-branch \
  --cov-report=term-missing
```

`--cov-branch` is intentional: a line-only percentage misses error handling, fallback paths, vault selection failures, and concurrent-update guards — exactly the code that tends to hurt later.

## Coverage status

The current `develop` baseline is **214 passing tests** and **49% package branch coverage**. This is an honest baseline, not a badge with lipstick on it.

Modules at 100% branch coverage:

- `memory/state_store.py` — corrupt-state recovery, atomic writes, cleanup, concurrent merge
- `memory/consolidation.py` — downscaling, pruning, dry-run and phantom-link paths
- `memory/quality.py` — dormancy and reactivation branches
- `retrieval/bm25.py` — empty queries, normalization, no-results behavior

The remaining work is concentrated in infrastructure boundaries: REST/WebSocket routes, CLI/MCP dispatch, graph/cache rebuilds, ChromaDB/embedding failure modes, and provider clients.

## Regression-test policy

Every bug fix gets a regression test that fails for the original behavior and asserts the repaired behavior. In particular, changes to multi-vault code must prove all of the following:

1. a request resolves only its selected `vault_id`;
2. embedding operations use that vault's `chroma_path` and collection name;
3. state writes and neurogenesis use that vault's filesystem path;
4. an unknown vault returns a useful error rather than silently falling back;
5. legacy single-vault configuration remains compatible.

Do not exclude application modules with `omit` or `# pragma: no cover` merely to improve a number. If a path is genuinely impossible to exercise in-process, document why and test the surrounding contract instead.
