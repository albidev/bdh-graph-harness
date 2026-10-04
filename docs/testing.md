# Testing and Coverage

The project uses `pytest` for unit, regression, API, and integration-style tests.
External systems (Ollama, ChromaDB services, LLM providers, and network calls) are mocked unless a test explicitly creates an isolated local ChromaDB collection in `tmp_path`.

## Setup

```bash
pip install -r requirements-dev.txt
```

Use the project interpreter explicitly (`.venv/bin/python`). A shared or global
Python may resolve `mcp` 2.x, where `FastMCP` moved to `mcp.server.mcpserver`:
`requirements.txt` pins `mcp>=1.0,<2.0`, and `mcp_server.py` accommodates both
majors, but the suite is only verified against the pinned one.

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

Test totals and coverage are run-specific. Record the exact commit, dependency versions, command and complete output; do not treat an old count as the current baseline. [coverage.md](coverage.md) is a historical snapshot, not a fresh measurement. Unit mocks alone do not verify provider behavior: use the disposable [demo smoke check](quickstart.md) for a real read-only path, then separately exercise learning and restore.

The remaining integration work includes REST/WebSocket routes, CLI/MCP dispatch, graph/cache rebuilds, ChromaDB/embedding failure modes, and provider clients.

## Regression-test policy

Every bug fix gets a regression test that fails for the original behavior and asserts the repaired behavior. In particular, changes to multi-vault code must prove all of the following:

1. a request resolves only its selected `vault_id`;
2. embedding operations use that vault's `chroma_path` and collection name;
3. state writes and neurogenesis use that vault's filesystem path;
4. an unknown vault returns a useful error rather than silently falling back;
5. legacy single-vault configuration remains compatible.

Do not exclude application modules with `omit` or `# pragma: no cover` merely to improve a number. If a path is genuinely impossible to exercise in-process, document why and test the surrounding contract instead.
