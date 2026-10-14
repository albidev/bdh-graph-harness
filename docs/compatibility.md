# Compatibility

## Supported platforms

The test suite and CI run on:

- **Linux** (`ubuntu-latest`)
- **Python 3.11** (verified)

No claims for macOS, Windows, or other Python versions are made from CI.
The code uses standard Python 3.11 features (match statements are not required; the code avoids them).

## What is verified

- `tests/` runs on Python 3.11 with `pytest`, branch coverage enabled.
- `.github/workflows/test-and-coverage.yml` pins `python-version: "3.11"`.
- `requirements.txt` requires `mcp>=2.3.0,<3.0`. `bdh_graph_harness/mcp_server.py` imports the SDK v2 `MCPServer` from `mcp.server.mcpserver` (aliased locally as `FastMCP`); it also retains the legacy SDK v1 import path as a compatibility fallback.
- `tests/test_mcp_sdk_compat.py` checks the resolved server import, required transports, and registered tools against the installed SDK. CI currently installs the declared SDK v2 range; it does not run a separate SDK v1 test job.

## What is NOT verified

- No automated tests on macOS or Windows.
- No SDK v1 test job or MCP SDK version matrix; only the dependency resolved from `mcp>=2.3.0,<3.0` is exercised by CI.
- No automated security or performance benchmarks.
- No production deployment guide.
- The `qwen3:0.6b` model is a demonstration selection; no quality or speed promise is made.

## Migration notes

- Config: always use `--config bdh-config.local.yaml`; no implicit `bdh-config.yaml` precedence.
- `llm_provider` aliases: `ollama` (local native), `ollama-cloud` (OpenAI-compatible), `omlx` (local OpenAI-compatible loopback), `openrouter` (external OpenAI-compatible).
- `local_only` gate: only `ollama` or `omlx` with loopback endpoint is allowed.
