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
- `requirements.txt` pins `mcp>=1.0,<2.0` to prevent SDK v2 breakages.

## What is NOT verified

- No automated tests on macOS or Windows.
- No automated security or performance benchmarks.
- No production deployment guide.
- The `qwen3:0.6b` model is a demonstration selection; no quality or speed promise is made.

## Migration notes

- Config: always use `--config bdh-config.local.yaml`; no implicit `bdh-config.yaml` precedence.
- `llm_provider` aliases: `ollama` (local native), `ollama-cloud` (OpenAI-compatible), `omlx` (local OpenAI-compatible loopback), `openrouter` (external OpenAI-compatible).
- `local_only` gate: only `ollama` or `omlx` with loopback endpoint is allowed.
