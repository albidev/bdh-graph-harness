# Coverage Baseline

> **Snapshot:** `develop` at `4741d46`, measured locally with Python 3.11 and `pytest-cov` on 10 July 2026. The authoritative result for every later commit is the **Test and coverage** GitHub Actions run and its attached `coverage.xml` / `coverage.json` artifact.

## Verified run

```text
214 passed in 1.86s
TOTAL  2875 statements · 1377 missed · 986 branches · 87 partial branches
Branch coverage: 50%
```

Command:

```bash
python -m pytest -q \
  --cov=bdh_graph_harness \
  --cov-branch \
  --cov-report=term-missing \
  --cov-report=xml:coverage.xml \
  --cov-report=json:coverage.json
```

## Fully covered modules

| Module | Statements | Branches | Coverage |
|---|---:|---:|---:|
| `memory/state_store.py` | 58 | 16 | 100% |
| `memory/consolidation.py` | 87 | 34 | 100% |
| `memory/quality.py` | 61 | 18 | 100% |
| `retrieval/bm25.py` | 91 | 36 | 100% |

## Why this file exists

This is the versioned starting line for the coverage programme: it records the suite, command, result, and modules already closed. It is **not** a fake badge and it is not a replacement for CI.

The GitHub Action runs the same command on pushes and pull requests to `develop` and `main`, then retains the machine-readable XML and JSON reports for 90 days. Update this snapshot when a deliberate coverage milestone is reached; otherwise let CI be the source of day-to-day truth.

See [`testing.md`](testing.md) for the test and regression policy.
