# Operations and privacy

## Writes and data flow

- **Source scan** (`--scan-sources`) reads Markdown without embeddings, LLM calls or persistent scan writes.
- **Startup / indexing** creates graph caches and Chroma data, even if later queries use `learn:false`. Stopping can save current state. The file watcher can update indexes when notes change.
- **HTTP queries** default to `learn:true` and `respond:true`. Explicit `learn:false` suppresses query plasticity/neurogenesis; `respond:false` also skips response generation. This is not a filesystem-wide read-only mode.
- **CLI/REPL, maintenance, refresh, staging/apply** have their own writes. Do not infer permissions from a read-only query flag.

| Data | Destination / control |
|---|---|
| Markdown text for embeddings | Configured `ollama_url` (local in the demo; a remote URL sends text off-host) |
| Query, activated note context, response/extraction input | Selected completion provider; global, per-vault and `llm_source_overrides` affect routing |
| Session/room transcript | Harness when a bridge/watcher submits it; then the selected source-specific completion provider |
| Learning | Vault `.bdh-state.json`, candidates and audit files; neurogenesis can create/merge notes when enabled |
| Retrieval observability | Deployment-local logs / telemetry; do not treat `learn:false` as "no logs" |

A hash in synthesis audit metadata is **not a guarantee that transcript text never exists elsewhere**: Hermes session DBs, bridge durable buffers, HTTP payloads and model input can contain it. Set access/retention policy for all of them. There is no universal automatic purge guarantee.

### Enforce local completion deliberately

The demo uses nested `llm.local_only: true`. The resolver accepts only `ollama` or `omlx` with loopback endpoints. This gates completion routing, not every network operation (check embeddings URL and optional integrations separately). `session_synthesis` and `room_synthesis` suppress completion fallback, but **do not force a local provider or a specific model**.

For a source-specific local route, use a real model you have pulled:

```yaml
llm_source_overrides:
  session_synthesis:
    provider: ollama
    model: qwen3:0.6b
    base_url: http://127.0.0.1:11434
    local_only: true
    timeout: 120
```

Repeat explicitly for `room_synthesis` if you use it. oMLX is an optional local OpenAI-compatible backend: choose `provider: omlx`, your actual served model, and `base_url: http://127.0.0.1:8083/v1`. No Apple-only server is required on Linux. Use `api_key_env` for credentials, not literal keys in versioned YAML.

## Backup and restore

1. Stop **all writers**: Harness, bridge-owning processes, standalone watchers, any editor/sync job changing the vault.
2. Back up the **whole vault**, including hidden state, candidate files, audit/operation journals, reconciliation records and caches, plus the actual configured Chroma directory (which may be outside the vault), operator config and policy files. A Git archive is not a state backup.
3. Protect the backup as private data. Verify archive integrity and record hashes before relying on it.
4. Restore into a separate directory first, point copied config at it, and verify stats/note hashes while other writers remain stopped. Keep the original until verification succeeds.
5. When restoring in place, keep notes, candidate status, journal and audit from the same consistent snapshot. Restart only the owning processes after verification.

The [quickstart](quickstart.md) exercises stop → copy → explicit learning → stop → preserve learned copy → restore on a disposable demo. Do not delete a Chroma collection to diagnose a crash without a verified backup; it may be an index problem, an environment problem, or neither.

## Timeouts and uncertain writes

A client timeout on a mutating POST does not prove rejection. Do not blindly retry. Inspect the exact candidate / operation / audit entry and stats with GET requests. Query plasticity is not generally an idempotent API; a stats delta alone may not identify the request when other clients are active. Record uncertainty instead of announcing success.

For directed merge, use preview revisions, stored correlation, explicit human confirmation and the endpoint's idempotency contract. Machine reconciliation advice is not write authority. See [directed merge evidence](directed-merge-evidence.md).

## Troubleshooting

| Symptom | Check |
|---|---|
| Wrong vault/config | Pass explicit `--config`; verify exports, `vault_path`, Chroma path, vault ID and stats |
| No activated notes | Source scan / exclusions / unresolved links, embedding model availability, then thresholds; raising the threshold normally reduces matches |
| Slow first startup | First embedding/index build; inspect logs. Source scan does not warm embeddings |
| Missing completion model | Pull the selected Ollama model or use an actually served local model; retrieval-only queries do not test completion |
| MCP initializes a second graph | HTTP-first backend is unreachable; inspect API host/port/config. Direct fallback can write caches and perform learning |
| Chroma native crash | Capture exact command and `python -X faulthandler` output in an isolated copy; report dependency versions; no unproven attribution |
| Invalid query source | Use a registered source from `memory/source_policy.py`, or omit `source`; arbitrary labels are not supported |

## Network boundary

Keep the demo loopback-only. `api_auth_token` can require a bearer token on API routes; the unauthenticated demo is **not a production deployment**. Binding to `0.0.0.0` does not add TLS, authorization isolation or safe handling of untrusted clients. Review auth coverage, firewall, reverse proxy/VPN, rate limits and file access before remote exposure. A VPN is transport restriction, not a substitute for application trust policy.

Advanced references: [dynamic edges](hebbian-dynamic-edges.md), [MCP](mcp-server.md), [testing](testing.md). Mission Control / Curate are optional integrations maintained separately.
