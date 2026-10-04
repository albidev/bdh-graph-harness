# Standalone demo quickstart

Start with fictional notes, not your real vault. **Hermes, Mission Control, Curate and Obsidian are not required.** Commands below use a POSIX shell and Python 3.11. Windows has not been validated.

## 1. Clone and install in a virtual environment

```bash
git clone https://github.com/albidev/bdh-graph-harness.git
cd bdh-graph-harness
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Install and start [Ollama](https://ollama.com/) separately, then select/pull both models:

```bash
ollama pull nomic-embed-text-v2-moe
ollama pull qwen3:0.6b
```

The first is the embedding model; the second is a small completion model for this demo, **not a quality recommendation**. You may select another available local completion model by editing the nested `llm.model` in your copied config. The embedding model has a short context; the demo notes are deliberately short.

## 2. Copy the demo into disposable storage

```bash
export BDH_DEMO_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/bdh-demo.XXXXXX")"
export BDH_DEMO_VAULT="$BDH_DEMO_ROOT/vault"
export BDH_DEMO_CHROMA="$BDH_DEMO_ROOT/chroma"
cp -R examples/demo-vault "$BDH_DEMO_VAULT"
cp examples/demo.yaml "$BDH_DEMO_ROOT/config.yaml"
python -m bdh_graph_harness --config "$BDH_DEMO_ROOT/config.yaml" --scan-sources
```

Expect a populated source scan with resolved links and no embeddings/LLM requests or persistent scan writes. Missing `${BDH_DEMO_VAULT}` / `${BDH_DEMO_CHROMA}` exports must be corrected before starting.

The demo binds to `127.0.0.1:18643`, selects local Ollama with `llm.local_only: true`, has no completion fallback chain, disables neurogenesis, and enables synthesis staging. **This config does not make every API operation read-only**: clients must set `learn:false` on each query. Startup itself writes caches.

## 3. Start the server in the foreground

```bash
python -m bdh_graph_harness --config "$BDH_DEMO_ROOT/config.yaml" --serve
```

Leave this terminal running. First startup computes embeddings. Wait for the ready/API message; if the port is occupied, stop here and choose another demo port in your copied config—do not kill an unrelated process.

## 4. Read-only verification in a second terminal

From the checkout, using the same venv and exported paths (copy the actual values from terminal 1):

```bash
curl --fail --max-time 10 http://127.0.0.1:18643/api/stats
python scripts/smoke_onboarding.py \
  --url http://127.0.0.1:18643 --vault "$BDH_DEMO_VAULT"
```

The smoke command requires a populated graph, sends **one** `learn:false, respond:false` query, then verifies unchanged query counter and vault learning artifacts (Markdown notes, `.bdh-state.json`, `.bdh-candidates`, `.bdh-audit`). Index/cache and observability writes are outside that check. It does not retry the POST. Do not edit the demo or run other clients during the check.

You can inspect the raw retrieval response:

```bash
curl --fail --max-time 30 http://127.0.0.1:18643/api/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"How does retrieval use embeddings and wikilinks?","learn":false,"respond":false}'
```

Expect `activated_notes` and empty `hebbian_updates` / `new_concepts`; the exact rankings are not a frozen golden answer.

## 5. Generate a response without learning (optional)

```bash
curl --fail --max-time 120 http://127.0.0.1:18643/api/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"How does retrieval use embeddings and wikilinks?","learn":false,"respond":true}'
```

Requires the completion model you pulled. The response comes from your configured local model. A provider error is not a successful smoke check; inspect server logs rather than substituting a plausible answer.

## 6. Explicitly try learning on the copied demo only

Stop the server with **Ctrl-C**, back up the entire demo directory, then restart it with the same `--config`:

```bash
# Only while the server is stopped:
cp -R "$BDH_DEMO_ROOT" "$BDH_DEMO_ROOT.before-learning"
```

After restart:

```bash
curl --fail --max-time 30 http://127.0.0.1:18643/api/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"retrieval embeddings wikilinks","learn":true,"respond":false}'
curl --fail --max-time 10 http://127.0.0.1:18643/api/stats
```

This exercises Hebbian learning without an LLM response. Neurogenesis remains disabled: the demo notes should not change. A populated activation normally creates/updates synapses and increments `queries_processed`. **Omitting `learn` means learning is enabled**, not a safe default. A timeout is ambiguous: do not blindly replay a mutating POST.

## 7. Stop and restore

Ctrl-C the foreground server. Preserve the learned copy before restoring:

```bash
mv "$BDH_DEMO_ROOT" "$BDH_DEMO_ROOT.after-learning"
cp -R "$BDH_DEMO_ROOT.before-learning" "$BDH_DEMO_ROOT"
python -m bdh_graph_harness --config "$BDH_DEMO_ROOT/config.yaml" --serve
```

Read `/api/stats` again to verify the pre-learning query count. Exported vault/cache paths still point to the restored directory. Keep both copies until verification is complete. Do not apply this disposable-directory recipe to a real vault without a proper backup.

## Using your own vault later

Copy `bdh-config.yaml` to `bdh-config.local.yaml`, edit storage/provider values, and **pass `--config bdh-config.local.yaml` to every command**. Automatic config discovery does not select that `.local` filename (the optional `start-server.sh` wrapper does).

CLI query text is positional: `python -m bdh_graph_harness --config bdh-config.local.yaml "your question"`. CLI/REPL queries use the learning pipeline; they are not substitutes for the read-only HTTP probe. `--refresh-embeddings` refreshes vectors; `--no-cache` skips the graph cache. Neither means "reset learned memory".

Next: [operations/privacy](operations.md), [compatibility](compatibility.md), [MCP](mcp-server.md), [human-confirmed directed merge](directed-merge-evidence.md). Curate is an optional external UI, not a demo prerequisite.
