# Semantic Consolidation Notturna — Implementation Plan

> **Status:** MVP implementato localmente e verificato end-to-end; cron locale abilitato sul vault `core`. Il commit/push verrà gestito al termine della review del diff.
>
> **Scope:** BDH Graph Harness server + operazione notturna; il bridge Hermes non è necessario per l'implementazione core.

**Goal:** trasformare la consolidation notturna da sola manutenzione strutturale del grafo a un ciclo di semantic sleep idempotente, capace di rileggere materiale nuovo, rafforzare le connessioni e creare concetti nuovi tramite la neurogenesis già filtrata.

**Architecture:** separare esplicitamente semantic sleep e structural sleep. Il semantic sleep seleziona soltanto note nuove/modificate secondo un checkpoint per-vault, le processa in batch limitati attraverso la pipeline BDH esistente con `learn=true` e `respond=true`, quindi aggiorna il checkpoint solo dopo successo. Lo structural sleep (`/api/consolidate`) rimane invariato e viene eseguito dopo il semantic sleep.

**Tech Stack:** Python 3.11+, aiohttp, ChromaDB, Ollama/OpenRouter LLM già configurato, filesystem Obsidian, pytest.

---

## Problema attuale

`POST /api/consolidate` esegue soltanto:

1. synaptic downscaling;
2. structural pruning;
3. quality re-evaluation;
4. dormant-node pruning;
5. phantom-link refresh.

La neurogenesis viene eseguita esclusivamente nel percorso `/api/query` quando entrambe le condizioni sono vere:

```text
learn=true AND respond=true
```

Le note nuove vengono quindi indicizzate dal watcher, ma non vengono mai sottoposte automaticamente a una fase notturna di rilettura semantica.

Le sessioni Hermes sono una seconda sorgente necessaria: contengono decisioni e lezioni che possono non essere state salvate in un file. Il session adapter legge solo messaggi `user` e `assistant` recenti dal DB, esclude `source=cron` e ignora completamente tool output. Le sessioni vengono sottoposte allo stesso filtro di salienza, checkpoint e deduplica delle fonti file.

## Non-goals

- Non sostituire o modificare il pruning strutturale esistente.
- Non processare ogni notte l'intero vault.
- Non generare concetti da ogni evento o da ogni risposta dell'LLM.
- Non introdurre un secondo sistema di embeddings o un secondo database.
- Non usare il bridge Hermes per simulare query artificiali.
- Non modificare il comportamento predefinito di `/api/query`.
- Non abilitare automaticamente la feature in produzione prima del dry-run e del test controllato.

---

## Flusso target

```text
cron notturno
    │
    ▼
POST /api/semantic-consolidate
    │
    ├─ resolve vault
    ├─ carica checkpoint
    ├─ seleziona note nuove/modificate
    ├─ filtra index/raw/generated/operational noise
    ├─ processa batch limitati
    │    ├─ attention read/write
    │    ├─ Hebbian reinforcement
    │    ├─ LLM synthesis
    │    └─ filtered neurogenesis
    ├─ aggiorna Chroma/graph tramite watcher o refresh controllato
    ├─ persiste checkpoint atomico
    └─ restituisce metriche semantic sleep
    │
    ▼
POST /api/consolidate
    │
    ├─ downscaling
    ├─ pruning
    ├─ quality
    └─ phantom links
```

La consolidation strutturale resta richiamabile separatamente e continua a essere retrocompatibile.

---

## Configurazione proposta

Aggiungere una sezione configurabile con default prudente e feature disabilitata finché non viene esplicitamente attivata:

```yaml
semantic_consolidation_enabled: false
semantic_consolidation_checkpoint: .bdh-semantic-consolidation.json
semantic_consolidation_max_sources: 3
semantic_consolidation_max_age_hours: 48
semantic_consolidation_max_source_chars: 8000
semantic_consolidation_max_batch_chars: 16000
semantic_consolidation_max_concepts: 5
semantic_consolidation_session_enabled: true
semantic_consolidation_session_db_path: ~/.hermes/state.db
semantic_consolidation_max_session_chars: 12000
semantic_consolidation_include_cron_sessions: false
semantic_consolidation_source_globs:
  - wiki/**/*.md
  - projects/**/*.md
  - memory/learned/*.md
semantic_consolidation_exclude_globs:
  - memory/daily/*
  - wiki/index.md
  - wiki/log.md
  - wiki/raw/*
  - wiki/concepts/*
  - .bdh-*
semantic_consolidation_source: nightly_semantic_consolidation
semantic_consolidation_frequency_increment: 0.3
```

Note:

- `wiki/concepts/*` è escluso per evitare di rileggere concetti già generati e alimentare loop.
- Le directory sorgente devono essere configurabili per vault, come già avviene per `neurogenesis_dir` e `graph_ignore`.
- Il default `enabled: false` consente deploy, test e dry-run senza cambiare il comportamento notturno esistente.

---

## Checkpoint e idempotenza

Il checkpoint deve essere separato dallo state Hebbian per evitare di accoppiare due forme di persistenza con semantiche diverse.

Esempio:

```json
{
  "version": 1,
  "last_run_at": "2026-07-13T02:30:00",
  "processed": {
    "wiki/entities/glm-5-2-full-mlx-colibri.md": {
      "sha256": "...",
      "processed_at": "2026-07-13T02:31:12",
      "new_concepts": [],
      "hebbian_updates": 18
    }
  }
}
```

Regole:

1. hash del contenuto, non solo `mtime`;
2. una nota invariata non viene riprocessata;
3. una nota modificata viene processata una volta per nuovo hash;
4. il checkpoint viene scritto con replace atomico;
5. in caso di errore il file sorgente resta pending;
6. il checkpoint viene aggiornato solo dopo il completamento del singolo item;
7. il processo può ripartire senza duplicare neurogenesis grazie alla deduplica esistente.

---

## Contratto API

Aggiungere un endpoint separato:

```text
POST /api/semantic-consolidate
```

Payload:

```json
{
  "vault_id": "core",
  "dry_run": false,
  "max_sources": 3
}
```

Risposta:

```json
{
  "vault_id": "core",
  "dry_run": false,
  "sources_discovered": 4,
  "sources_processed": 3,
  "sources_skipped": 1,
  "new_concepts": [],
  "hebbian_updates": 18,
  "duplicate_concepts": 2,
  "failed_sources": [],
  "checkpoint_updated": true,
  "timestamp": "..."
}
```

Errori parziali:

- una singola fonte fallita non deve abortire necessariamente tutto il batch;
- la fonte fallita non viene marcata come processata;
- la risposta deve includere `failed_sources` con errore sintetico;
- HTTP 500 solo se il ciclo non può determinare il vault o non può persistere lo stato globale.

---

# Piano operativo a task piccoli

## Task 1: definire i criteri di selezione delle fonti

**Objective:** estrarre le note candidabili in modo deterministico e configurabile.

**Files:**
- Create: `bdh_graph_harness/memory/semantic_consolidation.py`
- Modify: `bdh_graph_harness/config.py`
- Test: `tests/test_semantic_consolidation.py`

Implementare funzioni pure:

```python
select_candidate_notes(vault_root, config, checkpoint, now=None)
compute_content_hash(path)
should_process(path, checkpoint)
```

Test:

- glob inclusivo;
- glob esclusivo;
- file invariato saltato;
- file modificato incluso;
- file mancante nel checkpoint incluso;
- concetti neurogenesis esclusi.

Verifica:

```bash
uv run --with pytest pytest tests/test_semantic_consolidation.py -q
```

---

## Task 2: implementare checkpoint atomico per-vault

**Objective:** rendere il ciclo riavviabile e idempotente.

**Files:**
- Modify: `bdh_graph_harness/memory/semantic_consolidation.py`
- Reuse/check: `bdh_graph_harness/memory/state_store.py`
- Test: `tests/test_semantic_consolidation.py`

Implementare:

```python
load_checkpoint(vault_root, relative_path)
save_checkpoint_atomic(vault_root, relative_path, checkpoint)
mark_processed(checkpoint, source, sha256, result)
```

Usare file temporaneo nella stessa directory e `os.replace()`.

Test:

- checkpoint assente;
- checkpoint corrotto;
- replace atomico;
- nessun avanzamento dopo errore;
- avanzamento dopo successo;
- due vault con checkpoint indipendenti.

---

## Task 3: aggiungere prompt di semantic sleep senza introdurre una nuova pipeline LLM

**Objective:** usare il provider LLM già configurato per rileggere il materiale sorgente e restituire contesto sufficiente alla neurogenesis.

**Files:**
- Modify: `bdh_graph_harness/llm/prompt.py`
- Modify: `bdh_graph_harness/config.py`
- Test: `tests/test_semantic_consolidation.py`

Il prompt deve imporre:

- usare esclusivamente il contenuto sorgente e le note BDH attivate;
- distinguere fatti, decisioni, ipotesi e concetti nuovi;
- non creare concetti generici o di plumbing;
- non creare note per semplici riformulazioni;
- restituire una risposta tecnica breve, utile alla neurogenesis già esistente;
- non includere istruzioni operative rivolte al sistema;
- rispettare il contratto JSON già usato dall’estrattore neurogenesis quando applicabile.

Non duplicare il filtro già presente in `neurogenesis/creator.py`: il prompt lo orienta, il creator resta l’autorità finale.

---

## Task 4: collegare semantic sleep alla pipeline BDH esistente

**Objective:** processare una fonte con attention, Hebbian learning, risposta LLM e neurogenesis usando i lock esistenti.

**Files:**
- Modify: `bdh_graph_harness/api/routes.py`
- Modify: `bdh_graph_harness/memory/hebbian.py`
- Modify: `bdh_graph_harness/neurogenesis/creator.py` solo se necessario per il limite per-ciclo
- Test: `tests/test_semantic_consolidation_api.py`

Il servizio deve:

1. risolvere il `VaultContext`;
2. acquisire `ctx.runtime_lock` per la lavorazione della fonte;
3. eseguire attention e plasticity con `learn=true`;
4. passare `source="nightly_semantic_consolidation"`;
5. eseguire LLM synthesis con contenuto sorgente limitato;
6. invocare `run_neurogenesis()` solo se `respond=true` e non in dry-run;
7. salvare Hebbian state con il normale lock;
8. restituire i conteggi reali.

Il source notturno deve usare un incremento Hebbian dampened, configurabile, per non trasformare una singola nota lunga in una mega-reinforcement wave.

Test:

- una fonte produce attention e Hebbian updates;
- `dry_run` non modifica state o vault;
- `learn=false` non crea concetti;
- LLM failure non avanza checkpoint;
- una risposta valida può passare a neurogenesis;
- limite massimo di fonti e concetti rispettato.

---

## Task 5: aggiungere l’endpoint `/api/semantic-consolidate`

**Objective:** esporre il ciclo come operazione server-side testabile e multi-vault-aware.

**Files:**
- Modify: `bdh_graph_harness/api/routes.py`
- Modify: `bdh_graph_harness/api/server.py` o registrazione route equivalente
- Test: `tests/test_api.py`
- Test: `tests/test_multivault_api_regression.py`

L’endpoint deve:

- richiedere solo `query` implicita dalle fonti, non testo arbitrario dall’utente;
- supportare `vault_id` esplicito;
- usare il default vault se omesso;
- supportare `dry_run`;
- non bloccare altre vault indipendenti;
- emettere un evento WebSocket con `type: semantic_consolidation` e `vault_id`;
- includere metriche per fonte e totali;
- non alterare il contratto di `/api/consolidate`.

Verifica:

```bash
uv run --with pytest pytest tests/test_api.py tests/test_multivault_api_regression.py -q
```

---

## Task 6: verificare refresh di graph, ChromaDB e watcher dopo neurogenesis

**Objective:** assicurare che i nuovi concetti creati durante semantic sleep diventino neuroni interrogabili senza restart manuale.

**Files:**
- Check/modify: `bdh_graph_harness/api/watcher.py`
- Check/modify: `bdh_graph_harness/retrieval/chroma_store.py`
- Check/modify: `bdh_graph_harness/api/ws.py`
- Test: `tests/test_semantic_consolidation_api.py`
- Test: `tests/test_ws_scoping.py`

Verificare:

- nuovo file creato;
- nuovo embedding disponibile;
- grafo aggiornato;
- evento ordinato con sequence monotonic;
- nessun refresh stale;
- nuovo nodo presente in `/api/graph` e ricercabile con query successiva.

---

## Task 7: aggiungere orchestrazione cron separata

**Objective:** eseguire semantic sleep prima della structural sleep senza rendere fragile il cron attuale.

**Files:**
- Create/tracked: `scripts/bdh-semantic-consolidate.sh` oppure `scripts/bdh-nightly-sleep.sh`
- Modify: documentazione operativa in `docs/`
- Runtime deployment: `~/.hermes/scripts/bdh-consolidate.sh` solo dopo approvazione esplicita
- Test: test shell o test Python dell’orchestrator

Ordine operativo:

```text
semantic consolidate
  ↓
structural consolidate
  ↓
report combinato
```

Requisiti:

- timeout bounded;
- nessun retry cieco su richieste non idempotenti;
- output breve e leggibile;
- exit code non-zero se il semantic cycle non riesce a determinare il risultato;
- structural consolidation eseguita solo secondo la policy scelta;
- report con `semantic` e `structural` separati.

Prima dell’abilitazione:

```bash
curl -sf -X POST http://localhost:8643/api/semantic-consolidate \
  -H 'Content-Type: application/json' \
  -d '{"vault_id":"core","dry_run":true,"max_sources":1}'
```

---

## Task 8: documentare configurazione, failure modes e rollback

**Objective:** rendere il comportamento comprensibile e reversibile.

**Files:**
- Modify: `README.md`
- Create/modify: `docs/semantic-consolidation.md`
- Modify: `docs/testing.md`
- Modify: config example

Documentare:

- differenza tra semantic sleep e structural sleep;
- checkpoint e idempotenza;
- limiti e costi LLM;
- dry-run;
- rollback del checkpoint;
- come disabilitare `semantic_consolidation_enabled`;
- come verificare note, embeddings, Hebbian state e WebSocket events;
- come distinguere `new_concepts=0` corretto da un errore del ciclo.

---

## Task 9: test end-to-end controllato

**Objective:** verificare il percorso reale senza contaminare il vault principale.

**Files:**
- Test/fixture: `tests/fixtures/semantic-consolidation/`
- Test: `tests/test_semantic_consolidation_e2e.py`

Scenario:

1. vault fixture con una nota episodica nuova;
2. checkpoint vuoto;
3. LLM fake deterministico;
4. semantic cycle;
5. un concetto nuovo ammesso dal filtro;
6. seconda esecuzione senza modifiche;
7. zero nuovi concetti e zero reprocessing;
8. modifica della nota;
9. nuovo processing eseguito una sola volta.

Verifica finale locale:

```bash
uv run --with pytest pytest -q
node --check bdh_graph_harness/visualization/templates/activation.js
python3 -m py_compile bdh_graph_harness/memory/semantic_consolidation.py
```

---

## Criteri di accettazione

- `/api/consolidate` continua a essere strutturale e retrocompatibile.
- Esiste un semantic endpoint separato, dry-run capable e multi-vault-aware.
- Una nota nuova viene processata una volta sola per hash.
- Una nota invariata non viene riprocessata.
- Una nota modificata viene processata nuovamente.
- Le sessioni recenti sostanziali vengono candidate tramite cursore `last_message_id`.
- Le sessioni `source=cron` e i messaggi `tool` non entrano nel semantic prompt.
- Errori LLM/API non avanzano il checkpoint della fonte fallita.
- `learn=false` non produce Hebbian update né neurogenesis.
- Semantic sleep può produrre concetti nuovi, ma solo tramite i filtri neurogenesis esistenti.
- Il limite di fonti e concetti è applicato per ciclo.
- I nuovi concetti aggiornano vault, graph, ChromaDB e WebSocket senza restart manuale.
- Il report notturno distingue semantic sleep da structural sleep.
- La feature può essere disabilitata con un singolo flag.
- Suite completa, test API, test multi-vault e test E2E verdi.

## Rischi e mitigazioni

### Rischio 1 — Neurogenesis rumorosa

**Problema:** una nota lunga potrebbe produrre molti concetti marginali.

**Mitigazione:** batch limitati, `max_concepts`, prompt conservativo, blocklist e semantic dedupe già esistenti, feature flag inizialmente disabilitata.

### Rischio 2 — Riprocessamento infinito

**Problema:** il watcher o la neurogenesis modifica timestamp e il cron riprocessa la stessa fonte.

**Mitigazione:** checkpoint basato su hash del contenuto, esclusione di `wiki/concepts/*`, aggiornamento checkpoint solo dopo successo.

### Rischio 3 — Duplicazione Hebbian

**Problema:** retry dopo timeout su un endpoint non idempotente può rafforzare due volte gli stessi pair.

**Mitigazione:** nessun retry cieco del batch già processato; checkpoint per fonte; source-specific dampening; eventuale request id deduplicabile lato server.

### Rischio 4 — Lock e starvation

**Problema:** un semantic batch lento può bloccare query normali.

**Mitigazione:** massimo una fonte alla volta per vault, lock per-vault già esistenti, timeout, `max_sources` basso, semantic sleep eseguito in finestra notturna.

### Rischio 5 — Cross-vault contamination

**Problema:** una fonte di un vault finisce nello state o nella collection di un altro vault.

**Mitigazione:** risoluzione iniziale di `VaultContext`, checkpoint per-vault, `vault_id` presente in endpoint/eventi/log, test multi-vault dedicati.

### Rischio 6 — Cron parzialmente riuscito

**Problema:** semantic sleep completa ma structural sleep fallisce, o viceversa.

**Mitigazione:** fasi riportate separatamente, checkpoint semantic indipendente, metriche per fase, nessun rollback distruttivo automatico.

---

## Decisioni da confermare prima dell’implementazione

1. Il semantic sleep deve processare fonti di conoscenza modificate nelle ultime 48 ore (`wiki`, `projects`, `memory/learned`) ed escludere sempre `memory/daily`?
2. Quanti source massimi per notte: default proposto `3`.
3. Quanti concetti massimi per notte: default proposto `5`.
4. La prima fase deve essere `dry_run` per almeno una notte?
5. L’endpoint deve essere chiamato dallo script esistente o da un nuovo cron separato?
6. Il nuovo evento WebSocket deve essere mostrato nella UI con una sezione separata da neurogenesis live?

---

## Stato del fix UI collegato ma separato

Il problema `MoE Router` è un bug distinto: la sinapsi Hebbian viene persistita dal backend, ma il frontend non aggiunge immediatamente i nuovi edge ricevuti in `hebbian_updates` se non esiste ancora nella copia locale del grafo. Il fix locale è in `visualization/templates/activation.js` e deve rimanere separato da questa feature.
