# Session synthesis activity and revert

The graph API exposes post-commit neurogenesis activity for Curate:

```text
GET  /api/synthesis-activity?vault_id=<configured-vault-id>
POST /api/synthesis/revert
```

The activity endpoint reads the selected vault's `.bdh-audit/synthesis.jsonl` and
`.bdh-audit/neurogenesis-operations.jsonl`. It returns the synthesis/session IDs,
provider/model, outcome, affected concept notes, and operation status. Vaults are
isolated: an operation recorded in `core` cannot be reverted through a
`crossnection` request.

## Reversible operations

Only session syntheses with operation journal data are revertible:

- `created`: the note is moved to `.bdh-audit/reverted/` and its `wiki/index.md`
  entry is removed. The archive is retained as evidence.
- `merged`: the pre-write note snapshot is restored.

Every merge writes its snapshot before the note mutation. Revert checks the
current note hash against the recorded `after_hash`; if a later edit is present,
it returns HTTP `409` and does not overwrite the note. Historical synthesis audit
entries created before the journal was deployed remain visible but are not made
artificially revertible.

After a successful revert the API returns `refresh_required: true`. Curate calls
`POST /api/refresh-graph` for the same vault before refreshing its activity view.
A prepared journal entry without a completed mutation is diagnostic only and is
not exposed as a revert action.

## Curate pre-write gate (session_synthesis staging + apply)

When `session_synthesis_staging_enabled: true`, `session_synthesis` writes are
routed through a human review gate instead of creating/merging notes directly.
Extraction and application are separated:

```text
POST /api/synthesis/stage      # extract concepts into pending candidate files
GET  /api/synthesis/candidates # list staged candidates for Curate review
POST /api/synthesis/apply      # apply one approved candidate (idempotent)
```

`POST /api/synthesis/stage` accepts `vault_id`, `synthesis_id`, `session_id`,
`transcript_sha256` (64-hex digest), `response_text`, `query`, `activated_notes`,
and `dry_run`. It extracts durable concepts and writes one candidate file per
concept under `.bdh-candidates/` (unless `dry_run`). Staging is side-effect
free: it never creates vault notes, merges evidence, mutates Hebbian state, or
records a `created`/`merged` synthesis outcome. Candidate metadata carries the
Curate correlation tuple (`candidate_id`, `synthesis_id`, `vault_id`,
`session_id`, `transcript_sha256`) plus provenance, but never the raw transcript.

`GET /api/synthesis/candidates` lists staged candidates, optionally filtered by
`status` and `synthesis_id`.

`POST /api/synthesis/apply` accepts only `candidate_id`, `synthesis_id`,
`session_id`, `vault_id`, and `source: "session_synthesis"` — never raw
transcript or concept text. It resolves the candidate from BDH-owned staged
data, correlation-checks it against the vault/synthesis/session, and applies it
exactly once (create/merge/noop) using the existing neurogenesis helpers and
operation journal. Repeating the identical apply returns the previous result
with no second note, Hebbian update, or journal row. Rejecting a candidate
(`update_candidate_status(..., "rejected")`) mutates only the candidate file and
never the vault or Hebbian state.

The Curate audit state machine (`.bdh-audit/curate.jsonl`) tracks each
candidate through `pending_review`, `rejected`, `created`, `merged`, `noop`,
`failed`, `reverted`, and `conflict`, keyed by `candidate_id` and correlated to
`synthesis_id`.
