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
