# Human-confirmed directed synthesis merge

Curate keeps three distinct actions: **Approve** uses BDH's automatic destination
selection, **Merge into…** adds evidence to a human-selected existing note, and
**Reject** records feedback. A candidate carrying `extra.curator_merge_target`
must not pass through automatic apply; `/api/synthesis/apply` returns HTTP 409
rather than silently creating a second note.

## BDH API

| Method | Path | Inputs |
|---|---|---|
| GET | `/api/synthesis/merge-targets` | `vault_id`, `candidate_id`, optional `q` |
| POST | `/api/synthesis/merge-preview` | `vault_id`, `candidate_id`, `target_node_id` |
| POST | `/api/synthesis/merge` | Above identity, `synthesis_id`, `session_id`, `source`, both revision hashes, `confirmed: true` |

The Curate plugin exposes corresponding authenticated paths under
`/api/local/synthesis/`. It derives correlation fields from the stored candidate,
not from untrusted browser values. See the sibling `mc-curate-plugin` README.

### Selection and preview

Targets come from the selected vault's live graph. The response contains a
maximum of 50 entries, exact `total_count`, and `has_more`; use `q` to narrow by
title or vault-relative filename. Only existing writable vault-local Markdown
notes are allowed. External nodes, path escapes, symlink targets, missing files,
`index.md`, and `log.md` are excluded. Browsers submit a graph node ID, not a path.

A curator hint is matched literally against title, relative path, basename or
slug, in that order. Missing or ambiguous hints return a warning and no suggested
selection. They never cause fuzzy target selection or note creation. The human
may choose another valid note explicitly.

Preview returns existing target text, candidate definition, and SHA256 hashes of
the candidate's review-relevant content/provenance and target text. Listing and
preview are read-only. Pending, pre-approved, and approved candidates can be
reviewed; none bypasses the explicit confirmation requirement.

### Confirmation and durability

BDH re-resolves the vault, candidate, target and revisions at confirmation, inside
the vault runtime lock. Invalid identity/confirmation yields 400; unavailable
candidates or nodes yield 404; disallowed targets yield 403; changed revisions,
correlation mismatch, terminal decisions or possible conflict yield 409. Invalid
requests do not mutate the note, candidate or durable ledgers.

`assimilate_evidence()` preserves existing note content and appends synthesis
provenance. A real merge returns `status: merged`, `applied: true`, the relative
`note_path`, and a reversible `operation_id`. Already-present evidence returns
`status: noop`, `applied: false`, without a note write or operation record. Both
outcomes close the Curate audit and staged candidate.

The audit binds idempotent retries to the original target and revision hashes;
two matching concurrent submissions write once. A retry with another target or
reviewed revision is rejected. An audit-write failure rolls back the note via
the operation journal rather than leaving an untracked merge.

Graph/file watchers ingest the changed note as usual. Directed merge does not
run automatic note creation, semantic destination selection, or Hebbian learning.
Existing candidates without a target hint keep the automatic Approve path.

## Reproducible verification

Run from the BDH repository with its pytest/aiohttp environment:

```bash
python3 -m pytest tests/test_directed_merge.py \
  tests/test_curate_directed_merge_integration.py \
  tests/test_synthesis_apply_invariants.py -q \
  --basetemp="${TMPDIR:?Set a scratch directory}/curate-merge-tests"
```

The integration test requires sibling Curate and Mission Control checkouts,
overridable with `CURATE_PLUGIN_REPO` and `CURATE_MC_ROOT`. It uses MC's real
manifest loader, real HTTP adapter, real BDH routes, and a temporary vault.
It checks confirmation, authentication metadata, stale revisions, automatic
Approve rejection, merge read-back, idempotency and revert read-back. It does not
restart a production service.

The separate Curate browser harness renders the real UI against intercepted
API fixtures. It covers desktop confirmation, cancel/Escape, missing hints and
bounded selection, stale-preview errors, mobile detail and duplicate clicks.
Build the host into a scratch `--outDir`, not its live `dist/`.

**Test isolation includes runtime home/configuration, not just `tmp_path`.**
Legacy tests that load the user's `HERMES_HOME` or global multi-vault configuration
can reach unrelated candidate directories or live providers despite a temporary
primary vault. Curate's pytest configuration forces a scratch home and a disabled
BDH address; gate invariant fixtures use an explicit temporary single-vault
configuration and stub extraction/dedupe.
