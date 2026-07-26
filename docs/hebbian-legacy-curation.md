# Legacy Hebbian Curation

The original Hebbian state is an immutable historical artifact. It is never edited, deleted, or loaded by the clean-room learning policy.

## State namespaces

| State file | Purpose | Live status |
| --- | --- | --- |
| `.bdh-state.json` | Legacy association archive | Quarantined |
| `.bdh-state-primary-seeds-v2.json` | New conservative learning policy | Shadow learning |
| `.bdh-state-legacy-curated-v1.json` | Explicitly reviewed legacy edges only | Experimental; default-off |

## Curation pipeline

1. Validate both graph nodes and embeddings.
2. Reject pairs below the configured cosine similarity floor.
3. Require non-trivial historical evidence (weight threshold).
4. Send only surviving pairs to semantic adjudication.
5. Promote only explicit `keep` verdicts at or above the confidence threshold.
6. Preserve verdict, relation type, confidence, and rationale on every promoted edge.

The pipeline is read-only until a new curated artifact is generated. It never rewrites the legacy state.

## Rollback

The live configuration controls the active state by filename:

```yaml
hebbian_state_file: .bdh-state-primary-seeds-v2.json
```

To stop clean-room learning, point it to another state file and restart the BDH service. To revert to legacy behavior, use `.bdh-state.json`; this is intentionally a manual operational decision, not an automatic fallback.

## Promotion rule

A curated artifact remains disabled until it passes an independent qualitative review against embedding-neighbor context. A curator must not promote it merely because its endpoints are semantically similar: the association must add useful future context rather than repeat or distract from primary retrieval.
