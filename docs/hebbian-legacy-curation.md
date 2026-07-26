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

## Blind validation: curated v1

The first local A/B blind comparison paired each of the 16 curated edges against an embedding-neighbor alternative. Three independent reviewers chose the candidate that would add more useful associative context:

| Result | Count |
| --- | ---: |
| Curated chosen | 10 |
| Embedding chosen | 5 |
| Neither | 1 |

The curated lane won **66.7% of decisive comparisons** (10/15), but an exact binomial test was not significant (`p = 0.151` one-sided; `p = 0.302` two-sided). The correct product decision is therefore **do not activate curated v1**. It remains a local experimental artifact and a useful signal for a larger review, not a production memory source.

## Promotion rule

A curated artifact remains disabled until it passes an independent qualitative review against embedding-neighbor context. A curator must not promote it merely because its endpoints are semantically similar: the association must add useful future context rather than repeat or distract from primary retrieval.
