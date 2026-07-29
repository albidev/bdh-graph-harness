# Golden Set v4 (English) — Evaluation Contract

## Status

`golden_set_v4_en.yaml` is a **candidate** benchmark, not a promoted golden set.
It contains English-only queries grounded in the external Markdown notes indexed by
the live federated corpus at the snapshot declared in the YAML manifest.

## Composition

- 30 positive queries with graded qrels (`2` primary, `1` supporting)
- 5 near-domain negatives: plausible technical questions not documented by the corpus
- 5 out-of-domain negatives
- 22 development queries and 18 holdout queries
- English source documents across Privacy Guard, Tinka, BDH, MLX, Hermes Avatar
  Widget, and Assess Chat

## What it measures

1. **Positive retrieval:** MRR, Recall@5, NDCG@5, and Precision@5 over positive qrels.
2. **Rejection:** false-positive activation rate and empty-result correctness on negatives.
3. **Generalization:** development and holdout metrics must be reported separately.

Do not average positives and negatives into one retrieval score. A system can have a
strong rejection gate and weak positive retrieval, or the opposite.

## Validation gates before promotion

1. Every positive qrel resolves to exactly one live graph node at the frozen snapshot.
2. Every source claim is checked by a human reviewer against the cited note.
3. Near-domain negatives are searched manually to confirm that their absence is
   intentional, not an omitted qrel.
4. Each entry receives `reviewer_id` and `reviewed_at` before `reviewed: true`.
5. No source document may be primary for more than two positive entries.
6. No benchmark or evaluation report may be used to tune configuration against the
   holdout split.

## Query variants

The original query is authoritative. Query expansion experiments must use the same
frozen qrels and report a paired comparison against original-only retrieval. Variants
may improve recall but must not worsen negative rejection or holdout NDCG without an
explicit tradeoff decision.
