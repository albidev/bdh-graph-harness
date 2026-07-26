# Associative Context Review Protocol

## Question

Does the Hebbian associative lane contribute useful context beyond direct retrieval, without introducing noise?

## What this does **not** measure

The generated review pack is edge-conditioned: it proves that learned non-static relations can be surfaced. It does not by itself prove response quality or general-user value.

## Review unit

Each record contains:

- a leakage-checked query generated from the source note;
- the primary seed that justified associative expansion;
- up to two associative candidates with source seed, weight, trust, and score;
- blank reviewer labels.

Do not judge an item by its score, title, or presumed retrieval lane. Judge only whether it would materially help answer the query.

## Labels

| Label | Meaning |
| --- | --- |
| `useful` | Adds material context, a relevant consequence, implementation detail, decision, or constraint. |
| `redundant` | Correct but already covered by direct retrieval or adds no material value. |
| `noise` | Unrelated or too generic to help. |
| `misleading` | Likely to steer the answer toward an incorrect or irrelevant conclusion. |

## Guardrails

- Keep `learn=false` for all evaluation retrieval.
- Primary ranking must be byte-for-byte invariant with the associative lane on/off.
- Record `reviewer_id` and `reviewed_at`; do not overwrite prior judgements.
- Before promotion, review a stratified sample across trust/weight deciles and query categories.

## Promotion criteria

1. Primary golden-set metrics remain invariant.
2. At least 60 independently judged candidates, including low-trust and high-trust strata.
3. `useful` rate beats static-neighbor and embedding-neighbor controls by a predeclared margin.
4. `misleading` rate is not above either control.
5. A paired answer-quality evaluation shows a positive effect before automatic bridge injection.
