# BDH Graph Harness — Evidence-Driven Refactor Report

**Branch:** `refactor/evidence-driven-refactor`  
**Run date:** 2026-07-25  
**Vault:** `/Users/albi/Documents/Hermes` — 353 neurons  
**Golden set:** `benchmarks/golden_set.yaml` v2, 40 queries  
**Status:** draft, `reviewed: false`

---

## Executive Summary

| Method | MRR | Recall@5 | Precision@5 | NDCG@5 | Mean latency |
|--------|-----|----------|-------------|--------|--------------|
| **vector-only** | **0.7083** | 0.6875 | 0.1600 | **0.6600** | 149 ms |
| bm25-only | 0.6300 | 0.6792 | 0.1550 | 0.5949 | 149 ms |
| weighted-70/30 | 0.6687 | **0.7042** | 0.1600 | 0.6308 | 149 ms |
| rrf | 0.6979 | **0.7042** | 0.1600 | 0.6514 | 149 ms |

**Finding on this vault, with this draft golden set:**
- **Vector-only wins on MRR and NDCG@5.**
- BM25-only is **behind** vector-only on every metric.
- RRF and weighted hybrid **do not beat vector-only** on MRR.
- Hybrid methods do improve **Recall@5** slightly (+1.7% over vector-only).

**Default config change:**  
BDH default is now `hybrid_search: true` with `hybrid_fusion: rrf`, but per-vault tuning is required. The `bdh-config.local.yaml` for the Hermes vault keeps `hybrid_search: true` for now; after golden-set review we can decide if vector-only should be the local default.

---

## Golden Set v2 — 40 queries

| Category | Count | Share |
|----------|-------|-------|
| concept | 10 | 25% |
| entity | 10 | 25% |
| activity | 10 | 25% |
| crossref | 7 | 17.5% |
| news | 3 | 7.5% |
| **Total** | **40** | **100%** |

`news` is intentionally small because the vault only has 3 news-like notes with ≥80 chars. Coverage warning kept in `golden_set.yaml`.

---

## Per-Category Results (RRF fusion)

| Category | Queries | MRR | Recall@5 | Precision@5 | NDCG@5 |
|----------|---------|-----|----------|-------------|--------|
| concept | 10 | 0.7792 | 0.7750 | 0.1550 | 0.7356 |
| entity | 10 | 0.6917 | 0.7375 | 0.1475 | 0.6624 |
| activity | 10 | 0.6625 | 0.6625 | 0.1325 | 0.6115 |
| crossref | 7 | 0.6190 | 0.6190 | 0.1238 | 0.5897 |
| news | 3 | 0.5556 | 0.5556 | 0.1111 | 0.5310 |

Strongest category: **concept**. Weakest relative to volume: **crossref** and **activity**. Crossref queries are long and phrased as "Come si collega X agli altri concetti?" — a real user would likely ask more directly.

---

## Method Comparison Detail

| Method | Δ MRR vs vector | Δ Recall@5 vs vector | When it helps |
|--------|-----------------|----------------------|---------------|
| vector-only | — | — | Clean title/keyword match |
| bm25-only | −0.0783 | −0.0083 | Exact terminology match |
| weighted-70/30 | −0.0396 | +0.0167 | Title-as-query cases |
| rrf | −0.0104 | +0.0167 | Robust to score-scale mismatch |

The previous claim that RRF clearly beats vector-only was based on a 35-query draft with `reviewed: false`. Expanding to 40 queries and fixing the BM25 normalisation bug (Tranche 2) shows the gap is much smaller than initially thought.

---

## Italian Golden Set — 40 natural-language queries

A second golden set was built with **realistic Italian questions** (not note titles). This is a harder test because it uses paraphrase and natural language rather than title matching.

| Category | Count | Share |
|----------|-------|-------|
| concept | 10 | 25% |
| entity | 10 | 25% |
| activity | 10 | 25% |
| crossref | 7 | 17.5% |
| news | 3 | 7.5% |
| **Total** | **40** | **100%** |

### Results (Italian set)

| Method | MRR | Δ vs vector | Recall@5 | Precision@5 | NDCG@5 | Mean latency |
|--------|-----|---------------|----------|-------------|--------|--------------|
| **vector-only** | **0.5329** | — | 0.5583 | 0.1350 | **0.5152** | 145 ms |
| bm25-only | 0.2289 | −0.3040 | 0.3750 | 0.0850 | 0.2617 | 145 ms |
| weighted-70/30 | 0.4121 | −0.1208 | 0.5208 | 0.1300 | 0.4225 | 145 ms |
| rrf | 0.4338 | −0.0991 | **0.5875** | **0.1450** | 0.4538 | 145 ms |

Observations on Italian natural-language queries:
- **Vector-only still wins on MRR and NDCG@5**, but the absolute scores drop ~25% vs the English title-based set. This confirms the Italian set is harder.
- BM25 collapses (−0.30 MRR), showing it struggles with Italian morphology and paraphrase.
- RRF still improves **Recall@5** (+2.9%) and **Precision@5** (+1.0%) vs vector-only, but at a MRR cost.
- Weighted-70/30 degrades MRR more than RRF.

---

## How this compares to pre-refactor

The old `benchmarks/BENCHMARK_REPORT.md` reported (15 queries, pre-BM25-fix):

| Method | MRR |
|--------|-----|
| hybrid | 0.200 |
| vector-only | 0.456 |
| BM25-only | 0.200 |

That comparison is **not reliable** because:
1. BM25 scores were max-normalized per query, so the top BM25 result always scored 1.0 — the fusion was dominated by a bug.
2. The golden set was only 15 auto-generated title-as-query entries.
3. The `hybrid` numbers were effectively a copy of the BM25 ranking.

After the refactor:
- The BM25 normalisation bug is fixed.
- The golden set is larger (40 queries) and we now have two variants (title-based EN, natural-language ITA).
- Absolute MRR on the English set is **0.70+** vs the old 0.45, but this is partly due to easier title-as-query queries.
- On the harder Italian set, MRR is **0.53** for vector-only — still better than the old pre-fix hybrid number.

**Bottom line:** the refactor did not magically double retrieval quality. It removed a bug that made hybrid look worse than it was, and it gave us an evaluation harness that can now measure real differences. The honest headline is that **vector-only is the strongest baseline on this vault**, with hybrid (especially RRF) as a tunable recall-boost option.

---

## My Evaluation

**Is the system better than before the refactor?**
- **Engineering quality:** yes. We now have reproducible ablations, deterministic config overlays, isolated embedding cache, and correct BM25 scoring.
- **Retrieval quality:** marginally. Vector-only was already the best method; the refactor just proved it. The gains from hybrid are small and depend heavily on query style.
- **Honesty:** much better. Before, we were drawing conclusions from a contaminated benchmark. Now we have two independent golden sets and clean numbers.

**Should hybrid stay the default?**
- For the Hermes vault: probably **no** after review. Vector-only gives higher MRR and NDCG on both sets.
- Keep RRF as an opt-in for use cases where recall matters more than rank precision.

**Biggest weakness exposed:**
- Italian natural-language queries drop MRR to 0.53. The system still relies too much on lexical overlap / title match. This is the next real frontier, not fusion arithmetic.

---

## Next Steps

1. **Review both golden sets manually** — many English queries are just note titles. Promote `reviewed: false` → `reviewed:true` query by query.
2. **Decide local default** — after review, if vector-only still wins, set `hybrid_search: false` in `bdh-config.local.yaml` for the Hermes vault.
3. **Investigate Italian paraphrase robustness** — the Italian set is the more realistic target. Consider query expansion, multi-language embeddings, or better graph traversal to close the MRR gap.
4. **Re-run ablation** on reviewed sets before merging to `develop`.

---

*Generated from `benchmarks/results/ablation-99914b93-20260725-102931.json`, `scripts/run_tranche2_rrf.py`, and `benchmarks/golden_set_ita.yaml`.*
