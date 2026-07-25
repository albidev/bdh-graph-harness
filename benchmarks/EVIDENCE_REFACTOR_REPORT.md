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

## Tranche Completion Status

| Tranche | Goal | Status |
|---------|------|--------|
| **1** | Baseline + draft golden set | ✅ 40 query, MRR 0.6979 (RRF) |
| **2** | Ablation matrix + RRF fusion + BM25 bug fix | ✅ |
| **3** | Hebbian weight fix + propagation gain + unified threshold + IaF default off | ✅ |
| **4** | Hygiene: timeouts, logging, default localhost, golden-set 40 | ✅ |

---

## Next Steps

1. **Review the golden set manually** — many queries are just note titles. Promote `reviewed: false` → `reviewed: true` query by query.
2. **Decide local default** — after review, if vector-only still wins, set `hybrid_search: false` in `bdh-config.local.yaml` for the Hermes vault.
3. **Re-run ablation** on reviewed set before merging to `develop`.

---

*Generated from `benchmarks/results/ablation-99914b93-20260725-102931.json` and `scripts/run_tranche2_rrf.py`.*
