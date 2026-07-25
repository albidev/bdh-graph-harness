# BM25 Analysis: Hybrid Search in BDH Graph Harness

## TL;DR

Earlier analysis concluded that BM25 did not help the Italian Hermes vault.
That conclusion was contaminated by a bug: `BM25Index.score_batch()`
max-normalized every query so the top document always scored exactly 1.0,
while vector similarities occupied a narrow band (~0.45–0.65). The weighted
fusion `0.3 * 1.0` dominated the vector signal, so all hybrid variants
returned the BM25 ranking.

After switching to **Reciprocal Rank Fusion (RRF)** over a shared candidate
pool, hybrid search now beats vector-only on this vault.

**Default: `hybrid_search: true`, `hybrid_fusion: rrf`**.

---

## Benchmark Results

Comparative benchmark run on 2026-07-25 against the Hermes vault (353 notes,
35 draft queries across 5 categories). The golden set is still under review
(`reviewed: false`), so these numbers are directional, not final.

### Summary Table

| Method            |    MRR | Recall@5 | Precision@5 | NDCG@5 | Δ vs vector |
|-------------------|--------|----------|-------------|--------|-------------|
| **vector-only**   | 0.6143 |   0.6095 |      0.1429 | 0.5672 | — |
| bm25-only         | 0.7151 |   0.7143 |      0.1714 | 0.6462 | +0.1008 |
| weighted-70/30    | 0.7390 |   0.7143 |      0.1714 | 0.6605 | +0.1247 |
| **rrf**           | 0.7248 |   0.7714 |      0.1829 | 0.6823 | +0.1105 |

### Key Findings

1. **RRF improves over vector-only.** Recall@5 +26% (0.6095 → 0.7714) and
   NDCG@5 +20% (0.5672 → 0.6823). MRR is slightly below weighted-sum but
   still +11% over vector-only.
2. **BM25 alone is already competitive.** With proper normalization it is
   not the 0.2004 failure reported earlier.
3. **The old "all identical MRR" result was a bug.** Max-normalized BM25
   scores always hit 1.0 for the top document, drowning the vector signal.
4. **Weighted-sum still edges RRF on MRR** in this run, but RRF wins on
   recall and NDCG. The two fusion methods are close; we keep both under
   `hybrid_fusion` for now and will pick one after the golden set is
   finalized.

---

## Technical Details

- **Vector model:** `nomic-embed-text-v2-moe` via Ollama (local)
- **BM25:** Custom in-memory implementation with BM25+ IDF variant
- **Fusion:** Reciprocal Rank Fusion with `k = 60`, min-max normalized to
  keep scores comparable to the vector branch.
- **Threshold:** Adaptive (`median + 0.3*std`, floor=0.05)
- **Dataset:** 35 draft queries (review pending), 5 categories
- **Vault:** 353 Obsidian notes (Italian + English technical terms)

---

## Configuration

```yaml
# bdh-config.yaml
hybrid_search: true
hybrid_fusion: rrf   # or 'weighted'
rrf_k: 60
```

To run the comparative benchmark:

```bash
python scripts/run_tranche2_rrf.py
```

---

*Analysis updated 2026-07-25. Prior conclusions from 2026-07-05 were
invalidated by a normalization bug in the BM25 batch scorer.*
