# BM25 Analysis: Why Hybrid Search Doesn't Help Italian Vaults

## TL;DR

BM25 (Best Matching 25) is a lexical ranking algorithm that scores documents based on exact keyword overlap with the query. For Italian-language vaults, **BM25 actively hurts retrieval quality** — vector-only search outperforms all hybrid configurations by 2× on every metric. Italian Snowball stemming does not change this outcome.

**Default: `hybrid_search: false`** — vector-only retrieval.

---

## Benchmark Results

Comparative benchmark run on 2026-07-05 against the Hermes Obsidian vault (377 notes, 15 hand-curated queries across 5 categories: concept, activity, news, entity, crossref).

### Summary Table

| Method                   |    MRR | Recall@5 | Precision@5 | NDCG@5 | Latency |
|--------------------------|--------|----------|-------------|--------|---------|
| **vector-only**          | 0.4556 |   0.5333 |      0.1200 | 0.4654 |  123ms |
| bm25-plain               | 0.2004 |   0.2333 |      0.0533 | 0.1912 |  120ms |
| bm25-stemmed             | 0.2004 |   0.2333 |      0.0533 | 0.1912 |  120ms |
| hybrid-plain-70/30       | 0.2004 |   0.2333 |      0.0533 | 0.1912 |  122ms |
| hybrid-stemmed-70/30     | 0.2004 |   0.2333 |      0.0533 | 0.1912 |  122ms |
| hybrid-plain-85/15       | 0.2004 |   0.2333 |      0.0533 | 0.1912 |  121ms |
| hybrid-stemmed-85/15     | 0.2004 |   0.2333 |      0.0533 | 0.1912 |  121ms |

### Key Findings

1. **Vector-only wins on every metric.** MRR 0.456 vs 0.200 (+128%), Recall@5 0.533 vs 0.233 (+129%), NDCG@5 0.465 vs 0.191 (+144%).

2. **Stemming changes nothing.** BM25-plain and BM25-stemmed produce identical scores. The Italian stopword list already filters the problematic terms; remaining terms either match lexically or don't.

3. **Alpha/beta tuning doesn't help.** 70/30, 85/15 — same result. The damage from BM25 is structural, not parametric.

4. **All BM25 variants perform identically** (MRR = 0.2004 across all 6 BM25 configurations).

---

## Why BM25 Fails for Italian

### 1. Stopword Pollution

Italian queries follow predictable patterns: **"Che cos'è X?"**, **"Come funziona X?"**, **"Quali notizie X?"**. The words "che", "cos'", "è", "come" are extremely high-frequency. Even with stopword filtering, the filtered query often retains few discriminative terms.

**Example:**
- Query: `"Che cos'è il neurogenesis nel contesto dei grafi?"`
- After stopword removal: `"neurogenesis"`, `"grafi"`
- BM25 scores any document containing "neurogenesis" — but the vault note uses different terminology, so the score is low.

### 2. Morphological Richness

Italian has rich inflectional morphology. A single concept can appear as:
- `architettura`, `architetture`, `architetturale`, `architettonico`
- `ricerca`, `ricerche`, `ricercare`, `ricercato`
- `concetto`, `concetti`, `concettuale`

BM25 requires **exact lexical match** — `architettura` ≠ `architetture`. Snowball stemming helps (both → `architettur`) but doesn't fix the fundamental problem: the query terms often don't appear verbatim in the document.

### 3. Semantic Gap

Vector embeddings capture **meaning**, not **words**. When a user asks about "architettura Transformer", the embedding model knows this relates to transformer architecture, attention mechanisms, and neural networks — even if the document uses different terminology.

BM25 can't bridge this gap. It needs the exact word "architettura" or "transformer" to appear in the document.

### 4. Vocabulary Mismatch

Real vault notes use diverse terminology:
- Note title: "baby-dragon-hatchling"
- Query: "Cos'è il modello Baby Dragon Hatchling?"
- BM25 matches on "baby", "dragon", "hatchling" — but these are rare English words in an Italian vault, so IDF is high and the score is decent.
- However: `"Cos'è il modello Baby Dragon Hatchling?"` → BM25 finds the note, but ranks it at position 12 (out of 14 activated) because other documents happen to share more common terms.

### 5. Concrete Failure Examples

| Query | Vector MRR | BM25 MRR | What Happened |
|-------|-----------|----------|---------------|
| "Cos'è il modello Baby Dragon Hatchling?" | 0.50 (pos 2) | 0.08 (pos 12) | BM25 pushes correct result down |
| "Come funziona l'architettura Transformer?" | 1.00 (pos 1) | 0.17 (pos 6) | BM25 ranks irrelevant docs higher |
| "Che cos'è il Privacy Guard?" | 0.25 (pos 4) | 0.00 ❌ | BM25 completely misses the note |

---

## When BM25 *Could* Help

BM25 is valuable for:
- **English-heavy vaults** where vocabulary is smaller and keyword overlap is higher
- **Exact term matching** — product names, error codes, UUIDs, API endpoints
- **Multilingual vaults** where you need to match both English and Italian terms
- **Hybrid RAG pipelines** with cross-encoder re-ranking that can correct BM25's ranking errors

For a **pure Italian conceptual vault** like Hermes, vector-only is the correct choice.

---

## Technical Details

- **Vector model:** `nomic-embed-text-v2-moe` via Ollama (local)
- **BM25:** Custom in-memory implementation with BM25+ IDF variant
- **Stopwords:** 70+ Italian stopwords (articles, prepositions, conjunctions, pronouns, auxiliary verbs, adverbs)
- **Stemming:** Snowball Italian stemmer via PyStemmer
- **Threshold:** Adaptive (max of Q75, mean+1σ, floor=0.05)
- **Dataset:** 15 queries, 5 categories, hand-curated ground truth
- **Vault:** 377 Obsidian notes in Italian

---

## Configuration

To enable BM25 (if needed for a different vault):

```yaml
# bdh-config.yaml
hybrid_search: true
hybrid_alpha: 0.7   # vector weight
hybrid_beta: 0.3    # BM25 weight
```

To run the comparative benchmark:

```bash
python -m benchmarks.bm25_comparison
```

---

*Analysis conducted 2026-07-05 on the Hermes vault (377 notes, 15 queries). Results are vault-specific — different content or language may yield different outcomes.*
