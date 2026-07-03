"""
BDH Graph Harness — BM25 index for hybrid search.

Self-contained in-memory BM25 index. Only depends on `re` and `math`.
"""

import re
import math
from collections import defaultdict

import logging

_logger = logging.getLogger('bdh')


class BM25Index:
    """In-memory BM25 index over note texts.

    Built once from the graph nodes, supports keyword scoring
    for hybrid search (vector + keyword combination).
    """

    def __init__(self, nodes, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = {}       # note_id -> token list
        self.doc_len = {}    # note_id -> int
        self.avg_dl = 0.0
        self.tf = {}         # note_id -> {term: freq}
        self.df = defaultdict(int)  # term -> doc count
        self.idf = {}        # term -> idf value
        self.N = 0
        self._build(nodes)

    @staticmethod
    def _tokenize(text):
        """Simple tokenizer: lowercase, split on non-alphanumeric."""
        return re.findall(r'[a-z0-9]+', text.lower())

    def _build(self, nodes):
        """Build the BM25 index from graph nodes."""
        self.N = len(nodes)
        if self.N == 0:
            return

        total_len = 0
        for note_id, node in nodes.items():
            text = node.get('text', '')
            tokens = self._tokenize(text)
            self.docs[note_id] = tokens
            self.doc_len[note_id] = len(tokens)
            total_len += len(tokens)

            # Term frequencies
            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            self.tf[note_id] = dict(tf)

            # Document frequencies
            for term in tf:
                self.df[term] += 1

        self.avg_dl = total_len / self.N if self.N > 0 else 0.0

        # Compute IDF (BM25+ variant: idf = ln(1 + (N - df + 0.5) / (df + 0.5)))
        for term, df in self.df.items():
            self.idf[term] = float(math.log(1.0 + (self.N - df + 0.5) / (df + 0.5)))

        _logger.info(f"BM25 index built: {self.N} docs, {len(self.df)} unique terms, avg_dl={self.avg_dl:.1f}")

    def score(self, query, note_id):
        """Compute BM25 score for a single note against a query."""
        if note_id not in self.docs or self.N == 0:
            return 0.0

        query_terms = self._tokenize(query)
        if not query_terms:
            return 0.0

        dl = self.doc_len[note_id]
        score = 0.0
        tf_map = self.tf.get(note_id, {})

        for term in query_terms:
            if term not in self.idf:
                continue
            f = tf_map.get(term, 0)
            if f == 0:
                continue
            idf = self.idf[term]
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1.0))
            score += idf * f * (self.k1 + 1) / denom

        # Normalize to [0, 1] range (approximate)
        return min(score / 10.0, 1.0)

    def search(self, query, note_ids=None, top_k=None):
        """Score all (or subset of) notes against query.

        Returns list of (note_id, score) sorted descending.
        """
        results = []
        ids = note_ids if note_ids is not None else self.docs.keys()
        for nid in ids:
            s = self.score(query, nid)
            if s > 0:
                results.append((nid, s))
        results.sort(key=lambda x: -x[1])
        if top_k:
            results = results[:top_k]
        return results