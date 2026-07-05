"""
BDH Graph Harness — BM25 index with Italian Snowball stemming.

Enhanced variant that uses Snowball stemming for Italian morphology.
Words like 'architettura' and 'architetture' share the same stem,
improving recall for morphologically rich languages.
"""
import re
import math
from collections import defaultdict

import logging
_logger = logging.getLogger('bdh')

# Try to load Snowball stemmer — graceful fallback if not installed
try:
    import Stemmer as _Stemmer
    _ITALIAN_STEMMER = _Stemmer.Stemmer('italian')
    _HAS_STEMMER = True
except ImportError:
    _ITALIAN_STEMMER = None
    _HAS_STEMMER = False
    _logger.warning("PyStemmer not installed — BM25 stemming disabled")

# Italian stop words — high-frequency words that match everywhere
# and pollute BM25 scores for Italian queries
_IT_STOP_WORDS = {
    # Articles
    'il', 'lo', 'la', 'i', 'gli', 'le', 'l', 'un', 'uno', 'una',
    # Prepositions
    'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra',
    # Contracted prepositions
    'del', 'dello', 'della', 'dei', 'degli', 'delle', 'al', 'allo',
    'alla', 'ai', 'agli', 'alle', 'dal', 'dallo', 'dalla', 'dai',
    'dagli', 'dalle', 'nel', 'nello', 'nella', 'nei', 'negli',
    'nelle', 'sul', 'sullo', 'sulla', 'sui', 'sugli', 'sulle',
    # Conjunctions
    'e', 'ed', 'o', 'ma', 'che', 'se', 'come', 'anche', 'perché',
    'quando', 'dove', 'mentre', 'però', 'anzi', 'oppure',
    # Pronouns
    'io', 'tu', 'lui', 'lei', 'noi', 'voi', 'loro', 'mi', 'ti',
    'ci', 'vi', 'si', 'me', 'te', 'ce', 've',
    # Demonstratives
    'questo', 'questa', 'questi', 'queste', 'quello', 'quella',
    'quelli', 'quelle',
    # Verbs (common auxiliary/linking)
    'è', 'e', 'essere', 'ha', 'ho', 'hanno', 'avere', 'sono',
    'era', 'può', 'fare', 'fatto',
    # Adverbs
    'non', 'più', 'già', 'ancora', 'sempre', 'molto', 'poco',
    'troppo', 'solo', 'anche', 'qui', 'là', 'ora', 'poi',
    # Misc
    'cosa', 'cos', 'qual', 'quale', 'quali',
    'dopo', 'prima', 'senza', 'sotto', 'sopra',
    # Common short tokens
    'the', 'and', 'or', 'is', 'of', 'to', 'in', 'for',
}


class BM25StemmedIndex:
    """In-memory BM25 index with Italian Snowball stemming.

    Built once from the graph nodes, supports keyword scoring
    for hybrid search (vector + keyword combination).
    Uses Snowball stemming for Italian morphological normalization.
    """
    def __init__(self, nodes, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = {}       # note_id -> token list (stemmed)
        self.doc_len = {}    # note_id -> int
        self.avg_dl = 0.0
        self.tf = {}         # note_id -> {term: freq}
        self.df = defaultdict(int)  # term -> doc count
        self.idf = {}        # term -> idf value
        self.N = 0
        self._stem_cache = {}  # original -> stemmed cache
        self._build(nodes)

    def _stem(self, word):
        """Stem a word using Italian Snowball stemmer."""
        if word in self._stem_cache:
            return self._stem_cache[word]
        if _HAS_STEMMER:
            stemmed = _ITALIAN_STEMMER.stemWord(word)
        else:
            stemmed = word
        self._stem_cache[word] = stemmed
        return stemmed

    @staticmethod
    def _tokenize(text):
        """Tokenize: lowercase, split on non-alphanumeric."""
        return re.findall(r'[a-z0-9]+', text.lower())

    def _tokenize_and_stem(self, text):
        """Tokenize, filter stop words, and apply Italian stemming."""
        tokens = self._tokenize(text)
        return [self._stem(t) for t in tokens if t not in _IT_STOP_WORDS and len(t) > 1]

    def _build(self, nodes):
        """Build the BM25 index from graph nodes."""
        self.N = len(nodes)
        if self.N == 0:
            return

        total_len = 0
        for note_id, node in nodes.items():
            text = node.get('text', '')
            tokens = self._tokenize_and_stem(text)
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

        # Compute IDF (BM25+ variant)
        for term, df in self.df.items():
            self.idf[term] = float(math.log(1.0 + (self.N - df + 0.5) / (df + 0.5)))

        _logger.info(f"BM25 stemmed index built: {self.N} docs, {len(self.df)} unique stems, avg_dl={self.avg_dl:.1f}")

    def score(self, query, note_id):
        """Compute BM25 score for a single note against a query."""
        if note_id not in self.docs or self.N == 0:
            return 0.0

        query_terms = self._tokenize_and_stem(query)
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

        return score

    def score_batch(self, query, note_ids=None):
        """Score all notes and return normalized scores [0, 1]."""
        ids = note_ids if note_ids is not None else list(self.docs.keys())
        raw_scores = {}
        for nid in ids:
            s = self.score(query, nid)
            if s > 0:
                raw_scores[nid] = s

        if not raw_scores:
            return {}

        max_score = max(raw_scores.values())
        return {nid: s / max_score for nid, s in raw_scores.items()}

    def search(self, query, note_ids=None, top_k=None):
        """Score all notes against query, returns sorted list."""
        results = []
        ids = note_ids if note_ids is not None else list(self.docs.keys())
        for nid in ids:
            s = self.score(query, nid)
            if s > 0:
                results.append((nid, s))
        results.sort(key=lambda x: -x[1])
        if top_k:
            results = results[:top_k]
        return results
