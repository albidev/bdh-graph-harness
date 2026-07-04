"""Benchmark dataset — ground truth queries with expected note activations.

The dataset is semi-automatic: we sample real vault notes and create queries
that should activate specific notes. Each entry has:
  - query: the user question
  - expected_notes: set of note IDs that should be in the top-K activation
  - category: classification for per-category analysis
"""
import os
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Ground truth dataset — hand-curated from the real vault
# ---------------------------------------------------------------------------

DATASET = [
    # ── Concept queries (should activate concept notes) ──────────────────
    {
        "query": "Che cos'è l'Hebbian learning?",
        "expected_notes": {"wiki/concepts/hebbian-learning"},
        "category": "concept",
    },
    {
        "query": "Cos'è il modello Baby Dragon Hatchling?",
        "expected_notes": {"wiki/concepts/baby-dragon-hatchling"},
        "category": "concept",
    },
    {
        "query": "Come funziona l'architettura Transformer?",
        "expected_notes": {"wiki/concepts/transformer-architecture"},
        "category": "concept",
    },
    {
        "query": "Cos'è il graph-based retrieval?",
        "expected_notes": {"wiki/concepts/bdh-graph-harness"},
        "category": "concept",
    },
    {
        "query": "Cos'è il neurogenesis nel contesto dei grafi?",
        "expected_notes": {"wiki/concepts/neurogenesis"},
        "category": "concept",
    },

    # ── Activity queries (should activate session/daily notes) ───────────
    {
        "query": "Cosa abbiamo fatto sul BDH il 3 luglio 2026?",
        "expected_notes": {"memory/daily/2026-07-03"},
        "category": "activity",
    },
    {
        "query": "Racconta la sessione di deploy su Azure dell'assess-chat",
        "expected_notes": {"memory/sessions/assess-chat-2026-06-13"},
        "category": "activity",
    },
    {
        "query": "Cosa è successo con il gateway che girava come root?",
        "expected_notes": {"memory/daily/2026-07-03"},
        "category": "activity",
    },

    # ── News queries (should activate morning signal/trends) ─────────────
    {
        "query": "Quali notizie AI ci sono state il 1 luglio 2026?",
        "expected_notes": {"daily/morning-signal-2026-07-01"},
        "category": "news",
    },
    {
        "query": "Cosa trending su X il 2 luglio 2026?",
        "expected_notes": {"daily/morning-trends-2026-07-02"},
        "category": "news",
    },
    {
        "query": "Qual è stata la notizia principale del 24 giugno 2026?",
        "expected_notes": {"daily/morning-signal-2026-06-24"},
        "category": "news",
    },

    # ── Entity queries (should activate entity notes) ────────────────────
    {
        "query": "Cos'è l'Hermes Avatar Widget?",
        "expected_notes": {"wiki/entities/hermes-avatar-widget"},
        "category": "entity",
    },
    {
        "query": "Che cos'è il Privacy Guard?",
        "expected_notes": {"wiki/entities/privacy-guard"},
        "category": "entity",
    },

    # ── Cross-reference queries (should activate multiple notes) ─────────
    {
        "query": "Come si collega l'Hebbian learning ai Transformer?",
        "expected_notes": {
            "wiki/concepts/hebbian-learning",
            "wiki/concepts/transformer-architecture",
        },
        "category": "crossref",
    },
    {
        "query": "Qual è la relazione tra BDH e il neurogenesis?",
        "expected_notes": {
            "wiki/concepts/baby-dragon-hatchling",
            "wiki/concepts/neurogenesis",
        },
        "category": "crossref",
    },
]


def load_dataset():
    """Return the full benchmark dataset."""
    return DATASET


def get_categories():
    """Return unique categories in the dataset."""
    return list({e["category"] for e in DATASET})


def filter_by_category(category):
    """Return entries matching a specific category."""
    return [e for e in DATASET if e["category"] == category]
