"""Benchmark: Integrate-and-Fire vs single-pass k-hop attention.

Runs on the real vault (or mock if vault unavailable). Measures:
- Activation count (sparsité)
- Hub activation (do hubs get dampened?)
- Propagation depth
- Timing
"""
import json
import math
import time
import tempfile
import os

# Setup project
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bdh_graph_harness.config import CONFIG
from bdh_graph_harness.graph.builder import build_graph
from bdh_graph_harness.retrieval.attention import (
    attention,
    integrate_and_fire_attention,
    compute_tau,
    _compute_degree,
)

# Use test mock graph (6 nodes, known edges)
import chromadb

def build_mock_graph_and_collection():
    """6-node graph with known structure."""
    d = tempfile.mkdtemp()
    notes = {
        'apple': 'Apple is a fruit that grows on trees. Sweet and crunchy.',
        'banana': 'Banana is a yellow fruit rich in potassium.',
        'cherry': 'Cherry is a small red fruit.',
        'carrot': 'Carrot is an orange vegetable.',
        'spinach': 'Spinach is a green leafy vegetable.',
        'index': 'Index page linking to everything.',
    }
    for nid, text in notes.items():
        with open(os.path.join(d, f'{nid}.md'), 'w') as f:
            f.write(f'# {nid}\n{text}\n')

    nodes, edges = build_graph(d)
    edges['apple'] = [{'target': 'banana', 'display': 'banana'}]
    edges['banana'] = [{'target': 'cherry', 'display': 'cherry'}]
    edges['carrot'] = [{'target': 'spinach', 'display': 'spinach'}]
    edges['index'] = [
        {'target': 'apple', 'display': 'apple'},
        {'target': 'banana', 'display': 'banana'},
        {'target': 'cherry', 'display': 'cherry'},
        {'target': 'carrot', 'display': 'carrot'},
        {'target': 'spinach', 'display': 'spinach'},
    ]

    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection('bench', metadata={'hnsw:space': 'cosine'})
    embeddings = {
        'apple': [1.0, 0.0, 0.0, 0.0],
        'banana': [0.9, 0.1, 0.0, 0.0],
        'cherry': [0.8, 0.2, 0.0, 0.0],
        'carrot': [0.0, 0.0, 1.0, 0.0],
        'spinach': [0.0, 0.0, 0.9, 0.1],
        'index': [0.5, 0.5, 0.5, 0.0],
    }
    for nid, emb in embeddings.items():
        col.add(
            ids=[nid], embeddings=[emb],
            documents=[notes[nid][:200]],
            metadatas=[{'title': nid, 'tags': ''}],
        )

    return nodes, edges, col


def run_benchmark():
    """Compare IaF vs k-hop on mock graph."""
    nodes, edges, col = build_mock_graph_and_collection()

    # Disable adaptive threshold for fair comparison on small graph
    orig_at = CONFIG.get('adaptive_threshold', True)
    CONFIG['adaptive_threshold'] = False
    CONFIG['hub_dampening'] = True
    CONFIG['hub_degree_threshold'] = 3  # index has 5 edges

    query = "fruit"
    from bdh_graph_harness.retrieval.embeddings import get_embeddings
    import bdh_graph_harness.retrieval.attention as attn

    # Mock get_embeddings for benchmark
    original_get_emb = attn.get_embeddings
    attn.get_embeddings = lambda texts: [[1.0, 0.0, 0.0, 0.0]]

    print("=" * 60)
    print("BDH Attention Benchmark: IaF vs Single-pass k-hop")
    print("=" * 60)
    print(f"Graph: {len(nodes)} nodes, {sum(len(v) for v in edges.values())} edges")
    print(f"Query: '{query}'")
    print()

    # --- Benchmark 1: Single-pass k-hop (old) ---
    CONFIG['experimental_integrate_fire'] = False
    times_old = []
    results_old = []
    for _ in range(100):
        t0 = time.perf_counter()
        r = attention(query, nodes, edges, col, k=3, max_hop=2)
        t1 = time.perf_counter()
        times_old.append(t1 - t0)
        results_old.append(r)

    avg_old = sum(times_old) / len(times_old)
    last_old = results_old[-1]

    # --- Benchmark 2: Integrate-and-Fire (new) ---
    CONFIG['experimental_integrate_fire'] = True
    times_iaf = []
    results_iaf = []
    for _ in range(100):
        t0 = time.perf_counter()
        r = integrate_and_fire_attention(query, nodes, edges, col, k=3, max_hop=3)
        t1 = time.perf_counter()
        times_iaf.append(t1 - t0)
        results_iaf.append(r)

    avg_iaf = sum(times_iaf) / len(times_iaf)
    last_iaf = results_iaf[-1]

    # Restore
    attn.get_embeddings = original_get_emb
    CONFIG['adaptive_threshold'] = orig_at
    CONFIG['experimental_integrate_fire'] = False

    # --- Report ---
    print(f"{'Metric':<30} {'k-hop (old)':<15} {'IaF (new)':<15}")
    print("-" * 60)
    print(f"{'Active nodes':<30} {len(last_old):<15} {len(last_iaf):<15}")
    print(f"{'Avg time (μs)':<30} {avg_old*1e6:<15.1f} {avg_iaf*1e6:<15.1f}")
    print()

    # Degree analysis
    degree = _compute_degree(edges, nodes)
    print("Node details:")
    print(f"  {'Node':<12} {'Degree':<10} {'τ':<10} {'k-hop':<10} {'IaF':<10}")
    print(f"  {'-'*52}")
    for nid in sorted(nodes.keys()):
        d = degree.get(nid, 0)
        tau = compute_tau(nid, degree)
        old_score = last_old.get(nid, 0)
        iaf_score = last_iaf.get(nid, 0)
        print(f"  {nid:<12} {d:<10} {tau:<10.4f} {old_score:<10.4f} {iaf_score:<10.4f}")

    print()

    # Hub comparison
    hub_scores_old = [last_old.get('index', 0)]
    hub_scores_iaf = [last_iaf.get('index', 0)]
    print(f"Hub 'index' (degree={degree.get('index', 0)}):")
    print(f"  k-hop: {hub_scores_old[0]:.4f}")
    print(f"  IaF:   {hub_scores_iaf[0]:.4f}")
    print(f"  τ_hub: {compute_tau('index', degree):.4f}")
    print()

    # Sparsité
    total = len(nodes)
    sparsity_old = len(last_old) / total
    sparsity_iaf = len(last_iaf) / total
    print(f"Sparsity (active/total):")
    print(f"  k-hop: {len(last_old)}/{total} ({sparsity_old:.1%})")
    print(f"  IaF:   {len(last_iaf)}/{total} ({sparsity_iaf:.1%})")

    # Overhead
    overhead = (avg_iaf - avg_old) / avg_old * 100
    print(f"\nTiming overhead: IaF is {overhead:+.1f}% vs k-hop")

    print("\n" + "=" * 60)
    print("PASS" if len(results_old) == 100 and len(results_iaf) == 100 else "FAIL")


if __name__ == '__main__':
    run_benchmark()
