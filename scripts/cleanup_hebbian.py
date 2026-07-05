"""Cleanup spurious Hebbian synapses.

Removes synapses where:
1. Frequency = 1 (only co-activated once)
2. Weight < 0.5 (never reinforced)
3. No direct wikilink exists between the two nodes

Strong synapses (f>1 or w>0.5) are preserved even if spurious —
they've been reinforced by repeated co-activation.
"""
import json
import sys
from pathlib import Path


def load_state(vault_root):
    state_path = Path(vault_root) / ".bdh-state.json"
    if not state_path.exists():
        print(f"No state file at {state_path}")
        sys.exit(1)
    with open(state_path) as f:
        return json.load(f), state_path


def save_state(state, state_path):
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def load_graph_edges(vault_root):
    """Load wikilink edges to know which node pairs are directly connected."""
    cache_path = Path(vault_root) / ".bdh-graph-cache.json"
    if not cache_path.exists():
        print(f"No graph cache at {cache_path} — cannot check wikilinks")
        return set()
    with open(cache_path) as f:
        graph = json.load(f)
    edges = graph.get('edges', {})
    connected = set()
    for src, targets in edges.items():
        for t in targets:
            target_id = t['target'] if isinstance(t, dict) else t
            connected.add(f"{src}|{target_id}")
            connected.add(f"{target_id}|{src}")
    return connected


def cleanup(vault_root, dry_run=False):
    state, state_path = load_state(vault_root)
    wikilinks = load_graph_edges(vault_root)

    synapses = state.get('synapses', {})
    total_before = len(synapses)

    to_remove = []
    for key, syn in synapses.items():
        freq = syn.get('frequency', 0)
        weight = syn.get('weight', 0)

        # Keep strong synapses
        if freq > 1 or weight >= 0.5:
            continue

        # Keep if direct wikilink exists
        if key in wikilinks:
            continue

        # This is a weak, non-wikilink synapse — remove
        to_remove.append(key)

    print(f"Synapses before: {total_before}")
    print(f"Spurious (f=1, w<0.5, no wikilink): {len(to_remove)}")
    print(f"Remaining: {total_before - len(to_remove)}")

    if to_remove:
        # Show what we're removing
        removed_voicebox = [k for k in to_remove if 'voicebox' in k]
        if removed_voicebox:
            print(f"\n  Removing {len(removed_voicebox)} voicebox synapses:")
            for k in removed_voicebox:
                a, b = k.split('|')
                print(f"    {a.split('/')[-1]} <-> {b.split('/')[-1]}")

        if not dry_run:
            for key in to_remove:
                del synapses[key]
            state['synapses'] = synapses
            save_state(state, state_path)
            print(f"\n✅ Cleaned {len(to_remove)} synapses. State saved.")
        else:
            print(f"\n[dry run] No changes made.")
    else:
        print("\n✅ No spurious synapses found.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean spurious Hebbian synapses")
    parser.add_argument("--vault", default="/Users/albi/Documents/Hermes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cleanup(args.vault, args.dry_run)
