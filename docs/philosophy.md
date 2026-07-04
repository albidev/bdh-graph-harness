# Philosophy — BDH Graph Harness

Why this project exists, what it borrows from neuroscience, and how Obsidian became an unlikely neural substrate.

---

## The observation

In September 2025, Adrian Kosowski et al. published [The Dragon Hatchling (BDH)](https://arxiv.org/abs/2509.26507) — a Large Language Model architecture based on a scale-free biologically inspired network of locally-interacting neuron particles. Its working memory relies entirely on synaptic plasticity with Hebbian learning using spiking neurons. Specific synapses strengthen whenever the model hears or reasons about a specific concept while processing language inputs.

The core idea: a network of neurons connected by weighted synapses. When two neurons fire together, their connection strengthens. When neurons stop being co-activated, connections decay. New neurons are born when the system encounters concepts it doesn't have a representation for.

It worked. Not at GPT scale, but structurally — the network demonstrated genuine learning from its own activity. The question became: can this principle be applied to something people actually use?

## The substrate

Obsidian vaults are already graphs. Every `[[wikilink]]` is an edge. Every note is a node. The topology is there — what's missing is weight.

A vault with 350 notes and 2,000 wikilinks is a static network. The links don't know which connections matter. Two notes linked in 2023 might be deeply related to a query you're making today, but the graph treats all edges equally. It's a brain frozen in time.

The BDH Graph Harness doesn't create a graph — it **animates** one. It reads the existing vault structure, embeds every note into a vector space, and then overlays a Hebbian plasticity layer that learns from usage. The vault becomes a living system: connections strengthen, weak ones decay, new concepts emerge.

## Why not RAG?

Retrieval-Augmented Generation is the standard approach: embed documents, retrieve the most similar chunks, feed them to an LLM. It works. It's also completely static — the same query always retrieves the same documents, regardless of what happened before.

The harness makes retrieval **stateful**. Every query modifies the graph. Notes that were co-activated during a search get their synaptic weight increased. The next time a related query comes in, those notes are more likely to activate — not because the embedding changed, but because the *topology* adapted.

This is the fundamental difference: RAG is a lookup table. The harness is a nervous system.

## The plasticity model

Hebbian learning is the oldest principle in computational neuroscience: *neurons that fire together wire together*. In the harness, this translates to a concrete algorithm:

1. A query arrives
2. Hybrid search (vector similarity + BM25 keyword matching) finds seed notes
3. k-hop graph traversal spreads activation from seeds through wikilink connections
4. An adaptive threshold filters noise — only nodes scoring above `max(Q75, mean+1σ, 0.15)` remain active
5. **All pairs of co-activated notes strengthen their synaptic weight**
6. Unused synapses decay by a factor of 0.95
7. The LLM generates a response grounded in the activated subset
8. New concepts are extracted and created as notes in the vault

The critical design choice: step 5 happens *before* step 7. The graph learns from what was *activated*, not from what the LLM chose to use. This is **online plasticity** — the system updates its internal model in real-time, before external validation. In neuroscience terms, the association forms at perception, not at recall.

## Neurogenesis

In the adult hippocampus, new neurons are born continuously — a process called adult neurogenesis. These new neurons integrate into existing circuits and contribute to memory formation.

The harness implements this directly. After each LLM response, a second pass extracts concepts that were discussed but don't exist in the vault. For each new concept, an atomic note is created in a `concepts/` directory with frontmatter tags, a one-sentence definition, and links to the seed notes that triggered its creation.

On the next startup, the incremental embedding process picks up these new notes, embeds them, and they join the graph. The vault has physically grown. The next query can activate these new neurons, creating connections the system didn't have before.

This is the self-reinforcing loop: queries create concepts, concepts create connections, connections shape future queries.

## Architecture as metaphor

The system maps loosely to cortical organization:

| Brain region | Harness component | Function |
|---|---|---|
| Sensory cortex | ChromaDB + BM25 | Raw perception — embedding similarity and keyword matching |
| Association cortex | k-hop traversal + attention spread | Connecting related concepts across the graph |
| Synaptic plasticity | Hebbian update + decay | Strengthening used connections, pruning unused ones |
| Hippocampal neurogenesis | Concept extraction + note creation | Forming new representations from experience |
| Prefrontal cortex | Adaptive threshold + hub dampening | Filtering noise, preventing dominant nodes from drowning the signal |

This isn't a neuroscience simulation — it's an engineering system that borrows structural principles. The mapping is useful for understanding *why* each component exists, not for making claims about biological equivalence.

## What the vault becomes

Before the harness, an Obsidian vault is a personal wiki. A knowledge base. A second brain (to use the popular metaphor).

After the harness, it's something different. The vault is a substrate for a system that:

- **Remembers what was queried together** — co-activated notes strengthen their connection
- **Forgets what's ignored** — unused synapses decay
- **Grows from its own activity** — neurogenesis creates new notes from conversations
- **Adapts its retrieval to usage patterns** — the graph's topology reflects what matters *to you*, not what's statistically similar in embedding space

The vault becomes a knowledge graph that learns.

## Open questions

This project is a working prototype, not a finished theory. Open areas:

- **Consolidation** — real brains consolidate memories during sleep. The harness doesn't yet have a periodic process to reorganize, merge, or prune the graph based on accumulated usage patterns
- **Multi-vault coordination** — currently single-vault. Different vaults could represent different knowledge domains with separate plasticity states
- **Quantitative evaluation** — how do you measure whether the graph is actually getting "smarter"? Retrieval precision over time? User satisfaction with LLM responses?
- **Forgetting curves** — the current decay is uniform. Real memory has variable retention based on emotional salience, repetition patterns, and context
- **Interference** — when new learning disrupts old memories. The harness doesn't model catastrophic forgetting or recovery

These aren't bugs — they're research directions. The system works today as a retrieval engine that improves with use. Whether it can approximate something closer to genuine learning is an open question worth exploring.

## Further reading

- Kosowski, A. et al. (2025). *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain.* arXiv:2509.26507
- Hebb, D.O. (1949). *The Organization of Behavior.* Wiley.
- Eriksson, P.S. et al. (1998). *Neurogenesis in the adult human hippocampus.* Nature Medicine, 4(11), 1313–1317.
- [BDH Graph Harness — README](../README.md)
- [MCP Server Documentation](./mcp-server.md)
- [Visualization Guide](./visualization.md)
