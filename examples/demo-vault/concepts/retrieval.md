---
title: Retrieval
tags: [demo, retrieval]
---
# Retrieval

Retrieval combines embeddings with wikilinks. A query selects semantic seeds; graph attention traverses linked notes, with bounded propagation. Hybrid retrieval may also use BM25.

A read-only query uses learn:false and respond:false. It returns activated notes without query plasticity or an LLM response. Learning can strengthen co-activated synapses when explicitly enabled.

See [[architecture]] and [[Nova]].
