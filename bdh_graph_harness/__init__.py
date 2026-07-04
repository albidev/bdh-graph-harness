"""BDH Graph Harness — neural vault knowledge engine.

Top-level package re-exports the key public API so callers can do::

    from bdh_graph_harness import load_config, build_graph, attention, ...
"""

# Config
from bdh_graph_harness.config import CONFIG, load_config

# Graph
from bdh_graph_harness.graph import build_graph

# Retrieval
from bdh_graph_harness.retrieval.attention import attention
from bdh_graph_harness.retrieval import get_embeddings

# Memory / Hebbian
from bdh_graph_harness.memory import hebbian_update, load_state, save_state, merge_states

# LLM
from bdh_graph_harness.llm import llm_respond, llm_stream

# Neurogenesis
from bdh_graph_harness.neurogenesis import extract_new_concepts, create_note

# API server
from bdh_graph_harness.api import start_api_server

__all__ = [
    "CONFIG",
    "load_config",
    "build_graph",
    "attention",
    "get_embeddings",
    "hebbian_update",
    "load_state",
    "save_state",
    "merge_states",
    "llm_respond",
    "llm_stream",
    "extract_new_concepts",
    "create_note",
    "start_api_server",
]