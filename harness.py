"""Compatibility shim — re-exports everything from the modular package.

Existing tests and scripts that do `import harness` will continue to work.
New code should import from bdh_graph_harness.* directly.
"""
import sys
import os

# Ensure the package is importable
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from bdh_graph_harness.config import (
    CONFIG, load_config, retry_with_backoff,
    OLLAMA_EMBED_URL, OLLAMA_LLM_URL,
    STATE_FILE, LOCK_FILE,
    logger,
)
from bdh_graph_harness.graph.parser import (
    WIKILINK_RE, FRONTMATTER_RE,
    extract_note_id, find_note_by_id,
    parse_frontmatter, parse_json_frontmatter_list, extract_wikilinks, extract_text,
)
from bdh_graph_harness.graph.builder import (
    build_graph, _full_graph_build, _incremental_graph_update,
    _save_graph_cache, _resolve_target,
    GRAPH_CACHE_FILE,
)
from bdh_graph_harness.retrieval.embeddings import get_embeddings, cosine_similarity
from bdh_graph_harness.retrieval.chroma_store import compute_all_embeddings
from bdh_graph_harness.retrieval.bm25 import BM25Index
from bdh_graph_harness.retrieval.attention import attention, compute_adaptive_threshold, format_context
from bdh_graph_harness.memory.state_store import load_state, save_state
from bdh_graph_harness.memory.hebbian import hebbian_update
from bdh_graph_harness.llm.prompt import build_messages
from bdh_graph_harness.llm.providers import (
    _build_llm_payload, _parse_llm_response, _parse_llm_stream_token,
    llm_respond, llm_stream,
)
from bdh_graph_harness.neurogenesis.creator import (
    extract_new_concepts, slugify, create_note,
    update_vault_index, append_to_vault_log,
)
from bdh_graph_harness.api.server import start_api_server
from bdh_graph_harness.visualization import render_viz_html as _viz_html

# Re-export format_context from where it actually lives
try:
    from bdh_graph_harness.llm.prompt import format_context as _fmt_ctx
except ImportError:
    _fmt_ctx = None

# Ensure format_context is available (tests may use it)
if 'format_context' not in dir():
    from bdh_graph_harness.retrieval.attention import format_context