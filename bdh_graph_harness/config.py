"""
BDH Graph Harness — Configuration module.

Contains CONFIG defaults, load_config(), retry_with_backoff(),
and global OLLAMA_EMBED_URL / OLLAMA_LLM_URL.
"""

import os
import sys
import json
import time
import logging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_FILE = ".bdh-state.json"
LOCK_FILE = ".bdh-state.lock"
DEFAULT_CONFIG_PATHS = [
    "bdh-config.yaml",
    "~/.bdh-config.yaml",
]

# ---------------------------------------------------------------------------
# Defaults (overridden by config file)
# ---------------------------------------------------------------------------

CONFIG = {
    'vault_path': os.path.expanduser('~/Documents/Hermes'),
    'ollama_url': 'http://127.0.0.1:11434',
    # Embedding (Ollama)
    'embedding_model': 'nomic-embed-text-v2-moe',
    # LLM provider (ollama | openrouter)
    'llm_provider': 'ollama',
    'llm_model': 'gemma4:12b-mlx',
    'openrouter_url': 'https://openrouter.ai/api/v1/chat/completions',
    'openrouter_key': '',  # set in config or env
    'llm_temperature': 0.3,
    'llm_max_ctx': 4096,
    'llm_timeout': 300,
    'chroma_path': '.bdh-chroma',
    'chroma_collection': 'notes',
    'seed_count': 5,
    'max_hop': 2,
    'active_threshold': 0.25,
    'hub_dampening': True,
    'hub_degree_threshold': 25,      # dampen only very high-degree hubs (e.g. wiki/index)
    'max_neighbors_per_hop': 10,
    'alpha': 0.7,
    'beta': 0.3,
    'decay': 0.95,
    'hebbian_min_score': 0.15,  # min activation score to create Hebbian synapse
    'neurogenesis_dir': 'wiki/concepts',
    'neurogenesis_enabled': True,
    'api_host': '127.0.0.1',
    'api_port': 8642,
    'python_exec': sys.executable,
    # Hybrid search (Phase 3.1)
    'hybrid_search': True,
    'hybrid_alpha': 0.7,   # weight for vector similarity
    'hybrid_beta': 0.3,    # weight for BM25 keyword score
    'bm25_k1': 1.5,
    'bm25_b': 0.75,
    # Adaptive threshold (Phase 3.3)
    'adaptive_threshold': True,
    'threshold_floor': 0.15,
    # Online plasticity (Phase 3.2)
    'online_plasticity': True,
    # Node quality (Phase 3.5)
    'quality_threshold': 0.25,           # below this → dormant
    'quality_reactivation_score': 0.50,  # activation to re-awaken
    'quality_prune_interval': 50,        # re-evaluate every N queries
    # Memory consolidation (Phase 4)
    'consolidation_downscale_factor': 0.90,    # global weight multiplier per cycle
    'consolidation_prune_weight_floor': 0.02,  # delete synapses below this weight
    'consolidation_dormant_persist_cycles': 3, # remove nodes dormant for N+ cycles
    'consolidation_prune_dormant_nodes': True,  # actually delete stale dormant nodes
    # Integrate-and-Fire attention model
    'experimental_integrate_fire': False,  # IaF attention — enable via bdh-config.yaml
    'iaf_tau_base': 0.15,       # base firing threshold
    'iaf_tau_k': 0.075,         # degree scaling factor: τ_j = base + k * log(1 + deg)
    'iaf_max_steps': 5,         # max integration steps
    'iaf_convergence_threshold': 1e-4,  # stop if activation change below this
    # Graph ignore: node IDs or glob patterns to exclude from the graph
    # These nodes are never loaded as neurons and never become activation targets
    'graph_ignore': [
        'wiki/index',           # table of contents, not knowledge
        'wiki/log',             # session log, not knowledge
        'wiki/raw/*',           # raw/unprocessed notes
    ],
    'stream_enabled': True,
}

# Derived config (set after loading)
OLLAMA_EMBED_URL = None
OLLAMA_LLM_URL = None

# Logging
logger = logging.getLogger('bdh')


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str | None = None):
    """Load configuration from YAML file and merge with defaults.

    Tries the given path, then DEFAULT_CONFIG_PATHS. Sets global
    OLLAMA_EMBED_URL and OLLAMA_LLM_URL derived from ollama_url.
    Returns the merged config dict.
    """
    global CONFIG, OLLAMA_EMBED_URL, OLLAMA_LLM_URL

    import yaml  # PyYAML

    merged = dict(CONFIG)  # start with defaults

    paths_to_try = []
    if config_path:
        paths_to_try.append(os.path.expanduser(config_path))
    paths_to_try.extend(os.path.expanduser(p) for p in DEFAULT_CONFIG_PATHS)

    loaded = False
    for p in paths_to_try:
        if os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f) or {}
            merged.update(file_config)
            logger.info(f"Loaded config from {p}")
            loaded = True
            break

    if not loaded:
        logger.warning("No config file found; using defaults")

    # Expand vault path
    merged['vault_path'] = os.path.expanduser(merged['vault_path'])

    # Expand ${ENV_VAR} in config values (e.g. openrouter_key: ${OPENROUTER_API_KEY})
    for key, val in merged.items():
        if isinstance(val, str) and val.startswith('${') and val.endswith('}'):
            env_var = val[2:-1]
            merged[key] = os.environ.get(env_var, '')

    # Derived URLs — embeddings always from Ollama
    OLLAMA_EMBED_URL = merged['ollama_url'].rstrip('/') + '/api/embed'
    OLLAMA_LLM_URL = merged['ollama_url'].rstrip('/') + '/api/chat'

    # LLM endpoint depends on provider
    if merged.get('llm_provider') == 'openrouter':
        OLLAMA_LLM_URL = merged.get('openrouter_url', 'https://openrouter.ai/api/v1/chat/completions')
        key = merged.get('openrouter_key', '')
        if not key:
            logger.warning("OpenRouter provider selected but no API key found!")
        logger.info(f"LLM provider: OpenRouter ({merged.get('llm_model')})")
    else:
        logger.info(f"LLM provider: Ollama ({merged.get('llm_model')})")

    # Update CONFIG in-place so modules that did `from config import CONFIG`
    # see the merged values (reassigning CONFIG = merged would break those refs).
    CONFIG.clear()
    CONFIG.update(merged)
    return merged


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def retry_with_backoff(fn, max_attempts=3, delay=2):
    """Call fn() with exponential backoff. Returns result or raises last exception."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            wait = delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)