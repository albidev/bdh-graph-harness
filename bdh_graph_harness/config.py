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
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

# Load .env file from project root (if present) before reading config
load_dotenv()

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
    # Optional alternate state file for reversible clean-room learning runs.
    'hebbian_state_file': STATE_FILE,
    # Optional read-only Markdown sources merged into the primary vault graph.
    # Each entry supports id, path, include, exclude, and writable (false by default).
    'external_sources': [],
    # OKF document/interchange read compatibility; runtime state stays BDH-owned.
    'okf_mode': False,
    # Trust-aware retrieval is gated by okf_mode and can be disabled for A/B tests.
    'okf_retrieval_policy_enabled': True,
    'okf_policy_verified_bonus': 1.08,
    'okf_policy_unverified_penalty': 0.95,
    'okf_policy_draft_multiplier': 0.85,
    'okf_policy_deprecated_multiplier': 0.50,
    'okf_policy_stale_multiplier': 0.60,
    'okf_policy_provenance_bonus': 1.03,
    'okf_policy_min_multiplier': 0.35,
    'okf_policy_max_multiplier': 1.20,
    'ollama_url': 'http://127.0.0.1:11434',
    # Embedding (Ollama)
    'embedding_model': 'nomic-embed-text-v2-moe',
    # LLM provider (ollama | ollama-cloud | nous | openrouter | omlx)
    'llm_provider': 'ollama',
    'llm_model': 'gemma4:12b-mlx',
    # Ordered completion failover candidates. The primary remains llm_provider.
    'llm_fallbacks': [],
    # Canonical OpenAI-compatible settings (used by Ollama Cloud/OpenRouter).
    'llm_base_url': '',
    'llm_api_key': '',
    'llm_provider_label': 'Ollama',
    'llm_transport': 'ollama-native',
    'openrouter_url': 'https://openrouter.ai/api/v1/chat/completions',
    'openrouter_key': '',  # legacy compatibility; prefer llm_api_key
    'llm_temperature': 0.3,
    'llm_max_ctx': 4096,
    'llm_timeout': 300,
    'embed_timeout': 120,  # also used by ChromaDB Ollama embedding function
    'chroma_path': '.bdh-chroma',
    'chroma_collection': 'notes',
    'seed_count': 5,
    'hebbian_learning_seed_count': 2,  # conservative write budget; separate from retrieval seeds
    'max_hop': 2,
    'active_threshold': 0.25,
    'hub_dampening': True,
    'hub_degree_threshold': 25,      # dampen only very high-degree hubs (e.g. wiki/index)
    'max_neighbors_per_hop': 10,
    'attention_relevance_batch_size': 256,
    'hop_decay': 0.5,  # score decay per hop (single application, not compound)
    'hebbian_gain': 0.0,  # multiplier on learned synapse weight during propagation; 0 = disabled
    # Dynamic Hebbian adjacency: learned co-activations are traversable edges,
    # independent of static wikilinks. Kept tightly capped to prevent drift.
    'hebbian_dynamic_edges_enabled': True,
    'hebbian_dynamic_min_weight': 0.15,
    'hebbian_dynamic_top_n': 3,
    'hebbian_dynamic_gain': 1.5,
    'hebbian_dynamic_hop_decay': 0.6,
    'hebbian_associative_context_enabled': False,
    'hebbian_associative_context_max_items': 2,
    'hebbian_associative_context_max_per_seed': 1,
    # Dynamic associations need query-local semantic evidence before they may
    # compete with declared wikilinks in retrieval.
    'hebbian_dynamic_query_relevance_floor': 0.35,
    # Confidence gate for learned-only traversal; static wikilinks ignore these.
    'hebbian_dynamic_frequency_saturation': 2.0,
    'hebbian_dynamic_unconsolidated_trust': 0.6,
    'hebbian_dynamic_recency_days': 30.0,
    'hebbian_dynamic_recency_floor': 0.4,
    'hebbian_dynamic_trust_floor': 0.25,
    'hebbian_dynamic_shadow_enabled': True,
    'alpha': 0.7,
    'beta': 0.3,
    'decay': 0.95,
    'hebbian_min_score': 0.15,  # min activation score to create Hebbian synapse
    'hebbian_frequency_scale': 10.0,  # log1p scale for non-saturating frequency compression
    'tau_recency_hours': 24.0,  # recency half-life in hours
    'neurogenesis_dir': 'wiki/concepts',
    'neurogenesis_enabled': True,
    'api_host': '127.0.0.1',
    'api_port': 8642,
    'api_auth_token': '',  # if set, requires Authorization: Bearer <token> on all API routes
    'python_exec': sys.executable,
    # Hybrid search (Phase 3.1)
    'hybrid_search': True,
    'hybrid_fusion': 'rrf',  # 'weighted' | 'rrf'
    'hybrid_alpha': 0.7,   # weight for vector similarity
    'hybrid_beta': 0.3,    # weight for BM25 keyword score
    'rrf_k': 60,
    'bm25_k1': 1.5,
    'bm25_b': 0.75,
    # Query-level abstention must run before seed selection. RRF scores are
    # rank-normalized per query and cannot be used as absolute confidence.
    'retrieval_abstention_enabled': True,
    'retrieval_min_vector_score': 0.58,
    'retrieval_min_bm25_matched_terms': 5,
    # Short queries that exactly name a graph entity may bypass the generic
    # evidence gate when the match is present in stable identity metadata.
    'retrieval_entity_match_enabled': True,
    # Adaptive threshold (Phase 3.3)
    'adaptive_threshold': False,
    'threshold_floor': 0.15,
    # Online plasticity (Phase 3.2)
    'online_plasticity': True,
    # Hebbian-aware seed ranking (Phase 5)
    'hebbian_seed_boost': True,
    'hebbian_boost_max': 0.5,              # max boost factor (+50%)
    'hebbian_boost_top_n': 3,              # only count top-N strongest synapses
    'hebbian_boost_weight_factor': 0.3,    # multiplier on summed weight
    'hebbian_boost_window_minutes': 10,    # recency window for "recently active"
    'hebbian_boost_min_weight': 0.15,      # min synapse weight to consider active
    # Multi-query retrieval (issue #19)
    'multi_query_enabled': False,
    'multi_query_max_variants': 3,
    'multi_query_fusion': 'rrb',  # 'rrb' | 'weighted'
    'rrb_k': 60,
    # Node quality (Phase 3.5)
    'quality_threshold': 0.25,           # below this → dormant
    'quality_reactivation_score': 0.50,  # activation to re-awaken
    'quality_prune_interval': 50,        # re-evaluate every N queries
    # Memory consolidation (Phase 4)
    'consolidation_downscale_factor': 0.90,    # global weight multiplier per cycle
    'consolidation_prune_weight_floor': 0.02,  # delete synapses below this weight
    'consolidation_weak_weight_threshold': 0.15,
    'consolidation_weak_max_frequency': 1.0,
    'consolidation_weak_min_age_hours': 48,
    # Safe consolidation guardrails. Candidates must survive confirmation before
    # deletion; a cycle aborts instead of committing an anomalous mass prune.
    'consolidation_prune_confirm_cycles': 2,
    'consolidation_max_prune_ratio': 0.35,
    'consolidation_max_prune_per_cycle': 0.15,
    'consolidation_protect_backbone': True,
    'consolidation_protect_recent_hours': 72,
    'consolidation_dormant_persist_cycles': 3, # remove nodes dormant for N+ consolidation cycles
    'consolidation_prune_dormant_nodes': True,  # actually delete stale dormant nodes
    # Interactive neurogenesis is conservative but can retain several independent durable concepts.
    'neurogenesis_max_concepts': 3,
    'neurogenesis_source_edges_enabled': True,
    # Semantic sleep — disabled until explicitly enabled in the vault config.
    'semantic_consolidation_enabled': False,
    'semantic_consolidation_checkpoint': '.bdh-semantic-consolidation.json',
    'semantic_consolidation_max_sources': 3,
    'semantic_consolidation_max_age_hours': 48,
    'semantic_consolidation_max_source_chars': 8000,
    'semantic_consolidation_max_batch_chars': 16000,
    'semantic_consolidation_max_concepts': 5,
    'semantic_consolidation_session_enabled': True,
    'semantic_consolidation_session_db_path': '~/.hermes/state.db',
    'semantic_consolidation_max_session_chars': 12000,
    'semantic_consolidation_include_cron_sessions': False,
    'semantic_consolidation_source_globs': [
        'wiki/**/*.md',
        'projects/**/*.md',
        'memory/learned/*.md',
    ],
    'semantic_consolidation_exclude_globs': [
        'memory/daily/*',
        'wiki/index.md',
        'wiki/log.md',
        'wiki/raw/*',
        'wiki/concepts/*',
        '.bdh-*',
    ],
    'semantic_consolidation_source': 'nightly_semantic_consolidation',
    'semantic_consolidation_frequency_increment': 0.3,
    # Integrate-and-Fire attention model
    'experimental_integrate_fire': False,  # IaF is experimental and currently underperforming; keep off by default
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
        'graphify-out/*',       # derived Graphify artifacts, not knowledge
    ],
    'stream_enabled': True,
}

# Derived config (set after loading)
OLLAMA_EMBED_URL = None
OLLAMA_LLM_URL = None

# Logging
logger = logging.getLogger('bdh')


# ---------------------------------------------------------------------------
# Per-vault LLM routing
# ---------------------------------------------------------------------------

def _expand_env_values(value):
    """Recursively expand ``${ENV_VAR}`` placeholders in config values."""
    if isinstance(value, dict):
        return {key: _expand_env_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_values(item) for item in value]
    if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
        return os.environ.get(value[2:-1], '')
    return value




def _resolve_nous_api_key() -> str:
    """Resolve a Nous Portal inference key without persisting or logging it.

    Prefer an explicit API key for standalone deployments.  When BDH runs next
    to Hermes, reuse the short-lived agent key from Hermes' auth store so the
    local config never needs to contain a bearer credential.
    """
    for env_name in ('NOUS_API_KEY', 'NOUS_AGENT_KEY', 'NOUS_ACCESS_TOKEN'):
        value = os.environ.get(env_name, '').strip()
        if value:
            return value

    auth_file = os.environ.get('NOUS_AUTH_FILE', '').strip()
    if auth_file:
        candidates = [Path(auth_file)]
    else:
        hermes_home = os.environ.get('HERMES_HOME', '').strip()
        candidates = [
            Path(hermes_home) / 'auth.json' if hermes_home else Path.home() / '.hermes' / 'auth.json',
            Path.home() / '.hermes' / 'auth.json',
        ]

    seen = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            provider = (data.get('providers') or {}).get('nous') or {}
            for key_name in ('agent_key', 'access_token'):
                value = provider.get(key_name, '')
                if isinstance(value, str) and value.strip():
                    return value.strip()
        except (OSError, ValueError, TypeError):
            continue
    return ''


def resolve_llm_config(base_config: dict | None = None, *, require_endpoint: bool = False) -> dict:
    """Return an immutable effective LLM config for a vault or global config.

    ``base_config`` is normally ``VaultConfig.settings``.  The optional nested
    ``llm`` block is the preferred per-vault form::

        llm:
          provider: ollama
          model: gemma4:26b-mlx
          base_url: http://127.0.0.1:11434

    Cloud vaults should use ``api_key_env`` rather than storing a secret in the
    YAML file.  Legacy flat keys (``llm_provider``, ``llm_model``, etc.) remain
    supported and continue to act as global defaults.

    The function never mutates :data:`CONFIG` and derives the endpoint locally,
    which is what prevents one vault's provider from bleeding into another.
    """
    effective = dict(CONFIG)
    if base_config:
        effective.update(_expand_env_values(base_config))

        nested = base_config.get('llm')
        if isinstance(nested, dict):
            expanded_nested = _expand_env_values(nested)
            nested = expanded_nested if isinstance(expanded_nested, dict) else {}
            provider = nested.get('provider', effective.get('llm_provider', 'ollama'))
            effective['llm_provider'] = provider

            aliases = {
                'model': 'llm_model',
                'temperature': 'llm_temperature',
                'max_ctx': 'llm_max_ctx',
                'max_tokens': 'llm_max_tokens',
                'timeout': 'llm_timeout',
            }
            for nested_key, flat_key in aliases.items():
                if nested_key in nested:
                    effective[flat_key] = nested[nested_key]

            if 'api_key' in nested:
                effective['llm_api_key'] = nested['api_key'] or ''
            elif 'api_key_env' in nested:
                api_key_env = nested.get('api_key_env')
                effective['llm_api_key'] = (
                    os.environ.get(str(api_key_env), '') if api_key_env else ''
                )
            elif provider == 'ollama':
                # A local vault must not inherit a cloud credential into its
                # effective runtime config, even though Ollama-native calls do
                # not send one on the wire.
                effective['llm_api_key'] = ''

            if nested.get('local_only'):
                effective['llm_local_only'] = True

            if 'base_url' in nested:
                base_url = str(nested['base_url']).rstrip('/')
                if provider == 'ollama':
                    effective['ollama_url'] = base_url
                elif provider == 'openrouter':
                    effective['openrouter_url'] = (
                        base_url if base_url.endswith('/chat/completions')
                        else f'{base_url}/chat/completions'
                    )
                else:
                    effective['llm_base_url'] = base_url

    provider = effective.get('llm_provider', 'ollama')
    if effective.get('llm_local_only'):
        if provider != 'ollama':
            raise ValueError(
                "llm.local_only=true requires the 'ollama' provider; "
                f"refusing provider '{provider}'"
            )
        local_url = str(effective.get('ollama_url') or '').rstrip('/')
        local_host = urlparse(local_url).hostname
        if local_host not in {'127.0.0.1', 'localhost', '::1'}:
            raise ValueError(
                "llm.local_only=true requires an Ollama endpoint on localhost; "
                f"got '{local_url}'"
            )
    if provider == 'ollama-cloud':
        base_url = str(effective.get('llm_base_url') or '').rstrip('/')
        endpoint = f'{base_url}/chat/completions' if base_url else ''
        effective['llm_provider_label'] = 'Ollama Cloud'
        effective['llm_transport'] = 'openai-compatible'
        effective['llm_endpoint'] = endpoint
        if require_endpoint and not base_url:
            raise ValueError('ollama-cloud requires llm_base_url')
    elif provider == 'nous':
        base_url = str(
            effective.get('llm_base_url') or 'https://inference-api.nousresearch.com/v1'
        ).rstrip('/')
        effective['llm_base_url'] = base_url
        effective['llm_api_key'] = effective.get('llm_api_key') or _resolve_nous_api_key()
        effective['llm_provider_label'] = 'Nous Portal'
        effective['llm_transport'] = 'openai-compatible'
        effective['llm_endpoint'] = f'{base_url}/chat/completions'
        if require_endpoint and not effective['llm_api_key']:
            raise ValueError('nous requires a runtime API key or Hermes auth file')
    elif provider == 'omlx':
        base_url = str(
            effective.get('llm_base_url') or 'http://127.0.0.1:8083/v1'
        ).rstrip('/')
        endpoint = f'{base_url}/chat/completions'
        effective['llm_base_url'] = base_url
        effective['llm_provider_label'] = 'oMLX'
        effective['llm_transport'] = 'openai-compatible'
        effective['llm_endpoint'] = endpoint
    elif provider == 'openrouter':
        endpoint = str(
            effective.get('openrouter_url')
            or 'https://openrouter.ai/api/v1/chat/completions'
        ).rstrip('/')
        if not endpoint.endswith('/chat/completions'):
            endpoint += '/chat/completions'
        if not effective.get('llm_api_key'):
            effective['llm_api_key'] = effective.get('openrouter_key', '')
        effective['llm_provider_label'] = 'OpenRouter'
        effective['llm_transport'] = 'openai-compatible'
        effective['llm_endpoint'] = endpoint
    else:
        base_url = str(
            effective.get('ollama_url') or 'http://127.0.0.1:11434'
        ).rstrip('/')
        endpoint = base_url if base_url.endswith('/api/chat') else f'{base_url}/api/chat'
        effective['ollama_url'] = base_url
        effective['llm_provider_label'] = 'Ollama'
        effective['llm_transport'] = 'ollama-native'
        effective['llm_endpoint'] = endpoint

    return effective


def resolve_llm_candidates(base_config: dict | None = None) -> list[dict]:
    """Resolve the primary LLM and ordered failover candidates.

    Failover is deliberately owned by BDH rather than Hermes' outer agent
    loop. Each candidate is an independent OpenAI-compatible or Ollama
    runtime config, so a cloud outage can fall through to local oMLX without
    changing global process state.
    """
    primary = resolve_llm_config(base_config, require_endpoint=True)
    candidates = [primary]
    fallback_specs = primary.get('llm_fallbacks') or []
    if not isinstance(fallback_specs, list):
        logger.warning('Ignoring invalid llm_fallbacks value; expected a list')
        return candidates

    # Do not let the primary nested ``llm`` block override a flat fallback
    # candidate. Other vault settings remain available to the candidate.
    fallback_base = dict(base_config or {})
    fallback_base.pop('llm', None)
    for index, spec in enumerate(fallback_specs):
        if not isinstance(spec, dict):
            logger.warning('Ignoring invalid LLM fallback #%s; expected a mapping', index + 1)
            continue
        provider = spec.get('provider')
        model = spec.get('model')
        if not provider or not model:
            logger.warning('Ignoring incomplete LLM fallback #%s', index + 1)
            continue

        candidate_input = dict(fallback_base)
        candidate_input.update({
            'llm_provider': provider,
            'llm_model': model,
            # A candidate cannot recursively expand another fallback chain.
            'llm_fallbacks': [],
        })
        if 'temperature' in spec:
            candidate_input['llm_temperature'] = spec['temperature']
        if 'max_ctx' in spec:
            candidate_input['llm_max_ctx'] = spec['max_ctx']
        if 'timeout' in spec:
            candidate_input['llm_timeout'] = spec['timeout']
        if 'api_key' in spec:
            candidate_input['llm_api_key'] = spec['api_key'] or ''
        elif 'api_key_env' in spec:
            candidate_input['llm_api_key'] = os.environ.get(str(spec['api_key_env']), '')
        elif provider == 'nous':
            candidate_input['llm_api_key'] = _resolve_nous_api_key()
        elif provider in {'omlx', 'ollama'}:
            candidate_input['llm_api_key'] = ''

        if 'base_url' in spec:
            candidate_input['llm_base_url'] = spec['base_url']
        elif provider == 'nous':
            candidate_input['llm_base_url'] = 'https://inference-api.nousresearch.com/v1'
        elif provider == 'omlx':
            candidate_input['llm_base_url'] = 'http://127.0.0.1:8083/v1'
        elif provider == 'ollama':
            candidate_input['ollama_url'] = 'http://127.0.0.1:11434'

        try:
            candidates.append(resolve_llm_config(candidate_input, require_endpoint=True))
        except (TypeError, ValueError) as exc:
            logger.warning('Ignoring invalid LLM fallback #%s: %s', index + 1, exc)
    return candidates


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

    # Expand vault path and nested ${ENV_VAR} placeholders.
    merged = _expand_env_values(merged)
    merged['vault_path'] = os.path.expanduser(merged['vault_path'])

    # Embeddings always use local Ollama.  Keep the safe default when a
    # cloud-only global LLM config omits the embedding URL.
    merged['ollama_url'] = merged.get('ollama_url') or 'http://127.0.0.1:11434'
    OLLAMA_EMBED_URL = merged['ollama_url'].rstrip('/') + '/api/embed'

    # LLM endpoint and diagnostics depend on the actual provider.  Resolve the
    # global config through the same code path used by per-vault requests.
    resolved_llm = resolve_llm_config(merged, require_endpoint=True)
    merged.update(resolved_llm)
    OLLAMA_LLM_URL = merged['llm_endpoint']
    provider = merged.get('llm_provider', 'ollama')
    if provider in {'ollama-cloud', 'nous', 'openrouter'} and not merged.get('llm_api_key'):
        logger.warning('%s provider selected but no llm_api_key found!', provider)
    logger.info(
        'LLM provider: %s (%s) via %s',
        merged.get('llm_provider_label'),
        merged.get('llm_model'),
        merged.get('llm_transport'),
    )

    # Update CONFIG in-place so modules that did `from config import CONFIG`
    # see the merged values (reassigning CONFIG = merged would break those refs).
    CONFIG.clear()
    CONFIG.update(merged)
    return merged


# ---------------------------------------------------------------------------
# Config overlay — for parametric evaluation
# ---------------------------------------------------------------------------

from contextlib import contextmanager


@contextmanager
def config_overlay(overrides: dict):
    """Temporarily override CONFIG keys inside a narrow scope.

    Use this for parametric ablations and tests: changes are applied
    in-process and reverted on exit. Does not touch the on-disk config
    file or persisted Hebbian state.
    """
    original = {k: CONFIG[k] for k in overrides if k in CONFIG}
    absent = [k for k in overrides if k not in CONFIG]
    CONFIG.update(overrides)
    try:
        yield
    finally:
        for k in overrides:
            if k in original:
                CONFIG[k] = original[k]
            elif k in absent:
                CONFIG.pop(k, None)


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