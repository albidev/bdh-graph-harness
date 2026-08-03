"""Memory subpackage — state persistence, Hebbian updates, node quality, and consolidation."""

from bdh_graph_harness.memory.state_store import load_state, save_state, merge_states
from bdh_graph_harness.memory.hebbian import hebbian_update
from bdh_graph_harness.memory.quality import (
    compute_node_quality,
    compute_all_qualities,
    prune_dormant,
    try_reactivate,
    quality_stats,
)
from bdh_graph_harness.memory.consolidation import (
    consolidate,
    synaptic_downscaling,
    structural_pruning,
    prune_stale_dormant,
    consolidation_stats,
)
from bdh_graph_harness.memory.source_policy import (
    get_frequency_increment,
    get_source_policy,
    use_user_prompt_for_retrieval,
    allowed_sources,
)

__all__ = [
    'load_state', 'save_state', 'merge_states',
    'hebbian_update',
    'compute_node_quality', 'compute_all_qualities',
    'prune_dormant', 'try_reactivate', 'quality_stats',
    'consolidate', 'synaptic_downscaling', 'structural_pruning',
    'prune_stale_dormant', 'consolidation_stats',
    'get_frequency_increment', 'get_source_policy',
    'use_user_prompt_for_retrieval', 'allowed_sources',
]
