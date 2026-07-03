"""Neurogenesis subpackage — concept extraction, note creation, and dedup."""

from bdh_graph_harness.neurogenesis.creator import (
    extract_new_concepts,
    slugify,
    create_note,
    update_vault_index,
    append_to_vault_log,
)
from bdh_graph_harness.neurogenesis.dedupe import is_duplicate

__all__ = [
    'extract_new_concepts',
    'slugify',
    'create_note',
    'update_vault_index',
    'append_to_vault_log',
    'is_duplicate',
]