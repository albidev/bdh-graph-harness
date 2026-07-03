"""Duplicate detection for neurogenesis — avoid recreating existing concepts."""


def is_duplicate(title, existing_titles):
    """Check if a concept title already exists (case-insensitive).

    Args:
        title: The candidate concept title.
        existing_titles: Iterable of existing note titles.

    Returns:
        True if the title matches an existing title (case-insensitive), False otherwise.
    """
    title_lower = title.lower().strip()
    for existing in existing_titles:
        if existing.lower().strip() == title_lower:
            return True
    return False