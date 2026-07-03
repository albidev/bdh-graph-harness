"""Prompt construction — system/user message building and context formatting."""


def format_context(active_notes, nodes):
    """Format active notes as context for an LLM."""
    sorted_notes = sorted(active_notes.items(), key=lambda x: -x[1])
    parts = []
    for note_id, score in sorted_notes:
        node = nodes.get(note_id)
        if not node:
            continue
        parts.append(f"### {node['title']} (activation: {score:.3f})\n{node['text'][:300]}\n")
    return "\n---\n".join(parts)


def build_messages(query, active_notes, nodes):
    """Build the system + user messages list for an LLM call.

    Returns a list of {"role", "content"} dicts.
    """
    context = format_context(active_notes, nodes)

    system_prompt = (
        "You are a knowledge assistant grounded in the user's Obsidian vault. "
        "Answer the user's question using ONLY the provided note context. "
        "If the context doesn't contain enough information, say so explicitly. "
        "Cite notes by name when you use information from them, e.g. '[from: Baby Dragon Hatchling]'. "
        "Keep responses concise and factual. Do not invent information not present in the context."
    )

    user_prompt = f"""## Activated Notes Context

{context}

## Question
{query}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return messages