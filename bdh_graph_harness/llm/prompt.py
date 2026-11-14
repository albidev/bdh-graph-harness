"""Prompt construction — system/user message building and context formatting.

Level B (graph-aware): the retrieval prompt stops treating the vault as a
flat vector dump and starts reasoning over the knowledge graph. It uses:
  - per-note activation score
  - node quality / dormant state (when a ``state`` dict is provided)
  - the Hebbian associative-context lane (learned correlations between
    concepts, built by attention and passed through ``associative_context``)
These signals are all OPTIONAL. When absent, the prompt degrades to the
previous behavior, so existing callers (CLI, MCP, tests) are unaffected.
"""

from bdh_graph_harness.config import CONFIG
from bdh_graph_harness.retrieval.context_packing import (
    DEFAULT_LLM_CONTEXT_BUDGET_CHARS,
    DEFAULT_MIN_NOTE_CHARS,
    PackItem,
    coerce_budget_chars,
    pack_items,
)

_NOTE_SEPARATOR = "\n---\n"


def _pack_note_context(active_notes, nodes, annotate=None, context_budget_chars=None):
    """Render activated notes as prompt sections within one shared budget.

    The budget is split across notes by activation score instead of cutting
    every note at the same length, so a long, highly activated note keeps more
    of its text while short notes cost only what they need.
    """
    budget = coerce_budget_chars(
        context_budget_chars,
        CONFIG.get('llm_context_budget_chars', DEFAULT_LLM_CONTEXT_BUDGET_CHARS),
    )
    items, headings = [], {}
    for note_id, score in sorted(active_notes.items(), key=lambda x: -x[1]):
        node = nodes.get(note_id)
        if not node:
            continue
        annotation = annotate(note_id) if annotate else ""
        heading = f"### {node['title']} (activation: {score:.3f}{annotation})\n"
        headings[note_id] = heading
        items.append(PackItem(
            key=note_id,
            score=score,
            text=node['text'],
            overhead=len(heading) + 1 + len(_NOTE_SEPARATOR),
        ))
    packed, _omitted = pack_items(
        items,
        budget,
        min_chars=CONFIG.get('retrieval_context_min_note_chars', DEFAULT_MIN_NOTE_CHARS),
    )
    return _NOTE_SEPARATOR.join(f"{headings[entry.key]}{entry.text}\n" for entry in packed)


def format_context(active_notes, nodes, context_budget_chars=None):
    """Format active notes as context for an LLM."""
    return _pack_note_context(active_notes, nodes, context_budget_chars=context_budget_chars)


def _format_associative_context(associative_context, nodes):
    """Format the Hebbian associative lane as an optional context block.

    ``associative_context`` is a list of items, each carrying at least an
    ``id`` (note id) and optionally ``weight`` / ``trust`` / ``relationship``.
    Returns an empty string when there is nothing meaningful to show.
    """
    if not associative_context:
        return ""
    lines = []
    for item in associative_context:
        node_id = item.get("id")
        if not node_id:
            continue
        node = nodes.get(node_id) if nodes else None
        label = node["title"] if node else node_id
        weight = item.get("weight")
        rel = item.get("relationship")
        extra = []
        if rel:
            extra.append(f"relationship={rel}")
        if weight is not None:
            extra.append(f"weight={round(weight, 3)}")
        suffix = f" ({', '.join(extra)})" if extra else ""
        lines.append(f"- {label}{suffix}")
    if not lines:
        return ""
    return "\n".join(lines)


def _node_state_note(note_id, nodes, state):
    """Return a compact quality/dormancy annotation for one note, or ''."""
    if not state or not nodes:
        return ""
    node = nodes.get(note_id)
    if not node:
        return ""
    dormant = state.get("dormant_nodes") or set()
    nq = state.get("node_quality") or {}
    quality = (nq.get(note_id) or {}).get("score", 0.0) if isinstance(nq, dict) else 0.0
    tags = []
    if note_id in dormant:
        tags.append("dormant")
    if isinstance(nq, dict) and quality:
        tags.append(f"quality={round(quality, 3)}")
    return f" [{', '.join(tags)}]" if tags else ""


def build_messages(query, active_notes, nodes, state=None, associative_context=None,
                   context_budget_chars=None):
    """Build the system + user messages list for an LLM call.

    Returns a list of {"role", "content"} dicts.

    ``state`` (optional): the BDH runtime state dict, carrying ``dormant_nodes``
    and ``node_quality``. Used to annotate notes with quality/dormancy so the
    LLM can weigh confident, consolidated concepts over sparse ones.

    ``associative_context`` (optional): a list of learned Hebbian neighbors
    (from attention's ``associative_context`` lane) that are NOT part of the
    primary activation. Used as a separate, non-ranking correlation lane.

    ``context_budget_chars`` (optional): size of the note context, shared
    across the activated notes by score. Defaults to ``llm_context_budget_chars``.

    When both are omitted, behavior matches the pre-Level-B prompt exactly.
    """
    has_state = isinstance(state, dict)
    has_assoc = bool(associative_context)

    if not active_notes:
        # No notes activated — be honest about it.
        system_prompt = (
            "You are a knowledge assistant grounded in the user's Obsidian vault. "
            "No notes in the vault were activated for this query. "
            "Tell the user that the vault doesn't contain information about this topic. "
            "Keep the response to one or two sentences. Do not attempt to answer the question."
        )
        user_prompt = f"## Question\n{query}\n\nNo vault notes were activated for this query."
        return [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

    # Primary context with optional per-note quality/dormancy annotation.
    context = _pack_note_context(
        active_notes,
        nodes,
        annotate=(lambda note_id: _node_state_note(note_id, nodes, state)) if has_state else None,
        context_budget_chars=context_budget_chars,
    )

    if not has_state and not has_assoc:
        # Pre-Level-B behavior, unchanged.
        system_prompt = (
            "You are a knowledge assistant grounded in the user's Obsidian vault. "
            "Answer the user's question using ONLY the provided note context. "
            "If the context doesn't contain enough information, say so explicitly. "
            "Cite notes by name when you use information from them, e.g. '[from: Baby Dragon Hatchling]'. "
            "Keep responses concise and factual. Do not invent information not present in the context."
        )
        user_prompt = f"## Activated Notes Context\n\n{context}\n\n## Question\n{query}\n"
        return [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

    # ── Level B: graph-aware prompt ──────────────────────────────────────
    system_prompt = (
        "You are a knowledge assistant grounded in the user's Obsidian vault knowledge graph. "
        "Answer the user's question using the provided note context as your primary evidence. "
        "Ground every claim in the notes: cite notes by name, e.g. '[from: Baby Dragon Hatchling]'. "
        "Keep responses concise and factual. Do not invent information not present in the context."
    )

    # Only add the graph-reasoning guidance when we actually have the signals.
    graph_notes = []
    if has_state:
        graph_notes.append(
            "Notes are annotated with an activation score and, when available, a "
            "quality/dormancy tag. Prefer well-consolidated notes (higher quality, "
            "not dormant) as primary evidence; treat dormant or low-quality notes as "
            "weaker support and say so if you must rely on them."
        )
    if has_assoc:
        graph_notes.append(
            "Below the primary context there is an 'Associative context' section: "
            "concepts the graph has learned to associate with the activated notes, "
            "drawn from repeated past use (Hebbian associations). These are NOT "
            "confirmed evidence. Use them only to suggest related directions or "
            "to note plausible connections, and always label them as inferred, "
            "never as facts."
        )
    if graph_notes:
        system_prompt += " " + " ".join(graph_notes)

    user_prompt = f"## Activated Notes Context\n\n{context}\n"
    assoc_block = _format_associative_context(associative_context, nodes)
    if assoc_block:
        user_prompt += f"\n## Associative context (learned correlations, inferred)\n\n{assoc_block}\n"
    user_prompt += f"\n## Question\n{query}\n"

    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]
